from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import os
import time
import torch
import torch.distributed as dist

from .collectives import Collectives, episode_rounds, parameter_hash
from .common import root_for, identity, read_json, write_json, file_hash
from routed_lm.data import verified, read_episodes
from routed_lm.network import load
from routed_lm.workflow import component
from training_revision.common import verify as verify_baseline
from training_revision.engine import walk_training
from training_revision.workflow import score


def train(config, mapping):
    torch.cuda.set_device(0)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda:0"),
                                timeout=timedelta(minutes=60))
    runtime = Collectives("cuda:0")
    try:
        _train(config, mapping, runtime)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(config, mapping, runtime):
    root = root_for(config)
    data_root, manifest = verified(config)
    baseline = read_json(root / "before_train/summary.json")
    verify_baseline(baseline, config)
    print(f"parallel rank={runtime.rank}/{runtime.world_size} physical_gpu={mapping['physical_index']} "
          f"uuid={mapping['uuid']}", flush=True)
    network = load(config)
    frozen = network.frozen_hash()
    assert frozen == baseline["frozen_hash"]
    parameters = [p for p in network.parameters() if p.requires_grad]
    assert parameters and all(p.dtype == torch.float32 for p in parameters)
    schemas = runtime.gather([(n, list(p.shape)) for n, p in network.named_parameters() if p.requires_grad])
    assert all(schema == schemas[0] for schema in schemas)
    runtime.broadcast_parameters(parameters)
    initial_hashes = runtime.gather(parameter_hash(parameters))
    assert len(set(initial_hashes)) == 1, initial_hashes
    initial = network.trainable_state() if runtime.rank == 0 else None
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    episodes = read_episodes(data_root / "train.jsonl")
    validation = read_episodes(data_root / "validation.jsonl") if runtime.rank == 0 else None
    folder = root / "training"
    folder.mkdir(parents=True, exist_ok=True)
    trace_path = folder / f"train_trace.rank{runtime.rank}.jsonl"
    coverage_path = folder / f"coverage.rank{runtime.rank}.jsonl"
    trace_enabled = config.get("trace_training", False)
    pending = updates = 0
    # Cumulative local [visits, processed labels, unforced hits, forced labels, episodes].
    local = [0, 0, 0, 0, 0]
    totals = [0, 0, 0, 0, 0]
    best_key, best_score, selected_update, last_validation = None, None, 0, -1
    initial_validation = score(network, validation, config)[0] if runtime.rank == 0 else None
    runtime.barrier()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    log_context = (folder / "log.jsonl").open("w") if runtime.rank == 0 else nullcontext(None)
    trace_context = trace_path.open("w") if trace_enabled else nullcontext(None)

    with log_context as log, trace_context as trace, coverage_path.open("w") as coverage:
        def validate_and_save():
            nonlocal best_key, best_score, selected_update, last_validation
            if runtime.rank == 0:
                print(f"parallel validation update={updates}", flush=True)
                validation_score = score(network, validation, config)[0]
                key = (validation_score["unforced_hit_fraction"], -validation_score["mean_ce_per_visited_block"])
                log.write(json.dumps({"update": updates, "visited_blocks": totals[0],
                    "processed_targets": totals[1], "unforced_hits": totals[2], "forced_targets": totals[3],
                    "episodes": totals[4], "validation": validation_score,
                    "elapsed_seconds": time.monotonic() - started}) + "\n")
                log.flush()
                if best_key is None or key > best_key:
                    best_key, best_score, selected_update = key, validation_score, updates
                    payload = {"parameters": network.trainable_state(), "update": updates,
                        "frozen_hash": frozen, "data_manifest_hash": file_hash(data_root / "manifest.json"),
                        "training_method": "synchronous data parallel; visited-head CE; forced continuation",
                        "world_size": runtime.world_size, **identity(config)}
                    temporary = folder / "best.pt.tmp"
                    torch.save(payload, temporary)
                    temporary.replace(folder / "best.pt")
            runtime.barrier()
            last_validation = updates

        def update(global_pending):
            nonlocal pending, updates
            runtime.average_gradients(parameters, global_pending)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite globally averaged gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            updates += 1
            if runtime.rank == 0:
                print(f"parallel update={updates} world_size={runtime.world_size} episodes={totals[4]} "
                      f"processed={totals[1]} hits={totals[2]} forced={totals[3]} "
                      f"visited_blocks={totals[0]} update_visits={global_pending}", flush=True)
            if updates % config["eval_every_updates"] == 0:
                validate_and_save()

        global_pending = 0
        assigned = 0
        for epoch in range(config["epochs"]):
            for group in episode_rounds(episodes, config["seed"], epoch, runtime.world_size):
                if runtime.rank < len(group):
                    episode = group[runtime.rank]
                    assigned += len(episode["targets"])
                    before = list(local)
                    losses = []
                    for event, loss, state in walk_training(network, episode, config["label_mode"], config["iteration_policy"]):
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite local loss")
                        losses.append(loss)
                        local[0] += 1
                        if event["accepted"] or event["forced"]:
                            local[1] += 1
                            local[2] += int(event["accepted"])
                            local[3] += int(event["forced"])
                        if trace is not None:
                            trace.write(json.dumps({"epoch": epoch + 1, "rank": runtime.rank, **event}) + "\n")
                        end = event["action"] in ("complete_hit", "complete_forced")
                        truncate = config.get("bptt_visits", 0) > 0 and len(losses) >= config["bptt_visits"]
                        if end or truncate:
                            torch.stack(losses).sum().backward()
                            pending += len(losses)
                            losses = []
                            state.detach_()
                    assert not losses
                    assert local[1] - before[1] == len(episode["targets"])
                    local[4] += 1
                    coverage.write(json.dumps({"epoch": epoch + 1, "rank": runtime.rank,
                        "episode_id": episode["episode_id"], "processed_targets": local[1] - before[1],
                        "unforced_hits": local[2] - before[2], "forced_targets": local[3] - before[3]}) + "\n")
                # Even an idle rank must participate in the last, uneven round.
                values = runtime.sum_values([pending, *local])
                global_pending, totals = values[0], values[1:]
                if global_pending >= config["accumulation_visits"]:
                    update(global_pending)
                    global_pending = 0
            if runtime.rank == 0:
                print(f"parallel epoch={epoch + 1}/{config['epochs']} processed={totals[1]} "
                      f"episodes={totals[4]}", flush=True)
        if global_pending:
            update(global_pending)
        if updates == 0:
            raise RuntimeError("No optimizer update")
        if last_validation != updates:
            validate_and_save()

    expected = sum(len(e["targets"]) for e in episodes) * config["epochs"]
    assert local[1] == assigned
    assert totals[1] == expected and totals[4] == len(episodes) * config["epochs"]
    assert network.frozen_hash() == frozen
    worker = {"rank": runtime.rank, "device": mapping, "processed_targets": local[1],
        "episodes": local[4], "visited_blocks": local[0], "unforced_hits": local[2], "forced_targets": local[3],
        "final_parameter_hash": parameter_hash(parameters), "coverage_sha256": file_hash(coverage_path),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    if trace_enabled:
        worker["trace_sha256"] = file_hash(trace_path)
    write_json(folder / f"worker{runtime.rank}.json", {**identity(config), **worker})
    workers = runtime.gather(worker)
    assert len({w["final_parameter_hash"] for w in workers}) == 1, "Model replicas diverged"
    if runtime.rank == 0:
        state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)["parameters"]
        changes = {}
        for name, value in state.items():
            group = component(name)
            changes[group] = changes.get(group, 0.) + (value - initial[name]).square().sum().item()
        changes = {key: value**0.5 for key, value in changes.items()}
        write_json(folder / "done.json", {**identity(config), "device": mapping,
            "devices": [w["device"] for w in workers], "world_size": runtime.world_size, "workers": workers,
            "replica_parameters_identical": True, "initial_parameter_hash": initial_hashes[0],
            "updates": updates, "selected_update": selected_update, "epochs": config["epochs"],
            "requested_training_targets": expected, "processed_training_targets": totals[1], "unreached_training_targets": 0,
            "unforced_training_hits": totals[2], "forced_training_targets": totals[3], "visited_blocks": totals[0],
            "processed_training_episodes": totals[4],
            "frozen_hash": frozen, "frozen_weights_unchanged": True,
            "checkpoint_sha256": file_hash(folder / "best.pt"), "component_parameter_change_l2": changes,
            "train_data_sha256": manifest["splits"]["train"]["sha256"],
            "initial_validation": initial_validation, "selected_validation": best_score,
            "bptt_visits": config.get("bptt_visits", 0), "full_episode_bptt": config.get("bptt_visits", 0) == 0,
            "label_mode": config["label_mode"], "iteration_policy": config["iteration_policy"],
            "accumulation_visits_global": config["accumulation_visits"],
            "gradient_reduction": "sum rank CE gradients / sum rank visited-head counts; clip after reduction",
            "trace_training": trace_enabled, "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": max(w["peak_cuda_allocated_bytes"] for w in workers),
            "training_method": "synchronous data parallel; token-index iteration with teacher-forced continuation"})
        print(f"Parallel training complete: GPUs={[w['device']['physical_index'] for w in workers]} "
              f"targets={totals[1]}/{expected} updates={updates} replicas_identical=True", flush=True)
    runtime.barrier()
