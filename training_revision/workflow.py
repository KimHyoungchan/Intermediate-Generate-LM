from __future__ import annotations

import csv
import random
import time
import torch

from .common import root_for, identity, verify, completed, read_json, write_json, file_hash
from routed_lm.data import verified, read_episodes
from routed_lm.network import load
from routed_lm.workflow import save_csv, component
from .engine import walk_training


@torch.no_grad()
def score(network, episodes, config, trace=None):
    visits, loss_sum = 0, 0.
    tokens, episode_rows = [], []
    for episode in episodes:
        rows = [{"episode_id": episode["episode_id"], "target_index": i, "target_id": t,
                 "status": "unreached", "visits": 0, "exit_layer": None,
                 "iteration_at_exit": None} for i, t in enumerate(episode["targets"])]
        for event, loss, _ in walk_training(network, episode, config["label_mode"], config["iteration_policy"]):
            visits += 1
            loss_sum += loss.item()
            row = rows[event["target_index"]]
            row["visits"] = event["visits_for_target"]
            if event["accepted"] or event["forced"]:
                row.update(status="hit" if event["accepted"] else "forced",
                           exit_layer=event["layer"], iteration_at_exit=event["iteration"])
            if trace:
                import json
                trace.write(json.dumps(event) + "\n")
        assert all(row["status"] != "unreached" for row in rows)
        hit_count = sum(row["status"] == "hit" for row in rows)
        episode_rows.append({"episode_id": episode["episode_id"], "targets": len(rows),
            "unforced_hits": hit_count, "forced_targets": len(rows) - hit_count,
            "all_correct_without_forcing": hit_count == len(rows)})
        tokens.extend(rows)
    hits = sum(row["status"] == "hit" for row in tokens)
    summary = {"requested_targets": len(tokens), "processed_targets": len(tokens), "unreached_targets": 0,
        "unforced_hits": hits, "forced_targets": len(tokens) - hits,
        "unforced_hit_fraction": hits / len(tokens),
        "all_correct_without_forcing_fraction": sum(e["all_correct_without_forcing"] for e in episode_rows) / len(episodes),
        "visited_blocks": visits, "mean_ce_per_visited_block": loss_sum / visits,
        "label_mode": config["label_mode"], "iteration_policy": config["iteration_policy"],
        "teacher_forcing": "After iteration exhaustion, supply the correct next input and continue at layer 1",
        "completion_semantics": "Processing all labels is not counted as correct free generation"}
    return summary, tokens, episode_rows


def restore(network, config):
    root = root_for(config)
    done = read_json(root / "training/done.json")
    verify(done, config)
    path = root / "training/best.pt"
    if file_hash(path) != done["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["frozen_hash"] != network.frozen_hash() or payload["data_manifest_hash"] != file_hash(root / "data/manifest.json"):
        raise ValueError("Checkpoint/model/data mismatch")
    network.restore_trainable(payload["parameters"])
    return done["checkpoint_sha256"]


def evaluate(config, trained, mapping, resume):
    root = root_for(config)
    data_root, manifest = verified(config)
    folder = root / ("after_train" if trained else "before_train")
    if completed(folder / "summary.json", config, resume):
        return
    folder.mkdir(parents=True, exist_ok=True)
    network = load(config)
    frozen = network.frozen_hash()
    checkpoint_hash = restore(network, config) if trained else None
    started = time.monotonic()
    with (folder / "trace.jsonl").open("w") as trace:
        summary, tokens, episodes = score(network, read_episodes(data_root / "test.jsonl"), config, trace)
    save_csv(folder / "tokens.csv", tokens)
    save_csv(folder / "episodes.csv", episodes)
    summary.update({**identity(config), "device": mapping, "model_id": config["model_id"],
        "frozen_hash": frozen, "checkpoint_sha256": checkpoint_hash,
        "eval_data_sha256": manifest["splits"]["test"]["sha256"],
        "elapsed_seconds": time.monotonic() - started, "condition": "after" if trained else "before"})
    write_json(folder / "summary.json", summary)
    print(f"Revised evaluation: processed={summary['processed_targets']}/{summary['requested_targets']} "
          f"hits={summary['unforced_hits']} forced={summary['forced_targets']}", flush=True)


def train(config, mapping, resume):
    import json
    root = root_for(config)
    data_root, _ = verified(config)
    folder = root / "training"
    if completed(folder / "done.json", config, resume):
        return
    baseline = read_json(root / "before_train/summary.json")
    verify(baseline, config)
    network = load(config)
    frozen = network.frozen_hash()
    assert frozen == baseline["frozen_hash"]
    initial = network.trainable_state()
    parameters = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    episodes = read_episodes(data_root / "train.jsonl")
    validation = read_episodes(data_root / "validation.jsonl")
    folder.mkdir(parents=True, exist_ok=True)
    pending, visits, updates, processed, hits, forced, episode_count = 0, 0, 0, 0, 0, 0, 0
    best_key, best_score, selected_update, last_validation = None, None, 0, -1
    initial_validation, _, _ = score(network, validation, config)
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    log = (folder / "log.jsonl").open("w")
    trace = (folder / "train_trace.jsonl").open("w")

    def validate_and_save():
        nonlocal best_key, best_score, selected_update, last_validation
        validation_score, _, _ = score(network, validation, config)
        key = (validation_score["unforced_hit_fraction"], -validation_score["mean_ce_per_visited_block"])
        last_validation = updates
        log.write(json.dumps({"update": updates, "visited_blocks": visits, "processed_targets": processed,
            "unforced_hits": hits, "forced_targets": forced, "validation": validation_score,
            "elapsed_seconds": time.monotonic() - started}) + "\n")
        log.flush()
        if best_key is None or key > best_key:
            best_key, best_score, selected_update = key, validation_score, updates
            payload = {"parameters": network.trainable_state(), "update": updates,
                "frozen_hash": frozen, "data_manifest_hash": file_hash(data_root / "manifest.json"),
                "training_method": "token-index iteration with forced continuation", **identity(config)}
            temporary = folder / "best.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(folder / "best.pt")

    def update():
        nonlocal pending, updates
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        updates += 1
        print(f"revision update={updates} episodes={episode_count} processed={processed} "
              f"hits={hits} forced={forced} visited_blocks={visits}", flush=True)
        if updates % config["eval_every_updates"] == 0:
            validate_and_save()

    try:
        for epoch in range(config["epochs"]):
            order = list(episodes)
            random.Random(config["seed"] + epoch).shuffle(order)
            for episode in order:
                losses = []
                for event, loss, state in walk_training(network, episode, config["label_mode"], config["iteration_policy"]):
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss")
                    losses.append(loss)
                    visits += 1
                    if event["accepted"] or event["forced"]:
                        processed += 1
                        hits += int(event["accepted"])
                        forced += int(event["forced"])
                    trace.write(json.dumps({"epoch": epoch + 1, **event}) + "\n")
                    end = event["action"] in ("complete_hit", "complete_forced")
                    truncate = config.get("bptt_visits", 0) > 0 and len(losses) >= config["bptt_visits"]
                    if end or truncate:
                        torch.stack(losses).sum().backward()
                        pending += len(losses)
                        losses = []
                        state.detach_()
                        if pending >= config["accumulation_visits"]:
                            update()
                assert not losses
                episode_count += 1
        if pending:
            update()
        if updates == 0:
            raise RuntimeError("No optimizer update")
        if last_validation != updates:
            validate_and_save()
    finally:
        log.close()
        trace.close()
    expected = sum(len(e["targets"]) for e in episodes) * config["epochs"]
    assert processed == expected, (processed, expected)
    assert network.frozen_hash() == frozen
    state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)["parameters"]
    changes = {}
    for name, value in state.items():
        group = component(name)
        changes[group] = changes.get(group, 0.) + (value - initial[name]).square().sum().item()
    changes = {k: v**0.5 for k, v in changes.items()}
    write_json(folder / "done.json", {**identity(config), "device": mapping,
        "updates": updates, "selected_update": selected_update, "epochs": config["epochs"],
        "requested_training_targets": expected, "processed_training_targets": processed, "unreached_training_targets": 0,
        "unforced_training_hits": hits, "forced_training_targets": forced, "visited_blocks": visits,
        "frozen_hash": frozen, "frozen_weights_unchanged": True,
        "checkpoint_sha256": file_hash(folder / "best.pt"), "component_parameter_change_l2": changes,
        "initial_validation": initial_validation, "selected_validation": best_score,
        "bptt_visits": config.get("bptt_visits", 0), "full_episode_bptt": config.get("bptt_visits", 0) == 0,
        "label_mode": config["label_mode"], "iteration_policy": config["iteration_policy"],
        "elapsed_seconds": time.monotonic() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_method": "token-index iteration with teacher-forced continuation; visited-head reference CE"})


def compare(config):
    root = root_for(config)
    a = read_json(root / "before_train/summary.json")
    b = read_json(root / "after_train/summary.json")
    for value in (a, b):
        verify(value, config)
    for key in ("frozen_hash", "eval_data_sha256", "label_mode", "iteration_policy", "requested_targets"):
        assert a[key] == b[key], key
    def read_tokens(folder):
        with (root / folder / "tokens.csv").open() as stream:
            return list(csv.DictReader(stream))
    aa, bb = read_tokens("before_train"), read_tokens("after_train")
    assert [(t["episode_id"], t["target_index"], t["target_id"]) for t in aa] == [(t["episode_id"], t["target_index"], t["target_id"]) for t in bb]
    rows = [{"metric": key, "before": a[key], "after": b[key], "delta": b[key] - a[key]}
            for key in ("processed_targets", "unforced_hits", "forced_targets", "unforced_hit_fraction",
                        "all_correct_without_forcing_fraction", "visited_blocks", "mean_ce_per_visited_block")]
    save_csv(root / "comparison.csv", rows)
    lines = ["# 새 iteration 학습 전후 비교", "", f"정답 정렬: {config['label_mode']}; iteration: {config['iteration_policy']}.", "",
             "| 지표 | 학습 전 | 학습 후 | 변화량 |", "| --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['before']} | {row['after']} | {row['delta']} |")
    lines += ["", "실패한 token은 loss를 남긴 뒤 정답 입력으로 다음 token 학습을 계속한다. 따라서 processed_targets는 생성 성공 수가 아니다.", "",
        "unforced_hits만 head가 정답을 직접 맞힌 token이며 forced_targets는 iteration 예산 후 정답을 공급한 token이다.", "",
        "Token index는 context를 제외한 target sequence의 0-based index다. 강제 전환은 layer 1부터, 정답 전환은 같은 layer부터 시작한다.", "",
        "input_token 정렬은 이미 입력에 있는 token 복원 과제이며 next_token 예측 성능과 구분한다.", ""]
    (root / "comparison.md").write_text("\n".join(lines))
    print(f"Revised training report: {root / 'comparison.md'}", flush=True)
