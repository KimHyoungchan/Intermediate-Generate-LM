from __future__ import annotations

import csv
import json
import time
import random
import torch

from .common import root_for, identity, verify, completed, read_json, write_json, file_hash, versions
from .data import verified, read_episodes
from .network import load
from .engine import walk


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_episodes(network, episodes, budget, trace_stream=None):
    rows, episode_rows = [], []
    visits, loss_sum, wraps, injections = 0, 0.0, 0, 0
    for episode in episodes:
        targets = [{"episode_id": episode["episode_id"], "target_index": i, "target_id": t,
                    "status": "unreached", "start_layer": None, "exit_layer": None,
                    "visits": 0, "wraps": 0} for i, t in enumerate(episode["targets"])]
        count, episode_loss = 0, 0.0
        for event, loss, state in walk(network, episode, budget):
            row = targets[event["target_index"]]
            if row["start_layer"] is None:
                row["start_layer"] = event["layer"]
            row["visits"] = event["visits_for_target"]
            row["status"] = "hit" if event["accepted"] else "failed" if event["action"] == "budget_exhausted" else "attempted"
            if event["accepted"]:
                row["exit_layer"] = event["layer"]
            if event["action"] == "wrap":
                row["wraps"] += 1
                wraps += 1
            injections += int(event["action"] == "emit")
            count += 1
            episode_loss += loss.item()
            if trace_stream:
                trace_stream.write(json.dumps(event) + "\n")
        assert not any(row["status"] == "attempted" for row in targets)
        resolved = sum(r["status"] == "hit" for r in targets)
        episode_rows.append({"episode_id": episode["episode_id"], "document_id": episode["document_id"],
                "requested_targets": len(targets), "resolved_targets": resolved,
                "completed": resolved == len(targets), "decode_block_visits": count})
        rows.extend(targets)
        visits += count
        loss_sum += episode_loss
    requested = len(rows)
    hit_rows = [r for r in rows if r["status"] == "hit"]
    failed = sum(r["status"] == "failed" for r in rows)
    summary = {"episodes": len(episodes), "requested_targets": requested,
        "resolved_targets": len(hit_rows), "failed_targets": failed,
        "unreached_targets": sum(r["status"] == "unreached" for r in rows),
        "resolved_requested_fraction": len(hit_rows) / requested,
        "episode_completion_fraction": sum(r["completed"] for r in episode_rows) / len(episodes),
        "decode_block_visits": visits, "decode_head_calls": visits, "wrap_transitions": wraps,
        "same_layer_token_injections": injections,
        "mean_visits_per_hit": sum(r["visits"] for r in hit_rows) / len(hit_rows) if hit_rows else None,
        "mean_exit_layer_given_hit": sum(r["exit_layer"] for r in hit_rows) / len(hit_rows) if hit_rows else None,
        "decode_visits_per_resolved_target_including_failures": visits / len(hit_rows) if hit_rows else None,
        "mean_nll_per_visited_block": loss_sum / max(visits, 1),
        "prompt_prefill_block_token_evaluations": sum((len(e["context"]) - 1) * network.n_layers for e in episodes),
        "layer_visits_budget_per_target": budget,
        "routing": "oracle exact next-token acceptance; strict stop at exhausted budget; no forced target insertion",
        "cache": "latest K/V for visited input positions only; same-position revisits replace K/V",
        "prefill": "full-depth prompt[:-1] only; last prompt token starts routed traversal at layer 1"}
    return summary, rows, episode_rows


def evaluate(config, trained, mapping, resume):
    root = root_for(config)
    data_root, data_manifest = verified(config)
    output = root / ("after_train" if trained else "before_train")
    if completed(output / "summary.json", config, resume):
        saved = read_json(output / "summary.json")
        if file_hash(output / "trace.jsonl") != saved["trace_sha256"]:
            raise ValueError("Routed trace changed")
        if trained and file_hash(root / "training/best.pt") != saved["checkpoint_sha256"]:
            raise ValueError("Routed checkpoint changed")
        return
    output.mkdir(parents=True, exist_ok=True)
    network = load(config)
    frozen_hash = network.frozen_hash()
    checkpoint_hash = None
    if trained:
        done = read_json(root / "training/done.json")
        verify(done, config)
        path = root / "training/best.pt"
        checkpoint_hash = file_hash(path)
        if checkpoint_hash != done["checkpoint_sha256"]:
            raise ValueError("Invalid routed checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["frozen_hash"] != frozen_hash or checkpoint["data_manifest_hash"] != file_hash(data_root / "manifest.json"):
            raise ValueError("Routed model/data mismatch")
        network.restore_trainable(checkpoint["parameters"])
    episodes = read_episodes(data_root / "test.jsonl")
    budget = config.get("max_visits_per_token") or network.n_layers * 2
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    with (output / "trace.jsonl").open("w") as trace:
        summary, token_rows, episode_rows = evaluate_episodes(network, episodes, budget, trace)
    torch.cuda.synchronize()
    save_csv(output / "tokens.csv", token_rows)
    save_csv(output / "episodes.csv", episode_rows)
    summary.update({**identity(config), "condition": "trained routed model" if trained else "untrained routed model",
        "device": mapping, "versions": versions(), "model_id": config["model_id"],
        "elapsed_seconds": time.monotonic() - start, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "frozen_hash": frozen_hash, "checkpoint_sha256": checkpoint_hash,
        "eval_data_sha256": data_manifest["splits"]["test"]["sha256"],
        "trace_sha256": file_hash(output / "trace.jsonl"), "tokens_sha256": file_hash(output / "tokens.csv"),
        "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad)})
    write_json(output / "summary.json", summary)
    print(f"Routed evaluation: resolved {summary['resolved_targets']}/{summary['requested_targets']}; "
          f"injections={summary['same_layer_token_injections']}, wraps={summary['wrap_transitions']}", flush=True)


def component(name):
    if ".lora_" in name:
        return "transformer_lora"
    if "wrap" in name:
        return "wrap"
    if "exit" in name:
        return "exit"
    return "token_entry_and_memory"


def train(config, mapping, resume):
    root = root_for(config)
    data_root, _ = verified(config)
    output = root / "training"
    if completed(output / "done.json", config, resume):
        if file_hash(output / "best.pt") != read_json(output / "done.json")["checkpoint_sha256"]:
            raise ValueError("Routed checkpoint changed")
        return
    baseline = read_json(root / "before_train/summary.json")
    verify(baseline, config)
    network = load(config)
    frozen_hash = network.frozen_hash()
    if frozen_hash != baseline["frozen_hash"]:
        raise ValueError("Frozen backbone differs from before-training model")
    initial = network.trainable_state()
    parameters = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    episodes = read_episodes(data_root / "train.jsonl")
    validation = read_episodes(data_root / "validation.jsonl")
    budget = config.get("max_visits_per_token") or network.n_layers * 2
    output.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    pending_visits, updates, visits, emitted, processed = 0, 0, 0, 0, 0
    best_score, best_validation, selected_update = None, None, 0
    last_validation_update = -1
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    initial_validation, _, _ = evaluate_episodes(network, validation, budget)
    log = (output / "log.jsonl").open("w")
    trace = (output / "train_trace.jsonl").open("w")

    def validate_and_save():
        nonlocal best_score, best_validation, selected_update, last_validation_update
        summary, _, _ = evaluate_episodes(network, validation, budget)
        score = (summary["resolved_requested_fraction"], -summary["mean_nll_per_visited_block"])
        last_validation_update = updates
        row = {"update": updates, "block_visits": visits, "accepted_targets": emitted,
            "validation_resolved_fraction": summary["resolved_requested_fraction"],
            "validation_nll": summary["mean_nll_per_visited_block"], "elapsed_seconds": time.monotonic() - started}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if best_score is None or score > best_score:
            best_score, best_validation, selected_update = score, summary, updates
            payload = {"parameters": network.trainable_state(), "update": updates,
                       "frozen_hash": frozen_hash, "data_manifest_hash": file_hash(data_root / "manifest.json"), **identity(config)}
            temporary = output / "best.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(output / "best.pt")

    def update():
        nonlocal pending_visits, updates
        for p in parameters:
            if p.grad is not None:
                p.grad.div_(pending_visits)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Routed non-finite gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending_visits = 0
        updates += 1
        print(f"routed update={updates} visited_blocks={visits} accepted_targets={emitted} episodes={processed}", flush=True)
        if updates % config["eval_every_updates"] == 0:
            validate_and_save()

    try:
        for epoch in range(config["epochs"]):
            ordered = list(episodes)
            random.Random(config["seed"] + epoch).shuffle(ordered)
            for episode in ordered:
                losses = []
                for event, loss, state in walk(network, episode, budget):
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Routed non-finite loss")
                    losses.append(loss)
                    visits += 1
                    emitted += int(event["accepted"])
                    trace.write(json.dumps({"epoch": epoch + 1, **event}) + "\n")
                    end = event["action"] in ("complete", "budget_exhausted")
                    if len(losses) >= config["bptt_visits"] or end:
                        torch.stack(losses).sum().backward()
                        pending_visits += len(losses)
                        losses = []
                        # Detach BEFORE generator executes token injection/wrap.
                        state.detach_()
                        if pending_visits >= config["accumulation_visits"]:
                            update()
                assert not losses
                processed += 1
        if pending_visits:
            update()
        if updates == 0:
            raise ValueError("No routed optimizer updates")
        if last_validation_update != updates:
            validate_and_save()
    finally:
        log.close()
        trace.close()
    assert network.frozen_hash() == frozen_hash
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)["parameters"]
    changed = {}
    for name, value in selected.items():
        group = component(name)
        changed[group] = changed.get(group, 0.0) + (value - initial[name]).square().sum().item()
    changed = {name: value**0.5 for name, value in changed.items()}
    write_json(output / "done.json", {**identity(config), "device": mapping, "updates": updates,
        "selected_update": selected_update, "epochs": config["epochs"], "episodes_processed": processed,
        "visited_blocks": visits, "accepted_targets": emitted, "initial_validation": initial_validation,
        "selected_validation": best_validation, "component_parameter_change_l2": changed,
        "frozen_hash": frozen_hash, "frozen_weights_unchanged": True,
        "checkpoint_sha256": file_hash(output / "best.pt"), "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "objective": "reference CE at each actually visited block, truncated BPTT through visited states",
        "training_path": "same-layer emit, next-layer reject, state-carrying wrap; strict stop on budget exhaustion",
        "bptt_visits": config["bptt_visits"], "resume_policy": "completed stage reuse only"})


def compare(config):
    root = root_for(config)
    summaries = [read_json(root / folder / "summary.json") for folder in ("before_train", "after_train")]
    for summary in summaries:
        verify(summary, config)
    a, b = summaries
    for key in ("frozen_hash", "eval_data_sha256", "requested_targets"):
        if a[key] != b[key]:
            raise ValueError(f"Routed A/B {key} mismatch")
    def records(folder):
        with (root / folder / "tokens.csv").open() as stream:
            return list(csv.DictReader(stream))
    aa, bb = records("before_train"), records("after_train")
    assert [(r["episode_id"], r["target_index"], r["target_id"]) for r in aa] == [(r["episode_id"], r["target_index"], r["target_id"]) for r in bb]
    common = [(x, y) for x, y in zip(aa, bb) if x["status"] == y["status"] == "hit"]
    comparison = []
    for key in ("resolved_requested_fraction", "episode_completion_fraction", "resolved_targets", "failed_targets", "unreached_targets",
                "same_layer_token_injections", "wrap_transitions", "mean_visits_per_hit", "mean_exit_layer_given_hit",
                "decode_visits_per_resolved_target_including_failures", "mean_nll_per_visited_block", "decode_block_visits"):
        before, after = a[key], b[key]
        comparison.append({"metric": key, "before": before, "after": after,
                           "delta": after - before if before is not None and after is not None else None})
    save_csv(root / "comparison.csv", comparison)
    change = sum(int(y["visits"]) - int(x["visits"]) for x, y in common) / len(common) if common else None
    summary = {**identity(config), "common_hit_targets": len(common), "common_hit_visits_delta": change,
        "newly_resolved_targets": sum(x["status"] != "hit" and y["status"] == "hit" for x, y in zip(aa, bb)),
        "lost_resolved_targets": sum(x["status"] == "hit" and y["status"] != "hit" for x, y in zip(aa, bb)),
        "checkpoint_sha256": b["checkpoint_sha256"], "eval_data_sha256": a["eval_data_sha256"]}
    write_json(root / "pair_manifest.json", summary)
    lines = ["# 실제 순환 경로 학습 전후 비교", "", "정답을 맞히면 즉시 다음 token을 같은 block에 넣고, 거기서 계산을 계속한 실행 결과다.", "",
             "| 지표 | 학습 전 | 학습 후 | 변화량 |", "| --- | ---: | ---: | ---: |"]
    for row in comparison:
        lines.append(f"| {row['metric']} | {row['before']} | {row['after']} | {row['delta']} |")
    lines += ["", f"공통 성공 target: {len(common)}, 공통 target의 실제 block 방문 수 변화: {change}.", "",
        "평가는 실제 다음 token label로 분기를 선택하는 oracle 실험이다. 실제 배포용 exit gate는 포함하지 않는다.", "",
        "예산을 소진하면 해당 episode를 종료한다. 정답을 강제로 주입하지 않으며 이후 target은 unreached로 남긴다.", "",
        "성공률·완료율과 함께 계산량을 읽어야 한다. 일찍 실패한 실행의 작은 총 방문 수를 가속으로 해석하지 않는다.", "",
        "Trace에는 매 block의 input/target/layer/action/next_layer/injected token을 저장한다. KV는 방문한 위치만 가지며 재방문은 같은 위치를 교체한다.", ""]
    (root / "comparison.md").write_text("\n".join(lines))
    print(f"Routed report: {root / 'comparison.md'}", flush=True)
