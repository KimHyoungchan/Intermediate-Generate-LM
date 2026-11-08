from __future__ import annotations

import csv
import json
from pathlib import Path
import torch
import yaml

from layerwise_lm.common import ROOT, digest, file_hash, write_json, read_json, versions
from routed_lm.common import source_hash as routed_hash, verify as verify_routed, root_for
from routed_lm.data import verified, read_episodes
from routed_lm.network import load
from .core import profile_episode, aggregate


def source_hash():
    paths = [ROOT / "run_loop_pilot.py", *sorted((ROOT / "loop_pilot").glob("*.py"))]
    return digest({"pilot": {str(p.relative_to(ROOT)): file_hash(p) for p in paths}, "routed": routed_hash()})


def identity(config):
    return {"config_hash": digest(config), "loop_pilot_source_hash": source_hash()}


def csv_write(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, condition, mapping, resume):
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    if condition == "report":
        from .report import report
        report(output, config)
        return
    base_path = Path(config["source_experiment_config"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base_config = yaml.safe_load(base_path.read_text())
    data_root, data_manifest = verified(base_config)
    all_episodes = read_episodes(data_root / "test.jsonl")
    if config["episodes"] > len(all_episodes):
        raise ValueError(f"Only {len(all_episodes)} fixed test episodes are available")
    episodes = all_episodes[:config["episodes"]]
    source_root = root_for(base_config)
    selection = {"episode_ids": [e["episode_id"] for e in episodes],
                 "test_sha256": data_manifest["splits"]["test"]["sha256"],
                 "targets_sha256": digest(episodes)}
    selected = ("before", "after") if condition == "both" else (condition,)
    pending = []
    for name in selected:
        path = output / name / "summary.json"
        if path.exists():
            saved = read_json(path)
            if any(saved.get(k) != v for k, v in identity(config).items()) or saved["selection"] != selection:
                raise ValueError("Changed pilot config/source/data; choose a fresh output_dir")
            if not resume:
                raise FileExistsError(f"Completed pilot exists: {path}; use --resume or new output_dir")
            if name == "after" and file_hash(source_root / "training/best.pt") != saved["checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed; choose a new output_dir")
            if saved.get("condition_finished", True):
                print(f"Reusing pilot: {path}", flush=True)
            else:
                print(f"Restarting interrupted condition (cache-state resume is not supported): {name}", flush=True)
                pending.append(name)
        else:
            pending.append(name)
    write_json(output / "resolved_config.json", {**identity(config), "config": config, "selection": selection})
    network = load(base_config) if pending else None
    frozen_hash = network.frozen_hash() if network is not None else None
    interrupted = False
    for name in pending:
        checkpoint_hash = None
        if name == "after":
            done = read_json(source_root / "training/done.json")
            verify_routed(done, base_config)
            path = source_root / "training/best.pt"
            checkpoint_hash = file_hash(path)
            if checkpoint_hash != done["checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["frozen_hash"] != frozen_hash or payload["data_manifest_hash"] != file_hash(data_root / "manifest.json"):
                raise ValueError("Checkpoint/model/dataset mismatch")
            network.restore_trainable(payload["parameters"])
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        episode_rows, token_rows = [], []
        with (folder / "progress.jsonl").open("w") as progress:
            for index, episode in enumerate(episodes):
                print(f"{name}: episode {index+1}/{len(episodes)}, target tokens={len(episode['targets'])}; per-token cap=NONE", flush=True)
                def emit(event):
                    progress.write(json.dumps(event) + "\n")
                    progress.flush()
                    if event["event"] == "hit" or event["total_wraps"] % config["log_every_wraps"] == 0:
                        print(f"  target={event['target_index']} loops={event['target_wraps']} "
                              f"resolved={event['resolved_targets']} blocks={event['visited_blocks']} "
                              f"elapsed={event['elapsed_seconds']:.1f}s", flush=True)
                result, tokens, _ = profile_episode(network, episode,
                    seconds=config["max_seconds_per_episode"],
                    max_total_visits=config["max_total_visits_per_episode"],
                    fingerprint=config.get("observe_repeated_states", True), progress=emit)
                episode_rows.append(result)
                token_rows.extend(tokens)
                csv_write(folder / "episodes.csv", episode_rows)
                csv_write(folder / "tokens.csv", token_rows)
                print(f"  {result['outcome']}: resolved={result['resolved_targets']}/{result['requested_targets']} "
                      f"observed_loops={result['observed_wraps']} ({result['stop_reason']})", flush=True)
                if result["stop_reason"] == "user_interrupt":
                    interrupted = True
                    break
        summary = aggregate(episode_rows, token_rows, len(episodes))
        summary.update({**identity(config), "condition": name, "model_id": base_config["model_id"],
            "condition_finished": not interrupted and len(episode_rows) == len(episodes),
            "source_experiment_config": str(base_path), "training_visit_budget": base_config["max_visits_per_token"],
            "evaluation_per_token_visit_budget": None, "selection": selection,
            "requested_targets_selected": sum(len(e["targets"]) for e in episodes),
            "requested_targets_profiled": len(token_rows),
            "checkpoint_sha256": checkpoint_hash, "frozen_hash": frozen_hash,
            "max_seconds_per_episode": config["max_seconds_per_episode"],
            "max_total_visits_per_episode": config["max_total_visits_per_episode"],
            "device": mapping, "versions": versions(),
            "episodes_sha256": file_hash(folder / "episodes.csv"),
            "tokens_sha256": file_hash(folder / "tokens.csv"),
            "routing": "oracle; actual same-layer insertion and state-carrying wrap",
            "retraining": False,
            "state_repeat_policy": "last 4096 distinct wrap-boundary states of current token; observation only; never an early-stop criterion"})
        write_json(folder / "summary.json", summary)
        if interrupted:
            break
    from .report import report
    report(output, config)
    if interrupted:
        raise SystemExit(130)
