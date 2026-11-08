from pathlib import Path
import argparse
import subprocess
import sys
import yaml
from .common import ROOT, root_for, isolate_gpu


def main():
    parser = argparse.ArgumentParser(description="Token-index iteration curriculum with forced continuation on training misses")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--label-mode", choices=("next_token", "input_token"))
    parser.add_argument("--iteration-policy", choices=("per_token", "global", "advance_on_last"))
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--stage", choices=("all", "prepare", "before", "train", "after", "compare"), default="all")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text())
    if args.label_mode:
        config["label_mode"] = args.label_mode
    if args.iteration_policy:
        config["iteration_policy"] = args.iteration_policy
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if config.get("label_mode") not in ("next_token", "input_token"):
        parser.error("Choose label_mode explicitly: next_token (A->B) or input_token (A->A)")
    if config.get("iteration_policy") not in ("per_token", "global", "advance_on_last"):
        parser.error("Choose iteration_policy explicitly: per_token, global, or advance_on_last")
    for key in ("epochs", "rank", "accumulation_visits", "eval_every_updates", "train_episodes", "validation_episodes", "eval_episodes"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if not isinstance(config.get("bptt_visits", 0), int) or config.get("bptt_visits", 0) < 0:
        parser.error("bptt_visits must be nonnegative; 0 keeps the entire episode graph")
    root = root_for(config)
    root.mkdir(parents=True, exist_ok=True)
    resolved = root / "resolved_config.yaml"
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != config:
        parser.error("Config changed: choose a fresh output_dir")
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.stage == "all":
        for stage in ("prepare", "before", "train", "after", "compare"):
            command = [sys.executable, str(ROOT / "run_training_revision.py"), "--config", str(resolved),
                       "--stage", stage, "--gpu", str(args.gpu)]
            if args.resume:
                command.append("--resume")
            print(f"\n=== revised {stage} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        return
    mapping = isolate_gpu(args.gpu, cpu=args.stage in ("prepare", "compare"))
    print(f"Revised training device: {mapping}; labels={config['label_mode']}; iterations={config['iteration_policy']}", flush=True)
    if args.stage == "prepare":
        from routed_lm.data import prepare
        prepare(config)
    else:
        from .workflow import evaluate, train, compare
        if args.stage in ("before", "after"):
            evaluate(config, trained=args.stage == "after", mapping=mapping, resume=args.resume)
        elif args.stage == "train":
            train(config, mapping=mapping, resume=args.resume)
        else:
            compare(config)
