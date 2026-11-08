from pathlib import Path
import argparse
import subprocess
import sys
import yaml
from .common import ROOT, isolate_gpu


def main():
    parser = argparse.ArgumentParser(description="Actual same-layer token insertion and state-carrying L-to-1 recurrence")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--stage", default="all", choices=("all", "prepare", "before", "train", "after", "compare"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = args.config.resolve()
    config = yaml.safe_load(path.read_text())
    for key in ("context_tokens", "continuation_tokens", "train_episodes", "validation_episodes", "eval_episodes",
                "rank", "epochs", "bptt_visits", "accumulation_visits", "eval_every_updates"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if args.stage == "all":
        for stage in ("prepare", "before", "train", "after", "compare"):
            command = [sys.executable, str(ROOT / "run_routed.py"), "--config", str(path), "--gpu", str(args.gpu), "--stage", stage]
            if args.resume:
                command.append("--resume")
            print(f"\n=== routed {stage} ===", flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
        return
    mapping = isolate_gpu(args.gpu, cpu=args.stage in ("prepare", "compare"))
    print(f"Routed device: {mapping}", flush=True)
    if args.stage == "prepare":
        from .data import prepare
        prepare(config)
    else:
        from .workflow import evaluate, train, compare
        if args.stage in ("before", "after"):
            evaluate(config, args.stage == "after", mapping, args.resume)
        elif args.stage == "train":
            train(config, mapping, args.resume)
        else:
            compare(config)
