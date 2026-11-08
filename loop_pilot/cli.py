from pathlib import Path
import argparse
import sys
import yaml

from layerwise_lm.common import ROOT, isolate_gpu


def main():
    parser = argparse.ArgumentParser(description="Measure loops to completion; no per-token visit cap")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--condition", choices=("both", "before", "after", "report"), default="both")
    parser.add_argument("--max-seconds", type=float, help="Episode watchdog seconds; 0 means no timeout")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text())
    if args.max_seconds is not None:
        config["max_seconds_per_episode"] = args.max_seconds
    if args.episodes is not None:
        config["episodes"] = args.episodes
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    for key in ("episodes", "log_every_wraps"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            parser.error(f"{key} must be a positive integer")
    for key in ("max_seconds_per_episode", "max_total_visits_per_episode"):
        if not isinstance(config.get(key), (int, float)) or config[key] < 0:
            parser.error(f"{key} must be nonnegative; 0 disables this watchdog")
    mapping = isolate_gpu(args.gpu, cpu=args.condition == "report")
    print(f"Loop pilot device: {mapping}", flush=True)
    from .runner import run
    run(config, args.condition, mapping, args.resume)
