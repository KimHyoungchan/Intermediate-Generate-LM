from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import yaml

from .common import ROOT, isolate_gpu


def main():
    parser = argparse.ArgumentParser(description="No-train vs trained-readout next-token experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", "prepare", "before", "train", "after", "compare"), default="all")
    parser.add_argument("--gpu", type=int, default=1, help="Physical nvidia-smi GPU index: 1, 2, or 3")
    parser.add_argument("--resume", action="store_true", help="Reuse verified completed stages; restart an incomplete stage")
    args = parser.parse_args()
    if args.gpu not in (1, 2, 3):
        parser.error("Only physical GPUs 1, 2, 3 may be used. GPU 0 is forbidden.")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("mode", "teacher_forced") != "teacher_forced":
        parser.error("This implementation supports teacher-forced evaluation only.")
    if config.get("batch_size", 1) != 1:
        parser.error("Use batch_size=1; projection_chunk_tokens controls LM-head memory.")
    for key in ("model_id", "model_revision", "output_dir", "seq_len", "context_overlap", "adapter"):
        if key not in config:
            parser.error(f"Missing config key: {key}")
    if not 1 <= config["context_overlap"] < config["seq_len"]:
        parser.error("Require 1 <= context_overlap < seq_len")
    if config["adapter"]["kind"] not in ("dense", "low_rank"):
        parser.error("adapter.kind must be dense or low_rank")
    if config["top_k"] < 2 or config["projection_chunk_tokens"] < 1:
        parser.error("Require top_k >= 2 and projection_chunk_tokens >= 1")
    if args.stage == "all":
        for stage in ("prepare", "before", "train", "after", "compare"):
            command = [sys.executable, str(ROOT / "run.py"), "--config", str(config_path),
                       "--stage", stage, "--gpu", str(args.gpu)]
            if args.resume:
                command.append("--resume")
            print(f"\n=== {stage} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        return
    mapping = isolate_gpu(args.gpu, cpu=args.stage in ("prepare", "compare"))
    print(f"Device mapping: {mapping}", flush=True)
    if args.stage == "prepare":
        from .data import prepare
        prepare(config)
    elif args.stage in ("before", "after"):
        from .evaluate import evaluate
        evaluate(config, trained=args.stage == "after", mapping=mapping, resume=args.resume)
    elif args.stage == "train":
        from .train import train
        train(config, mapping=mapping, resume=args.resume)
    elif args.stage == "compare":
        from .compare import compare
        compare(config)
