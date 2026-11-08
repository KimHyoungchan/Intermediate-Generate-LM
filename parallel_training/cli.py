from pathlib import Path
import argparse
import os
import subprocess
import sys
import yaml

from layerwise_lm.common import gpu_inventory, isolate_gpu
from .common import ROOT, root_for, identity, verify, read_json, write_json, completed


def gpu_list(value):
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use physical GPU indices, e.g. 1,2,3") from error
    if not result or any(index not in (1, 2, 3) for index in result):
        raise argparse.ArgumentTypeError("Only physical GPUs 1,2,3 are allowed; GPU 0 is forbidden")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU training of the routed next-token objective")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpus", type=gpu_list, help="Physical GPU indices, e.g. 1,2,3 or 1,3")
    parser.add_argument("--stage", choices=("all", "prepare", "before", "train", "after", "compare"), default="all")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--accumulation-visits", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages; interrupted training restarts")
    parser.add_argument("--distributed-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text())
    for name in ("output_dir", "epochs", "train_episodes", "accumulation_visits"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    gpus = args.gpus if args.gpus is not None else gpu_list(",".join(map(str, config.get("training_gpus", [1, 2, 3]))))
    config["training_gpus"] = gpus
    for name in ("context_tokens", "continuation_tokens", "epochs", "rank", "train_episodes",
                 "validation_episodes", "eval_episodes", "accumulation_visits", "eval_every_updates"):
        if type(config.get(name)) is not int or config[name] < 1:
            parser.error(f"{name} must be a positive integer")
    if type(config.get("bptt_visits", 0)) is not int or config.get("bptt_visits", 0) < 0:
        parser.error("bptt_visits must be nonnegative")
    if config.get("label_mode") != "next_token" or config.get("iteration_policy") != "global":
        parser.error("This experiment requires label_mode=next_token and iteration_policy=global")
    if float(config.get("learning_rate", 0)) <= 0:
        parser.error("learning_rate must be positive")
    root = root_for(config)
    resolved = root / "resolved_config.yaml"
    record = root / "parallel_run.json"
    if args.distributed_worker:
        if args.stage != "train" or int(os.environ.get("WORLD_SIZE", "0")) != len(gpus):
            parser.error("Workers must be started by this launcher's torchrun command")
        rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["RANK"]) != rank or not 0 <= rank < len(gpus):
            parser.error("Only single-node distributed training is supported")
        if not resolved.exists() or yaml.safe_load(resolved.read_text()) != config:
            parser.error("Worker configuration differs from the launcher's resolved configuration")
        verify(read_json(record), config)
        mapping = isolate_gpu(gpus[rank])
        # Import torch only after each process has exactly one permitted GPU.
        from .trainer import train
        train(config, mapping)
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Run this launcher directly; it starts torchrun itself")
    root.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != config:
        parser.error("Configuration changed: choose a fresh output_dir")
    if record.exists():
        verify(read_json(record), config)
    else:
        write_json(record, {**identity(config), "training_gpus": gpus,
            "strategy": "synchronous data parallel; visit-weighted all-reduce; no episode padding"})
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.stage == "all":
        for stage in ("prepare", "before", "train", "after", "compare"):
            command = [sys.executable, str(ROOT / "run_training_parallel.py"), "--config", str(resolved),
                       "--gpus", ",".join(map(str, gpus)), "--stage", stage]
            if args.resume:
                command.append("--resume")
            print(f"\n=== parallel experiment: {stage} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        return
    if args.stage != "train":
        command = [sys.executable, str(ROOT / "run_training_revision.py"), "--config", str(resolved),
                   "--gpu", str(gpus[0]), "--stage", args.stage]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if completed(root / "training/done.json", config, args.resume):
        return
    # Validate prerequisites before paying for N model replicas.
    from routed_lm.data import verified
    from training_revision.common import verify as verify_baseline
    verified(config)
    verify_baseline(read_json(root / "before_train/summary.json"), config)
    if len(gpus) == 1:
        mapping = isolate_gpu(gpus[0])
        from .trainer import train
        train(config, mapping)
        return
    inventory = gpu_inventory()
    if any(index not in inventory for index in gpus):
        parser.error("A requested GPU is unavailable")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(inventory[index] for index in gpus)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("OMP_NUM_THREADS", "1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={len(gpus)}", str(ROOT / "run_training_parallel.py"),
               "--config", str(resolved), "--gpus", ",".join(map(str, gpus)),
               "--stage", "train", "--distributed-worker"]
    print(f"Train one model across physical GPUs {gpus}; one replica per GPU", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
