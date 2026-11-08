from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import os
import importlib.metadata


ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hash():
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "src").rglob("*.py"))})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def gpu_inventory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    return {int(line.split(",")[0]): line.split(",")[1].strip()
            for line in result.stdout.splitlines() if line.strip()}


def isolate_gpu(index, cpu=False):
    if index not in (1, 2, 3):
        raise ValueError("Only physical GPUs 1, 2, 3 are allowed; physical GPU 0 is forbidden.")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return {"physical_index": None, "uuid": None, "logical_device": "cpu"}
    inventory = gpu_inventory()
    if index not in inventory:
        raise ValueError(f"Physical GPU {index} is unavailable")
    os.environ["CUDA_VISIBLE_DEVICES"] = inventory[index]
    return {"physical_index": index, "uuid": inventory[index], "logical_device": "cuda:0"}


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def versions():
    names = ("torch", "transformers", "datasets", "numpy", "matplotlib", "PyYAML", "huggingface-hub")
    return {n: importlib.metadata.version(n) for n in names}


def run_root(config):
    path = Path(config["output_dir"])
    return path if path.is_absolute() else ROOT / path


def identity(config):
    return {"config_hash": digest(config), "source_hash": source_hash()}


def verify_identity(record, config):
    for key, value in identity(config).items():
        if record.get(key) != value:
            raise ValueError(f"{key} changed. Use a new output_dir to avoid mixing runs.")


def check_complete(path, config, resume):
    if not Path(path).exists():
        return False
    verify_identity(read_json(path), config)
    if not resume:
        raise FileExistsError(f"Completed stage exists: {path}. Use --resume or a new output_dir.")
    print(f"Reusing completed stage: {path}", flush=True)
    return True
