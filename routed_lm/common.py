from pathlib import Path
from layerwise_lm.common import ROOT, digest, file_hash, write_json, read_json, isolate_gpu, versions


def source_hash():
    paths = [ROOT / "run_routed.py", *sorted((ROOT / "routed_lm").glob("*.py"))]
    paths += [ROOT / "src/layerwise_lm" / name for name in ("common.py", "model.py", "data.py")]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in paths})


def identity(config):
    return {"config_hash": digest(config), "routed_source_hash": source_hash()}


def verify(record, config):
    for key, value in identity(config).items():
        if record.get(key) != value:
            raise ValueError(f"Routed {key} mismatch; use a new output_dir")


def root_for(config):
    path = Path(config["output_dir"])
    return path if path.is_absolute() else ROOT / path


def completed(path, config, resume):
    if not path.exists():
        return False
    verify(read_json(path), config)
    if not resume:
        raise FileExistsError(f"Completed result exists: {path}; use --resume or new output_dir")
    print(f"Reuse: {path}", flush=True)
    return True
