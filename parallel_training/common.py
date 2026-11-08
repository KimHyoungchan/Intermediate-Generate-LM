from training_revision.common import (
    ROOT, digest, file_hash, read_json, write_json, root_for,
    identity as training_identity, verify as verify_training,
)


def source_hash():
    paths = [ROOT / "run_training_parallel.py", *sorted((ROOT / "parallel_training").glob("*.py"))]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in paths})


def identity(config):
    return {**training_identity(config), "parallel_training_source_hash": source_hash()}


def verify(record, config):
    verify_training(record, config)
    if record.get("parallel_training_source_hash") != source_hash():
        raise ValueError("Parallel training source changed; use a fresh output_dir")


def completed(path, config, resume):
    if not path.exists():
        return False
    verify(read_json(path), config)
    if not resume:
        raise FileExistsError(f"Completed stage exists: {path}; use --resume or a fresh output_dir")
    print(f"Reuse parallel training: {path}", flush=True)
    return True
