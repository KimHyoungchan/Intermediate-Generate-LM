from layerwise_lm.common import ROOT, digest, file_hash, write_json, read_json, versions, isolate_gpu
from routed_lm.common import identity as architecture_identity, verify as verify_architecture, root_for


def source_hash():
    paths = [ROOT / "run_training_revision.py", *sorted((ROOT / "training_revision").glob("*.py"))]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in paths})


def identity(config):
    return {**architecture_identity(config), "training_revision_source_hash": source_hash()}


def verify(record, config):
    verify_architecture(record, config)
    if record.get("training_revision_source_hash") != source_hash():
        raise ValueError("Changed revision training source; use a fresh output directory")


def completed(path, config, resume):
    if not path.exists():
        return False
    verify(read_json(path), config)
    if not resume:
        raise FileExistsError(f"Completed stage exists: {path}; use --resume or fresh output_dir")
    print(f"Reuse revised training stage: {path}", flush=True)
    return True
