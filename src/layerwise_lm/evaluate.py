from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import torch

from .common import check_complete, file_hash, identity, read_json, run_root, versions, verify_identity, write_json
from .data import read_windows, verified_data
from .metrics import summarize, write_csv
from .model import load_model, logits_for, make_readouts, raw_hidden, verify_model, weight_hash


def load_trained(readouts, config, manifest, base_hash):
    path = run_root(config) / "training" / "best.pt"
    done = read_json(path.parent / "done.json")
    verify_identity(done, config)
    if file_hash(path) != done["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["data_manifest_hash"] != file_hash(run_root(config) / "data" / "manifest.json"):
        raise ValueError("Checkpoint was trained against another prepared dataset")
    if payload["backbone_sha256"] != base_hash:
        raise ValueError("Checkpoint backbone mismatch")
    readouts.load_state_dict(payload["readouts"])
    return done["checkpoint_sha256"], payload["optimizer_step"]


def evaluate(config, trained, mapping, resume=False):
    root = run_root(config)
    output = root / ("after_train" if trained else "before_train")
    data_root, data_manifest = verified_data(config)
    if check_complete(output / "summary.json", config, resume):
        summary = read_json(output / "summary.json")
        if file_hash(output / "predictions.npz") != summary["predictions_sha256"]:
            raise ValueError("Saved predictions changed")
        if trained and file_hash(root / "training" / "best.pt") != summary["checkpoint_sha256"]:
            raise ValueError("Trained checkpoint changed")
        return
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(config)
    base_hash = weight_hash(model)
    readouts = make_readouts(model, config).eval()
    checkpoint_hash, step = (None, 0)
    if trained:
        checkpoint_hash, step = load_trained(readouts, config, data_manifest, base_hash)
    count = data_manifest["splits"]["evaluation"]["tokens"]
    layers, k = model.config.num_hidden_layers, config["top_k"]
    topk = np.empty((count, layers, k), dtype=np.int32)
    scores = {key: np.empty((count, layers), dtype=np.float32)
              for key in ("reference_logprob", "max_probability", "entropy", "margin")}
    reference = np.empty(count, dtype=np.int32)
    positions_all = np.empty(count, dtype=np.int64)
    documents = []
    offset, verification = 0, None
    chunk = config["projection_chunk_tokens"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for window_no, record in enumerate(read_windows(data_root / "evaluation.jsonl")):
            ids = torch.tensor([record["input_ids"]], device="cuda:0")
            targets_at = torch.tensor(record["target_positions"], device="cuda:0")
            targets = ids[0, targets_at]
            n = len(targets)
            if verification is None:
                verification = verify_model(model, ids)
            hidden = raw_hidden(model, ids, targets_at - 1)
            for l, h in enumerate(hidden):
                adapter = readouts[l] if l < layers - 1 else None
                for start in range(0, n, chunk):
                    stop = min(n, start + chunk)
                    z = logits_for(model, h[start:stop], adapter).float()
                    logp = z.log_softmax(dim=-1)
                    probability = logp.exp()
                    values, indices = z.topk(k, dim=-1)
                    dest = slice(offset + start, offset + stop)
                    topk[dest, l] = indices.cpu().numpy()
                    scores["reference_logprob"][dest, l] = logp.gather(1, targets[start:stop, None]).squeeze(1).cpu().numpy()
                    scores["max_probability"][dest, l] = logp.gather(1, indices[:, :1]).exp().squeeze(1).cpu().numpy()
                    scores["entropy"][dest, l] = -(probability * logp).sum(dim=-1).cpu().numpy()
                    scores["margin"][dest, l] = (values[:, 0] - values[:, 1]).cpu().numpy()
            reference[offset:offset + n] = targets.cpu().numpy()
            positions_all[offset:offset + n] = np.array(record["target_positions"]) + record["start"]
            documents.extend([record["document_id"]] * n)
            offset += n
            if window_no % 10 == 0:
                print(f"{'after' if trained else 'before'}: {offset}/{count} targets", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert offset == count
    summary, per_layer, token_metrics = summarize(topk, reference, scores["reference_logprob"])
    documents = np.array(documents)
    prediction_path = output / "predictions.npz"
    with prediction_path.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, topk=topk, reference=reference, teacher=topk[:, -1, 0],
                            document_ids=documents, absolute_positions=positions_all, **scores, **token_metrics)
    prediction_path.with_suffix(".npz.tmp").replace(prediction_path)
    write_csv(output / "layer_metrics.csv", per_layer)
    write_csv(output / "token_metrics.csv", [
        {"document_id": documents[i], "absolute_position": int(positions_all[i]),
         **{key: int(value[i]) for key, value in token_metrics.items()}}
        for i in range(count)])
    summary.update({**identity(config), "condition": "B-CE" if trained else "A-no-train",
                    "model_id": config["model_id"], "model_revision": config["model_revision"],
                    "backbone_sha256": base_hash, "device": mapping, "versions": versions(),
                    "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
                    "evaluation_data_sha256": data_manifest["splits"]["evaluation"]["sha256"],
                    "documents": len(np.unique(documents)), "checkpoint_sha256": checkpoint_hash,
                    "optimizer_step": step, "adapter_parameters": sum(p.numel() for p in readouts.parameters()),
                    "evaluation_seconds_with_instrumentation": elapsed,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "verification": verification, "predictions_sha256": file_hash(prediction_path),
                    "depth_null_encoding": "0 in NPZ/CSV denotes no hit; real layers are 1..L",
                    "margin_definition": "top1 logit minus top2 logit"})
    write_json(output / "summary.json", summary)
    print(f"Evaluation complete: {output}", flush=True)
