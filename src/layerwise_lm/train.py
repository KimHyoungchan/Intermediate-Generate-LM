from __future__ import annotations

import json
import random
import time
import torch
import torch.nn.functional as F

from .common import check_complete, file_hash, identity, run_root, versions, write_json
from .data import read_windows, verified_data
from .model import load_model, logits_for, make_readouts, raw_hidden, weight_hash


@torch.inference_mode()
def validation_nll(model, readouts, records, chunk):
    total, tokens = 0.0, 0
    readouts.eval()
    for record in records:
        ids = torch.tensor([record["input_ids"]], device="cuda:0")
        pos = torch.tensor(record["target_positions"], device="cuda:0")
        targets = ids[0, pos]
        hidden = raw_hidden(model, ids, pos - 1)
        for l, adapter in enumerate(readouts):
            for start in range(0, len(pos), chunk):
                z = logits_for(model, hidden[l][start:start + chunk], adapter).float()
                total += F.cross_entropy(z, targets[start:start + chunk], reduction="sum").item()
        tokens += len(pos)
    readouts.train()
    return total / (tokens * len(readouts))


def train(config, mapping, resume=False):
    root = run_root(config)
    output = root / "training"
    data_root, manifest = verified_data(config)
    if check_complete(output / "done.json", config, resume):
        from .common import read_json
        if file_hash(output / "best.pt") != read_json(output / "done.json")["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        return
    # Require an actual matching baseline before training.
    from .common import read_json, verify_identity
    baseline = read_json(root / "before_train" / "summary.json")
    verify_identity(baseline, config)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(config)
    original_hash = weight_hash(model)
    if original_hash != baseline["backbone_sha256"]:
        raise ValueError("Training backbone differs from baseline")
    readouts = make_readouts(model, config)
    initial_state = {k: v.detach().cpu().clone() for k, v in readouts.state_dict().items()}
    optimizer = torch.optim.AdamW(readouts.parameters(), lr=config["learning_rate"],
                                 weight_decay=config.get("weight_decay", 0.0))
    train_records = list(read_windows(data_root / "train.jsonl"))
    validation = list(read_windows(data_root / "validation.jsonl"))
    chunk = config["projection_chunk_tokens"]
    initial_val = validation_nll(model, readouts, validation, chunk)
    print(f"Initial validation intermediate-layer NLL: {initial_val:.5f}", flush=True)
    step, pending, seen, interval_loss, interval_tokens = 0, 0, 0, 0.0, 0
    best_val, best_step, last_validated_step = float("inf"), 0, 0
    log_path = output / "log.jsonl"
    log_path.write_text("")
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def validate_and_save():
        nonlocal best_val, best_step, last_validated_step
        val = validation_nll(model, readouts, validation, chunk)
        last_validated_step = step
        event = {"step": step, "train_tokens_seen": seen, "validation_nll": val,
                 "train_nll_since_validation": interval_loss / max(interval_tokens, 1),
                 "elapsed_seconds": time.monotonic() - started}
        with log_path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(f"step={step} tokens={seen} val_nll={val:.5f}", flush=True)
        if val < best_val:
            best_val, best_step = val, step
            payload = {"readouts": {k: v.detach().cpu() for k, v in readouts.state_dict().items()},
                       "optimizer_step": step, "validation_nll": val,
                       "backbone_sha256": original_hash,
                       "data_manifest_hash": file_hash(data_root / "manifest.json"),
                       **identity(config)}
            temporary = output / "best.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(output / "best.pt")

    def optimizer_step():
        nonlocal step, pending, interval_loss, interval_tokens
        for parameter in readouts.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        norm = torch.nn.utils.clip_grad_norm_(readouts.parameters(), config.get("max_grad_norm", 1.0))
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite adapter gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        step += 1
        if step % config["eval_every_steps"] == 0:
            validate_and_save()
            interval_loss, interval_tokens = 0.0, 0

    readouts.train()
    for epoch in range(config.get("epochs", 1)):
        random.Random(config["seed"] + epoch).shuffle(train_records)
        for record in train_records:
            ids = torch.tensor([record["input_ids"]], device="cuda:0")
            pos = torch.tensor(record["target_positions"], device="cuda:0")
            targets = ids[0, pos]
            hidden = raw_hidden(model, ids, pos - 1)
            for l, adapter in enumerate(readouts):
                for start in range(0, len(pos), chunk):
                    z = logits_for(model, hidden[l][start:start + chunk], adapter).float()
                    loss_sum = F.cross_entropy(z, targets[start:start + chunk], reduction="sum") / len(readouts)
                    if not torch.isfinite(loss_sum):
                        raise FloatingPointError("Non-finite adapter loss")
                    loss_sum.backward()
                    interval_loss += loss_sum.item()
            pending += len(pos)
            seen += len(pos)
            interval_tokens += len(pos)
            if pending >= config["gradient_accumulation_tokens"]:
                optimizer_step()
    if pending:
        optimizer_step()
    if last_validated_step != step:
        validate_and_save()
    if step == 0:
        raise ValueError("No optimizer update was performed")
    unchanged = weight_hash(model) == original_hash
    if not unchanged:
        raise AssertionError("Frozen backbone/norm/LM head changed")
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    changed = any(not torch.equal(initial_state[k], v) for k, v in checkpoint["readouts"].items())
    if not changed:
        raise AssertionError("Selected trained checkpoint is still identity")
    write_json(output / "done.json", {**identity(config), "device": mapping, "versions": versions(),
               "loss": "mean reference next-token CE across all intermediate layers",
               "initial_validation_nll": initial_val, "best_validation_nll": best_val,
               "best_optimizer_step": best_step, "optimizer_steps": step, "train_tokens_seen": seen,
               "epochs": config.get("epochs", 1), "backbone_unchanged": unchanged,
               "trained_checkpoint_changed": changed, "backbone_sha256": original_hash,
               "adapter_parameters": sum(p.numel() for p in readouts.parameters()),
               "checkpoint_sha256": file_hash(output / "best.pt"),
               "elapsed_seconds": time.monotonic() - started,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "resume_policy": "skip completed stages; incomplete training restarts from identity"})
    print(f"Training complete: {output / 'best.pt'}", flush=True)
