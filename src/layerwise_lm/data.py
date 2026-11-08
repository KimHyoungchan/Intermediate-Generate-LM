from __future__ import annotations

import json
from pathlib import Path
import random
import re

from .common import digest, file_hash, identity, read_json, run_root, verify_identity, write_json


def articles(rows, split):
    """Restore top-level WikiText articles, preserving their internal newlines."""
    pending = []
    number = 0
    for row in rows:
        text = row["text"]
        if re.fullmatch(r"= [^=]+ =", text.strip()) and pending:
            value = "".join(pending)
            if value.strip():
                yield {"id": f"{split}:{number}:{digest(value)[:12]}", "text": value}
                number += 1
            pending = []
        pending.append(text if text.endswith("\n") else text + "\n")
    if "".join(pending).strip():
        value = "".join(pending)
        yield {"id": f"{split}:{number}:{digest(value)[:12]}", "text": value}


def windows(tokens, seq_len, overlap):
    for start in range(0, len(tokens), seq_len - overlap):
        chunk = tokens[start:start + seq_len]
        if len(chunk) > overlap:
            yield start, chunk, list(range(overlap, len(chunk)))


def read_windows(path):
    with open(path) as stream:
        for line in stream:
            yield json.loads(line)


def verified_data(config):
    root = run_root(config) / "data"
    manifest = read_json(root / "manifest.json")
    verify_identity(manifest, config)
    for name, info in manifest["splits"].items():
        if file_hash(root / f"{name}.jsonl") != info["sha256"]:
            raise ValueError(f"Prepared data changed: {name}")
    return root, manifest


def prepare(config):
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from huggingface_hub import HfApi

    root = run_root(config) / "data"
    if (root / "manifest.json").exists():
        verified_data(config)
        print(f"Prepared data verified: {root}", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"],
                                             local_files_only=config.get("local_model_only", True))
    local = config.get("data_jsonl")
    if local:
        path = Path(local)
        content = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        documents = {s: [dict(id=str(r["id"]), text=r["text"]) for r in content if r["split"] == s]
                     for s in ("train", "validation", "test")}
        dataset_revision = file_hash(path)
        fingerprint = {"local_file_sha256": dataset_revision}
    else:
        dataset_id = config.get("dataset_id", "Salesforce/wikitext")
        dataset_revision = config.get("dataset_revision") or HfApi().dataset_info(dataset_id).sha
        dataset = load_dataset(dataset_id, config.get("dataset_config", "wikitext-2-raw-v1"),
                               revision=dataset_revision)
        documents = {s: list(articles(dataset[s], s)) for s in ("train", "validation", "test")}
        fingerprint = {s: dataset[s]._fingerprint for s in documents}
    validation = [d for d in documents["validation"] if int(digest(d["id"])[:8], 16) % 2 == 0]
    evaluation_split = config.get("eval_split", "test")
    if evaluation_split not in ("test", "validation"):
        raise ValueError("eval_split must be test or validation")
    # In pilots, use the reserved half of validation for evaluation.
    evaluation = documents["test"] if evaluation_split == "test" else [
        d for d in documents["validation"] if int(digest(d["id"])[:8], 16) % 2 == 1]
    selections = {"train": documents["train"], "validation": validation, "evaluation": evaluation}
    caps = {"train": config["train_tokens"], "validation": config["validation_tokens"],
            "evaluation": config["eval_tokens"]}
    information = {}
    special = set(tokenizer.all_special_ids)
    for split, docs in selections.items():
        docs = list(docs)
        random.Random(config["seed"]).shuffle(docs)
        total, count = 0, 0
        selected_docs = set()
        temporary = root / f"{split}.jsonl.tmp"
        with temporary.open("w") as stream:
            for doc in docs:
                tokens = tokenizer(doc["text"], add_special_tokens=False, truncation=False,
                                   verbose=False)["input_ids"]
                for start, chunk, positions in windows(tokens, config["seq_len"], config["context_overlap"]):
                    positions = [j for j in positions if chunk[j] not in special][:caps[split] - total]
                    if not positions:
                        continue
                    record = {"document_id": doc["id"], "start": start,
                              "input_ids": chunk[:positions[-1] + 1], "target_positions": positions}
                    stream.write(json.dumps(record) + "\n")
                    total += len(positions)
                    count += 1
                    selected_docs.add(doc["id"])
                    if total >= caps[split]:
                        break
                if total >= caps[split]:
                    break
        if total == 0:
            raise ValueError(f"No valid target tokens in {split}")
        destination = root / f"{split}.jsonl"
        temporary.replace(destination)
        information[split] = {"tokens": total, "windows": count, "documents": len(selected_docs),
                              "sha256": file_hash(destination)}
        print(f"Prepared {split}: {information[split]}", flush=True)
    write_json(root / "manifest.json", {**identity(config), "dataset_revision": dataset_revision,
               "dataset_fingerprint": fingerprint, "eval_source_split": evaluation_split,
               "splits": information, "tokenizer_revision": config["model_revision"],
               "special_token_policy": "no added special tokens; exclude special targets"})
