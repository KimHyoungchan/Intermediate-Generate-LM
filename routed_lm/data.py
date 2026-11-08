from __future__ import annotations

import json
import random
from .common import root_for, identity, verify, file_hash, write_json, read_json, digest


def read_episodes(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(config):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from layerwise_lm.data import articles
    root = root_for(config) / "data"
    if (root / "manifest.json").exists():
        verified(config)
        return
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], local_files_only=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=config["dataset_revision"])
    specials = set(tokenizer.all_special_ids)
    manifest = {**identity(config), "dataset_id": "Salesforce/wikitext", "dataset_revision": config["dataset_revision"],
                "sampling": "document-balanced cyclic selection; random offsets; no duplicate target positions within split", "splits": {}}
    length = config["context_tokens"] + config["continuation_tokens"]
    for split, cap in (("train", config["train_episodes"]), ("validation", config["validation_episodes"]), ("test", config["eval_episodes"])):
        docs = []
        for doc in articles(ds[split], split):
            tokens = tokenizer(doc["text"], add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            if len(tokens) >= length:
                docs.append((doc["id"], tokens))
        rng = random.Random(config["seed"])
        rng.shuffle(docs)
        episodes, used = [], set()
        attempts = 0
        while len(episodes) < cap and attempts < max(cap * 1000, 1000):
            doc_id, tokens = docs[attempts % len(docs)]
            attempts += 1
            start = rng.randrange(len(tokens) - length + 1)
            sequence = tokens[start:start + length]
            target_positions = range(start + config["context_tokens"], start + length)
            keys = {(doc_id, position) for position in target_positions}
            if keys & used or specials & set(sequence):
                continue
            used.update(keys)
            episodes.append({"episode_id": digest([doc_id, start, length])[:24], "document_id": doc_id,
                "start": start, "context": sequence[:config["context_tokens"]],
                "targets": sequence[config["context_tokens"]:]})
        if len(episodes) != cap:
            raise ValueError(f"Insufficient non-overlapping {split} episodes: {len(episodes)}/{cap}")
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(e) + "\n" for e in episodes))
        manifest["splits"][split] = {"episodes": len(episodes), "requested_target_tokens": len(used),
            "documents": len({e["document_id"] for e in episodes}), "sha256": file_hash(path),
            "fingerprint": ds[split]._fingerprint}
        print(f"Prepared routed {split}: {manifest['splits'][split]}", flush=True)
    write_json(root / "manifest.json", manifest)


def verified(config):
    root = root_for(config) / "data"
    manifest = read_json(root / "manifest.json")
    verify(manifest, config)
    for split, info in manifest["splits"].items():
        if file_hash(root / f"{split}.jsonl") != info["sha256"]:
            raise ValueError(f"Changed routed dataset: {split}")
    return root, manifest
