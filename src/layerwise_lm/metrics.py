from __future__ import annotations

import csv
import math
import numpy as np


def depth_and_stability(correct):
    hit = correct.any(axis=1)
    first = np.where(hit, correct.argmax(axis=1) + 1, 0).astype(np.int32)
    stable_mask = ~np.maximum.accumulate(~correct[:, ::-1], axis=1)[:, ::-1]
    stable = np.where(stable_mask.any(axis=1), stable_mask.argmax(axis=1) + 1, 0).astype(np.int32)
    flip = (np.maximum.accumulate(correct, axis=1) & ~correct).any(axis=1)
    return first, stable, flip


def eligible_mean(values, mask=None):
    selected = values if mask is None else values[mask]
    return float(selected.mean()) if len(selected) else None


def summarize(topk, reference, logprob):
    top1 = topk[:, :, 0]
    teacher = top1[:, -1]
    correct = top1 == reference[:, None]
    agreement = top1 == teacher[:, None]
    first, stable, flip = depth_and_stability(correct)
    teacher_first, teacher_stable, _ = depth_and_stability(agreement)
    layers = top1.shape[1]
    cost = np.where(first > 0, first, layers)
    final_correct = correct[:, -1]
    hit = first > 0
    summary = {
        "tokens": len(reference), "layers": layers,
        "reference_final_accuracy": float(final_correct.mean()),
        "reference_hit_coverage": float(hit.mean()), "reference_no_hit_rate": float((~hit).mean()),
        "reference_first_hit_mean": eligible_mean(first, hit),
        "reference_first_hit_median": float(np.median(first[hit])) if hit.any() else None,
        "reference_first_hit_p90": float(np.quantile(first[hit], 0.9)) if hit.any() else None,
        "reference_stable_mean": eligible_mean(stable, stable > 0),
        "reference_stable_coverage": float((stable > 0).mean()),
        "reference_first_hit_mean_final_correct": eligible_mean(first, final_correct),
        "reference_flip_rate_given_hit": eligible_mean(flip.astype(float), hit),
        "reference_early_hit_final_wrong_rate": float((hit & ~final_correct).mean()),
        "teacher_first_hit_mean": float(teacher_first.mean()),
        "teacher_stable_mean": float(teacher_stable.mean()),
        "oracle_fullscan_mean": float(cost.mean()),
        "oracle_block_saving": float(1 - cost.mean() / layers),
        "teacher_oracle_block_saving": float(1 - teacher_first.mean() / layers),
    }
    per_layer = []
    for i in range(layers):
        nll = float(-logprob[:, i].mean(dtype=np.float64))
        per_layer.append({"layer": i + 1, "reference_accuracy": float(correct[:, i].mean()),
                          "teacher_agreement": float(agreement[:, i].mean()),
                          "reference_topk_accuracy": float((topk[:, i] == reference[:, None]).any(axis=1).mean()),
                          "reference_nll": nll, "reference_perplexity": math.exp(nll) if nll < 700 else None,
                          "reference_first_hit_cdf": float(((first > 0) & (first <= i + 1)).mean()),
                          "teacher_first_hit_cdf": float((teacher_first <= i + 1).mean())})
    token_metrics = {"reference_first_hit": first, "reference_stable": stable,
                     "teacher_first_hit": teacher_first, "teacher_stable": teacher_stable,
                     "reference_flip": flip, "oracle_cost": cost}
    assert teacher_first.min() > 0 and agreement[:, -1].all()
    assert np.isclose(per_layer[-1]["reference_first_hit_cdf"], summary["reference_hit_coverage"])
    assert np.array_equal(stable > 0, final_correct)
    return summary, per_layer, token_metrics


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_delta_ci(before, after, documents, mask_before=None, mask_after=None, draws=1000, seed=42):
    """Cluster bootstrap: resample the same document indices for both conditions."""
    _, indices = np.unique(documents, return_inverse=True)
    n = indices.max() + 1
    def aggregate(values, mask):
        mask = np.ones(len(values), dtype=bool) if mask is None else mask
        return (np.bincount(indices, weights=np.where(mask, values, 0), minlength=n),
                np.bincount(indices, weights=mask.astype(float), minlength=n))
    a_sum, a_count = aggregate(before, mask_before)
    b_sum, b_count = aggregate(after, mask_after)
    if not a_count.sum() or not b_count.sum():
        return None, None
    samples = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    da, db = a_count[samples].sum(axis=1), b_count[samples].sum(axis=1)
    keep = (da > 0) & (db > 0)
    delta = b_sum[samples].sum(axis=1)[keep] / db[keep] - a_sum[samples].sum(axis=1)[keep] / da[keep]
    return tuple(float(x) for x in np.quantile(delta, [0.025, 0.975]))
