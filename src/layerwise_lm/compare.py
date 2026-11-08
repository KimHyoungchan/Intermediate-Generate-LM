from __future__ import annotations

import math
import numpy as np

from .common import file_hash, identity, read_json, run_root, verify_identity, write_json
from .data import verified_data
from .metrics import eligible_mean, paired_delta_ci, write_csv


def compare(config):
    root = run_root(config)
    verified_data(config)
    a_summary = read_json(root / "before_train" / "summary.json")
    b_summary = read_json(root / "after_train" / "summary.json")
    training = read_json(root / "training" / "done.json")
    for summary in (a_summary, b_summary, training):
        verify_identity(summary, config)
    for name, summary in (("before_train", a_summary), ("after_train", b_summary)):
        if file_hash(root / name / "predictions.npz") != summary["predictions_sha256"]:
            raise ValueError(f"Predictions changed: {name}")
    if file_hash(root / "training" / "best.pt") != b_summary["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from after-training evaluation")
    for key in ("backbone_sha256", "evaluation_data_sha256"):
        if a_summary[key] != b_summary[key]:
            raise ValueError(f"A/B mismatch: {key}")
    with np.load(root / "before_train" / "predictions.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    with np.load(root / "after_train" / "predictions.npz", allow_pickle=False) as f:
        b = {k: f[k] for k in f.files}
    for key in ("reference", "teacher", "document_ids", "absolute_positions"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"A/B target alignment mismatch: {key}")
    if not np.array_equal(a["topk"][:, -1], b["topk"][:, -1]):
        raise ValueError("Final-layer predictions changed despite frozen backbone/head")
    np.testing.assert_allclose(a["reference_logprob"][:, -1], b["reference_logprob"][:, -1], atol=1e-5, rtol=1e-5)
    da, db = a["reference_first_hit"], b["reference_first_hit"]
    both = (da > 0) & (db > 0)
    transitions = {
        "both_hit": int(both.sum()),
        "no_hit_to_hit": int(((da == 0) & (db > 0)).sum()),
        "hit_to_no_hit": int(((da > 0) & (db == 0)).sum()),
        "neither_hit": int(((da == 0) & (db == 0)).sum()),
        "common_hit_depth_delta": eligible_mean((db - da).astype(float), both),
    }
    docs = a["document_ids"]
    specifications = [
        ("reference_hit_coverage", (da > 0).astype(float), (db > 0).astype(float), None, None),
        ("reference_no_hit_rate", (da == 0).astype(float), (db == 0).astype(float), None, None),
        ("reference_first_hit_mean", da, db, da > 0, db > 0),
        ("reference_stable_mean", a["reference_stable"], b["reference_stable"],
         a["reference_stable"] > 0, b["reference_stable"] > 0),
        ("teacher_first_hit_mean", a["teacher_first_hit"], b["teacher_first_hit"], None, None),
        ("oracle_fullscan_mean", a["oracle_cost"], b["oracle_cost"], None, None),
        ("common_hit_first_depth", da, db, both, both),
    ]
    rows = []
    for name, av, bv, am, bm in specifications:
        before, after = eligible_mean(av, am), eligible_mean(bv, bm)
        ci_low, ci_high = paired_delta_ci(av, bv, docs, am, bm,
                                          draws=config.get("bootstrap_draws", 1000), seed=config["seed"])
        rows.append({"metric": name, "before": before, "after": after,
                     "delta_after_minus_before": after - before if before is not None and after is not None else None,
                     "delta_ci95_low": ci_low, "delta_ci95_high": ci_high})
    write_csv(root / "comparison.csv", rows)
    layer_rows = []
    for l in range(a_summary["layers"]):
        ar = a["topk"][:, l, 0] == a["reference"]
        br = b["topk"][:, l, 0] == b["reference"]
        ci = paired_delta_ci(ar.astype(float), br.astype(float), docs,
                             draws=config.get("bootstrap_draws", 1000), seed=config["seed"])
        layer_rows.append({"layer": l + 1, "accuracy_before": float(ar.mean()), "accuracy_after": float(br.mean()),
                           "accuracy_delta": float(br.mean() - ar.mean()), "delta_ci95_low": ci[0], "delta_ci95_high": ci[1],
                           "nll_before": float(-a["reference_logprob"][:, l].mean()),
                           "nll_after": float(-b["reference_logprob"][:, l].mean())})
    write_csv(root / "layer_comparison.csv", layer_rows)
    write_json(root / "pair_manifest.json", {**identity(config), "model_id": config["model_id"],
               "model_revision": config["model_revision"], "tokens": a_summary["tokens"],
               "documents": a_summary["documents"], "transitions": transitions,
               "backbone_sha256": a_summary["backbone_sha256"],
               "evaluation_data_sha256": a_summary["evaluation_data_sha256"],
               "before_summary_sha256": file_hash(root / "before_train" / "summary.json"),
               "after_summary_sha256": file_hash(root / "after_train" / "summary.json"),
               "checkpoint_sha256": b_summary["checkpoint_sha256"],
               "bootstrap_unit": "document", "delta_direction": "after minus before"})
    make_figures(root, a, b, layer_rows)
    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"
    report = ["# 학습 전후 비교", "", f"Model: `{config['model_id']}`", "",
              f"평가 token: {a_summary['tokens']:,}, 문서: {a_summary['documents']}. 동일 prefix·mask의 paired 비교.", "",
              "| 지표 | 학습 전 | 학습 후 | 변화량 B−A | 변화량 95% CI |",
              "| --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        report.append(f"| {row['metric']} | {fmt(row['before'])} | {fmt(row['after'])} | "
                      f"{fmt(row['delta_after_minus_before'])} | [{fmt(row['delta_ci95_low'])}, {fmt(row['delta_ci95_high'])}] |")
    report += ["", f"실패→성공: {transitions['no_hit_to_hit']:,}; 성공→실패: {transitions['hit_to_no_hit']:,}; "
               f"공통 성공: {transitions['both_hit']:,}; 둘 다 실패: {transitions['neither_hit']:,}.", "",
               f"Adapter optimizer steps: {training['optimizer_steps']}; selected step: {training['best_optimizer_step']}.", "",
               f"Validation NLL: {training['initial_validation_nll']:.6f} → {training['best_validation_nll']:.6f}.", "",
               f"Final-layer accuracy: {a_summary['reference_final_accuracy']:.6f} → {b_summary['reference_final_accuracy']:.6f} (고정 backbone/head).", "",
               "최초·안정 깊이와 NLL은 낮을수록 좋고 coverage는 높을수록 좋다. 성공 subset 평균의 변화는 공통 성공 분석과 함께 해석한다.", "",
               "Oracle block 수는 LM Head·gate·KV 보완 비용을 제외한다. 실제 속도 향상이나 순환 경로의 성공을 나타내지 않는다.", "",
               "문서 단위 bootstrap CI다. 문서 수가 적은 smoke/pilot은 성능 결론에 사용하지 않는다.", "",
               "![Layer accuracy](figures/accuracy.png)", "", "![First-hit CDF](figures/first_hit_cdf.png)", ""]
    (root / "comparison.md").write_text("\n".join(report))
    print(f"Comparison report: {root / 'comparison.md'}", flush=True)


def make_figures(root, a, b, layer_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    layers = np.arange(1, len(layer_rows) + 1)
    for name in ("accuracy", "first_hit_cdf", "first_hit_histogram", "nll"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, data, color in (("Before training", a, "tab:blue"), ("After training", b, "tab:orange")):
            depth = data["reference_first_hit"]
            if name == "accuracy":
                values = (data["topk"][:, :, 0] == data["reference"][:, None]).mean(axis=0)
            elif name == "first_hit_cdf":
                values = np.array([((depth > 0) & (depth <= l)).mean() for l in layers])
            elif name == "first_hit_histogram":
                categories = np.concatenate([layers, [0]])
                values = np.array([(depth == l).mean() for l in categories])
                shift = -0.18 if label.startswith("Before") else 0.18
                ax.bar(np.arange(len(categories)) + shift, values, width=0.36, label=label, color=color)
                ax.set_xticks(np.arange(len(categories)), [str(l) for l in layers] + ["No hit"], rotation=45)
                continue
            else:
                values = -data["reference_logprob"].mean(axis=0)
            ax.plot(layers, values, label=label, color=color)
        ax.set_xlabel("Transformer block (1-based)")
        ax.set_ylabel("Reference NLL" if name == "nll" else "Fraction of evaluation tokens")
        ax.set_title(name.replace("_", " ").title())
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{suffix}", dpi=160)
        plt.close(fig)
