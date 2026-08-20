#!/usr/bin/env python
"""Disagreement-feature diagnostics for the guided teacher trio (Phase B / M3).

For each dataset, computes phi(x) (Eq. 6) and the "harm from naive averaging"
oracle diagnostic, then reports correlations between phi(x) and harm. Constructing
harm targets and measuring logistic-regression AUC use the pool's gold labels.
AUC is stratified 5-fold out-of-fold (mean ± fold SD), not an in-sample fit.
This is a retrospective supervised diagnostic, not a label-free or deployable
method; only phi(x) is label-independent. Gold labels are never fed to the student.

Usage:
    uv run python scripts/analyse_disagreement.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.disagreement.features import FEATURE_GROUPS, FEATURE_NAMES, compute_disagreement_features
from src.disagreement.harm_auc import stratified_oof_logreg_auc
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    _logits_to_probs,
    average_teacher_soft_labels,
    load_multi_teacher_logits,
)

DATASETS = ["ag_news", "yahoo", "amazon", "amazon3"]
REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def analyse_dataset(dataset_name: str) -> dict:
    dataset = load_dataset_by_name(dataset_name)
    num_classes = dataset.num_classes
    gold_labels = np.asarray(dataset.train[dataset.label_column])

    stacked_logits = load_multi_teacher_logits(GUIDED_TEACHERS, dataset_name, num_classes)
    probs = _logits_to_probs(stacked_logits, temperature=1.0)
    phi = compute_disagreement_features(probs)

    per_teacher_pred = probs.argmax(axis=-1)  # (K, N)
    any_teacher_correct = (per_teacher_pred == gold_labels[None, :]).any(axis=0)

    mean_probs = average_teacher_soft_labels(GUIDED_TEACHERS, dataset_name, num_classes, temperature=1.0)
    naive_mean_correct = mean_probs.argmax(axis=-1) == gold_labels

    harm = any_teacher_correct & ~naive_mean_correct

    correlations = {}
    for i, name in enumerate(FEATURE_NAMES):
        r, _p = pearsonr(harm.astype(np.float64), phi[:, i].astype(np.float64))
        correlations[name] = float(r) if np.isfinite(r) else 0.0

    agg_idx = FEATURE_GROUPS["aggregate"]
    x_agg = phi[:, agg_idx]
    oof = stratified_oof_logreg_auc(x_agg, harm)

    top1_agreement_rate = float(phi[:, FEATURE_NAMES.index("top1_agreement")].mean())
    disagreement_rate = 1.0 - top1_agreement_rate
    mean_pairwise_kl = float(phi[:, FEATURE_NAMES.index("kl_mean")].mean())

    return {
        "dataset": dataset_name,
        "n_samples": int(len(gold_labels)),
        "phi": phi,
        "harm": harm,
        "harm_rate": float(harm.mean()),
        "correlations": correlations,
        # Mean OOF AUC; key name retained for report compatibility.
        "logreg_auc_aggregate_features": oof["auc_mean"],
        "logreg_auc_oof_std": oof["auc_std"],
        "logreg_auc_oof_folds": oof["auc_folds"],
        "logreg_auc_oof_n_splits": oof["n_splits"],
        "disagreement_rate": disagreement_rate,
        "mean_pairwise_kl": mean_pairwise_kl,
    }


def save_csv(result: dict, path: Path) -> None:
    phi = result["phi"]
    harm = result["harm"]
    fieldnames = ["sample_idx", *FEATURE_NAMES, "harm"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(phi.shape[0]):
            row = {"sample_idx": idx, "harm": int(harm[idx])}
            row.update({name: float(phi[idx, i]) for i, name in enumerate(FEATURE_NAMES)})
            writer.writerow(row)


def save_summary(result: dict, path: Path) -> None:
    summary = {
        "dataset": result["dataset"],
        "n_samples": result["n_samples"],
        "harm_rate": result["harm_rate"],
        "disagreement_rate": result["disagreement_rate"],
        "mean_pairwise_kl": result["mean_pairwise_kl"],
        "logreg_auc_aggregate_features": result["logreg_auc_aggregate_features"],
        "logreg_auc_oof_std": result["logreg_auc_oof_std"],
        "logreg_auc_oof_folds": result["logreg_auc_oof_folds"],
        "logreg_auc_oof_n_splits": result["logreg_auc_oof_n_splits"],
        "correlations_with_harm": result["correlations"],
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plot_correlations(result: dict, path: Path) -> None:
    names = FEATURE_NAMES
    values = [result["correlations"][n] for n in names]
    colors = ["#d7191c" if v >= 0 else "#2c7bb6" for v in values]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Pearson correlation with harm")
    ax.set_title(
        f"{result['dataset']}: phi(x) vs. harm-from-naive-averaging "
        f"(rate={result['harm_rate']:.2%}, "
        f"OOF AUC={result['logreg_auc_aggregate_features']:.3f}"
        f"±{result['logreg_auc_oof_std']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        print(f"\n=== {dataset_name} ===")
        result = analyse_dataset(dataset_name)

        print(f"  n_samples: {result['n_samples']}")
        print(f"  harm rate (any teacher right, naive mean wrong): {result['harm_rate']:.2%}")
        print(f"  top-1 disagreement rate: {result['disagreement_rate']:.2%}")
        print(f"  mean pairwise KL: {result['mean_pairwise_kl']:.4f}")
        print(
            "  logistic-regression stratified 5-fold OOF AUC "
            f"(aggregate features -> harm): "
            f"{result['logreg_auc_aggregate_features']:.4f} "
            f"± {result['logreg_auc_oof_std']:.4f}"
        )
        print("  correlations with harm:")
        for name, r in result["correlations"].items():
            print(f"    {name:>18}: {r:+.4f}")

        save_csv(result, REPORT_DIR / f"disagreement_{dataset_name}.csv")
        save_summary(result, REPORT_DIR / f"disagreement_{dataset_name}_summary.json")
        plot_correlations(result, FIGURE_DIR / f"disagreement_{dataset_name}_correlations.pdf")
        print(f"  Saved reports/figures for {dataset_name}.")


if __name__ == "__main__":
    main()
