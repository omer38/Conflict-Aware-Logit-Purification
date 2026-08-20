#!/usr/bin/env python
"""Re-run harm analysis against the prior-and-scale-aligned mean.

The historical filename is retained for compatibility. The aligned mean performs
class-prior matching to the known uniform pool prior plus confidence-scale
alignment; it is not described as generally label-free probability calibration.

Original motivation (router_improvement_plan.md
§4b, labelfree_calibration_results.md §4b): "recompute the harm analysis against
the calibrated prob-mean baseline — the expected drop in harm rate quantifies how
much 'conflict' was really miscalibration, a headline table for the dissertation."

Harm is redefined against class-prior matching + confidence-scale alignment,
then arithmetic probability mean, instead of the naive
logit mean, holding phi(x) (raw, tau=1 teacher disagreement features) fixed.
Harm, rescue and newly-harmed churn are constructed from the gold class; the
logistic harm predictor and its AUC are supervised retrospective diagnostics.
AUC is stratified 5-fold out-of-fold (mean ± fold SD), not an in-sample fit.
Cached logits make this analysis cheap, not label-free or deployable.
Compares against the existing logit-mean harm analysis
(``outputs/reports/disagreement_{dataset}_summary.json``, from
``scripts/analyse_disagreement.py``).

Cached logits only, no GPU.

Usage:
    uv run python scripts/harm_vs_calibrated_mean.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.disagreement.features import FEATURE_GROUPS, compute_disagreement_features
from src.disagreement.harm_auc import stratified_oof_logreg_auc
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    _logits_to_probs,
    average_teacher_soft_labels,
    load_multi_teacher_logits,
    prior_scale_aligned_teacher_soft_labels,
    uniform_class_prior,
)

DATASETS = ["ag_news", "yahoo", "amazon", "amazon3"]
REPORT_DIR = Path("outputs/reports")
OLD_SUMMARY = REPORT_DIR / "disagreement_{ds}_summary.json"


def _oof_auc(x_agg: np.ndarray, harm: np.ndarray) -> dict:
    return stratified_oof_logreg_auc(x_agg, harm)


def analyse_dataset(dataset_name: str) -> dict:
    dataset = load_dataset_by_name(dataset_name)
    num_classes = dataset.num_classes
    gold_labels = np.asarray(dataset.train[dataset.label_column])

    stacked_logits = load_multi_teacher_logits(GUIDED_TEACHERS, dataset_name, num_classes)
    probs = _logits_to_probs(stacked_logits, temperature=1.0)
    phi = compute_disagreement_features(probs)
    x_agg = phi[:, FEATURE_GROUPS["aggregate"]]

    per_teacher_pred = probs.argmax(axis=-1)
    any_teacher_correct = (per_teacher_pred == gold_labels[None, :]).any(axis=0)

    logit_mean_probs = average_teacher_soft_labels(
        GUIDED_TEACHERS, dataset_name, num_classes, temperature=1.0
    )
    aligned_mean_probs = prior_scale_aligned_teacher_soft_labels(
        GUIDED_TEACHERS,
        dataset_name,
        num_classes,
        temperature=1.0,
        target_prior=uniform_class_prior(num_classes),
    )

    logit_mean_correct = logit_mean_probs.argmax(axis=-1) == gold_labels
    aligned_mean_correct = aligned_mean_probs.argmax(axis=-1) == gold_labels

    harm_logit_mean = any_teacher_correct & ~logit_mean_correct
    harm_aligned_mean = any_teacher_correct & ~aligned_mean_correct

    # Samples harmed by the logit mean but rescued by the aligned mean.
    rescued = harm_logit_mean & ~harm_aligned_mean
    # Samples newly harmed by the aligned mean that the logit mean got right.
    newly_harmed = harm_aligned_mean & ~harm_logit_mean

    oof_logit = _oof_auc(x_agg, harm_logit_mean)
    oof_aligned = _oof_auc(x_agg, harm_aligned_mean)

    return {
        "dataset": dataset_name,
        "n_samples": int(len(gold_labels)),
        "harm_rate_logit_mean": float(harm_logit_mean.mean()),
        "harm_rate_calibrated_mean": float(harm_aligned_mean.mean()),
        "harm_rate_drop_pp": float((harm_logit_mean.mean() - harm_aligned_mean.mean()) * 100),
        "rescued_rate": float(rescued.mean()),
        "newly_harmed_rate": float(newly_harmed.mean()),
        "auc_logit_mean": oof_logit["auc_mean"],
        "auc_logit_mean_std": oof_logit["auc_std"],
        "auc_calibrated_mean": oof_aligned["auc_mean"],
        "auc_calibrated_mean_std": oof_aligned["auc_std"],
        "auc_oof_n_splits": oof_logit["n_splits"],
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for dataset_name in DATASETS:
        print(f"\n=== {dataset_name} ===")
        r = analyse_dataset(dataset_name)
        results.append(r)
        print(f"  harm rate (logit mean):       {r['harm_rate_logit_mean']:.2%}")
        print(f"  harm rate (aligned mean):     {r['harm_rate_calibrated_mean']:.2%}")
        print(f"  drop:                         {r['harm_rate_drop_pp']:+.2f} pp")
        print(f"  rescued / newly-harmed:       {r['rescued_rate']:.2%} / {r['newly_harmed_rate']:.2%}")
        print(
            f"  OOF AUC (logit mean):         {r['auc_logit_mean']:.4f} "
            f"± {r['auc_logit_mean_std']:.4f}"
        )
        print(
            f"  OOF AUC (aligned mean):       {r['auc_calibrated_mean']:.4f} "
            f"± {r['auc_calibrated_mean_std']:.4f}"
        )

    out_path = REPORT_DIR / "harm_vs_calibrated_mean.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")

    lines = [
        "> **Diagnostic only.** Harm, rescue, newly-harmed churn and AUC use gold",
        "> training-pool labels. They are retrospective supervised diagnostics,",
        "> not label-free or deployable. AUC is stratified 5-fold out-of-fold",
        "> mean ± fold SD.",
        "",
        "| Dataset | Harm rate (logit mean) | Harm rate (prior-and-scale-aligned mean) | Drop | OOF AUC (logit mean) | OOF AUC (aligned mean) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['harm_rate_logit_mean']:.2%} | "
            f"{r['harm_rate_calibrated_mean']:.2%} | {r['harm_rate_drop_pp']:+.2f} pp | "
            f"{r['auc_logit_mean']:.3f}±{r['auc_logit_mean_std']:.3f} | "
            f"{r['auc_calibrated_mean']:.3f}±{r['auc_calibrated_mean_std']:.3f} |"
        )
    table_path = REPORT_DIR / "harm_vs_calibrated_mean.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
