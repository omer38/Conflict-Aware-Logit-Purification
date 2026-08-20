#!/usr/bin/env python
"""Class-prior sensitivity analysis using cached teacher logits only.

This simulates label-prior shift by importance-weighting the balanced cached
working pools. It does not simulate covariate shift within a class and requires
no teacher queries, student training, or GPU.

Run:
    python scripts/prior_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    _align_confidence_scales_logits,
    _logits_to_probs,
    _match_class_prior_logits,
    load_multi_teacher_logits,
    skew_class_prior,
    uniform_class_prior,
)


REPORT_DIR = Path("outputs/reports")
JSON_PATH = REPORT_DIR / "prior_sensitivity.json"
MARKDOWN_PATH = REPORT_DIR / "prior_sensitivity.md"

# Counts from the locally cached, un-rebalanced source train splits. Amazon-3
# groups the balanced Amazon-5 source as (1-2 stars, 3 stars, 4-5 stars).
NATURAL_SOURCE_COUNTS = {
    "ag_news": [30_000, 30_000, 30_000, 30_000],
    "yahoo": [
        132_698,
        132_191,
        131_412,
        125_624,
        125_340,
        126_798,
        127_355,
        124_050,
        128_688,
        131_965,
    ],
    "amazon": [600_000, 600_000, 600_000, 600_000, 600_000],
    "amazon3": [1_200_000, 600_000, 1_200_000],
}


def _normalise(values: list[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / values.sum()


def _importance_weights(labels: np.ndarray, true_prior: np.ndarray) -> np.ndarray:
    empirical = np.bincount(labels, minlength=len(true_prior)).astype(np.float64)
    empirical /= empirical.sum()
    weights = true_prior[labels] / empirical[labels]
    return weights / weights.mean()


def _aggregate(
    stacked: np.ndarray,
    sample_weights: np.ndarray,
    assumed_prior: np.ndarray | None,
    *,
    align_scales: bool,
) -> np.ndarray:
    adjusted = stacked
    if assumed_prior is not None:
        adjusted = _match_class_prior_logits(
            adjusted,
            assumed_prior,
            sample_weights=sample_weights,
        )
    if align_scales:
        adjusted = _align_confidence_scales_logits(
            adjusted,
            sample_weights=sample_weights,
        )
    return _logits_to_probs(adjusted, temperature=1.0).mean(axis=0)


def _metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    true_prior: np.ndarray,
) -> dict[str, float]:
    weights = weights / weights.sum()
    predictions = probabilities.argmax(axis=1)
    accuracy = float(weights[predictions == labels].sum())

    f1_values: list[float] = []
    for class_id in range(probabilities.shape[1]):
        true_positive = weights[(predictions == class_id) & (labels == class_id)].sum()
        false_positive = weights[(predictions == class_id) & (labels != class_id)].sum()
        false_negative = weights[(predictions != class_id) & (labels == class_id)].sum()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(float(2 * true_positive / denominator) if denominator else 0.0)

    true_probabilities = probabilities[np.arange(len(labels)), labels]
    nll = float(-(weights * np.log(np.clip(true_probabilities, 1e-12, None))).sum())
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = float((weights[:, None] * (probabilities - one_hot) ** 2).sum())

    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            confidence <= upper if np.isclose(upper, 1.0) else confidence < upper
        )
        bin_weight = weights[mask].sum()
        if bin_weight:
            bin_accuracy = np.average(correct[mask], weights=weights[mask])
            bin_confidence = np.average(confidence[mask], weights=weights[mask])
            ece += float(bin_weight * abs(bin_accuracy - bin_confidence))

    predicted_marginal = np.average(probabilities, axis=0, weights=weights)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "prior_l1": float(np.abs(predicted_marginal - true_prior).sum()),
    }


def _evaluate(
    stacked: np.ndarray,
    labels: np.ndarray,
    true_prior: np.ndarray,
    assumed_prior: np.ndarray | None,
    *,
    align_scales: bool,
) -> dict[str, float]:
    weights = _importance_weights(labels, true_prior)
    probabilities = _aggregate(
        stacked,
        weights,
        assumed_prior,
        align_scales=align_scales,
    )
    return _metrics(probabilities, labels, weights, true_prior)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _format_prior(prior: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in prior) + "]"


def _format_metric(metrics: dict[str, float]) -> str:
    return f"{metrics['accuracy']:.3f} / {metrics['nll']:.3f} / {metrics['ece']:.3f}"


def run_dataset(dataset_name: str) -> dict:
    dataset = load_dataset_by_name(dataset_name)
    labels = np.asarray(dataset.train[dataset.label_column], dtype=np.int64)
    stacked = load_multi_teacher_logits(
        GUIDED_TEACHERS,
        dataset_name,
        dataset.num_classes,
    )
    uniform = uniform_class_prior(dataset.num_classes)
    natural = _normalise(NATURAL_SOURCE_COUNTS[dataset_name])
    balanced_weights = _importance_weights(labels, uniform)

    balanced = {
        "probability_mean": _evaluate(
            stacked, labels, uniform, None, align_scales=False
        ),
        "scale_alignment_only": _evaluate(
            stacked, labels, uniform, None, align_scales=True
        ),
        "correct_uniform_prior": _evaluate(
            stacked, labels, uniform, uniform, align_scales=True
        ),
        "mildly_misspecified_prior": _mean_metrics(
            [
                _evaluate(
                    stacked,
                    labels,
                    uniform,
                    skew_class_prior(dataset.num_classes, 0.15, dominant),
                    align_scales=True,
                )
                for dominant in range(dataset.num_classes)
            ]
        ),
        "strongly_misspecified_prior": _mean_metrics(
            [
                _evaluate(
                    stacked,
                    labels,
                    uniform,
                    skew_class_prior(dataset.num_classes, 0.50, dominant),
                    align_scales=True,
                )
                for dominant in range(dataset.num_classes)
            ]
        ),
    }

    natural_scenario = {
        "probability_mean": _evaluate(
            stacked, labels, natural, None, align_scales=False
        ),
        "scale_alignment_only": _evaluate(
            stacked, labels, natural, None, align_scales=True
        ),
        "correct_natural_prior": _evaluate(
            stacked, labels, natural, natural, align_scales=True
        ),
        "forced_uniform_prior": _evaluate(
            stacked, labels, natural, uniform, align_scales=True
        ),
    }

    strong_skew_rows: dict[str, list[dict[str, float]]] = {
        "scale_alignment_only": [],
        "correct_skewed_prior": [],
        "forced_uniform_prior": [],
        "wrong_dominant_class": [],
    }
    for dominant in range(dataset.num_classes):
        true_prior = skew_class_prior(dataset.num_classes, 0.50, dominant)
        wrong_prior = skew_class_prior(
            dataset.num_classes,
            0.50,
            (dominant + 1) % dataset.num_classes,
        )
        strong_skew_rows["scale_alignment_only"].append(
            _evaluate(stacked, labels, true_prior, None, align_scales=True)
        )
        strong_skew_rows["correct_skewed_prior"].append(
            _evaluate(stacked, labels, true_prior, true_prior, align_scales=True)
        )
        strong_skew_rows["forced_uniform_prior"].append(
            _evaluate(stacked, labels, true_prior, uniform, align_scales=True)
        )
        strong_skew_rows["wrong_dominant_class"].append(
            _evaluate(stacked, labels, true_prior, wrong_prior, align_scales=True)
        )

    return {
        "num_classes": dataset.num_classes,
        "working_pool_prior": uniform.tolist(),
        "natural_source_prior": natural.tolist(),
        "natural_source_counts": NATURAL_SOURCE_COUNTS[dataset_name],
        "natural_source_is_uniform": bool(np.allclose(natural, uniform, atol=1e-3)),
        "balanced_pool": balanced,
        "natural_source_label_shift": natural_scenario,
        "strong_skew_label_shift": {
            key: _mean_metrics(rows) for key, rows in strong_skew_rows.items()
        },
        "balanced_pool_weight_check": float(balanced_weights.mean()),
    }


def write_markdown(results: dict[str, dict]) -> None:
    lines = [
        "# Class-prior sensitivity of prior-and-scale-aligned teacher aggregation",
        "",
        "This is a CPU-only cached-logit **label-shift simulation**. Gold labels are used",
        "only to construct and evaluate simulated class priors; the correction itself sees",
        "teacher logits, unlabeled pool frequencies, and a supplied target prior. It does",
        "not test covariate shift or a new deployment domain.",
        "Like the deployed pool-level method, matching is transductive: offsets are fitted",
        "and evaluated on the same simulated unlabeled pool.",
        "",
        "Reported cells are accuracy / NLL / 10-bin ECE at teacher-target level (tau=1).",
        "Mild and strong synthetic results average over every possible dominant class.",
        "",
        "## Source priors",
        "",
        "| Dataset | Un-rebalanced source prior | Distinct from uniform? |",
        "|---|---|---|",
    ]
    for dataset_name, result in results.items():
        prior = np.asarray(result["natural_source_prior"])
        lines.append(
            f"| {dataset_name} | `{_format_prior(prior)}` | "
            f"{'yes' if not result['natural_source_is_uniform'] else 'no'} |"
        )

    sections = [
        (
            "Balanced working pool: assumed-prior sensitivity",
            "balanced_pool",
            [
                "probability_mean",
                "scale_alignment_only",
                "correct_uniform_prior",
                "mildly_misspecified_prior",
                "strongly_misspecified_prior",
            ],
        ),
        (
            "Natural source prior under label shift",
            "natural_source_label_shift",
            [
                "probability_mean",
                "scale_alignment_only",
                "correct_natural_prior",
                "forced_uniform_prior",
            ],
        ),
        (
            "Strong synthetic skew under label shift",
            "strong_skew_label_shift",
            [
                "scale_alignment_only",
                "correct_skewed_prior",
                "forced_uniform_prior",
                "wrong_dominant_class",
            ],
        ),
    ]
    for title, result_key, methods in sections:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Method | AG News | Yahoo | Amazon | Amazon-3 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            cells = [
                _format_metric(results[dataset][result_key][method])
                for dataset in ("ag_news", "yahoo", "amazon", "amazon3")
            ]
            lines.append(f"| {method.replace('_', ' ')} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Class-prior matching is unsupervised conditional on a supplied prior; it is",
            "  not label-free in an end-to-end setup where labels established that prior.",
            "- Confidence-scale alignment equalises mean entropy; it does not guarantee",
            "  empirical probability calibration.",
            "- The natural-source row is not an imbalance stress test for datasets whose",
            "  raw benchmark source is already uniform.",
            "- Importance weighting reuses cached class-conditional examples and therefore",
            "  supports claims about prior shift only, not general deployment shift.",
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = {
        dataset_name: run_dataset(dataset_name)
        for dataset_name in ("ag_news", "yahoo", "amazon", "amazon3")
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results)
    print(f"Saved {JSON_PATH}")
    print(f"Saved {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
