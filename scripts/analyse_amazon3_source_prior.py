#!/usr/bin/env python
"""Analyse the Amazon-3 source-prior transfer experiment.

Student-level replication under a source-induced non-uniform prior: the
40/20/40 prior comes from grouping the balanced, gold-constructed Amazon-5
benchmark, not from natural deployment prevalence. Class-prior matching and
confidence-scale alignment are fit on a separate unlabelled reference batch
and transferred (out of sample) to the student working pool; the estimated
prior is a constrained soft-BBSE estimate from a fixed committee predictor,
using no additional teacher queries and no labels beyond the declared
per-class budget.

Inputs (read-only):
  - outputs/student/amazon3_source_prior_transfer_v1/  (24 result cells)

Outputs:
  - outputs/reports/amazon3_source_prior_transfer_v1.json
  - outputs/reports/amazon3_source_prior_transfer_v1.md

Also exposes ``--check-cell`` (single-file validation for the GPU runner).

Run:
    uv run python scripts/analyse_amazon3_source_prior.py
    uv run python scripts/analyse_amazon3_source_prior.py \\
        --check-cell PATH --method scale_only --labels-per-class 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train_amazon3_source_prior as experiment  # noqa: E402

NEW_CELLS_DIR = Path("outputs/student/amazon3_source_prior_transfer_v1")
REPORT_DIR = Path("outputs/reports")
JSON_PATH = REPORT_DIR / "amazon3_source_prior_transfer_v1.json"
MARKDOWN_PATH = REPORT_DIR / "amazon3_source_prior_transfer_v1.md"

METHODS = experiment.METHODS  # ("scale_only", "uniform_prior", "estimated_prior_bbse", "oracle_prior")
LABEL_BUDGETS = experiment.LABEL_BUDGETS
SEEDS = experiment.SEEDS
TRUE_PRIOR = experiment.TRUE_PRIOR
EXPECTED_TEST_SIZE = 10000

REQUIRED_STUDENT_HYPERPARAMS = {
    "model_name": "answerdotai/ModernBERT-base",
    "epochs": 3,
    "alpha": 0.5,
    "temperature": 2.0,
    "lr": 2e-5,
    "batch_size": 16,
    "max_length": 256,
}

REQUIRED_TOP_LEVEL_KEYS = (
    "experiment_id", "method", "dataset", "parent_dataset", "group_map",
    "true_prior", "assumed_prior", "prior_estimator", "reference_batch",
    "working_pool", "labelled_split", "prompt_label_accounting",
    "alignment_fit", "alignment_application", "teacher_target_diagnostics",
    "student_config", "metrics", "per_class_metrics", "dataset_identity",
    "paired_test_data",
)

N_CELLS_EXPECTED = len(METHODS) * len(LABEL_BUDGETS) * len(SEEDS)  # 24

FILENAME_RE = re.compile(
    r"^amazon3_source-prior-transfer_(?P<method>scale_only|uniform_prior|"
    r"estimated_prior_bbse|oracle_prior)_L(?P<labels>\d+)_s(?P<seed>\d+)\.json$"
)

PAIRED_BOOTSTRAP_SEED = 20260103
PAIRED_BOOTSTRAP_REPLICATES = 1000

PRIMARY_COMPARISON = ("estimated_prior_bbse", "scale_only")
SECONDARY_COMPARISONS = (
    ("estimated_prior_bbse", "uniform_prior"),
    ("oracle_prior", "estimated_prior_bbse"),
    ("uniform_prior", "scale_only"),
)
ALL_COMPARISONS = (PRIMARY_COMPARISON, *SECONDARY_COMPARISONS)


def result_path(output_dir: Path, method: str, labels_per_class: int, seed: int) -> Path:
    return output_dir / f"amazon3_source-prior-transfer_{method}_L{labels_per_class}_s{seed}.json"


def _finite_unit_interval(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _prior_matches(values: object, expected: np.ndarray | None, atol: float = 1e-6) -> bool:
    if expected is None:
        return values is None
    if not isinstance(values, list) or len(values) != len(expected):
        return False
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(arr)) and np.allclose(arr, expected, atol=atol))


_INDEPENDENT_SPLIT_CACHE: dict[str, object] = {}
_INDEPENDENT_PRIOR_CACHE: dict[tuple[int, int], dict] = {}


def _independent_reference_working_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the reference/working split independently (not JSON-trusting)."""
    if "train_grouped" not in _INDEPENDENT_SPLIT_CACHE:
        amazon = experiment.load_amazon()
        train_grouped, _ = experiment.build_grouped_labels(amazon)
        demo_by_group = experiment.grouped_prompt_demo_indices(amazon)
        reference_indices, working_indices = experiment.build_reference_working_split(
            train_grouped, demo_by_group
        )
        _INDEPENDENT_SPLIT_CACHE["train_grouped"] = train_grouped
        _INDEPENDENT_SPLIT_CACHE["demo_by_group"] = demo_by_group
        _INDEPENDENT_SPLIT_CACHE["reference_indices"] = reference_indices
        _INDEPENDENT_SPLIT_CACHE["working_indices"] = working_indices
    return (
        _INDEPENDENT_SPLIT_CACHE["train_grouped"],
        _INDEPENDENT_SPLIT_CACHE["reference_indices"],
        _INDEPENDENT_SPLIT_CACHE["working_indices"],
    )


def independent_prior_estimate(labels_per_class: int, seed: int) -> dict:
    """Recompute the BBSE prior estimate for (L, seed) from scratch (cached)."""
    key = (labels_per_class, seed)
    if key not in _INDEPENDENT_PRIOR_CACHE:
        train_grouped, reference_indices, working_indices = _independent_reference_working_split()
        demo_by_group = _INDEPENDENT_SPLIT_CACHE["demo_by_group"]
        labelled_indices, unlabelled_indices = experiment.select_labelled_indices(
            working_indices, train_grouped, demo_by_group, labels_per_class, seed
        )
        stacked_raw = experiment.stack_grouped_teacher_logits()
        payload, estimate = experiment.run_prior_estimation(
            stacked_raw, train_grouped, labelled_indices, reference_indices
        )
        _INDEPENDENT_PRIOR_CACHE[key] = {
            "payload": payload,
            "estimate": estimate,
            "labelled_indices": labelled_indices,
            "unlabelled_indices": unlabelled_indices,
        }
    return _INDEPENDENT_PRIOR_CACHE[key]


def validate_cell(data: dict, *, method: str, labels_per_class: int, seed: int) -> list[str]:
    """Return a list of validation error strings (empty iff the cell is valid)."""
    errors: list[str] = []

    missing_keys = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in data]
    if missing_keys:
        errors.append(f"missing top-level keys: {missing_keys}")
        return errors  # cannot check further without the basics

    if data.get("experiment_id") != experiment.EXPERIMENT_ID:
        errors.append(f"experiment_id mismatch: {data.get('experiment_id')}")
    if data.get("method") != method:
        errors.append(f"method mismatch: {data.get('method')}")
    if data.get("dataset") != "amazon3_source_prior_transfer":
        errors.append(f"dataset mismatch: {data.get('dataset')}")
    if data.get("parent_dataset") != "amazon":
        errors.append(f"parent_dataset mismatch: {data.get('parent_dataset')}")
    if not _prior_matches(data.get("true_prior"), TRUE_PRIOR):
        errors.append(f"true_prior mismatch: {data.get('true_prior')}")

    fixed_assumed_priors = {
        "scale_only": None,
        "uniform_prior": experiment.UNIFORM_PRIOR,
        "oracle_prior": TRUE_PRIOR,
    }
    if method in fixed_assumed_priors:
        if not _prior_matches(data.get("assumed_prior"), fixed_assumed_priors[method]):
            errors.append(f"assumed_prior mismatch for {method}: {data.get('assumed_prior')}")

    labelled_split = data.get("labelled_split", {})
    if labelled_split.get("labels_per_class") != labels_per_class:
        errors.append("labelled_split.labels_per_class mismatch")
    if labelled_split.get("seed") != seed:
        errors.append("labelled_split.seed mismatch")
    labelled_indices = labelled_split.get("labelled_indices", [])
    if len(labelled_indices) != 3 * labels_per_class:
        errors.append(f"labelled_indices length {len(labelled_indices)} != {3 * labels_per_class}")

    reference_batch = data.get("reference_batch", {})
    working_pool = data.get("working_pool", {})
    if reference_batch.get("size") != 500:
        errors.append(f"reference_batch.size != 500: {reference_batch.get('size')}")
    if working_pool.get("size") != 2000:
        errors.append(f"working_pool.size != 2000: {working_pool.get('size')}")

    prior_estimator = data.get("prior_estimator", {})
    if prior_estimator.get("matrix_rank") != 3:
        errors.append(f"prior_estimator.matrix_rank != 3: {prior_estimator.get('matrix_rank')}")
    if prior_estimator.get("solver_success") is not True:
        errors.append("prior_estimator.solver_success is not True")
    if method == "estimated_prior_bbse":
        if not _prior_matches(data.get("assumed_prior"), np.asarray(prior_estimator.get("estimated_prior"))):
            errors.append("assumed_prior does not equal prior_estimator.estimated_prior for estimated_prior_bbse")

    alignment_fit = data.get("alignment_fit", {})
    expected_prior_matching = method != "scale_only"
    if alignment_fit.get("prior_matching") is not expected_prior_matching:
        errors.append(f"alignment_fit.prior_matching mismatch for {method}")
    if method == "scale_only":
        bias = np.asarray(alignment_fit.get("bias", []), dtype=np.float64)
        if bias.size and not np.allclose(bias, 0.0):
            errors.append("scale_only alignment_fit.bias is not all-zero")

    metrics = data.get("metrics", {})
    for key in ("accuracy", "macro_f1"):
        if not _finite_unit_interval(metrics.get(key)):
            errors.append(f"invalid metric {key}: {metrics.get(key)}")

    paired = data.get("paired_test_data", {})
    for key in ("sample_indices", "labels", "predictions"):
        values = paired.get(key, [])
        if len(values) != EXPECTED_TEST_SIZE:
            errors.append(f"paired_test_data.{key} length {len(values)} != {EXPECTED_TEST_SIZE}")

    cfg = data.get("student_config", {})
    for key, expected in REQUIRED_STUDENT_HYPERPARAMS.items():
        if cfg.get(key) != expected:
            errors.append(f"student_config.{key}={cfg.get(key)!r} != {expected!r}")

    dataset_identity = data.get("dataset_identity", {})
    if dataset_identity.get("test", {}).get("size") != EXPECTED_TEST_SIZE:
        errors.append("dataset_identity.test.size != 10000")

    return errors


def validate_cell_independent(data: dict, *, method: str, labels_per_class: int, seed: int) -> list[str]:
    """Deeper, independently-recomputed checks (splits/estimate not JSON-trusted)."""
    errors: list[str] = []
    train_grouped, reference_indices, working_indices = _independent_reference_working_split()

    expected_ref_hash = experiment.indices_hash(reference_indices)
    expected_work_hash = experiment.indices_hash(working_indices)
    if data.get("reference_batch", {}).get("identity_hash") != expected_ref_hash:
        errors.append("reference_batch.identity_hash does not match the independently recomputed split")
    if data.get("working_pool", {}).get("identity_hash") != expected_work_hash:
        errors.append("working_pool.identity_hash does not match the independently recomputed split")

    cached = independent_prior_estimate(labels_per_class, seed)
    expected_labelled = cached["labelled_indices"].tolist()
    if data.get("labelled_split", {}).get("labelled_indices") != expected_labelled:
        errors.append("labelled_split.labelled_indices does not match the independently recomputed selection")

    expected_pi_hat = cached["estimate"].estimated_prior
    stored_pi_hat = np.asarray(data.get("prior_estimator", {}).get("estimated_prior"), dtype=np.float64)
    if not np.allclose(stored_pi_hat, expected_pi_hat, atol=1e-6):
        errors.append(
            f"prior_estimator.estimated_prior {stored_pi_hat.tolist()} does not match the "
            f"independently reproduced estimate {expected_pi_hat.tolist()}"
        )

    return errors


def load_cell(directory: Path, method: str, labels_per_class: int, seed: int, *, deep: bool = True) -> dict:
    path = result_path(directory, method, labels_per_class, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing result cell: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_cell(data, method=method, labels_per_class=labels_per_class, seed=seed)
    if not errors and deep:
        errors = validate_cell_independent(data, method=method, labels_per_class=labels_per_class, seed=seed)
    if errors:
        raise ValueError(f"invalid cell {path}: {'; '.join(errors)}")
    return data


def scan_directory(directory: Path) -> dict[tuple, Path]:
    found: dict[tuple, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        key = (match.group("method"), int(match.group("labels")), int(match.group("seed")))
        found[key] = path
    return found


def expected_keys() -> set[tuple]:
    return {(method, lpc, seed) for method in METHODS for lpc in LABEL_BUDGETS for seed in SEEDS}


def scan_grid_completeness() -> dict:
    found = scan_directory(NEW_CELLS_DIR)
    expected = expected_keys()
    missing = sorted(str(k) for k in (expected - set(found)))
    unexpected = sorted(str(k) for k in (set(found) - expected))
    return {"missing_cells": missing, "unexpected_cells": unexpected, "n_found": len(found)}


def assert_pair_compatible(cell_a: dict, cell_b: dict) -> None:
    a_test = cell_a["dataset_identity"]["test"]
    b_test = cell_b["dataset_identity"]["test"]
    if a_test.get("sha256") != b_test.get("sha256"):
        raise ValueError("paired cells have mismatched test SHA-256 identities")
    a_paired = cell_a["paired_test_data"]
    b_paired = cell_b["paired_test_data"]
    if a_paired["labels"] != b_paired["labels"]:
        raise ValueError("paired cells have mismatched test labels")
    if cell_a["labelled_split"]["labels_per_class"] != cell_b["labelled_split"]["labels_per_class"]:
        raise ValueError("paired cells have mismatched label budgets")
    if cell_a["labelled_split"]["seed"] != cell_b["labelled_split"]["seed"]:
        raise ValueError("paired cells have mismatched seeds")
    if cell_a["working_pool"]["identity_hash"] != cell_b["working_pool"]["identity_hash"]:
        raise ValueError("paired cells have mismatched working-pool identities")
    for key, expected in REQUIRED_STUDENT_HYPERPARAMS.items():
        if cell_a["student_config"].get(key) != expected or cell_b["student_config"].get(key) != expected:
            raise ValueError(f"paired cells disagree on hyperparameter {key}")


def mcnemar_exact(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n_discordant = only_a + only_b
    p_value = 1.0 if n_discordant == 0 else float(binomtest(only_a, n_discordant, p=0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "n_discordant": n_discordant, "p_value": p_value}


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        holm = (m - rank) * p_values[idx]
        running = max(running, holm)
        adjusted[idx] = min(1.0, running)
    return adjusted


def stratified_bootstrap_f1_delta(
    labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, num_classes: int
) -> dict:
    """Paired stratified bootstrap CI for macro-F1(A) - macro-F1(B).

    Resamples within each true test class (keeping the 40/20/40 evaluation
    prior fixed) and uses the same resampled indices for both methods.
    """
    rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    deltas = np.empty(PAIRED_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for b in range(PAIRED_BOOTSTRAP_REPLICATES):
        resampled = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in class_indices]
        )
        f1_a = f1_score(labels[resampled], pred_a[resampled], average="macro")
        f1_b = f1_score(labels[resampled], pred_b[resampled], average="macro")
        deltas[b] = f1_a - f1_b
    return {
        "delta_ci_2_5": float(np.percentile(deltas, 2.5)),
        "delta_ci_97_5": float(np.percentile(deltas, 97.5)),
        "n_replicates": PAIRED_BOOTSTRAP_REPLICATES,
        "seed": PAIRED_BOOTSTRAP_SEED,
    }


def mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in values],
    }


def analyse_comparison(
    cells_a: dict[int, dict], cells_b: dict[int, dict], *, label_a: str, label_b: str
) -> dict:
    seed_rows: list[dict] = []
    deltas_acc: list[float] = []
    deltas_f1: list[float] = []
    a_wins = b_wins = ties = 0

    for seed in SEEDS:
        cell_a, cell_b = cells_a[seed], cells_b[seed]
        assert_pair_compatible(cell_a, cell_b)

        a_acc, b_acc = cell_a["metrics"]["accuracy"], cell_b["metrics"]["accuracy"]
        a_f1, b_f1 = cell_a["metrics"]["macro_f1"], cell_b["metrics"]["macro_f1"]
        d_acc, d_f1 = a_acc - b_acc, a_f1 - b_f1
        deltas_acc.append(d_acc)
        deltas_f1.append(d_f1)

        if abs(d_acc) < 1e-12:
            ties += 1
            winner = "tie"
        elif d_acc > 0:
            a_wins += 1
            winner = label_a
        else:
            b_wins += 1
            winner = label_b

        labels = np.asarray(cell_a["paired_test_data"]["labels"], dtype=np.int64)
        pred_a = np.asarray(cell_a["paired_test_data"]["predictions"], dtype=np.int64)
        pred_b = np.asarray(cell_b["paired_test_data"]["predictions"], dtype=np.int64)
        mcnemar = mcnemar_exact(labels, pred_a, pred_b)
        f1_bootstrap = stratified_bootstrap_f1_delta(labels, pred_a, pred_b, num_classes=3)

        seed_rows.append(
            {
                "seed": seed,
                f"{label_a}_accuracy": a_acc,
                f"{label_b}_accuracy": b_acc,
                "delta_accuracy": d_acc,
                f"{label_a}_macro_f1": a_f1,
                f"{label_b}_macro_f1": b_f1,
                "delta_macro_f1": d_f1,
                "accuracy_winner": winner,
                "mcnemar": mcnemar,
                "macro_f1_bootstrap_ci": f1_bootstrap,
            }
        )

    p_values = [row["mcnemar"]["p_value"] for row in seed_rows]
    for row, adj in zip(seed_rows, holm_adjust(p_values)):
        row["mcnemar"]["p_value_holm"] = float(adj)

    return {
        "a": label_a,
        "b": label_b,
        "n_seeds": len(SEEDS),
        "paired_delta": {"accuracy": mean_std(deltas_acc), "macro_f1": mean_std(deltas_f1)},
        "seed_wins": {label_a: a_wins, label_b: b_wins, "tie": ties},
        "seeds": seed_rows,
    }


def analyse_group(labels_per_class: int) -> dict:
    cells_by_method: dict[str, dict[int, dict]] = {}
    method_summaries: dict[str, dict] = {}

    for method in METHODS:
        cells_by_method[method] = {}
        acc_values, f1_values = [], []
        seed_metrics = []
        for seed in SEEDS:
            cell = load_cell(NEW_CELLS_DIR, method, labels_per_class, seed)
            cells_by_method[method][seed] = cell
            acc, f1 = cell["metrics"]["accuracy"], cell["metrics"]["macro_f1"]
            acc_values.append(acc)
            f1_values.append(f1)
            seed_metrics.append(
                {
                    "seed": seed,
                    "accuracy": acc,
                    "macro_f1": f1,
                    "estimated_or_assumed_prior": cell["assumed_prior"],
                    "prior_l1_error": cell["prior_estimator"]["l1_error"],
                    "working_marginal_error_l1": cell["alignment_application"]["working_marginal_error_l1"],
                }
            )
        method_summaries[method] = {
            "accuracy": mean_std(acc_values),
            "macro_f1": mean_std(f1_values),
            "seeds": seed_metrics,
        }

    comparisons = {
        f"{a}_minus_{b}": analyse_comparison(cells_by_method[a], cells_by_method[b], label_a=a, label_b=b)
        for a, b in ALL_COMPARISONS
    }

    return {
        "labels_per_class": labels_per_class,
        "n_seeds": len(SEEDS),
        "methods": method_summaries,
        "comparisons": comparisons,
    }


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(groups: list[dict], completeness: dict) -> str:
    lines = [
        "# Amazon-3 source-prior transfer experiment",
        "",
        "**Student-level replication under a source-induced non-uniform prior.** "
        "The working pool has a 40/20/40 class prior induced by grouping the "
        "balanced, gold-constructed Amazon-5 benchmark (1-2 stars -> negative, "
        "3 stars -> neutral, 4-5 stars -> positive); it is **not** natural "
        "deployment prevalence. The prior is estimated with a constrained "
        "soft-BBSE estimator from a fixed committee predictor on a separate "
        "500-row unlabelled reference batch, then alignment parameters "
        "(class-prior bias, confidence scale) are fit on that reference batch "
        "and transferred out of sample to a disjoint 2,000-row student working "
        "pool. No additional teacher queries and no labels beyond the declared "
        "per-class budget were used.",
        "",
        f"Grid: {len(METHODS)} methods x {len(LABEL_BUDGETS)} label budgets x "
        f"{len(SEEDS)} seeds = {N_CELLS_EXPECTED} cells. Seeds: "
        f"{', '.join(str(s) for s in SEEDS)}.",
        "",
        "Primary comparison: `estimated_prior_bbse - scale_only` (macro-F1 primary, "
        "accuracy secondary). Secondary: estimated vs uniform, oracle vs estimated "
        "(prior-estimation gap), uniform vs scale_only (cost of a wrong prior). "
        "With three seeds, effect direction/size/consistency are emphasised over "
        "seed-level statistical significance.",
        "",
    ]

    for group in groups:
        lpc = group["labels_per_class"]
        lines.append(f"## L={lpc}")
        lines.append("")
        lines.append("### Method summary (mean +/- SD across 3 seeds)")
        lines.append("")
        lines.append("| method | accuracy | macro-F1 |")
        lines.append("|---|---:|---:|")
        for method in METHODS:
            m = group["methods"][method]
            lines.append(
                f"| {method} | {fmt_pct(m['accuracy']['mean'])} +/- {fmt_pct(m['accuracy']['std'])} | "
                f"{fmt_pct(m['macro_f1']['mean'])} +/- {fmt_pct(m['macro_f1']['std'])} |"
            )
        lines.append("")
        lines.append("### Prior estimates by seed (estimated_prior_bbse)")
        lines.append("")
        lines.append("| seed | estimated prior | L1 error vs true | working-marginal L1 error |")
        lines.append("|---:|---|---:|---:|")
        for row in group["methods"]["estimated_prior_bbse"]["seeds"]:
            prior_str = "[" + ", ".join(f"{v:.3f}" for v in row["estimated_or_assumed_prior"]) + "]"
            lines.append(
                f"| {row['seed']} | {prior_str} | {row['prior_l1_error']:.4f} | "
                f"{row['working_marginal_error_l1']:.4f} |"
            )
        lines.append("")

        lines.append("### Paired comparisons (per seed)")
        lines.append("")
        for a, b in ALL_COMPARISONS:
            cmp = group["comparisons"][f"{a}_minus_{b}"]
            d_acc = cmp["paired_delta"]["accuracy"]
            d_f1 = cmp["paired_delta"]["macro_f1"]
            wins = cmp["seed_wins"]
            lines.append(f"#### {a} - {b}")
            lines.append("")
            lines.append(
                f"- Mean paired delta-macroF1: {100 * d_f1['mean']:+.2f} +/- {100 * d_f1['std']:.2f}; "
                f"per-seed: {', '.join(f'{100 * v:+.2f}' for v in d_f1['values'])}"
            )
            lines.append(
                f"- Mean paired delta-acc: {100 * d_acc['mean']:+.2f} +/- {100 * d_acc['std']:.2f}; "
                f"per-seed: {', '.join(f'{100 * v:+.2f}' for v in d_acc['values'])}"
            )
            lines.append(f"- Wins/ties/losses (accuracy, {a} vs {b}): {wins[a]}/{wins['tie']}/{wins[b]}")
            lines.append("")
            lines.append(
                "| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for row in cmp["seeds"]:
                mc = row["mcnemar"]
                ci = row["macro_f1_bootstrap_ci"]
                lines.append(
                    f"| {row['seed']} | {100 * row['delta_macro_f1']:+.2f} | "
                    f"[{100 * ci['delta_ci_2_5']:+.2f}, {100 * ci['delta_ci_97_5']:+.2f}] | "
                    f"{100 * row['delta_accuracy']:+.2f} | {mc['p_value']:.4g} | {mc['p_value_holm']:.4g} |"
                )
            lines.append("")

    lines.append("## Completeness / provenance")
    lines.append("")
    lines.append(f"- Cells found: {completeness['n_found']} / {N_CELLS_EXPECTED}")
    lines.append(f"- Missing cells: {len(completeness['missing_cells'])}")
    lines.append(f"- Unexpected cells: {len(completeness['unexpected_cells'])}")
    lines.append(f"- Source: `{NEW_CELLS_DIR}`")
    lines.append(
        "- This is a controlled label-shift study (source-induced non-uniform prior via "
        "grouping a balanced benchmark), not a test of covariate shift, domain shift, or "
        "natural deployment prevalence."
    )
    lines.append("")
    return "\n".join(lines)


def check_cell_cli(args: argparse.Namespace) -> int:
    path = Path(args.check_cell)
    if not path.is_file() or path.stat().st_size == 0:
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 1
    errors = validate_cell(data, method=args.method, labels_per_class=args.labels_per_class, seed=args.seed)
    if not errors and not args.shallow:
        errors = validate_cell_independent(
            data, method=args.method, labels_per_class=args.labels_per_class, seed=args.seed
        )
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse the Amazon-3 source-prior transfer experiment.")
    parser.add_argument("--check-cell", default=None, help="Validate a single result JSON and exit 0/1.")
    parser.add_argument("--method", default=None, choices=list(METHODS))
    parser.add_argument("--labels-per-class", type=int, default=None, choices=list(LABEL_BUDGETS))
    parser.add_argument("--seed", type=int, default=None, choices=list(SEEDS))
    parser.add_argument(
        "--shallow", action="store_true", help="Skip the independent-recomputation checks (faster)."
    )
    args = parser.parse_args()

    if args.check_cell is not None:
        required = ("method", "labels_per_class", "seed")
        missing = [n for n in required if getattr(args, n) is None]
        if missing:
            print(f"--check-cell requires: {', '.join('--' + m.replace('_', '-') for m in missing)}", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(check_cell_cli(args))

    completeness = scan_grid_completeness()
    groups: list[dict] = []
    failures: list[str] = list(completeness["missing_cells"])
    failures += [f"unexpected cell: {k}" for k in completeness["unexpected_cells"]]

    if not failures:
        for labels_per_class in LABEL_BUDGETS:
            try:
                groups.append(analyse_group(labels_per_class))
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"L={labels_per_class}: {exc}")

    if failures:
        print(f"Amazon-3 source-prior-transfer analysis incomplete (expected {N_CELLS_EXPECTED} cells):", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment.EXPERIMENT_ID,
        "true_prior": TRUE_PRIOR.tolist(),
        "methods": list(METHODS),
        "label_budgets": list(LABEL_BUDGETS),
        "seeds": list(SEEDS),
        "n_cells_expected": N_CELLS_EXPECTED,
        "primary_comparison": list(PRIMARY_COMPARISON),
        "secondary_comparisons": [list(c) for c in SECONDARY_COMPARISONS],
        "source": str(NEW_CELLS_DIR),
        "completeness": completeness,
        "groups": groups,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(render_markdown(groups, completeness), encoding="utf-8")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
