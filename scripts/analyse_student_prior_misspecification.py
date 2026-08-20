#!/usr/bin/env python
"""Analyse the student-level A.12 prior-misspecification experiment.

Balanced-pool, predeclared-prior sensitivity check: only the *assumed* class
prior fed to prior-and-scale-aligned matching changes across conditions
(correct / mild / strong). The actual training pool and the fixed balanced
test set are identical in every condition. Class 0 is the predeclared
misspecification direction (not averaged over dominant classes); this is not
a label-shift, unbalanced-pool, covariate-shift, or domain-shift experiment.

Inputs (read-only):
  - outputs/student/prompt_inclusive_core_3seed_v1/   (18 existing correct-prior cells)
  - outputs/student/prior_misspecification_a12_v1/    (36 new mild/strong cells)

Outputs:
  - outputs/reports/student_prior_misspecification_a12.json
  - outputs/reports/student_prior_misspecification_a12.md

Also exposes ``--check-cell`` so the GPU runner can validate a single result
file before deciding whether to skip or quarantine it.

Run:
    uv run python scripts/analyse_student_prior_misspecification.py
    uv run python scripts/analyse_student_prior_misspecification.py \\
        --check-cell PATH --dataset ag_news --condition mild \\
        --labels-per-class 10 --seed 42
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name  # noqa: E402
from src.student.soft_labels import (  # noqa: E402
    GUIDED_TEACHERS,
    MULTI_PRIOR_SCALE_ALIGNED_TEACHER,
    prior_scale_aligned_teacher_soft_labels,
    skew_class_prior,
    teacher_target_diagnostics,
    uniform_class_prior,
)

BASELINE_DIR = Path("outputs/student/prompt_inclusive_core_3seed_v1")
NEW_CELLS_DIR = Path("outputs/student/prior_misspecification_a12_v1")
REPORT_DIR = Path("outputs/reports")
JSON_PATH = REPORT_DIR / "student_prior_misspecification_a12.json"
MARKDOWN_PATH = REPORT_DIR / "student_prior_misspecification_a12.md"

MODE = "kd"
TEACHER = MULTI_PRIOR_SCALE_ALIGNED_TEACHER
EXPECTED_AGGREGATION = "prior_matched_scale_aligned_prob_mean"
KD_TEMPERATURE = 2.0

DATASETS = ("ag_news", "amazon", "yahoo")
LABEL_BUDGETS = (10, 50)
SEEDS = (42, 123, 456)
CONDITIONS = ("correct", "mild", "strong")
DOMINANT_CLASS = 0
STRENGTH = {"mild": 0.15, "strong": 0.50}

NUM_CLASSES = {"ag_news": 4, "amazon": 5, "yahoo": 10}
EXPECTED_TEST_SIZE = {"ag_news": 7600, "amazon": 10000, "yahoo": 10000}

REQUIRED_KD_HYPERPARAMS = {
    "model_name": "answerdotai/ModernBERT-base",
    "epochs": 3,
    "alpha": 0.5,
    "temperature": 2.0,
    "lr": 2e-5,
    "batch_size": 16,
    "max_length": 256,
}

N_CELLS_EXPECTED = len(DATASETS) * len(LABEL_BUDGETS) * len(SEEDS) * len(CONDITIONS)
N_CORRECT_BASELINES_EXPECTED = len(DATASETS) * len(LABEL_BUDGETS) * len(SEEDS)
N_NEW_CELLS_EXPECTED = len(DATASETS) * len(LABEL_BUDGETS) * len(SEEDS) * 2

FILENAME_RE = re.compile(
    r"^(?P<dataset>[a-z0-9_]+)_kd_"
    + re.escape(TEACHER)
    + r"(?:_prior-(?P<condition>mild|strong)-d(?P<dominant>\d+))?"
    + r"_L(?P<labels>\d+)_s(?P<seed>\d+)_prompt-inclusive-v1\.json$"
)


def compute_assumed_prior(num_classes: int, condition: str, dominant_class: int) -> np.ndarray:
    """Reconstruct the assumed prior for a condition (no stored-value trust)."""
    if condition == "correct":
        return uniform_class_prior(num_classes)
    return skew_class_prior(num_classes, STRENGTH[condition], dominant_class)


def prior_filename_tag(condition: str, dominant_class: int) -> str:
    """The one naming convention shared by train_student.py, the runner, and this script."""
    return "" if condition == "correct" else f"_prior-{condition}-d{dominant_class}"


def result_path(
    output_dir: Path,
    dataset: str,
    condition: str,
    labels_per_class: int,
    seed: int,
    dominant_class: int = DOMINANT_CLASS,
) -> Path:
    tag = prior_filename_tag(condition, dominant_class)
    return output_dir / (
        f"{dataset}_{MODE}_{TEACHER}{tag}_L{labels_per_class}_s{seed}_prompt-inclusive-v1.json"
    )


def input_dir_for_condition(condition: str) -> Path:
    return BASELINE_DIR if condition == "correct" else NEW_CELLS_DIR


def _finite_unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _prior_matches(values: object, expected: np.ndarray, atol: float = 1e-6) -> bool:
    if not isinstance(values, list) or len(values) != len(expected):
        return False
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    return bool(np.allclose(arr, expected, atol=atol))


def validate_cell(
    data: dict,
    *,
    dataset: str,
    condition: str,
    labels_per_class: int,
    seed: int,
    dominant_class: int = DOMINANT_CLASS,
) -> list[str]:
    """Return a list of validation error strings (empty iff the cell is valid)."""
    errors: list[str] = []
    if data.get("dataset") != dataset:
        errors.append(f"dataset mismatch: {data.get('dataset')}")
    if data.get("mode") != MODE:
        errors.append(f"mode mismatch: {data.get('mode')}")
    if data.get("teacher") != TEACHER:
        errors.append(f"teacher mismatch: {data.get('teacher')}")
    if data.get("labels_per_class") != labels_per_class:
        errors.append(f"L mismatch: {data.get('labels_per_class')}")
    if data.get("seed") != seed:
        errors.append(f"seed mismatch: {data.get('seed')}")

    accounting = data.get("label_accounting", {})
    if accounting.get("policy") != "prompt-inclusive-v1":
        errors.append("label_accounting.policy != prompt-inclusive-v1")
    if accounting.get("declared_labels_per_class") != labels_per_class:
        errors.append("declared_labels_per_class mismatch")
    if accounting.get("prompt_demo_labels_per_class") != 2:
        errors.append("prompt_demo_labels_per_class != 2")
    if accounting.get("student_ce_labels_per_class") != labels_per_class:
        errors.append("student_ce_labels_per_class mismatch")
    if accounting.get("router_calibration_labels_per_class") != 0:
        errors.append("router_calibration_labels_per_class != 0")
    if accounting.get("calibration_reserved") is not False:
        errors.append("calibration_reserved must be false")

    if data.get("dataset_identity", {}).get("artifact_version") != "large-test-v1":
        errors.append("dataset_identity.artifact_version != large-test-v1 (legacy/mismatched artifact)")

    metrics = data.get("metrics", {})
    for key in ("accuracy", "macro_f1"):
        if not _finite_unit_interval(metrics.get(key)):
            errors.append(f"invalid metric {key}: {metrics.get(key)}")

    expected_n = EXPECTED_TEST_SIZE.get(dataset)
    paired = data.get("paired_test_data", {})
    for key in ("sample_indices", "labels", "predictions"):
        values = paired.get(key, [])
        if expected_n is not None and len(values) != expected_n:
            errors.append(f"paired_test_data.{key} length {len(values)} != {expected_n}")

    cfg = data.get("config", {})
    for key, expected in REQUIRED_KD_HYPERPARAMS.items():
        if cfg.get(key) != expected:
            errors.append(f"config.{key}={cfg.get(key)!r} != {expected!r}")

    if data.get("aggregation") != EXPECTED_AGGREGATION:
        errors.append(f"aggregation={data.get('aggregation')!r} != {EXPECTED_AGGREGATION!r}")

    num_classes = NUM_CLASSES.get(dataset)
    prior_meta = data.get("class_prior_assumption", {})
    if condition == "correct":
        if prior_meta.get("source") != "known_uniform_prior_from_gold-balanced_working_pool":
            errors.append("class_prior_assumption.source mismatch for correct condition")
        if prior_meta.get("uses_individual_labels_during_matching") is not False:
            errors.append("class_prior_assumption.uses_individual_labels_during_matching must be false")
        if num_classes is not None and not _prior_matches(
            prior_meta.get("values"), uniform_class_prior(num_classes)
        ):
            errors.append("class_prior_assumption.values is not the uniform prior")
        if "condition" in prior_meta:
            errors.append("correct-condition cell must not carry a condition tag")
    else:
        if prior_meta.get("condition") != condition:
            errors.append(f"class_prior_assumption.condition mismatch: {prior_meta.get('condition')}")
        if prior_meta.get("source") != (
            "predeclared_A12_style_misspecification_on_gold_balanced_working_pool"
        ):
            errors.append("class_prior_assumption.source mismatch for misspecified condition")
        if prior_meta.get("uses_individual_labels_during_matching") is not False:
            errors.append("class_prior_assumption.uses_individual_labels_during_matching must be false")
        expected_strength = STRENGTH[condition]
        if prior_meta.get("strength") != expected_strength:
            errors.append(f"class_prior_assumption.strength mismatch: {prior_meta.get('strength')}")
        if prior_meta.get("dominant_class") != dominant_class:
            errors.append(f"class_prior_assumption.dominant_class mismatch: {prior_meta.get('dominant_class')}")
        if num_classes is not None:
            expected_values = skew_class_prior(num_classes, expected_strength, dominant_class)
            if not _prior_matches(prior_meta.get("values"), expected_values):
                errors.append("class_prior_assumption.values does not match the registered condition")
            if not _prior_matches(
                prior_meta.get("true_working_pool_prior"), uniform_class_prior(num_classes)
            ):
                errors.append("class_prior_assumption.true_working_pool_prior is not uniform")

        diagnostics = data.get("teacher_target_diagnostics", {})
        if diagnostics.get("scope") != "offline_full_balanced_working_pool":
            errors.append("teacher_target_diagnostics.scope mismatch")
        if diagnostics.get("uses_gold_labels_for_evaluation_only") is not True:
            errors.append("teacher_target_diagnostics.uses_gold_labels_for_evaluation_only must be true")
        for key in ("accuracy", "macro_f1", "nll"):
            value = diagnostics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"teacher_target_diagnostics.{key} is not finite")
        if not isinstance(diagnostics.get("num_rows"), int) or diagnostics.get("num_rows", 0) <= 0:
            errors.append("teacher_target_diagnostics.num_rows must be a positive int")

    return errors


def assert_pair_compatible(cell_a: dict, cell_b: dict) -> None:
    a_test = cell_a["dataset_identity"]["test"]
    b_test = cell_b["dataset_identity"]["test"]
    if a_test.get("sha256") != b_test.get("sha256"):
        raise ValueError("paired cells have mismatched test SHA-256 identities")
    a_paired = cell_a["paired_test_data"]
    b_paired = cell_b["paired_test_data"]
    if a_paired["sample_indices"] != b_paired["sample_indices"]:
        raise ValueError("paired cells have mismatched sample_indices")
    if a_paired["labels"] != b_paired["labels"]:
        raise ValueError("paired cells have mismatched test labels")
    if cell_a["dataset"] != cell_b["dataset"]:
        raise ValueError("paired cells have mismatched datasets")
    if cell_a["labels_per_class"] != cell_b["labels_per_class"]:
        raise ValueError("paired cells have mismatched label budgets")
    if cell_a["seed"] != cell_b["seed"]:
        raise ValueError("paired cells have mismatched seeds")
    for key in ("model_name", "epochs", "temperature", "lr", "batch_size", "max_length"):
        if cell_a["config"].get(key) != cell_b["config"].get(key):
            raise ValueError(f"paired cells disagree on hyperparameter {key}")


def load_cell(
    dataset: str,
    condition: str,
    labels_per_class: int,
    seed: int,
    dominant_class: int = DOMINANT_CLASS,
) -> dict:
    path = result_path(
        input_dir_for_condition(condition), dataset, condition, labels_per_class, seed, dominant_class
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing result cell: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_cell(
        data,
        dataset=dataset,
        condition=condition,
        labels_per_class=labels_per_class,
        seed=seed,
        dominant_class=dominant_class,
    )
    if errors:
        raise ValueError(f"invalid cell {path}: {'; '.join(errors)}")
    return data


def scan_directory(directory: Path) -> dict[tuple, Path]:
    """Map every filename in *directory* matching our teacher to a grid key.

    Non-matching files (other teachers/methods sharing the directory, e.g. the
    other 191 prompt-inclusive-core cells) are silently ignored.
    """
    found: dict[tuple, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        condition = match.group("condition") or "correct"
        dominant = int(match.group("dominant")) if match.group("dominant") else DOMINANT_CLASS
        key = (
            match.group("dataset"),
            int(match.group("labels")),
            int(match.group("seed")),
            condition,
            dominant,
        )
        found[key] = path
    return found


def expected_keys() -> set[tuple]:
    return {
        (dataset, lpc, seed, condition, DOMINANT_CLASS)
        for dataset in DATASETS
        for lpc in LABEL_BUDGETS
        for seed in SEEDS
        for condition in CONDITIONS
    }


def scan_grid_completeness() -> dict:
    """Detect missing/duplicate/unexpected cells across both source directories."""
    baseline_found = scan_directory(BASELINE_DIR)
    new_found = scan_directory(NEW_CELLS_DIR)

    duplicates = sorted(str(k) for k in (set(baseline_found) & set(new_found)))
    # "correct" belongs in the baseline dir only; mild/strong in the new dir only.
    misplaced = [
        str(k) for k, path in baseline_found.items() if k[3] != "correct"
    ] + [str(k) for k, path in new_found.items() if k[3] == "correct"]

    all_found = dict(baseline_found)
    all_found.update(new_found)
    expected = expected_keys()
    missing = sorted(str(k) for k in (expected - set(all_found)))
    unexpected = sorted(
        str(k) for k in (set(all_found) - expected) if k not in set(baseline_found) & set(new_found)
    )
    return {
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "misplaced_cells": sorted(misplaced),
        "unexpected_cells": unexpected,
        "n_found": len(all_found),
    }


def mcnemar_exact(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n_discordant = only_a + only_b
    p_value = 1.0 if n_discordant == 0 else float(binomtest(only_a, n_discordant, p=0.5).pvalue)
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx]
        holm = (m - rank) * raw
        running = max(running, holm)
        adjusted[idx] = min(1.0, running)
    return adjusted


def mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in values],
    }


def paired_delta_stats(deltas: list[float]) -> dict:
    stats = mean_std(deltas)
    n = len(deltas)
    stats["se"] = float(stats["std"] / math.sqrt(n)) if n else 0.0
    stats["se_note"] = "descriptive SE of seed-level paired deltas; not a standalone significance claim"
    return stats


def analyse_comparison(
    cells_a: dict[int, dict],
    cells_b: dict[int, dict],
    *,
    label_a: str,
    label_b: str,
) -> dict:
    seed_rows: list[dict] = []
    deltas_acc: list[float] = []
    deltas_f1: list[float] = []
    a_wins = b_wins = ties = 0

    for seed in SEEDS:
        cell_a = cells_a[seed]
        cell_b = cells_b[seed]
        assert_pair_compatible(cell_a, cell_b)

        a_acc = float(cell_a["metrics"]["accuracy"])
        b_acc = float(cell_b["metrics"]["accuracy"])
        a_f1 = float(cell_a["metrics"]["macro_f1"])
        b_f1 = float(cell_b["metrics"]["macro_f1"])
        d_acc = a_acc - b_acc
        d_f1 = a_f1 - b_f1
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
            }
        )

    p_values = [row["mcnemar"]["p_value"] for row in seed_rows]
    for row, adj in zip(seed_rows, holm_adjust(p_values)):
        row["mcnemar"]["p_value_holm"] = float(adj)

    return {
        "a": label_a,
        "b": label_b,
        "n_seeds": len(SEEDS),
        "paired_delta": {
            "accuracy": paired_delta_stats(deltas_acc),
            "macro_f1": paired_delta_stats(deltas_f1),
        },
        "seed_wins": {label_a: a_wins, label_b: b_wins, "tie": ties},
        "seeds": seed_rows,
    }


_TEACHER_TARGET_CACHE: dict[tuple, dict] = {}


def offline_teacher_target(dataset_name: str, condition: str, dominant_class: int) -> dict:
    """Independent (not JSON-trusting) offline teacher-target diagnostic.

    Cached teacher logits only, no GPU, no training. Uses gold labels for
    evaluation only; identical to what train_student.py computes and stores
    for mild/strong, and reconstructed here for correct (not stored in the
    pre-existing 18 baseline files).
    """
    key = (dataset_name, condition, dominant_class)
    if key in _TEACHER_TARGET_CACHE:
        return _TEACHER_TARGET_CACHE[key]
    dataset = load_dataset_by_name(dataset_name)
    assumed_prior = compute_assumed_prior(dataset.num_classes, condition, dominant_class)
    soft_probs = prior_scale_aligned_teacher_soft_labels(
        GUIDED_TEACHERS, dataset_name, dataset.num_classes, KD_TEMPERATURE, assumed_prior
    )
    gold_labels = np.asarray(dataset.train[dataset.label_column], dtype=np.int64)
    diagnostics = teacher_target_diagnostics(soft_probs, gold_labels)
    _TEACHER_TARGET_CACHE[key] = diagnostics
    return diagnostics


def analyse_group(dataset: str, labels_per_class: int) -> dict:
    conditions_payload: dict[str, dict] = {}
    cells_by_condition: dict[str, dict[int, dict]] = {}

    for condition in CONDITIONS:
        acc_values: list[float] = []
        f1_values: list[float] = []
        seed_metrics: list[dict] = []
        cells_by_condition[condition] = {}
        for seed in SEEDS:
            cell = load_cell(dataset, condition, labels_per_class, seed)
            cells_by_condition[condition][seed] = cell
            acc = float(cell["metrics"]["accuracy"])
            f1 = float(cell["metrics"]["macro_f1"])
            acc_values.append(acc)
            f1_values.append(f1)
            seed_metrics.append({"seed": seed, "accuracy": acc, "macro_f1": f1})

        teacher_target = offline_teacher_target(dataset, condition, DOMINANT_CLASS)
        conditions_payload[condition] = {
            "accuracy": mean_std(acc_values),
            "macro_f1": mean_std(f1_values),
            "teacher_target_accuracy": teacher_target["accuracy"],
            "teacher_target_macro_f1": teacher_target["macro_f1"],
            "teacher_target_nll": teacher_target["nll"],
            "seeds": seed_metrics,
        }

    comparisons = {
        "mild_minus_correct": analyse_comparison(
            cells_by_condition["mild"], cells_by_condition["correct"], label_a="mild", label_b="correct"
        ),
        "strong_minus_correct": analyse_comparison(
            cells_by_condition["strong"], cells_by_condition["correct"], label_a="strong", label_b="correct"
        ),
        "strong_minus_mild": analyse_comparison(
            cells_by_condition["strong"], cells_by_condition["mild"], label_a="strong", label_b="mild"
        ),
    }

    return {
        "dataset": dataset,
        "labels_per_class": labels_per_class,
        "n_seeds": len(SEEDS),
        "validation_status": "ok",
        "conditions": conditions_payload,
        "comparisons": comparisons,
    }


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(groups: list[dict], completeness: dict) -> str:
    lines = [
        "# Student-level A.12 prior-misspecification experiment",
        "",
        "We retrained students using the mild and strong misspecification strengths from "
        "Table A.12, with class 0 predeclared as the dominant assumed class. The actual "
        "training pool and the fixed balanced test set are identical across conditions; "
        "only the prior supplied to prior-and-scale-aligned matching changes. This is an "
        "assumed-prior sensitivity experiment on balanced pools, not a label-shift, "
        "unbalanced-pool, covariate-shift, or domain-shift experiment.",
        "",
        "> The student experiment uses one predeclared misspecification direction on three "
        "balanced datasets; it does not average over all possible dominant classes.",
        "",
        f"Seeds: {', '.join(str(s) for s in SEEDS)}. Grid: "
        f"{len(DATASETS)} datasets x {len(LABEL_BUDGETS)} label budgets x {len(SEEDS)} seeds "
        f"x {len(CONDITIONS)} conditions = {N_CELLS_EXPECTED} cells "
        f"({N_CORRECT_BASELINES_EXPECTED} pre-existing correct-prior baselines + "
        f"{N_NEW_CELLS_EXPECTED} new mild/strong cells).",
        "",
        "Primary comparisons: mild − correct, strong − correct. Secondary: strong − mild. "
        "With three seeds, effect sizes/direction/dataset- and budget-dependence are "
        "emphasised over seed-level significance.",
        "",
    ]

    for group in groups:
        ds = group["dataset"]
        lpc = group["labels_per_class"]
        lines.append(f"## {ds} · L={lpc}")
        lines.append("")
        lines.append("### Condition summary (accuracy / macro-F1, student + teacher-target)")
        lines.append("")
        lines.append(
            "| condition | student acc mean±SD | student F1 mean±SD | "
            "teacher-target acc | teacher-target NLL |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for condition in CONDITIONS:
            c = group["conditions"][condition]
            lines.append(
                f"| {condition} | {fmt_pct(c['accuracy']['mean'])} ± "
                f"{fmt_pct(c['accuracy']['std'])} | {fmt_pct(c['macro_f1']['mean'])} ± "
                f"{fmt_pct(c['macro_f1']['std'])} | {fmt_pct(c['teacher_target_accuracy'])} | "
                f"{c['teacher_target_nll']:.4f} |"
            )
        lines.append("")
        lines.append("### Paired comparisons (per seed)")
        lines.append("")
        for cmp_id, desc in (
            ("mild_minus_correct", "mild − correct"),
            ("strong_minus_correct", "strong − correct"),
            ("strong_minus_mild", "strong − mild (secondary)"),
        ):
            cmp = group["comparisons"][cmp_id]
            d_acc = cmp["paired_delta"]["accuracy"]
            wins = cmp["seed_wins"]
            lines.append(f"#### {desc}")
            lines.append("")
            lines.append(
                f"- Mean paired Δacc: {100 * d_acc['mean']:+.2f} ± {100 * d_acc['std']:.2f} "
                f"(descriptive SE {100 * d_acc['se']:.2f}); per-seed: "
                + ", ".join(f"{100 * v:+.2f}" for v in d_acc["values"])
            )
            lines.append(
                f"- Wins/ties/losses ({cmp['a']} vs {cmp['b']}): "
                f"{wins[cmp['a']]}/{wins['tie']}/{wins[cmp['b']]}"
            )
            lines.append("")
            lines.append(
                "| seed | A acc | B acc | Δacc | discordant | only_A | only_B | "
                "McNemar p | Holm p |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for row in cmp["seeds"]:
                mc = row["mcnemar"]
                a_acc = row[f"{cmp['a']}_accuracy"]
                b_acc = row[f"{cmp['b']}_accuracy"]
                lines.append(
                    f"| {row['seed']} | {fmt_pct(a_acc)} | {fmt_pct(b_acc)} | "
                    f"{100 * row['delta_accuracy']:+.2f} | {mc['n_discordant']} | "
                    f"{mc['only_a_correct']} | {mc['only_b_correct']} | "
                    f"{mc['p_value']:.4g} | {mc['p_value_holm']:.4g} |"
                )
            lines.append("")

    lines.append("## Completeness / provenance")
    lines.append("")
    lines.append(f"- Cells found matching this experiment's teacher: {completeness['n_found']}")
    lines.append(f"- Missing cells: {len(completeness['missing_cells'])}")
    lines.append(f"- Duplicate cells: {len(completeness['duplicate_cells'])}")
    lines.append(f"- Misplaced cells (wrong source directory for their condition): {len(completeness['misplaced_cells'])}")
    lines.append(
        f"- Unexpected cells (out-of-scope dataset/budget/seed/condition, excluded from the 54-cell grid): "
        f"{len(completeness['unexpected_cells'])}"
    )
    if completeness["unexpected_cells"]:
        lines.append(f"  - {', '.join(completeness['unexpected_cells'])}")
    lines.append(
        f"- Baseline source: `{BASELINE_DIR}` ({N_CORRECT_BASELINES_EXPECTED} correct-prior cells, not retrained)"
    )
    lines.append(f"- New-cell source: `{NEW_CELLS_DIR}` ({N_NEW_CELLS_EXPECTED} mild/strong cells)")
    lines.append("- Predeclared misspecification direction: class 0 (fixed before examining results)")
    lines.append(
        "- No result here is an average over dominant classes; this is a single predeclared direction."
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
    errors = validate_cell(
        data,
        dataset=args.dataset,
        condition=args.condition,
        labels_per_class=args.labels_per_class,
        seed=args.seed,
        dominant_class=args.dominant_class,
    )
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse the student-level A.12 prior-misspecification experiment."
    )
    parser.add_argument(
        "--check-cell",
        default=None,
        help="Validate a single result JSON and exit 0/1 (for the GPU runner).",
    )
    parser.add_argument("--dataset", default=None, choices=list(DATASETS))
    parser.add_argument("--condition", default=None, choices=list(CONDITIONS))
    parser.add_argument("--labels-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dominant-class", type=int, default=DOMINANT_CLASS)
    args = parser.parse_args()

    if args.check_cell is not None:
        required = ("dataset", "condition", "labels_per_class", "seed")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            print(
                f"--check-cell requires: {', '.join('--' + m.replace('_', '-') for m in missing)}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise SystemExit(check_cell_cli(args))

    completeness = scan_grid_completeness()

    groups: list[dict] = []
    failures: list[str] = list(completeness["missing_cells"])
    failures += [f"duplicate cell: {k}" for k in completeness["duplicate_cells"]]
    failures += [f"misplaced cell: {k}" for k in completeness["misplaced_cells"]]

    if not failures:
        for dataset in DATASETS:
            for labels_per_class in LABEL_BUDGETS:
                try:
                    groups.append(analyse_group(dataset, labels_per_class))
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    failures.append(f"{dataset} L={labels_per_class}: {exc}")

    if failures:
        print(
            f"Student prior-misspecification analysis incomplete (expected {N_CELLS_EXPECTED} cells):",
            file=sys.stderr,
        )
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "student_prior_misspecification_a12_v1",
        "label_accounting": "prompt-inclusive-v1",
        "artifact_version": "large-test-v1",
        "teacher": TEACHER,
        "aggregation": EXPECTED_AGGREGATION,
        "datasets": list(DATASETS),
        "label_budgets": list(LABEL_BUDGETS),
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "dominant_class": DOMINANT_CLASS,
        "strengths": STRENGTH,
        "n_cells_expected": N_CELLS_EXPECTED,
        "n_correct_baselines_expected": N_CORRECT_BASELINES_EXPECTED,
        "n_new_cells_expected": N_NEW_CELLS_EXPECTED,
        "baseline_source": str(BASELINE_DIR),
        "new_cells_source": str(NEW_CELLS_DIR),
        "scope_note": (
            "Table A.12 averages each misspecification strength over every possible "
            "dominant class. This student experiment uses class 0 as one predeclared "
            "representative direction; it is not an exact student replication of the "
            "class-averaged A.12 values."
        ),
        "completeness": completeness,
        "groups": groups,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(render_markdown(groups, completeness), encoding="utf-8")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
