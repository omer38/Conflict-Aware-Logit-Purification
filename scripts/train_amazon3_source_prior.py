#!/usr/bin/env python
"""Amazon-3 source-prior transfer: student-level replication under a
source-induced non-uniform prior.

This is **not** a natural-deployment-prevalence experiment. The 40/20/40
prior arises only from grouping the balanced, gold-constructed Amazon-5
benchmark (500/class train, 2000/class test) into 1-2 stars -> negative,
3 stars -> neutral, 4-5 stars -> positive. A fixed pool seed splits the 2,500
grouped training rows into a 500-row unlabelled reference batch and a
2,000-row student working pool (both exactly 40/20/40); the fixed,
10,000-row grouped test set is untouched. Class-prior matching and
confidence-scale alignment are fit on the reference batch only and
transferred (applied without refitting) to the working pool -- out of
sample, unlike the transductive alignment used elsewhere in the thesis.

No teacher queries are made anywhere in this module: guided-teacher grouped
logits are derived from the existing cached 5-class Amazon logits via the
same transform as ``scripts/build_amazon3.py``.

Usage:
    uv run python scripts/train_amazon3_source_prior.py \\
        --method estimated_prior_bbse --labels-per-class 10 --seed 42
    uv run python scripts/train_amazon3_source_prior.py \\
        --method scale_only --labels-per-class 50 --seed 456 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StudentConfig  # noqa: E402
from src.data.loading import (  # noqa: E402
    DATA_ARTIFACT_VERSION,
    TextClassificationDataset,
    load_amazon,
    split_identity,
)
from src.data.subsampling import (  # noqa: E402
    DataSplits,
    PROMPT_DEMOS_PER_CLASS,
    prompt_demo_indices,
)
from src.student.model import build_student, select_device  # noqa: E402
from src.student.prior_estimation import (  # noqa: E402
    apply_bias_corrected_calibration,
    bootstrap_prior_ci,
    committee_probs,
    confusion_moment_matrix,
    estimate_prior_bbse,
    estimate_prior_mlls,
    fit_bias_corrected_calibration,
    prior_error_metrics,
    reference_marginal as compute_reference_marginal,
)
from src.student.soft_labels import (  # noqa: E402
    GUIDED_TEACHERS,
    REPORT_DIR,
    _logits_to_probs,
    apply_confidence_scale,
    apply_prior_bias,
    fit_confidence_scale,
    fit_prior_bias,
    load_teacher_logits,
    probability_mean_kd_target,
    teacher_target_diagnostics_full,
)
from src.student.train import build_examples, evaluate, fit, set_seed  # noqa: E402

import scripts.build_amazon3 as amazon3_builder  # noqa: E402

EXPERIMENT_ID = "amazon3_source_prior_transfer_v1"
MLLS_EXPERIMENT_ID = "amazon3_source_prior_transfer_mlls_v1"

GROUP_MAP: dict[int, int] = dict(amazon3_builder.GROUP_MAP)  # {0:0,1:0,2:1,3:2,4:2}
GROUP_NAMES = ["negative", "neutral", "positive"]
GROUP_MAP_DESCRIPTION = {"1-2 stars": "negative", "3 stars": "neutral", "4-5 stars": "positive"}

TRUE_PRIOR = np.array([0.4, 0.2, 0.4], dtype=np.float64)
UNIFORM_PRIOR = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

# Single fixed, predeclared pool seed for the reference/working split. It is
# a separate concept from the per-run labelled-sampling seed below.
POOL_SPLIT_SEED = 20260101
REFERENCE_PER_GROUP = {0: 200, 1: 100, 2: 200}
WORKING_PER_GROUP = {0: 800, 1: 400, 2: 800}
FULL_GROUP_TRAIN_COUNTS = {0: 1000, 1: 500, 2: 1000}
FULL_GROUP_TEST_COUNTS = {0: 4000, 1: 2000, 2: 4000}

METHODS = ("scale_only", "uniform_prior", "estimated_prior_bbse", "oracle_prior")
MLLS_METHOD = "estimated_prior_mlls_bcts"
CLI_METHODS = (*METHODS, MLLS_METHOD)
LABEL_BUDGETS = (10, 50)
SEEDS = (42, 123, 456)

# Fixed analysis seed for the bootstrap prior CI -- a CPU diagnostic only,
# never used to alter the prior supplied to student training.
PRIOR_BOOTSTRAP_SEED = 20260102
PRIOR_BOOTSTRAP_REPLICATES = 1000

STUDENT_MODEL_NAME = "answerdotai/ModernBERT-base"
STUDENT_EPOCHS = 3
STUDENT_LR = 2e-5
STUDENT_BATCH_SIZE = 16
STUDENT_MAX_LENGTH = 256
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5

DEFAULT_OUTPUT_DIR = "outputs/student/amazon3_source_prior_transfer_v1"


# ---------------------------------------------------------------------------
# Deterministic grouped-dataset construction (no new artifact is saved).
# ---------------------------------------------------------------------------
def build_grouped_labels(amazon: TextClassificationDataset) -> tuple[np.ndarray, np.ndarray]:
    """Group Amazon-5 star ratings into 3 classes, preserving row order (no rebalancing)."""
    group_vector = np.array([GROUP_MAP[c] for c in range(5)], dtype=np.int64)
    train_labels5 = np.asarray(amazon.train["label"], dtype=np.int64)
    test_labels5 = np.asarray(amazon.test["label"], dtype=np.int64)
    return group_vector[train_labels5], group_vector[test_labels5]


def build_grouped_dataset(
    amazon: TextClassificationDataset,
    train_grouped: np.ndarray,
    test_grouped: np.ndarray,
) -> TextClassificationDataset:
    """Full grouped (non-rebalanced) Amazon-3 dataset, built in memory only.

    Reuses the exact review texts from ``data/amazon``; never writes a new
    dataset artifact and never overwrites ``data/amazon`` or ``data/amazon3``.
    """
    train = amazon.train.remove_columns(["label"]).add_column("label", train_grouped.tolist())
    test = amazon.test.remove_columns(["label"]).add_column("label", test_grouped.tolist())
    return TextClassificationDataset(
        name="amazon3_source_prior_transfer",
        train=train,
        test=test,
        text_column="text",
        label_column="label",
        num_classes=3,
        label_names=GROUP_NAMES,
        metadata={
            "artifact_version": DATA_ARTIFACT_VERSION,
            "parent_dataset": "amazon",
            "source": (
                "grouped in-memory from data/amazon (5-class, gold-balanced); "
                "not rebalanced; not written to disk"
            ),
            "group_map": GROUP_MAP_DESCRIPTION,
        },
    )


def grouped_prompt_demo_indices(amazon: TextClassificationDataset) -> dict[int, list[int]]:
    """Map the fixed 5-class teacher-prompt demonstrations onto grouped classes.

    Two demonstrations per original star class -> 4 grouped-negative, 2
    grouped-neutral, 4 grouped-positive (10 total); unequal by construction.
    """
    demos_by_star = prompt_demo_indices(amazon, per_class=PROMPT_DEMOS_PER_CLASS)
    grouped: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for star_class, indices in demos_by_star.items():
        grouped[GROUP_MAP[star_class]].extend(indices)
    return {group: sorted(indices) for group, indices in grouped.items()}


def build_reference_working_split(
    train_grouped: np.ndarray,
    demo_by_group: dict[int, list[int]],
    pool_seed: int = POOL_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the grouped train set into a disjoint, exhaustive reference/working pair.

    Every prompt demonstration is forced into the working pool and excluded
    from the reference batch. Both resulting pools carry the exact 40/20/40
    prior by construction (each group is split at a fixed ratio).
    """
    rng = np.random.default_rng(pool_seed)
    reference_parts: list[np.ndarray] = []
    working_parts: list[np.ndarray] = []

    for group in range(3):
        group_indices = np.where(train_grouped == group)[0]
        if len(group_indices) != FULL_GROUP_TRAIN_COUNTS[group]:
            raise RuntimeError(
                f"group {group} has {len(group_indices)} rows, "
                f"expected {FULL_GROUP_TRAIN_COUNTS[group]}"
            )
        demo_indices = np.asarray(demo_by_group[group], dtype=np.int64)
        non_demo = np.setdiff1d(group_indices, demo_indices)
        n_reference = REFERENCE_PER_GROUP[group]
        reference_group = rng.choice(non_demo, size=n_reference, replace=False)
        working_group = np.setdiff1d(group_indices, reference_group)
        reference_parts.append(reference_group)
        working_parts.append(working_group)

    reference_indices = np.sort(np.concatenate(reference_parts)).astype(np.int64)
    working_indices = np.sort(np.concatenate(working_parts)).astype(np.int64)

    if np.intersect1d(reference_indices, working_indices).size:
        raise RuntimeError("reference and working indices are not disjoint")
    if reference_indices.size + working_indices.size != len(train_grouped):
        raise RuntimeError("reference and working indices do not exhaust the grouped train set")
    for group in range(3):
        ref_count = int((train_grouped[reference_indices] == group).sum())
        work_count = int((train_grouped[working_indices] == group).sum())
        if ref_count != REFERENCE_PER_GROUP[group] or work_count != WORKING_PER_GROUP[group]:
            raise RuntimeError(
                f"group {group} split counts wrong: reference={ref_count} working={work_count}"
            )

    all_demo_indices = np.asarray(
        [i for indices in demo_by_group.values() for i in indices], dtype=np.int64
    )
    if np.intersect1d(all_demo_indices, reference_indices).size:
        raise RuntimeError("a prompt-demonstration index leaked into the reference batch")
    if not np.all(np.isin(all_demo_indices, working_indices)):
        raise RuntimeError("not every prompt-demonstration index is in the working pool")

    return reference_indices, working_indices


def select_labelled_indices(
    working_indices: np.ndarray,
    train_grouped: np.ndarray,
    demo_by_group: dict[int, list[int]],
    labels_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly ``labels_per_class`` working-pool labels per grouped class.

    Forces every original prompt demonstration into the selection, then
    samples the remainder without replacement using ``seed``.
    """
    rng = np.random.default_rng(seed)
    labelled_parts: list[np.ndarray] = []
    unlabelled_parts: list[np.ndarray] = []

    for group in range(3):
        group_working = working_indices[train_grouped[working_indices] == group]
        demo_indices = np.asarray(demo_by_group[group], dtype=np.int64)
        if not np.all(np.isin(demo_indices, group_working)):
            raise RuntimeError(f"group {group} prompt demonstrations are missing from the working pool")
        if len(demo_indices) > labels_per_class:
            raise ValueError(
                f"labels_per_class={labels_per_class} is smaller than the "
                f"{len(demo_indices)} forced prompt demonstrations for group {group}"
            )
        non_demo_working = np.setdiff1d(group_working, demo_indices)
        n_extra = labels_per_class - len(demo_indices)
        extra = rng.choice(non_demo_working, size=n_extra, replace=False)
        chosen = np.concatenate([demo_indices, extra])
        labelled_parts.append(chosen)
        unlabelled_parts.append(np.setdiff1d(group_working, chosen))

    labelled_indices = np.sort(np.concatenate(labelled_parts)).astype(np.int64)
    unlabelled_indices = np.sort(np.concatenate(unlabelled_parts)).astype(np.int64)

    if labelled_indices.size != 3 * labels_per_class:
        raise RuntimeError("labelled index count != 3 * labels_per_class")
    if np.intersect1d(labelled_indices, unlabelled_indices).size:
        raise RuntimeError("labelled and unlabelled indices are not disjoint")
    union = np.union1d(labelled_indices, unlabelled_indices)
    if union.size != working_indices.size or np.setdiff1d(union, working_indices).size:
        raise RuntimeError("labelled/unlabelled indices do not exactly partition the working pool")

    return labelled_indices, unlabelled_indices


# ---------------------------------------------------------------------------
# Grouped teacher logits, derived from the existing 5-class cache (no queries).
# ---------------------------------------------------------------------------
def derive_grouped_teacher_logits(teacher: str) -> np.ndarray:
    """Grouped 3-class pseudo-logits for *teacher*, from cached 5-class logits.

    Uses ``amazon3_builder._grouped_pseudo_logits`` verbatim -- the same
    implementation and numerical conventions as ``scripts/build_amazon3.py``
    -- applied to ``load_teacher_logits(teacher, "amazon", 5)``. Row order
    (position i <-> ``load_amazon().train[i]``) is preserved.
    """
    logits5 = load_teacher_logits(teacher, "amazon", 5)
    return amazon3_builder._grouped_pseudo_logits(logits5)


def stack_grouped_teacher_logits() -> np.ndarray:
    """``(K, N, C)`` grouped pseudo-logits for the three guided teachers."""
    return np.stack([derive_grouped_teacher_logits(t) for t in GUIDED_TEACHERS], axis=0)


def existing_amazon3_grouped_logits(teacher: str) -> tuple[np.ndarray, np.ndarray]:
    """Read back the cached Amazon-3 builder CSV and the rows it was built from.

    Used only to verify :func:`derive_grouped_teacher_logits` reproduces the
    existing cached Amazon-3 pseudo-logits for the rows the balanced builder
    selected (``scripts/build_amazon3.py``); not used by the training flow.
    """
    amazon = load_amazon()
    labels5 = np.asarray(amazon.train["label"], dtype=np.int64)
    train_chosen = amazon3_builder._grouped_balanced_indices(labels5, amazon3_builder.SEED)
    report_path = REPORT_DIR / f"{teacher}_amazon3.csv"
    rows: dict[int, list[float]] = {}
    with report_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row["sample_idx"])] = [
                float(row["logit_A"]), float(row["logit_B"]), float(row["logit_C"])
            ]
    cached = np.array([rows[i] for i in range(len(train_chosen))], dtype=np.float64)
    return np.asarray(train_chosen, dtype=np.int64), cached


# ---------------------------------------------------------------------------
# Prior estimation (BBSE) and out-of-sample alignment.
# ---------------------------------------------------------------------------
def indices_hash(indices: np.ndarray) -> str:
    arr = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def run_prior_estimation(
    stacked_raw: np.ndarray,
    train_grouped: np.ndarray,
    labelled_indices: np.ndarray,
    reference_indices: np.ndarray,
) -> dict:
    """Fixed-committee soft-BBSE estimate, plus bootstrap CI and error metrics.

    The committee predictor never sees any assumed/target prior and never
    sees reference-batch labels (only reference-batch predictions).
    """
    g_all = committee_probs(stacked_raw)  # (N, 3); fixed, temp=1, label-free
    declared_probs = g_all[labelled_indices]
    declared_labels = train_grouped[labelled_indices]
    reference_probs = g_all[reference_indices]

    matrix = confusion_moment_matrix(declared_probs, declared_labels, num_classes=3)
    mu = compute_reference_marginal(reference_probs)
    estimate = estimate_prior_bbse(matrix, mu)
    error_metrics = prior_error_metrics(estimate.estimated_prior, TRUE_PRIOR)
    bootstrap = bootstrap_prior_ci(
        declared_probs,
        declared_labels,
        reference_probs,
        num_classes=3,
        n_replicates=PRIOR_BOOTSTRAP_REPLICATES,
        seed=PRIOR_BOOTSTRAP_SEED,
    )

    payload = estimate.to_json()
    payload.update(error_metrics)
    payload["bootstrap_ci"] = bootstrap.to_json()
    payload["true_prior"] = TRUE_PRIOR.tolist()
    payload["committee"] = {
        "teachers": list(GUIDED_TEACHERS),
        "aggregation": "arithmetic_probability_mean",
        "temperature": 1.0,
        "uses_labels": False,
        "uses_prior_matching": False,
        "uses_confidence_scaling": False,
        "fixed_across_methods": True,
    }
    payload["n_declared_labels"] = int(len(labelled_indices))
    payload["n_reference_rows"] = int(len(reference_indices))
    return payload, estimate


def run_prior_estimation_mlls(
    stacked_raw: np.ndarray,
    train_grouped: np.ndarray,
    labelled_indices: np.ndarray,
    reference_indices: np.ndarray,
) -> tuple[dict, object]:
    """Bias-corrected calibration plus maximum-likelihood prior estimation.

    The deliberately stratified declared-label sample has a known uniform
    source prior. It calibrates the same fixed committee used by BBSE, while
    the disjoint reference batch contributes predictions only.
    """
    g_all = committee_probs(stacked_raw)
    declared_probs = g_all[labelled_indices]
    declared_labels = train_grouped[labelled_indices]
    reference_probs = g_all[reference_indices]
    source_prior = np.full(3, 1.0 / 3.0, dtype=np.float64)

    calibration = fit_bias_corrected_calibration(declared_probs, declared_labels)
    calibrated_reference = apply_bias_corrected_calibration(reference_probs, calibration)
    estimate = estimate_prior_mlls(calibrated_reference, source_prior)
    payload = estimate.to_json()
    payload.update(prior_error_metrics(estimate.estimated_prior, TRUE_PRIOR))
    payload.update(
        {
            "estimator": "maximum_likelihood_label_shift",
            "calibration_method": "bias_corrected_temperature_scaling",
            "calibration": calibration.to_json(),
            "true_prior": TRUE_PRIOR.tolist(),
            "committee": {
                "teachers": list(GUIDED_TEACHERS),
                "aggregation": "arithmetic_probability_mean",
                "temperature": 1.0,
                "uses_labels": False,
                "uses_prior_matching": False,
                "uses_confidence_scaling": False,
                "fixed_across_methods": True,
            },
            "n_declared_labels": int(len(labelled_indices)),
            "n_reference_rows": int(len(reference_indices)),
            "reference_labels_used": False,
        }
    )
    return payload, estimate


def fit_apply_alignment(
    method: str,
    stacked_raw: np.ndarray,
    reference_indices: np.ndarray,
    working_indices: np.ndarray,
    assumed_prior: np.ndarray | None,
) -> dict:
    """Fit prior-bias/confidence-scale parameters on the reference batch only,
    then apply the frozen parameters to the working pool (out of sample)."""
    reference_logits = stacked_raw[:, reference_indices, :]
    working_logits = stacked_raw[:, working_indices, :]
    num_teachers, _, num_classes = stacked_raw.shape
    prior_matching = method != "scale_only"

    if prior_matching:
        if assumed_prior is None:
            raise ValueError(f"method {method!r} requires assumed_prior")
        bias = fit_prior_bias(reference_logits, assumed_prior)
        matched_reference_logits = apply_prior_bias(reference_logits, bias)
        matched_working_logits = apply_prior_bias(working_logits, bias)
    else:
        bias = np.zeros((num_teachers, num_classes), dtype=np.float64)
        matched_reference_logits = reference_logits
        matched_working_logits = working_logits

    scales = fit_confidence_scale(matched_reference_logits)
    aligned_reference_logits = apply_confidence_scale(matched_reference_logits, scales)
    aligned_working_logits = apply_confidence_scale(matched_working_logits, scales)

    reference_probs = _logits_to_probs(aligned_reference_logits, temperature=1.0)
    working_probs = _logits_to_probs(aligned_working_logits, temperature=1.0)
    achieved_reference_marginal = reference_probs.mean(axis=1)  # (K, C)
    achieved_working_marginal = working_probs.mean(axis=1)  # (K, C)
    reference_entropy = float(
        (-(reference_probs * np.log(np.clip(reference_probs, 1e-12, None))).sum(-1)).mean()
    )
    working_entropy = float(
        (-(working_probs * np.log(np.clip(working_probs, 1e-12, None))).sum(-1)).mean()
    )

    reference_marginal_error = (
        float(np.abs(achieved_reference_marginal.mean(axis=0) - assumed_prior).sum())
        if prior_matching
        else None
    )
    working_marginal_error = float(np.abs(achieved_working_marginal.mean(axis=0) - TRUE_PRIOR).sum())

    p_bar_working = working_probs.mean(axis=0)  # pre-KD-temperature aggregate (N_working, C)
    working_kd_probs = probability_mean_kd_target(aligned_working_logits, temperature=KD_TEMPERATURE)

    fit_payload = {
        "method": method,
        "prior_matching": prior_matching,
        "assumed_prior": assumed_prior.tolist() if assumed_prior is not None else None,
        "bias": bias.tolist(),
        "scales": scales.tolist(),
        "order": ["prior_matching", "confidence_scale_alignment", "probability_averaging", "kd_temperature"],
        "fitting_split": "reference_batch",
        "teachers": list(GUIDED_TEACHERS),
    }
    application_payload = {
        "application_split": "working_pool",
        "achieved_reference_marginal_per_teacher": achieved_reference_marginal.tolist(),
        "achieved_working_marginal_per_teacher": achieved_working_marginal.tolist(),
        "reference_marginal_error_l1": reference_marginal_error,
        "working_marginal_error_l1": working_marginal_error,
        "reference_mean_entropy": reference_entropy,
        "working_mean_entropy": working_entropy,
    }
    return {
        "alignment_fit": fit_payload,
        "alignment_application": application_payload,
        "working_kd_probs": working_kd_probs,
        "p_bar_working": p_bar_working,
    }


# ---------------------------------------------------------------------------
# Full CPU-only cell preparation (shared by --dry-run and real training).
# ---------------------------------------------------------------------------
def prepare_cell(method: str, labels_per_class: int, seed: int) -> dict:
    if method not in CLI_METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {CLI_METHODS}")
    if labels_per_class not in LABEL_BUDGETS:
        raise ValueError(f"labels_per_class must be one of {LABEL_BUDGETS}, got {labels_per_class}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed}")

    amazon = load_amazon()
    train_grouped, test_grouped = build_grouped_labels(amazon)
    demo_by_group = grouped_prompt_demo_indices(amazon)
    reference_indices, working_indices = build_reference_working_split(train_grouped, demo_by_group)
    labelled_indices, unlabelled_indices = select_labelled_indices(
        working_indices, train_grouped, demo_by_group, labels_per_class, seed
    )

    grouped_dataset = build_grouped_dataset(amazon, train_grouped, test_grouped)
    train_identity = split_identity(grouped_dataset.train, "text", "label")
    test_identity = split_identity(grouped_dataset.test, "text", "label")

    stacked_raw = stack_grouped_teacher_logits()
    probs_check = _logits_to_probs(stacked_raw, temperature=1.0)
    if not np.isfinite(stacked_raw).all() or not np.isfinite(probs_check).all():
        raise RuntimeError("grouped teacher logits/probabilities contain non-finite values")

    if method == MLLS_METHOD:
        prior_estimator_payload, prior_estimate = run_prior_estimation_mlls(
            stacked_raw, train_grouped, labelled_indices, reference_indices
        )
        calibration_success = prior_estimator_payload["calibration"]["solver_success"]
        if not prior_estimator_payload["solver_success"] or not calibration_success:
            raise RuntimeError(
                "BCTS+MLLS prior estimation failed "
                f"(calibration_success={calibration_success}, "
                f"mlls_success={prior_estimator_payload['solver_success']}). "
                "Stopping before GPU training."
            )
    else:
        prior_estimator_payload, prior_estimate = run_prior_estimation(
            stacked_raw, train_grouped, labelled_indices, reference_indices
        )
        if not prior_estimator_payload["solver_success"] or prior_estimator_payload["matrix_rank"] < 3:
            raise RuntimeError(
                "Prior estimator failed or the confusion-moment matrix is rank-deficient "
                f"(success={prior_estimator_payload['solver_success']}, "
                f"rank={prior_estimator_payload['matrix_rank']}). Stopping before GPU training."
            )

    assumed_prior_by_method: dict[str, np.ndarray | None] = {
        "scale_only": None,
        "uniform_prior": UNIFORM_PRIOR,
        "estimated_prior_bbse": prior_estimate.estimated_prior,
        MLLS_METHOD: prior_estimate.estimated_prior,
        "oracle_prior": TRUE_PRIOR,
    }
    assumed_prior = assumed_prior_by_method[method]

    alignment = fit_apply_alignment(method, stacked_raw, reference_indices, working_indices, assumed_prior)

    gold_working = train_grouped[working_indices]
    teacher_diag = teacher_target_diagnostics_full(alignment["p_bar_working"], gold_working, TRUE_PRIOR)
    teacher_diag["scope"] = "offline_working_pool_source_induced_prior"

    all_demo_indices = np.asarray(
        sorted(i for indices in demo_by_group.values() for i in indices), dtype=np.int64
    )
    prompt_label_accounting = {
        "total_prompt_labels": int(len(all_demo_indices)),
        "per_group": {GROUP_NAMES[g]: len(demo_by_group[g]) for g in range(3)},
        "original_star_demo_indices": all_demo_indices.tolist(),
        "note": (
            "Grouped demonstration counts are unequal by construction (4 negative, "
            "2 neutral, 4 positive) because the cached teacher prompts used 2 "
            "demonstrations per original 5-class star rating, not 2 per grouped class."
        ),
    }

    soft_probs_full = np.full((len(grouped_dataset.train), 3), np.nan, dtype=np.float64)
    soft_probs_full[working_indices] = alignment["working_kd_probs"]

    result = {
        "experiment_id": MLLS_EXPERIMENT_ID if method == MLLS_METHOD else EXPERIMENT_ID,
        "method": method,
        "dataset": "amazon3_source_prior_transfer",
        "parent_dataset": "amazon",
        "group_map": GROUP_MAP_DESCRIPTION,
        "true_prior": TRUE_PRIOR.tolist(),
        "assumed_prior": assumed_prior.tolist() if assumed_prior is not None else None,
        "prior_estimator": prior_estimator_payload,
        "reference_batch": {
            "indices": reference_indices.tolist(),
            "identity_hash": indices_hash(reference_indices),
            "size": int(len(reference_indices)),
            "per_group": {GROUP_NAMES[g]: REFERENCE_PER_GROUP[g] for g in range(3)},
            "pool_split_seed": POOL_SPLIT_SEED,
            "audit_true_group_counts": {
                GROUP_NAMES[g]: int((train_grouped[reference_indices] == g).sum()) for g in range(3)
            },
            "note": "Reference labels are used only to construct/audit this split, never by the estimator.",
        },
        "working_pool": {
            "indices": working_indices.tolist(),
            "identity_hash": indices_hash(working_indices),
            "size": int(len(working_indices)),
            "per_group": {GROUP_NAMES[g]: WORKING_PER_GROUP[g] for g in range(3)},
            "pool_split_seed": POOL_SPLIT_SEED,
        },
        "labelled_split": {
            "labelled_indices": labelled_indices.tolist(),
            "unlabelled_indices": unlabelled_indices.tolist(),
            "labels_per_class": labels_per_class,
            "seed": seed,
        },
        "prompt_label_accounting": prompt_label_accounting,
        "alignment_fit": alignment["alignment_fit"],
        "alignment_application": alignment["alignment_application"],
        "teacher_target_diagnostics": teacher_diag,
        "dataset_identity": {
            "artifact_version": grouped_dataset.metadata["artifact_version"],
            "parent_source": amazon.metadata.get("source"),
            "train": train_identity,
            "test": test_identity,
            "test_per_class": FULL_GROUP_TEST_COUNTS,
            "constructed_on_the_fly": True,
        },
    }
    return {
        "result": result,
        "grouped_dataset": grouped_dataset,
        "labelled_indices": labelled_indices,
        "unlabelled_indices": unlabelled_indices,
        "all_demo_indices": all_demo_indices,
        "soft_probs_full": soft_probs_full,
        "train_grouped": train_grouped,
        "test_grouped": test_grouped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one Amazon-3 source-prior-transfer student cell."
    )
    parser.add_argument("--method", required=True, choices=CLI_METHODS)
    parser.add_argument("--labels-per-class", type=int, required=True, choices=LABEL_BUDGETS)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and validate the split/estimator/alignment pipeline and print "
            "diagnostics, then exit before loading or training ModernBERT. Writes "
            "no result file and makes no teacher queries."
        ),
    )
    args = parser.parse_args()

    print(
        f"=== {MLLS_EXPERIMENT_ID if args.method == MLLS_METHOD else EXPERIMENT_ID}: "
        f"method={args.method} L={args.labels_per_class} seed={args.seed} ==="
    )
    prepared = prepare_cell(args.method, args.labels_per_class, args.seed)
    result = prepared["result"]
    prior_meta = result["prior_estimator"]

    print(f"True prior: {result['true_prior']} | assumed prior: {result['assumed_prior']}")
    if args.method == MLLS_METHOD:
        calibration = prior_meta["calibration"]
        print(
            f"BCTS+MLLS estimate: {prior_meta['estimated_prior']} "
            f"(l1_error={prior_meta['l1_error']:.4f}, "
            f"tv_error={prior_meta['total_variation_error']:.4f}, "
            f"temperature={calibration['temperature']:.4f}, "
            f"iterations={prior_meta['solver_iterations']}, "
            f"solver_success={prior_meta['solver_success']})"
        )
    else:
        print(
            f"BBSE estimate: {prior_meta['estimated_prior']} "
            f"(l1_error={prior_meta['l1_error']:.4f}, tv_error={prior_meta['total_variation_error']:.4f}, "
            f"rank={prior_meta['matrix_rank']}, cond={prior_meta['condition_number']:.3f}, "
            f"residual={prior_meta['residual']:.6f}, solver_success={prior_meta['solver_success']})"
        )
    print(
        f"Reference batch: n={result['reference_batch']['size']} hash={result['reference_batch']['identity_hash'][:16]}... | "
        f"Working pool: n={result['working_pool']['size']} hash={result['working_pool']['identity_hash'][:16]}..."
    )
    print(
        f"Working-pool teacher-marginal L1 error vs true prior: "
        f"{result['alignment_application']['working_marginal_error_l1']:.4f}"
    )

    if args.dry_run:
        soft_probs_full = prepared["soft_probs_full"]
        working_indices = np.asarray(result["working_pool"]["indices"], dtype=np.int64)
        working_soft = soft_probs_full[working_indices]
        if not np.isfinite(working_soft).all():
            raise RuntimeError("working-pool KD targets contain non-finite values")
        if np.any(working_soft < 0):
            raise RuntimeError("working-pool KD targets contain negative values")
        row_sums = working_soft.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise RuntimeError("working-pool KD targets do not sum to 1")
        print(
            f"KD targets for {len(working_indices)} working-pool rows: all finite, non-negative, "
            f"rows sum to 1 (max |sum-1|={float(np.max(np.abs(row_sums - 1.0))):.2e})"
        )
        print("Dry run OK: exiting before loading or training ModernBERT. No teacher queries made.")
        return

    grouped_dataset = prepared["grouped_dataset"]
    labelled_indices = prepared["labelled_indices"]
    unlabelled_indices = prepared["unlabelled_indices"]
    all_demo_indices = prepared["all_demo_indices"]
    soft_probs_full = prepared["soft_probs_full"]

    splits = DataSplits(
        labeled_train=grouped_dataset.train.select(labelled_indices.tolist()),
        labeled_cal=grouped_dataset.train.select(labelled_indices[:1].tolist()),
        unlabeled=grouped_dataset.train.select(unlabelled_indices.tolist()),
        labeled_train_indices=labelled_indices,
        labeled_cal_indices=np.array([], dtype=np.int64),
        unlabeled_indices=unlabelled_indices,
        labeled_indices=labelled_indices,
        prompt_demo_indices=all_demo_indices,
        labels_per_class=args.labels_per_class,
        seed=args.seed,
        cal_ratio=0.0,
    )

    config = StudentConfig(
        model_name=STUDENT_MODEL_NAME,
        max_length=STUDENT_MAX_LENGTH,
        lr=STUDENT_LR,
        batch_size=STUDENT_BATCH_SIZE,
        epochs=STUDENT_EPOCHS,
        temperature=KD_TEMPERATURE,
        alpha=KD_ALPHA,
    )
    set_seed(args.seed)
    device = select_device()
    print(f"Device: {device}")

    train_ds = build_examples(grouped_dataset, splits, "kd", soft_probs_full, reserve_calibration=False)
    print(f"Training on {len(train_ds)} working-pool examples (alpha={config.alpha}, tau={config.temperature})")

    model, tokenizer = build_student(config.model_name, grouped_dataset.num_classes)
    fit(model, tokenizer, train_ds, config, device)
    evaluation = evaluate(model, tokenizer, grouped_dataset, config, device, return_predictions=True)
    test_labels = np.asarray(evaluation.pop("test_labels"), dtype=np.int64)
    test_predictions = np.asarray(evaluation.pop("test_predictions"), dtype=np.int64)
    metrics = evaluation

    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_predictions, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(test_labels, test_predictions, labels=[0, 1, 2])

    print(f"\nTest accuracy: {metrics['accuracy']:.4f} | macro-F1: {metrics['macro_f1']:.4f}")

    result["student_config"] = vars(config)
    result["metrics"] = metrics
    result["per_class_metrics"] = {
        "labels": GROUP_NAMES,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    result["paired_test_data"] = {
        "sample_indices": list(range(len(test_labels))),
        "labels": test_labels.tolist(),
        "predictions": test_predictions.tolist(),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"amazon3_source-prior-transfer_{args.method}_L{args.labels_per_class}_s{args.seed}.json"
    )
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
