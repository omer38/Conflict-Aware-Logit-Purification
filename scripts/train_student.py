#!/usr/bin/env python
"""Train a ModernBERT student via supervised CE or single-teacher KD.

Usage:
    uv run python scripts/train_student.py --dataset ag_news --mode supervised \
        --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher qwen3.5-27b-vllm-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-mean-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-entropy-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-probmean-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-prior-scale-aligned-guided --labels-per-class 10 --seed 42
    # Assumed-prior misspecification (Table A.12 style; balanced pool unchanged).
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-prior-scale-aligned-guided --prior-condition mild \
        --prior-dominant-class 0 --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-cascade-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-cascade-guided --cascade-order gemma-first \
        --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-uniform-budget-guided --uniform-target-queries 1.4 \
        --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-prior-matched-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-scale-aligned-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher qwen-prior-matched-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher oracle-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-router-linear-guided --labels-per-class 10 --seed 42
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-router-mlp-guided --labels-per-class 10 --seed 42
    # Calibration-allocation diagnostic: 3 CE / 7 router-calibration labels
    # per class at L=10.  This is not a main-result configuration.
    uv run python scripts/train_student.py --dataset ag_news --mode kd \
        --teacher multi-router-linear-guided --labels-per-class 10 \
        --cal-ratio 0.7 --seed 42 --output-dir outputs/student/router_cal70_diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StudentConfig
from src.data.loading import load_dataset_by_name, split_identity
from src.data.subsampling import PROMPT_DEMOS_PER_CLASS, subsample_low_label
from src.disagreement.features import FEATURE_GROUPS, compute_features_for_dataset
from src.router.oracle import MULTI_ORACLE_TEACHER, oracle_teacher_soft_labels
from src.router.purify import purified_soft_labels
from src.router.train_router import (
    MULTI_ROUTER_LINEAR_TEACHER,
    MULTI_ROUTER_MLP_TEACHER,
    entropy_prior_log_weights,
    loocv_router_score,
    select_router_or_fallback,
    standardize_features,
    train_router,
)
from src.student.model import build_student, select_device
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    MULTI_CALIBRATED_TEACHER,
    MULTI_CASCADE_TEACHER,
    MULTI_DEPTH_MATCHED_RANDOM_TEACHER,
    MULTI_ENTROPY_TEACHER,
    MULTI_ENTROPYEQ_TEACHER,
    MULTI_MARGINAL_TEACHER,
    MULTI_MEAN_TEACHER,
    MULTI_PROBMEAN_TEACHER,
    MULTI_UNIFORM_BUDGET_TEACHER,
    QWEN_CALIBRATED_TEACHER,
    MULTI_PRIOR_MATCHED_TEACHER,
    MULTI_PRIOR_SCALE_ALIGNED_TEACHER,
    MULTI_SCALE_ALIGNED_TEACHER,
    QWEN_PRIOR_MATCHED_TEACHER,
    average_teacher_soft_labels,
    cascade_teacher_soft_labels,
    depth_matched_random_teacher_soft_labels,
    entropy_weighted_teacher_soft_labels,
    load_multi_teacher_logits,
    prior_matched_single_teacher_soft_labels,
    class_prior_assumption_metadata,
    prior_matched_teacher_soft_labels,
    prior_scale_aligned_teacher_soft_labels,
    prob_mean_teacher_soft_labels,
    require_prior_condition_supported,
    scale_aligned_teacher_soft_labels,
    skew_class_prior,
    teacher_soft_labels,
    teacher_target_diagnostics,
    uniform_budget_teacher_soft_labels,
    uniform_class_prior,
)
from src.student.train import build_examples, evaluate, fit, set_seed

ROUTER_WEIGHT_DECAY_GRID = [0.0, 1e-2, 1e-1]
ROUTER_DROPOUT = 0.3
LABEL_ACCOUNTING_TAG = "prompt-inclusive-v1"
DEFAULT_ROUTER_CAL_RATIO = 0.3


def _format_queries_tag(queries: float) -> str:
    """Encode a query budget for filenames, e.g. 1.4 -> q1p4."""
    text = f"{queries:g}".replace(".", "p")
    return f"q{text}"


def _calibration_ratio_tag(cal_ratio: float) -> str:
    """Return a filename suffix for non-default router calibration splits."""
    if np.isclose(cal_ratio, DEFAULT_ROUTER_CAL_RATIO):
        return ""
    return f"_cal{cal_ratio * 100:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a student model (Stage 4).")
    parser.add_argument("--dataset", required=True, choices=["amazon", "amazon3", "yahoo", "ag_news"])
    parser.add_argument("--mode", required=True, choices=["supervised", "kd"])
    parser.add_argument(
        "--teacher",
        default="qwen3.5-27b-vllm-guided",
        help=(
            "Teacher report name, multi-mean-guided, multi-entropy-guided, "
            "multi-probmean-guided, multi-prior-scale-aligned-guided, "
            "multi-prior-matched-guided, multi-scale-aligned-guided, "
            "qwen-prior-matched-guided, multi-cascade-guided, "
            "multi-uniform-budget-guided, multi-depth-matched-random-guided, "
            "oracle-guided, "
            "multi-router-linear-guided, or multi-router-mlp-guided."
        ),
    )
    parser.add_argument("--labels-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.5, help="KD/CE mix (ignored for supervised).")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", default="outputs/student")
    parser.add_argument(
        "--oracle-temperature",
        type=float,
        default=1.0,
        help="Ground-truth gating temperature for oracle-guided (default 1.0).",
    )
    parser.add_argument(
        "--cascade-target-queries",
        type=float,
        default=1.4,
        help="Mean teacher queries/sample for multi-cascade-guided (default 1.4).",
    )
    parser.add_argument(
        "--cascade-order",
        default="qwen-first",
        choices=["qwen-first", "gemma-first", "mistral-first"],
        help="Which teacher is queried on every sample for multi-cascade-guided "
        "(default qwen-first; order-sensitivity ablation).",
    )
    parser.add_argument(
        "--uniform-target-queries",
        type=float,
        default=1.4,
        help="Mean teacher queries/sample for multi-uniform-budget-guided (default 1.4).",
    )
    parser.add_argument(
        "--depth-matched-target-queries",
        type=float,
        default=1.4,
        help=(
            "Cascade-matched mean teacher queries/sample for "
            "multi-depth-matched-random-guided (default 1.4)."
        ),
    )
    parser.add_argument(
        "--allocation-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for random allocation policies. "
            "Defaults to --seed when omitted."
        ),
    )
    parser.add_argument(
        "--router-feature-groups",
        default="all",
        choices=["all", "aggregate"],
        help="Disagreement feature subset for router training (default: all 11).",
    )
    parser.add_argument(
        "--prior-condition",
        default="correct",
        choices=["correct", "mild", "strong"],
        help=(
            "Assumed class prior supplied to prior-and-scale-aligned matching. "
            "'correct' is the true uniform working-pool prior (default, backward "
            "compatible). 'mild'/'strong' apply the Table A.12 skew construction "
            "(strength 0.15/0.50) toward --prior-dominant-class; the actual "
            "working pool and test set stay balanced and unchanged."
        ),
    )
    parser.add_argument(
        "--prior-dominant-class",
        type=int,
        default=0,
        help="Class index that receives the excess prior mass for mild/strong.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and validate the assumed-prior soft targets, print "
            "diagnostics, then exit before loading or training ModernBERT. "
            "Writes no result file and makes no teacher queries."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.cal_ratio < 1.0:
        parser.error("--cal-ratio must be strictly between 0 and 1")
    try:
        require_prior_condition_supported(args.mode, args.teacher, args.prior_condition)
    except ValueError as exc:
        parser.error(str(exc))

    config = StudentConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        temperature=args.temperature,
        alpha=0.0 if args.mode == "supervised" else args.alpha,
    )

    set_seed(args.seed)
    device = select_device()
    teacher_name = args.teacher if args.mode == "kd" else "none"
    print(f"Mode: {args.mode} | teacher: {teacher_name} | device: {device}")

    dataset = load_dataset_by_name(args.dataset)
    train_identity = dataset.metadata.get("train_identity") or split_identity(
        dataset.train, dataset.text_column, dataset.label_column
    )
    test_identity = dataset.metadata.get("test_identity") or split_identity(
        dataset.test, dataset.text_column, dataset.label_column
    )
    target_prior = uniform_class_prior(dataset.num_classes)
    if args.prior_condition == "correct":
        assumed_prior = target_prior
        prior_strength = None
    elif args.prior_condition == "mild":
        prior_strength = 0.15
        assumed_prior = skew_class_prior(
            dataset.num_classes, prior_strength, args.prior_dominant_class
        )
    else:
        prior_strength = 0.50
        assumed_prior = skew_class_prior(
            dataset.num_classes, prior_strength, args.prior_dominant_class
        )
    calibration_tag = _calibration_ratio_tag(args.cal_ratio)
    prior_tag = (
        f"_prior-{args.prior_condition}-d{args.prior_dominant_class}"
        if args.prior_condition != "correct"
        else ""
    )
    splits = subsample_low_label(
        dataset, args.labels_per_class, args.seed, args.cal_ratio
    )
    reserve_calibration = args.mode == "kd" and args.teacher in (
        MULTI_ROUTER_LINEAR_TEACHER,
        MULTI_ROUTER_MLP_TEACHER,
    )
    cal_per_class = len(splits.labeled_cal_indices) // dataset.num_classes
    ce_per_class = (
        args.labels_per_class - cal_per_class
        if reserve_calibration
        else args.labels_per_class
    )
    print(
        f"Dataset: {dataset.name} ({dataset.num_classes} classes) | "
        f"declared_labeled={len(splits.labeled_indices)} "
        f"prompt_demos/class={PROMPT_DEMOS_PER_CLASS} CE/class={ce_per_class} "
        f"router_cal/class={cal_per_class if reserve_calibration else 0} "
        f"unlabeled={len(splits.unlabeled)} test={len(dataset.test)}"
    )

    soft_probs = None
    aggregation = None
    teachers_used: list[str] | None = None
    router_type = None
    loocv_accuracy = None
    router_meta: dict | None = None
    queries_per_sample = None
    assumed_class_prior: list[float] | None = None
    budget_policy: dict | None = None
    allocation_seed: int | None = None
    if args.mode == "kd":
        if args.teacher == MULTI_MEAN_TEACHER:
            soft_probs = average_teacher_soft_labels(
                GUIDED_TEACHERS, args.dataset, dataset.num_classes, config.temperature
            )
            aggregation = "logit_mean"
            teachers_used = list(GUIDED_TEACHERS)
        elif args.teacher == MULTI_ENTROPY_TEACHER:
            soft_probs = entropy_weighted_teacher_soft_labels(
                GUIDED_TEACHERS, args.dataset, dataset.num_classes, config.temperature
            )
            aggregation = "entropy_weighted"
            teachers_used = list(GUIDED_TEACHERS)
        elif args.teacher == MULTI_PROBMEAN_TEACHER:
            soft_probs = prob_mean_teacher_soft_labels(
                GUIDED_TEACHERS, args.dataset, dataset.num_classes, config.temperature
            )
            aggregation = "prob_mean"
            teachers_used = list(GUIDED_TEACHERS)
        elif args.teacher in (
            MULTI_PRIOR_SCALE_ALIGNED_TEACHER,
            MULTI_CALIBRATED_TEACHER,
        ):
            soft_probs = prior_scale_aligned_teacher_soft_labels(
                GUIDED_TEACHERS,
                args.dataset,
                dataset.num_classes,
                config.temperature,
                assumed_prior,
            )
            aggregation = "prior_matched_scale_aligned_prob_mean"
            teachers_used = list(GUIDED_TEACHERS)
            assumed_class_prior = assumed_prior.tolist()
        elif args.teacher in (MULTI_PRIOR_MATCHED_TEACHER, MULTI_MARGINAL_TEACHER):
            soft_probs = prior_matched_teacher_soft_labels(
                GUIDED_TEACHERS,
                args.dataset,
                dataset.num_classes,
                config.temperature,
                target_prior,
            )
            aggregation = "prior_matched_prob_mean"
            teachers_used = list(GUIDED_TEACHERS)
            assumed_class_prior = target_prior.tolist()
        elif args.teacher in (MULTI_SCALE_ALIGNED_TEACHER, MULTI_ENTROPYEQ_TEACHER):
            soft_probs = scale_aligned_teacher_soft_labels(
                GUIDED_TEACHERS, args.dataset, dataset.num_classes, config.temperature
            )
            aggregation = "confidence_scale_aligned_prob_mean"
            teachers_used = list(GUIDED_TEACHERS)
        elif args.teacher in (QWEN_PRIOR_MATCHED_TEACHER, QWEN_CALIBRATED_TEACHER):
            soft_probs = prior_matched_single_teacher_soft_labels(
                GUIDED_TEACHERS[0],
                args.dataset,
                dataset.num_classes,
                config.temperature,
                target_prior,
            )
            aggregation = "prior_matched_single"
            teachers_used = [GUIDED_TEACHERS[0]]
            assumed_class_prior = target_prior.tolist()
        elif args.teacher == MULTI_CASCADE_TEACHER:
            cascade_order = {
                "qwen-first": (GUIDED_TEACHERS[0], GUIDED_TEACHERS[1], GUIDED_TEACHERS[2]),
                "gemma-first": (GUIDED_TEACHERS[1], GUIDED_TEACHERS[0], GUIDED_TEACHERS[2]),
                "mistral-first": (GUIDED_TEACHERS[2], GUIDED_TEACHERS[0], GUIDED_TEACHERS[1]),
            }[args.cascade_order]
            soft_probs, queries_per_sample = cascade_teacher_soft_labels(
                cascade_order,
                args.dataset,
                dataset.num_classes,
                config.temperature,
                target_queries=args.cascade_target_queries,
                target_prior=target_prior,
            )
            aggregation = "cascade_prob_mean"
            teachers_used = list(cascade_order)
            assumed_class_prior = target_prior.tolist()
            print(f"Cascade realised queries/sample: {queries_per_sample:.3f}")
        elif args.teacher == MULTI_UNIFORM_BUDGET_TEACHER:
            allocation_seed = (
                args.allocation_seed if args.allocation_seed is not None else args.seed
            )
            soft_probs, budget_policy = uniform_budget_teacher_soft_labels(
                GUIDED_TEACHERS,
                args.dataset,
                dataset.num_classes,
                config.temperature,
                target_queries=args.uniform_target_queries,
                allocation_seed=allocation_seed,
                target_prior=target_prior,
            )
            aggregation = "uniform_budget_prior_matched_scale_aligned_prob_mean"
            teachers_used = list(GUIDED_TEACHERS)
            assumed_class_prior = target_prior.tolist()
            queries_per_sample = budget_policy["realised_queries_per_sample"]
            print(
                f"Uniform-budget realised queries/sample: {queries_per_sample:.3f} "
                f"(selected={budget_policy['selected_count']}, "
                f"allocation_seed={allocation_seed})"
            )
        elif args.teacher == MULTI_DEPTH_MATCHED_RANDOM_TEACHER:
            allocation_seed = (
                args.allocation_seed if args.allocation_seed is not None else args.seed
            )
            soft_probs, budget_policy = depth_matched_random_teacher_soft_labels(
                GUIDED_TEACHERS,
                args.dataset,
                dataset.num_classes,
                config.temperature,
                target_queries=args.depth_matched_target_queries,
                allocation_seed=allocation_seed,
                target_prior=target_prior,
            )
            aggregation = (
                "depth_matched_random_prior_matched_scale_aligned_prob_mean"
            )
            teachers_used = list(GUIDED_TEACHERS)
            assumed_class_prior = target_prior.tolist()
            queries_per_sample = budget_policy["realised_queries_per_sample"]
            print(
                f"Depth-matched random realised queries/sample: "
                f"{queries_per_sample:.3f} "
                f"(depths={budget_policy['depth_1_count']}/"
                f"{budget_policy['depth_2_count']}/"
                f"{budget_policy['depth_3_count']}, "
                f"allocation_seed={allocation_seed})"
            )
        elif args.teacher == MULTI_ORACLE_TEACHER:
            soft_probs = oracle_teacher_soft_labels(
                GUIDED_TEACHERS,
                dataset,
                dataset.num_classes,
                config.temperature,
                oracle_temperature=args.oracle_temperature,
            )
            aggregation = "oracle"
            teachers_used = list(GUIDED_TEACHERS)
        elif args.teacher in (MULTI_ROUTER_LINEAR_TEACHER, MULTI_ROUTER_MLP_TEACHER):
            router_type = "linear" if args.teacher == MULTI_ROUTER_LINEAR_TEACHER else "mlp"

            phi_all = compute_features_for_dataset(GUIDED_TEACHERS, args.dataset, dataset.num_classes)
            if args.router_feature_groups == "aggregate":
                agg_idx = FEATURE_GROUPS["aggregate"]
                phi_all = phi_all[:, agg_idx]
                feature_groups_used = ["aggregate"]
            else:
                feature_groups_used = list(FEATURE_GROUPS.keys())

            phi_all, phi_mean, phi_std = standardize_features(phi_all)
            stacked_logits = load_multi_teacher_logits(GUIDED_TEACHERS, args.dataset, dataset.num_classes)
            logits_all = np.transpose(stacked_logits, (1, 0, 2))  # (N, K, C) for router math
            log_prior_all = entropy_prior_log_weights(logits_all)

            cal_idx = splits.labeled_cal_indices
            phi_cal = phi_all[cal_idx]
            logits_cal = logits_all[cal_idx]
            log_prior_cal = log_prior_all[cal_idx]
            labels_cal = np.asarray(
                dataset.train.select(cal_idx.tolist())[dataset.label_column], dtype=np.int64
            )

            loocv_table = [
                loocv_router_score(
                    phi_cal, logits_cal, labels_cal, log_prior_cal, rt, config.temperature,
                    weight_decay=wd, dropout=ROUTER_DROPOUT, seed=args.seed,
                )
                for rt in ("linear", "mlp")
                for wd in ROUTER_WEIGHT_DECAY_GRID
            ]
            router_dir = Path("outputs/router")
            router_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if args.router_feature_groups == "all" else f"_{args.router_feature_groups}"
            loocv_path = router_dir / (
                f"{args.dataset}_L{args.labels_per_class}_s{args.seed}_{LABEL_ACCOUNTING_TAG}"
                f"{calibration_tag}{suffix}_loocv.json"
            )
            loocv_path.write_text(json.dumps(loocv_table, indent=2) + "\n", encoding="utf-8")
            print(f"Saved LOOCV table ({len(loocv_table)} configs): {loocv_path}")

            best = max(
                (s for s in loocv_table if s["router_type"] == router_type),
                key=lambda s: s["accuracy"],
            )
            loocv_accuracy = best["accuracy"]
            print(
                f"Selected {router_type} router via LOOCV: "
                f"weight_decay={best['weight_decay']}, loocv_accuracy={loocv_accuracy:.4f}"
            )

            gate = select_router_or_fallback(
                loocv_accuracy, logits_cal, labels_cal, config.temperature
            )
            print(
                f"Safety gate: mean_cal={gate['mean_cal_accuracy']:.4f}, "
                f"entropy_cal={gate['entropy_cal_accuracy']:.4f}, "
                f"use_router={gate['use_router']}"
            )

            if gate["use_router"]:
                final_router = train_router(
                    phi_cal, logits_cal, labels_cal, log_prior_cal,
                    router_type, config.temperature,
                    weight_decay=best["weight_decay"], dropout=ROUTER_DROPOUT, seed=args.seed,
                )
                soft_probs = purified_soft_labels(
                    final_router, phi_all, logits_all, log_prior_all, config.temperature
                )
                aggregation = f"router_{router_type}"
            elif gate["fallback_to"] == "mean":
                soft_probs = average_teacher_soft_labels(
                    GUIDED_TEACHERS, args.dataset, dataset.num_classes, config.temperature
                )
                aggregation = "logit_mean_fallback"
            else:
                soft_probs = entropy_weighted_teacher_soft_labels(
                    GUIDED_TEACHERS, args.dataset, dataset.num_classes, config.temperature
                )
                aggregation = "entropy_weighted_fallback"

            teachers_used = list(GUIDED_TEACHERS)
            router_meta = {
                "router_version": "v2",
                "router_used": gate["use_router"],
                "fallback_to": gate["fallback_to"],
                "heuristic_cal_accuracies": {
                    "mean": gate["mean_cal_accuracy"],
                    "entropy": gate["entropy_cal_accuracy"],
                },
                "feature_groups_used": feature_groups_used,
                "phi_standardization": {"mean": phi_mean.tolist(), "std": phi_std.tolist()},
                "selected_weight_decay": best["weight_decay"],
            }
        else:
            soft_probs = teacher_soft_labels(
                args.teacher, args.dataset, dataset.num_classes, config.temperature
            )

    if args.dry_run:
        print("\n=== DRY RUN: validating assumed-prior soft targets, no training ===")
        print(f"True working-pool prior ({dataset.num_classes} classes): {target_prior.tolist()}")
        if assumed_class_prior is not None:
            print(
                f"Assumed prior (condition={args.prior_condition}, "
                f"dominant_class={args.prior_dominant_class}): {assumed_class_prior}"
            )
            expected_assumed_prior = assumed_prior
            if not np.allclose(
                np.asarray(assumed_class_prior, dtype=np.float64),
                expected_assumed_prior,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "assumed prior does not match the registered "
                    f"{args.prior_condition} condition"
                )
        if soft_probs is not None:
            expected_shape = (len(dataset.train), dataset.num_classes)
            if soft_probs.shape != expected_shape:
                raise RuntimeError(
                    f"soft target shape {soft_probs.shape} != {expected_shape}"
                )
            if not np.isfinite(soft_probs).all():
                raise RuntimeError("soft targets contain non-finite values")
            if np.any(soft_probs < 0):
                raise RuntimeError("soft targets contain negative values")
            row_sums = soft_probs.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-5):
                raise RuntimeError("soft target rows do not all sum to one")
            gold_labels = np.asarray(dataset.train[dataset.label_column], dtype=np.int64)
            diagnostics = teacher_target_diagnostics(soft_probs, gold_labels)
            print(
                f"Soft target shape {soft_probs.shape}: all finite, all "
                f"non-negative, rows sum to 1 (max |sum-1|="
                f"{float(np.max(np.abs(row_sums - 1.0))):.2e})"
            )
            print(f"Teacher-target diagnostics (gold labels used for evaluation only): {diagnostics}")
        print(
            f"Declared labeled={len(splits.labeled_indices)} "
            f"unlabeled={len(splits.unlabeled)} test={len(dataset.test)} "
            "(dataset selection unaffected by --dry-run)"
        )
        print("Dry run OK: exiting before loading or training ModernBERT.")
        return

    train_ds = build_examples(
        dataset,
        splits,
        args.mode,
        soft_probs,
        reserve_calibration=reserve_calibration,
    )
    print(f"Training on {len(train_ds)} examples (alpha={config.alpha}, tau={config.temperature})")

    model, tokenizer = build_student(config.model_name, dataset.num_classes)
    fit(model, tokenizer, train_ds, config, device)
    evaluation = evaluate(
        model,
        tokenizer,
        dataset,
        config,
        device,
        return_predictions=True,
    )
    test_labels = evaluation.pop("test_labels")
    test_predictions = evaluation.pop("test_predictions")
    metrics = evaluation

    print(f"\nTest accuracy: {metrics['accuracy']:.4f} | macro-F1: {metrics['macro_f1']:.4f}")

    result = {
        "dataset": args.dataset,
        "mode": args.mode,
        "teacher": teacher_name,
        "labels_per_class": args.labels_per_class,
        "seed": args.seed,
        "cal_ratio": args.cal_ratio,
        "label_accounting": {
            "policy": LABEL_ACCOUNTING_TAG,
            "declared_labels_per_class": args.labels_per_class,
            "prompt_demo_labels_per_class": PROMPT_DEMOS_PER_CLASS,
            "student_ce_labels_per_class": ce_per_class,
            "router_calibration_labels_per_class": cal_per_class if reserve_calibration else 0,
            "student_ce_fraction": (1.0 - args.cal_ratio) if reserve_calibration else 1.0,
            "router_calibration_fraction": args.cal_ratio if reserve_calibration else 0.0,
            "calibration_reserved": reserve_calibration,
            "prompt_demo_indices": splits.prompt_demo_indices.tolist(),
        },
        "config": vars(config),
        "metrics": metrics,
        "dataset_identity": {
            "artifact_version": dataset.metadata.get("artifact_version", "legacy"),
            "source": dataset.metadata.get("source"),
            "train": train_identity,
            "test": test_identity,
            "test_per_class": dataset.metadata.get("test_per_class"),
        },
        "paired_test_data": {
            "sample_indices": list(range(len(test_labels))),
            "labels": test_labels,
            "predictions": test_predictions,
        },
    }
    if aggregation is not None:
        result["aggregation"] = aggregation
        result["teachers"] = teachers_used
    if assumed_class_prior is not None:
        result["class_prior_assumption"] = class_prior_assumption_metadata(
            args.prior_condition,
            np.asarray(assumed_class_prior, dtype=np.float64),
            target_prior,
            args.prior_dominant_class,
            prior_strength,
        )
        if args.prior_condition != "correct":
            gold_labels = np.asarray(
                dataset.train[dataset.label_column], dtype=np.int64
            )
            result["teacher_target_diagnostics"] = teacher_target_diagnostics(
                soft_probs, gold_labels
            )
    if router_type is not None:
        result["router_type"] = router_type
        result["loocv_accuracy"] = loocv_accuracy
        if router_meta is not None:
            result.update(router_meta)
    if args.mode == "kd" and args.teacher == MULTI_ORACLE_TEACHER:
        result["oracle_temperature"] = args.oracle_temperature
    if queries_per_sample is not None:
        result["queries_per_sample"] = queries_per_sample
    if budget_policy is not None:
        result["budget_policy"] = budget_policy
    if args.teacher == MULTI_CASCADE_TEACHER and args.cascade_order != "qwen-first":
        result["cascade_order"] = args.cascade_order
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order_suffix = (
        f"_ord-{args.cascade_order}"
        if args.teacher == MULTI_CASCADE_TEACHER and args.cascade_order != "qwen-first"
        else ""
    )
    budget_tag = ""
    allocation_tag = ""
    if args.teacher == MULTI_CASCADE_TEACHER:
        budget_tag = f"_{_format_queries_tag(args.cascade_target_queries)}"
    elif args.teacher == MULTI_UNIFORM_BUDGET_TEACHER:
        assert allocation_seed is not None
        budget_tag = f"_{_format_queries_tag(args.uniform_target_queries)}"
        allocation_tag = f"_a{allocation_seed}"
    elif args.teacher == MULTI_DEPTH_MATCHED_RANDOM_TEACHER:
        assert allocation_seed is not None
        budget_tag = f"_{_format_queries_tag(args.depth_matched_target_queries)}"
        allocation_tag = f"_a{allocation_seed}"
    out_path = out_dir / (
        f"{args.dataset}_{args.mode}_{teacher_name}{order_suffix}{budget_tag}"
        f"{prior_tag}_L{args.labels_per_class}_s{args.seed}{allocation_tag}_"
        f"{LABEL_ACCOUNTING_TAG}{calibration_tag}.json"
    )
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
