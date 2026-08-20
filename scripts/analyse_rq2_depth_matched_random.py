#!/usr/bin/env python
"""Analyse cascade versus depth-matched random RQ2 student results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student.soft_labels import (
    GUIDED_TEACHERS,
    MULTI_CASCADE_TEACHER,
    MULTI_DEPTH_MATCHED_RANDOM_TEACHER,
    REPORT_DIR,
    _cascade_allocation_masks,
    _cascade_teacher_probabilities,
    uniform_class_prior,
)

DATASETS = ("ag_news", "yahoo", "amazon", "amazon3")
LABEL_BUDGETS = (10, 50)
SEEDS = (42, 123, 456)
QUERY_RATES = (1.4, 2.0)
NUM_CLASSES = {"ag_news": 4, "yahoo": 10, "amazon": 5, "amazon3": 3}
EXPECTED_TRAIN_SIZE = {
    "ag_news": 6000,
    "yahoo": 5000,
    "amazon": 2500,
    "amazon3": 1500,
}
EXPECTED_TEST_SIZE = {
    "ag_news": 7600,
    "yahoo": 10000,
    "amazon": 10000,
    "amazon3": 6000,
}
REQUIRED_CONFIG = {
    "model_name": "answerdotai/ModernBERT-base",
    "max_length": 256,
    "lr": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "temperature": 2.0,
    "alpha": 0.5,
}


def query_tag(target_queries: float) -> str:
    return f"q{target_queries:g}".replace(".", "p")


def input_dir_for(target_queries: float) -> Path:
    if target_queries == 1.4:
        return Path("outputs/student/rq2_matched_q1p4_v1")
    if target_queries == 2.0:
        return Path("outputs/student/rq2_matched_q2_3seed_v1")
    raise ValueError(f"unsupported analysis query rate: {target_queries}")


def result_path(
    teacher: str,
    dataset: str,
    labels_per_class: int,
    seed: int,
    target_queries: float,
) -> Path:
    allocation_tag = (
        f"_a{seed}" if teacher == MULTI_DEPTH_MATCHED_RANDOM_TEACHER else ""
    )
    name = (
        f"{dataset}_kd_{teacher}_{query_tag(target_queries)}_"
        f"L{labels_per_class}_s{seed}{allocation_tag}_prompt-inclusive-v1.json"
    )
    return input_dir_for(target_queries) / name


@lru_cache(maxsize=None)
def source_depth_policy(dataset: str, target_queries: float) -> dict:
    num_classes = NUM_CLASSES[dataset]
    probabilities = _cascade_teacher_probabilities(
        GUIDED_TEACHERS,
        dataset,
        num_classes,
        REPORT_DIR,
        uniform_class_prior(num_classes),
    )
    ask2, ask3, realised = _cascade_allocation_masks(
        probabilities, target_queries
    )
    n_samples = int(probabilities.shape[1])
    n1 = int((~ask2).sum())
    n2 = int((ask2 & ~ask3).sum())
    n3 = int(ask3.sum())
    return {
        "n_samples": n_samples,
        "depth_1_count": n1,
        "depth_2_count": n2,
        "depth_3_count": n3,
        "realised_queries_per_sample": float(realised),
    }


def _finite_unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_random_cell(
    data: dict,
    *,
    dataset: str,
    labels_per_class: int,
    seed: int,
    target_queries: float,
) -> list[str]:
    errors: list[str] = []
    expected = source_depth_policy(dataset, target_queries)
    basic_expected = {
        "dataset": dataset,
        "mode": "kd",
        "teacher": MULTI_DEPTH_MATCHED_RANDOM_TEACHER,
        "labels_per_class": labels_per_class,
        "seed": seed,
        "aggregation": (
            "depth_matched_random_prior_matched_scale_aligned_prob_mean"
        ),
    }
    for key, value in basic_expected.items():
        if data.get(key) != value:
            errors.append(f"{key}={data.get(key)!r} != {value!r}")
    if data.get("label_accounting", {}).get("policy") != "prompt-inclusive-v1":
        errors.append("label_accounting.policy != prompt-inclusive-v1")
    if data.get("dataset_identity", {}).get("artifact_version") != "large-test-v1":
        errors.append("dataset_identity.artifact_version != large-test-v1")
    if data.get("teachers") != list(GUIDED_TEACHERS):
        errors.append("teacher order does not match Qwen, Gemma, Mistral")

    for key, value in REQUIRED_CONFIG.items():
        if data.get("config", {}).get(key) != value:
            errors.append(f"config.{key}={data.get('config', {}).get(key)!r} != {value!r}")
    for key in ("accuracy", "macro_f1"):
        if not _finite_unit_interval(data.get("metrics", {}).get(key)):
            errors.append(f"invalid metric: {key}")

    paired = data.get("paired_test_data", {})
    for key in ("sample_indices", "labels", "predictions"):
        if len(paired.get(key, [])) != EXPECTED_TEST_SIZE[dataset]:
            errors.append(f"invalid paired_test_data.{key} length")

    policy = data.get("budget_policy", {})
    policy_expected = {
        "policy": "depth_matched_random",
        "source_policy": "entropy_cascade",
        "target_queries_per_sample": float(target_queries),
        "allocation_seed": seed,
        "n_samples": EXPECTED_TRAIN_SIZE[dataset],
        "primary_teacher": GUIDED_TEACHERS[0],
        "teachers": list(GUIDED_TEACHERS),
    }
    for key, value in policy_expected.items():
        if policy.get(key) != value:
            errors.append(f"budget_policy.{key}={policy.get(key)!r} != {value!r}")
    for key in (
        "n_samples",
        "depth_1_count",
        "depth_2_count",
        "depth_3_count",
    ):
        if policy.get(key) != expected[key]:
            errors.append(
                f"budget_policy.{key}={policy.get(key)!r} != source {expected[key]!r}"
            )

    n1 = policy.get("depth_1_count")
    n2 = policy.get("depth_2_count")
    n3 = policy.get("depth_3_count")
    n_samples = policy.get("n_samples")
    if all(isinstance(v, int) for v in (n1, n2, n3, n_samples)) and n_samples:
        if n1 + n2 + n3 != n_samples:
            errors.append("depth counts do not sum to n_samples")
        count_rate = (n1 + 2 * n2 + 3 * n3) / n_samples
        policy_rate = policy.get("realised_queries_per_sample")
        if not isinstance(policy_rate, (int, float)) or not np.isclose(
            float(policy_rate), count_rate, rtol=0.0, atol=1e-12
        ):
            errors.append("realised rate does not equal the depth-count rate")

    realised = policy.get("realised_queries_per_sample")
    source_realised = expected["realised_queries_per_sample"]
    if not isinstance(realised, (int, float)) or float(realised) != source_realised:
        errors.append(
            f"realised rate {realised!r} != cascade source {source_realised!r}"
        )
    if data.get("queries_per_sample") != realised:
        errors.append("top-level and policy realised rates differ")
    if isinstance(realised, (int, float)) and abs(float(realised) - target_queries) > 1e-3:
        errors.append("realised rate is outside requested-rate tolerance")

    fractions = (
        ("depth_1_fraction", n1),
        ("depth_2_fraction", n2),
        ("depth_3_fraction", n3),
    )
    if isinstance(n_samples, int) and n_samples:
        for key, count in fractions:
            value = policy.get(key)
            if not isinstance(count, int) or not isinstance(value, (int, float)):
                errors.append(f"invalid budget_policy.{key}")
            elif float(value) != count / n_samples:
                errors.append(f"budget_policy.{key} does not match its count")
    return errors


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cascade_cell(
    data: dict,
    *,
    dataset: str,
    labels_per_class: int,
    seed: int,
    target_queries: float,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "dataset": dataset,
        "mode": "kd",
        "teacher": MULTI_CASCADE_TEACHER,
        "labels_per_class": labels_per_class,
        "seed": seed,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            errors.append(f"{key} mismatch")
    for key, value in REQUIRED_CONFIG.items():
        if data.get("config", {}).get(key) != value:
            errors.append(f"config.{key} mismatch")
    source_rate = source_depth_policy(dataset, target_queries)[
        "realised_queries_per_sample"
    ]
    result_rate = data.get("queries_per_sample")
    if not isinstance(result_rate, (int, float)) or abs(
        float(result_rate) - target_queries
    ) > 1e-3:
        errors.append("stored cascade rate is outside requested-rate tolerance")
    if abs(source_rate - target_queries) > 1e-3:
        errors.append("recomputed cascade rate is outside requested-rate tolerance")
    return errors


def assert_pair_compatible(cascade: dict, random: dict) -> None:
    if (
        cascade["dataset_identity"]["test"].get("sha256")
        != random["dataset_identity"]["test"].get("sha256")
    ):
        raise ValueError("test SHA-256 mismatch")
    for key in ("sample_indices", "labels"):
        if cascade["paired_test_data"][key] != random["paired_test_data"][key]:
            raise ValueError(f"paired test {key} mismatch")
    for key in ("seed", "labels_per_class"):
        if cascade[key] != random[key]:
            raise ValueError(f"paired {key} mismatch")
    if cascade["config"] != random["config"]:
        raise ValueError("student hyperparameters mismatch")


def mcnemar_exact(
    labels: np.ndarray, pred_cascade: np.ndarray, pred_random: np.ndarray
) -> dict:
    correct_cascade = pred_cascade == labels
    correct_random = pred_random == labels
    only_cascade = int(np.sum(correct_cascade & ~correct_random))
    only_random = int(np.sum(~correct_cascade & correct_random))
    discordant = only_cascade + only_random
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(only_cascade, discordant, p=0.5).pvalue)
    )
    return {
        "only_cascade_correct": only_cascade,
        "only_depth_matched_random_correct": only_random,
        "n_discordant": discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "values": [float(value) for value in values],
    }


def analyse_group(dataset: str, labels_per_class: int, target_queries: float) -> dict:
    cascade_accuracy: list[float] = []
    random_accuracy: list[float] = []
    cascade_f1: list[float] = []
    random_f1: list[float] = []
    differences: list[float] = []
    stored_cascade_rates: list[float] = []
    random_rates: list[float] = []
    seed_rows: list[dict] = []

    for seed in SEEDS:
        cascade_path = result_path(
            MULTI_CASCADE_TEACHER, dataset, labels_per_class, seed, target_queries
        )
        random_path = result_path(
            MULTI_DEPTH_MATCHED_RANDOM_TEACHER,
            dataset,
            labels_per_class,
            seed,
            target_queries,
        )
        if not cascade_path.is_file() or not random_path.is_file():
            raise FileNotFoundError(f"missing pair: {cascade_path}, {random_path}")
        cascade = load_json(cascade_path)
        random = load_json(random_path)
        cascade_errors = validate_cascade_cell(
            cascade,
            dataset=dataset,
            labels_per_class=labels_per_class,
            seed=seed,
            target_queries=target_queries,
        )
        random_errors = validate_random_cell(
            random,
            dataset=dataset,
            labels_per_class=labels_per_class,
            seed=seed,
            target_queries=target_queries,
        )
        if cascade_errors or random_errors:
            raise ValueError("; ".join(cascade_errors + random_errors))
        assert_pair_compatible(cascade, random)

        c_acc = float(cascade["metrics"]["accuracy"])
        r_acc = float(random["metrics"]["accuracy"])
        c_f1 = float(cascade["metrics"]["macro_f1"])
        r_f1 = float(random["metrics"]["macro_f1"])
        difference = c_acc - r_acc
        labels = np.asarray(cascade["paired_test_data"]["labels"], dtype=np.int64)
        pred_c = np.asarray(cascade["paired_test_data"]["predictions"], dtype=np.int64)
        pred_r = np.asarray(random["paired_test_data"]["predictions"], dtype=np.int64)
        mcnemar = mcnemar_exact(labels, pred_c, pred_r)

        cascade_accuracy.append(c_acc)
        random_accuracy.append(r_acc)
        cascade_f1.append(c_f1)
        random_f1.append(r_f1)
        differences.append(difference)
        stored_cascade_rates.append(float(cascade["queries_per_sample"]))
        random_rates.append(float(random["queries_per_sample"]))
        seed_rows.append(
            {
                "seed": seed,
                "cascade_accuracy": c_acc,
                "depth_matched_random_accuracy": r_acc,
                "cascade_minus_random_accuracy": difference,
                "direction": (
                    "cascade"
                    if difference > 0
                    else "depth_matched_random"
                    if difference < 0
                    else "tie"
                ),
                "mcnemar": mcnemar,
            }
        )

    adjusted = holm_adjust([row["mcnemar"]["p_value"] for row in seed_rows])
    for row, value in zip(seed_rows, adjusted):
        row["mcnemar"]["p_value_holm"] = value

    all_positive = all(value > 0 for value in differences)
    all_negative = all(value < 0 for value in differences)
    all_significant = all(value < 0.05 for value in adjusted)
    robust_winner = (
        "cascade"
        if all_positive and all_significant
        else "depth_matched_random"
        if all_negative and all_significant
        else "none"
    )
    return {
        "dataset": dataset,
        "labels_per_class": labels_per_class,
        "target_queries_per_sample": target_queries,
        "source_depth_policy": source_depth_policy(dataset, target_queries),
        "query_rate_audit": {
            "requested_queries_per_sample": target_queries,
            "recomputed_cascade_queries_per_sample": source_depth_policy(
                dataset, target_queries
            )["realised_queries_per_sample"],
            "depth_matched_random_queries_per_sample": random_rates,
            "stored_historical_cascade_queries_per_sample": stored_cascade_rates,
            "random_matches_recomputed_cascade_exactly": all(
                value
                == source_depth_policy(dataset, target_queries)[
                    "realised_queries_per_sample"
                ]
                for value in random_rates
            ),
            "stored_cascade_matches_recomputed_exactly": all(
                value
                == source_depth_policy(dataset, target_queries)[
                    "realised_queries_per_sample"
                ]
                for value in stored_cascade_rates
            ),
            "all_rates_within_requested_tolerance": all(
                abs(value - target_queries) <= 1e-3
                for value in stored_cascade_rates + random_rates
            ),
        },
        "cascade": {
            "accuracy": stats(cascade_accuracy),
            "macro_f1": stats(cascade_f1),
        },
        "depth_matched_random": {
            "accuracy": stats(random_accuracy),
            "macro_f1": stats(random_f1),
        },
        "paired_cascade_minus_random_accuracy": stats(differences),
        "seed_directions": [row["direction"] for row in seed_rows],
        "maximum_holm_p": max(adjusted),
        "robust_winner": robust_winner,
        "seeds": seed_rows,
    }


def render_markdown(groups: list[dict]) -> str:
    lines = [
        "# RQ2 depth-matched random control",
        "",
        "Cascade versus random allocation with identical cascade-derived query-depth "
        "counts. Exact paired McNemar tests are Holm-adjusted across the three seeds "
        "within each dataset–budget–query-rate cell.",
        "",
        "| Dataset | L | q/N | Cascade accuracy | Random accuracy | Mean difference | "
        "Seed directions | Maximum Holm p | Robust winner |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for group in groups:
        cascade = group["cascade"]["accuracy"]["mean"]
        random = group["depth_matched_random"]["accuracy"]["mean"]
        difference = group["paired_cascade_minus_random_accuracy"]["mean"]
        directions = ", ".join(group["seed_directions"])
        lines.append(
            f"| {group['dataset']} | {group['labels_per_class']} | "
            f"{group['target_queries_per_sample']:g} | {100*cascade:.2f}% | "
            f"{100*random:.2f}% | {100*difference:+.2f} pp | {directions} | "
            f"{group['maximum_holm_p']:.4g} | {group['robust_winner']} |"
        )
    mismatches = [
        group
        for group in groups
        if not group["query_rate_audit"]["stored_cascade_matches_recomputed_exactly"]
    ]
    if mismatches:
        lines.extend(
            [
                "",
                "## Query-rate audit",
                "",
                "The new control exactly matches the cascade rule recomputed from the "
                "current tracked teacher logits in every cell. Pre-existing cascade "
                "JSONs differ from that recomputation by one query in the following "
                "cells, while remaining within the established ±0.001 tolerance:",
                "",
            ]
        )
        for group in mismatches:
            audit = group["query_rate_audit"]
            lines.append(
                f"- {group['dataset']}, L={group['labels_per_class']}, "
                f"q={group['target_queries_per_sample']:g}: stored "
                f"{audit['stored_historical_cascade_queries_per_sample'][0]:.7f}, "
                f"recomputed/random "
                f"{audit['recomputed_cascade_queries_per_sample']:.7f}"
            )
    return "\n".join(lines) + "\n"


def check_cell(args: argparse.Namespace) -> int:
    path = Path(args.check_cell)
    if not path.is_file() or path.stat().st_size == 0:
        print(f"missing or empty result: {path}", file=sys.stderr)
        return 1
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid JSON {path}: {exc}", file=sys.stderr)
        return 1
    errors = validate_random_cell(
        data,
        dataset=args.dataset,
        labels_per_class=args.labels_per_class,
        seed=args.seed,
        target_queries=args.target_queries,
    )
    cascade_path = result_path(
        MULTI_CASCADE_TEACHER,
        args.dataset,
        args.labels_per_class,
        args.seed,
        args.target_queries,
    )
    if not cascade_path.is_file():
        errors.append(f"missing paired cascade result: {cascade_path}")
    else:
        try:
            cascade = load_json(cascade_path)
            errors.extend(
                validate_cascade_cell(
                    cascade,
                    dataset=args.dataset,
                    labels_per_class=args.labels_per_class,
                    seed=args.seed,
                    target_queries=args.target_queries,
                )
            )
            assert_pair_compatible(cascade, data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid paired cascade result: {exc}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"VALID {path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-cell")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--labels-per-class", type=int, choices=LABEL_BUDGETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--target-queries", type=float, choices=QUERY_RATES)
    parser.add_argument("--report-dir", default="outputs/reports")
    args = parser.parse_args()

    if args.check_cell:
        required = (
            args.dataset,
            args.labels_per_class,
            args.seed,
            args.target_queries,
        )
        if any(value is None for value in required):
            parser.error(
                "--check-cell requires --dataset, --labels-per-class, --seed, "
                "and --target-queries"
            )
        raise SystemExit(check_cell(args))

    groups: list[dict] = []
    errors: list[str] = []
    for target_queries in QUERY_RATES:
        for dataset in DATASETS:
            for labels_per_class in LABEL_BUDGETS:
                try:
                    groups.append(
                        analyse_group(dataset, labels_per_class, target_queries)
                    )
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"{dataset} L={labels_per_class} q={target_queries}: {exc}"
                    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    payload = {
        "experiment": "rq2_depth_matched_random",
        "policies": [
            MULTI_CASCADE_TEACHER,
            MULTI_DEPTH_MATCHED_RANDOM_TEACHER,
        ],
        "query_rates": list(QUERY_RATES),
        "datasets": list(DATASETS),
        "label_budgets": list(LABEL_BUDGETS),
        "seeds": list(SEEDS),
        "n_random_result_cells": 48,
        "validation_status": "complete",
        "groups": groups,
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "rq2_depth_matched_random.json"
    markdown_path = report_dir / "rq2_depth_matched_random.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(groups), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
