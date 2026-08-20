#!/usr/bin/env python
"""Analyse matched-budget cascade vs uniform student results at a fixed q/N.

Read-only over ``outputs/student/rq2_matched_q1p4_v1/``. Writes:
  - outputs/reports/rq2_matched_students.json
  - outputs/reports/rq2_matched_students.md

Run:
    uv run python scripts/analyse_rq2_matched_students.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student.soft_labels import MULTI_CASCADE_TEACHER, MULTI_UNIFORM_BUDGET_TEACHER

DATASETS = ("ag_news", "yahoo", "amazon", "amazon3")
LABEL_BUDGETS = (10, 50)
SEEDS = (42, 123, 456, 789, 2024)

# Budget-dependent settings; overridden by --target-queries/--budget-tag/--seeds.
TARGET_QUERIES = 1.4
BUDGET_TAG = "q1p4"
EXPECTED_TEST_SIZE = {
    "ag_news": 7600,
    "yahoo": 10000,
    "amazon": 10000,
    "amazon3": 6000,
}
REQUIRED_HYPERPARAMS = {
    "model_name": "answerdotai/ModernBERT-base",
    "epochs": 3,
    "alpha": 0.5,
    "temperature": 2.0,
    "lr": 2e-5,
    "max_length": 256,
}


def result_path(
    output_dir: Path,
    dataset: str,
    teacher: str,
    labels_per_class: int,
    seed: int,
) -> Path:
    if teacher == MULTI_UNIFORM_BUDGET_TEACHER:
        name = (
            f"{dataset}_kd_{teacher}_{BUDGET_TAG}_L{labels_per_class}_s{seed}_a{seed}"
            f"_prompt-inclusive-v1.json"
        )
    else:
        name = (
            f"{dataset}_kd_{teacher}_{BUDGET_TAG}_L{labels_per_class}_s{seed}"
            f"_prompt-inclusive-v1.json"
        )
    return output_dir / name


def _finite_unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_cell(
    data: dict,
    *,
    dataset: str,
    teacher: str,
    labels_per_class: int,
    seed: int,
) -> list[str]:
    errors: list[str] = []
    if data.get("dataset") != dataset:
        errors.append(f"dataset mismatch: {data.get('dataset')}")
    if data.get("teacher") != teacher:
        errors.append(f"teacher mismatch: {data.get('teacher')}")
    if data.get("labels_per_class") != labels_per_class:
        errors.append(f"L mismatch: {data.get('labels_per_class')}")
    if data.get("seed") != seed:
        errors.append(f"seed mismatch: {data.get('seed')}")
    if data.get("label_accounting", {}).get("policy") != "prompt-inclusive-v1":
        errors.append("label_accounting.policy != prompt-inclusive-v1")
    if data.get("dataset_identity", {}).get("artifact_version") != "large-test-v1":
        errors.append("dataset_identity.artifact_version != large-test-v1")
    q = data.get("queries_per_sample")
    if not isinstance(q, (int, float)) or abs(float(q) - TARGET_QUERIES) > 1e-3:
        errors.append(f"queries_per_sample not ~{TARGET_QUERIES}: {q}")
    metrics = data.get("metrics", {})
    for key in ("accuracy", "macro_f1"):
        if not _finite_unit_interval(metrics.get(key)):
            errors.append(f"invalid metric {key}: {metrics.get(key)}")
    expected_n = EXPECTED_TEST_SIZE[dataset]
    paired = data.get("paired_test_data", {})
    for key in ("sample_indices", "labels", "predictions"):
        values = paired.get(key, [])
        if len(values) != expected_n:
            errors.append(f"paired_test_data.{key} length {len(values)} != {expected_n}")
    cfg = data.get("config", {})
    for key, expected in REQUIRED_HYPERPARAMS.items():
        if cfg.get(key) != expected:
            errors.append(f"config.{key}={cfg.get(key)!r} != {expected!r}")
    # Batch size may be reduced under OOM, but must be recorded and positive.
    batch_size = cfg.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        errors.append(f"invalid batch_size: {batch_size}")
    if teacher == MULTI_UNIFORM_BUDGET_TEACHER:
        policy = data.get("budget_policy", {})
        if policy.get("allocation_seed") != seed:
            errors.append(
                f"budget_policy.allocation_seed={policy.get('allocation_seed')} != {seed}"
            )
        rq = policy.get("realised_queries_per_sample")
        if not isinstance(rq, (int, float)) or abs(float(rq) - TARGET_QUERIES) > 1e-3:
            errors.append(f"budget_policy realised budget not ~{TARGET_QUERIES}: {rq}")
    return errors


def load_cell(
    output_dir: Path,
    dataset: str,
    teacher: str,
    labels_per_class: int,
    seed: int,
) -> dict:
    path = result_path(output_dir, dataset, teacher, labels_per_class, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing result cell: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_cell(
        data,
        dataset=dataset,
        teacher=teacher,
        labels_per_class=labels_per_class,
        seed=seed,
    )
    if errors:
        raise ValueError(f"invalid cell {path}: {'; '.join(errors)}")
    return data


def assert_pair_compatible(cascade: dict, uniform: dict) -> None:
    c_test = cascade["dataset_identity"]["test"]
    u_test = uniform["dataset_identity"]["test"]
    if c_test.get("sha256") != u_test.get("sha256"):
        raise ValueError("paired cells have mismatched test SHA-256 identities")
    c_paired = cascade["paired_test_data"]
    u_paired = uniform["paired_test_data"]
    if c_paired["sample_indices"] != u_paired["sample_indices"]:
        raise ValueError("paired cells have mismatched sample_indices")
    if c_paired["labels"] != u_paired["labels"]:
        raise ValueError("paired cells have mismatched test labels")
    if cascade["labels_per_class"] != uniform["labels_per_class"]:
        raise ValueError("paired cells have mismatched label budgets")
    if cascade["seed"] != uniform["seed"]:
        raise ValueError("paired cells have mismatched seeds")
    for key, expected in REQUIRED_HYPERPARAMS.items():
        if cascade["config"].get(key) != expected or uniform["config"].get(key) != expected:
            raise ValueError(f"paired cells disagree on hyperparameter {key}")
    if cascade["config"].get("batch_size") != uniform["config"].get("batch_size"):
        raise ValueError("paired cells have mismatched batch_size")
    # Cascade bisection is discrete; both must be within tolerance of the target.
    if abs(float(cascade["queries_per_sample"]) - TARGET_QUERIES) > 1e-3:
        raise ValueError(f"cascade realised query budget not ~{TARGET_QUERIES}")
    if abs(float(uniform["queries_per_sample"]) - TARGET_QUERIES) > 1e-3:
        raise ValueError(f"uniform realised query budget not ~{TARGET_QUERIES}")


def mcnemar_exact(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test: A=cascade, B=uniform; H0: equal error rates."""
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n_discordant = only_a + only_b
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(only_a, n_discordant, p=0.5).pvalue)
    return {
        "only_cascade_correct": only_a,
        "only_uniform_correct": only_b,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in values],
    }


def analyse_group(
    output_dir: Path,
    dataset: str,
    labels_per_class: int,
) -> dict:
    cascade_acc: list[float] = []
    cascade_f1: list[float] = []
    uniform_acc: list[float] = []
    uniform_f1: list[float] = []
    paired_deltas_acc: list[float] = []
    paired_deltas_f1: list[float] = []
    seed_rows: list[dict] = []
    cascade_wins = uniform_wins = ties = 0

    for seed in SEEDS:
        cascade = load_cell(
            output_dir, dataset, MULTI_CASCADE_TEACHER, labels_per_class, seed
        )
        uniform = load_cell(
            output_dir, dataset, MULTI_UNIFORM_BUDGET_TEACHER, labels_per_class, seed
        )
        assert_pair_compatible(cascade, uniform)

        c_acc = float(cascade["metrics"]["accuracy"])
        u_acc = float(uniform["metrics"]["accuracy"])
        c_f1 = float(cascade["metrics"]["macro_f1"])
        u_f1 = float(uniform["metrics"]["macro_f1"])
        d_acc = c_acc - u_acc
        d_f1 = c_f1 - u_f1

        cascade_acc.append(c_acc)
        cascade_f1.append(c_f1)
        uniform_acc.append(u_acc)
        uniform_f1.append(u_f1)
        paired_deltas_acc.append(d_acc)
        paired_deltas_f1.append(d_f1)

        if abs(d_acc) < 1e-12:
            ties += 1
            winner = "tie"
        elif d_acc > 0:
            cascade_wins += 1
            winner = "cascade"
        else:
            uniform_wins += 1
            winner = "uniform"

        labels = np.asarray(cascade["paired_test_data"]["labels"], dtype=np.int64)
        pred_c = np.asarray(cascade["paired_test_data"]["predictions"], dtype=np.int64)
        pred_u = np.asarray(uniform["paired_test_data"]["predictions"], dtype=np.int64)
        mcnemar = mcnemar_exact(labels, pred_c, pred_u)

        seed_rows.append(
            {
                "seed": seed,
                "cascade_accuracy": c_acc,
                "uniform_accuracy": u_acc,
                "delta_accuracy": d_acc,
                "cascade_macro_f1": c_f1,
                "uniform_macro_f1": u_f1,
                "delta_macro_f1": d_f1,
                "accuracy_winner": winner,
                "queries_per_sample_cascade": float(cascade["queries_per_sample"]),
                "queries_per_sample_uniform": float(uniform["queries_per_sample"]),
                "mcnemar": mcnemar,
                "batch_size": cascade["config"]["batch_size"],
            }
        )

    delta_acc = mean_std(paired_deltas_acc)
    # Uncertainty on the mean paired delta: sample SE of the five seed deltas.
    delta_acc["se"] = (
        float(delta_acc["std"] / math.sqrt(len(paired_deltas_acc)))
        if paired_deltas_acc
        else 0.0
    )
    delta_f1 = mean_std(paired_deltas_f1)
    delta_f1["se"] = (
        float(delta_f1["std"] / math.sqrt(len(paired_deltas_f1)))
        if paired_deltas_f1
        else 0.0
    )

    # Holm correction over the five per-seed McNemar p-values in this cell.
    p_values = [row["mcnemar"]["p_value"] for row in seed_rows]
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx]
        holm = (m - rank) * raw
        running = max(running, holm)
        adjusted[idx] = min(1.0, running)
    for row, adj in zip(seed_rows, adjusted):
        row["mcnemar"]["p_value_holm"] = float(adj)

    return {
        "dataset": dataset,
        "labels_per_class": labels_per_class,
        "n_seeds": len(SEEDS),
        "validation_status": "ok",
        "cascade": {
            "accuracy": mean_std(cascade_acc),
            "macro_f1": mean_std(cascade_f1),
        },
        "uniform": {
            "accuracy": mean_std(uniform_acc),
            "macro_f1": mean_std(uniform_f1),
        },
        "paired_delta_cascade_minus_uniform": {
            "accuracy": delta_acc,
            "macro_f1": delta_f1,
        },
        "seed_wins": {
            "cascade": cascade_wins,
            "uniform": uniform_wins,
            "tie": ties,
        },
        "seeds": seed_rows,
    }


def fmt_mean_std(stat: dict, as_pct: bool = True) -> str:
    if as_pct:
        return f"{100 * stat['mean']:.2f}% ± {100 * stat['std']:.2f}"
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def render_markdown(groups: list[dict]) -> str:
    lines = [
        f"# RQ2 matched-budget student results (q/N = {TARGET_QUERIES})",
        "",
        "Cascade (`multi-cascade-guided`) versus uniform allocation "
        "(`multi-uniform-budget-guided`) at matched teacher-query budget.",
        "",
        "Paired McNemar tests are computed **per training seed** on the fixed "
        "`large-test-v1` test rows. Holm-adjusted p-values are reported within "
        "each dataset×L cell (5 seeds). With only five seeds, seed-level "
        "deltas are the primary evidence.",
        "",
    ]
    for group in groups:
        ds = group["dataset"]
        l = group["labels_per_class"]
        lines.append(f"## {ds} · L={l}")
        lines.append("")
        lines.append(
            f"- Cascade accuracy: {fmt_mean_std(group['cascade']['accuracy'])}"
        )
        lines.append(
            f"- Uniform accuracy: {fmt_mean_std(group['uniform']['accuracy'])}"
        )
        lines.append(
            f"- Cascade macro-F1: {fmt_mean_std(group['cascade']['macro_f1'])}"
        )
        lines.append(
            f"- Uniform macro-F1: {fmt_mean_std(group['uniform']['macro_f1'])}"
        )
        d_acc = group["paired_delta_cascade_minus_uniform"]["accuracy"]
        lines.append(
            f"- Mean paired Δacc (cascade−uniform): "
            f"{100 * d_acc['mean']:+.2f} ± {100 * d_acc['std']:.2f} "
            f"(SE {100 * d_acc['se']:.2f})"
        )
        wins = group["seed_wins"]
        lines.append(
            f"- Seed wins (accuracy): cascade={wins['cascade']}, "
            f"uniform={wins['uniform']}, tie={wins['tie']}"
        )
        lines.append(
            f"- Realised q/N: cascade≈"
            f"{group['seeds'][0]['queries_per_sample_cascade']:.4f}, "
            f"uniform="
            f"{group['seeds'][0]['queries_per_sample_uniform']:.4f}"
        )
        lines.append("")
        lines.append(
            "| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        for row in group["seeds"]:
            lines.append(
                f"| {row['seed']} | {100 * row['cascade_accuracy']:.2f}% | "
                f"{100 * row['uniform_accuracy']:.2f}% | "
                f"{100 * row['delta_accuracy']:+.2f} | "
                f"{row['mcnemar']['p_value']:.4g} | "
                f"{row['mcnemar']['p_value_holm']:.4g} | "
                f"{row['accuracy_winner']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    global TARGET_QUERIES, BUDGET_TAG, SEEDS
    parser = argparse.ArgumentParser(
        description="Analyse RQ2 matched-budget student results."
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/student/rq2_matched_q1p4_v1",
        help="Directory containing the 80 result JSON files.",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/reports",
        help="Directory for JSON/Markdown analysis outputs.",
    )
    parser.add_argument(
        "--target-queries",
        type=float,
        default=TARGET_QUERIES,
        help="Matched query budget q/N these cells realise (default 1.4).",
    )
    parser.add_argument(
        "--budget-tag",
        default=BUDGET_TAG,
        help="Filename budget tag, e.g. q1p4 or q2 (default q1p4).",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated training seeds (default 42,123,456,789,2024).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment name recorded in the JSON (default rq2_matched_<tag>_v1).",
    )
    parser.add_argument(
        "--report-stem",
        default="rq2_matched_students",
        help="Basename for the generated .json/.md reports.",
    )
    args = parser.parse_args()

    TARGET_QUERIES = args.target_queries
    BUDGET_TAG = args.budget_tag
    SEEDS = tuple(int(x) for x in args.seeds.split(","))

    input_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    groups: list[dict] = []
    missing: list[str] = []
    for dataset in DATASETS:
        for labels_per_class in LABEL_BUDGETS:
            try:
                groups.append(analyse_group(input_dir, dataset, labels_per_class))
            except (FileNotFoundError, ValueError) as exc:
                missing.append(f"{dataset} L={labels_per_class}: {exc}")

    if missing:
        print("RQ2 matched-budget analysis incomplete:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)

    payload = {
        "experiment": args.experiment or f"rq2_matched_{BUDGET_TAG}_v1",
        "target_queries_per_sample": TARGET_QUERIES,
        "policies": [MULTI_CASCADE_TEACHER, MULTI_UNIFORM_BUDGET_TEACHER],
        "seeds": list(SEEDS),
        "label_budgets": list(LABEL_BUDGETS),
        "datasets": list(DATASETS),
        "n_cells_expected": 80,
        "groups": groups,
    }
    json_path = report_dir / f"{args.report_stem}.json"
    md_path = report_dir / f"{args.report_stem}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(groups), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
