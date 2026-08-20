#!/usr/bin/env python
"""Analyse prompt-inclusive core student results (3-seed mandatory grid).

Read-only over ``outputs/student/prompt_inclusive_core_3seed_v1/``. Writes:
  - outputs/reports/prompt_inclusive_core_3seed.json
  - outputs/reports/prompt_inclusive_core_3seed.md

Also exposes ``--check-cell`` for the resumable GPU runner.

Run:
    uv run python scripts/analyse_prompt_inclusive_core_students.py
    uv run python scripts/analyse_prompt_inclusive_core_students.py \\
        --check-cell PATH --dataset ag_news --mode kd \\
        --teacher multi-mean-guided --labels-per-class 10 --seed 42
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

DATASETS = ("ag_news", "yahoo", "amazon", "amazon3")
LABEL_BUDGETS = (10, 50)
SEEDS = (42, 123, 456)
EXPECTED_TEST_SIZE = {
    "ag_news": 7600,
    "yahoo": 10000,
    "amazon": 10000,
    "amazon3": 6000,
}
NUM_CLASSES = {
    "ag_news": 4,
    "yahoo": 10,
    "amazon": 5,
    "amazon3": 3,
}

# Canonical method keys used in reports / comparisons.
METHODS: dict[str, dict[str, str]] = {
    "supervised": {"mode": "supervised", "teacher": "none"},
    "raw_qwen": {"mode": "kd", "teacher": "qwen3.5-27b-vllm-guided"},
    "prior_matched_qwen": {"mode": "kd", "teacher": "qwen-prior-matched-guided"},
    "logit_mean": {"mode": "kd", "teacher": "multi-mean-guided"},
    "prob_mean": {"mode": "kd", "teacher": "multi-probmean-guided"},
    "prior_matched": {"mode": "kd", "teacher": "multi-prior-matched-guided"},
    "scale_aligned": {"mode": "kd", "teacher": "multi-scale-aligned-guided"},
    "combined": {"mode": "kd", "teacher": "multi-prior-scale-aligned-guided"},
}

METHOD_ORDER = tuple(METHODS.keys())

EXPECTED_AGGREGATION = {
    "qwen-prior-matched-guided": "prior_matched_single",
    "multi-mean-guided": "logit_mean",
    "multi-probmean-guided": "prob_mean",
    "multi-prior-matched-guided": "prior_matched_prob_mean",
    "multi-scale-aligned-guided": "confidence_scale_aligned_prob_mean",
    "multi-prior-scale-aligned-guided": "prior_matched_scale_aligned_prob_mean",
}

PRIOR_DEPENDENT_TEACHERS = frozenset(
    {
        "qwen-prior-matched-guided",
        "multi-prior-matched-guided",
        "multi-prior-scale-aligned-guided",
    }
)

REQUIRED_KD_HYPERPARAMS = {
    "model_name": "answerdotai/ModernBERT-base",
    "epochs": 3,
    "alpha": 0.5,
    "temperature": 2.0,
    "lr": 2e-5,
    "batch_size": 16,
    "max_length": 256,
}

REQUIRED_SUPERVISED_HYPERPARAMS = {
    **REQUIRED_KD_HYPERPARAMS,
    "alpha": 0.0,
}

# (comparison_id, method_a, method_b, description) — report A minus B.
COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("raw_qwen_minus_supervised", "raw_qwen", "supervised", "raw Qwen − supervised"),
    (
        "prior_matched_qwen_minus_raw_qwen",
        "prior_matched_qwen",
        "raw_qwen",
        "prior-matched Qwen − raw Qwen",
    ),
    ("logit_mean_minus_raw_qwen", "logit_mean", "raw_qwen", "logit mean − raw Qwen"),
    (
        "prob_mean_minus_logit_mean",
        "prob_mean",
        "logit_mean",
        "probability mean − logit mean",
    ),
    (
        "prior_matched_minus_prob_mean",
        "prior_matched",
        "prob_mean",
        "prior-matched mean − probability mean",
    ),
    (
        "scale_aligned_minus_prob_mean",
        "scale_aligned",
        "prob_mean",
        "scale-aligned mean − probability mean",
    ),
    (
        "combined_minus_prob_mean",
        "combined",
        "prob_mean",
        "combined alignment − probability mean",
    ),
    (
        "combined_minus_logit_mean",
        "combined",
        "logit_mean",
        "combined alignment − logit mean",
    ),
    (
        "combined_minus_prior_matched",
        "combined",
        "prior_matched",
        "combined alignment − prior-matched mean",
    ),
    (
        "combined_minus_prior_matched_qwen",
        "combined",
        "prior_matched_qwen",
        "combined alignment − prior-matched Qwen",
    ),
)

N_CELLS_EXPECTED = (
    len(DATASETS) * len(LABEL_BUDGETS) * len(SEEDS) * len(METHODS)
)


def result_path(
    output_dir: Path,
    dataset: str,
    mode: str,
    teacher: str,
    labels_per_class: int,
    seed: int,
) -> Path:
    name = (
        f"{dataset}_{mode}_{teacher}_L{labels_per_class}_s{seed}"
        f"_prompt-inclusive-v1.json"
    )
    return output_dir / name


def method_result_path(
    output_dir: Path,
    dataset: str,
    method_key: str,
    labels_per_class: int,
    seed: int,
) -> Path:
    spec = METHODS[method_key]
    return result_path(
        output_dir,
        dataset,
        spec["mode"],
        spec["teacher"],
        labels_per_class,
        seed,
    )


def _finite_unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _uniform_prior_ok(values: object, num_classes: int, atol: float = 1e-6) -> bool:
    if not isinstance(values, list) or len(values) != num_classes:
        return False
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if abs(float(arr.sum()) - 1.0) > atol:
        return False
    expected = 1.0 / num_classes
    return bool(np.allclose(arr, expected, atol=atol))


def validate_cell(
    data: dict,
    *,
    dataset: str,
    mode: str,
    teacher: str,
    labels_per_class: int,
    seed: int,
) -> list[str]:
    """Return a list of validation error strings (empty iff the cell is valid)."""
    errors: list[str] = []
    if data.get("dataset") != dataset:
        errors.append(f"dataset mismatch: {data.get('dataset')}")
    if data.get("mode") != mode:
        errors.append(f"mode mismatch: {data.get('mode')}")
    if data.get("teacher") != teacher:
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
        errors.append("dataset_identity.artifact_version != large-test-v1")

    metrics = data.get("metrics", {})
    for key in ("accuracy", "macro_f1"):
        if not _finite_unit_interval(metrics.get(key)):
            errors.append(f"invalid metric {key}: {metrics.get(key)}")

    expected_n = EXPECTED_TEST_SIZE[dataset]
    paired = data.get("paired_test_data", {})
    for key in ("sample_indices", "labels", "predictions"):
        values = paired.get(key, [])
        if len(values) != expected_n:
            errors.append(
                f"paired_test_data.{key} length {len(values)} != {expected_n}"
            )

    required = (
        REQUIRED_SUPERVISED_HYPERPARAMS if mode == "supervised" else REQUIRED_KD_HYPERPARAMS
    )
    cfg = data.get("config", {})
    for key, expected in required.items():
        if cfg.get(key) != expected:
            errors.append(f"config.{key}={cfg.get(key)!r} != {expected!r}")

    expected_agg = EXPECTED_AGGREGATION.get(teacher)
    if expected_agg is not None and data.get("aggregation") != expected_agg:
        errors.append(
            f"aggregation={data.get('aggregation')!r} != {expected_agg!r}"
        )

    if teacher in PRIOR_DEPENDENT_TEACHERS:
        prior = data.get("class_prior_assumption", {})
        if prior.get("source") != (
            "known_uniform_prior_from_gold-balanced_working_pool"
        ):
            errors.append("class_prior_assumption.source mismatch")
        if prior.get("uses_individual_labels_during_matching") is not False:
            errors.append(
                "class_prior_assumption.uses_individual_labels_during_matching "
                "must be false"
            )
        if not _uniform_prior_ok(prior.get("values"), NUM_CLASSES[dataset]):
            errors.append("class_prior_assumption.values not uniform of length C")

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
    for key in (
        "model_name",
        "epochs",
        "temperature",
        "lr",
        "batch_size",
        "max_length",
    ):
        if cell_a["config"].get(key) != cell_b["config"].get(key):
            raise ValueError(f"paired cells disagree on hyperparameter {key}")


def load_cell(
    output_dir: Path,
    dataset: str,
    mode: str,
    teacher: str,
    labels_per_class: int,
    seed: int,
) -> dict:
    path = result_path(output_dir, dataset, mode, teacher, labels_per_class, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing result cell: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_cell(
        data,
        dataset=dataset,
        mode=mode,
        teacher=teacher,
        labels_per_class=labels_per_class,
        seed=seed,
    )
    if errors:
        raise ValueError(f"invalid cell {path}: {'; '.join(errors)}")
    return data


def load_method_cell(
    output_dir: Path,
    dataset: str,
    method_key: str,
    labels_per_class: int,
    seed: int,
) -> dict:
    spec = METHODS[method_key]
    return load_cell(
        output_dir,
        dataset,
        spec["mode"],
        spec["teacher"],
        labels_per_class,
        seed,
    )


def mcnemar_exact(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test: A vs B; H0 equal error rates.

    ``only_a_correct`` / ``only_b_correct`` are the discordant counts.
    """
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
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment; preserves input order."""
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
    stats["se_note"] = "descriptive SE of seed-level paired deltas"
    return stats


def analyse_comparison(
    cells_a: dict[int, dict],
    cells_b: dict[int, dict],
    *,
    method_a: str,
    method_b: str,
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
            winner = method_a
        else:
            b_wins += 1
            winner = method_b

        labels = np.asarray(cell_a["paired_test_data"]["labels"], dtype=np.int64)
        pred_a = np.asarray(cell_a["paired_test_data"]["predictions"], dtype=np.int64)
        pred_b = np.asarray(cell_b["paired_test_data"]["predictions"], dtype=np.int64)
        mcnemar = mcnemar_exact(labels, pred_a, pred_b)

        seed_rows.append(
            {
                "seed": seed,
                f"{method_a}_accuracy": a_acc,
                f"{method_b}_accuracy": b_acc,
                "delta_accuracy": d_acc,
                f"{method_a}_macro_f1": a_f1,
                f"{method_b}_macro_f1": b_f1,
                "delta_macro_f1": d_f1,
                "accuracy_winner": winner,
                "mcnemar": mcnemar,
            }
        )

    p_values = [row["mcnemar"]["p_value"] for row in seed_rows]
    for row, adj in zip(seed_rows, holm_adjust(p_values)):
        row["mcnemar"]["p_value_holm"] = float(adj)

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_seeds": len(SEEDS),
        "paired_delta": {
            "accuracy": paired_delta_stats(deltas_acc),
            "macro_f1": paired_delta_stats(deltas_f1),
        },
        "seed_wins": {
            method_a: a_wins,
            method_b: b_wins,
            "tie": ties,
        },
        "seeds": seed_rows,
    }


def analyse_group(
    output_dir: Path,
    dataset: str,
    labels_per_class: int,
) -> dict:
    methods_payload: dict[str, dict] = {}
    cells_by_method: dict[str, dict[int, dict]] = {}

    for method_key in METHOD_ORDER:
        acc_values: list[float] = []
        f1_values: list[float] = []
        seed_metrics: list[dict] = []
        cells_by_method[method_key] = {}
        for seed in SEEDS:
            cell = load_method_cell(
                output_dir, dataset, method_key, labels_per_class, seed
            )
            cells_by_method[method_key][seed] = cell
            acc = float(cell["metrics"]["accuracy"])
            f1 = float(cell["metrics"]["macro_f1"])
            acc_values.append(acc)
            f1_values.append(f1)
            seed_metrics.append(
                {
                    "seed": seed,
                    "accuracy": acc,
                    "macro_f1": f1,
                    "teacher": cell["teacher"],
                    "mode": cell["mode"],
                }
            )
        # Within-method paired identity check across seeds is not required;
        # cross-method pairing is checked in comparisons below.
        methods_payload[method_key] = {
            "teacher": METHODS[method_key]["teacher"],
            "mode": METHODS[method_key]["mode"],
            "accuracy": mean_std(acc_values),
            "macro_f1": mean_std(f1_values),
            "seeds": seed_metrics,
        }

    comparisons: dict[str, dict] = {}
    for cmp_id, method_a, method_b, _desc in COMPARISONS:
        comparisons[cmp_id] = analyse_comparison(
            cells_by_method[method_a],
            cells_by_method[method_b],
            method_a=method_a,
            method_b=method_b,
        )
        comparisons[cmp_id]["description"] = _desc

    return {
        "dataset": dataset,
        "labels_per_class": labels_per_class,
        "n_seeds": len(SEEDS),
        "validation_status": "ok",
        "methods": methods_payload,
        "comparisons": comparisons,
    }


def fmt_mean_std(stat: dict, as_pct: bool = True) -> str:
    if as_pct:
        return f"{100 * stat['mean']:.2f}% ± {100 * stat['std']:.2f}"
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def fmt_seed_values(stat: dict, as_pct: bool = True) -> str:
    if as_pct:
        return ", ".join(f"{100 * v:.2f}%" for v in stat["values"])
    return ", ".join(f"{v:.4f}" for v in stat["values"])


def render_markdown(groups: list[dict]) -> str:
    lines = [
        "# Prompt-inclusive core student results (3-seed)",
        "",
        "Mandatory like-for-like grid under `prompt-inclusive-v1` label "
        "accounting and `large-test-v1` evaluation.",
        "",
        f"Seeds: {', '.join(str(s) for s in SEEDS)}. "
        "Report every seed value; mean ± sample SD is secondary.",
        "",
        "Paired McNemar tests are computed **per training seed** on the fixed "
        "test rows. Holm-adjusted p-values correct the three seed-level tests "
        "within each dataset × L × comparison cell. Seed-level SE of paired "
        "deltas is descriptive only.",
        "",
        "This is a three-seed pilot, not an established population-level claim.",
        "",
    ]

    for group in groups:
        ds = group["dataset"]
        lpc = group["labels_per_class"]
        lines.append(f"## {ds} · L={lpc}")
        lines.append("")
        lines.append("### Method summary (accuracy / macro-F1)")
        lines.append("")
        lines.append(
            "| method | teacher | acc mean±SD | acc seeds | "
            "F1 mean±SD | F1 seeds |"
        )
        lines.append("|---|---|---:|---|---:|---|")
        for method_key in METHOD_ORDER:
            m = group["methods"][method_key]
            lines.append(
                f"| {method_key} | `{m['teacher']}` | "
                f"{fmt_mean_std(m['accuracy'])} | "
                f"{fmt_seed_values(m['accuracy'])} | "
                f"{fmt_mean_std(m['macro_f1'])} | "
                f"{fmt_seed_values(m['macro_f1'])} |"
            )
        lines.append("")
        lines.append("### Pairwise comparisons (A − B)")
        lines.append("")
        for cmp_id, method_a, method_b, desc in COMPARISONS:
            cmp = group["comparisons"][cmp_id]
            d_acc = cmp["paired_delta"]["accuracy"]
            wins = cmp["seed_wins"]
            lines.append(f"#### {desc}")
            lines.append("")
            lines.append(
                f"- Mean paired Δacc: {100 * d_acc['mean']:+.2f} ± "
                f"{100 * d_acc['std']:.2f} "
                f"(descriptive SE {100 * d_acc['se']:.2f})"
            )
            lines.append(
                f"- Seed wins (accuracy): {method_a}={wins[method_a]}, "
                f"{method_b}={wins[method_b]}, tie={wins['tie']}"
            )
            lines.append("")
            lines.append(
                "| seed | A acc | B acc | Δacc | only_A | only_B | "
                "McNemar p | Holm p | winner |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in cmp["seeds"]:
                mc = row["mcnemar"]
                lines.append(
                    f"| {row['seed']} | "
                    f"{100 * row[f'{method_a}_accuracy']:.2f}% | "
                    f"{100 * row[f'{method_b}_accuracy']:.2f}% | "
                    f"{100 * row['delta_accuracy']:+.2f} | "
                    f"{mc['only_a_correct']} | {mc['only_b_correct']} | "
                    f"{mc['p_value']:.4g} | {mc['p_value_holm']:.4g} | "
                    f"{row['accuracy_winner']} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


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
        mode=args.mode,
        teacher=args.teacher,
        labels_per_class=args.labels_per_class,
        seed=args.seed,
    )
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse prompt-inclusive core 3-seed student results."
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/student/prompt_inclusive_core_3seed_v1",
        help="Directory containing the 192 result JSON files.",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/reports",
        help="Directory for JSON/Markdown analysis outputs.",
    )
    parser.add_argument(
        "--check-cell",
        default=None,
        help="Validate a single result JSON and exit 0/1 (for the GPU runner).",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", default=None, choices=["supervised", "kd"])
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--labels-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.check_cell is not None:
        required = ("dataset", "mode", "teacher", "labels_per_class", "seed")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            print(
                f"--check-cell requires: {', '.join('--' + m.replace('_', '-') for m in missing)}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise SystemExit(check_cell_cli(args))

    input_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    groups: list[dict] = []
    failures: list[str] = []
    for dataset in DATASETS:
        for labels_per_class in LABEL_BUDGETS:
            try:
                groups.append(analyse_group(input_dir, dataset, labels_per_class))
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"{dataset} L={labels_per_class}: {exc}")

    if failures:
        print(
            "Prompt-inclusive core analysis incomplete "
            f"(expected {N_CELLS_EXPECTED} cells):",
            file=sys.stderr,
        )
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)

    payload = {
        "experiment": "prompt_inclusive_core_3seed_v1",
        "label_accounting": "prompt-inclusive-v1",
        "artifact_version": "large-test-v1",
        "methods": {
            key: {"mode": spec["mode"], "teacher": spec["teacher"]}
            for key, spec in METHODS.items()
        },
        "comparisons": [
            {
                "id": cmp_id,
                "method_a": method_a,
                "method_b": method_b,
                "description": desc,
            }
            for cmp_id, method_a, method_b, desc in COMPARISONS
        ],
        "seeds": list(SEEDS),
        "label_budgets": list(LABEL_BUDGETS),
        "datasets": list(DATASETS),
        "n_cells_expected": N_CELLS_EXPECTED,
        "groups": groups,
    }
    json_path = report_dir / "prompt_inclusive_core_3seed.json"
    md_path = report_dir / "prompt_inclusive_core_3seed.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(groups), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
