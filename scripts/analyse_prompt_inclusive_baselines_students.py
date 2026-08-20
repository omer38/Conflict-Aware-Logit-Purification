#!/usr/bin/env python
"""Analyse the prompt-inclusive baseline student grid (Phase 2, 3 seeds).

Read-only over ``outputs/student/prompt_inclusive_baselines_3seed_v1/`` for the
three baseline methods (Gemma single, Mistral single, entropy-weighted
committee), joined against the Phase 1 core grid in
``outputs/student/prompt_inclusive_core_3seed_v1/``.

Both grids use the same three seeds, the same ``large-test-v1`` fixed test rows
and the same matched configuration, so cells are *paired* and support exact
McNemar tests with Holm correction within each dataset x L cell.

Writes:
  - outputs/reports/prompt_inclusive_baselines_3seed.json
  - outputs/reports/prompt_inclusive_baselines_3seed.md

Run:
    python3 scripts/analyse_prompt_inclusive_baselines_students.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
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

# Phase 2 baselines.
BASELINE_METHODS: dict[str, str] = {
    "gemma": "gemma4-26b-vllm-guided",
    "mistral": "mistral-small-24b-vllm-guided",
    "entropy": "multi-entropy-guided",
}
# Phase 1 cells needed as comparators.
CORE_METHODS: dict[str, str] = {
    "raw_qwen": "qwen3.5-27b-vllm-guided",
    "prior_matched_qwen": "qwen-prior-matched-guided",
    "prob_mean": "multi-probmean-guided",
    "combined": "multi-prior-scale-aligned-guided",
}
SINGLE_TEACHERS = ("raw_qwen", "gemma", "mistral")

REQUIRED_HYPERPARAMS = {
    "model_name": "answerdotai/ModernBERT-base",
    "epochs": 3,
    "alpha": 0.5,
    "temperature": 2.0,
    "lr": 2e-5,
    "batch_size": 16,
    "max_length": 256,
}

N_CELLS_EXPECTED = (
    len(DATASETS) * len(LABEL_BUDGETS) * len(SEEDS) * len(BASELINE_METHODS)
)


def cell_path(base: Path, dataset: str, teacher: str, labels: int, seed: int) -> Path:
    return base / (
        f"{dataset}_kd_{teacher}_L{labels}_s{seed}_prompt-inclusive-v1.json"
    )


def validate(data: dict, dataset: str, teacher: str, labels: int, seed: int) -> list[str]:
    """Return a list of contract violations; empty means the cell is valid."""
    errors: list[str] = []
    if data.get("dataset") != dataset:
        errors.append(f"dataset={data.get('dataset')!r}")
    if data.get("mode") != "kd":
        errors.append(f"mode={data.get('mode')!r}")
    if data.get("teacher") != teacher:
        errors.append(f"teacher={data.get('teacher')!r}")
    if data.get("labels_per_class") != labels:
        errors.append(f"labels_per_class={data.get('labels_per_class')!r}")
    if data.get("seed") != seed:
        errors.append(f"seed={data.get('seed')!r}")
    if data.get("label_accounting", {}).get("policy") != "prompt-inclusive-v1":
        errors.append("label_accounting.policy != prompt-inclusive-v1")
    if data.get("dataset_identity", {}).get("artifact_version") != "large-test-v1":
        errors.append("dataset_identity.artifact_version != large-test-v1")
    if teacher == BASELINE_METHODS["entropy"] and data.get("aggregation") != "entropy_weighted":
        errors.append(f"aggregation={data.get('aggregation')!r} != entropy_weighted")

    metrics = data.get("metrics", {})
    for key in ("accuracy", "macro_f1"):
        value = metrics.get(key)
        if not (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        ):
            errors.append(f"metric {key}={value!r}")

    expected_n = EXPECTED_TEST_SIZE[dataset]
    paired = data.get("paired_test_data", {})
    for key in ("sample_indices", "labels", "predictions"):
        if len(paired.get(key, [])) != expected_n:
            errors.append(f"paired_test_data.{key} length != {expected_n}")

    cfg = data.get("config", {})
    for key, expected in REQUIRED_HYPERPARAMS.items():
        if cfg.get(key) != expected:
            errors.append(f"config.{key}={cfg.get(key)!r} != {expected!r}")
    return errors


def load_cell(base: Path, dataset: str, teacher: str, labels: int, seed: int) -> dict:
    path = cell_path(base, dataset, teacher, labels, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing cell: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate(data, dataset, teacher, labels, seed)
    if errors:
        raise ValueError(f"invalid cell {path.name}: {'; '.join(errors)}")
    return data


def mcnemar_exact(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact binomial McNemar test over discordant pairs."""
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n_disc = only_a + only_b
    p_value = 1.0 if n_disc == 0 else float(binomtest(only_a, n_disc, p=0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "p_value": p_value}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_values[idx]
        running = max(running, min(1.0, val))
        adjusted[idx] = running
    return adjusted


def summarise(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


def compare(cells_a: dict, cells_b: dict, name_a: str, name_b: str) -> dict:
    """Paired per-seed comparison between two methods within one cell."""
    deltas, seed_rows, p_values = [], [], []
    for seed in SEEDS:
        a, b = cells_a[seed], cells_b[seed]
        labels = np.asarray(a["paired_test_data"]["labels"])
        pa = np.asarray(a["paired_test_data"]["predictions"])
        pb = np.asarray(b["paired_test_data"]["predictions"])
        if not np.array_equal(labels, np.asarray(b["paired_test_data"]["labels"])):
            raise ValueError("paired label vectors differ; cells are not comparable")
        res = mcnemar_exact(labels, pa, pb)
        delta = (a["metrics"]["accuracy"] - b["metrics"]["accuracy"]) * 100.0
        deltas.append(delta)
        p_values.append(res["p_value"])
        seed_rows.append({"seed": seed, "delta_accuracy_pp": delta, **res})
    for row, adj in zip(seed_rows, holm_adjust(p_values)):
        row["holm_p"] = adj
    wins_a = sum(1 for d in deltas if d > 0)
    wins_b = sum(1 for d in deltas if d < 0)
    return {
        "method_a": name_a,
        "method_b": name_b,
        "mean_delta_accuracy_pp": statistics.mean(deltas),
        "std_delta_accuracy_pp": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "seed_wins_a": wins_a,
        "seed_wins_b": wins_b,
        "all_seeds_same_sign": wins_a == len(SEEDS) or wins_b == len(SEEDS),
        "holm_significant_seeds": sum(1 for r in seed_rows if r["holm_p"] < 0.05),
        "seeds": seed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines-dir",
        default="outputs/student/prompt_inclusive_baselines_3seed_v1",
    )
    parser.add_argument(
        "--core-dir",
        default="outputs/student/prompt_inclusive_core_3seed_v1",
    )
    parser.add_argument("--report-dir", default="outputs/reports")
    args = parser.parse_args()

    base_dir = Path(args.baselines_dir)
    core_dir = Path(args.core_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    groups: list[dict] = []

    for dataset in DATASETS:
        for labels in LABEL_BUDGETS:
            cells: dict[str, dict[int, dict]] = {}
            for key, teacher in BASELINE_METHODS.items():
                cells[key] = {}
                for seed in SEEDS:
                    try:
                        cells[key][seed] = load_cell(base_dir, dataset, teacher, labels, seed)
                    except (FileNotFoundError, ValueError) as exc:
                        problems.append(str(exc))
            for key, teacher in CORE_METHODS.items():
                cells[key] = {}
                for seed in SEEDS:
                    try:
                        cells[key][seed] = load_cell(core_dir, dataset, teacher, labels, seed)
                    except (FileNotFoundError, ValueError) as exc:
                        problems.append(str(exc))
            if problems:
                continue

            methods = {
                key: {
                    "teacher": BASELINE_METHODS.get(key) or CORE_METHODS.get(key),
                    "accuracy": summarise(
                        [cells[key][s]["metrics"]["accuracy"] for s in SEEDS]
                    ),
                    "macro_f1": summarise(
                        [cells[key][s]["metrics"]["macro_f1"] for s in SEEDS]
                    ),
                }
                for key in list(BASELINE_METHODS) + list(CORE_METHODS)
            }

            best_single = max(
                SINGLE_TEACHERS, key=lambda k: methods[k]["accuracy"]["mean"]
            )

            comparisons = [
                compare(cells["gemma"], cells["raw_qwen"], "gemma", "raw_qwen"),
                compare(cells["mistral"], cells["raw_qwen"], "mistral", "raw_qwen"),
                compare(cells["entropy"], cells["prob_mean"], "entropy", "prob_mean"),
                compare(cells["entropy"], cells["combined"], "entropy", "combined"),
                compare(
                    cells["combined"], cells[best_single], "combined", f"best_single:{best_single}"
                ),
                compare(
                    cells["prior_matched_qwen"],
                    cells[best_single],
                    "prior_matched_qwen",
                    f"best_single:{best_single}",
                ),
            ]

            groups.append(
                {
                    "dataset": dataset,
                    "labels_per_class": labels,
                    "n_seeds": len(SEEDS),
                    "validation_status": "ok",
                    "best_single_teacher": best_single,
                    "methods": methods,
                    "comparisons": comparisons,
                }
            )

    if problems:
        print("Baseline analysis incomplete:", file=sys.stderr)
        for line in problems[:40]:
            print(f"  - {line}", file=sys.stderr)
        raise SystemExit(1)

    payload = {
        "experiment": "prompt_inclusive_baselines_3seed_v1",
        "label_accounting": "prompt-inclusive-v1",
        "artifact_version": "large-test-v1",
        "baseline_methods": BASELINE_METHODS,
        "core_comparators": CORE_METHODS,
        "seeds": list(SEEDS),
        "label_budgets": list(LABEL_BUDGETS),
        "datasets": list(DATASETS),
        "n_cells_expected": N_CELLS_EXPECTED,
        "groups": groups,
    }
    json_path = report_dir / "prompt_inclusive_baselines_3seed.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Markdown report.
    lines = [
        "# Prompt-inclusive baseline student results (Phase 2, 3 seeds)",
        "",
        "Gemma single, Mistral single and the entropy-weighted committee under "
        "`prompt-inclusive-v1` accounting on the `large-test-v1` fixed tests, "
        "joined against the Phase 1 core grid.",
        "",
        f"Seeds: {', '.join(str(s) for s in SEEDS)}. Paired exact McNemar tests are "
        "computed per seed on identical test rows; Holm adjustment is applied across "
        "the three seed-level tests within each comparison.",
        "",
    ]
    for g in groups:
        lines += [
            f"## {g['dataset']} · L={g['labels_per_class']}",
            "",
            f"Best single teacher: **{g['best_single_teacher']}**",
            "",
            "| Method | Accuracy | Macro-F1 | Seed accuracies |",
            "|---|---:|---:|---|",
        ]
        for key, m in g["methods"].items():
            a, f = m["accuracy"], m["macro_f1"]
            seeds = ", ".join(f"{v * 100:.2f}%" for v in a["values"])
            lines.append(
                f"| {key} | {a['mean'] * 100:.2f}% ± {a['std'] * 100:.2f} "
                f"| {f['mean'] * 100:.2f}% ± {f['std'] * 100:.2f} | {seeds} |"
            )
        lines += ["", "| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |", "|---|---:|---:|---|---:|"]
        for c in g["comparisons"]:
            lines.append(
                f"| {c['method_a']} − {c['method_b']} | {c['mean_delta_accuracy_pp']:+.2f} "
                f"± {c['std_delta_accuracy_pp']:.2f} | {c['seed_wins_a']}/{c['seed_wins_b']} "
                f"| {'yes' if c['all_seeds_same_sign'] else 'no'} | {c['holm_significant_seeds']} |"
            )
        lines.append("")
    md_path = report_dir / "prompt_inclusive_baselines_3seed.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
