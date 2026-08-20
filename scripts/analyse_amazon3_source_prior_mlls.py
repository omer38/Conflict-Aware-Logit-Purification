#!/usr/bin/env python
"""Validate and analyse the six-cell BCTS+MLLS Amazon-3 follow-up.

The new student cells live in an isolated directory. Existing scale-only,
uniform-prior, and oracle-prior cells are read from the completed v1 grid;
they are never regenerated or overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.analyse_amazon3_source_prior as base_analysis  # noqa: E402
import scripts.train_amazon3_source_prior as experiment  # noqa: E402

NEW_DIR = Path("outputs/student/amazon3_source_prior_transfer_mlls_v1")
BASELINE_DIR = Path("outputs/student/amazon3_source_prior_transfer_v1")
REPORT_JSON = Path("outputs/reports/amazon3_source_prior_transfer_mlls_v1.json")
REPORT_MD = Path("outputs/reports/amazon3_source_prior_transfer_mlls_v1.md")

METHOD = experiment.MLLS_METHOD
BASELINES = ("scale_only", "uniform_prior", "oracle_prior")
REPORT_METHODS = ("scale_only", "uniform_prior", METHOD, "oracle_prior")
LABEL_BUDGETS = experiment.LABEL_BUDGETS
SEEDS = experiment.SEEDS
EXPECTED_CELLS = len(LABEL_BUDGETS) * len(SEEDS)
COMPARISONS = (
    (METHOD, "scale_only"),
    (METHOD, "uniform_prior"),
    ("oracle_prior", METHOD),
)


def result_path(labels_per_class: int, seed: int) -> Path:
    return NEW_DIR / (
        f"amazon3_source-prior-transfer_{METHOD}_L{labels_per_class}_s{seed}.json"
    )


def _finite_unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_cell(data: dict, *, labels_per_class: int, seed: int, deep: bool = True) -> list[str]:
    errors: list[str] = []
    if data.get("experiment_id") != experiment.MLLS_EXPERIMENT_ID:
        errors.append(f"experiment_id mismatch: {data.get('experiment_id')}")
    if data.get("method") != METHOD:
        errors.append(f"method mismatch: {data.get('method')}")
    if data.get("true_prior") != experiment.TRUE_PRIOR.tolist():
        errors.append("true_prior mismatch")

    split = data.get("labelled_split", {})
    if split.get("labels_per_class") != labels_per_class:
        errors.append("labelled_split.labels_per_class mismatch")
    if split.get("seed") != seed:
        errors.append("labelled_split.seed mismatch")
    if len(split.get("labelled_indices", [])) != 3 * labels_per_class:
        errors.append("labelled index count mismatch")
    if data.get("reference_batch", {}).get("size") != 500:
        errors.append("reference batch size mismatch")
    if data.get("working_pool", {}).get("size") != 2000:
        errors.append("working pool size mismatch")

    estimator = data.get("prior_estimator", {})
    if estimator.get("estimator") != "maximum_likelihood_label_shift":
        errors.append("estimator name mismatch")
    if estimator.get("calibration_method") != "bias_corrected_temperature_scaling":
        errors.append("calibration method mismatch")
    if estimator.get("solver_success") is not True:
        errors.append("MLLS solver did not succeed")
    if estimator.get("calibration", {}).get("solver_success") is not True:
        errors.append("BCTS solver did not succeed")
    estimate = np.asarray(estimator.get("estimated_prior", []), dtype=np.float64)
    assumed = np.asarray(data.get("assumed_prior", []), dtype=np.float64)
    if estimate.shape != (3,) or not np.allclose(estimate.sum(), 1.0, atol=1e-6):
        errors.append("estimated prior is not a three-class simplex vector")
    if assumed.shape != (3,) or estimate.shape != assumed.shape or not np.allclose(
        estimate, assumed, atol=1e-8
    ):
        errors.append("assumed prior does not equal the BCTS+MLLS estimate")

    metrics = data.get("metrics", {})
    for key in ("accuracy", "macro_f1"):
        if not _finite_unit_interval(metrics.get(key)):
            errors.append(f"invalid metric {key}: {metrics.get(key)}")
    paired = data.get("paired_test_data", {})
    for key in ("sample_indices", "labels", "predictions"):
        if len(paired.get(key, [])) != base_analysis.EXPECTED_TEST_SIZE:
            errors.append(f"paired_test_data.{key} length mismatch")
    for key, expected in base_analysis.REQUIRED_STUDENT_HYPERPARAMS.items():
        if data.get("student_config", {}).get(key) != expected:
            errors.append(f"student_config.{key} mismatch")

    if deep and not errors:
        reproduced = experiment.prepare_cell(METHOD, labels_per_class, seed)["result"]
        if data["labelled_split"]["labelled_indices"] != reproduced["labelled_split"]["labelled_indices"]:
            errors.append("labelled split does not match independent reproduction")
        if data["reference_batch"]["identity_hash"] != reproduced["reference_batch"]["identity_hash"]:
            errors.append("reference split hash does not match independent reproduction")
        if data["working_pool"]["identity_hash"] != reproduced["working_pool"]["identity_hash"]:
            errors.append("working split hash does not match independent reproduction")
        expected_prior = np.asarray(reproduced["assumed_prior"], dtype=np.float64)
        if not np.allclose(assumed, expected_prior, atol=1e-6):
            errors.append("estimated prior does not match independent reproduction")
    return errors


def load_new_cell(labels_per_class: int, seed: int) -> dict:
    path = result_path(labels_per_class, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing BCTS+MLLS result: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_cell(data, labels_per_class=labels_per_class, seed=seed)
    if errors:
        raise ValueError(f"invalid cell {path}: {'; '.join(errors)}")
    return data


def mean_std(values: list[float]) -> dict:
    return base_analysis.mean_std(values)


def analyse_group(labels_per_class: int) -> dict:
    cells: dict[str, dict[int, dict]] = {method: {} for method in REPORT_METHODS}
    for seed in SEEDS:
        cells[METHOD][seed] = load_new_cell(labels_per_class, seed)
        for baseline in BASELINES:
            cells[baseline][seed] = base_analysis.load_cell(
                BASELINE_DIR, baseline, labels_per_class, seed
            )

    methods: dict[str, dict] = {}
    for method in REPORT_METHODS:
        acc = [cells[method][seed]["metrics"]["accuracy"] for seed in SEEDS]
        f1 = [cells[method][seed]["metrics"]["macro_f1"] for seed in SEEDS]
        methods[method] = {
            "accuracy": mean_std(acc),
            "macro_f1": mean_std(f1),
            "seeds": [
                {
                    "seed": seed,
                    "accuracy": cells[method][seed]["metrics"]["accuracy"],
                    "macro_f1": cells[method][seed]["metrics"]["macro_f1"],
                    "assumed_prior": cells[method][seed]["assumed_prior"],
                    "prior_l1_error": cells[method][seed]["prior_estimator"]["l1_error"],
                }
                for seed in SEEDS
            ],
        }

    comparisons = {
        f"{a}_minus_{b}": base_analysis.analyse_comparison(
            cells[a], cells[b], label_a=a, label_b=b
        )
        for a, b in COMPARISONS
    }
    return {
        "labels_per_class": labels_per_class,
        "methods": methods,
        "comparisons": comparisons,
    }


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(groups: list[dict]) -> str:
    lines = [
        "# Amazon-3 calibrated maximum-likelihood prior follow-up",
        "",
        "This isolated follow-up replaces confusion-moment inversion with "
        "bias-corrected temperature scaling plus maximum-likelihood label-shift "
        "estimation (BCTS+MLLS). It was motivated after the original BBSE "
        "estimator showed small-sample instability and is therefore reported as "
        "a follow-up robustness analysis, not a predeclared primary estimator.",
        "",
        "The same 500-row unlabelled reference batch, 2,000-row working pool, "
        "declared labels, teacher cache, student configuration, and seeds are used. "
        "Existing baseline and oracle cells are reused without retraining.",
        "",
    ]
    for group in groups:
        lpc = group["labels_per_class"]
        lines.extend(
            [
                f"## L={lpc}",
                "",
                "| method | accuracy | macro-F1 |",
                "|---|---:|---:|",
            ]
        )
        for method in REPORT_METHODS:
            row = group["methods"][method]
            lines.append(
                f"| {method} | {fmt_pct(row['accuracy']['mean'])} +/- "
                f"{fmt_pct(row['accuracy']['std'])} | "
                f"{fmt_pct(row['macro_f1']['mean'])} +/- "
                f"{fmt_pct(row['macro_f1']['std'])} |"
            )
        lines.extend(
            [
                "",
                "### BCTS+MLLS prior estimates",
                "",
                "| seed | estimated prior | L1 error |",
                "|---:|---|---:|",
            ]
        )
        for row in group["methods"][METHOD]["seeds"]:
            prior = "[" + ", ".join(f"{v:.3f}" for v in row["assumed_prior"]) + "]"
            lines.append(f"| {row['seed']} | {prior} | {row['prior_l1_error']:.4f} |")
        lines.append("")
        for a, b in COMPARISONS:
            comparison = group["comparisons"][f"{a}_minus_{b}"]
            d_acc = comparison["paired_delta"]["accuracy"]
            d_f1 = comparison["paired_delta"]["macro_f1"]
            wins = comparison["seed_wins"]
            lines.extend(
                [
                    f"### {a} - {b}",
                    "",
                    f"- Mean delta macro-F1: {100 * d_f1['mean']:+.2f} +/- "
                    f"{100 * d_f1['std']:.2f} pp",
                    f"- Mean delta accuracy: {100 * d_acc['mean']:+.2f} +/- "
                    f"{100 * d_acc['std']:.2f} pp",
                    f"- Accuracy wins/ties/losses: {wins[a]}/{wins['tie']}/{wins[b]}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Provenance",
            "",
            f"- New valid cells: {EXPECTED_CELLS}/{EXPECTED_CELLS}",
            f"- New-cell directory: `{NEW_DIR}`",
            f"- Reused baseline directory: `{BASELINE_DIR}`",
            "- No new teacher queries were made.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-cell", type=Path)
    parser.add_argument("--labels-per-class", type=int, choices=LABEL_BUDGETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()

    if args.check_cell is not None:
        if args.labels_per_class is None or args.seed is None:
            parser.error("--check-cell requires --labels-per-class and --seed")
        data = json.loads(args.check_cell.read_text(encoding="utf-8"))
        errors = validate_cell(
            data, labels_per_class=args.labels_per_class, seed=args.seed
        )
        if errors:
            print("\n".join(errors), file=sys.stderr)
            raise SystemExit(1)
        print("valid")
        return

    groups = [analyse_group(lpc) for lpc in LABEL_BUDGETS]
    report = {
        "experiment_id": experiment.MLLS_EXPERIMENT_ID,
        "method": METHOD,
        "labels_per_class": list(LABEL_BUDGETS),
        "seeds": list(SEEDS),
        "groups": groups,
        "follow_up_not_predeclared": True,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(groups), encoding="utf-8")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
