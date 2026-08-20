"""Focused tests for prompt-inclusive core student analysis/validation.

Synthetic small JSON fixtures only — does not train ModernBERT.

Run:
    uv run python tests/test_prompt_inclusive_core_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyse_prompt_inclusive_core_students import (  # noqa: E402
    SEEDS,
    analyse_comparison,
    analyse_group,
    assert_pair_compatible,
    holm_adjust,
    mean_std,
    mcnemar_exact,
    paired_delta_stats,
    result_path,
    validate_cell,
)


def _base_cell(
    *,
    dataset: str = "ag_news",
    mode: str = "supervised",
    teacher: str = "none",
    labels_per_class: int = 10,
    seed: int = 42,
    n_test: int = 8,
    alpha: float = 0.0,
    artifact_version: str = "large-test-v1",
    policy: str = "prompt-inclusive-v1",
    accuracy: float = 0.5,
    macro_f1: float = 0.5,
    labels: list[int] | None = None,
    predictions: list[int] | None = None,
    aggregation: str | None = None,
    class_prior: dict | None = None,
) -> dict:
    if labels is None:
        labels = [0, 1, 0, 1, 0, 1, 0, 1][:n_test]
    if predictions is None:
        predictions = list(labels)
    cell = {
        "dataset": dataset,
        "mode": mode,
        "teacher": teacher,
        "labels_per_class": labels_per_class,
        "seed": seed,
        "label_accounting": {
            "policy": policy,
            "declared_labels_per_class": labels_per_class,
            "prompt_demo_labels_per_class": 2,
            "student_ce_labels_per_class": labels_per_class,
            "router_calibration_labels_per_class": 0,
            "calibration_reserved": False,
        },
        "config": {
            "model_name": "answerdotai/ModernBERT-base",
            "epochs": 3,
            "alpha": alpha,
            "temperature": 2.0,
            "lr": 2e-5,
            "batch_size": 16,
            "max_length": 256,
        },
        "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
        "dataset_identity": {
            "artifact_version": artifact_version,
            "test": {
                "size": n_test,
                "sha256": "test-sha-fixed",
                "class_counts": [n_test // 2, n_test // 2],
            },
        },
        "paired_test_data": {
            "sample_indices": list(range(n_test)),
            "labels": labels,
            "predictions": predictions,
        },
    }
    if aggregation is not None:
        cell["aggregation"] = aggregation
    if class_prior is not None:
        cell["class_prior_assumption"] = class_prior
    return cell


def _patch_expected_sizes(test_case: unittest.TestCase) -> None:
    """Shrink expected test sizes so fixtures stay tiny."""
    import scripts.analyse_prompt_inclusive_core_students as mod

    test_case._orig_sizes = dict(mod.EXPECTED_TEST_SIZE)
    mod.EXPECTED_TEST_SIZE.update(
        {ds: 8 for ds in ("ag_news", "yahoo", "amazon", "amazon3")}
    )
    test_case.addCleanup(
        lambda: mod.EXPECTED_TEST_SIZE.update(test_case._orig_sizes)
    )


class ValidateCellTests(unittest.TestCase):
    def setUp(self) -> None:
        _patch_expected_sizes(self)

    def test_valid_supervised_cell(self) -> None:
        cell = _base_cell(mode="supervised", teacher="none", alpha=0.0)
        errors = validate_cell(
            cell,
            dataset="ag_news",
            mode="supervised",
            teacher="none",
            labels_per_class=10,
            seed=42,
        )
        self.assertEqual(errors, [])

    def test_valid_kd_cell(self) -> None:
        cell = _base_cell(
            mode="kd",
            teacher="multi-mean-guided",
            alpha=0.5,
            aggregation="logit_mean",
        )
        errors = validate_cell(
            cell,
            dataset="ag_news",
            mode="kd",
            teacher="multi-mean-guided",
            labels_per_class=10,
            seed=42,
        )
        self.assertEqual(errors, [])

    def test_rejects_wrong_label_accounting_policy(self) -> None:
        cell = _base_cell(policy="legacy")
        errors = validate_cell(
            cell,
            dataset="ag_news",
            mode="supervised",
            teacher="none",
            labels_per_class=10,
            seed=42,
        )
        self.assertTrue(any("prompt-inclusive-v1" in e for e in errors))

    def test_rejects_wrong_evaluation_artifact(self) -> None:
        cell = _base_cell(artifact_version="legacy")
        errors = validate_cell(
            cell,
            dataset="ag_news",
            mode="supervised",
            teacher="none",
            labels_per_class=10,
            seed=42,
        )
        self.assertTrue(any("large-test-v1" in e for e in errors))

    def test_rejects_wrong_prediction_length(self) -> None:
        cell = _base_cell(n_test=8)
        cell["paired_test_data"]["predictions"] = [0, 1, 0]
        errors = validate_cell(
            cell,
            dataset="ag_news",
            mode="supervised",
            teacher="none",
            labels_per_class=10,
            seed=42,
        )
        self.assertTrue(any("predictions" in e for e in errors))

    def test_rejects_wrong_configuration(self) -> None:
        cell = _base_cell()
        cell["config"]["batch_size"] = 8
        errors = validate_cell(
            cell,
            dataset="ag_news",
            mode="supervised",
            teacher="none",
            labels_per_class=10,
            seed=42,
        )
        self.assertTrue(any("batch_size" in e for e in errors))

    def test_rejects_mismatched_paired_test_identities(self) -> None:
        a = _base_cell(seed=42, accuracy=0.6)
        b = _base_cell(seed=42, accuracy=0.5)
        b["dataset_identity"]["test"]["sha256"] = "other-sha"
        with self.assertRaises(ValueError):
            assert_pair_compatible(a, b)


class GridAndStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        _patch_expected_sizes(self)

    def _write_full_grid(self, root: Path) -> None:
        import scripts.analyse_prompt_inclusive_core_students as mod

        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        # Deterministic per-method prediction flips so comparisons are non-trivial.
        pred_by_method = {
            "supervised": [0, 1, 0, 1, 0, 1, 0, 0],
            "raw_qwen": [0, 1, 0, 1, 0, 1, 1, 1],
            "prior_matched_qwen": [0, 1, 0, 1, 0, 1, 1, 0],
            "logit_mean": [0, 1, 0, 1, 1, 1, 0, 1],
            "prob_mean": [0, 1, 0, 1, 0, 0, 0, 1],
            "prior_matched": [0, 1, 1, 1, 0, 1, 0, 1],
            "scale_aligned": [0, 0, 0, 1, 0, 1, 0, 1],
            "combined": [0, 1, 0, 1, 0, 1, 0, 1],
        }
        for dataset in mod.DATASETS:
            for lpc in mod.LABEL_BUDGETS:
                for seed in mod.SEEDS:
                    for method_key, spec in mod.METHODS.items():
                        preds = pred_by_method[method_key]
                        # Slight seed-dependent flip to create seed variance.
                        preds = list(preds)
                        if seed != 42:
                            preds[seed % 8] = 1 - preds[seed % 8]
                        acc = float(np.mean(np.asarray(preds) == np.asarray(labels)))
                        cell = _base_cell(
                            dataset=dataset,
                            mode=spec["mode"],
                            teacher=spec["teacher"],
                            labels_per_class=lpc,
                            seed=seed,
                            alpha=0.0 if spec["mode"] == "supervised" else 0.5,
                            accuracy=acc,
                            macro_f1=acc,
                            labels=labels,
                            predictions=preds,
                            aggregation=mod.EXPECTED_AGGREGATION.get(spec["teacher"]),
                            class_prior=(
                                {
                                    "values": [0.25, 0.25, 0.25, 0.25],
                                    "source": (
                                        "known_uniform_prior_from_gold-balanced_"
                                        "working_pool"
                                    ),
                                    "uses_individual_labels_during_matching": False,
                                }
                                if spec["teacher"] in mod.PRIOR_DEPENDENT_TEACHERS
                                else None
                            ),
                        )
                        # NUM_CLASSES for non-ag_news needs matching prior length.
                        if (
                            spec["teacher"] in mod.PRIOR_DEPENDENT_TEACHERS
                            and dataset != "ag_news"
                        ):
                            c = mod.NUM_CLASSES[dataset]
                            cell["class_prior_assumption"]["values"] = [1.0 / c] * c
                        path = result_path(
                            root,
                            dataset,
                            spec["mode"],
                            spec["teacher"],
                            lpc,
                            seed,
                        )
                        path.write_text(
                            json.dumps(cell, indent=2) + "\n", encoding="utf-8"
                        )

    def test_incomplete_grid_fails_rather_than_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Write only one cell.
            cell = _base_cell()
            path = result_path(root, "ag_news", "supervised", "none", 10, 42)
            path.write_text(json.dumps(cell) + "\n", encoding="utf-8")
            with self.assertRaises((FileNotFoundError, ValueError)):
                analyse_group(root, "ag_news", 10)

    def test_mean_sample_sd_and_paired_deltas(self) -> None:
        stats = mean_std([0.1, 0.2, 0.3])
        self.assertAlmostEqual(stats["mean"], 0.2)
        self.assertAlmostEqual(stats["std"], math.sqrt(0.01), places=12)
        deltas = paired_delta_stats([0.05, -0.01, 0.02])
        self.assertIn("se", deltas)
        self.assertEqual(deltas["se_note"], "descriptive SE of seed-level paired deltas")
        self.assertAlmostEqual(deltas["mean"], (0.05 - 0.01 + 0.02) / 3)

    def test_mcnemar_discordant_counts(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
        pred_a = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int64)
        pred_b = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64)
        # A-only correct: idx 4 (0 vs 1 wrong for B); B-only: idx 1,3,5
        # Wait carefully:
        # idx0: A ok, B ok
        # idx1: A wrong (1!=0), B ok (0==0) -> only_b
        # idx2: A ok, B ok
        # idx3: A wrong (0!=1), B ok (1==1) -> only_b
        # idx4: A ok (0==0), B wrong (1!=0) -> only_a
        # idx5: A ok (1==1), B wrong (0!=1) -> only_a
        result = mcnemar_exact(labels, pred_a, pred_b)
        self.assertEqual(result["only_a_correct"], 2)
        self.assertEqual(result["only_b_correct"], 2)
        self.assertEqual(result["n_discordant"], 4)
        self.assertAlmostEqual(result["p_value"], 1.0)

    def test_mcnemar_zero_discordant(self) -> None:
        labels = np.asarray([0, 1, 0], dtype=np.int64)
        preds = np.asarray([0, 1, 0], dtype=np.int64)
        result = mcnemar_exact(labels, preds, preds)
        self.assertEqual(result["n_discordant"], 0)
        self.assertEqual(result["p_value"], 1.0)

    def test_holm_correction_three_seed_tests(self) -> None:
        raw = [0.01, 0.04, 0.03]
        adj = holm_adjust(raw)
        # Sorted: 0.01, 0.03, 0.04
        # rank0: 3*0.01=0.03
        # rank1: max(0.03, 2*0.03)=0.06
        # rank2: max(0.06, 1*0.04)=0.06
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertAlmostEqual(adj[2], 0.06)
        self.assertAlmostEqual(adj[1], 0.06)

    def test_full_synthetic_grid_analyses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_full_grid(root)
            group = analyse_group(root, "ag_news", 10)
            self.assertEqual(group["validation_status"], "ok")
            self.assertEqual(len(group["methods"]), 8)
            self.assertEqual(len(group["comparisons"]), 10)
            cmp0 = group["comparisons"]["raw_qwen_minus_supervised"]
            self.assertEqual(cmp0["n_seeds"], len(SEEDS))
            self.assertIn("se", cmp0["paired_delta"]["accuracy"])

    def test_analyse_comparison_seed_wins(self) -> None:
        cells_a = {}
        cells_b = {}
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        for seed in SEEDS:
            # A always perfect; B always wrong on last example.
            cells_a[seed] = _base_cell(
                seed=seed,
                accuracy=1.0,
                labels=labels,
                predictions=labels,
            )
            bad = list(labels)
            bad[-1] = 1 - bad[-1]
            cells_b[seed] = _base_cell(
                seed=seed,
                accuracy=float(np.mean(np.asarray(bad) == np.asarray(labels))),
                labels=labels,
                predictions=bad,
            )
        result = analyse_comparison(
            cells_a, cells_b, method_a="combined", method_b="prob_mean"
        )
        self.assertEqual(result["seed_wins"]["combined"], 3)
        self.assertEqual(result["seed_wins"]["prob_mean"], 0)
        self.assertEqual(result["seed_wins"]["tie"], 0)


if __name__ == "__main__":
    unittest.main()
