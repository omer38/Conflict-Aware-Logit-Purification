"""Tests for the student-level A.12 prior-misspecification experiment.

Covers the shared ``skew_class_prior`` helper, ``train_student.py``'s
prior-condition wiring, the dedicated validator/analyser, and a synthetic
54-cell fixture. Only two tests shell out to ``train_student.py --dry-run``
(fast: no ModernBERT load, no GPU wait, no teacher queries); everything else
is a pure in-process unit test.

Run:
    uv run pytest tests/test_student_prior_misspecification.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.student.soft_labels import (  # noqa: E402
    class_prior_assumption_metadata,
    require_prior_condition_supported,
    skew_class_prior,
    uniform_class_prior,
)

import scripts.analyse_student_prior_misspecification as analysis  # noqa: E402


# ---------------------------------------------------------------------------
# 1-8: exact A.12 skew vectors, sum-to-one, finite/positive.
# ---------------------------------------------------------------------------
class SkewClassPriorVectorTests(unittest.TestCase):
    def test_ag_news_mild_vector(self) -> None:
        np.testing.assert_allclose(
            skew_class_prior(4, 0.15, 0), [0.3625, 0.2125, 0.2125, 0.2125]
        )

    def test_ag_news_strong_vector(self) -> None:
        np.testing.assert_allclose(skew_class_prior(4, 0.50, 0), [0.625, 0.125, 0.125, 0.125])

    def test_amazon5_mild_vector(self) -> None:
        np.testing.assert_allclose(
            skew_class_prior(5, 0.15, 0), [0.32, 0.17, 0.17, 0.17, 0.17]
        )

    def test_amazon5_strong_vector(self) -> None:
        np.testing.assert_allclose(skew_class_prior(5, 0.50, 0), [0.60, 0.10, 0.10, 0.10, 0.10])

    def test_yahoo_mild_vector(self) -> None:
        np.testing.assert_allclose(
            skew_class_prior(10, 0.15, 0),
            [0.235, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085],
        )

    def test_yahoo_strong_vector(self) -> None:
        np.testing.assert_allclose(
            skew_class_prior(10, 0.50, 0),
            [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        )

    def test_all_priors_sum_to_one(self) -> None:
        for num_classes in (2, 3, 4, 5, 10):
            for strength in (0.01, 0.15, 0.50, 0.99):
                for dominant in range(num_classes):
                    prior = skew_class_prior(num_classes, strength, dominant)
                    self.assertAlmostEqual(float(prior.sum()), 1.0, places=12)

    def test_all_priors_finite_and_positive(self) -> None:
        for num_classes in (2, 3, 4, 5, 10):
            for strength in (0.01, 0.15, 0.50, 0.99):
                for dominant in range(num_classes):
                    prior = skew_class_prior(num_classes, strength, dominant)
                    self.assertTrue(np.all(np.isfinite(prior)))
                    self.assertTrue(np.all(prior > 0))


# ---------------------------------------------------------------------------
# 9-10: invalid inputs are rejected.
# ---------------------------------------------------------------------------
class SkewClassPriorValidationTests(unittest.TestCase):
    def test_invalid_strengths_fail(self) -> None:
        for strength in (0.0, 1.0, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                skew_class_prior(4, strength, 0)

    def test_invalid_dominant_classes_fail(self) -> None:
        for dominant in (-1, 4, 5, 100):
            with self.assertRaises(ValueError):
                skew_class_prior(4, 0.15, dominant)

    def test_invalid_num_classes_fail(self) -> None:
        with self.assertRaises(ValueError):
            skew_class_prior(1, 0.15, 0)
        with self.assertRaises(ValueError):
            skew_class_prior(0, 0.15, 0)


# ---------------------------------------------------------------------------
# 11, 14: correct condition stays uniform; metadata records assumed + true priors.
# ---------------------------------------------------------------------------
class ClassPriorAssumptionMetadataTests(unittest.TestCase):
    def test_correct_condition_is_uniform_and_backward_compatible(self) -> None:
        uniform = uniform_class_prior(4)
        meta = class_prior_assumption_metadata("correct", uniform, uniform, 0, None)
        self.assertEqual(
            meta,
            {
                "values": [0.25, 0.25, 0.25, 0.25],
                "source": "known_uniform_prior_from_gold-balanced_working_pool",
                "uses_individual_labels_during_matching": False,
            },
        )

    def test_misspecified_metadata_records_assumed_and_true_priors(self) -> None:
        uniform = uniform_class_prior(4)
        mild = skew_class_prior(4, 0.15, 0)
        meta = class_prior_assumption_metadata("mild", mild, uniform, 0, 0.15)
        self.assertEqual(meta["condition"], "mild")
        self.assertEqual(meta["values"], mild.tolist())
        self.assertEqual(meta["true_working_pool_prior"], uniform.tolist())
        self.assertEqual(meta["strength"], 0.15)
        self.assertEqual(meta["dominant_class"], 0)
        self.assertEqual(
            meta["source"],
            "predeclared_A12_style_misspecification_on_gold_balanced_working_pool",
        )
        self.assertFalse(meta["uses_individual_labels_during_matching"])


# ---------------------------------------------------------------------------
# 12: misspecified conditions rejected for irrelevant teacher methods.
# ---------------------------------------------------------------------------
class PriorConditionTeacherGuardTests(unittest.TestCase):
    def test_rejects_mild_for_unrelated_teacher(self) -> None:
        with self.assertRaises(ValueError):
            require_prior_condition_supported("kd", "multi-mean-guided", "mild")

    def test_rejects_strong_for_supervised_mode(self) -> None:
        with self.assertRaises(ValueError):
            require_prior_condition_supported("supervised", "none", "strong")

    def test_allows_mild_strong_for_registered_teacher(self) -> None:
        require_prior_condition_supported(
            "kd", "multi-prior-scale-aligned-guided", "mild"
        )
        require_prior_condition_supported(
            "kd", "multi-prior-scale-aligned-guided", "strong"
        )

    def test_correct_condition_always_allowed(self) -> None:
        require_prior_condition_supported("supervised", "none", "correct")
        require_prior_condition_supported("kd", "multi-mean-guided", "correct")


# ---------------------------------------------------------------------------
# 13: mild/strong filenames cannot collide with correct outputs.
# ---------------------------------------------------------------------------
class FilenameConventionTests(unittest.TestCase):
    def test_correct_tag_is_empty(self) -> None:
        self.assertEqual(analysis.prior_filename_tag("correct", 0), "")

    def test_misspecified_tags_are_nonempty_and_distinct(self) -> None:
        mild_tag = analysis.prior_filename_tag("mild", 0)
        strong_tag = analysis.prior_filename_tag("strong", 0)
        self.assertEqual(mild_tag, "_prior-mild-d0")
        self.assertEqual(strong_tag, "_prior-strong-d0")
        self.assertNotEqual(mild_tag, strong_tag)

    def test_filenames_do_not_collide_across_conditions(self) -> None:
        tmp = Path("/tmp/_unused_for_paths_only")
        paths = {
            condition: analysis.result_path(tmp, "ag_news", condition, 10, 42)
            for condition in ("correct", "mild", "strong")
        }
        self.assertEqual(len(set(paths.values())), 3)
        self.assertNotIn("_prior-", paths["correct"].name)
        self.assertIn("_prior-mild-d0_", paths["mild"].name)
        self.assertIn("_prior-strong-d0_", paths["strong"].name)


# ---------------------------------------------------------------------------
# 15-17: validator rejects incorrect prior / strength / mismatched test identity.
# ---------------------------------------------------------------------------
def _valid_cell(
    *,
    dataset: str = "ag_news",
    condition: str = "correct",
    labels_per_class: int = 10,
    seed: int = 42,
    n_test: int = 8,
    dominant_class: int = 0,
) -> dict:
    num_classes = analysis.NUM_CLASSES[dataset]
    labels = [i % num_classes for i in range(n_test)]
    cell = {
        "dataset": dataset,
        "mode": "kd",
        "teacher": analysis.TEACHER,
        "labels_per_class": labels_per_class,
        "seed": seed,
        "label_accounting": {
            "policy": "prompt-inclusive-v1",
            "declared_labels_per_class": labels_per_class,
            "prompt_demo_labels_per_class": 2,
            "student_ce_labels_per_class": labels_per_class,
            "router_calibration_labels_per_class": 0,
            "calibration_reserved": False,
        },
        "config": dict(analysis.REQUIRED_KD_HYPERPARAMS),
        "metrics": {"accuracy": 0.5, "macro_f1": 0.5},
        "dataset_identity": {
            "artifact_version": "large-test-v1",
            "test": {"size": n_test, "sha256": f"test-sha-{dataset}"},
        },
        "paired_test_data": {
            "sample_indices": list(range(n_test)),
            "labels": labels,
            "predictions": list(labels),
        },
        "aggregation": analysis.EXPECTED_AGGREGATION,
    }
    if condition == "correct":
        cell["class_prior_assumption"] = {
            "values": uniform_class_prior(num_classes).tolist(),
            "source": "known_uniform_prior_from_gold-balanced_working_pool",
            "uses_individual_labels_during_matching": False,
        }
    else:
        strength = analysis.STRENGTH[condition]
        cell["class_prior_assumption"] = {
            "condition": condition,
            "values": skew_class_prior(num_classes, strength, dominant_class).tolist(),
            "true_working_pool_prior": uniform_class_prior(num_classes).tolist(),
            "strength": strength,
            "dominant_class": dominant_class,
            "source": "predeclared_A12_style_misspecification_on_gold_balanced_working_pool",
            "uses_individual_labels_during_matching": False,
        }
        cell["teacher_target_diagnostics"] = {
            "scope": "offline_full_balanced_working_pool",
            "uses_gold_labels_for_evaluation_only": True,
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "nll": 1.0,
            "num_rows": 6000,
        }
    return cell


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_sizes = dict(analysis.EXPECTED_TEST_SIZE)
        analysis.EXPECTED_TEST_SIZE.update({ds: 8 for ds in analysis.DATASETS})
        self.addCleanup(lambda: analysis.EXPECTED_TEST_SIZE.update(self._orig_sizes))

    def test_valid_correct_cell_has_no_errors(self) -> None:
        cell = _valid_cell(condition="correct")
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="correct", labels_per_class=10, seed=42
        )
        self.assertEqual(errors, [])

    def test_valid_mild_cell_has_no_errors(self) -> None:
        cell = _valid_cell(condition="mild")
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="mild", labels_per_class=10, seed=42
        )
        self.assertEqual(errors, [])

    def test_validator_rejects_incorrect_prior(self) -> None:
        cell = _valid_cell(condition="mild")
        cell["class_prior_assumption"]["values"] = [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="mild", labels_per_class=10, seed=42
        )
        self.assertTrue(any("values does not match" in e for e in errors))

    def test_validator_rejects_incorrect_strength(self) -> None:
        cell = _valid_cell(condition="mild")
        cell["class_prior_assumption"]["strength"] = 0.99
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="mild", labels_per_class=10, seed=42
        )
        self.assertTrue(any("strength mismatch" in e for e in errors))

    def test_validator_rejects_incorrect_dominant_class(self) -> None:
        cell = _valid_cell(condition="strong")
        cell["class_prior_assumption"]["dominant_class"] = 3
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="strong", labels_per_class=10, seed=42
        )
        self.assertTrue(any("dominant_class mismatch" in e for e in errors))

    def test_validator_rejects_mismatched_test_identity(self) -> None:
        a = _valid_cell(condition="correct", seed=42)
        b = _valid_cell(condition="mild", seed=42)
        b["dataset_identity"]["test"]["sha256"] = "some-other-sha"
        with self.assertRaises(ValueError):
            analysis.assert_pair_compatible(a, b)

    def test_validator_rejects_non_finite_metrics(self) -> None:
        cell = _valid_cell(condition="correct")
        cell["metrics"]["accuracy"] = float("nan")
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="correct", labels_per_class=10, seed=42
        )
        self.assertTrue(any("invalid metric accuracy" in e for e in errors))

    def test_validator_rejects_wrong_prediction_length(self) -> None:
        cell = _valid_cell(condition="correct")
        cell["paired_test_data"]["predictions"] = [0, 1]
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="correct", labels_per_class=10, seed=42
        )
        self.assertTrue(any("predictions" in e for e in errors))

    def test_validator_rejects_legacy_artifact_version(self) -> None:
        cell = _valid_cell(condition="correct")
        cell["dataset_identity"]["artifact_version"] = "legacy"
        errors = analysis.validate_cell(
            cell, dataset="ag_news", condition="correct", labels_per_class=10, seed=42
        )
        self.assertTrue(any("legacy/mismatched artifact" in e for e in errors))


# ---------------------------------------------------------------------------
# 18-19: validator detects missing / duplicate cells across the grid.
# ---------------------------------------------------------------------------
class GridCompletenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.baseline_dir = Path(self._tmp.name) / "baseline"
        self.new_dir = Path(self._tmp.name) / "new"
        self.baseline_dir.mkdir()
        self.new_dir.mkdir()

        self._orig_baseline = analysis.BASELINE_DIR
        self._orig_new = analysis.NEW_CELLS_DIR
        analysis.BASELINE_DIR = self.baseline_dir
        analysis.NEW_CELLS_DIR = self.new_dir
        self.addCleanup(self._restore_dirs)

    def _restore_dirs(self) -> None:
        analysis.BASELINE_DIR = self._orig_baseline
        analysis.NEW_CELLS_DIR = self._orig_new

    def _write(self, directory: Path, dataset: str, condition: str, lpc: int, seed: int) -> None:
        path = analysis.result_path(directory, dataset, condition, lpc, seed)
        path.write_text(json.dumps(_valid_cell(dataset=dataset, condition=condition, labels_per_class=lpc, seed=seed)))

    def _write_full_grid(self) -> None:
        for dataset in analysis.DATASETS:
            for lpc in analysis.LABEL_BUDGETS:
                for seed in analysis.SEEDS:
                    self._write(self.baseline_dir, dataset, "correct", lpc, seed)
                    self._write(self.new_dir, dataset, "mild", lpc, seed)
                    self._write(self.new_dir, dataset, "strong", lpc, seed)

    def test_detects_missing_cells(self) -> None:
        self._write_full_grid()
        # Remove one mild cell.
        missing_path = analysis.result_path(self.new_dir, "ag_news", "mild", 10, 42)
        missing_path.unlink()

        completeness = analysis.scan_grid_completeness()
        self.assertIn("('ag_news', 10, 42, 'mild', 0)", completeness["missing_cells"])
        self.assertEqual(len(completeness["missing_cells"]), 1)

    def test_detects_duplicate_cells(self) -> None:
        self._write_full_grid()
        # Accidentally also write the same mild cell into the baseline dir.
        self._write(self.baseline_dir, "ag_news", "mild", 10, 42)

        completeness = analysis.scan_grid_completeness()
        # A "correct"-only baseline dir now contains a mild file: this is
        # both a duplicate (present under two roots) and misplaced.
        self.assertTrue(
            completeness["duplicate_cells"] or completeness["misplaced_cells"]
        )


# ---------------------------------------------------------------------------
# 20: synthetic 54-cell fixture produces the expected paired deltas.
# ---------------------------------------------------------------------------
class SyntheticFullGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.baseline_dir = Path(self._tmp.name) / "baseline"
        self.new_dir = Path(self._tmp.name) / "new"
        self.baseline_dir.mkdir()
        self.new_dir.mkdir()

        self._orig_baseline = analysis.BASELINE_DIR
        self._orig_new = analysis.NEW_CELLS_DIR
        self._orig_sizes = dict(analysis.EXPECTED_TEST_SIZE)
        self._orig_offline_teacher_target = analysis.offline_teacher_target
        analysis.BASELINE_DIR = self.baseline_dir
        analysis.NEW_CELLS_DIR = self.new_dir
        analysis.EXPECTED_TEST_SIZE.update({ds: 8 for ds in analysis.DATASETS})
        analysis.offline_teacher_target = lambda *a, **k: {  # noqa: ARG005
            "accuracy": 0.7,
            "macro_f1": 0.7,
            "nll": 0.9,
        }
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        analysis.BASELINE_DIR = self._orig_baseline
        analysis.NEW_CELLS_DIR = self._orig_new
        analysis.EXPECTED_TEST_SIZE.update(self._orig_sizes)
        analysis.offline_teacher_target = self._orig_offline_teacher_target

    def _write_cell(self, dataset: str, condition: str, lpc: int, seed: int, predictions: list[int], labels: list[int]) -> None:
        cell = _valid_cell(dataset=dataset, condition=condition, labels_per_class=lpc, seed=seed, n_test=len(labels))
        cell["paired_test_data"]["labels"] = labels
        cell["paired_test_data"]["predictions"] = predictions
        accuracy = float(np.mean(np.asarray(predictions) == np.asarray(labels)))
        cell["metrics"] = {"accuracy": accuracy, "macro_f1": accuracy}
        directory = self.baseline_dir if condition == "correct" else self.new_dir
        path = analysis.result_path(directory, dataset, condition, lpc, seed)
        path.write_text(json.dumps(cell))

    def test_full_54_cell_fixture_produces_expected_paired_deltas(self) -> None:
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        # Fixed, seed-independent predictions per condition: correct is
        # perfect; mild misses one; strong misses two of the same rows.
        preds_by_condition = {
            "correct": [0, 1, 0, 1, 0, 1, 0, 1],
            "mild": [0, 1, 0, 1, 0, 1, 0, 0],
            "strong": [0, 1, 0, 1, 0, 1, 1, 0],
        }
        for dataset in analysis.DATASETS:
            for lpc in analysis.LABEL_BUDGETS:
                for seed in analysis.SEEDS:
                    for condition in analysis.CONDITIONS:
                        self._write_cell(
                            dataset, condition, lpc, seed, preds_by_condition[condition], labels
                        )

        completeness = analysis.scan_grid_completeness()
        self.assertEqual(completeness["missing_cells"], [])
        self.assertEqual(completeness["duplicate_cells"], [])
        self.assertEqual(completeness["n_found"], analysis.N_CELLS_EXPECTED)

        group = analysis.analyse_group("ag_news", 10)
        self.assertEqual(group["validation_status"], "ok")

        mild_vs_correct = group["comparisons"]["mild_minus_correct"]
        strong_vs_correct = group["comparisons"]["strong_minus_correct"]
        strong_vs_mild = group["comparisons"]["strong_minus_mild"]

        # correct=8/8=1.0, mild=7/8=0.875, strong=6/8=0.75 (identical every seed).
        self.assertAlmostEqual(mild_vs_correct["paired_delta"]["accuracy"]["mean"], -0.125)
        self.assertAlmostEqual(mild_vs_correct["paired_delta"]["accuracy"]["std"], 0.0)
        self.assertAlmostEqual(strong_vs_correct["paired_delta"]["accuracy"]["mean"], -0.25)
        self.assertAlmostEqual(strong_vs_mild["paired_delta"]["accuracy"]["mean"], -0.125)

        # correct wins every seed against both mild and strong.
        self.assertEqual(mild_vs_correct["seed_wins"]["correct"], 3)
        self.assertEqual(mild_vs_correct["seed_wins"]["mild"], 0)
        self.assertEqual(strong_vs_correct["seed_wins"]["correct"], 3)


# ---------------------------------------------------------------------------
# 21: the A.12 cached-logit report's numerics are unaffected by centralisation.
# ---------------------------------------------------------------------------
class PriorSensitivityCentralisationTests(unittest.TestCase):
    def test_skew_class_prior_matches_original_inline_formula(self) -> None:
        """The refactor must reproduce prior_sensitivity.py's original arithmetic bit-for-bit."""
        for num_classes, strength, dominant in (
            (4, 0.15, 0),
            (4, 0.50, 2),
            (5, 0.15, 3),
            (10, 0.50, 7),
        ):
            original = np.full(num_classes, (1.0 - strength) / num_classes, dtype=np.float64)
            original[dominant] += strength
            np.testing.assert_array_equal(
                skew_class_prior(num_classes, strength, dominant), original
            )

    def test_prior_sensitivity_pipeline_unchanged_by_centralisation(self) -> None:
        """run_dataset() gives bit-identical output whether skew_class_prior or the
        original private _skew_prior-equivalent formula is used, within one process
        (avoids unrelated cross-environment floating-point noise)."""
        import importlib

        prior_sensitivity = importlib.import_module("scripts.prior_sensitivity")

        def legacy_skew_prior(num_classes: int, strength: float, dominant: int) -> np.ndarray:
            prior = np.full(num_classes, (1.0 - strength) / num_classes, dtype=np.float64)
            prior[dominant] += strength
            return prior

        result_new = prior_sensitivity.run_dataset("ag_news")

        original_skew = prior_sensitivity.skew_class_prior
        prior_sensitivity.skew_class_prior = legacy_skew_prior
        try:
            result_legacy = prior_sensitivity.run_dataset("ag_news")
        finally:
            prior_sensitivity.skew_class_prior = original_skew

        self.assertEqual(result_new, result_legacy)


# ---------------------------------------------------------------------------
# CLI integration checks (fast: --dry-run never loads ModernBERT or the GPU).
# ---------------------------------------------------------------------------
class TrainStudentCliIntegrationTests(unittest.TestCase):
    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "scripts/train_student.py", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_dry_run_mild_ag_news_prints_exact_vector_and_writes_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run(
                [
                    "--dataset", "ag_news",
                    "--mode", "kd",
                    "--teacher", "multi-prior-scale-aligned-guided",
                    "--prior-condition", "mild",
                    "--prior-dominant-class", "0",
                    "--labels-per-class", "10",
                    "--seed", "42",
                    "--output-dir", tmp,
                    "--dry-run",
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("[0.3625, 0.2125, 0.2125, 0.2125]", proc.stdout)
            self.assertIn("Dry run OK", proc.stdout)
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_cli_rejects_mild_for_unrelated_teacher(self) -> None:
        proc = self._run(
            [
                "--dataset", "ag_news",
                "--mode", "kd",
                "--teacher", "multi-mean-guided",
                "--prior-condition", "mild",
                "--labels-per-class", "10",
                "--seed", "42",
                "--dry-run",
            ]
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("only defined for", proc.stderr)


if __name__ == "__main__":
    unittest.main()
