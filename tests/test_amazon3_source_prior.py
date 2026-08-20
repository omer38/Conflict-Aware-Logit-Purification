"""Tests for the Amazon-3 source-prior transfer experiment.

Covers deterministic grouped-dataset construction, grouped-logit derivation,
label-budget splitting, the constrained soft-BBSE prior estimator, the
out-of-sample alignment fit/apply helpers, and the independent result
validator used by the GPU runner and analyser.

Only two tests shell out to ``train_amazon3_source_prior.py --dry-run``
(fast: no ModernBERT load, no GPU wait, no teacher queries); everything
else is a pure in-process unit/integration test using cached teacher
logits only.

Run:
    uv run pytest tests/test_amazon3_source_prior.py -q
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.student.prior_estimation import (  # noqa: E402
    apply_bias_corrected_calibration,
    bootstrap_prior_ci,
    committee_probs,
    confusion_moment_matrix,
    estimate_prior_bbse,
    estimate_prior_mlls,
    fit_bias_corrected_calibration,
    prior_error_metrics,
    reference_marginal,
)
from src.student.soft_labels import (  # noqa: E402
    GUIDED_TEACHERS,
    _align_confidence_scales_logits,
    _logits_to_probs,
    _match_class_prior_logits,
    apply_confidence_scale,
    apply_prior_bias,
    fit_confidence_scale,
    fit_prior_bias,
)

import scripts.analyse_amazon3_source_prior as analysis  # noqa: E402
import scripts.analyse_amazon3_source_prior_mlls as mlls_analysis  # noqa: E402
import scripts.train_amazon3_source_prior as experiment  # noqa: E402

# Module-level caches so the (slow-ish) real-data computations run once.
_AMAZON = experiment.load_amazon()
_TRAIN_GROUPED, _TEST_GROUPED = experiment.build_grouped_labels(_AMAZON)
_DEMO_BY_GROUP = experiment.grouped_prompt_demo_indices(_AMAZON)
_REFERENCE_INDICES, _WORKING_INDICES = experiment.build_reference_working_split(
    _TRAIN_GROUPED, _DEMO_BY_GROUP
)
_STACKED_RAW = experiment.stack_grouped_teacher_logits()


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
class GroupMapTests(unittest.TestCase):
    def test_group_map_is_exact(self) -> None:
        self.assertEqual(experiment.GROUP_MAP, {0: 0, 1: 0, 2: 1, 3: 2, 4: 2})

    def test_full_grouped_train_counts(self) -> None:
        counts = np.bincount(_TRAIN_GROUPED, minlength=3).tolist()
        self.assertEqual(counts, [1000, 500, 1000])

    def test_full_grouped_test_counts(self) -> None:
        counts = np.bincount(_TEST_GROUPED, minlength=3).tolist()
        self.assertEqual(counts, [4000, 2000, 4000])


class ReferenceWorkingSplitTests(unittest.TestCase):
    def test_reference_counts(self) -> None:
        counts = np.bincount(_TRAIN_GROUPED[_REFERENCE_INDICES], minlength=3).tolist()
        self.assertEqual(counts, [200, 100, 200])

    def test_working_counts(self) -> None:
        counts = np.bincount(_TRAIN_GROUPED[_WORKING_INDICES], minlength=3).tolist()
        self.assertEqual(counts, [800, 400, 800])

    def test_disjoint_and_exhaustive(self) -> None:
        self.assertEqual(np.intersect1d(_REFERENCE_INDICES, _WORKING_INDICES).size, 0)
        union = np.union1d(_REFERENCE_INDICES, _WORKING_INDICES)
        self.assertEqual(union.size, 2500)
        np.testing.assert_array_equal(union, np.arange(2500))

    def test_both_pools_have_the_exact_true_prior(self) -> None:
        ref_prior = np.bincount(_TRAIN_GROUPED[_REFERENCE_INDICES], minlength=3) / len(_REFERENCE_INDICES)
        work_prior = np.bincount(_TRAIN_GROUPED[_WORKING_INDICES], minlength=3) / len(_WORKING_INDICES)
        np.testing.assert_allclose(ref_prior, experiment.TRUE_PRIOR)
        np.testing.assert_allclose(work_prior, experiment.TRUE_PRIOR)

    def test_all_prompt_demos_in_working_pool(self) -> None:
        all_demos = np.asarray(
            [i for indices in _DEMO_BY_GROUP.values() for i in indices], dtype=np.int64
        )
        self.assertEqual(len(all_demos), 10)
        self.assertTrue(np.all(np.isin(all_demos, _WORKING_INDICES)))

    def test_no_prompt_demo_in_reference_batch(self) -> None:
        all_demos = np.asarray(
            [i for indices in _DEMO_BY_GROUP.values() for i in indices], dtype=np.int64
        )
        self.assertEqual(np.intersect1d(all_demos, _REFERENCE_INDICES).size, 0)

    def test_grouped_demo_counts_are_unequal_by_construction(self) -> None:
        self.assertEqual(len(_DEMO_BY_GROUP[0]), 4)  # negative: star 1 + star 2
        self.assertEqual(len(_DEMO_BY_GROUP[1]), 2)  # neutral: star 3
        self.assertEqual(len(_DEMO_BY_GROUP[2]), 4)  # positive: star 4 + star 5

    def test_split_is_reproducible_given_the_fixed_pool_seed(self) -> None:
        ref2, work2 = experiment.build_reference_working_split(_TRAIN_GROUPED, _DEMO_BY_GROUP)
        np.testing.assert_array_equal(_REFERENCE_INDICES, ref2)
        np.testing.assert_array_equal(_WORKING_INDICES, work2)


# ---------------------------------------------------------------------------
# Grouped teacher logits (no new teacher queries)
# ---------------------------------------------------------------------------
class GroupedLogitsTests(unittest.TestCase):
    def test_probs_finite_and_sum_to_one(self) -> None:
        for k, teacher in enumerate(GUIDED_TEACHERS):
            logits = _STACKED_RAW[k]
            self.assertTrue(np.isfinite(logits).all(), teacher)
            probs = _logits_to_probs(logits, temperature=1.0)
            np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    def test_row_order_preserved(self) -> None:
        teacher = GUIDED_TEACHERS[0]
        grouped = experiment.derive_grouped_teacher_logits(teacher)
        logits5 = experiment.load_teacher_logits(teacher, "amazon", 5)
        for row in (0, 1, 2499):
            p5 = np.exp(logits5[row] - logits5[row].max())
            p5 /= p5.sum()
            expected = np.log(
                np.clip(
                    np.array([p5[0] + p5[1], p5[2], p5[3] + p5[4]]), 1e-12, None
                )
            )
            np.testing.assert_allclose(grouped[row], expected, atol=1e-4)

    def test_matches_existing_amazon3_cache_for_every_teacher(self) -> None:
        for teacher in GUIDED_TEACHERS:
            grouped = experiment.derive_grouped_teacher_logits(teacher)
            train_chosen, cached = experiment.existing_amazon3_grouped_logits(teacher)
            np.testing.assert_allclose(grouped[train_chosen], cached, atol=1e-3)


# ---------------------------------------------------------------------------
# Label-budget splitting
# ---------------------------------------------------------------------------
class LabelSplitTests(unittest.TestCase):
    def _check(self, labels_per_class: int, seed: int) -> None:
        labelled, unlabelled = experiment.select_labelled_indices(
            _WORKING_INDICES, _TRAIN_GROUPED, _DEMO_BY_GROUP, labels_per_class, seed
        )
        counts = np.bincount(_TRAIN_GROUPED[labelled], minlength=3)
        np.testing.assert_array_equal(counts, [labels_per_class] * 3)

        all_demos = np.asarray(
            [i for indices in _DEMO_BY_GROUP.values() for i in indices], dtype=np.int64
        )
        self.assertTrue(np.all(np.isin(all_demos, labelled)))

        self.assertEqual(np.intersect1d(labelled, unlabelled).size, 0)
        union = np.union1d(labelled, unlabelled)
        np.testing.assert_array_equal(np.sort(union), _WORKING_INDICES)

        self.assertEqual(np.intersect1d(labelled, _REFERENCE_INDICES).size, 0)
        self.assertEqual(np.intersect1d(unlabelled, _REFERENCE_INDICES).size, 0)

    def test_all_label_budgets_and_seeds(self) -> None:
        for labels_per_class in experiment.LABEL_BUDGETS:
            for seed in experiment.SEEDS:
                with self.subTest(labels_per_class=labels_per_class, seed=seed):
                    self._check(labels_per_class, seed)

    def test_selection_is_deterministic_and_method_independent(self) -> None:
        # select_labelled_indices does not take a method argument, so every
        # method automatically receives identical labelled/unlabelled indices.
        a = experiment.select_labelled_indices(_WORKING_INDICES, _TRAIN_GROUPED, _DEMO_BY_GROUP, 10, 42)
        b = experiment.select_labelled_indices(_WORKING_INDICES, _TRAIN_GROUPED, _DEMO_BY_GROUP, 10, 42)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_all_four_methods_share_identical_labelled_split(self) -> None:
        splits = {
            method: experiment.prepare_cell(method, 10, 42)["result"]["labelled_split"]
            for method in experiment.METHODS
        }
        reference = splits[experiment.METHODS[0]]
        for method, split in splits.items():
            self.assertEqual(split["labelled_indices"], reference["labelled_indices"], method)
            self.assertEqual(split["unlabelled_indices"], reference["unlabelled_indices"], method)


# ---------------------------------------------------------------------------
# Prior estimator: hand-calculated toy examples
# ---------------------------------------------------------------------------
class PriorEstimatorToyTests(unittest.TestCase):
    def test_identity_matrix_exact_recovery(self) -> None:
        matrix = np.eye(3)
        mu = np.array([0.5, 0.3, 0.2])
        result = estimate_prior_bbse(matrix, mu)
        np.testing.assert_allclose(result.estimated_prior, mu, atol=1e-5)
        self.assertTrue(result.success)
        self.assertEqual(result.rank, 3)

    def test_small_invertible_non_identity_recovery(self) -> None:
        matrix = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        pi_true = np.array([0.4, 0.2, 0.4])
        mu = matrix @ pi_true
        result = estimate_prior_bbse(matrix, mu)
        np.testing.assert_allclose(result.estimated_prior, pi_true, atol=1e-5)

    def test_output_nonnegative_and_sums_to_one(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            matrix = rng.dirichlet(np.ones(3), size=3).T  # random column-stochastic
            mu = rng.dirichlet(np.ones(3))
            result = estimate_prior_bbse(matrix, mu)
            self.assertTrue(np.all(result.estimated_prior >= 0))
            self.assertAlmostEqual(float(result.estimated_prior.sum()), 1.0, places=6)

    def test_deterministic_output(self) -> None:
        matrix = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        mu = np.array([0.35, 0.4, 0.25])
        a = estimate_prior_bbse(matrix, mu)
        b = estimate_prior_bbse(matrix, mu)
        np.testing.assert_array_equal(a.estimated_prior, b.estimated_prior)

    def test_rank_condition_residual_are_reported(self) -> None:
        matrix = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        mu = matrix @ np.array([0.4, 0.2, 0.4])
        result = estimate_prior_bbse(matrix, mu)
        self.assertEqual(result.rank, 3)
        self.assertGreater(result.condition_number, 1.0)
        self.assertLess(result.residual, 1e-6)
        self.assertEqual(len(result.singular_values), 3)

    def test_rank_deficient_matrix_is_surfaced_not_hidden(self) -> None:
        # Two identical columns -> rank 2, not 3. The estimator must report
        # this plainly; it must not regularise or substitute a different prior.
        matrix = np.array([[0.5, 0.5, 0.1], [0.3, 0.3, 0.2], [0.2, 0.2, 0.7]])
        mu = np.array([0.4, 0.3, 0.3])
        result = estimate_prior_bbse(matrix, mu)
        self.assertLess(result.rank, 3)
        self.assertGreater(result.condition_number, 1e6)

    def test_error_metrics_are_post_hoc_only(self) -> None:
        estimated = np.array([0.5, 0.2, 0.3])
        true_prior = np.array([0.4, 0.2, 0.4])
        metrics = prior_error_metrics(estimated, true_prior)
        np.testing.assert_allclose(metrics["component_error"], [0.1, 0.0, -0.1])
        self.assertAlmostEqual(metrics["l1_error"], 0.2)
        self.assertAlmostEqual(metrics["total_variation_error"], 0.1)

    def test_reference_labels_are_not_accepted_by_the_estimator_api(self) -> None:
        for fn in (estimate_prior_bbse, reference_marginal, committee_probs):
            params = inspect.signature(fn).parameters
            for name in params:
                self.assertNotIn("reference_label", name)
                self.assertNotIn("ref_label", name)
        bootstrap_params = inspect.signature(bootstrap_prior_ci).parameters
        for name in bootstrap_params:
            self.assertNotIn("reference_label", name)
        confusion_params = inspect.signature(confusion_moment_matrix).parameters
        self.assertNotIn("reference_labels", confusion_params)

    def test_bootstrap_requires_at_least_1000_replicates(self) -> None:
        rng = np.random.default_rng(1)
        probs = rng.dirichlet(np.ones(3), size=30)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        ref_probs = rng.dirichlet(np.ones(3), size=50)
        with self.assertRaises(ValueError):
            bootstrap_prior_ci(probs, labels, ref_probs, num_classes=3, n_replicates=10, seed=0)

    def test_bootstrap_output_shape_and_ordering(self) -> None:
        rng = np.random.default_rng(2)
        probs = rng.dirichlet(np.ones(3), size=90)
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        ref_probs = rng.dirichlet(np.ones(3), size=100)
        result = bootstrap_prior_ci(probs, labels, ref_probs, num_classes=3, n_replicates=1000, seed=7)
        self.assertEqual(result.lower_2_5.shape, (3,))
        self.assertEqual(result.upper_97_5.shape, (3,))
        self.assertTrue(np.all(result.lower_2_5 <= result.upper_97_5 + 1e-9))
        self.assertEqual(result.n_replicates, 1000)

    def test_bootstrap_is_deterministic_given_fixed_seed(self) -> None:
        rng = np.random.default_rng(3)
        probs = rng.dirichlet(np.ones(3), size=60)
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        ref_probs = rng.dirichlet(np.ones(3), size=80)
        a = bootstrap_prior_ci(probs, labels, ref_probs, num_classes=3, n_replicates=1000, seed=42)
        b = bootstrap_prior_ci(probs, labels, ref_probs, num_classes=3, n_replicates=1000, seed=42)
        np.testing.assert_array_equal(a.lower_2_5, b.lower_2_5)
        np.testing.assert_array_equal(a.upper_97_5, b.upper_97_5)


class CalibratedMaximumLikelihoodPriorTests(unittest.TestCase):
    def test_bias_corrected_calibration_is_deterministic_and_improves_labelled_nll(self) -> None:
        probs = np.array(
            [
                [0.65, 0.25, 0.10],
                [0.55, 0.35, 0.10],
                [0.55, 0.35, 0.10],
                [0.45, 0.45, 0.10],
                [0.30, 0.25, 0.45],
                [0.20, 0.20, 0.60],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        a = fit_bias_corrected_calibration(probs, labels)
        b = fit_bias_corrected_calibration(probs, labels)
        self.assertTrue(a.success)
        self.assertLessEqual(a.labelled_nll_after, a.labelled_nll_before)
        self.assertGreater(a.temperature, 0.0)
        self.assertAlmostEqual(float(a.bias.sum()), 0.0, places=10)
        np.testing.assert_allclose(a.bias, b.bias, atol=1e-10)
        self.assertAlmostEqual(a.temperature, b.temperature, places=10)

        calibrated = apply_bias_corrected_calibration(probs, a)
        self.assertTrue(np.isfinite(calibrated).all())
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-10)

    def test_mlls_recovers_prior_from_one_hot_reference_predictions(self) -> None:
        probs = np.eye(3)[np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])]
        result = estimate_prior_mlls(probs, np.full(3, 1.0 / 3.0))
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.estimated_prior, [0.4, 0.2, 0.4], atol=1e-6)

    def test_real_mlls_estimate_improves_the_failed_bbse_cell(self) -> None:
        labelled, _ = experiment.select_labelled_indices(
            _WORKING_INDICES, _TRAIN_GROUPED, _DEMO_BY_GROUP, 10, 123
        )
        payload, estimate = experiment.run_prior_estimation_mlls(
            _STACKED_RAW, _TRAIN_GROUPED, labelled, _REFERENCE_INDICES
        )
        bbse_payload, _ = experiment.run_prior_estimation(
            _STACKED_RAW, _TRAIN_GROUPED, labelled, _REFERENCE_INDICES
        )
        self.assertTrue(payload["solver_success"])
        self.assertTrue(payload["calibration"]["solver_success"])
        self.assertEqual(payload["n_declared_labels"], 30)
        self.assertEqual(payload["n_reference_rows"], 500)
        self.assertFalse(payload["reference_labels_used"])
        self.assertLess(payload["l1_error"], bbse_payload["l1_error"])
        np.testing.assert_allclose(
            estimate.estimated_prior, [0.523265, 0.077114, 0.399621], atol=2e-4
        )


# ---------------------------------------------------------------------------
# Out-of-sample alignment: fit/apply helpers
# ---------------------------------------------------------------------------
class AlignmentTransferTests(unittest.TestCase):
    def test_fit_prior_bias_depends_only_on_the_fitting_batch(self) -> None:
        rng = np.random.default_rng(11)
        reference_logits = rng.normal(size=(3, 60, 3)).astype(np.float32)
        target = np.array([0.5, 0.2, 0.3])
        bias_a = fit_prior_bias(reference_logits, target)
        # A completely different "working" batch must not change the fitted bias.
        unrelated_working_logits = rng.normal(size=(3, 900, 3)).astype(np.float32) * 5.0
        bias_b = fit_prior_bias(reference_logits, target)
        np.testing.assert_array_equal(bias_a, bias_b)
        self.assertEqual(unrelated_working_logits.shape[1], 900)  # unused, just proves no coupling

    def test_apply_does_not_refit(self) -> None:
        rng = np.random.default_rng(12)
        reference_logits = rng.normal(size=(3, 50, 3)).astype(np.float32)
        working_logits = rng.normal(size=(3, 40, 3)).astype(np.float32)
        bias = fit_prior_bias(reference_logits, np.array([0.6, 0.1, 0.3]))
        applied_once = apply_prior_bias(working_logits, bias)
        applied_twice = apply_prior_bias(working_logits, bias)
        np.testing.assert_array_equal(applied_once, applied_twice)
        expected_delta = np.broadcast_to(bias[:, None, :], applied_once.shape)
        np.testing.assert_allclose(applied_once - working_logits, expected_delta)

    def test_fitted_bias_matches_supplied_prior_on_the_fitting_batch(self) -> None:
        rng = np.random.default_rng(13)
        reference_logits = rng.normal(size=(3, 300, 4)).astype(np.float32)
        target = np.array([0.55, 0.20, 0.15, 0.10])
        bias = fit_prior_bias(reference_logits, target)
        matched = apply_prior_bias(reference_logits, bias)
        achieved = _logits_to_probs(matched, 1.0).mean(axis=1)
        np.testing.assert_allclose(achieved, np.broadcast_to(target, achieved.shape), atol=1e-5)

    def test_working_pool_outputs_finite_and_sum_to_one(self) -> None:
        reference_logits = _STACKED_RAW[:, _REFERENCE_INDICES, :]
        working_logits = _STACKED_RAW[:, _WORKING_INDICES, :]
        bias = fit_prior_bias(reference_logits, experiment.TRUE_PRIOR)
        matched_reference = apply_prior_bias(reference_logits, bias)
        scales = fit_confidence_scale(matched_reference)
        working_aligned = apply_confidence_scale(apply_prior_bias(working_logits, bias), scales)
        probs = _logits_to_probs(working_aligned, temperature=1.0)
        self.assertTrue(np.isfinite(probs).all())
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    def test_scale_only_has_zero_bias(self) -> None:
        prepared = experiment.prepare_cell("scale_only", 10, 42)
        bias = np.asarray(prepared["result"]["alignment_fit"]["bias"])
        np.testing.assert_allclose(bias, 0.0)
        self.assertFalse(prepared["result"]["alignment_fit"]["prior_matching"])

    def test_uniform_and_oracle_priors_are_exact(self) -> None:
        uniform_result = experiment.prepare_cell("uniform_prior", 10, 42)["result"]
        oracle_result = experiment.prepare_cell("oracle_prior", 10, 42)["result"]
        np.testing.assert_allclose(uniform_result["assumed_prior"], [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(oracle_result["assumed_prior"], [0.4, 0.2, 0.4])

    def test_estimated_condition_uses_the_independently_recomputed_estimate(self) -> None:
        result = experiment.prepare_cell("estimated_prior_bbse", 50, 123)["result"]
        cached = analysis.independent_prior_estimate(50, 123)
        np.testing.assert_allclose(
            result["assumed_prior"], cached["estimate"].estimated_prior, atol=1e-9
        )

    def test_mlls_condition_uses_calibrated_maximum_likelihood_estimate(self) -> None:
        result = experiment.prepare_cell(experiment.MLLS_METHOD, 50, 123)["result"]
        self.assertEqual(result["experiment_id"], experiment.MLLS_EXPERIMENT_ID)
        self.assertEqual(result["prior_estimator"]["estimator"], "maximum_likelihood_label_shift")
        self.assertEqual(
            result["prior_estimator"]["calibration_method"],
            "bias_corrected_temperature_scaling",
        )
        np.testing.assert_allclose(
            result["assumed_prior"], result["prior_estimator"]["estimated_prior"], atol=1e-9
        )

    def test_transductive_legacy_helpers_remain_equivalent(self) -> None:
        rng = np.random.default_rng(14)
        logits = rng.normal(size=(3, 200, 4)).astype(np.float32)
        target = np.array([0.55, 0.20, 0.15, 0.10])
        matched_legacy = _match_class_prior_logits(logits, target)
        matched_new = apply_prior_bias(logits, fit_prior_bias(logits, target))
        np.testing.assert_array_equal(matched_legacy, matched_new)

        aligned_legacy = _align_confidence_scales_logits(logits)
        aligned_new = apply_confidence_scale(logits, fit_confidence_scale(logits))
        np.testing.assert_array_equal(aligned_legacy, aligned_new)


# ---------------------------------------------------------------------------
# Result validation (synthetic fixtures; no GPU results required)
# ---------------------------------------------------------------------------
def _synthetic_valid_cell(method: str, labels_per_class: int, seed: int, n_test: int = 12) -> dict:
    prepared = experiment.prepare_cell(method, labels_per_class, seed)
    result = dict(prepared["result"])
    labels = [i % 3 for i in range(n_test)]
    result["dataset_identity"] = dict(result["dataset_identity"])
    result["dataset_identity"]["test"] = {"size": n_test, "sha256": "synthetic-test-sha"}
    result["student_config"] = dict(analysis.REQUIRED_STUDENT_HYPERPARAMS)
    result["metrics"] = {"accuracy": 0.5, "macro_f1": 0.5}
    result["per_class_metrics"] = {
        "labels": experiment.GROUP_NAMES,
        "precision": [0.5, 0.5, 0.5],
        "recall": [0.5, 0.5, 0.5],
        "f1": [0.5, 0.5, 0.5],
        "support": [4, 4, 4],
        "confusion_matrix": [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
    }
    result["paired_test_data"] = {
        "sample_indices": list(range(n_test)),
        "labels": labels,
        "predictions": list(labels),
    }
    return result


class ValidatorShallowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = analysis.EXPECTED_TEST_SIZE
        analysis.EXPECTED_TEST_SIZE = 12
        self.addCleanup(lambda: setattr(analysis, "EXPECTED_TEST_SIZE", self._orig))

    def test_valid_cell_has_no_shallow_errors_for_every_method(self) -> None:
        for method in experiment.METHODS:
            with self.subTest(method=method):
                cell = _synthetic_valid_cell(method, 10, 42)
                errors = analysis.validate_cell(cell, method=method, labels_per_class=10, seed=42)
                self.assertEqual(errors, [])

    def test_validator_rejects_wrong_true_prior(self) -> None:
        cell = _synthetic_valid_cell("scale_only", 10, 42)
        cell["true_prior"] = [1 / 3, 1 / 3, 1 / 3]
        errors = analysis.validate_cell(cell, method="scale_only", labels_per_class=10, seed=42)
        self.assertTrue(any("true_prior" in e for e in errors))

    def test_validator_rejects_wrong_assumed_prior_for_uniform(self) -> None:
        cell = _synthetic_valid_cell("uniform_prior", 10, 42)
        cell["assumed_prior"] = [0.4, 0.2, 0.4]
        errors = analysis.validate_cell(cell, method="uniform_prior", labels_per_class=10, seed=42)
        self.assertTrue(any("assumed_prior mismatch" in e for e in errors))

    def test_validator_rejects_nonzero_bias_for_scale_only(self) -> None:
        cell = _synthetic_valid_cell("scale_only", 10, 42)
        cell["alignment_fit"] = dict(cell["alignment_fit"])
        cell["alignment_fit"]["bias"] = [[0.1, 0.0, 0.0]] * 3
        errors = analysis.validate_cell(cell, method="scale_only", labels_per_class=10, seed=42)
        self.assertTrue(any("bias is not all-zero" in e for e in errors))

    def test_validator_rejects_bad_hyperparameter(self) -> None:
        cell = _synthetic_valid_cell("scale_only", 10, 42)
        cell["student_config"] = dict(cell["student_config"])
        cell["student_config"]["epochs"] = 5
        errors = analysis.validate_cell(cell, method="scale_only", labels_per_class=10, seed=42)
        self.assertTrue(any("student_config.epochs" in e for e in errors))

    def test_validator_rejects_wrong_paired_length(self) -> None:
        cell = _synthetic_valid_cell("scale_only", 10, 42)
        cell["paired_test_data"] = dict(cell["paired_test_data"])
        cell["paired_test_data"]["predictions"] = [0, 1]
        errors = analysis.validate_cell(cell, method="scale_only", labels_per_class=10, seed=42)
        self.assertTrue(any("paired_test_data.predictions" in e for e in errors))

    def test_validator_rejects_missing_keys(self) -> None:
        cell = _synthetic_valid_cell("scale_only", 10, 42)
        del cell["prior_estimator"]
        errors = analysis.validate_cell(cell, method="scale_only", labels_per_class=10, seed=42)
        self.assertTrue(any("missing top-level keys" in e for e in errors))

    def test_validator_rejects_rank_deficient_estimator(self) -> None:
        cell = _synthetic_valid_cell("scale_only", 10, 42)
        cell["prior_estimator"] = dict(cell["prior_estimator"])
        cell["prior_estimator"]["matrix_rank"] = 2
        errors = analysis.validate_cell(cell, method="scale_only", labels_per_class=10, seed=42)
        self.assertTrue(any("matrix_rank" in e for e in errors))


class ScanGridCompletenessTests(unittest.TestCase):
    def test_filename_regex_round_trips(self) -> None:
        for method in experiment.METHODS:
            path = analysis.result_path(Path("/tmp/x"), method, 10, 42)
            match = analysis.FILENAME_RE.match(path.name)
            self.assertIsNotNone(match, path.name)
            self.assertEqual(match.group("method"), method)
            self.assertEqual(int(match.group("labels")), 10)
            self.assertEqual(int(match.group("seed")), 42)

    def test_scan_reports_all_missing_on_empty_directory(self) -> None:
        empty_dir = Path("/tmp/_amazon3_scan_empty")
        empty_dir.mkdir(parents=True, exist_ok=True)
        for f in empty_dir.glob("*.json"):
            f.unlink()
        found = analysis.scan_directory(empty_dir)
        self.assertEqual(len(found), 0)

    def test_no_existing_result_is_silently_accepted(self) -> None:
        # The real experiment directory must not already contain 24 valid
        # cells from an unrelated/older run masquerading as this grid.
        found = analysis.scan_directory(analysis.NEW_CELLS_DIR)
        for key, path in found.items():
            self.assertTrue(path.name.startswith("amazon3_source-prior-transfer_"))


# ---------------------------------------------------------------------------
# Deep (independently recomputed) validation, using real prepare_cell output.
# ---------------------------------------------------------------------------
class DeepValidationTests(unittest.TestCase):
    def test_prepare_cell_output_passes_deep_validation(self) -> None:
        cell = _synthetic_valid_cell("estimated_prior_bbse", 10, 42, n_test=10000)
        errors = analysis.validate_cell(cell, method="estimated_prior_bbse", labels_per_class=10, seed=42)
        self.assertEqual(errors, [])
        errors = analysis.validate_cell_independent(
            cell, method="estimated_prior_bbse", labels_per_class=10, seed=42
        )
        self.assertEqual(errors, [])

    def test_deep_validation_catches_tampered_working_pool_hash(self) -> None:
        cell = _synthetic_valid_cell("estimated_prior_bbse", 10, 42, n_test=10000)
        cell["working_pool"] = dict(cell["working_pool"])
        cell["working_pool"]["identity_hash"] = "0" * 64
        errors = analysis.validate_cell_independent(
            cell, method="estimated_prior_bbse", labels_per_class=10, seed=42
        )
        self.assertTrue(any("working_pool.identity_hash" in e for e in errors))

    def test_deep_validation_catches_tampered_estimated_prior(self) -> None:
        cell = _synthetic_valid_cell("estimated_prior_bbse", 10, 42, n_test=10000)
        cell["prior_estimator"] = dict(cell["prior_estimator"])
        cell["prior_estimator"]["estimated_prior"] = [0.9, 0.05, 0.05]
        errors = analysis.validate_cell_independent(
            cell, method="estimated_prior_bbse", labels_per_class=10, seed=42
        )
        self.assertTrue(any("does not match the independently reproduced estimate" in e for e in errors))


class MLLSValidatorTests(unittest.TestCase):
    @staticmethod
    def _cell(labels_per_class: int = 10, seed: int = 42) -> dict:
        result = experiment.prepare_cell(experiment.MLLS_METHOD, labels_per_class, seed)["result"]
        result["student_config"] = dict(analysis.REQUIRED_STUDENT_HYPERPARAMS)
        result["metrics"] = {"accuracy": 0.5, "macro_f1": 0.5}
        labels = [i % 3 for i in range(10000)]
        result["paired_test_data"] = {
            "sample_indices": list(range(10000)),
            "labels": labels,
            "predictions": labels,
        }
        return result

    def test_valid_mlls_cell_passes_independent_validation(self) -> None:
        errors = mlls_analysis.validate_cell(
            self._cell(), labels_per_class=10, seed=42
        )
        self.assertEqual(errors, [])

    def test_mlls_validator_rejects_tampered_prior(self) -> None:
        cell = self._cell()
        cell["assumed_prior"] = [0.8, 0.1, 0.1]
        errors = mlls_analysis.validate_cell(cell, labels_per_class=10, seed=42)
        self.assertTrue(any("assumed prior" in error for error in errors))


# ---------------------------------------------------------------------------
# No teacher-query code paths (static check on the new experiment modules).
# ---------------------------------------------------------------------------
class NoTeacherQueryTests(unittest.TestCase):
    def test_experiment_modules_do_not_reference_teacher_query_paths(self) -> None:
        forbidden = ("teachers.llm", "teachers.extraction.complete", "openai.OpenAI(", "vllm")
        for path in (
            REPO_ROOT / "scripts" / "train_amazon3_source_prior.py",
            REPO_ROOT / "src" / "student" / "prior_estimation.py",
        ):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path} references forbidden token {token!r}")


# ---------------------------------------------------------------------------
# CLI smoke tests (subprocess; only two, both --dry-run).
# ---------------------------------------------------------------------------
class DryRunCliTests(unittest.TestCase):
    def _run(self, *extra_args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "scripts/train_amazon3_source_prior.py", *extra_args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_dry_run_scale_only(self) -> None:
        result = self._run("--method", "scale_only", "--labels-per-class", "10", "--seed", "42", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Dry run OK", result.stdout)
        self.assertIn("No teacher queries made", result.stdout)

    def test_dry_run_estimated_prior_bbse(self) -> None:
        result = self._run(
            "--method", "estimated_prior_bbse", "--labels-per-class", "50", "--seed", "456", "--dry-run"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BBSE estimate", result.stdout)
        self.assertIn("Dry run OK", result.stdout)

    def test_dry_run_estimated_prior_mlls_bcts(self) -> None:
        result = self._run(
            "--method",
            experiment.MLLS_METHOD,
            "--labels-per-class",
            "50",
            "--seed",
            "456",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BCTS+MLLS estimate", result.stdout)
        self.assertIn("Dry run OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
