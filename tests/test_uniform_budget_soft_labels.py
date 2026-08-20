"""Focused tests for matched-budget uniform teacher soft labels.

Run:
    uv run python tests/test_uniform_budget_soft_labels.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student import soft_labels as soft_labels_mod
from src.student.soft_labels import (
    MULTI_UNIFORM_BUDGET_TEACHER,
    _align_confidence_scales_logits,
    _logits_to_probs,
    _match_class_prior_logits,
    _probs_to_kd_target,
    uniform_budget_teacher_soft_labels,
    uniform_class_prior,
)


TEACHERS = ("t0", "t1", "t2")


def _aligned_probs(stacked: np.ndarray, num_classes: int) -> np.ndarray:
    prior = uniform_class_prior(num_classes)
    aligned = _align_confidence_scales_logits(_match_class_prior_logits(stacked, prior))
    return _logits_to_probs(aligned, temperature=1.0)


def _with_synthetic_logits(stacked: np.ndarray):
    """Monkeypatch load_multi_teacher_logits to return *stacked*."""

    class _Patch:
        def __enter__(self):
            self._orig = soft_labels_mod.load_multi_teacher_logits
            soft_labels_mod.load_multi_teacher_logits = (
                lambda *args, **kwargs: stacked
            )
            return self

        def __exit__(self, exc_type, exc, tb):
            soft_labels_mod.load_multi_teacher_logits = self._orig
            return False

    return _Patch()


def test_output_is_proper_distribution() -> None:
    print("--- test_output_is_proper_distribution ---")
    rng = np.random.default_rng(0)
    n, c = 50, 4
    stacked = rng.normal(size=(3, n, c)).astype(np.float32)
    with _with_synthetic_logits(stacked):
        probs, meta = uniform_budget_teacher_soft_labels(
            TEACHERS, "toy", c, temperature=2.0, target_queries=1.4, allocation_seed=42
        )
    assert probs.shape == (n, c)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert meta["policy"] == "uniform_budget"
    print("  PASSED\n")


def test_budget_selects_exact_fraction() -> None:
    print("--- test_budget_selects_exact_fraction ---")
    rng = np.random.default_rng(1)
    n, c = 100, 3
    stacked = rng.normal(size=(3, n, c)).astype(np.float32)
    with _with_synthetic_logits(stacked):
        _, meta = uniform_budget_teacher_soft_labels(
            TEACHERS, "toy", c, temperature=2.0, target_queries=1.4, allocation_seed=7
        )
    assert meta["selected_count"] == 20
    assert abs(meta["realised_queries_per_sample"] - 1.4) < 1e-12
    assert abs(meta["selected_fraction"] - 0.2) < 1e-12
    print("  PASSED\n")


def test_allocation_seed_determinism() -> None:
    print("--- test_allocation_seed_determinism ---")
    rng = np.random.default_rng(2)
    n, c = 80, 4
    stacked = rng.normal(size=(3, n, c)).astype(np.float32)
    with _with_synthetic_logits(stacked):
        a1, m1 = uniform_budget_teacher_soft_labels(
            TEACHERS, "toy", c, temperature=2.0, target_queries=1.4, allocation_seed=123
        )
        a2, m2 = uniform_budget_teacher_soft_labels(
            TEACHERS, "toy", c, temperature=2.0, target_queries=1.4, allocation_seed=123
        )
        b1, mb = uniform_budget_teacher_soft_labels(
            TEACHERS, "toy", c, temperature=2.0, target_queries=1.4, allocation_seed=999
        )
    assert np.array_equal(a1, a2)
    assert m1["selected_count"] == m2["selected_count"]
    assert not np.array_equal(a1, b1)
    assert m1["allocation_seed"] == 123
    assert mb["allocation_seed"] == 999
    print("  PASSED\n")


def test_unselected_are_primary_only() -> None:
    print("--- test_unselected_are_primary_only ---")
    n, c = 40, 3
    # Extreme secondary teachers: if leaked into unselected rows they dominate.
    stacked = np.zeros((3, n, c), dtype=np.float32)
    stacked[0, :, 0] = 3.0
    stacked[0, :, 1] = 0.0
    stacked[0, :, 2] = -3.0
    stacked[1, :, :] = 50.0
    stacked[1, :, 2] = -50.0
    stacked[2, :, :] = -50.0
    stacked[2, :, 1] = 50.0

    p = _aligned_probs(stacked, c)
    expected_primary = _probs_to_kd_target(p[0], temperature=2.0)
    expected_committee = _probs_to_kd_target(p.mean(axis=0), temperature=2.0)

    with _with_synthetic_logits(stacked):
        probs, meta = uniform_budget_teacher_soft_labels(
            TEACHERS, "toy", c, temperature=2.0, target_queries=1.4, allocation_seed=42
        )

    selected_count = meta["selected_count"]
    rng = np.random.default_rng(42)
    selected = np.sort(rng.choice(n, size=selected_count, replace=False))
    mask = np.ones(n, dtype=bool)
    mask[selected] = False

    assert np.allclose(probs[mask], expected_primary[mask], atol=1e-5)
    assert np.allclose(probs[selected], expected_committee[selected], atol=1e-5)
    print("  PASSED\n")


def test_invalid_budgets_raise() -> None:
    print("--- test_invalid_budgets_raise ---")
    rng = np.random.default_rng(3)
    stacked = rng.normal(size=(3, 30, 2)).astype(np.float32)
    with _with_synthetic_logits(stacked):
        for bad_q in (0.5, 0.0, 3.1, 4.0):
            try:
                uniform_budget_teacher_soft_labels(
                    TEACHERS, "toy", 2, temperature=2.0, target_queries=bad_q
                )
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for q={bad_q}")
    print("  PASSED\n")


def test_no_label_argument() -> None:
    print("--- test_no_label_argument ---")
    params = inspect.signature(uniform_budget_teacher_soft_labels).parameters
    assert "labels" not in params
    assert "label" not in params
    assert MULTI_UNIFORM_BUDGET_TEACHER == "multi-uniform-budget-guided"
    print("  PASSED\n")


def test_result_metadata_schema() -> None:
    print("--- test_result_metadata_schema ---")
    rng = np.random.default_rng(4)
    n, c = 50, 4
    stacked = rng.normal(size=(3, n, c)).astype(np.float32)
    seed = 42
    with _with_synthetic_logits(stacked):
        probs, budget_policy = uniform_budget_teacher_soft_labels(
            TEACHERS,
            "toy",
            c,
            temperature=2.0,
            target_queries=1.4,
            allocation_seed=seed,
        )

    # Mirror the fields train_student records for RQ2 cells.
    result = {
        "dataset": "toy",
        "teacher": MULTI_UNIFORM_BUDGET_TEACHER,
        "labels_per_class": 10,
        "seed": seed,
        "queries_per_sample": budget_policy["realised_queries_per_sample"],
        "budget_policy": budget_policy,
        "label_accounting": {"policy": "prompt-inclusive-v1"},
        "dataset_identity": {"artifact_version": "large-test-v1"},
        "paired_test_data": {
            "sample_indices": list(range(5)),
            "labels": [0, 1, 2, 3, 0],
            "predictions": [0, 1, 2, 3, 1],
        },
        "metrics": {"accuracy": 0.8, "macro_f1": 0.75},
    }
    assert abs(result["queries_per_sample"] - 1.4) < 1e-12
    assert result["budget_policy"]["allocation_seed"] == seed
    assert result["budget_policy"]["selected_count"] == 10
    assert result["dataset_identity"]["artifact_version"] == "large-test-v1"
    assert len(result["paired_test_data"]["predictions"]) == 5
    # Filename encoding used by train_student for this experiment.
    expected_name = (
        f"toy_kd_{MULTI_UNIFORM_BUDGET_TEACHER}_q1p4_L10_s{seed}_a{seed}"
        f"_prompt-inclusive-v1.json"
    )
    assert expected_name == (
        "toy_kd_multi-uniform-budget-guided_q1p4_L10_s42_a42_prompt-inclusive-v1.json"
    )
    assert probs.shape == (n, c)
    print("  PASSED\n")


def main() -> None:
    test_output_is_proper_distribution()
    test_budget_selects_exact_fraction()
    test_allocation_seed_determinism()
    test_unselected_are_primary_only()
    test_invalid_budgets_raise()
    test_no_label_argument()
    test_result_metadata_schema()
    print("All uniform-budget soft-label tests passed.")


if __name__ == "__main__":
    main()
