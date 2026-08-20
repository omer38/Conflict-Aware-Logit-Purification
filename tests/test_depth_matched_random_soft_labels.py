"""Focused CPU-only tests for depth-matched random teacher allocation."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student.soft_labels import (
    MULTI_DEPTH_MATCHED_RANDOM_TEACHER,
    _cascade_allocation_masks,
    _logits_to_probs,
    _probs_to_kd_target,
    depth_matched_random_teacher_soft_labels,
)


TEACHERS = ("qwen", "gemma", "mistral")


def _synthetic_logits() -> np.ndarray:
    rng = np.random.default_rng(17)
    logits = rng.normal(size=(3, 40, 4)).astype(np.float32)
    logits[0] *= np.linspace(2.5, 0.25, 40)[:, None]
    return logits


def _run(
    stacked: np.ndarray,
    *,
    target_queries: float = 1.4,
    allocation_seed: int = 42,
    temperature: float = 2.0,
) -> tuple[np.ndarray, dict]:
    identity = lambda values, *args, **kwargs: values
    with (
        patch(
            "src.student.soft_labels.load_multi_teacher_logits",
            return_value=stacked,
        ),
        patch("src.student.soft_labels._match_class_prior_logits", side_effect=identity),
        patch(
            "src.student.soft_labels._align_confidence_scales_logits",
            side_effect=identity,
        ),
    ):
        return depth_matched_random_teacher_soft_labels(
            TEACHERS,
            "synthetic",
            num_classes=stacked.shape[-1],
            temperature=temperature,
            target_queries=target_queries,
            allocation_seed=allocation_seed,
        )


def test_distribution_depth_counts_and_realised_rate() -> None:
    stacked = _synthetic_logits()
    targets, metadata = _run(stacked)
    p = _logits_to_probs(stacked, temperature=1.0)
    ask2, ask3, cascade_realised = _cascade_allocation_masks(p, 1.4)
    expected_counts = (
        int((~ask2).sum()),
        int((ask2 & ~ask3).sum()),
        int(ask3.sum()),
    )

    assert targets.shape == (40, 4)
    assert np.isfinite(targets).all()
    assert (targets >= 0).all()
    assert np.allclose(targets.sum(axis=1), 1.0, atol=1e-6)
    assert (
        metadata["depth_1_count"],
        metadata["depth_2_count"],
        metadata["depth_3_count"],
    ) == expected_counts
    n1, n2, n3 = expected_counts
    count_realised = (n1 + 2 * n2 + 3 * n3) / 40
    assert np.isclose(
        metadata["realised_queries_per_sample"],
        count_realised,
        rtol=0.0,
        atol=1e-12,
    )
    assert metadata["realised_queries_per_sample"] == cascade_realised
    assert n1 + n2 + n3 == metadata["n_samples"] == 40


def test_seeded_assignment_and_exact_targets() -> None:
    stacked = _synthetic_logits()
    p = _logits_to_probs(stacked, temperature=1.0)
    targets_a, metadata_a = _run(stacked, allocation_seed=123)
    targets_a_repeat, metadata_a_repeat = _run(stacked, allocation_seed=123)
    targets_b, metadata_b = _run(stacked, allocation_seed=999)

    assert np.array_equal(targets_a, targets_a_repeat)
    assert metadata_a == metadata_a_repeat
    assert not np.array_equal(targets_a, targets_b)
    assert metadata_a["depth_1_count"] == metadata_b["depth_1_count"]
    assert metadata_a["depth_2_count"] == metadata_b["depth_2_count"]
    assert metadata_a["depth_3_count"] == metadata_b["depth_3_count"]

    n2 = metadata_a["depth_2_count"]
    n3 = metadata_a["depth_3_count"]
    permutation = np.random.default_rng(123).permutation(metadata_a["n_samples"])
    depth3 = permutation[:n3]
    depth2 = permutation[n3:n3 + n2]
    depth1_mask = np.ones(metadata_a["n_samples"], dtype=bool)
    depth1_mask[depth2] = False
    depth1_mask[depth3] = False

    assert np.intersect1d(depth2, depth3).size == 0
    expected_depth1 = _probs_to_kd_target(p[0], temperature=2.0)
    expected_depth2 = _probs_to_kd_target((p[0] + p[1]) / 2.0, temperature=2.0)
    expected_depth3 = _probs_to_kd_target(
        (p[0] + p[1] + p[2]) / 3.0,
        temperature=2.0,
    )
    assert np.allclose(targets_a[depth1_mask], expected_depth1[depth1_mask])
    assert np.allclose(targets_a[depth2], expected_depth2[depth2])
    assert np.allclose(targets_a[depth3], expected_depth3[depth3])


def test_metadata_schema_and_label_independence() -> None:
    _, metadata = _run(_synthetic_logits())
    assert MULTI_DEPTH_MATCHED_RANDOM_TEACHER == (
        "multi-depth-matched-random-guided"
    )
    assert metadata["policy"] == "depth_matched_random"
    assert metadata["source_policy"] == "entropy_cascade"
    assert metadata["primary_teacher"] == TEACHERS[0]
    assert metadata["teachers"] == list(TEACHERS)
    assert "indices" not in metadata

    parameters = inspect.signature(
        depth_matched_random_teacher_soft_labels
    ).parameters
    assert "labels" not in parameters
    assert "gold_labels" not in parameters


def test_invalid_budgets_and_committee_size_raise() -> None:
    stacked = _synthetic_logits()
    for bad_q in (0.9, 3.1, np.nan, np.inf):
        try:
            _run(stacked, target_queries=bad_q)
        except ValueError as exc:
            assert "between 1 and 3" in str(exc)
        else:
            raise AssertionError(f"Expected invalid target_queries={bad_q} to fail")

    try:
        depth_matched_random_teacher_soft_labels(
            TEACHERS[:2],
            "synthetic",
            num_classes=4,
            temperature=2.0,
        )
    except ValueError as exc:
        assert "exactly three teachers" in str(exc)
    else:
        raise AssertionError("Expected a two-teacher committee to fail")


def test_unattainable_ceiling_is_reported_honestly() -> None:
    n = 20
    primary = np.tile(np.array([0.55, 0.45]), (n, 1))
    secondary = np.tile(np.array([0.70, 0.30]), (n, 1))
    secondary[:4] = np.array([0.30, 0.70])
    tertiary = np.tile(np.array([0.20, 0.80]), (n, 1))
    stacked = np.log(np.stack([primary, secondary, tertiary])).astype(np.float32)

    _, metadata = _run(stacked, target_queries=3.0)
    assert metadata["target_queries_per_sample"] == 3.0
    assert metadata["depth_1_count"] == 0
    assert metadata["depth_2_count"] == 16
    assert metadata["depth_3_count"] == 4
    assert np.isclose(metadata["realised_queries_per_sample"], 2.2)
    assert metadata["realised_queries_per_sample"] < 3.0


def main() -> None:
    test_distribution_depth_counts_and_realised_rate()
    test_seeded_assignment_and_exact_targets()
    test_metadata_schema_and_label_independence()
    test_invalid_budgets_and_committee_size_raise()
    test_unattainable_ceiling_is_reported_honestly()
    print("Depth-matched random soft-label checks passed.")


if __name__ == "__main__":
    main()
