"""Focused CPU-only tests for the entropy query cascade."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student.soft_labels import cascade_teacher_soft_labels


TEACHERS = ("primary", "secondary", "tertiary")


def _synthetic_probabilities() -> np.ndarray:
    confidence = np.linspace(0.98, 0.55, 10)
    primary = np.column_stack(
        [confidence, (1.0 - confidence) / 2.0, (1.0 - confidence) / 2.0]
    )
    secondary = np.tile(np.array([0.70, 0.20, 0.10]), (10, 1))
    secondary[[6, 9]] = np.array([0.20, 0.70, 0.10])
    tertiary = np.tile(np.array([0.10, 0.20, 0.70]), (10, 1))
    return np.stack([primary, secondary, tertiary])


def _run_cascade() -> tuple[np.ndarray, float, np.ndarray]:
    probabilities = _synthetic_probabilities()
    logits = np.log(probabilities)
    identity = lambda values, *args, **kwargs: values
    with (
        patch(
            "src.student.soft_labels.load_multi_teacher_logits",
            return_value=logits,
        ),
        patch("src.student.soft_labels._match_class_prior_logits", side_effect=identity),
        patch(
            "src.student.soft_labels._align_confidence_scales_logits",
            side_effect=identity,
        ),
    ):
        targets, realised = cascade_teacher_soft_labels(
            TEACHERS,
            "synthetic",
            num_classes=3,
            temperature=1.0,
            target_queries=1.4,
        )
    return targets, realised, probabilities


def test_cascade_budget_and_escalation_rules() -> None:
    targets, realised, probabilities = _run_cascade()
    primary, secondary, tertiary = probabilities

    assert targets.shape == (10, 3)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert np.isclose(realised, 1.4)

    # Only the three highest-entropy primary predictions cross the threshold.
    assert np.allclose(targets[:7], primary[:7])
    assert np.allclose(targets[7:9], 0.5 * (primary[7:9] + secondary[7:9]))

    # The third teacher is queried only for an escalated primary/secondary
    # argmax disagreement. Row 6 disagrees but is below the entropy threshold.
    assert primary[6].argmax() != secondary[6].argmax()
    assert np.allclose(targets[6], primary[6])
    assert primary[9].argmax() != secondary[9].argmax()
    assert np.allclose(
        targets[9],
        (primary[9] + secondary[9] + tertiary[9]) / 3.0,
    )


def test_cascade_is_label_independent() -> None:
    parameters = inspect.signature(cascade_teacher_soft_labels).parameters
    assert "labels" not in parameters
    assert "gold_labels" not in parameters


def test_cascade_budget_respects_discrete_tolerance() -> None:
    probabilities = _synthetic_probabilities()[:, :6]
    logits = np.log(probabilities)
    identity = lambda values, *args, **kwargs: values
    with (
        patch(
            "src.student.soft_labels.load_multi_teacher_logits",
            return_value=logits,
        ),
        patch("src.student.soft_labels._match_class_prior_logits", side_effect=identity),
        patch(
            "src.student.soft_labels._align_confidence_scales_logits",
            side_effect=identity,
        ),
    ):
        _, realised = cascade_teacher_soft_labels(
            TEACHERS,
            "synthetic",
            num_classes=3,
            temperature=1.0,
            target_queries=1.4,
        )
    assert abs(realised - 1.4) <= 1.0 / probabilities.shape[1]


def test_cascade_validates_budget_and_committee_size() -> None:
    for target in (0.9, 3.1):
        try:
            cascade_teacher_soft_labels(
                TEACHERS,
                "synthetic",
                num_classes=3,
                temperature=1.0,
                target_queries=target,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid target_queries={target} to fail")

    try:
        cascade_teacher_soft_labels(
            TEACHERS[:2],
            "synthetic",
            num_classes=3,
            temperature=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a two-teacher cascade to fail")


def main() -> None:
    test_cascade_budget_and_escalation_rules()
    test_cascade_is_label_independent()
    test_cascade_budget_respects_discrete_tolerance()
    test_cascade_validates_budget_and_committee_size()
    print("Cascade soft-label checks passed.")


if __name__ == "__main__":
    main()
