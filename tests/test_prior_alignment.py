"""Tests for explicit class-prior matching and confidence-scale alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student.soft_labels import (
    GUIDED_TEACHERS,
    _align_confidence_scales_logits,
    _logits_to_probs,
    _match_class_prior_logits,
    calibrated_mean_teacher_soft_labels,
    prior_scale_aligned_teacher_soft_labels,
    uniform_class_prior,
)


def test_class_prior_matching_hits_supplied_prior() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(3, 200, 4)).astype(np.float32)
    target = np.array([0.55, 0.20, 0.15, 0.10])
    matched = _match_class_prior_logits(logits, target)
    marginals = _logits_to_probs(matched, 1.0).mean(axis=1)
    assert np.allclose(marginals, target[None, :], atol=1e-5)


def test_weighted_class_prior_matching() -> None:
    rng = np.random.default_rng(8)
    logits = rng.normal(size=(2, 120, 3)).astype(np.float32)
    weights = np.linspace(0.25, 2.0, 120)
    target = np.array([0.20, 0.30, 0.50])
    matched = _match_class_prior_logits(
        logits,
        target,
        sample_weights=weights,
    )
    probabilities = _logits_to_probs(matched, 1.0)
    marginals = np.stack(
        [np.average(probabilities[k], axis=0, weights=weights) for k in range(2)]
    )
    assert np.allclose(marginals, target[None, :], atol=1e-5)


def test_confidence_scale_alignment_preserves_argmax() -> None:
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(3, 100, 5)).astype(np.float32)
    logits[1] *= 4.0
    logits[2] *= 0.5
    aligned = _align_confidence_scales_logits(logits)
    assert np.array_equal(aligned.argmax(axis=-1), logits.argmax(axis=-1))

    probabilities = _logits_to_probs(aligned, 1.0)
    entropies = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(-1)
    assert np.allclose(entropies.mean(axis=1), entropies.mean(axis=1)[0], atol=1e-5)


def test_historical_uniform_wrapper_is_unchanged() -> None:
    historical = calibrated_mean_teacher_soft_labels(
        GUIDED_TEACHERS,
        "ag_news",
        num_classes=4,
        temperature=2.0,
    )
    explicit = prior_scale_aligned_teacher_soft_labels(
        GUIDED_TEACHERS,
        "ag_news",
        num_classes=4,
        temperature=2.0,
        target_prior=uniform_class_prior(4),
    )
    assert np.array_equal(historical, explicit)


def main() -> None:
    test_class_prior_matching_hits_supplied_prior()
    test_weighted_class_prior_matching()
    test_confidence_scale_alignment_preserves_argmax()
    test_historical_uniform_wrapper_is_unchanged()
    print("Prior matching and confidence-scale alignment checks passed.")


if __name__ == "__main__":
    main()
