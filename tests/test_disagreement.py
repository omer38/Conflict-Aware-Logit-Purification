"""Tests for the teacher-disagreement feature extraction phi(x) (Phase B / M3).

Covers synthetic agreement/disagreement/shared-uncertainty cases and a
shape/sanity check on real guided teacher logits.

Run:
    uv run python tests/test_disagreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.disagreement.features import (
    FEATURE_NAMES,
    compute_disagreement_features,
    compute_features_for_dataset,
)
from src.student.soft_labels import GUIDED_TEACHERS

DATASET = "ag_news"


def _feature(features: np.ndarray, name: str) -> np.ndarray:
    return features[:, FEATURE_NAMES.index(name)]


def _make_case(rows: list[list[float]]) -> np.ndarray:
    """Build a ``(K, 1, C)`` probs array from one distribution per teacher."""
    return np.array(rows, dtype=np.float32)[:, None, :]


def test_synthetic_agreement_disagreement_uncertainty() -> None:
    """Full agreement, full disagreement, and shared uncertainty are ordered as expected."""
    print("--- test_synthetic_agreement_disagreement_uncertainty ---")

    full_agreement = _make_case([[0.9, 0.05, 0.05]] * 3)
    full_disagreement = _make_case(
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
    )
    shared_uncertainty = _make_case([[0.34, 0.33, 0.33]] * 3)

    feats_agree = compute_disagreement_features(full_agreement)
    feats_disagree = compute_disagreement_features(full_disagreement)
    feats_uncertain = compute_disagreement_features(shared_uncertainty)

    kl_agree = _feature(feats_agree, "kl_mean")[0]
    kl_disagree = _feature(feats_disagree, "kl_mean")[0]
    kl_uncertain = _feature(feats_uncertain, "kl_mean")[0]
    assert kl_agree < kl_disagree, (kl_agree, kl_disagree)
    assert kl_uncertain < kl_disagree, (kl_uncertain, kl_disagree)

    top1_agree = _feature(feats_agree, "top1_agreement")[0]
    top1_disagree = _feature(feats_disagree, "top1_agreement")[0]
    assert top1_agree > top1_disagree, (top1_agree, top1_disagree)
    assert np.isclose(top1_agree, 1.0)
    assert np.isclose(top1_disagree, 0.0)

    h_agree = _feature(feats_agree, "H_mean")[0]
    h_uncertain = _feature(feats_uncertain, "H_mean")[0]
    assert h_uncertain > h_agree, (h_uncertain, h_agree)

    tau_agree = _feature(feats_agree, "kendall_tau_mean")[0]
    tau_disagree = _feature(feats_disagree, "kendall_tau_mean")[0]
    assert tau_agree > tau_disagree, (tau_agree, tau_disagree)

    print(
        f"  kl_mean agree={kl_agree:.4f} uncertain={kl_uncertain:.4f} disagree={kl_disagree:.4f}"
    )
    print(f"  top1_agreement agree={top1_agree:.2f} disagree={top1_disagree:.2f}")
    print(f"  H_mean agree={h_agree:.4f} uncertain={h_uncertain:.4f}")
    print(f"  kendall_tau_mean agree={tau_agree:.4f} disagree={tau_disagree:.4f}")
    print("  PASSED\n")


def test_real_data_shape_and_sanity() -> None:
    """phi(x) on real ag_news guided logits has the expected shape and ranges."""
    print("--- test_real_data_shape_and_sanity ---")
    ds = load_dataset_by_name(DATASET)
    features = compute_features_for_dataset(GUIDED_TEACHERS, DATASET, ds.num_classes)

    assert features.shape == (len(ds.train), 11), features.shape
    assert not np.isnan(features).any(), "disagreement features contain NaN"

    top1 = _feature(features, "top1_agreement")
    assert top1.min() >= 0.0 and top1.max() <= 1.0, (top1.min(), top1.max())

    h_mean = _feature(features, "H_mean")
    assert h_mean.min() >= 0.0, h_mean.min()

    print(f"  {features.shape} features, no NaN, ranges valid. PASSED\n")


def main() -> None:
    test_synthetic_agreement_disagreement_uncertainty()
    test_real_data_shape_and_sanity()
    print("=== All disagreement tests passed. ===")


if __name__ == "__main__":
    main()
