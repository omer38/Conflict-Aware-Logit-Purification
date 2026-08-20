"""Tests for stratified out-of-fold harm-prediction AUC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.disagreement.harm_auc import stratified_oof_logreg_auc


def test_oof_auc_high_on_separable_data() -> None:
    rng = np.random.default_rng(0)
    n = 400
    # One strong feature separates rare harm from non-harm.
    harm = np.zeros(n, dtype=np.int64)
    harm[:80] = 1
    x = rng.normal(size=(n, 5))
    x[:, 0] = harm.astype(np.float64) + 0.15 * rng.normal(size=n)

    result = stratified_oof_logreg_auc(x, harm, n_splits=5, random_state=0)
    assert result["n_splits"] == 5
    assert len(result["auc_folds"]) == 5
    assert result["auc_mean"] > 0.9
    assert result["auc_std"] >= 0.0
    assert np.isfinite(result["auc_mean"])


def test_oof_auc_nan_when_harm_constant() -> None:
    x = np.zeros((50, 3))
    y = np.zeros(50, dtype=np.int64)
    result = stratified_oof_logreg_auc(x, y)
    assert np.isnan(result["auc_mean"])
    assert result["n_splits"] == 0
    assert result["auc_folds"] == []


if __name__ == "__main__":
    test_oof_auc_high_on_separable_data()
    test_oof_auc_nan_when_harm_constant()
    print("All harm_auc tests passed.")
