"""Stratified out-of-fold AUC for the retrospective harm diagnostic."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def stratified_oof_logreg_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 0,
    max_iter: int = 1000,
) -> dict:
    """Mean / std of stratified K-fold out-of-fold logistic-regression AUCs.

    Each fold trains on the complementary training indices and scores only the
    held-out fold, so the reported mean is not an in-sample fit statistic.
    Fold standard deviation uses sample std (ddof=1).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(np.int64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "auc_folds": [],
            "n_splits": 0,
        }

    n_splits_eff = min(int(n_splits), n_pos, n_neg)
    if n_splits_eff < 2:
        return {
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "auc_folds": [],
            "n_splits": 0,
        }

    skf = StratifiedKFold(
        n_splits=n_splits_eff, shuffle=True, random_state=random_state
    )
    fold_aucs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=max_iter)
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.predict_proba(X[test_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[test_idx], scores)))

    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "auc_folds": fold_aucs,
        "n_splits": n_splits_eff,
    }
