"""Oracle upper-bound teacher purification (baseline #4).

Uses ground-truth labels for every training sample (including "unlabeled"
ones) to weight teachers per sample. This is an explicit cheating diagnostic
baseline establishing an upper bound on how well any conflict-aware
combination could do -- it is not a deployable method, since a real router
never sees labels for the unlabeled pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.loading import TextClassificationDataset
from ..student.soft_labels import REPORT_DIR, _logits_to_probs, load_multi_teacher_logits

MULTI_ORACLE_TEACHER = "oracle-guided"


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def oracle_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset: TextClassificationDataset,
    num_classes: int,
    temperature: float,
    oracle_temperature: float = 1.0,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Ground-truth-weighted teacher combination: upper bound, not deployable.

    Each teacher's per-sample weight is ``softmax_i(log p_i(y_true) / oracle_temperature)``,
    so the teacher that assigns the most probability to the true class dominates
    the combined logits for that sample.
    """
    stacked_logits = load_multi_teacher_logits(teachers, dataset.name, num_classes, report_dir)  # (K,N,C)
    probs = _logits_to_probs(stacked_logits, temperature=1.0)  # raw per-teacher probs
    labels = np.asarray(dataset.train[dataset.label_column])  # gold for every sample (upper-bound only)
    p_true = np.take_along_axis(probs, labels[None, :, None], axis=-1)[..., 0]  # (K,N)
    log_w = np.log(np.clip(p_true, 1e-12, None)) / oracle_temperature
    weights = _softmax(log_w, axis=0)  # (K,N), sums to 1 over teachers
    z_bar = (weights[:, :, None] * stacked_logits).sum(axis=0)  # (N,C)
    return _logits_to_probs(z_bar, temperature)  # final purified distribution at KD tau
