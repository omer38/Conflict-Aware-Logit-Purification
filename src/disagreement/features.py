"""Teacher-disagreement feature extraction phi(x) (proposal Eq. 6).

Operates on the guided teachers' per-class probabilities ``p_i(x) =
softmax(z_i(x))``, ``i = 1..K`` (K=3), computed at raw temperature (tau=1.0)
so the features reflect genuine teacher confidence/disagreement rather than
the softened KD distribution used later to form the purified label.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from ..student.soft_labels import REPORT_DIR, _logits_to_probs, load_multi_teacher_logits

FEATURE_NAMES: list[str] = [
    "H_mean",
    "kl_mean",
    "kl_max",
    "top1_agreement",
    "kendall_tau_mean",
    "top3_var_1",
    "top3_var_2",
    "top3_var_3",
    "delta_0",
    "delta_1",
    "delta_2",
]

FEATURE_GROUPS: dict[str, list[int]] = {
    "aggregate": [0, 1, 2, 3, 4],
    "class_aware": [5, 6, 7],
    "teacher_specific": [8, 9, 10],
}


def _teacher_pairs(num_teachers: int) -> list[tuple[int, int]]:
    return list(combinations(range(num_teachers), 2))


def mean_distribution(probs: np.ndarray) -> np.ndarray:
    """Consensus distribution ``p_bar = mean_i p_i``. ``(K,N,C) -> (N,C)``."""
    return probs.mean(axis=0)


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy over the last (class) axis."""
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``KL(p || q)`` over the last axis, with both inputs clipped away from 0."""
    log_ratio = np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None))
    return (p * log_ratio).sum(axis=-1)


def pairwise_symmetric_kl(probs: np.ndarray) -> np.ndarray:
    """Symmetric KL for every teacher pair. ``(K,N,C) -> (num_pairs, N)``."""
    pairs = _teacher_pairs(probs.shape[0])
    return np.stack(
        [0.5 * (_kl(probs[i], probs[j]) + _kl(probs[j], probs[i])) for i, j in pairs],
        axis=0,
    )


def top1_agreement(probs: np.ndarray) -> np.ndarray:
    """Fraction of teacher pairs whose argmax class matches, per sample."""
    K, N, _ = probs.shape
    pairs = _teacher_pairs(K)
    argmaxes = probs.argmax(axis=-1)  # (K, N)
    agreements = np.stack(
        [(argmaxes[i] == argmaxes[j]).astype(np.float32) for i, j in pairs], axis=0
    )
    return agreements.mean(axis=0)


def kendall_tau_mean(probs: np.ndarray) -> np.ndarray:
    """Mean pairwise Kendall tau-a over class rankings, fully vectorized.

    For each teacher, ``sign(p[:, c1] - p[:, c2])`` over all class pairs
    ``(c1 < c2)`` encodes its class ranking; tau-a for a teacher pair is the
    normalised sum of matching signs (concordant minus discordant, divided by
    the number of class pairs), averaged over all teacher pairs.
    """
    K, N, C = probs.shape
    diff = probs[:, :, :, None] - probs[:, :, None, :]  # (K, N, C, C)
    sign = np.sign(diff)
    iu = np.triu_indices(C, k=1)
    num_class_pairs = len(iu[0])
    sign_upper = sign[:, :, iu[0], iu[1]]  # (K, N, num_class_pairs)

    pairs = _teacher_pairs(K)
    tau_per_pair = np.stack(
        [
            (sign_upper[i] * sign_upper[j]).sum(axis=-1) / num_class_pairs
            for i, j in pairs
        ],
        axis=0,
    )  # (num_pairs, N)
    return tau_per_pair.mean(axis=0).astype(np.float32)


def top3_class_variance(probs: np.ndarray) -> np.ndarray:
    """Cross-teacher variance on the top-3 consensus classes. ``(N, 3)``."""
    _, N, _ = probs.shape
    mean_dist = mean_distribution(probs)  # (N, C)
    top3_idx = np.argsort(-mean_dist, axis=-1)[:, :3]  # (N, 3)
    n_idx = np.arange(N)[:, None]
    selected = probs[:, n_idx, top3_idx]  # (K, N, 3)
    return selected.var(axis=0).astype(np.float32)


def per_teacher_kl_from_consensus(probs: np.ndarray) -> np.ndarray:
    """``delta_i = KL(p_i || p_bar)`` for every teacher. ``(K, N)``."""
    p_bar = mean_distribution(probs)[None, :, :]  # (1, N, C)
    return _kl(probs, p_bar).astype(np.float32)


def compute_disagreement_features(probs: np.ndarray) -> np.ndarray:
    """Assemble the ``(N, 11)`` feature matrix in ``FEATURE_NAMES`` order."""
    K = probs.shape[0]
    if K != 3:
        raise ValueError(
            f"compute_disagreement_features expects exactly 3 teachers (K=3), got K={K}"
        )

    p_bar = mean_distribution(probs)
    h_mean = entropy(p_bar)
    pkl = pairwise_symmetric_kl(probs)
    kl_mean = pkl.mean(axis=0)
    kl_max = pkl.max(axis=0)
    top1_agr = top1_agreement(probs)
    tau_mean = kendall_tau_mean(probs)
    top3_var = top3_class_variance(probs)
    delta = per_teacher_kl_from_consensus(probs)

    return np.column_stack(
        [
            h_mean,
            kl_mean,
            kl_max,
            top1_agr,
            tau_mean,
            top3_var[:, 0],
            top3_var[:, 1],
            top3_var[:, 2],
            delta[0],
            delta[1],
            delta[2],
        ]
    ).astype(np.float32)


def compute_features_for_dataset(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Load guided teacher logits and compute ``phi(x)`` for every sample.

    Row order matches ``sample_idx`` / ``dataset.train`` order, like the
    existing soft-label loaders in ``src.student.soft_labels``.
    """
    stacked_logits = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    probs = _logits_to_probs(stacked_logits, temperature=1.0)
    return compute_disagreement_features(probs)
