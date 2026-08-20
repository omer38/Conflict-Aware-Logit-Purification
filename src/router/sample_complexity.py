"""Sample-complexity utilities for conflict-aware routers.

Used by ``scripts/router_learning_curve.py`` to (A) measure the learnable
ceiling of a φ-router vs calibrated probability-mean, (B) sweep n_cal, and
(C) compare sample-efficiency variants.  Proxy metrics are purification-target
quality on held-out train-pool labels — no student GPU.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..disagreement.features import FEATURE_GROUPS, compute_disagreement_features
from ..student.soft_labels import REPORT_DIR, GUIDED_TEACHERS
from .purify import purified_soft_labels
from .train_router import (
    entropy_prior_log_weights,
    router_weights,
    train_router,
    uniform_log_prior,
)

DATASETS: dict[str, int] = {"ag_news": 4, "yahoo": 10, "amazon": 5}
PER_CLASS_GRID: tuple[int, ...] = (3, 5, 10, 20, 50, 100, 200)
N_CURVE_SEEDS = 20
N_CEILING_SEEDS = 5
ENTROPY_TOL = 0.05
ORACLE_GAP_FRACTION = 0.5
GATE_PASS_RATE = 0.8
CAR_ENTROPY_COEF = 0.1
V2_TEMPERATURE = 2.0
CAR_TEMPERATURE = 1.0


@dataclass(frozen=True)
class Variant:
    """A router / aggregator configuration for the learning-curve sweep."""

    name: str
    kind: str  # "router" | "baseline"
    router_type: str = "linear"
    features: str = "all"  # "all" | "aggregate"
    prior: str = "entropy"  # "entropy" | "uniform"
    mixture_space: str = "logits"
    calibrate: bool = False
    entropy_coef: float = 0.0
    sparse: bool = False
    sampling: str = "random"  # "random" | "disagreement"
    pseudo: bool = False
    stage_a: bool = True
    stage_curve: bool = True


VARIANTS: tuple[Variant, ...] = (
    Variant(name="v2_linear_all", kind="router", stage_curve=False),
    Variant(name="v2_mlp_all", kind="router", router_type="mlp", stage_curve=False),
    Variant(name="v2_linear_agg", kind="router", features="aggregate", stage_curve=False),
    Variant(
        name="car_linear_all",
        kind="router",
        prior="uniform",
        mixture_space="probs",
        calibrate=True,
        entropy_coef=CAR_ENTROPY_COEF,
    ),
    Variant(
        name="car_mlp_all",
        kind="router",
        router_type="mlp",
        prior="uniform",
        mixture_space="probs",
        calibrate=True,
        entropy_coef=CAR_ENTROPY_COEF,
        stage_curve=False,
    ),
    Variant(
        name="car_linear_agg",
        kind="router",
        features="aggregate",
        prior="uniform",
        mixture_space="probs",
        calibrate=True,
        entropy_coef=CAR_ENTROPY_COEF,
    ),
    Variant(
        name="car_linear_sparse",
        kind="router",
        prior="uniform",
        mixture_space="probs",
        calibrate=True,
        entropy_coef=CAR_ENTROPY_COEF,
        sparse=True,
    ),
    Variant(
        name="car_linear_disagree",
        kind="router",
        prior="uniform",
        mixture_space="probs",
        calibrate=True,
        entropy_coef=CAR_ENTROPY_COEF,
        sampling="disagreement",
        sparse=True,
    ),
    Variant(
        name="car_linear_pseudo",
        kind="router",
        prior="uniform",
        mixture_space="probs",
        calibrate=True,
        entropy_coef=CAR_ENTROPY_COEF,
        pseudo=True,
        sparse=True,
    ),
    Variant(name="drop_outlier_lambda", kind="baseline", calibrate=True, stage_a=True),
    Variant(name="class_conditional", kind="baseline", calibrate=True, stage_a=True),
)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = z - z.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=axis, keepdims=True)).astype(np.float32)


def nll_of(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def entropy_of(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1).mean())


def accuracy_of(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(axis=1) == y).mean())


def metrics_of(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"acc": accuracy_of(p, y), "nll": nll_of(p, y), "H": entropy_of(p)}


def load_teacher_stack(
    dataset: str,
    num_classes: int,
    report_dir: str | Path = REPORT_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return logits ``(K, N, C)`` and gold labels ``(N,)`` aligned by sample_idx."""
    logits = []
    labels = None
    report_dir = Path(report_dir)
    for teacher in GUIDED_TEACHERS:
        path = report_dir / f"{teacher}_{dataset}.csv"
        rows = sorted(
            csv.DictReader(path.open(encoding="utf-8")),
            key=lambda r: int(r["sample_idx"]),
        )
        z = np.array(
            [[float(r[f"logit_{chr(65 + c)}"]) for c in range(num_classes)] for r in rows],
            dtype=np.float32,
        )
        logits.append(z)
        if labels is None:
            labels = np.array([int(r["true_label_id"]) for r in rows], dtype=np.int64)
    return np.stack(logits, axis=0), labels


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    grid = np.geomspace(0.5, 8.0, 20)
    return float(min(grid, key=lambda t: nll_of(softmax(z / t), y)))


def calibrate_logits(logits_knc: np.ndarray, y: np.ndarray, cal_idx: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Fit one temperature per teacher on ``cal_idx``; return scaled logits and τ."""
    taus = [fit_temperature(logits_knc[i, cal_idx], y[cal_idx]) for i in range(logits_knc.shape[0])]
    scaled = np.stack([logits_knc[i] / taus[i] for i in range(logits_knc.shape[0])], axis=0)
    return scaled.astype(np.float32), taus


def disagreement_mask(probs_knc: np.ndarray) -> np.ndarray:
    votes = probs_knc.argmax(axis=-1)
    agree = (votes[0] == votes[1]) & (votes[1] == votes[2])
    return ~agree


def majority_pseudo_labels(probs_knc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-vs-1 majority vote labels. Returns (indices, pseudo_y) for 2-vs-1 rows."""
    votes = probs_knc.argmax(axis=-1)  # (K, N)
    n = votes.shape[1]
    idx, labels = [], []
    for j in range(n):
        vals, counts = np.unique(votes[:, j], return_counts=True)
        if counts.max() == 2:
            idx.append(j)
            labels.append(int(vals[counts.argmax()]))
    return np.asarray(idx, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def drop_outlier_probs(probs_knc: np.ndarray, disagree: np.ndarray) -> np.ndarray:
    k = probs_knc.shape[0]
    pbar = probs_knc.mean(axis=0)
    log_p = np.log(np.clip(probs_knc, 1e-12, None))
    log_bar = np.log(np.clip(pbar, 1e-12, None))
    delta = (probs_knc * (log_p - log_bar[None])).sum(axis=2)
    outlier = delta.argmax(axis=0)
    out = pbar.copy()
    for i in range(k):
        sel = disagree & (outlier == i)
        others = [j for j in range(k) if j != i]
        out[sel] = probs_knc[others][:, sel].mean(axis=0)
    return out


def oracle_weight_mixture(logits_knc: np.ndarray, y: np.ndarray, oracle_tau: float) -> np.ndarray:
    p = softmax(logits_knc, axis=-1)
    p_true = np.take_along_axis(p, y[None, :, None], axis=-1)[..., 0]
    log_w = np.log(np.clip(p_true, 1e-12, None)) / oracle_tau
    w = softmax(log_w, axis=0)
    z_bar = np.einsum("kn,knc->nc", w, logits_knc)
    return softmax(z_bar, axis=-1)


def stratified_indices(
    y: np.ndarray,
    n_per_class: int,
    rng: np.random.Generator,
    pool: np.ndarray | None = None,
    num_classes: int | None = None,
) -> np.ndarray:
    """Draw ``n_per_class`` labelled rows per class, optionally restricted to ``pool``."""
    c_max = int(y.max()) + 1 if num_classes is None else num_classes
    chosen: list[int] = []
    for c in range(c_max):
        candidates = np.where(y == c)[0]
        if pool is not None:
            candidates = np.intersect1d(candidates, pool, assume_unique=False)
        if len(candidates) == 0:
            continue
        take = min(int(n_per_class), len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return np.asarray(chosen, dtype=np.int64)


def feature_matrix(probs_knc: np.ndarray, feature_set: str) -> np.ndarray:
    phi = compute_disagreement_features(probs_knc)
    if feature_set == "aggregate":
        return phi[:, FEATURE_GROUPS["aggregate"]]
    if feature_set != "all":
        raise ValueError(f"Unknown feature set '{feature_set}'")
    return phi


def slice_phi(phi_all: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return phi_all[idx]


def fit_drop_outlier_lambda(
    p_mean: np.ndarray,
    p_drop: np.ndarray,
    y: np.ndarray,
    cal_idx: np.ndarray,
) -> float:
    best_lam, best_nll = 1.0, nll_of(p_mean[cal_idx], y[cal_idx])
    for lam in np.linspace(0.0, 1.0, 11):
        mixed = lam * p_mean + (1.0 - lam) * p_drop
        score = nll_of(mixed[cal_idx], y[cal_idx])
        if score < best_nll:
            best_nll, best_lam = score, float(lam)
    return best_lam


def class_conditional_mixture(
    teacher_probs_knc: np.ndarray,
    y: np.ndarray,
    cal_idx: np.ndarray,
) -> np.ndarray:
    """Per-consensus-class teacher weights estimated on the calibration split."""
    k, n, c = teacher_probs_knc.shape
    consensus = teacher_probs_knc.mean(axis=0).argmax(axis=1)
    weights = np.full((c, k), 1.0 / k, dtype=np.float32)
    for cls in range(c):
        members = cal_idx[consensus[cal_idx] == cls]
        if len(members) == 0:
            continue
        scores = np.array(
            [nll_of(teacher_probs_knc[i, members], y[members]) for i in range(k)],
            dtype=np.float64,
        )
        weights[cls] = softmax(-scores[None, :], axis=-1)[0]
    w = weights[consensus]
    return np.einsum("nk,knc->nc", w, teacher_probs_knc)


def apply_sparse_prior(
    p_router: np.ndarray,
    p_prior: np.ndarray,
    disagree: np.ndarray,
) -> np.ndarray:
    out = p_prior.copy()
    out[disagree] = p_router[disagree]
    return out


def train_and_purify(
    variant: Variant,
    phi_all: np.ndarray,
    logits_nkc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    log_prior: np.ndarray,
    disagree: np.ndarray,
    temperature: float,
    weight_decay: float,
    seed: int,
    epochs: int = 300,
    entropy_floor: float | None = None,
    pseudo_idx: np.ndarray | None = None,
    pseudo_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a φ-router and return (mixture probs on all N, teacher weights on all N)."""
    if variant.pseudo:
        if pseudo_idx is None or len(pseudo_idx) == 0:
            raise ValueError("pseudo variant requires 2-vs-1 majority indices")
        fit_idx, fit_y = pseudo_idx, pseudo_y
    elif variant.sparse:
        fit_idx = train_idx[disagree[train_idx]]
        fit_y = y[fit_idx]
        if len(fit_idx) < 2:
            fit_idx, fit_y = train_idx, y[train_idx]
    else:
        fit_idx, fit_y = train_idx, y[train_idx]

    phi_fit, mean, std = _standardize_from_idx(phi_all, fit_idx)
    phi_all_z = _apply_standardize(phi_all, mean, std)

    router = train_router(
        phi_fit,
        logits_nkc[fit_idx],
        fit_y,
        log_prior[fit_idx],
        variant.router_type,
        temperature,
        epochs=epochs,
        weight_decay=weight_decay,
        seed=seed,
        mixture_space=variant.mixture_space,
        entropy_coef=variant.entropy_coef,
        entropy_floor=entropy_floor,
    )
    p = purified_soft_labels(
        router,
        phi_all_z,
        logits_nkc,
        log_prior,
        temperature,
        mixture_space=variant.mixture_space,
        disagree_mask=disagree if variant.sparse else None,
    )
    with torch.no_grad():
        w = router_weights(
            router,
            torch.as_tensor(phi_all_z, dtype=torch.float32),
            torch.as_tensor(log_prior, dtype=torch.float32),
        ).numpy()
    return p, w


def _standardize_from_idx(phi: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = phi[idx].mean(axis=0)
    std = phi[idx].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((phi[idx] - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _apply_standardize(phi: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-8, 1.0, std)
    return ((phi - mean) / std).astype(np.float32)


def percentile_band(values: list[float], lo: float = 10.0, hi: float = 90.0) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, lo)),
        "p90": float(np.percentile(arr, hi)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
    }


def nstar_beats_mean(per_n: list[dict]) -> int | None:
    """Smallest n where NLL beats calibrated mean without sharpening, with a working gate."""
    ordered = sorted(per_n, key=lambda r: r["n_cal"])
    for row in ordered:
        if (
            row["nll_gap_p10"] < 0.0
            and row["entropy_ok_rate"] >= GATE_PASS_RATE
            and row["gate_pass_rate"] >= GATE_PASS_RATE
        ):
            return int(row["n_cal"])
    return None


def nstar_from_curve(
    per_n: list[dict],
    nll_key: str = "nll_gap_p10",
    entropy_ok_key: str = "entropy_ok_rate",
    gap_closed_key: str = "oracle_gap_closed_p10",
    gate_key: str = "gate_pass_rate",
) -> int | None:
    """Smallest n_cal whose 10th-percentile still satisfies the 'good router' criteria."""
    ordered = sorted(per_n, key=lambda r: r["n_cal"])
    for row in ordered:
        nll_ok = row[nll_key] < 0.0
        ent_ok = row[entropy_ok_key] >= GATE_PASS_RATE
        gap_ok = row[gap_closed_key] >= ORACLE_GAP_FRACTION
        gate_ok = row[gate_key] >= GATE_PASS_RATE
        if nll_ok and ent_ok and gap_ok and gate_ok:
            return int(row["n_cal"])
    return None


def nstar_temperature(per_n: list[dict], full_nll: float, rel_tol: float = 0.02) -> int | None:
    """Smallest n where median calib-mean NLL is within ``rel_tol`` of the full-pool value."""
    ordered = sorted(per_n, key=lambda r: r["n_cal"])
    target = full_nll * (1.0 + rel_tol) if full_nll > 0 else full_nll + abs(full_nll) * rel_tol
    for row in ordered:
        if row["calibmean_nll_median"] <= target:
            return int(row["n_cal"])
    return None


def prepare_features_and_prior(
    variant: Variant,
    logits_knc: np.ndarray,
    y: np.ndarray,
    cal_idx: np.ndarray,
    phi_precomputed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float] | None]:
    """Return phi (N, d), logits_nkc, log_prior (N, K), disagree mask, optional taus.

    ``phi_precomputed`` should be the full 11-d raw disagreement matrix.  When
    supplied, Kendall-tau features are not recomputed (required for the n-grid).
    """
    taus = None
    working = logits_knc
    if variant.calibrate:
        working, taus = calibrate_logits(logits_knc, y, cal_idx)
    probs = softmax(working, axis=-1)
    if phi_precomputed is None:
        phi = feature_matrix(probs, variant.features)
    else:
        phi = phi_precomputed
        if variant.features == "aggregate":
            phi = phi[:, FEATURE_GROUPS["aggregate"]]
    logits_nkc = np.transpose(working, (1, 0, 2))
    if variant.prior == "uniform":
        log_prior = uniform_log_prior(logits_nkc.shape[0], logits_nkc.shape[1])
    else:
        log_prior = entropy_prior_log_weights(logits_nkc)
    disagree = disagreement_mask(probs)
    return phi, logits_nkc, log_prior, disagree, taus


def baseline_mixture(
    variant: Variant,
    logits_knc: np.ndarray,
    y: np.ndarray,
    cal_idx: np.ndarray,
) -> np.ndarray:
    working = logits_knc
    if variant.calibrate:
        working, _ = calibrate_logits(logits_knc, y, cal_idx)
    probs = softmax(working, axis=-1)
    p_mean = probs.mean(axis=0)
    if variant.name == "drop_outlier_lambda":
        disagree = disagreement_mask(probs)
        p_drop = drop_outlier_probs(probs, disagree)
        lam = fit_drop_outlier_lambda(p_mean, p_drop, y, cal_idx)
        return lam * p_mean + (1.0 - lam) * p_drop
    if variant.name == "class_conditional":
        return class_conditional_mixture(probs, y, cal_idx)
    raise ValueError(f"Not a baseline variant: {variant.name}")


def calibrated_prob_mean(logits_knc: np.ndarray, y: np.ndarray, cal_idx: np.ndarray) -> np.ndarray:
    scaled, _ = calibrate_logits(logits_knc, y, cal_idx)
    return softmax(scaled, axis=-1).mean(axis=0)


def prior_mixture_probs(
    logits_nkc: np.ndarray,
    log_prior: np.ndarray,
    temperature: float,
    mixture_space: str,
) -> np.ndarray:
    w = softmax(log_prior, axis=-1)
    if mixture_space == "logits":
        z_bar = np.einsum("nk,nkc->nc", w, logits_nkc)
        return softmax(z_bar / temperature, axis=-1)
    p_t = softmax(logits_nkc / temperature, axis=-1)
    return np.einsum("nk,nkc->nc", w, p_t)


def weight_entropy(weights: np.ndarray) -> float:
    return entropy_of(weights)
