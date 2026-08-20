"""Training and leave-one-out model selection for the conflict-aware router
(proposal Eq. 9, Stage 3 / M4).

The router is trained only on ``labeled_cal`` -- full-batch Adam, since cal
sets are tiny (e.g. ~3 examples/class at L=10).  v2 reparameterises the
router as a learned correction on top of a fixed entropy-weighted prior:
``w = softmax(correction(phi) + log_prior)``.  A LOOCV safety gate falls
back to the best heuristic baseline when the learned router does not clearly
beat it on the calibration set.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..student.soft_labels import _entropy_weights, _logits_to_probs
from .model import DEFAULT_HIDDEN_DIM, LinearRouter, MLPRouter

ROUTER_TYPES = ("linear", "mlp")
MULTI_ROUTER_LINEAR_TEACHER = "multi-router-linear-guided"
MULTI_ROUTER_MLP_TEACHER = "multi-router-mlp-guided"


def _build_router(router_type: str, feature_dim: int, num_teachers: int, dropout: float) -> torch.nn.Module:
    if router_type == "linear":
        return LinearRouter(feature_dim, num_teachers)
    if router_type == "mlp":
        return MLPRouter(feature_dim, DEFAULT_HIDDEN_DIM, num_teachers, dropout=dropout)
    raise ValueError(f"Unknown router_type '{router_type}' (expected one of {ROUTER_TYPES}).")


def standardize_features(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score phi(x) using population statistics over all N samples."""
    mean = phi.mean(axis=0)
    std = phi.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((phi - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def entropy_prior_log_weights(logits: np.ndarray) -> np.ndarray:
    """Log of entropy-weighted teacher weights per sample.

    ``logits`` has shape ``(N, K, C)``.  Returns ``(N, K)`` log-weights that
    sum (in log-space via softmax) to the same mixture as ``multi-entropy-guided``.
    """
    # transpose to (K, N, C) for shared helper
    stacked = np.transpose(logits, (1, 0, 2))
    probs = _logits_to_probs(stacked, temperature=1.0)
    weights = _entropy_weights(probs)  # (K, N)
    log_w = np.log(np.clip(weights, 1e-12, None))
    return log_w.T.astype(np.float32)  # (N, K)


def uniform_log_prior(n_samples: int, num_teachers: int) -> np.ndarray:
    """Log-weights of a uniform teacher mixture (calibrated-mean shrinkage prior)."""
    return np.zeros((n_samples, num_teachers), dtype=np.float32)


def router_weights(
    router: torch.nn.Module,
    phi: torch.Tensor,
    log_prior: torch.Tensor,
) -> torch.Tensor:
    """Combine learned correction with entropy prior: ``(batch, K)`` weights."""
    correction = router(phi)
    return F.softmax(correction + log_prior, dim=-1)


def _mixture_logits(
    weights: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    return (weights.unsqueeze(-1) * logits).sum(dim=1)


def mixture_probs(
    router: torch.nn.Module,
    phi: torch.Tensor,
    logits: torch.Tensor,
    log_prior: torch.Tensor,
    temperature: float,
    mixture_space: str = "logits",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(batch, C)`` mixture probabilities and ``(batch, K)`` weights."""
    w = router_weights(router, phi, log_prior)
    if mixture_space == "logits":
        z_bar = _mixture_logits(w, logits)
        return F.softmax(z_bar / temperature, dim=-1), w
    if mixture_space == "probs":
        teacher_probs = F.softmax(logits / temperature, dim=-1)
        return (w.unsqueeze(-1) * teacher_probs).sum(dim=1), w
    raise ValueError(f"Unknown mixture_space '{mixture_space}'")


def router_loss(
    router: torch.nn.Module,
    phi_cal: torch.Tensor,
    logits_cal: torch.Tensor,
    labels_cal: torch.Tensor,
    log_prior_cal: torch.Tensor,
    temperature: float,
    mixture_space: str = "logits",
    entropy_coef: float = 0.0,
    entropy_floor: float | None = None,
) -> torch.Tensor:
    """Cross-entropy / NLL of the router-purified mixture vs the true label (Eq. 9).

    v2 defaults (``mixture_space='logits'``, ``entropy_coef=0``) are unchanged.
    CAR training uses probability-space mixing plus an entropy floor so the
    router cannot improve NLL solely by sharpening the target.
    """
    p, _ = mixture_probs(
        router, phi_cal, logits_cal, log_prior_cal, temperature, mixture_space
    )
    nll = F.nll_loss(torch.log(p.clamp_min(1e-12)), labels_cal)
    if entropy_coef == 0.0:
        return nll
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean()
    if entropy_floor is None:
        return nll - entropy_coef * entropy
    return nll + entropy_coef * F.relu(entropy_floor - entropy)


def train_router(
    phi_cal: np.ndarray,
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    log_prior_cal: np.ndarray,
    router_type: str,
    temperature: float,
    epochs: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    dropout: float = 0.3,
    seed: int = 42,
    mixture_space: str = "logits",
    entropy_coef: float = 0.0,
    entropy_floor: float | None = None,
) -> torch.nn.Module:
    """Train a router on the (tiny) calibration set with full-batch Adam."""
    torch.manual_seed(seed)
    phi_t = torch.as_tensor(phi_cal, dtype=torch.float32)
    logits_t = torch.as_tensor(logits_cal, dtype=torch.float32)
    labels_t = torch.as_tensor(labels_cal, dtype=torch.long)
    log_prior_t = torch.as_tensor(log_prior_cal, dtype=torch.float32)

    router = _build_router(router_type, phi_t.shape[1], logits_t.shape[1], dropout)
    optimizer = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=weight_decay)

    router.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = router_loss(
            router, phi_t, logits_t, labels_t, log_prior_t, temperature,
            mixture_space=mixture_space,
            entropy_coef=entropy_coef,
            entropy_floor=entropy_floor,
        )
        loss.backward()
        optimizer.step()

    router.eval()
    return router


def _heuristic_cal_accuracy(
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    temperature: float,
    method: str,
) -> float:
    """Calibration-set accuracy for a fixed heuristic (mean or entropy-weighted)."""
    if method == "mean":
        z_bar = logits_cal.mean(axis=1)
    elif method == "entropy":
        log_prior = entropy_prior_log_weights(logits_cal)
        w = np.exp(log_prior - log_prior.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        z_bar = (w[:, :, None] * logits_cal).sum(axis=1)
    else:
        raise ValueError(f"Unknown heuristic method '{method}'")

    preds = (z_bar / temperature).argmax(axis=1)
    return float((preds == labels_cal).mean())


def select_router_or_fallback_nll(
    router_nll: float,
    baseline_nll: float,
) -> dict:
    """NLL safety gate: deploy the router only if it strictly beats the baseline."""
    use_router = router_nll < baseline_nll
    return {
        "use_router": use_router,
        "router_nll": float(router_nll),
        "baseline_nll": float(baseline_nll),
        "fallback_to": None if use_router else "baseline",
    }


def select_router_or_fallback(
    router_loocv_accuracy: float,
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    temperature: float,
) -> dict:
    """Decide whether to deploy the learned router or fall back to a heuristic.

    The router is used only when its LOOCV accuracy strictly exceeds the
    better of naive-mean and entropy-weighted baselines on ``labeled_cal``.
    """
    mean_acc = _heuristic_cal_accuracy(logits_cal, labels_cal, temperature, "mean")
    entropy_acc = _heuristic_cal_accuracy(logits_cal, labels_cal, temperature, "entropy")

    if entropy_acc >= mean_acc:
        best_heuristic = "entropy"
        best_heuristic_acc = entropy_acc
    else:
        best_heuristic = "mean"
        best_heuristic_acc = mean_acc

    use_router = router_loocv_accuracy > best_heuristic_acc
    return {
        "use_router": use_router,
        "router_loocv_accuracy": router_loocv_accuracy,
        "mean_cal_accuracy": mean_acc,
        "entropy_cal_accuracy": entropy_acc,
        "best_heuristic": best_heuristic,
        "best_heuristic_accuracy": best_heuristic_acc,
        "fallback_to": None if use_router else best_heuristic,
    }


def loocv_router_score(
    phi_cal: np.ndarray,
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    log_prior_cal: np.ndarray,
    router_type: str,
    temperature: float,
    **hparams,
) -> dict:
    """Leave-one-out diagnostic: train on n-1 cal points, score the held-out one."""
    n = len(labels_cal)
    losses: list[float] = []
    correct = 0

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        router = train_router(
            phi_cal[mask], logits_cal[mask], labels_cal[mask], log_prior_cal[mask],
            router_type, temperature, **hparams,
        )

        phi_i = torch.as_tensor(phi_cal[i : i + 1], dtype=torch.float32)
        logits_i = torch.as_tensor(logits_cal[i : i + 1], dtype=torch.float32)
        label_i = torch.as_tensor(labels_cal[i : i + 1], dtype=torch.long)
        log_prior_i = torch.as_tensor(log_prior_cal[i : i + 1], dtype=torch.float32)

        with torch.no_grad():
            loss = router_loss(router, phi_i, logits_i, label_i, log_prior_i, temperature)
            w = router_weights(router, phi_i, log_prior_i)
            z_bar = _mixture_logits(w, logits_i)
            pred = int(z_bar.argmax(dim=-1).item())

        losses.append(float(loss.item()))
        correct += int(pred == int(labels_cal[i]))

    return {
        "router_type": router_type,
        "n": n,
        "mean_loss": float(np.mean(losses)),
        "accuracy": correct / n,
        **hparams,
    }
