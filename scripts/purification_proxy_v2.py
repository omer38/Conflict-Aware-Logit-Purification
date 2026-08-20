"""Second round of purification-quality proxy experiments on cached logits.

Terminology note: the historical experiment called its operations "marginal
calibration" and "entropy equalization". They are now described as class-prior
matching and confidence-scale alignment. The former assumes a supplied prior;
the latter does not guarantee reliability calibration.

Follows up scripts/router_proxy_experiments.py with label-free purification
ideas that attack failure modes the first round exposed:

  1. Gemma is severely overconfident (NLL 1.7-4.2, entropy ~0): its large
     logit magnitudes dominate the logit-mean. -> per-teacher *entropy
     confidence-scale alignment* (label-independent temperature) and *logit standardization*
     (per-sample z-score, Sun et al. CVPR'24).
  2. The working pool is class-balanced by construction, so each teacher's
     average prediction should be uniform; deviations are systematic class
     bias. -> *class-prior matching* under the known uniform pool prior and
     *Sinkhorn balancing* of the consensus.
  3. In a retrospective gold-labelled analysis, delta_k = KL(p_k || p_bar)
     is the strongest teacher-specific feature associated with harm. ->
     *consensus-proximity weighting* w_k ~ exp(-beta * delta_k): a continuous
     per-sample rule that uses delta_k itself, not the unavailable harm label,
     to down-weight outlier teachers instead of picking winners.
  4. Per-sample routers are unlearnable from 12-30 cal points, but a single
     *global* convex weight vector (2 free params) might not be. -> global
     simplex weights selected by cal-set NLL (proper score, finer resolution
     than 0/1 accuracy), 20 cal draws.
  5. The oracle showed target *sharpness* is the lever that survives
     distillation. -> *agreement-routed temperature*: sharpen the consensus
     where all teachers agree, soften it where they disagree.

All methods output a final (N, C) target distribution; we report argmax
accuracy, NLL of the true class (proper score = soft-target quality), and
mean entropy (dark-knowledge proxy), overall and on the disagreement subset.

Usage: uv run python scripts/purification_proxy_v2.py
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent / "outputs" / "reports"
TEACHERS = (
    "qwen3.5-27b-vllm-guided",
    "gemma4-26b-vllm-guided",
    "mistral-small-24b-vllm-guided",
)
SHORT = ("qwen", "gemma", "mistral")
DATASETS = {"ag_news": 4, "yahoo": 10, "amazon": 5}
CAL_SIZES = {"ag_news": 12, "yahoo": 30, "amazon": 15}  # 30% of L=10 per class
N_CAL_SEEDS = 20


def load(ds: str, C: int) -> tuple[np.ndarray, np.ndarray]:
    logits, labels = [], None
    for t in TEACHERS:
        rows = sorted(
            csv.DictReader(open(BASE / f"{t}_{ds}.csv", encoding="utf-8")),
            key=lambda r: int(r["sample_idx"]),
        )
        z = np.array(
            [[float(r[f"logit_{chr(65 + c)}"]) for c in range(C)] for r in rows]
        )
        logits.append(z)
        if labels is None:
            labels = np.array([int(r["true_label_id"]) for r in rows])
    return np.stack(logits), labels  # (K,N,C), (N,)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def nll_of(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def entropy_rows(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum(-1)


# ---------------------------------------------------------------- methods


def match_class_prior(
    z: np.ndarray,
    target_prior: np.ndarray,
    n_iter: int = 100,
) -> np.ndarray:
    """Match each teacher's pool-average prediction to ``target_prior``.

    No individual labels are used during fitting, but the prior is supplied.
    Returns bias-adjusted logits, same shape (K, N, C).
    """
    C = z.shape[-1]
    out = np.empty_like(z)
    for k in range(z.shape[0]):
        b = np.zeros(C)
        for _ in range(n_iter):
            p_mean = softmax(z[k] + b).mean(0)
            b += np.log(target_prior) - np.log(np.clip(p_mean, 1e-12, None))
        out[k] = z[k] + b
    return out


def align_confidence_scales(z: np.ndarray) -> np.ndarray:
    """Per-teacher temperature so all teachers have the same mean entropy.

    Target = the highest per-teacher mean entropy at t=1 (soften overconfident
    teachers to match the least confident one). This does not establish empirical
    probability calibration. Returns scaled logits.
    """
    target = max(entropy_rows(softmax(z[k])).mean() for k in range(z.shape[0]))
    out = np.empty_like(z)
    for k in range(z.shape[0]):
        lo, hi = 0.25, 16.0
        for _ in range(50):
            t = 0.5 * (lo + hi)
            if entropy_rows(softmax(z[k] / t)).mean() < target:
                lo = t
            else:
                hi = t
        out[k] = z[k] / (0.5 * (lo + hi))
    return out


def logit_standardize(z: np.ndarray) -> np.ndarray:
    """Per-sample per-teacher z-score of logits (Logit Standardization, CVPR'24)."""
    mu = z.mean(-1, keepdims=True)
    sd = z.std(-1, keepdims=True)
    return (z - mu) / np.clip(sd, 1e-8, None)


def consensus_proximity(p: np.ndarray, beta: float, n_iter: int = 2) -> np.ndarray:
    """w_k(x) ~ exp(-beta * KL(p_k || p_bar)): continuous outlier down-weighting."""
    pbar = p.mean(0)
    for _ in range(n_iter):
        delta = np.stack([kl_rows(p[k], pbar) for k in range(p.shape[0])])  # (K,N)
        w = np.exp(-beta * delta)
        w /= w.sum(0, keepdims=True)
        pbar = (w[:, :, None] * p).sum(0)
    return pbar


def sinkhorn_balance(p: np.ndarray, C: int, n_iter: int = 20) -> np.ndarray:
    """Rescale a (N,C) distribution so class marginals over the pool are uniform."""
    out = p.copy()
    for _ in range(n_iter):
        out = out * (1.0 / C) / np.clip(out.mean(0, keepdims=True), 1e-12, None)
        out /= out.sum(1, keepdims=True)
    return out


def agreement_temperature(p: np.ndarray, pbar: np.ndarray) -> np.ndarray:
    """Sharpen consensus where teachers agree, soften where they conflict.

    tau(x) = 0.5 (all 3 agree) / 1.0 (2 agree) / 2.0 (all differ), applied to
    log p_bar. Label-free per-sample temperature routing.
    """
    votes = p.argmax(-1)  # (K, N)
    pairs = list(combinations(range(p.shape[0]), 2))
    agree = np.stack([(votes[i] == votes[j]) for i, j in pairs]).mean(0)  # (N,)
    tau = np.where(agree > 0.99, 0.5, np.where(agree > 0.01, 1.0, 2.0))
    return softmax(np.log(np.clip(pbar, 1e-12, None)) / tau[:, None])


def stratified_cal(y: np.ndarray, C: int, size: int, rng) -> np.ndarray:
    idx = []
    for c in range(C):
        pool = np.where(y == c)[0]
        idx.extend(rng.choice(pool, size=max(1, size // C), replace=False))
    return np.array(idx)


def simplex_grid(K: int, step: float = 0.1) -> np.ndarray:
    n = round(1.0 / step)
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((i * step, j * step, 1.0 - (i + j) * step))
    return np.array(pts)


# ---------------------------------------------------------------- main


def main() -> None:
    for ds, C in DATASETS.items():
        zs, y = load(ds, C)
        K, N, _ = zs.shape
        p1 = softmax(zs)
        votes = p1.argmax(2)
        dis = ~((votes[0] == votes[1]) & (votes[1] == votes[2]))

        rows: list[tuple] = []

        def ev(p, name):
            rows.append(
                (
                    name,
                    (p.argmax(1) == y).mean(),
                    nll_of(p, y),
                    float(entropy_rows(p).mean()),
                    (p[dis].argmax(1) == y[dis]).mean(),
                    nll_of(p[dis], y[dis]),
                )
            )

        # --- reference points from round 1
        ev(softmax(zs.mean(0)), "logit-mean (baseline#2)")
        pm = p1.mean(0)
        ev(pm, "prob-mean (round-1 best label-free)")

        # --- leave-one-teacher-out prob-means (who is hurting?)
        for k in range(K):
            others = [j for j in range(K) if j != k]
            ev(p1[others].mean(0), f"prob-mean minus {SHORT[k]}")

        # --- 1. class-prior matching (known balanced-pool prior)
        uniform_prior = np.full(C, 1.0 / C)
        z_mc = match_class_prior(zs, uniform_prior)
        ev(softmax(z_mc).mean(0), "prior-match -> prob-mean")

        # --- 2. confidence-scale alignment (label-independent temperature)
        z_eq = align_confidence_scales(zs)
        ev(softmax(z_eq).mean(0), "scale-align -> prob-mean")

        # --- 3. logit standardization then mean
        ev(softmax(logit_standardize(zs).mean(0)), "logit-standardize -> mean")

        # --- 4. consensus-proximity weighting on raw probs
        for beta in (1.0, 2.0):
            ev(consensus_proximity(p1, beta), f"consensus-prox beta={beta}")

        # --- combos: fix calibration first, then robust-combine
        p_fix = softmax(align_confidence_scales(match_class_prior(zs, uniform_prior)))
        ev(p_fix.mean(0), "prior-match + scale-align -> prob-mean")
        ev(consensus_proximity(p_fix, 2.0), "prior-match + scale-align -> cons-prox b=2")

        # --- 5. Sinkhorn balancing of the consensus
        ev(sinkhorn_balance(pm, C), "prob-mean -> sinkhorn-balance")

        # --- 6. agreement-routed temperature on the consensus
        ev(agreement_temperature(p1, pm), "prob-mean -> agree-tau (0.5/1/2)")
        ev(
            agreement_temperature(p1, p_fix.mean(0)),
            "prior-match + scale-align -> agree-tau",
        )

        # --- 7. global convex weights from tiny cal (NLL-selected, 20 seeds)
        grid = simplex_grid(K)
        accs, nlls = [], []
        for s in range(N_CAL_SEEDS):
            rng = np.random.default_rng(s)
            cal = stratified_cal(y, C, CAL_SIZES[ds], rng)
            best_w, best_nll = None, np.inf
            for w in grid:
                p_cal = np.einsum("k,knc->nc", w, p1[:, cal])
                v = nll_of(p_cal, y[cal])
                if v < best_nll:
                    best_nll, best_w = v, w
            p_full = np.einsum("k,knc->nc", best_w, p1)
            accs.append((p_full.argmax(1) == y).mean())
            nlls.append(nll_of(p_full, y))
        rows.append(
            (
                f"global simplex weights (cal={CAL_SIZES[ds]}, NLL-sel)",
                (np.mean(accs), np.std(accs)),
                (np.mean(nlls), np.std(nlls)),
                float("nan"),
                float("nan"),
                float("nan"),
            )
        )

        # --- oracle reference
        conf_true = p1[:, np.arange(N), y]
        w_or = softmax(np.log(np.clip(conf_true, 1e-12, None)) / 3.0, axis=0)
        ev(softmax(np.einsum("kn,knc->nc", w_or, zs)), "ORACLE weight tau_or=3")

        print(f"\n===== {ds} (N={N}, C={C}, disagree-rate={dis.mean():.3f}) =====")
        print(
            f"{'method':44s} {'acc':>15s} {'nll':>15s} {'H':>6s} "
            f"{'acc|dis':>8s} {'nll|dis':>8s}"
        )
        for name, acc, nll, ent, acc_d, nll_d in rows:
            if isinstance(acc, tuple):
                print(
                    f"{name:44s} {acc[0]:.4f}±{acc[1]:.4f} {nll[0]:.4f}±{nll[1]:.4f}"
                )
            else:
                print(
                    f"{name:44s} {acc:15.4f} {nll:15.4f} {ent:6.3f} "
                    f"{acc_d:8.4f} {nll_d:8.4f}"
                )


if __name__ == "__main__":
    main()
