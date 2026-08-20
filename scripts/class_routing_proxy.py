"""Class-conditional routing proxy: is per-CLASS teacher expertise exploitable?

Per-sample routing failed (report §5-6), but Figure 5/8 show teachers differ
per class (e.g. Mistral on Family & Rel., Qwen on Sports). Class-conditional
weighting has K*C parameters, shared across thousands of samples -- a far
easier estimation problem than per-sample weights. The catch: at aggregation
time the true class is unknown (chicken-and-egg), so we compare:

  oracle ceilings (condition on TRUE class; not deployable):
    - best single teacher per true class
    - best convex weights per true class (grid)
  deployable estimators:
    - weights conditioned on the consensus PREDICTED class (per-class
      accuracies estimated from pool labels = diagnostic upper-ish bound)
    - soft Dawid-Skene EM on prior-matched, scale-aligned probabilities
      matrices ARE the per-class expertise, posterior does the routing)
    - DS blended with the aligned mean

All on top of class-prior matching under the known uniform pool prior plus
confidence-scale alignment, the strongest base from
scripts/purification_proxy_v2.py. Neither operation is conventional empirical
probability calibration.

Usage: uv run python scripts/class_routing_proxy.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent / "outputs" / "reports"
TEACHERS = (
    "qwen3.5-27b-vllm-guided",
    "gemma4-26b-vllm-guided",
    "mistral-small-24b-vllm-guided",
)
DATASETS = {"ag_news": 4, "yahoo": 10, "amazon": 5}


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
    return np.stack(logits), labels


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def entropy_rows(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def match_uniform_class_prior(z: np.ndarray, C: int, n_iter: int = 100) -> np.ndarray:
    out = np.empty_like(z)
    for k in range(z.shape[0]):
        b = np.zeros(C)
        for _ in range(n_iter):
            p_mean = softmax(z[k] + b).mean(0)
            b += np.log(1.0 / C) - np.log(np.clip(p_mean, 1e-12, None))
        out[k] = z[k] + b
    return out


def align_confidence_scales(z: np.ndarray) -> np.ndarray:
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


def nll_of(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def simplex_grid(step: float = 0.1) -> np.ndarray:
    n = round(1.0 / step)
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((i * step, j * step, 1.0 - (i + j) * step))
    return np.array(pts)


def soft_dawid_skene(p: np.ndarray, C: int, n_iter: int = 50, smooth: float = 1.0):
    """EM with soft votes. p: (K,N,C) aligned teacher probabilities.

    Returns (posterior (N,C), confusion matrices (K,C,C) with conf[k, true, said]).
    """
    K, N, _ = p.shape
    mu = p.mean(0)  # init with consensus
    log_p = np.log(np.clip(p, 1e-12, None))
    for _ in range(n_iter):
        # M-step: expected confusion matrices under posterior mu
        conf = np.einsum("nc,knd->kcd", mu, p) + smooth
        conf /= conf.sum(-1, keepdims=True)
        # E-step: posterior with uniform prior (pool is class-balanced)
        log_mu = np.einsum("knd,kcd->nc", p, np.log(np.clip(conf, 1e-12, None)))
        log_mu -= log_mu.max(1, keepdims=True)
        mu = np.exp(log_mu)
        mu /= mu.sum(1, keepdims=True)
    return mu, conf


def main() -> None:
    grid = simplex_grid()
    for ds, C in DATASETS.items():
        zs, y = load(ds, C)
        K, N, _ = zs.shape
        p = softmax(align_confidence_scales(match_uniform_class_prior(zs, C)))  # (K,N,C)
        pbar = p.mean(0)
        votes = p.argmax(-1)

        rows: list[tuple[str, float, float]] = []

        def ev(pt, name):
            rows.append((name, float((pt.argmax(1) == y).mean()), nll_of(pt, y)))

        ev(pbar, "prior-and-scale-aligned prob-mean (base)")

        # --- oracle: best single teacher per TRUE class
        acc_kc = np.stack(
            [[(votes[k][y == c] == c).mean() for c in range(C)] for k in range(K)]
        )  # (K,C) per-teacher per-class accuracy (pool labels)
        best_k = acc_kc.argmax(0)  # (C,)
        pred_sel = votes[best_k[y], np.arange(N)]
        rows.append(
            ("ORACLE best teacher per TRUE class", float((pred_sel == y).mean()), float("nan"))
        )

        # --- oracle: best convex weights per TRUE class
        pt = np.empty_like(pbar)
        for c in range(C):
            idx = np.where(y == c)[0]
            best_acc, best_p = -1.0, None
            for w in grid:
                mix = np.einsum("k,knc->nc", w, p[:, idx])
                a = (mix.argmax(1) == c).mean()
                if a > best_acc:
                    best_acc, best_p = a, mix
            pt[idx] = best_p
        ev(pt, "ORACLE convex weights per TRUE class")

        # --- deployable-ish: weights conditioned on consensus PREDICTED class
        # (per-class accuracy from pool labels; selection uses only predictions)
        chat = pbar.argmax(1)
        w_c = acc_kc / acc_kc.sum(0, keepdims=True)  # (K,C) normalised expertise
        mix = np.einsum("kn,knc->nc", w_c[:, chat], p)
        ev(mix, "expertise weights @ predicted class")

        # --- Unsupervised soft Dawid-Skene EM on aligned probabilities.
        mu, conf = soft_dawid_skene(p, C)
        ev(mu, "soft Dawid-Skene posterior (0 labels)")
        for lam in (0.25, 0.5):
            ev((1 - lam) * pbar + lam * mu, f"aligned-mean + {lam}*DS blend (0 labels)")

        print(f"\n===== {ds} (N={N}, C={C}) =====")
        print("per-class acc spread (max-min over teachers), mean over classes: "
              f"{(acc_kc.max(0) - acc_kc.min(0)).mean():.3f}")
        print(f"{'method':44s} {'acc':>8s} {'nll':>8s}")
        for name, a, nll in rows:
            print(f"{name:44s} {a:8.4f} {nll:8.4f}")


if __name__ == "__main__":
    main()
