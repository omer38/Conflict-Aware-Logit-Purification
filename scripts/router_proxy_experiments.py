"""Purification-quality proxy experiments on cached guided teacher logits.

Evaluates candidate multi-teacher aggregation schemes WITHOUT student training,
using three target-quality metrics against ground-truth train-pool labels:

  - acc: argmax accuracy of the purified distribution (top-1 quality)
  - nll: negative log-likelihood of the true class (proper score; soft quality)
  - H:   mean entropy of the target (dark-knowledge proxy; too low = one-hot)

Methods:
  baselines : single teachers, logit-mean (#2), entropy-weighted (#3)
  probmean  : arithmetic probability mean (calibration-robust, label-free)
  calibmean : per-teacher temperature scaling fit on a tiny labelled cal set
              (1 param/teacher), then prob-mean; averaged over 20 cal draws
  dropout   : drop the max-KL-from-consensus teacher on disagreement samples
  ds        : label-free Dawid-Skene EM posterior (+ blend with mean)
  oracles   : per-sample best-teacher selection; soft oracle weighting

Usage: uv run python scripts/router_proxy_experiments.py
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


def entropy_of(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum(1).mean())


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    grid = np.geomspace(0.5, 8, 20)
    return float(min(grid, key=lambda t: nll_of(softmax(z / t), y)))


def stratified_cal(y: np.ndarray, C: int, size: int, rng) -> np.ndarray:
    idx = []
    for c in range(C):
        pool = np.where(y == c)[0]
        idx.extend(rng.choice(pool, size=max(1, size // C), replace=False))
    return np.array(idx)


def dawid_skene(votes: np.ndarray, C: int, n_iter: int = 30, smooth: float = 0.1):
    """Hard-vote Dawid-Skene EM; returns (N,C) posterior over the true label."""
    K, N = votes.shape
    mu = np.zeros((N, C))
    for i in range(K):
        mu[np.arange(N), votes[i]] += 1
    mu /= mu.sum(1, keepdims=True)
    for _ in range(n_iter):
        pi = mu.mean(0)
        conf = np.zeros((K, C, C))  # conf[i, true, said]
        for i in range(K):
            for k in range(C):
                conf[i, :, k] = mu[votes[i] == k].sum(0)
            conf[i] += smooth
            conf[i] /= conf[i].sum(1, keepdims=True)
        log_mu = np.log(np.clip(pi, 1e-12, None))[None, :].repeat(N, 0)
        for i in range(K):
            log_mu += np.log(np.clip(conf[i, :, votes[i]], 1e-12, None))
        log_mu -= log_mu.max(1, keepdims=True)
        mu = np.exp(log_mu)
        mu /= mu.sum(1, keepdims=True)
    return mu


def drop_outlier(probs: np.ndarray, dis: np.ndarray) -> np.ndarray:
    """probs (K,N,C): on disagreement samples, drop the teacher with max
    KL-from-consensus and average the rest (probability space)."""
    K = probs.shape[0]
    pbar = probs.mean(0)
    delta = (
        probs
        * (np.log(np.clip(probs, 1e-12, None)) - np.log(np.clip(pbar, 1e-12, None))[None])
    ).sum(2)
    outlier = delta.argmax(0)
    out = pbar.copy()
    for i in range(K):
        sel = dis & (outlier == i)
        others = [j for j in range(K) if j != i]
        out[sel] = probs[others][:, sel].mean(0)
    return out


def main() -> None:
    for ds, C in DATASETS.items():
        zs, y = load(ds, C)
        K, N, _ = zs.shape
        p1 = softmax(zs)
        votes = p1.argmax(2)
        agree = (votes[0] == votes[1]) & (votes[1] == votes[2])
        dis = ~agree
        rows: list[tuple] = []

        def ev(p, name):
            rows.append((name, (p.argmax(1) == y).mean(), nll_of(p, y), entropy_of(p)))

        for i, t in enumerate(TEACHERS):
            ev(p1[i], f"single:{t.split('-')[0]}")
        ev(softmax(zs.mean(0)), "logit-mean (baseline#2)")
        H = -(p1 * np.log(np.clip(p1, 1e-12, None))).sum(2)
        w = np.exp(-H)
        w /= w.sum(0, keepdims=True)
        ev(softmax(np.einsum("kn,knc->nc", w, zs)), "entropy-weighted (baseline#3)")
        ev(p1.mean(0), "prob-mean (label-free)")
        ev(drop_outlier(p1, dis), "prob-mean + drop-outlier (label-free)")

        mu = dawid_skene(votes, C)
        ev(mu, "Dawid-Skene posterior (label-free)")
        pm = p1.mean(0)
        blend = pm.copy()
        blend[dis] = 0.5 * mu[dis] + 0.5 * pm[dis]
        ev(blend, "prob-mean, +DS blend on disagree (label-free)")

        # calibrated prob-mean over tiny-cal draws
        accs, nlls, ents = [], [], []
        for s in range(N_CAL_SEEDS):
            rng = np.random.default_rng(s)
            cal = stratified_cal(y, C, CAL_SIZES[ds], rng)
            taus = [fit_temperature(zs[i][cal], y[cal]) for i in range(K)]
            pc = np.stack([softmax(zs[i] / taus[i]) for i in range(K)]).mean(0)
            accs.append((pc.argmax(1) == y).mean())
            nlls.append(nll_of(pc, y))
            ents.append(entropy_of(pc))
        rows.append(
            (
                f"calibrated prob-mean (cal={CAL_SIZES[ds]}, {N_CAL_SEEDS} seeds)",
                (np.mean(accs), np.std(accs)),
                (np.mean(nlls), np.std(nlls)),
                np.mean(ents),
            )
        )

        conf_true = p1[:, np.arange(N), y]
        ev(softmax(zs[conf_true.argmax(0), np.arange(N)]), "ORACLE best-teacher select")
        w_or = softmax(np.log(np.clip(conf_true, 1e-12, None)) / 3.0, axis=0)
        ev(softmax(np.einsum("kn,knc->nc", w_or, zs)), "ORACLE weight tau_or=3")

        print(f"\n===== {ds} (N={N}, C={C}, agree-rate={agree.mean():.3f}) =====")
        print(f"{'method':52s} {'acc':>15s} {'nll':>15s} {'H':>7s}")
        for name, acc, nll, ent in rows:
            if isinstance(acc, tuple):
                print(
                    f"{name:52s} {acc[0]:.4f}±{acc[1]:.4f} {nll[0]:.4f}±{nll[1]:.4f} {ent:7.3f}"
                )
            else:
                print(f"{name:52s} {acc:15.4f} {nll:15.4f} {ent:7.3f}")


if __name__ == "__main__":
    main()
