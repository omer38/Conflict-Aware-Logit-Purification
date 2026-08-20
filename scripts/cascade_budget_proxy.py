"""RQ2 proxy: disagreement-aware teacher-query budgeting from cached logits.

Simulates building the KD target set under a query budget T < 3N using a
confidence cascade:

  1. query teacher 1 on every sample;
  2. if its prior-matched, scale-aligned prediction entropy exceeds a threshold h, query
     teacher 2;
  3. if teachers 1 and 2 disagree on the argmax, query teacher 3 as a
     tie-breaker;
  4. aggregate whatever was queried with the prior-and-scale-aligned prob-mean.

Sweeping h traces an accuracy-vs-budget curve. The uniform-allocation
baseline queries all 3 teachers on a random fraction f of samples and a
single teacher on the rest (expectation over which teacher is first /
single, since under label scarcity the best teacher is unknown a priori).

Also reports single-teacher accuracy after class-prior matching and
confidence-scale alignment, to separate alignment effects from ensembling.
The matching assumes the known uniform prior induced by the gold-balanced pool;
it is not conventional empirical probability calibration.

Usage: uv run python scripts/cascade_budget_proxy.py
"""

from __future__ import annotations

import csv
from itertools import permutations
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent / "outputs" / "reports"
TEACHERS = (
    "qwen3.5-27b-vllm-guided",
    "gemma4-26b-vllm-guided",
    "mistral-small-24b-vllm-guided",
)
SHORT = ("qwen", "gemma", "mistral")
DATASETS = {"ag_news": 4, "yahoo": 10, "amazon": 5, "amazon3": 3}
THRESH_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


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


def cascade(p: np.ndarray, order: tuple[int, int, int], h: float, y: np.ndarray):
    """Run the entropy cascade; return (accuracy, mean queries per sample)."""
    _, N, _ = p.shape
    k1, k2, k3 = order
    ent1 = entropy_rows(p[k1])
    ask2 = ent1 > h
    votes1 = p[k1].argmax(-1)
    votes2 = p[k2].argmax(-1)
    ask3 = ask2 & (votes1 != votes2)

    target = p[k1].copy()
    two = ask2 & ~ask3
    target[two] = 0.5 * (p[k1][two] + p[k2][two])
    target[ask3] = (p[k1][ask3] + p[k2][ask3] + p[k3][ask3]) / 3.0

    acc = float((target.argmax(-1) == y).mean())
    queries = float(1.0 + ask2.mean() + ask3.mean())
    return acc, queries


def main() -> None:
    for ds, C in DATASETS.items():
        zs, y = load(ds, C)
        K, N, _ = zs.shape
        p = softmax(align_confidence_scales(match_uniform_class_prior(zs, C)))

        singles = [(p[k].argmax(-1) == y).mean() for k in range(K)]
        full = ((p.mean(0)).argmax(-1) == y).mean()
        print(f"\n===== {ds} (N={N}, C={C}) =====")
        print("aligned singles: " + "  ".join(f"{SHORT[k]}={singles[k]:.4f}" for k in range(K)))
        print(f"aligned prob-mean (3N queries): {full:.4f}   expected single (1N): {np.mean(singles):.4f}")

        print(f"{'policy':34s} {'queries/N':>9s} {'acc':>8s}")
        # uniform-allocation baseline: all-3 on fraction f, one random teacher on rest
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            acc = f * full + (1 - f) * np.mean(singles)
            print(f"uniform alloc f={f:<4}                  {1 + 2 * f:9.2f} {acc:8.4f}")

        # cascade, averaged over all teacher orders (best teacher unknown a priori)
        for q in THRESH_QUANTILES:
            accs, costs = [], []
            for order in permutations(range(K)):
                h = np.quantile(entropy_rows(p[order[0]]), 1.0 - q) if q > 0 else np.inf
                a, c = cascade(p, order, h, y)
                accs.append(a)
                costs.append(c)
            print(
                f"cascade top-{q:<4} entropy escalation    "
                f"{np.mean(costs):9.2f} {np.mean(accs):8.4f}  (order-avg)"
            )


if __name__ == "__main__":
    main()
