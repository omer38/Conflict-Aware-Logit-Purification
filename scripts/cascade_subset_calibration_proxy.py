"""Honest deployment accounting for the entropy cascade (progress_report5_draft.md
§7 pending item iv; student_level_calibration_cascade_results.md caveat (iii)).

The production cascade (``src.student.soft_labels.cascade_teacher_soft_labels``)
aligns *all* K teachers (class-prior bias + confidence-scale temperature)
using their full-pool (3N) logits before running the cascade. That is a
"cheat": in a genuine 1.4 q/N deployment you never query teacher 2 or 3 on the
full pool, so you cannot compute their full-pool alignment constants for
free.

This script compares that full-pool ("dishonest") alignment against a
subset-only ("honest") variant where each escalated teacher's prior-matching
bias and confidence-scale temperature are estimated *only* from the subset of
samples it was actually queried on (the ask-2 / ask-3 subsets):

  - Teacher 1 (queried on 100% of the pool) is aligned on the full pool in
    both variants -- this is always legitimate.
  - Teacher 1's aligned mean entropy (full pool) is the scale-alignment
    target in both variants.
  - Teachers 2 and 3 (honest variant): prior-matching bias and confidence-scale
    temperature are fit using only their own escalated subset (ask2 / ask3
    rows), which also breaks the "pool is class-balanced by construction"
    assumption uniform class-prior matching relies on -- we report the escalated
    subsets' class balance to quantify that risk.

Reports accuracy / NLL of the final cascade target under both variants, plus
the realised queries/sample and the escalated-subset class-balance deviation,
for all four datasets. Cached logits only, no GPU.

Usage: uv run python scripts/cascade_subset_calibration_proxy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent / "outputs" / "reports"
TEACHERS = (
    "qwen3.5-27b-vllm-guided",
    "gemma4-26b-vllm-guided",
    "mistral-small-24b-vllm-guided",
)
DATASETS = {"ag_news": 4, "yahoo": 10, "amazon": 5, "amazon3": 3}
TARGET_QUERIES = 1.4


def load(ds: str, C: int) -> tuple[np.ndarray, np.ndarray]:
    import csv

    logits, labels = [], None
    for t in TEACHERS:
        rows = sorted(
            csv.DictReader(open(BASE / f"{t}_{ds}.csv", encoding="utf-8")),
            key=lambda r: int(r["sample_idx"]),
        )
        z = np.array([[float(r[f"logit_{chr(65 + c)}"]) for c in range(C)] for r in rows])
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


def nll_of(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def uniform_prior_bias(z: np.ndarray, C: int, n_iter: int = 100) -> np.ndarray:
    """Additive per-class bias so the mean prediction over *z*'s own rows is
    uniform. Fit on whatever subset of rows is passed in."""
    b = np.zeros(C)
    for _ in range(n_iter):
        p_mean = softmax(z + b).mean(0)
        b += np.log(1.0 / C) - np.log(np.clip(p_mean, 1e-12, None))
    return b


def equalize_temperature(z: np.ndarray, target_entropy: float) -> float:
    """Bisect a temperature so mean entropy of softmax(z/t) over *z*'s own
    rows matches *target_entropy*."""
    lo, hi = 0.25, 16.0
    for _ in range(50):
        t = 0.5 * (lo + hi)
        if entropy_rows(softmax(z / t)).mean() < target_entropy:
            lo = t
        else:
            hi = t
    return 0.5 * (lo + hi)


def class_balance_deviation(y_subset: np.ndarray, C: int) -> float:
    """Max absolute deviation of the subset's class proportions from uniform
    (1/C); 0 = perfectly balanced, matches the doc's "pool is class-balanced
    by construction" assumption."""
    if len(y_subset) == 0:
        return float("nan")
    counts = np.bincount(y_subset, minlength=C).astype(np.float64)
    props = counts / counts.sum()
    return float(np.max(np.abs(props - 1.0 / C)))


def _escalation_masks(zs: np.ndarray, C: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Which samples get escalated, using the production procedure (full-pool
    alignment for the escalation *decision* only -- this part uses no individual
    labels but assumes the known uniform pool prior, and is identical to
    ``cascade_teacher_soft_labels``. Shared by both variants below so the
    honesty check isolates the alignment
    *constants* used for the final targets, not the escalation policy itself.
    """
    b1 = uniform_prior_bias(zs[0], C)
    p1 = softmax(zs[0] + b1)
    target_entropy = entropy_rows(p1).mean()

    b2_full = uniform_prior_bias(zs[1], C)
    t2_full = equalize_temperature(zs[1] + b2_full, target_entropy)
    p2_full = softmax((zs[1] + b2_full) / t2_full)

    ent1 = entropy_rows(p1)
    votes1 = p1.argmax(-1)
    votes2_full = p2_full.argmax(-1)

    lo, hi = float(ent1.min()) - 1e-6, float(ent1.max()) + 1e-6
    for _ in range(60):
        h = 0.5 * (lo + hi)
        ask2_try = ent1 > h
        ask3_try = ask2_try & (votes1 != votes2_full)
        queries = 1.0 + float(ask2_try.mean()) + float(ask3_try.mean())
        if queries > TARGET_QUERIES:
            lo = h
        else:
            hi = h
    h = 0.5 * (lo + hi)
    ask2 = ent1 > h
    ask3 = ask2 & (votes1 != votes2_full)
    return ask2, ask3, p1, target_entropy


def run_cascade(zs: np.ndarray, y: np.ndarray, C: int, honest: bool) -> dict:
    ask2, ask3, p1, target_entropy = _escalation_masks(zs, C)

    if honest:
        b2 = uniform_prior_bias(zs[1][ask2], C)
        t2 = equalize_temperature(zs[1][ask2] + b2, target_entropy)
        b3 = uniform_prior_bias(zs[2][ask3], C) if ask3.any() else np.zeros(C)
        t3 = equalize_temperature(zs[2][ask3] + b3, target_entropy) if ask3.any() else 1.0
    else:
        b2 = uniform_prior_bias(zs[1], C)
        t2 = equalize_temperature(zs[1] + b2, target_entropy)
        b3 = uniform_prior_bias(zs[2], C)
        t3 = equalize_temperature(zs[2] + b3, target_entropy)
    p2 = softmax((zs[1] + b2) / t2)
    p3 = softmax((zs[2] + b3) / t3)

    p_bar = p1.copy()
    two_only = ask2 & ~ask3
    p_bar[two_only] = 0.5 * (p1[two_only] + p2[two_only])
    p_bar[ask3] = (p1[ask3] + p2[ask3] + p3[ask3]) / 3.0

    realised_queries = 1.0 + float(ask2.mean()) + float(ask3.mean())
    return {
        "accuracy": float((p_bar.argmax(-1) == y).mean()),
        "nll": nll_of(p_bar, y),
        "queries_per_sample": realised_queries,
        "ask2_rate": float(ask2.mean()),
        "ask3_rate": float(ask3.mean()),
        "ask2_class_balance_dev": class_balance_deviation(y[ask2], C),
        "ask3_class_balance_dev": class_balance_deviation(y[ask3], C) if ask3.any() else float("nan"),
        "pool_class_balance_dev": class_balance_deviation(y, C),
    }


def main() -> None:
    results = []
    for ds, C in DATASETS.items():
        zs, y = load(ds, C)
        dishonest = run_cascade(zs, y, C, honest=False)
        honest = run_cascade(zs, y, C, honest=True)
        row = {"dataset": ds, "full_pool_alignment": dishonest, "subset_only_alignment": honest}
        results.append(row)

        print(f"\n===== {ds} (N={len(y)}, C={C}) =====")
        print(f"  pool class-balance deviation:        {dishonest['pool_class_balance_dev']:.4f}")
        print(f"  ask2 rate / ask3 rate:                {dishonest['ask2_rate']:.3f} / {dishonest['ask3_rate']:.3f}")
        print(f"  ask2 subset class-balance deviation: {dishonest['ask2_class_balance_dev']:.4f}")
        print(f"  ask3 subset class-balance deviation: {dishonest['ask3_class_balance_dev']:.4f}")
        print(
            f"  full-pool alignment:   acc={dishonest['accuracy']:.4f}  nll={dishonest['nll']:.4f}  "
            f"q/N={dishonest['queries_per_sample']:.3f}"
        )
        print(
            f"  subset-only alignment: acc={honest['accuracy']:.4f}  nll={honest['nll']:.4f}  "
            f"q/N={honest['queries_per_sample']:.3f}"
        )
        print(
            f"  delta (subset - full pool): acc={honest['accuracy'] - dishonest['accuracy']:+.4f}  "
            f"nll={honest['nll'] - dishonest['nll']:+.4f}"
        )

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cascade_subset_calibration_proxy.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")

    lines = [
        "| Dataset | Pool balance dev | ask2/ask3 rate | Full-pool alignment (acc/NLL) | "
        "Subset-only alignment (acc/NLL) | Delta acc | Delta NLL |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results:
        d, hn = row["full_pool_alignment"], row["subset_only_alignment"]
        lines.append(
            f"| {row['dataset']} | {d['pool_class_balance_dev']:.3f} | "
            f"{d['ask2_rate']:.2%} / {d['ask3_rate']:.2%} | {d['accuracy']:.4f} / {d['nll']:.4f} | "
            f"{hn['accuracy']:.4f} / {hn['nll']:.4f} | {hn['accuracy'] - d['accuracy']:+.4f} | "
            f"{hn['nll'] - d['nll']:+.4f} |"
        )
    table_path = out_dir / "cascade_subset_calibration_proxy.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
