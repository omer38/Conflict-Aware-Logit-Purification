#!/usr/bin/env python
"""Publication-quality figures from cached teacher logits (no API calls).

Produces, per teacher/dataset, vector PDFs in outputs/figures/:

  1. {teacher}_{dataset}_confusion.pdf    -- row-normalized confusion matrix
  2. {teacher}_{dataset}_confidence.pdf   -- confidence histogram, correct vs wrong
  3. {teacher}_{dataset}_floor.pdf        -- floored-classes-per-sample distribution
  4. {teacher}_{dataset}_reliability.pdf  -- reliability diagram (calibration)

Usage:
    uv run python scripts/plot_logits.py --teacher gpt-4.1-mini --dataset amazon
    uv run python scripts/plot_logits.py --teacher gpt-4.1-mini --dataset ag_news --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import TextClassificationDataset, load_dataset_by_name
from src.teachers.llm import TEACHER_PRESETS
from src.teachers.extraction import load_cached_logits

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

SINGLE_COL = 3.3  # inches, typical single-column width


def softmax_observed(logits: np.ndarray, floored: np.ndarray) -> np.ndarray:
    """Row-wise softmax over observed entries only; floored classes get zero mass.

    This keeps the imputed floor value out of the visualized distributions.
    """
    masked = np.where(floored, -np.inf, logits.astype(np.float64))
    shifted = masked - masked.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=1, keepdims=True)


def plot_confusion(
    labels: np.ndarray,
    preds: np.ndarray,
    dataset: TextClassificationDataset,
    path: Path,
) -> None:
    c = dataset.num_classes
    matrix = np.zeros((c, c))
    np.add.at(matrix, (labels, preds), 1)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    side = max(SINGLE_COL, 0.32 * c)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    # Banking15 names are long; shorten for tick labels.
    names = [n[:24] + "…" if len(n) > 25 else n for n in dataset.label_names]
    ax.set_xticks(range(c), names, rotation=90)
    ax.set_yticks(range(c), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    # Annotate cells when the grid is small enough to stay legible.
    if c <= 15:
        for i in range(c):
            for j in range(c):
                if norm[i, j] >= 0.005:
                    color = "white" if norm[i, j] > 0.5 else "black"
                    ax.text(
                        j, i, f"{norm[i, j]:.2f}",
                        ha="center", va="center", color=color, fontsize=6,
                    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized frequency")
    fig.savefig(path)
    plt.close(fig)


def plot_confidence_hist(
    confidence: np.ndarray,
    correct: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    bins = np.linspace(0, 1, 21)
    ax.hist(
        confidence[correct], bins=bins, alpha=0.7, label="Correct",
        color="#2c7bb6", edgecolor="white", linewidth=0.3,
    )
    ax.hist(
        confidence[~correct], bins=bins, alpha=0.7, label="Incorrect",
        color="#d7191c", edgecolor="white", linewidth=0.3,
    )
    ax.set_xlabel("Teacher confidence (max softmax probability)")
    ax.set_ylabel("Samples")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_floor_hist(num_floored: np.ndarray, num_classes: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    counts = np.bincount(num_floored, minlength=num_classes + 1)
    ax.bar(range(num_classes + 1), counts, color="#5e6b7a", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Floored classes per sample (absent from top-20 logprobs)")
    ax.set_ylabel("Samples")
    ax.set_xticks(range(0, num_classes + 1, max(1, (num_classes + 1) // 8)))
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(
    confidence: np.ndarray,
    correct: np.ndarray,
    path: Path,
    n_bins: int = 10,
) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)

    bin_conf, bin_acc, bin_frac = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_conf.append(confidence[mask].mean())
        bin_acc.append(correct[mask].mean())
        bin_frac.append(mask.mean())

    ece = float(np.sum([f * abs(a - c) for f, a, c in zip(bin_frac, bin_acc, bin_conf)]))

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey", label="Perfect calibration")
    ax.plot(bin_conf, bin_acc, marker="o", ms=4, lw=1.2, color="#2c7bb6", label="Teacher")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.97, 0.05, f"ECE = {ece:.3f}", ha="right", va="bottom", transform=ax.transAxes)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures from cached teacher logits.")
    parser.add_argument(
        "--teacher",
        required=True,
        choices=list(TEACHER_PRESETS.keys()),
        help="Teacher model preset name.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amazon", "yahoo", "ag_news"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/logits",
        help="Directory containing cached NPZ files (default: cache/logits).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/figures",
        help="Directory for output PDFs (default: outputs/figures).",
    )
    args = parser.parse_args()

    config = TEACHER_PRESETS[args.teacher]
    cached = load_cached_logits(config.name, args.dataset, args.cache_dir)
    if cached is None:
        print(
            f"No cache found at {args.cache_dir}/{config.name}_{args.dataset}.npz. "
            "Run extract_logits.py first."
        )
        sys.exit(1)

    logits, extracted_mask, floored = cached
    dataset = load_dataset_by_name(args.dataset)

    idx = np.where(extracted_mask)[0]
    if idx.size == 0:
        print("Cache exists but no samples are extracted yet.")
        sys.exit(1)

    logits = logits[idx]
    floored = floored[idx]
    labels = np.asarray(dataset.train[dataset.label_column])[idx]

    probs = softmax_observed(logits, floored)
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = preds == labels
    num_floored = floored.sum(axis=1).astype(int)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.name}_{dataset.name}"

    plot_confusion(labels, preds, dataset, out_dir / f"{stem}_confusion.pdf")
    plot_confidence_hist(confidence, correct, out_dir / f"{stem}_confidence.pdf")
    plot_floor_hist(num_floored, dataset.num_classes, out_dir / f"{stem}_floor.pdf")
    plot_reliability(confidence, correct, out_dir / f"{stem}_reliability.pdf")

    print(f"Teacher: {config.name}  Dataset: {dataset.name}")
    print(f"Samples: {idx.size}  Accuracy: {correct.mean():.1%}  Mean floored: {num_floored.mean():.2f}")
    print(f"Figures written to {out_dir}/{stem}_*.pdf")


if __name__ == "__main__":
    main()
