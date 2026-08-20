"""
Comprehensive analysis of vLLM / vLLM-guided teacher logits.
Produces tables (printed as Markdown) and publication-quality figures.

Run:
    python3 scripts/analyse_vllm_logits.py
"""

from __future__ import annotations

import csv
import math
import json
import sys
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student.soft_labels import (  # noqa: E402
    _align_confidence_scales_logits,
    _logits_to_probs,
    load_multi_teacher_logits,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR = Path("outputs/figures/vllm_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("outputs/reports")

MODELS = ["gemma4-26b", "mistral-small-24b", "qwen3.5-27b"]
MODES  = ["vllm", "vllm-guided"]
DATASETS = ["ag_news", "yahoo"]

LOGPROB_FLOOR = -30.0

AG_NEWS_LABELS  = ["World", "Sports", "Business", "Sci/Tech"]
YAHOO_LABELS = [
    "Society & Culture", "Science & Math", "Health",
    "Education & Ref", "Computers & Internet", "Sports",
    "Business & Finance", "Entertainment", "Family & Rel.",
    "Politics & Gov.",
]

MODEL_DISPLAY = {
    "gemma4-26b":       "Gemma4-26B",
    "mistral-small-24b":"Mistral-24B",
    "qwen3.5-27b":      "Qwen3.5-27B",
}
MODEL_COLORS = {
    "gemma4-26b":       "#e07b54",
    "mistral-small-24b":"#5b8db8",
    "qwen3.5-27b":      "#6dab6d",
}

plt.rcParams.update({
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
    "savefig.dpi": 150,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softmax(logits: list[float]) -> np.ndarray:
    a = np.array(logits, dtype=np.float64)
    a -= a.max()
    e = np.exp(a)
    return e / e.sum()


def entropy_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def entropy_nats(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


def eff_classes(probs: np.ndarray) -> float:
    """Effective number of classes = 2^H (using bits)."""
    return 2 ** entropy_bits(probs)


def load_csv(model: str, mode: str, dataset: str) -> list[dict]:
    path = REPORT_DIR / f"{model}-{mode}_{dataset}.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_row(row: dict, logit_keys: list[str]):
    logits = np.array([float(row[k]) for k in logit_keys], dtype=np.float64)
    probs  = softmax(logits)
    floored = any(abs(float(row[k]) - LOGPROB_FLOOR) < 0.5 for k in logit_keys)
    return {
        "idx":        int(row["sample_idx"]),
        "true":       int(row["true_label_id"]),
        "pred":       int(row["pred_label_id"]),
        "correct":    int(row["correct"]) == 1,
        "logits":     logits,
        "probs":      probs,
        "max_prob":   float(probs.max()),
        "dark_know":  float(1.0 - probs.max()),
        "entropy":    entropy_nats(probs),
        "eff_cls":    eff_classes(probs),
        "margin":     float(np.sort(probs)[-1] - np.sort(probs)[-2]),
        "floored":    floored,
    }


def load_all(model: str, mode: str, dataset: str) -> list[dict]:
    rows = load_csv(model, mode, dataset)
    logit_keys = [k for k in rows[0].keys() if k.startswith("logit_")]
    return [parse_row(r, logit_keys) for r in rows]


def confidence_aligned_eff_classes(
    models: list[str],
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-model effective classes after label-free scale alignment."""
    teacher_names = [f"{model}-vllm" for model in models]
    num_classes = len(YAHOO_LABELS) if dataset == "yahoo" else len(AG_NEWS_LABELS)
    stacked = load_multi_teacher_logits(
        teacher_names,
        dataset,
        num_classes,
        REPORT_DIR,
    )
    aligned = _align_confidence_scales_logits(stacked)
    probabilities = _logits_to_probs(aligned, temperature=1.0)
    entropy = -(
        probabilities * np.log(np.clip(probabilities, 1e-12, None))
    ).sum(axis=-1)

    reference = load_all(models[0], "vllm", dataset)
    true_labels = np.full(stacked.shape[1], -1, dtype=np.int64)
    for row in reference:
        true_labels[row["idx"]] = row["true"]
    if np.any(true_labels < 0):
        raise ValueError(f"Missing true labels in {models[0]} vllm {dataset} report")
    return np.exp(entropy), true_labels


# ===========================================================================
# TABLE 1 – headline accuracy / floor summary (already known, reformatted)
# ===========================================================================

def table1_accuracy():
    print("\n" + "="*70)
    print("TABLE 1 – Accuracy & Floor Fraction per Model / Mode / Dataset")
    print("="*70)
    print(f"{'Model':<20} {'Mode':<14} {'Dataset':<10} {'n':>6}  {'Acc':>7}  {'Floor%':>7}")
    print("-"*70)
    for model, mode, dataset in product(MODELS, MODES, DATASETS):
        path = REPORT_DIR / f"{model}-{mode}_{dataset}_summary.json"
        s = json.loads(path.read_text())
        print(f"{MODEL_DISPLAY[model]:<20} {mode:<14} {dataset:<10} "
              f"{s['n_samples_reported']:>6}  "
              f"{s['accuracy']:>7.2%}  "
              f"{s['floor_fraction']:>7.2%}")
    print()


# ===========================================================================
# TABLE 2 – confidence statistics
# ===========================================================================

def table2_confidence():
    print("\n" + "="*80)
    print("TABLE 2 – Confidence Statistics (only 'vllm' mode shown; guided is identical)")
    print("  max_prob: softmax probability of predicted class (teacher confidence)")
    print("  margin:   gap between top-1 and top-2 softmax probability")
    print("  corr_conf / wrong_conf: mean max_prob for correct / incorrect samples")
    print("  gap:      correct_conf - wrong_conf  (calibration proxy)")
    print("  hi_conf:  fraction of samples where max_prob > 0.95")
    print("="*80)
    hdr = f"{'Model':<16} {'Dataset':<10} {'mean_mp':>8} {'margin':>8} {'corr':>7} {'wrong':>7} {'gap':>6} {'hi>0.95':>8}"
    print(hdr)
    print("-"*80)
    for model, dataset in product(MODELS, DATASETS):
        data = load_all(model, "vllm", dataset)
        mp    = np.array([d["max_prob"] for d in data])
        margins = np.array([d["margin"]   for d in data])
        correct = np.array([d["correct"]  for d in data])
        corr_mp = mp[correct].mean()
        wrong_mp = mp[~correct].mean() if (~correct).any() else float("nan")
        gap = corr_mp - wrong_mp
        hi  = (mp > 0.95).mean()
        print(f"{MODEL_DISPLAY[model]:<16} {dataset:<10} "
              f"{mp.mean():>8.4f} {margins.mean():>8.4f} "
              f"{corr_mp:>7.4f} {wrong_mp:>7.4f} {gap:>6.4f} {hi:>7.2%}")
    print()


# ===========================================================================
# TABLE 3 – dark knowledge and entropy
# ===========================================================================

def table3_dark_knowledge():
    print("\n" + "="*80)
    print("TABLE 3 – Soft Label Richness (Dark Knowledge) by Model / Dataset")
    print("  dark_know: 1 - max_prob  (probability mass on non-top classes)")
    print("  entropy:   Shannon entropy in nats of the soft label distribution")
    print("  eff_cls:   Effective number of classes = 2^H  (1 = hard label, C = uniform)")
    print("  For KD, higher dark_know / eff_cls means richer pedagogical signal.")
    print("="*80)
    hdr = f"{'Model':<16} {'Dataset':<10} {'dark_know':>10} {'entropy':>9} {'eff_cls':>8}"
    print(hdr)
    print("-"*80)
    for model, dataset in product(MODELS, DATASETS):
        data = load_all(model, "vllm", dataset)
        dk   = np.mean([d["dark_know"] for d in data])
        ent  = np.mean([d["entropy"]   for d in data])
        ec   = np.mean([d["eff_cls"]   for d in data])
        print(f"{MODEL_DISPLAY[model]:<16} {dataset:<10} {dk:>10.4f} {ent:>9.4f} {ec:>8.3f}")
    print()


# ===========================================================================
# TABLE 4 – vllm vs vllm-guided delta
# ===========================================================================

def table4_guided_delta():
    print("\n" + "="*80)
    print("TABLE 4 – Guided vs Unguided Comparison per Model / Dataset")
    print("  floored_samples: samples with ≥1 floored class (imputed -30 floor) in unguided mode")
    print("  changed_preds:   samples where guided changes the argmax prediction")
    print("  mean_logit_diff: mean of max(|logit_i_guided - logit_i_unguided|) per sample")
    print("  dk_delta:        mean dark_know(guided) - mean dark_know(unguided)")
    print("  -> For Mistral/Qwen: truly zero effect; for Gemma: fix for floored classes only")
    print("="*80)
    hdr = f"{'Model':<16} {'Dataset':<10} {'floored_n':>10} {'floored%':>9} {'chg_pred':>9} {'logit_diff':>11} {'dk_delta':>9}"
    print(hdr)
    print("-"*80)
    for model, dataset in product(MODELS, DATASETS):
        d_std  = {r["idx"]: r for r in load_all(model, "vllm",        dataset)}
        d_guid = {r["idx"]: r for r in load_all(model, "vllm-guided", dataset)}
        common = sorted(set(d_std) & set(d_guid))
        floored_n = sum(1 for i in common if d_std[i]["floored"])
        floored_p = floored_n / len(common)
        chg = sum(1 for i in common if d_std[i]["pred"] != d_guid[i]["pred"])
        diffs = [float(np.abs(d_std[i]["logits"] - d_guid[i]["logits"]).max()) for i in common]
        mean_diff = np.mean(diffs)
        dk_delta = np.mean([d_guid[i]["dark_know"] - d_std[i]["dark_know"] for i in common])
        print(f"{MODEL_DISPLAY[model]:<16} {dataset:<10} "
              f"{floored_n:>10} {floored_p:>8.2%} {chg:>9} "
              f"{mean_diff:>11.6f} {dk_delta:>+9.6f}")
    print()


# ===========================================================================
# TABLE 5 – per-class breakdown on Yahoo
# ===========================================================================

def table5_per_class_yahoo():
    print("\n" + "="*90)
    print("TABLE 5 – Per-Class Breakdown on Yahoo (vllm mode)")
    print("  Shows which classes are easy/hard and how much uncertainty each teacher encodes.")
    print("  eff_cls ≈ 1 means teacher is near-deterministic for that class (hard-label-like).")
    print("  eff_cls >> 1 means teacher is genuinely uncertain → rich KD signal.")
    print("="*90)
    for model in MODELS:
        data = load_all(model, "vllm", "yahoo")
        print(f"\n  {MODEL_DISPLAY[model]}")
        print(f"  {'Class':<26} {'n':>5} {'acc':>6} {'conf':>7} {'dk':>7} {'eff_cls':>8}")
        print(f"  {'-'*62}")
        by_class: dict[int, list] = {}
        for d in data:
            by_class.setdefault(d["true"], []).append(d)
        for c in sorted(by_class):
            items = by_class[c]
            acc  = np.mean([d["correct"]  for d in items])
            conf = np.mean([d["max_prob"] for d in items])
            dk   = np.mean([d["dark_know"] for d in items])
            ec   = np.mean([d["eff_cls"]  for d in items])
            print(f"  {YAHOO_LABELS[c]:<26} {len(items):>5} {acc:>6.2%} {conf:>7.3f} {dk:>7.4f} {ec:>8.3f}")
    print()


# ===========================================================================
# FIGURE 1 – Confidence histogram: correct vs wrong, 3 models × 2 datasets
# ===========================================================================

def fig1_confidence_histograms():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharey=False)
    bins = np.linspace(0, 1, 26)

    for col, model in enumerate(MODELS):
        for row, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            data = load_all(model, "vllm", dataset)
            mp   = np.array([d["max_prob"] for d in data])
            corr = np.array([d["correct"]  for d in data])

            ax.hist(mp[corr],  bins=bins, alpha=0.75, color="#2c7bb6",
                    label="Correct", edgecolor="white", linewidth=0.3)
            ax.hist(mp[~corr], bins=bins, alpha=0.75, color="#d7191c",
                    label="Wrong",   edgecolor="white", linewidth=0.3)
            ax.set_yscale("log")

            mean_c = mp[corr].mean()
            mean_w = mp[~corr].mean() if (~corr).any() else float("nan")
            ax.axvline(mean_c, color="#2c7bb6", lw=1.2, ls="--")
            ax.axvline(mean_w, color="#d7191c", lw=1.2, ls="--")

            if row == 0:
                ax.set_title(MODEL_DISPLAY[model])
            if col == 0:
                ax.set_ylabel(f"{dataset.replace('_', ' ').title()}\nSamples (log)")
            if row == 1:
                ax.set_xlabel("Max softmax probability (confidence)")
            if col == 0 and row == 0:
                ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Figure 1 – Teacher confidence distribution: correct vs wrong predictions\n"
        "Dashed lines = group means. Log y-axis. A well-calibrated teacher shows\n"
        "clearly separated means; overconfident teachers (Gemma) overlap heavily.",
        fontsize=8, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig1_confidence_histograms.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# FIGURE 2 – Dark knowledge bar chart: model × dataset
# ===========================================================================

def fig2_dark_knowledge():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    dataset_labels = {"ag_news": "AG News (4 classes)", "yahoo": "Yahoo (10 classes)"}

    for ax, dataset in zip(axes, DATASETS):
        values = []
        colors = []
        labels = []
        for model in MODELS:
            data = load_all(model, "vllm", dataset)
            dk   = np.mean([d["dark_know"] for d in data])
            ec   = np.mean([d["eff_cls"]   for d in data])
            values.append((dk, ec))
            colors.append(MODEL_COLORS[model])
            labels.append(MODEL_DISPLAY[model])

        x = np.arange(len(MODELS))
        dk_vals = [v[0] for v in values]
        ec_vals = [v[1] for v in values]

        bars = ax.bar(x, [v * 100 for v in dk_vals], color=colors, edgecolor="white",
                      linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Dark knowledge  (% of prob. mass on non-top classes)")
        ax.set_title(dataset_labels[dataset])
        ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)

        # Annotate with eff_cls
        for bar, ec in zip(bars, ec_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"eff_cls={ec:.2f}",
                ha="center", va="bottom", fontsize=7, color="#333",
            )

    fig.suptitle(
        "Figure 2 – Soft label richness: dark knowledge & effective number of classes\n"
        "'Dark knowledge' = probability mass the teacher spreads over non-winning classes.\n"
        "eff_cls = 2^H ∈ [1, C]; near 1 means the soft label is essentially a hard label.",
        fontsize=8,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig2_dark_knowledge.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# FIGURE 3 – Reliability (calibration) diagrams
# ===========================================================================

def calibration_curve(data: list[dict], n_bins: int = 10):
    mp   = np.array([d["max_prob"] for d in data])
    corr = np.array([d["correct"]  for d in data], dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx  = np.clip(np.digitize(mp, bins) - 1, 0, n_bins - 1)
    bc, ba, bf = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        bc.append(mp[mask].mean())
        ba.append(corr[mask].mean())
        bf.append(mask.mean())
    ece = float(np.sum([f * abs(a - c) for f, a, c in zip(bf, ba, bc)]))
    return bc, ba, ece


def fig3_reliability():
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5), sharex=True, sharey=True)

    for col, model in enumerate(MODELS):
        for row, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            data = load_all(model, "vllm", dataset)
            bc, ba, ece = calibration_curve(data)

            ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey", label="Perfect")
            ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.07, color="red")
            ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.07, color="blue")
            ax.plot(bc, ba, marker="o", ms=4, lw=1.4,
                    color=MODEL_COLORS[model], label=MODEL_DISPLAY[model])
            ax.text(0.04, 0.92, f"ECE = {ece:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")

            if row == 0:
                ax.set_title(MODEL_DISPLAY[model])
            if col == 0:
                ax.set_ylabel(f"{dataset.replace('_',' ').title()}\nAccuracy")
            if row == 1:
                ax.set_xlabel("Confidence")

    fig.suptitle(
        "Figure 3 – Reliability (calibration) diagrams\n"
        "Red region = over-confident; blue = under-confident. ECE = Expected Calibration Error.\n"
        "A perfect teacher lies on the diagonal. Gemma is severely over-confident on AG News.",
        fontsize=8, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig3_reliability.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# FIGURE 4 – Guided vs Unguided: logit scatter for Gemma (floored samples)
# ===========================================================================

def fig4_guided_vs_unguided_gemma():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    for ax, dataset in zip(axes, DATASETS):
        d_std  = {r["idx"]: r for r in load_all("gemma4-26b", "vllm",        dataset)}
        d_guid = {r["idx"]: r for r in load_all("gemma4-26b", "vllm-guided", dataset)}

        floored_idx   = [i for i in d_std if d_std[i]["floored"]]
        unfloored_idx = [i for i in d_std if not d_std[i]["floored"] and i in d_guid]

        # Max prob comparison
        mp_std_f  = [d_std[i]["max_prob"]  for i in floored_idx if i in d_guid]
        mp_guid_f = [d_guid[i]["max_prob"] for i in floored_idx if i in d_guid]
        mp_std_u  = [d_std[i]["max_prob"]  for i in unfloored_idx[:500]]
        mp_guid_u = [d_guid[i]["max_prob"] for i in unfloored_idx[:500]]

        ax.scatter(mp_std_u, mp_guid_u, s=3, alpha=0.3, color="#aaa", label="No floor", zorder=2)
        ax.scatter(mp_std_f, mp_guid_f, s=10, alpha=0.7, color="#e07b54",
                   label=f"Had floored class (n={len(mp_std_f)})", zorder=3)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
        ax.set_xlabel("max_prob (unguided vLLM)")
        ax.set_ylabel("max_prob (guided vLLM)")
        ax.set_xlim(0.5, 1.01); ax.set_ylim(0.5, 1.01)
        ax.set_title(dataset.replace("_", " ").title())
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        "Figure 4 – Gemma4-26B: max_prob comparison unguided vs guided\n"
        "Each point = one sample. Floored samples (orange) had ≥1 class absent from\n"
        "top-50 logprobs → imputed floor value. Guided decoding recovers the real logprob.\n"
        "Points on the diagonal = no change. Floored samples rarely shift the argmax.",
        fontsize=8,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig4_guided_vs_unguided_gemma.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# FIGURE 5 – Per-class confidence & accuracy on Yahoo (grouped bar)
# ===========================================================================

def fig5_per_class_yahoo():
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))

    x = np.arange(len(YAHOO_LABELS))
    width = 0.25
    aligned_eff_cls, true_labels = confidence_aligned_eff_classes(MODELS, "yahoo")
    aligned_class_means = np.array(
        [
            [
                aligned_eff_cls[i, true_labels == class_id].mean()
                for class_id in range(len(YAHOO_LABELS))
            ]
            for i in range(len(MODELS))
        ]
    )

    for ax_idx, (ax, metric, ylabel) in enumerate(zip(
        axes,
        ["acc", "eff_cls"],
        [
            "Per-class accuracy",
            "Mean effective classes after confidence-scale alignment",
        ],
    )):
        for i, model in enumerate(MODELS):
            if metric == "acc":
                data = load_all(model, "vllm", "yahoo")
                by_class: dict[int, list] = {}
                for d in data:
                    by_class.setdefault(d["true"], []).append(d)
                vals = [np.mean([d["correct"]  for d in by_class[c]]) for c in range(10)]
            else:
                vals = aligned_class_means[i]

            ax.bar(x + (i - 1) * width, vals, width, label=MODEL_DISPLAY[model],
                   color=MODEL_COLORS[model], edgecolor="white", linewidth=0.4, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(YAHOO_LABELS, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
        if ax_idx == 0:
            ax.legend(frameon=False)
            ax.set_ylim(0, 1.05)
            # horizontal line at random baseline
            ax.axhline(0.1, color="grey", lw=0.8, ls=":", label="Random (0.10)")
        else:
            max_cls = len(YAHOO_LABELS)
            ax.axhline(1.0, color="grey", lw=0.8, ls=":", label="Hard label (1.0)")
            ax.axhline(max_cls, color="grey", lw=0.8, ls="--", label=f"Uniform ({max_cls})")
            ax.set_ylim(0.8, max(3.2, float(aligned_class_means.max()) + 0.2))

    fig.suptitle(
        "Figure 5 - Per-class breakdown on Yahoo Answers (vllm mode)\n"
        "Top: raw accuracy. Bottom: effective classes (eff_cls = exp(H)) after label-free\n"
        "confidence-scale alignment; raw entropy differences primarily reflected calibration scale.",
        fontsize=8,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig5_per_class_yahoo.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# FIGURE 6 – Entropy CDF: soft label richness distribution
# ===========================================================================

def fig6_entropy_cdf():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            data = load_all(model, "vllm", dataset)
            entropies = np.sort([d["entropy"] for d in data])
            cdf = np.arange(1, len(entropies) + 1) / len(entropies)
            ax.plot(entropies, cdf, lw=1.4, color=MODEL_COLORS[model],
                    label=MODEL_DISPLAY[model])

        ax.set_xlabel("Shannon entropy (nats)")
        ax.set_ylabel("Cumulative fraction of samples")
        ax.set_title(dataset.replace("_", " ").title())
        ax.legend(frameon=False)
        ax.grid(lw=0.3, alpha=0.4)

    fig.suptitle(
        "Figure 6 – CDF of soft label entropy across samples\n"
        "A curve shifted right = more uncertain teacher = richer soft labels for KD.\n"
        "Gemma is almost always near 0 entropy (hard-label-like). Mistral/Qwen spread further right on Yahoo.",
        fontsize=8,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig6_entropy_cdf.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# FIGURE 7 – Margin distribution: top-1 minus top-2 probability
# ===========================================================================

def fig7_margin_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    bins = np.linspace(0, 1, 31)

    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            data = load_all(model, "vllm", dataset)
            margins = np.array([d["margin"] for d in data])
            ax.hist(margins, bins=bins, histtype="step", lw=1.6,
                    color=MODEL_COLORS[model], label=MODEL_DISPLAY[model], density=True)

        ax.set_xlabel("Margin (top-1 − top-2 softmax prob)")
        ax.set_ylabel("Density")
        ax.set_title(dataset.replace("_", " ").title())
        ax.legend(frameon=False)

    fig.suptitle(
        "Figure 7 – Distribution of decision margin (top-1 minus top-2 probability)\n"
        "Margin ≈ 1 = winner takes all; margin ≈ 0 = two classes are nearly tied.\n"
        "Low-margin samples are the most valuable for KD: they tell the student which\n"
        "classes are confusable. Gemma's margin is near 1 for almost all samples.",
        fontsize=8,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig7_margin_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# FIGURE 8 – Confusion matrices on Yahoo (vllm, all 3 models)
# ===========================================================================

def fig8_confusion_yahoo():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    C = len(YAHOO_LABELS)
    short = [l[:12] for l in YAHOO_LABELS]

    for ax, model in zip(axes, MODELS):
        data = load_all(model, "vllm", "yahoo")
        matrix = np.zeros((C, C))
        for d in data:
            matrix[d["true"], d["pred"]] += 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(C), short, rotation=90, fontsize=6.5)
        ax.set_yticks(range(C), short, fontsize=6.5)
        ax.set_title(MODEL_DISPLAY[model], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("True", fontsize=8)

        for i in range(C):
            for j in range(C):
                if norm[i, j] >= 0.05:
                    col = "white" if norm[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                            fontsize=5, color=col)

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Row-normalised frequency")
    fig.suptitle(
        "Figure 8 – Row-normalised confusion matrices on Yahoo Answers (vllm mode)\n"
        "Diagonal = accuracy per class. Off-diagonal = systematic confusions.\n"
        "Society & Culture ↔ Education, Business ↔ Politics are common error pairs across all models.",
        fontsize=8,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig8_confusion_yahoo.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("\n### TABLES ###")
    table1_accuracy()
    table2_confidence()
    table3_dark_knowledge()
    table4_guided_delta()
    table5_per_class_yahoo()

    print("\n### FIGURES ###")
    fig1_confidence_histograms()
    fig2_dark_knowledge()
    fig3_reliability()
    fig4_guided_vs_unguided_gemma()
    fig5_per_class_yahoo()
    fig6_entropy_cdf()
    fig7_margin_distribution()
    fig8_confusion_yahoo()

    print(f"\nAll outputs written to {OUT_DIR}/")
