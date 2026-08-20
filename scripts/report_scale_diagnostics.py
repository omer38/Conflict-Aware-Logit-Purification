#!/usr/bin/env python
"""Report fitted confidence scales and comparable committee-target entropies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    _fit_confidence_scales,
    _logits_to_probs,
    _match_class_prior_logits,
    _probs_to_kd_target,
    load_multi_teacher_logits,
    uniform_class_prior,
)

DATASETS = ("ag_news", "yahoo", "amazon", "amazon3")
DISPLAY_NAMES = {
    "ag_news": "AG News",
    "yahoo": "Yahoo",
    "amazon": "Amazon",
    "amazon3": "Amazon-3",
}


def mean_entropy(probabilities: np.ndarray) -> float:
    """Return mean per-example Shannon entropy in nats."""
    per_sample = -(
        probabilities * np.log(np.clip(probabilities, 1e-12, None))
    ).sum(axis=-1)
    return float(per_sample.mean())


def diagnostics_for_dataset(dataset_name: str, kd_temperature: float) -> dict:
    dataset = load_dataset_by_name(dataset_name)
    stacked = load_multi_teacher_logits(
        GUIDED_TEACHERS, dataset_name, dataset.num_classes
    )

    raw_target = _logits_to_probs(stacked, temperature=1.0).mean(axis=0)
    prior_matched = _match_class_prior_logits(
        stacked, uniform_class_prior(dataset.num_classes)
    )
    scales = _fit_confidence_scales(prior_matched)
    aligned_logits = prior_matched / scales[:, None, None]
    aligned_target = _logits_to_probs(aligned_logits, temperature=1.0).mean(axis=0)
    kd_target = _probs_to_kd_target(aligned_target, kd_temperature)

    return {
        "dataset": dataset_name,
        "display_name": DISPLAY_NAMES[dataset_name],
        "teachers": list(GUIDED_TEACHERS),
        "fitted_scales": {
            teacher: float(scale)
            for teacher, scale in zip(GUIDED_TEACHERS, scales, strict=True)
        },
        "entropy_units": "nats",
        "h_raw": mean_entropy(raw_target),
        "h_aligned": mean_entropy(aligned_target),
        "h_kd": mean_entropy(kd_target),
        "kd_temperature": kd_temperature,
        "definitions": {
            "h_raw": "entropy of the unaligned arithmetic probability-mean target",
            "h_aligned": (
                "entropy after uniform-prior matching and confidence-scale "
                "alignment, before KD softening"
            ),
            "h_kd": "entropy of the aligned target after KD softening",
        },
    }


def latex_table(rows: list[dict], kd_temperature: float) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Confidence-scale alignment diagnostics. Entropies (nats) "
            r"are measured for the probability-mean target before alignment, "
            r"after prior-and-scale alignment, and after applying the KD "
            rf"temperature \(\tau={kd_temperature:g}\).}}"
        ),
        r"\label{tab:scale-diagnostics}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & \(t_{\mathrm{Qwen}}\) & \(t_{\mathrm{Gemma}}\) & "
            r"\(t_{\mathrm{Mistral}}\) & \(H_{\mathrm{raw}}\) & "
            r"\(H_{\mathrm{aligned}}\) & \(H_{\mathrm{KD}}\) \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        scales = list(row["fitted_scales"].values())
        lines.append(
            f"{row['display_name']} & "
            + " & ".join(f"{value:.3f}" for value in scales)
            + f" & {row['h_raw']:.3f} & {row['h_aligned']:.3f} "
            + f"& {row['h_kd']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/reports")
    )
    args = parser.parse_args()

    rows = [
        diagnostics_for_dataset(dataset_name, args.temperature)
        for dataset_name in DATASETS
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "scale_diagnostics.json"
    tex_path = args.output_dir / "scale_diagnostics.tex"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    tex_path.write_text(latex_table(rows, args.temperature), encoding="utf-8")
    print(latex_table(rows, args.temperature))
    print(f"Saved {json_path} and {tex_path}")


if __name__ == "__main__":
    main()
