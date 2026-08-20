#!/usr/bin/env python
"""Extract teacher logits for a dataset and cache them to disk.

Usage:
    uv run python scripts/extract_logits.py --teacher gpt-4.1-mini --dataset ag_news
    uv run python scripts/extract_logits.py --teacher gemma4-26b-vllm --dataset ag_news --limit 10
    uv run python scripts/extract_logits.py --teacher mistral-small-24b-vllm --dataset ag_news --limit 10
    uv run python scripts/extract_logits.py --teacher deepseek-r1 --dataset amazon --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.teachers.llm import TEACHER_PRESETS
from src.teachers.extraction import (
    default_report_path,
    extract_logits_for_dataset,
    load_cached_logits,
    write_extraction_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract teacher logits for a dataset.")
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
        help="Dataset to extract logits for.",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/logits",
        help="Directory to store cached logit files (default: cache/logits).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only extract the first N samples (for testing).",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Write a CSV report plus JSON summary after extraction. "
            "Default path: outputs/reports/{teacher}_{dataset}.csv"
        ),
    )
    args = parser.parse_args()

    config = TEACHER_PRESETS[args.teacher]
    print(f"Teacher: {config.name} (model={config.model}, base_url={config.base_url})")

    dataset = load_dataset_by_name(args.dataset)
    print(f"Dataset: {dataset.name} ({len(dataset.train)} train samples, {dataset.num_classes} classes)")

    logits = extract_logits_for_dataset(
        config=config,
        dataset=dataset,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )

    # The cache always exists after extraction; re-load it for the floored mask.
    _, extracted_mask, floored = load_cached_logits(config.name, dataset.name, args.cache_dir)

    # Print summary statistics.
    n_extracted = logits.shape[0] if args.limit is None else args.limit
    logits_subset = logits[:n_extracted]

    probs = np.exp(logits_subset - logits_subset.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    floor_frac = floored[:n_extracted].sum() / floored[:n_extracted].size
    predictions = logits_subset.argmax(axis=1)
    labels = np.array(dataset.train[:n_extracted]["label"])
    accuracy = (predictions == labels).mean()

    print(f"\nSummary:")
    print(f"  Logit shape:    {logits_subset.shape}")
    print(f"  Floor values:   {floor_frac:.1%} of entries")
    print(f"  Top-1 accuracy: {accuracy:.1%} (vs ground-truth labels)")
    print(f"  Prob sums:      min={probs.sum(axis=1).min():.4f}, max={probs.sum(axis=1).max():.4f}")

    if args.report is not None:
        report_path = (
            default_report_path(config.name, dataset.name)
            if args.report == "auto"
            else Path(args.report)
        )
        csv_path, summary_path = write_extraction_report(
            logits,
            dataset,
            config.name,
            report_path,
            extracted_mask=extracted_mask,
            floored=floored,
            limit=args.limit,
        )
        print(f"\nReport CSV:     {csv_path}")
        print(f"Report summary: {summary_path}")


if __name__ == "__main__":
    main()
