#!/usr/bin/env python
"""Build a CSV/JSON report from cached teacher logits (no API calls).

Usage:
    uv run python scripts/export_extraction_report.py --teacher gpt-4.1-mini --dataset amazon
    uv run python scripts/export_extraction_report.py --teacher gpt-4.1-mini --dataset yahoo --report outputs/reports/custom.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.teachers.llm import TEACHER_PRESETS
from src.teachers.extraction import (
    default_report_path,
    load_cached_logits,
    write_extraction_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a CSV/JSON report from cached teacher logits."
    )
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
        "--report",
        nargs="?",
        const="auto",
        default="auto",
        metavar="PATH",
        help=(
            "CSV output path (default: outputs/reports/{teacher}_{dataset}.csv)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only include the first N samples in the report.",
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
    print(f"Report CSV:     {csv_path}")
    print(f"Report summary: {summary_path}")


if __name__ == "__main__":
    main()
