#!/usr/bin/env python
"""Compare supervised, single-teacher KD, and multi-teacher KD student results.

Usage:
    uv run python scripts/compare_kd_results.py
    uv run python scripts/compare_kd_results.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router.oracle import MULTI_ORACLE_TEACHER
from src.router.train_router import MULTI_ROUTER_LINEAR_TEACHER, MULTI_ROUTER_MLP_TEACHER
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    MULTI_ENTROPY_TEACHER,
    MULTI_MEAN_TEACHER,
)

SINGLE_TEACHERS = list(GUIDED_TEACHERS)
DATASETS = ["ag_news", "yahoo", "amazon"]
L_VALUES = [10, 50, 500]


def load_results(output_dir: Path, seed: int) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("seed") != seed:
            continue
        rows.append(data)
    return rows


def acc_pct(row: dict | None) -> str:
    if row is None:
        return "—"
    return f"{row['metrics']['accuracy'] * 100:.1f}%"


def delta_str(row: dict | None, ref: dict | None) -> str:
    if row is None or ref is None:
        return "—"
    return f"{(row['metrics']['accuracy'] - ref['metrics']['accuracy']) * 100:+.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KD experiment results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/student")
    args = parser.parse_args()

    results = load_results(Path(args.output_dir), args.seed)
    by_key = {
        (r["dataset"], r["mode"], r.get("teacher"), r["labels_per_class"]): r
        for r in results
    }

    for ds in DATASETS:
        print(f"\n=== {ds.upper()} (seed={args.seed}) ===")
        print(
            f"{'L':>4} | {'Supervised':>10} | {'Best single':>11} | "
            f"{'Mean':>8} | {'Entropy':>8} | {'Ent vs best':>11} | "
            f"{'Oracle':>8} | {'Router-L':>8} | {'Router-M':>8}"
        )
        print("-" * 104)
        for l in L_VALUES:
            sup = by_key.get((ds, "supervised", "none", l))
            singles = [by_key.get((ds, "kd", t, l)) for t in SINGLE_TEACHERS]
            singles = [s for s in singles if s is not None]
            best = max(singles, key=lambda r: r["metrics"]["accuracy"]) if singles else None
            mean_row = by_key.get((ds, "kd", MULTI_MEAN_TEACHER, l))
            ent_row = by_key.get((ds, "kd", MULTI_ENTROPY_TEACHER, l))
            # Oracle/router experiments are scoped to Yahoo-L10 and AG-News-L10
            # only (see docs/plans/phase_b_c_d_plan.md); other rows show "—".
            oracle_row = by_key.get((ds, "kd", MULTI_ORACLE_TEACHER, l))
            router_linear_row = by_key.get((ds, "kd", MULTI_ROUTER_LINEAR_TEACHER, l))
            router_mlp_row = by_key.get((ds, "kd", MULTI_ROUTER_MLP_TEACHER, l))

            best_s = acc_pct(best)
            if best is not None:
                short = best["teacher"].split("-")[0].replace("qwen3.5", "qwen")
                best_s = f"{best_s} ({short})"

            print(
                f"{l:4d} | {acc_pct(sup):>10} | {best_s:>11} | "
                f"{acc_pct(mean_row):>8} | {acc_pct(ent_row):>8} | "
                f"{delta_str(ent_row, best):>11} | "
                f"{acc_pct(oracle_row):>8} | {acc_pct(router_linear_row):>8} | "
                f"{acc_pct(router_mlp_row):>8}"
            )


if __name__ == "__main__":
    main()
