"""Router sample-complexity learning curves (Stages A–C).

Fits φ-routers on cached teacher logits — no student GPU, no new LLM queries.

Stage A: 80/20 learnable ceiling vs calibrated prob-mean vs soft oracle.
Stage B/C: isolated n_cal curves on the same 20% holdout, including CAR,
smaller hypothesis class, disagreement-only labels, and majority pseudo-labels.

Usage:
    uv run python scripts/router_learning_curve.py
    uv run python scripts/router_learning_curve.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router.sample_complexity import (
    CAR_TEMPERATURE,
    DATASETS,
    ENTROPY_TOL,
    GATE_PASS_RATE,
    N_CEILING_SEEDS,
    N_CURVE_SEEDS,
    ORACLE_GAP_FRACTION,
    PER_CLASS_GRID,
    VARIANTS,
    V2_TEMPERATURE,
    Variant,
    baseline_mixture,
    calibrated_prob_mean,
    entropy_of,
    feature_matrix,
    load_teacher_stack,
    majority_pseudo_labels,
    metrics_of,
    nll_of,
    nstar_from_curve,
    nstar_beats_mean,
    nstar_temperature,
    oracle_weight_mixture,
    percentile_band,
    prepare_features_and_prior,
    prior_mixture_probs,
    softmax,
    stratified_indices,
    train_and_purify,
    weight_entropy,
)
from src.router.train_router import select_router_or_fallback_nll
from src.student.soft_labels import REPORT_DIR

OUT_DIR = Path("outputs/router/learning_curves")


def _variant_by_name(name: str) -> Variant:
    for v in VARIANTS:
        if v.name == name:
            return v
    raise KeyError(name)


def _eval_split(y: np.ndarray, num_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    eval_idx = stratified_indices(y, max(1, int(0.2 * len(y) / num_classes)), rng, num_classes=num_classes)
    train_pool = np.setdiff1d(np.arange(len(y)), eval_idx, assume_unique=False)
    return train_pool, eval_idx


def _score_row(p: np.ndarray, y: np.ndarray, idx: np.ndarray, prefix: str) -> dict[str, float]:
    m = metrics_of(p[idx], y[idx])
    return {f"{prefix}_{k}": v for k, v in m.items()}


def _entropy_floor_value(p_prior: np.ndarray, idx: np.ndarray) -> float:
    return entropy_of(p_prior[idx])


def run_stage_a(
    dataset: str,
    logits_knc: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    n_seeds: int,
    phi_raw: np.ndarray,
) -> dict:
    k, n, _ = logits_knc.shape
    oracle_p = oracle_weight_mixture(logits_knc, y, oracle_tau=3.0)
    records: list[dict] = []

    for seed in range(n_seeds):
        train_pool, eval_idx = _eval_split(y, num_classes, seed=10_000 + seed)
        p_calib = calibrated_prob_mean(logits_knc, y, train_pool)
        p_mean_raw = softmax(logits_knc, axis=-1).mean(axis=0)
        base = {
            "dataset": dataset,
            "seed": seed,
            "n_train": int(len(train_pool)),
            "n_eval": int(len(eval_idx)),
            **_score_row(p_calib, y, eval_idx, "calibmean"),
            **_score_row(p_mean_raw, y, eval_idx, "probmean"),
            **_score_row(oracle_p, y, eval_idx, "oracle_tau3"),
        }

        for variant in VARIANTS:
            if not variant.stage_a:
                continue
            row = {"variant": variant.name, **base}
            if variant.kind == "baseline":
                p = baseline_mixture(variant, logits_knc, y, train_pool)
                row.update(_score_row(p, y, eval_idx, "router"))
                row["weight_H"] = float("nan")
                row["survives"] = (
                    row["router_nll"] < row["calibmean_nll"]
                    and row["router_H"] >= row["calibmean_H"] - ENTROPY_TOL
                )
                records.append(row)
                continue

            phi, logits_nkc, log_prior, disagree, _ = prepare_features_and_prior(
                variant, logits_knc, y, train_pool, phi_precomputed=phi_raw
            )
            temperature = CAR_TEMPERATURE if variant.calibrate else V2_TEMPERATURE
            p_prior = prior_mixture_probs(
                logits_nkc, log_prior, temperature, variant.mixture_space
            )
            floor = _entropy_floor_value(p_prior, train_pool)
            pseudo_idx = pseudo_y = None
            if variant.pseudo:
                probs = softmax(np.transpose(logits_nkc, (1, 0, 2)), axis=-1)
                pidx, py = majority_pseudo_labels(probs)
                keep = np.isin(pidx, train_pool)
                pseudo_idx, pseudo_y = pidx[keep], py[keep]
            p, w = train_and_purify(
                variant,
                phi,
                logits_nkc,
                y,
                train_pool,
                log_prior,
                disagree,
                temperature,
                weight_decay=0.0,
                seed=seed,
                entropy_floor=floor,
                pseudo_idx=pseudo_idx,
                pseudo_y=pseudo_y,
            )
            row.update(_score_row(p, y, eval_idx, "router"))
            row["weight_H"] = weight_entropy(w[eval_idx])
            row["nll_gap"] = row["router_nll"] - row["calibmean_nll"]
            oracle_gap = row["calibmean_nll"] - row["oracle_tau3_nll"]
            row["oracle_gap"] = oracle_gap
            row["oracle_gap_closed"] = (
                (row["calibmean_nll"] - row["router_nll"]) / oracle_gap if oracle_gap > 1e-4 else 0.0
            )
            row["survives"] = (
                row["router_nll"] < row["calibmean_nll"]
                and row["router_H"] >= row["calibmean_H"] - ENTROPY_TOL
            )
            records.append(row)

        # In-sample (all labels) ceiling for the best CAR linear router.
        variant = _variant_by_name("car_linear_all")
        all_idx = np.arange(n)
        phi, logits_nkc, log_prior, disagree, _ = prepare_features_and_prior(
            variant, logits_knc, y, all_idx, phi_precomputed=phi_raw
        )
        p_calib_all = calibrated_prob_mean(logits_knc, y, all_idx)
        p_prior = prior_mixture_probs(logits_nkc, log_prior, CAR_TEMPERATURE, variant.mixture_space)
        p, w = train_and_purify(
            variant, phi, logits_nkc, y, all_idx, log_prior, disagree,
            CAR_TEMPERATURE, weight_decay=0.0, seed=seed,
            entropy_floor=_entropy_floor_value(p_prior, all_idx),
        )
        ins = {
            "dataset": dataset,
            "seed": seed,
            "variant": "car_linear_all_insample",
            "n_train": n,
            "n_eval": n,
            **_score_row(p, y, all_idx, "router"),
            **_score_row(p_calib_all, y, all_idx, "calibmean"),
            **_score_row(oracle_p, y, all_idx, "oracle_tau3"),
            "weight_H": weight_entropy(w),
        }
        ins["nll_gap"] = ins["router_nll"] - ins["calibmean_nll"]
        og = ins["calibmean_nll"] - ins["oracle_tau3_nll"]
        ins["oracle_gap_closed"] = (ins["calibmean_nll"] - ins["router_nll"]) / og if og > 1e-4 else 0.0
        ins["survives"] = (
            ins["router_nll"] < ins["calibmean_nll"]
            and ins["router_H"] >= ins["calibmean_H"] - ENTROPY_TOL
        )
        records.append(ins)

    summary = _summarise_stage_a(dataset, records)
    return {"dataset": dataset, "records": records, "summary": summary}


def _summarise_stage_a(dataset: str, records: list[dict]) -> dict:
    by_var: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_var[r["variant"]].append(r)
    out = {}
    for name, rows in by_var.items():
        nll_gap = [r["router_nll"] - r["calibmean_nll"] for r in rows]
        survives = [bool(r["survives"]) for r in rows]
        out[name] = {
            "n_seeds": len(rows),
            "router_nll": percentile_band([r["router_nll"] for r in rows]),
            "calibmean_nll": percentile_band([r["calibmean_nll"] for r in rows]),
            "nll_gap": percentile_band(nll_gap),
            "router_H": percentile_band([r["router_H"] for r in rows]),
            "calibmean_H": percentile_band([r["calibmean_H"] for r in rows]),
            "router_acc": percentile_band([r["router_acc"] for r in rows]),
            "oracle_gap_closed": percentile_band(
                [r.get("oracle_gap_closed", 0.0) for r in rows]
            ),
            "survive_rate": float(np.mean(survives)),
            "survives": bool(np.median(nll_gap) < 0 and np.mean(survives) >= 0.5),
        }
    return out


def run_curves(
    dataset: str,
    logits_knc: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    variants: list[Variant],
    n_seeds: int,
    per_class_grid: tuple[int, ...],
    phi_raw: np.ndarray,
) -> dict:
    curve_rows: list[dict] = []
    k, n, _ = logits_knc.shape
    oracle_p = oracle_weight_mixture(logits_knc, y, oracle_tau=3.0)

    for seed in range(n_seeds):
        train_pool, eval_idx = _eval_split(y, num_classes, seed=10_000 + seed)
        counts = np.bincount(y[train_pool], minlength=num_classes)
        max_per_class = int(counts.min())
        grid = [g for g in per_class_grid if g <= max_per_class]

        p_calib_full = calibrated_prob_mean(logits_knc, y, train_pool)

        for n_per_class in grid:
            rng = np.random.default_rng(20_000 + seed * 1000 + n_per_class)
            random_cal = stratified_indices(y, n_per_class, rng, pool=train_pool, num_classes=num_classes)
            p_calib = calibrated_prob_mean(logits_knc, y, random_cal)
            calib_eval = metrics_of(p_calib[eval_idx], y[eval_idx])
            oracle_eval = metrics_of(oracle_p[eval_idx], y[eval_idx])
            oracle_gap = calib_eval["nll"] - oracle_eval["nll"]

            for variant in variants:
                if variant.pseudo and n_per_class not in (10, 50, 200):
                    continue
                if variant.kind == "baseline":
                    p = baseline_mixture(variant, logits_knc, y, random_cal)
                    m_eval = metrics_of(p[eval_idx], y[eval_idx])
                    m_cal = metrics_of(p[random_cal], y[random_cal])
                    gate = select_router_or_fallback_nll(m_cal["nll"], nll_of(p_calib[random_cal], y[random_cal]))
                    p_gated = p if gate["use_router"] else p_calib
                    m_gated = metrics_of(p_gated[eval_idx], y[eval_idx])
                    curve_rows.append(
                        _curve_record(
                            dataset, variant.name, seed, len(random_cal), n_per_class,
                            calib_eval, oracle_eval, oracle_gap, m_eval, m_gated, gate,
                            entropy_of(p[eval_idx]),
                        )
                    )
                    continue

                cal_idx = random_cal
                if variant.sampling == "disagreement":
                    # Calibrate τ / features on a random labelled slice, then
                    # restrict router supervision to disagreement rows.
                    phi_probe, _, _, disagree_probe, _ = prepare_features_and_prior(
                        variant, logits_knc, y, random_cal, phi_precomputed=phi_raw
                    )
                    cal_idx = stratified_indices(
                        y, n_per_class, rng,
                        pool=train_pool[disagree_probe[train_pool]],
                        num_classes=num_classes,
                    )
                    if len(cal_idx) < num_classes:
                        cal_idx = random_cal

                phi, logits_nkc, log_prior, disagree, _ = prepare_features_and_prior(
                    variant, logits_knc, y, cal_idx, phi_precomputed=phi_raw
                )
                temperature = CAR_TEMPERATURE if variant.calibrate else V2_TEMPERATURE
                p_prior = prior_mixture_probs(
                    logits_nkc, log_prior, temperature, variant.mixture_space
                )
                pseudo_idx = pseudo_y = None
                if variant.pseudo:
                    probs = softmax(np.transpose(logits_nkc, (1, 0, 2)), axis=-1)
                    pidx, py = majority_pseudo_labels(probs)
                    keep = np.isin(pidx, train_pool)
                    pseudo_idx, pseudo_y = pidx[keep], py[keep]

                p, w = train_and_purify(
                    variant, phi, logits_nkc, y, cal_idx, log_prior, disagree,
                    temperature, weight_decay=0.01, seed=seed,
                    epochs=100,
                    entropy_floor=_entropy_floor_value(p_prior, cal_idx),
                    pseudo_idx=pseudo_idx, pseudo_y=pseudo_y,
                )
                # Same-n calibrated mean in the *working* (possibly τ-scaled) space.
                p_calib_v = calibrated_prob_mean(logits_knc, y, cal_idx)
                m_eval = metrics_of(p[eval_idx], y[eval_idx])
                m_cal = metrics_of(p[cal_idx], y[cal_idx])
                gate = select_router_or_fallback_nll(
                    m_cal["nll"], nll_of(p_calib_v[cal_idx], y[cal_idx])
                )
                p_gated = p if gate["use_router"] else p_calib_v
                m_gated = metrics_of(p_gated[eval_idx], y[eval_idx])
                curve_rows.append(
                    _curve_record(
                        dataset, variant.name, seed, int(len(cal_idx)), n_per_class,
                        calib_eval, oracle_eval, oracle_gap, m_eval, m_gated, gate,
                        entropy_of(p[eval_idx]),
                        extra={
                            "weight_H": weight_entropy(w[eval_idx]),
                            "calibmean_full_nll": nll_of(p_calib_full[eval_idx], y[eval_idx]),
                        },
                    )
                )

            # Temperature-only baseline (n*_τ): same-n calibrated mean already in calib_eval.
            curve_rows.append(
                _curve_record(
                    dataset, "calibmean", seed, int(len(random_cal)), n_per_class,
                    calib_eval, oracle_eval, oracle_gap, calib_eval, calib_eval,
                    {"use_router": False, "router_nll": calib_eval["nll"], "baseline_nll": calib_eval["nll"], "fallback_to": "baseline"},
                    calib_eval["H"],
                    extra={"calibmean_full_nll": nll_of(p_calib_full[eval_idx], y[eval_idx])},
                )
            )

    aggregated = _aggregate_curves(dataset, curve_rows)
    return {"dataset": dataset, "rows": curve_rows, "aggregated": aggregated}


def _curve_record(
    dataset: str,
    variant: str,
    seed: int,
    n_cal: int,
    n_per_class: int,
    calib_eval: dict,
    oracle_eval: dict,
    oracle_gap: float,
    m_eval: dict,
    m_gated: dict,
    gate: dict,
    router_H: float,
    extra: dict | None = None,
) -> dict:
    closed = (calib_eval["nll"] - m_eval["nll"]) / oracle_gap if oracle_gap > 1e-4 else 0.0
    row = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "n_cal": n_cal,
        "n_per_class": n_per_class,
        "router_acc": m_eval["acc"],
        "router_nll": m_eval["nll"],
        "router_H": router_H,
        "gated_nll": m_gated["nll"],
        "gated_acc": m_gated["acc"],
        "calibmean_nll": calib_eval["nll"],
        "calibmean_acc": calib_eval["acc"],
        "calibmean_H": calib_eval["H"],
        "oracle_nll": oracle_eval["nll"],
        "nll_gap": m_eval["nll"] - calib_eval["nll"],
        "oracle_gap_closed": closed,
        "entropy_ok": router_H >= calib_eval["H"] - ENTROPY_TOL,
        "gate_pass": bool(gate["use_router"]),
    }
    if extra:
        row.update(extra)
    return row


def _aggregate_curves(dataset: str, rows: list[dict]) -> dict:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["variant"], int(r["n_per_class"]))].append(r)
    out: dict[str, list[dict]] = defaultdict(list)
    for (variant, n_per_class), items in grouped.items():
        nll_gaps = [r["nll_gap"] for r in items]
        closed = [r["oracle_gap_closed"] for r in items]
        band = percentile_band(nll_gaps)
        closed_band = percentile_band(closed)
        out[variant].append(
            {
                "n_cal": int(np.median([r["n_cal"] for r in items])),
                "n_per_class": n_per_class,
                "n_per_class_median": float(n_per_class),
                "n_seeds": len(items),
                "nll_gap_median": band["median"],
                "nll_gap_p10": band["p10"],
                "nll_gap_p90": band["p90"],
                "router_nll_median": float(np.median([r["router_nll"] for r in items])),
                "calibmean_nll_median": float(np.median([r["calibmean_nll"] for r in items])),
                "gated_nll_median": float(np.median([r["gated_nll"] for r in items])),
                "router_H_median": float(np.median([r["router_H"] for r in items])),
                "oracle_gap_closed_median": closed_band["median"],
                "oracle_gap_closed_p10": closed_band["p10"],
                "entropy_ok_rate": float(np.mean([r["entropy_ok"] for r in items])),
                "gate_pass_rate": float(np.mean([r["gate_pass"] for r in items])),
                "router_acc_median": float(np.median([r["router_acc"] for r in items])),
            }
        )
    for variant in out:
        out[variant].sort(key=lambda r: r["n_per_class"])
    return dict(out)


def fit_nstars(curve_agg: dict, stage_a_summary: dict) -> dict:
    nstars = {}
    calib_rows = curve_agg.get("calibmean", [])
    full_nll = None
    if calib_rows:
        full_nll = min(r["calibmean_nll_median"] for r in calib_rows)
        # Prefer the largest-n point as the full-pool calibrated mean.
        full_nll = calib_rows[-1]["calibmean_nll_median"]
    nstars["nstar_tau"] = nstar_temperature(calib_rows, full_nll) if calib_rows and full_nll is not None else None
    for variant, rows in curve_agg.items():
        if variant == "calibmean":
            continue
        survives_a = stage_a_summary.get(variant, {}).get("survives", False)
        nstars[variant] = {
            "stage_a_survives": survives_a,
            "nstar_proxy": nstar_from_curve(rows) if survives_a else None,
            "nstar_proxy_ignoring_ceiling": nstar_from_curve(rows),
            "nstar_nll": nstar_beats_mean(rows) if survives_a else None,
        }
    return nstars


def plot_curves(dataset: str, aggregated: dict, nstars: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {
        "car_linear_all": "#1f77b4",
        "car_linear_agg": "#ff7f0e",
        "car_linear_sparse": "#2ca02c",
        "car_linear_disagree": "#d62728",
        "car_linear_pseudo": "#9467bd",
        "drop_outlier_lambda": "#8c564b",
        "class_conditional": "#e377c2",
        "calibmean": "#7f7f7f",
        "v2_linear_all": "#bcbd22",
    }
    for variant, rows in aggregated.items():
        xs = [r["n_cal"] for r in rows]
        med = [r["nll_gap_median"] for r in rows]
        lo = [r["nll_gap_p10"] for r in rows]
        hi = [r["nll_gap_p90"] for r in rows]
        color = colors.get(variant, None)
        ax.plot(xs, med, marker="o", label=variant, color=color)
        ax.fill_between(xs, lo, hi, alpha=0.12, color=color)
        nstar = nstars.get(variant, {}).get("nstar_proxy") if isinstance(nstars.get(variant), dict) else None
        if nstar:
            ax.axvline(nstar, linestyle="--", alpha=0.5, color=color)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("n_cal (log)")
    ax.set_ylabel("Held-out NLL gap vs same-n calibrated mean (<0 = router better)")
    ax.set_title(f"{dataset}: router sample-complexity curves")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_summary_md(payload: dict, path: Path) -> None:
    lines = [
        "# Router sample complexity",
        "",
        "Proxy learning curves on cached teacher logits. Evaluation is always the",
        "same stratified 20% holdout per seed; n_cal is drawn from the remaining 80%.",
        "A variant *survives* Stage A if median OOS NLL beats calibrated prob-mean",
        "without collapsing entropy (H >= H_calibmean - 0.05).",
        "",
        f"Good-router n* requires all of: NLL gap p10 < 0, entropy-ok rate "
        f">= {GATE_PASS_RATE:.0%}, oracle-gap-closed p10 >= {ORACLE_GAP_FRACTION:.0%},",
        f"gate-pass rate >= {GATE_PASS_RATE:.0%}.",
        "",
    ]
    for ds, blob in payload["datasets"].items():
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("### Stage A — learnable ceiling (80/20)")
        lines.append("")
        lines.append("| variant | NLL gap median | survive rate | oracle-gap closed | survives |")
        lines.append("|---|---:|---:|---:|---|")
        for name, s in blob["stage_a"]["summary"].items():
            gap = s["nll_gap"]["median"]
            lines.append(
                f"| {name} | {gap:+.4f} | {s['survive_rate']:.0%} | "
                f"{s['oracle_gap_closed']['median']:.2f} | {s['survives']} |"
            )
        lines.append("")
        nstars = blob["nstars"]
        lines.append("### n*")
        lines.append("")
        lines.append(f"- n*_τ (calibrated mean saturates): `{nstars.get('nstar_tau')}`")
        for name, ns in nstars.items():
            if name == "nstar_tau":
                continue
            lines.append(
                f"- `{name}`: n*_proxy={ns.get('nstar_proxy')} "
                f"n*_nll={ns.get('nstar_nll')} "
                f"(Stage A survives={ns.get('stage_a_survives')}; "
                f"ignoring ceiling={ns.get('nstar_proxy_ignoring_ceiling')})"
            )
        lines.append("")
        lines.append("![curve]({}_nll_gap.png)".format(ds))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Router sample-complexity learning curves")
    p.add_argument("--quick", action="store_true", help="2 seeds, tiny n-grid, AG News only")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--curve-seeds", type=int, default=None)
    p.add_argument("--ceiling-seeds", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or list(DATASETS)
    if args.quick:
        datasets = ["ag_news"]
        n_curve = 2
        n_ceil = 2
        grid = (3, 10)
    else:
        n_curve = args.curve_seeds or N_CURVE_SEEDS
        n_ceil = args.ceiling_seeds or N_CEILING_SEEDS
        grid = PER_CLASS_GRID
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"datasets": {}, "config": {"curve_seeds": n_curve, "ceiling_seeds": n_ceil, "grid": list(grid)}}
    for ds in datasets:
        C = DATASETS[ds]
        print(f"\n===== {ds} =====")
        logits, y = load_teacher_stack(ds, C, REPORT_DIR)
        print(f"Loaded logits {logits.shape}, labels {y.shape}")
        phi_raw = feature_matrix(softmax(logits, axis=-1), "all")
        print(f"Cached φ(x) {phi_raw.shape}")
        stage_a = run_stage_a(ds, logits, y, C, n_ceil, phi_raw)
        survivors = [name for name, s in stage_a["summary"].items() if s["survives"] and name in {v.name for v in VARIANTS}]
        print("Stage A survivors:", survivors or "(none)")
        curve_variants = [
            v for v in VARIANTS
            if v.stage_curve and (v.kind == "baseline" or v.calibrate or v.name in survivors)
        ]
        # Always include CAR efficiency family even if Stage A failed, so Δn* is defined.
        names = {v.name for v in curve_variants}
        for v in VARIANTS:
            if v.stage_curve and v.name not in names:
                curve_variants.append(v)
        print("Curve variants:", [v.name for v in curve_variants])
        curves = run_curves(ds, logits, y, C, curve_variants, n_curve, grid, phi_raw)
        nstars = fit_nstars(curves["aggregated"], stage_a["summary"])
        plot_curves(ds, curves["aggregated"], nstars, out_dir / f"{ds}_nll_gap.png")
        ds_payload = {
            "stage_a": {"summary": stage_a["summary"], "records": stage_a["records"]},
            "curves": {"aggregated": curves["aggregated"], "rows": curves["rows"]},
            "nstars": nstars,
        }
        payload["datasets"][ds] = ds_payload
        (out_dir / f"{ds}.json").write_text(json.dumps(ds_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_dir / ds}.json")

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary_md(payload, out_dir / "summary.md")
    print(f"\nDone. Summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
