"""Stage E: student confirmation at n*_proxy, saturation, and the full-pool ceiling.

Trains ModernBERT on isolated extra router labels (student CE keeps the full L=10
set).  Skips variants that failed Stage A.  If n*_proxy is undefined, trains the
full-pool ceiling against calibrated mean and soft oracle τ=3.

Usage:
    uv run python scripts/train_router_student_confirm.py
    uv run python scripts/train_router_student_confirm.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StudentConfig
from src.data.loading import load_dataset_by_name, split_identity
from src.data.subsampling import PROMPT_DEMOS_PER_CLASS, subsample_low_label
from src.router.oracle import oracle_teacher_soft_labels
from src.router.sample_complexity import (
    CAR_TEMPERATURE,
    DATASETS,
    VARIANTS,
    Variant,
    calibrated_prob_mean,
    feature_matrix,
    load_teacher_stack,
    majority_pseudo_labels,
    prepare_features_and_prior,
    prior_mixture_probs,
    softmax,
    stratified_indices,
    train_and_purify,
)
from src.student.model import build_student, select_device
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    _probs_to_kd_target,
    average_teacher_soft_labels,
)
from src.student.train import build_examples, evaluate, fit, set_seed

CURVE_DIR = Path("outputs/router/learning_curves")
OUT_DIR = Path("outputs/student/router_sample_complexity_v1")
MEAN_SRC = Path("outputs/student/prompt_inclusive_core_3seed_v1")
SEEDS = (42, 123, 456)
L_CE = 10
KD_TAU = 2.0
LABEL_ACCOUNTING_TAG = "prompt-inclusive-v1"


def mcnemar_exact(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n_discordant = only_a + only_b
    p_value = 1.0 if n_discordant == 0 else float(binomtest(only_a, n_discordant, p=0.5).pvalue)
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def _variant(name: str) -> Variant:
    for v in VARIANTS:
        if v.name == name:
            return v
    raise KeyError(name)


def _best_surviving_router(nstars: dict) -> str | None:
    preferred = (
        "car_linear_all",
        "car_linear_sparse",
        "car_linear_agg",
        "car_linear_disagree",
        "car_linear_pseudo",
        "drop_outlier_lambda",
        "class_conditional",
        "v2_linear_all",
    )
    for name in preferred:
        info = nstars.get(name)
        if isinstance(info, dict) and info.get("stage_a_survives"):
            return name
    return None


def _kd_targets(p_tau1: np.ndarray) -> np.ndarray:
    return _probs_to_kd_target(p_tau1, KD_TAU)


def build_router_targets(
    variant: Variant,
    logits_knc: np.ndarray,
    y: np.ndarray,
    phi_raw: np.ndarray,
    cal_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    if variant.kind == "baseline":
        from src.router.sample_complexity import baseline_mixture

        p = baseline_mixture(variant, logits_knc, y, cal_idx)
        return _kd_targets(p)

    phi, logits_nkc, log_prior, disagree, _ = prepare_features_and_prior(
        variant, logits_knc, y, cal_idx, phi_precomputed=phi_raw
    )
    p_prior = prior_mixture_probs(logits_nkc, log_prior, CAR_TEMPERATURE, variant.mixture_space)
    pseudo_idx = pseudo_y = None
    if variant.pseudo:
        probs = softmax(np.transpose(logits_nkc, (1, 0, 2)), axis=-1)
        pidx, py = majority_pseudo_labels(probs)
        pseudo_idx, pseudo_y = pidx, py
    p, _ = train_and_purify(
        variant, phi, logits_nkc, y, cal_idx, log_prior, disagree,
        CAR_TEMPERATURE if variant.calibrate else 2.0,
        weight_decay=0.01, seed=seed,
        entropy_floor=float(-(p_prior * np.log(np.clip(p_prior, 1e-12, None))).sum(1).mean()),
        pseudo_idx=pseudo_idx, pseudo_y=pseudo_y,
    )
    return _kd_targets(p)


def extra_cal_from_unlabeled(
    y: np.ndarray,
    unlabeled_idx: np.ndarray,
    n_per_class: int,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    return stratified_indices(y, n_per_class, rng, pool=unlabeled_idx, num_classes=num_classes)


def train_one(
    dataset_name: str,
    teacher_tag: str,
    soft_probs: np.ndarray,
    seed: int,
    output_dir: Path,
    extra_meta: dict,
    dry_run: bool,
) -> Path:
    dataset = load_dataset_by_name(dataset_name)
    splits = subsample_low_label(dataset, L_CE, seed, 0.3)
    config = StudentConfig(temperature=KD_TAU, alpha=0.5, epochs=3, batch_size=16, max_length=256)
    out = output_dir / (
        f"{dataset_name}_kd_{teacher_tag}_L{L_CE}_s{seed}_{LABEL_ACCOUNTING_TAG}.json"
    )
    if out.exists() and not dry_run:
        print(f"Skip existing {out.name}")
        return out
    if dry_run:
        print(f"DRY RUN would train {out.name}  soft={soft_probs.shape}")
        return out

    set_seed(seed)
    device = select_device()
    train_ds = build_examples(dataset, splits, "kd", soft_probs, reserve_calibration=False)
    model, tokenizer = build_student(config.model_name, dataset.num_classes)
    fit(model, tokenizer, train_ds, config, device)
    evaluation = evaluate(model, tokenizer, dataset, config, device, return_predictions=True)
    test_labels = evaluation.pop("test_labels")
    test_predictions = evaluation.pop("test_predictions")
    result = {
        "dataset": dataset_name,
        "mode": "kd",
        "teacher": teacher_tag,
        "labels_per_class": L_CE,
        "seed": seed,
        "label_accounting": {
            "policy": LABEL_ACCOUNTING_TAG,
            "declared_labels_per_class": L_CE,
            "prompt_demo_labels_per_class": PROMPT_DEMOS_PER_CLASS,
            "student_ce_labels_per_class": L_CE,
            "router_calibration_labels_per_class": extra_meta.get("n_cal", 0),
            "calibration_reserved": False,
            "router_labels_isolated": True,
        },
        "config": vars(config),
        "metrics": evaluation,
        "dataset_identity": {
            "artifact_version": dataset.metadata.get("artifact_version", "legacy"),
            "train": dataset.metadata.get("train_identity") or split_identity(
                dataset.train, dataset.text_column, dataset.label_column
            ),
            "test": dataset.metadata.get("test_identity") or split_identity(
                dataset.test, dataset.text_column, dataset.label_column
            ),
        },
        "paired_test_data": {
            "sample_indices": list(range(len(test_labels))),
            "labels": test_labels,
            "predictions": test_predictions,
        },
        "router_sample_complexity": extra_meta,
        "device": str(device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}  acc={evaluation['accuracy']:.4f}")
    return out


def planned_jobs(summary: dict) -> list[dict]:
    jobs = []
    for ds, blob in summary["datasets"].items():
        nstars = blob["nstars"]
        best = _best_surviving_router(nstars)
        nstar = None
        if best:
            info = nstars.get(best, {})
            nstar = info.get("nstar_proxy") or info.get("nstar_nll")
        points = [("fullpool", None)]
        if nstar:
            n_per = max(1, int(round(nstar / DATASETS[ds])))
            points.append(("nstar", n_per))
        jobs.append({"dataset": ds, "best_variant": best, "points": points, "nstar": nstar})
    return jobs


def analyse_dir(output_dir: Path) -> dict:
    files = sorted(output_dir.glob("*.json"))
    grouped: dict[tuple[str, str], dict[int, dict]] = {}
    for path in files:
        if path.name in {"mcnemar.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in data:
            continue
        key = (data["dataset"], data["teacher"])
        grouped.setdefault(key, {})[int(data["seed"])] = data

    comparisons = []
    datasets = sorted({k[0] for k in grouped})
    for ds in datasets:
        mean_runs = grouped.get((ds, "multi-mean-guided"), {})
        for (dataset, teacher), runs in grouped.items():
            if dataset != ds or teacher == "multi-mean-guided":
                continue
            for seed, run in runs.items():
                base = mean_runs.get(seed)
                if base is None:
                    continue
                labels = np.asarray(run["paired_test_data"]["labels"])
                pred_a = np.asarray(run["paired_test_data"]["predictions"])
                pred_b = np.asarray(base["paired_test_data"]["predictions"])
                stats = mcnemar_exact(labels, pred_a, pred_b)
                comparisons.append(
                    {
                        "dataset": ds,
                        "teacher": teacher,
                        "seed": seed,
                        "acc": run["metrics"]["accuracy"],
                        "mean_acc": base["metrics"]["accuracy"],
                        "delta": run["metrics"]["accuracy"] - base["metrics"]["accuracy"],
                        **stats,
                    }
                )
    return {"comparisons": comparisons}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=CURVE_DIR / "summary.json")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--datasets", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"Missing {args.summary}; run scripts/router_learning_curve.py first")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    jobs = planned_jobs(summary)
    if args.datasets:
        jobs = [j for j in jobs if j["dataset"] in args.datasets]
    print("Stage E plan:")
    print(json.dumps(jobs, indent=2))

    cache: dict[str, tuple] = {}
    for job in jobs:
        ds = job["dataset"]
        C = DATASETS[ds]
        if ds not in cache:
            logits, y = load_teacher_stack(ds, C)
            phi = feature_matrix(softmax(logits, axis=-1), "all")
            cache[ds] = (logits, y, phi)
        logits, y, phi = cache[ds]
        dataset = load_dataset_by_name(ds)

        for seed in SEEDS:
            splits = subsample_low_label(dataset, L_CE, seed, 0.3)
            unlabeled = splits.unlabeled_indices

            mean_src = MEAN_SRC / (
                f"{ds}_kd_multi-mean-guided_L{L_CE}_s{seed}_{LABEL_ACCOUNTING_TAG}.json"
            )
            mean_dst = args.output_dir / mean_src.name
            if mean_src.exists() and not mean_dst.exists():
                args.output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(mean_src, mean_dst)
                print(f"Reused mean baseline {mean_dst.name}")
            elif not mean_src.exists():
                mean_p = average_teacher_soft_labels(GUIDED_TEACHERS, ds, C, KD_TAU)
                train_one(ds, "multi-mean-guided", mean_p, seed, args.output_dir, {"method": "logit_mean"}, args.dry_run)

            if seed == 42:
                oracle_p = oracle_teacher_soft_labels(
                    GUIDED_TEACHERS, dataset, C, KD_TAU, oracle_temperature=3.0
                )
                train_one(
                    ds, "oracle-tau3-guided", oracle_p, seed, args.output_dir,
                    {"method": "oracle_tau3"}, args.dry_run,
                )

            for point_name, n_per in job["points"]:
                if point_name == "current_l10":
                    cal_idx = extra_cal_from_unlabeled(y, unlabeled, 3, C, seed)
                elif point_name == "fullpool":
                    cal_idx = np.arange(len(y))
                else:
                    cal_idx = extra_cal_from_unlabeled(y, unlabeled, int(n_per), C, seed)

                p_cal = _kd_targets(calibrated_prob_mean(logits, y, cal_idx))
                train_one(
                    ds, f"calibmean-{point_name}", p_cal, seed, args.output_dir,
                    {"method": "calibmean", "point": point_name, "n_cal": int(len(cal_idx))},
                    args.dry_run,
                )
                if job["best_variant"] is None:
                    continue
                variant = _variant(job["best_variant"])
                soft = build_router_targets(variant, logits, y, phi, cal_idx, seed)
                train_one(
                    ds, f"{variant.name}-{point_name}", soft, seed, args.output_dir,
                    {
                        "method": "router",
                        "variant": variant.name,
                        "point": point_name,
                        "n_cal": int(len(cal_idx)),
                    },
                    args.dry_run,
                )

    if not args.dry_run:
        analysis = analyse_dir(args.output_dir)
        analysis_path = args.output_dir / "mcnemar.json"
        analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        md_lines = ["# Stage E student confirmation (McNemar vs logit-mean)", ""]
        md_lines.append("| dataset | teacher | seed | acc | mean acc | Δ | McNemar p |")
        md_lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in analysis["comparisons"]:
            md_lines.append(
                f"| {row['dataset']} | {row['teacher']} | {row['seed']} | "
                f"{row['acc']:.4f} | {row['mean_acc']:.4f} | {row['delta']:+.4f} | "
                f"{row['p_value']:.4f} |"
            )
        (args.output_dir / "mcnemar.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"Wrote {analysis_path}")


if __name__ == "__main__":
    main()
