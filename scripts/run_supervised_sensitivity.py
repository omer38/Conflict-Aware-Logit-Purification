#!/usr/bin/env python
"""Label-budget-respecting CV and final supervised sensitivity runs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StudentConfig
from src.data.loading import load_dataset_by_name
from src.data.subsampling import subsample_low_label
from src.student.model import build_student, select_device
from src.student.train import (
    build_supervised_examples,
    evaluate_train_indices,
    fit,
    set_seed,
)

DATASETS = ("ag_news", "yahoo", "amazon", "amazon3")
DISPLAY_NAMES = {
    "ag_news": "AG News",
    "yahoo": "Yahoo",
    "amazon": "Amazon",
    "amazon3": "Amazon-3",
}
LABEL_BUDGETS = (10, 50)
SEEDS = (42, 123, 456)
LEARNING_RATES = (1e-5, 2e-5, 5e-5)
EPOCH_CHECKPOINTS = (2, 3, 5)
N_SPLITS = 5
CV_DIR = Path("outputs/student/supervised_sensitivity_cv_v1")
FINAL_DIR = Path("outputs/student/supervised_sensitivity_final_v1")
REPORT_DIR = Path("outputs/reports")
CORE_DIR = Path("outputs/student/prompt_inclusive_core_3seed_v1")
SELECTION_PATH = REPORT_DIR / "supervised_sensitivity_selection.json"


def lr_tag(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("-", "m")


def cv_path(
    dataset_name: str,
    labels_per_class: int,
    seed: int,
    learning_rate: float,
    fold: int,
) -> Path:
    return CV_DIR / (
        f"{dataset_name}_L{labels_per_class}_s{seed}_"
        f"lr{lr_tag(learning_rate)}_fold{fold}.json"
    )


def valid_cv_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return set(result.get("checkpoints", {})) == {
        str(epoch) for epoch in EPOCH_CHECKPOINTS
    }


def run_cv() -> None:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f"CV device: {device}")

    for dataset_name in DATASETS:
        dataset = load_dataset_by_name(dataset_name)
        all_labels = np.asarray(dataset.train[dataset.label_column], dtype=np.int64)
        for labels_per_class in LABEL_BUDGETS:
            for seed in SEEDS:
                splits = subsample_low_label(dataset, labels_per_class, seed)
                labeled_indices = splits.labeled_indices
                labeled_labels = all_labels[labeled_indices]
                folds = list(
                    StratifiedKFold(
                        n_splits=N_SPLITS, shuffle=True, random_state=seed
                    ).split(labeled_indices, labeled_labels)
                )
                for learning_rate in LEARNING_RATES:
                    for fold, (train_pos, val_pos) in enumerate(folds):
                        output_path = cv_path(
                            dataset_name,
                            labels_per_class,
                            seed,
                            learning_rate,
                            fold,
                        )
                        if valid_cv_result(output_path):
                            print(f"SKIP {output_path}")
                            continue

                        train_indices = labeled_indices[train_pos]
                        val_indices = labeled_indices[val_pos]
                        run_seed = seed * 100 + fold
                        set_seed(run_seed)
                        config = StudentConfig(
                            lr=learning_rate,
                            epochs=max(EPOCH_CHECKPOINTS),
                            batch_size=16,
                            max_length=256,
                            alpha=0.0,
                        )
                        train_ds = build_supervised_examples(dataset, train_indices)
                        model, tokenizer = build_student(
                            config.model_name, dataset.num_classes
                        )
                        checkpoints: dict[str, dict] = {}

                        def record_checkpoint(epoch: int) -> None:
                            if epoch not in EPOCH_CHECKPOINTS:
                                return
                            metrics = evaluate_train_indices(
                                model,
                                tokenizer,
                                dataset,
                                val_indices,
                                config,
                                device,
                            )
                            checkpoints[str(epoch)] = metrics
                            print(
                                f"  validation epoch={epoch}: "
                                f"acc={metrics['accuracy']:.4f}, "
                                f"macro_f1={metrics['macro_f1']:.4f}"
                            )

                        print(
                            f"START dataset={dataset_name} L={labels_per_class} "
                            f"seed={seed} lr={learning_rate:g} fold={fold}"
                        )
                        fit(
                            model,
                            tokenizer,
                            train_ds,
                            config,
                            device,
                            epoch_callback=record_checkpoint,
                        )
                        result = {
                            "protocol": "prompt-inclusive-supervised-cv-v1",
                            "dataset": dataset_name,
                            "labels_per_class": labels_per_class,
                            "seed": seed,
                            "learning_rate": learning_rate,
                            "fold": fold,
                            "n_splits": N_SPLITS,
                            "run_seed": run_seed,
                            "selection_split": "labelled training pool only",
                            "selection_metric": "macro_f1",
                            "train_indices": train_indices.tolist(),
                            "validation_indices": val_indices.tolist(),
                            "checkpoints": checkpoints,
                        }
                        output_path.write_text(
                            json.dumps(result, indent=2) + "\n", encoding="utf-8"
                        )
                        del model, tokenizer, train_ds
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()


def select_configs() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selections: dict[str, dict] = {}
    for dataset_name in DATASETS:
        for labels_per_class in LABEL_BUDGETS:
            candidates = []
            for learning_rate in LEARNING_RATES:
                for epochs in EPOCH_CHECKPOINTS:
                    fold_metrics = []
                    for seed in SEEDS:
                        for fold in range(N_SPLITS):
                            path = cv_path(
                                dataset_name,
                                labels_per_class,
                                seed,
                                learning_rate,
                                fold,
                            )
                            if not valid_cv_result(path):
                                raise RuntimeError(f"Missing or invalid CV result: {path}")
                            result = json.loads(path.read_text(encoding="utf-8"))
                            fold_metrics.append(result["checkpoints"][str(epochs)])
                    candidates.append(
                        {
                            "learning_rate": learning_rate,
                            "epochs": epochs,
                            "mean_macro_f1": float(
                                np.mean([row["macro_f1"] for row in fold_metrics])
                            ),
                            "mean_accuracy": float(
                                np.mean([row["accuracy"] for row in fold_metrics])
                            ),
                            "n_fold_scores": len(fold_metrics),
                        }
                    )

            def selection_key(row: dict) -> tuple:
                return (
                    row["mean_macro_f1"],
                    row["mean_accuracy"],
                    -abs(math.log(row["learning_rate"] / 2e-5)),
                    -abs(row["epochs"] - 3),
                )

            selected = max(candidates, key=selection_key)
            key = f"{dataset_name}_L{labels_per_class}"
            selections[key] = {
                "dataset": dataset_name,
                "labels_per_class": labels_per_class,
                "selected": selected,
                "candidates": candidates,
                "selection_rule": (
                    "highest mean macro-F1 across 5 folds and seeds 42/123/456; "
                    "ties broken by mean accuracy, then proximity to fixed config"
                ),
                "test_metrics_used_for_selection": False,
            }

    SELECTION_PATH.write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {SELECTION_PATH}")
    return selections


def final_result_path(dataset_name: str, labels_per_class: int, seed: int) -> Path:
    return FINAL_DIR / (
        f"{dataset_name}_supervised_none_L{labels_per_class}_s{seed}_"
        "prompt-inclusive-v1.json"
    )


def run_final(selections: dict) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASETS:
        for labels_per_class in LABEL_BUDGETS:
            selected = selections[f"{dataset_name}_L{labels_per_class}"]["selected"]
            for seed in SEEDS:
                output_path = final_result_path(
                    dataset_name, labels_per_class, seed
                )
                if output_path.exists():
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                    config = result.get("config", {})
                    if (
                        config.get("lr") == selected["learning_rate"]
                        and config.get("epochs") == selected["epochs"]
                    ):
                        print(f"SKIP valid final {output_path}")
                        continue
                    raise RuntimeError(
                        f"Existing final result has wrong selected config: {output_path}"
                    )
                command = [
                    sys.executable,
                    "scripts/train_student.py",
                    "--dataset",
                    dataset_name,
                    "--mode",
                    "supervised",
                    "--labels-per-class",
                    str(labels_per_class),
                    "--seed",
                    str(seed),
                    "--lr",
                    str(selected["learning_rate"]),
                    "--epochs",
                    str(selected["epochs"]),
                    "--batch-size",
                    "16",
                    "--max-length",
                    "256",
                    "--output-dir",
                    str(FINAL_DIR),
                ]
                print("RUN", " ".join(command))
                subprocess.run(command, check=True)


def load_accuracy(path: Path) -> float:
    if not path.exists():
        raise RuntimeError(f"Missing result: {path}")
    return float(json.loads(path.read_text(encoding="utf-8"))["metrics"]["accuracy"])


def core_result_path(
    dataset_name: str,
    method: str,
    labels_per_class: int,
    seed: int,
) -> Path:
    mode, teacher = {
        "supervised": ("supervised", "none"),
        "raw_qwen": ("kd", "qwen3.5-27b-vllm-guided"),
    }[method]
    return CORE_DIR / (
        f"{dataset_name}_{mode}_{teacher}_L{labels_per_class}_s{seed}_"
        "prompt-inclusive-v1.json"
    )


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def report_results(selections: dict) -> None:
    rows = []
    for dataset_name in DATASETS:
        for labels_per_class in LABEL_BUDGETS:
            fixed = [
                load_accuracy(
                    core_result_path(
                        dataset_name, "supervised", labels_per_class, seed
                    )
                )
                for seed in SEEDS
            ]
            raw_qwen = [
                load_accuracy(
                    core_result_path(
                        dataset_name, "raw_qwen", labels_per_class, seed
                    )
                )
                for seed in SEEDS
            ]
            selected_supervised = [
                load_accuracy(
                    final_result_path(dataset_name, labels_per_class, seed)
                )
                for seed in SEEDS
            ]
            fixed_mean, fixed_sd = mean_sd(fixed)
            selected_mean, selected_sd = mean_sd(selected_supervised)
            raw_mean, raw_sd = mean_sd(raw_qwen)
            selection = selections[f"{dataset_name}_L{labels_per_class}"]["selected"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "display_name": DISPLAY_NAMES[dataset_name],
                    "labels_per_class": labels_per_class,
                    "fixed_supervised": {"mean": fixed_mean, "sd": fixed_sd},
                    "cv_selected_supervised": {
                        "mean": selected_mean,
                        "sd": selected_sd,
                    },
                    "selected_learning_rate": selection["learning_rate"],
                    "selected_epochs": selection["epochs"],
                    "raw_qwen_kd": {"mean": raw_mean, "sd": raw_sd},
                    "conclusion_survives": raw_mean > selected_mean,
                }
            )

    survived = sum(row["conclusion_survives"] for row in rows)
    report = {
        "protocol": "prompt-inclusive-supervised-cv-v1",
        "selection_uses_test_metrics": False,
        "seeds": list(SEEDS),
        "rows": rows,
        "conclusions_surviving": survived,
        "total_conclusions": len(rows),
    }
    json_path = REPORT_DIR / "supervised_sensitivity.json"
    tex_path = REPORT_DIR / "supervised_sensitivity.tex"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Sensitivity of the supervised baseline to "
            r"label-budget-respecting hyperparameter selection. Selection uses "
            r"only the labelled training set. Accuracies are mean \(\pm\) "
            r"sample standard deviation over three seeds.}"
        ),
        r"\label{tab:supervised-sensitivity}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & \(L\) & Fixed supervised & CV-selected supervised & "
            r"Selected \((\eta,E)\) & Raw-Qwen KD \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        fixed = row["fixed_supervised"]
        selected = row["cv_selected_supervised"]
        raw = row["raw_qwen_kd"]
        learning_rate = row["selected_learning_rate"]
        tex_lines.append(
            f"{row['display_name']} & {row['labels_per_class']} & "
            f"${100 * fixed['mean']:.2f} \\pm {100 * fixed['sd']:.2f}$ & "
            f"${100 * selected['mean']:.2f} \\pm {100 * selected['sd']:.2f}$ & "
            f"$({learning_rate:.0e},{row['selected_epochs']})$ & "
            f"${100 * raw['mean']:.2f} \\pm {100 * raw['sd']:.2f}$ \\\\"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            (
                f"% Main-text result: Raw-Qwen KD remains more accurate than "
                f"the CV-selected supervised baseline in {survived} of "
                f"{len(rows)} dataset--budget cells."
            ),
            "",
        ]
    )
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print("\n".join(tex_lines))
    print(f"Saved {json_path} and {tex_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("cv", "select", "final", "report", "all"),
        default="all",
    )
    args = parser.parse_args()

    if args.phase in ("cv", "all"):
        run_cv()
    if args.phase in ("select", "all"):
        selections = select_configs()
    else:
        if not SELECTION_PATH.exists():
            raise RuntimeError(f"Run --phase select first; missing {SELECTION_PATH}")
        selections = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if args.phase in ("final", "all"):
        run_final(selections)
    if args.phase in ("report", "all"):
        report_results(selections)


if __name__ == "__main__":
    main()
