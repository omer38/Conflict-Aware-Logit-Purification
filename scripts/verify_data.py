#!/usr/bin/env python
"""Verify the data loading and low-label subsampling pipeline.

Loads each dataset, runs subsampling for every (L, seed) combination in the
default ExperimentGrid, and prints class-distribution diagnostics.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentGrid
from src.data.loading import load_dataset_by_name, split_identity
from src.data.subsampling import subsample_low_label


def print_header(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def class_distribution(split, label_column: str) -> Counter:
    return Counter(split[label_column])


def verify_artifact_identity(dataset) -> list[str]:
    """Verify saved metadata identifies the exact loaded train/test rows."""
    errors: list[str] = []
    for split_name in ("train", "test"):
        expected = dataset.metadata.get(f"{split_name}_identity")
        if expected is None:
            errors.append(f"Missing {split_name}_identity in artifact metadata")
            continue
        actual = split_identity(
            getattr(dataset, split_name), dataset.text_column, dataset.label_column
        )
        if actual != expected:
            errors.append(
                f"{split_name} identity mismatch: metadata={expected}, actual={actual}"
            )
    return errors


def verify_split(
    dataset_name: str,
    num_classes: int,
    label_column: str,
    splits,
    labels_per_class: int,
    cal_ratio: float,
) -> list[str]:
    """Return a list of error messages (empty means all checks passed)."""
    errors: list[str] = []

    train_dist = class_distribution(splits.labeled_train, label_column)
    cal_dist = class_distribution(splits.labeled_cal, label_column)
    unlabeled_dist = class_distribution(splits.unlabeled, label_column)

    total_labeled = len(splits.labeled_train) + len(splits.labeled_cal)
    expected_labeled = labels_per_class * num_classes

    if total_labeled != expected_labeled:
        errors.append(
            f"Total labeled = {total_labeled}, expected {expected_labeled}"
        )

    for cls in range(num_classes):
        n_train = train_dist.get(cls, 0)
        n_cal = cal_dist.get(cls, 0)
        if n_train + n_cal != labels_per_class:
            errors.append(
                f"Class {cls}: train({n_train}) + cal({n_cal}) = "
                f"{n_train + n_cal}, expected {labels_per_class}"
            )
        if n_train < 1:
            errors.append(f"Class {cls}: 0 train examples")
        if n_cal < 1:
            errors.append(f"Class {cls}: 0 calibration examples")

    if len(unlabeled_dist) != num_classes:
        errors.append(
            f"Unlabeled covers {len(unlabeled_dist)} classes, expected {num_classes}"
        )

    return errors


def main() -> None:
    grid = ExperimentGrid()

    dataset_cache: dict[str, object] = {}
    all_passed = True

    for cfg in grid.iter_configs():
        if cfg.dataset_name not in dataset_cache:
            print_header(f"Loading {cfg.dataset_name}")
            ds = load_dataset_by_name(cfg.dataset_name)
            dataset_cache[cfg.dataset_name] = ds
            print(f"  Classes     : {ds.num_classes}")
            print(f"  Train size  : {len(ds.train)}")
            print(f"  Test size   : {len(ds.test)}")
            print(f"  Label names : {ds.label_names[:5]}{'...' if ds.num_classes > 5 else ''}")
            identity_errors = verify_artifact_identity(ds)
            if identity_errors:
                all_passed = False
                for error in identity_errors:
                    print(f"  FAIL  {error}")
            else:
                print("  OK    Saved train/test identities match loaded rows")

        ds = dataset_cache[cfg.dataset_name]

        tag = f"{cfg.dataset_name} | L={cfg.labels_per_class} | seed={cfg.seed}"
        splits = subsample_low_label(ds, cfg.labels_per_class, cfg.seed, cfg.cal_ratio)

        errors = verify_split(
            cfg.dataset_name,
            ds.num_classes,
            ds.label_column,
            splits,
            cfg.labels_per_class,
            cfg.cal_ratio,
        )

        n_train = len(splits.labeled_train)
        n_cal = len(splits.labeled_cal)
        n_unlabeled = len(splits.unlabeled)
        total = n_train + n_cal + n_unlabeled

        if errors:
            all_passed = False
            print(f"\n  FAIL  {tag}")
            for e in errors:
                print(f"    - {e}")
        else:
            print(f"  OK    {tag}  |  train={n_train}  cal={n_cal}  unlabeled={n_unlabeled}  total={total}")

    # Amazon-3 is a derived robustness dataset outside the default experiment grid,
    # but its persisted row identities need the same integrity check.
    if "amazon3" not in dataset_cache:
        print_header("Loading amazon3")
        ds_amazon3 = load_dataset_by_name("amazon3")
        print(f"  Classes     : {ds_amazon3.num_classes}")
        print(f"  Train size  : {len(ds_amazon3.train)}")
        print(f"  Test size   : {len(ds_amazon3.test)}")
        identity_errors = verify_artifact_identity(ds_amazon3)
        if identity_errors:
            all_passed = False
            for error in identity_errors:
                print(f"  FAIL  {error}")
        else:
            print("  OK    Saved train/test identities match loaded rows")

    # Reproducibility check: same seed must give identical indices.
    print_header("Reproducibility check")
    ds = dataset_cache[grid.dataset_names[0]]
    s1 = subsample_low_label(ds, 10, seed=42)
    s2 = subsample_low_label(ds, 10, seed=42)
    s3 = subsample_low_label(ds, 10, seed=99)

    same = (
        s1.labeled_train[ds.text_column] == s2.labeled_train[ds.text_column]
        and s1.labeled_cal[ds.text_column] == s2.labeled_cal[ds.text_column]
    )
    diff = s1.labeled_train[ds.text_column] != s3.labeled_train[ds.text_column]

    if same and diff:
        print("  OK    Same seed -> identical splits; different seed -> different splits")
    else:
        all_passed = False
        if not same:
            print("  FAIL  Same seed produced different splits!")
        if not diff:
            print("  FAIL  Different seeds produced identical splits!")

    print_header("Summary")
    if all_passed:
        print("  All checks passed.")
    else:
        print("  Some checks FAILED -- see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
