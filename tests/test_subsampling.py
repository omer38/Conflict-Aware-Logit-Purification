"""Prompt-inclusive low-label sampling checks.

Run:
    python tests/test_subsampling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.data.subsampling import (
    PROMPT_DEMOS_PER_CLASS,
    prompt_demo_indices,
    subsample_low_label,
)


def test_prompt_demos_are_counted_in_every_d_l() -> None:
    """Every D_L contains the fixed teacher-prompt labels, for each dataset."""
    for dataset_name in ("ag_news", "yahoo", "amazon", "amazon3"):
        dataset = load_dataset_by_name(dataset_name)
        splits = subsample_low_label(dataset, labels_per_class=10, seed=42)
        demos = prompt_demo_indices(dataset)
        declared = set(splits.labeled_indices.tolist())
        unlabeled = set(splits.unlabeled_indices.tolist())

        assert len(splits.labeled_indices) == 10 * dataset.num_classes
        assert len(splits.labeled_train_indices) == 7 * dataset.num_classes
        assert len(splits.labeled_cal_indices) == 3 * dataset.num_classes
        for class_id, class_demos in demos.items():
            assert len(class_demos) == PROMPT_DEMOS_PER_CLASS, class_id
            assert set(class_demos).issubset(declared), class_id
            assert not set(class_demos) & unlabeled, class_id


def test_l_equals_prompt_budget_is_valid() -> None:
    """At L=2, D_L consists exactly of the two fixed demos per class."""
    dataset = load_dataset_by_name("ag_news")
    splits = subsample_low_label(dataset, labels_per_class=2, seed=42)
    expected = {
        idx
        for class_demos in prompt_demo_indices(dataset).values()
        for idx in class_demos
    }
    assert set(splits.labeled_indices.tolist()) == expected


def test_calibration_allocation_diagnostic_split() -> None:
    """The diagnostic reserves 70% of L=10 labels for the learned router."""
    dataset = load_dataset_by_name("ag_news")
    splits = subsample_low_label(
        dataset, labels_per_class=10, seed=42, cal_ratio=0.7
    )

    assert len(splits.labeled_train_indices) == 3 * dataset.num_classes
    assert len(splits.labeled_cal_indices) == 7 * dataset.num_classes


def main() -> None:
    test_prompt_demos_are_counted_in_every_d_l()
    test_l_equals_prompt_budget_is_valid()
    test_calibration_allocation_diagnostic_split()
    print("Prompt-inclusive subsampling checks passed.")


if __name__ == "__main__":
    main()
