"""Checks for stable dataset split identities used to protect teacher-cache joins."""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import split_identity


def test_split_identity_is_stable_and_order_sensitive() -> None:
    split = Dataset.from_dict(
        {"text": ["alpha", "beta", "gamma"], "label": [0, 1, 0]}
    )
    same = Dataset.from_dict(
        {"text": ["alpha", "beta", "gamma"], "label": [0, 1, 0]}
    )
    reordered = split.select([1, 0, 2])
    relabelled = Dataset.from_dict(
        {"text": ["alpha", "beta", "gamma"], "label": [0, 0, 0]}
    )

    identity = split_identity(split, "text", "label")
    assert identity == split_identity(same, "text", "label")
    assert identity["size"] == 3
    assert identity["class_counts"] == [2, 1]
    assert identity["sha256"] != split_identity(reordered, "text", "label")["sha256"]
    assert identity["sha256"] != split_identity(relabelled, "text", "label")["sha256"]


if __name__ == "__main__":
    test_split_identity_is_stable_and_order_sensitive()
    print("Dataset identity checks passed.")
