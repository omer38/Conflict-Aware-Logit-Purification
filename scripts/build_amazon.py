#!/usr/bin/env python
"""Build and save the balanced Amazon Review Full subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import (
    AMAZON_DIR,
    AMAZON_NUM_CLASSES,
    AMAZON_TEST_PER_CLASS,
    AMAZON_TRAIN_PER_CLASS,
    DATA_ARTIFACT_VERSION,
    build_amazon_splits,
    split_identity,
)


def main() -> None:
    ds = load_dataset("vgaraujov/amazon_review_full")
    amazon, label_names = build_amazon_splits(ds)

    AMAZON_DIR.mkdir(parents=True, exist_ok=True)
    amazon.save_to_disk(AMAZON_DIR)

    metadata = {
        "source": "vgaraujov/amazon_review_full",
        "artifact_version": DATA_ARTIFACT_VERSION,
        "num_classes": AMAZON_NUM_CLASSES,
        "label_names": label_names,
        "train_per_class": AMAZON_TRAIN_PER_CLASS,
        "test_per_class": AMAZON_TEST_PER_CLASS,
        "seed": 42,
        "train_size": len(amazon["train"]),
        "test_size": len(amazon["test"]),
        "train_identity": split_identity(amazon["train"], "text", "label"),
        "test_identity": split_identity(amazon["test"], "text", "label"),
    }
    (AMAZON_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved amazon to {AMAZON_DIR}")
    print(f"  Classes     : {metadata['num_classes']}")
    print(f"  Train size  : {metadata['train_size']}")
    print(f"  Test size   : {metadata['test_size']}")
    print(f"  Label names : {label_names}")


if __name__ == "__main__":
    main()
