#!/usr/bin/env python
"""Build and save the balanced Yahoo Answers Topics subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import (
    YAHOO_DIR,
    YAHOO_TEST_PER_CLASS,
    YAHOO_TRAIN_PER_CLASS,
    DATA_ARTIFACT_VERSION,
    build_yahoo_splits,
    split_identity,
)


def main() -> None:
    ds = load_dataset("mteb/yahoo_answers_topics")
    yahoo, label_names = build_yahoo_splits(ds)

    YAHOO_DIR.mkdir(parents=True, exist_ok=True)
    yahoo.save_to_disk(YAHOO_DIR)

    metadata = {
        "source": "mteb/yahoo_answers_topics",
        "artifact_version": DATA_ARTIFACT_VERSION,
        "num_classes": len(label_names),
        "label_names": label_names,
        "train_per_class": YAHOO_TRAIN_PER_CLASS,
        "test_per_class": YAHOO_TEST_PER_CLASS,
        "seed": 42,
        "train_size": len(yahoo["train"]),
        "test_size": len(yahoo["test"]),
        "train_identity": split_identity(yahoo["train"], "text", "label"),
        "test_identity": split_identity(yahoo["test"], "text", "label"),
    }
    (YAHOO_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved yahoo to {YAHOO_DIR}")
    print(f"  Classes     : {metadata['num_classes']}")
    print(f"  Train size  : {metadata['train_size']}")
    print(f"  Test size   : {metadata['test_size']}")
    print(f"  Label names : {label_names}")


if __name__ == "__main__":
    main()
