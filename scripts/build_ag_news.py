#!/usr/bin/env python
"""Build and save the capped AG News subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import (
    AG_NEWS_DIR,
    AG_NEWS_TEST_PER_CLASS,
    AG_NEWS_TRAIN_PER_CLASS,
    DATA_ARTIFACT_VERSION,
    build_ag_news_splits,
    split_identity,
)


def main() -> None:
    ds = load_dataset("fancyzhx/ag_news")
    ag_news, label_names = build_ag_news_splits(ds)

    AG_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    ag_news.save_to_disk(AG_NEWS_DIR)

    metadata = {
        "source": "fancyzhx/ag_news",
        "artifact_version": DATA_ARTIFACT_VERSION,
        "num_classes": len(label_names),
        "label_names": label_names,
        "train_per_class": AG_NEWS_TRAIN_PER_CLASS,
        "test_per_class": AG_NEWS_TEST_PER_CLASS,
        "seed": 42,
        "train_size": len(ag_news["train"]),
        "test_size": len(ag_news["test"]),
        "train_identity": split_identity(ag_news["train"], "text", "label"),
        "test_identity": split_identity(ag_news["test"], "text", "label"),
    }
    (AG_NEWS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved ag_news to {AG_NEWS_DIR}")
    print(f"  Classes     : {metadata['num_classes']}")
    print(f"  Train size  : {metadata['train_size']}")
    print(f"  Test size   : {metadata['test_size']}")
    print(f"  Label names : {label_names}")


if __name__ == "__main__":
    main()
