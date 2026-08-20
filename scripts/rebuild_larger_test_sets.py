#!/usr/bin/env python
"""Safely rebuild the four datasets with larger, fixed evaluation splits.

All candidate artifacts are written to a staging directory first. The rebuild is
installed only after every training split has identical ordered text/label content
by stable hash and every guided-teacher cache still matches train-row positions.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_amazon3 import amazon3_metadata, build_amazon3_splits
from src.data.loading import (
    AG_NEWS_DIR,
    AG_NEWS_TEST_PER_CLASS,
    AG_NEWS_TRAIN_PER_CLASS,
    AMAZON3_DIR,
    AMAZON_DIR,
    AMAZON_NUM_CLASSES,
    AMAZON_TEST_PER_CLASS,
    AMAZON_TRAIN_PER_CLASS,
    DATA_ARTIFACT_VERSION,
    YAHOO_DIR,
    YAHOO_TEST_PER_CLASS,
    YAHOO_TRAIN_PER_CLASS,
    TextClassificationDataset,
    build_ag_news_splits,
    build_amazon_splits,
    build_yahoo_splits,
    split_identity,
)
from src.student.soft_labels import GUIDED_TEACHERS, REPORT_DIR

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _metadata(
    source: str,
    dataset: DatasetDict,
    label_names: list[str],
    train_per_class: int,
    test_per_class: int,
) -> dict:
    return {
        "source": source,
        "artifact_version": DATA_ARTIFACT_VERSION,
        "num_classes": len(label_names),
        "label_names": label_names,
        "train_per_class": train_per_class,
        "test_per_class": test_per_class,
        "seed": 42,
        "train_size": len(dataset["train"]),
        "test_size": len(dataset["test"]),
        "train_identity": split_identity(dataset["train"], "text", "label"),
        "test_identity": split_identity(dataset["test"], "text", "label"),
    }


def _verify_train_unchanged(name: str, old: DatasetDict, new: DatasetDict) -> None:
    old_identity = split_identity(old["train"], "text", "label")
    new_identity = split_identity(new["train"], "text", "label")
    if old_identity != new_identity:
        raise RuntimeError(
            f"{name}: training split changed; refusing to replace artifacts\n"
            f"old={old_identity}\nnew={new_identity}"
        )
    print(f"  {name}: train identity preserved ({new_identity['sha256'][:12]}...) ")


def _verify_test_shape(
    name: str, dataset: DatasetDict, per_class: int, num_classes: int
) -> None:
    identity = split_identity(dataset["test"], "text", "label")
    expected_counts = [per_class] * num_classes
    if identity["class_counts"] != expected_counts:
        raise RuntimeError(
            f"{name}: test counts {identity['class_counts']} != {expected_counts}"
        )
    print(
        f"  {name}: test={identity['size']} counts={identity['class_counts']} "
        f"sha256={identity['sha256'][:12]}..."
    )


def _verify_teacher_cache_alignment(name: str, train_split) -> None:
    labels = train_split["label"]
    for teacher in GUIDED_TEACHERS:
        path = REPORT_DIR / f"{teacher}_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing teacher cache required for {name}: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(labels):
            raise RuntimeError(
                f"{path}: {len(rows)} rows != {len(labels)} training examples"
            )
        for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
            if int(row["sample_idx"]) != index:
                raise RuntimeError(f"{path}: non-positional sample_idx at row {index}")
            if "true_label_id" in row and int(row["true_label_id"]) != int(label):
                raise RuntimeError(f"{path}: label mismatch at train row {index}")
    print(f"  {name}: all guided-teacher caches align with the preserved train split")


def _save_staged(path: Path, dataset: DatasetDict, metadata: dict) -> None:
    dataset.save_to_disk(path)
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _install_staged(staging_root: Path, targets: dict[str, Path]) -> None:
    backup_root = Path(tempfile.mkdtemp(prefix=".larger-test-backup-", dir=DATA_ROOT))
    installed: list[str] = []
    try:
        for name, target in targets.items():
            backup = backup_root / name
            target.rename(backup)
            try:
                (staging_root / name).rename(target)
            except Exception:
                backup.rename(target)
                raise
            installed.append(name)
    except Exception:
        for name in reversed(installed):
            target = targets[name]
            (staging_root / name).parent.mkdir(parents=True, exist_ok=True)
            target.rename(staging_root / name)
            (backup_root / name).rename(target)
        raise
    else:
        shutil.rmtree(backup_root)


def main() -> None:
    targets = {
        "ag_news": AG_NEWS_DIR,
        "yahoo": YAHOO_DIR,
        "amazon": AMAZON_DIR,
        "amazon3": AMAZON3_DIR,
    }
    old = {name: load_from_disk(path) for name, path in targets.items()}

    print("Building candidates from source datasets...")
    ag_news, ag_names = build_ag_news_splits(load_dataset("fancyzhx/ag_news"))
    yahoo, yahoo_names = build_yahoo_splits(load_dataset("mteb/yahoo_answers_topics"))
    amazon, amazon_names = build_amazon_splits(
        load_dataset("vgaraujov/amazon_review_full")
    )
    amazon_container = TextClassificationDataset(
        name="amazon",
        train=amazon["train"],
        test=amazon["test"],
        text_column="text",
        label_column="label",
        num_classes=AMAZON_NUM_CLASSES,
        label_names=amazon_names,
    )
    amazon3, _ = build_amazon3_splits(amazon_container)

    candidates = {
        "ag_news": ag_news,
        "yahoo": yahoo,
        "amazon": amazon,
        "amazon3": amazon3,
    }
    metadata = {
        "ag_news": _metadata(
            "fancyzhx/ag_news",
            ag_news,
            ag_names,
            AG_NEWS_TRAIN_PER_CLASS,
            AG_NEWS_TEST_PER_CLASS,
        ),
        "yahoo": _metadata(
            "mteb/yahoo_answers_topics",
            yahoo,
            yahoo_names,
            YAHOO_TRAIN_PER_CLASS,
            YAHOO_TEST_PER_CLASS,
        ),
        "amazon": _metadata(
            "vgaraujov/amazon_review_full",
            amazon,
            amazon_names,
            AMAZON_TRAIN_PER_CLASS,
            AMAZON_TEST_PER_CLASS,
        ),
        "amazon3": {
            **amazon3_metadata(amazon3),
            "artifact_version": DATA_ARTIFACT_VERSION,
        },
    }

    expected_test_shape = {
        "ag_news": (AG_NEWS_TEST_PER_CLASS, len(ag_names)),
        "yahoo": (YAHOO_TEST_PER_CLASS, len(yahoo_names)),
        "amazon": (AMAZON_TEST_PER_CLASS, AMAZON_NUM_CLASSES),
        "amazon3": (AMAZON_TEST_PER_CLASS, 3),
    }
    print("Verifying candidates before writing anything...")
    for name, candidate in candidates.items():
        _verify_train_unchanged(name, old[name], candidate)
        per_class, num_classes = expected_test_shape[name]
        _verify_test_shape(name, candidate, per_class, num_classes)
        _verify_teacher_cache_alignment(name, candidate["train"])

    staging_root = Path(tempfile.mkdtemp(prefix=".larger-test-staging-", dir=DATA_ROOT))
    try:
        for name, candidate in candidates.items():
            _save_staged(staging_root / name, candidate, metadata[name])
        _install_staged(staging_root, targets)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    print("Installed larger fixed test sets successfully.")


if __name__ == "__main__":
    main()
