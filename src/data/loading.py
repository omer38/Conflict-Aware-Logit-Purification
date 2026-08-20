"""Dataset loading wrappers for Amazon, Yahoo, and AG News."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

DATA_ARTIFACT_VERSION = "large-test-v1"
AMAZON_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "amazon"
AMAZON_NUM_CLASSES = 5
AMAZON_TRAIN_PER_CLASS = 500
AMAZON_TEST_PER_CLASS = 2000
AMAZON_LABEL_NAMES = [
    "1 star",
    "2 stars",
    "3 stars",
    "4 stars",
    "5 stars",
]
AMAZON3_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "amazon3"
YAHOO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "yahoo"
YAHOO_TRAIN_PER_CLASS = 500
YAHOO_TEST_PER_CLASS = 1000
AG_NEWS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ag_news"
AG_NEWS_TRAIN_PER_CLASS = 1500
AG_NEWS_TEST_PER_CLASS = 1900


@dataclass # Instead of writing lots of boilerplate code, dataclass automatically creates __init__, __repr__, __eq__ etc
class TextClassificationDataset:
    """Unified container for a text-classification dataset.

    Attributes:
        name: Dataset identifier (e.g. "amazon", "yahoo", "ag_news").
        train: HuggingFace Dataset split used for training / subsampling.
        test: HuggingFace Dataset split used for final evaluation.
        text_column: Name of the column containing input text.
        label_column: Name of the column containing integer labels.
        num_classes: Total number of classes.
        label_names: Human-readable class names, indexed by label id.
    """

    name: str
    train: object  # datasets.Dataset
    test: object  # datasets.Dataset
    text_column: str
    label_column: str
    num_classes: int
    label_names: list[str]
    metadata: dict = field(default_factory=dict)


def split_identity(split: Dataset, text_column: str, label_column: str) -> dict:
    """Return a stable content identity for a text-classification split.

    The digest includes row order, text bytes and labels. It therefore protects
    the positional join between training rows and cached teacher logits.
    """
    digest = hashlib.sha256()
    labels = np.asarray(split[label_column], dtype=np.int64)
    texts = split[text_column]
    digest.update(len(labels).to_bytes(8, byteorder="big", signed=False))
    for text, label in zip(texts, labels, strict=True):
        text_bytes = str(text).encode("utf-8")
        digest.update(len(text_bytes).to_bytes(8, byteorder="big", signed=False))
        digest.update(text_bytes)
        digest.update(int(label).to_bytes(8, byteorder="big", signed=True))

    class_counts = np.bincount(labels).astype(int).tolist() if len(labels) else []
    return {
        "size": len(labels),
        "class_counts": class_counts,
        "sha256": digest.hexdigest(),
    }


def _balanced_subsample(
    split: Dataset,
    label_column: str,
    per_class: int,
    seed: int,
    num_classes: int,
) -> Dataset:
    """Sample up to per_class examples per class with a fixed random seed."""
    labels = np.asarray(split[label_column])
    rng = np.random.default_rng(seed)
    chosen_indices: list[int] = []

    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) < per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_indices)} examples, "
                f"need at least {per_class}"
            )
        chosen = rng.choice(cls_indices, size=per_class, replace=False)
        chosen_indices.extend(chosen.tolist())

    return split.select(chosen_indices)


def _format_amazon_split(split: Dataset) -> Dataset:
    """Combine title and content into a single text field."""

    def combine(row: dict) -> dict:
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        text = f"{title}\n\n{content}" if title else content
        return {"text": text, "label": row["label"]}

    return split.map(combine, remove_columns=split.column_names)


def build_amazon_splits(
    ds: DatasetDict,
    train_per_class: int = AMAZON_TRAIN_PER_CLASS,
    test_per_class: int = AMAZON_TEST_PER_CLASS,
    seed: int = 42,
) -> tuple[DatasetDict, list[str]]:
    """Balanced row subset of Amazon Review Full (all 5 star-rating classes kept)."""
    formatted = DatasetDict(
        {
            "train": _format_amazon_split(ds["train"]),
            "test": _format_amazon_split(ds["test"]),
        }
    )

    return (
        DatasetDict(
            {
                "train": _balanced_subsample(
                    formatted["train"], "label", train_per_class, seed, AMAZON_NUM_CLASSES
                ),
                "test": _balanced_subsample(
                    formatted["test"], "label", test_per_class, seed + 1, AMAZON_NUM_CLASSES
                ),
            }
        ),
        AMAZON_LABEL_NAMES,
    )


def build_yahoo_splits(
    ds: DatasetDict,
    train_per_class: int = YAHOO_TRAIN_PER_CLASS,
    test_per_class: int = YAHOO_TEST_PER_CLASS,
    seed: int = 42,
) -> tuple[DatasetDict, list[str]]:
    """Balanced row subset of Yahoo Answers Topics (all classes kept)."""
    label_names = ds["train"].features["label"].names
    num_classes = len(label_names)

    return (
        DatasetDict(
            {
                "train": _balanced_subsample(
                    ds["train"], "label", train_per_class, seed, num_classes
                ),
                "test": _balanced_subsample(
                    ds["test"], "label", test_per_class, seed + 1, num_classes
                ),
            }
        ),
        label_names,
    )


def build_ag_news_splits(
    ds: DatasetDict,
    train_per_class: int = AG_NEWS_TRAIN_PER_CLASS,
    test_per_class: int = AG_NEWS_TEST_PER_CLASS,
    seed: int = 42,
) -> tuple[DatasetDict, list[str]]:
    """Balanced row subset of AG News (all 4 classes kept)."""
    label_names = ds["train"].features["label"].names
    num_classes = len(label_names)

    return (
        DatasetDict(
            {
                "train": _balanced_subsample(
                    ds["train"], "label", train_per_class, seed, num_classes
                ),
                "test": _balanced_subsample(
                    ds["test"], "label", test_per_class, seed + 1, num_classes
                ),
            }
        ),
        label_names,
    )


def load_amazon() -> TextClassificationDataset:
    """Load the capped Amazon Review Full dataset (5 classes, local artifact)."""
    if not AMAZON_DIR.exists():
        raise FileNotFoundError(
            f"amazon not found at {AMAZON_DIR}. "
            "Run: uv run python scripts/build_amazon.py"
        )

    ds = load_from_disk(AMAZON_DIR)
    metadata = json.loads((AMAZON_DIR / "metadata.json").read_text())
    return TextClassificationDataset(
        name="amazon",
        train=ds["train"],
        test=ds["test"],
        text_column="text",
        label_column="label",
        num_classes=metadata["num_classes"],
        label_names=metadata["label_names"],
        metadata=metadata,
    )


def load_amazon3() -> TextClassificationDataset:
    """Load Amazon-3 (Amazon-5 star ratings grouped into negative/neutral/positive
    and rebalanced; secondary/robustness dataset, built by ``scripts/build_amazon3.py``).
    """
    if not AMAZON3_DIR.exists():
        raise FileNotFoundError(
            f"amazon3 not found at {AMAZON3_DIR}. "
            "Run: uv run python scripts/build_amazon3.py"
        )

    ds = load_from_disk(AMAZON3_DIR)
    metadata = json.loads((AMAZON3_DIR / "metadata.json").read_text())
    return TextClassificationDataset(
        name="amazon3",
        train=ds["train"],
        test=ds["test"],
        text_column="text",
        label_column="label",
        num_classes=metadata["num_classes"],
        label_names=metadata["label_names"],
        metadata=metadata,
    )


def load_yahoo() -> TextClassificationDataset:
    """Load the Yahoo Answers Topics subset (10 classes, local artifact)."""
    if not YAHOO_DIR.exists():
        raise FileNotFoundError(
            f"yahoo not found at {YAHOO_DIR}. "
            "Run: uv run python scripts/build_yahoo.py"
        )

    ds = load_from_disk(YAHOO_DIR)
    metadata = json.loads((YAHOO_DIR / "metadata.json").read_text())
    return TextClassificationDataset(
        name="yahoo",
        train=ds["train"],
        test=ds["test"],
        text_column="text",
        label_column="label",
        num_classes=metadata["num_classes"],
        label_names=metadata["label_names"],
        metadata=metadata,
    )


def load_ag_news() -> TextClassificationDataset:
    """Load the capped AG News topic-classification dataset (4 classes, local artifact)."""
    if not AG_NEWS_DIR.exists():
        raise FileNotFoundError(
            f"ag_news not found at {AG_NEWS_DIR}. "
            "Run: uv run python scripts/build_ag_news.py"
        )

    ds = load_from_disk(AG_NEWS_DIR)
    metadata = json.loads((AG_NEWS_DIR / "metadata.json").read_text())
    return TextClassificationDataset(
        name="ag_news",
        train=ds["train"],
        test=ds["test"],
        text_column="text",
        label_column="label",
        num_classes=metadata["num_classes"],
        label_names=metadata["label_names"],
        metadata=metadata,
    )


_LOADERS: dict[str, callable] = {
    "amazon": load_amazon,
    "amazon3": load_amazon3,
    "yahoo": load_yahoo,
    "ag_news": load_ag_news,
}


def load_dataset_by_name(name: str) -> TextClassificationDataset:
    """Load a dataset by its string identifier.

    Instead of doing dataset = load_amazon() or dataset = load_ag_news(), this allow us to use dataset = load_dataset_by_name("amazon")
    Hence, the dataset can be changed from the config without changing the training code.
    """
    try:
        loader = _LOADERS[name]
    except KeyError:
        available = ", ".join(sorted(_LOADERS))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}") from None
    return loader()
