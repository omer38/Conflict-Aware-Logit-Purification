#!/usr/bin/env python
"""Build Amazon-3 (grouped ordinal sentiment), a secondary/robustness dataset that
groups Amazon-5's star ratings into 1-2 stars -> negative, 3 stars -> neutral,
4-5 stars -> positive, then rebalances to the smallest resulting group.

Reuses the exact same review texts as Amazon-5 (no new source download) and
derives each guided teacher's Amazon-3 logits from its *existing* Amazon-5
logits by softmax + marginalisation over each group (no new LLM queries):
``pseudo_logit_group = log(sum_{c in group} softmax(z)_c)``.

Amazon-5 remains the primary reported dataset; Amazon-3 is an explicit
robustness check on whether the ordinal-adjacency findings are an artifact of
label granularity (see docs/amazon3_ordinal_grouping.md).

Usage:
    uv run python scripts/build_amazon3.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from datasets import DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import DATA_ARTIFACT_VERSION, load_amazon, split_identity
from src.student.soft_labels import GUIDED_TEACHERS, REPORT_DIR, load_teacher_logits
from src.teachers.extraction import make_verbalizers

AMAZON3_DIR = Path(__file__).resolve().parent.parent / "data" / "amazon3"
GROUP_MAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}  # 1-2 stars->negative, 3->neutral, 4-5->positive
GROUP_NAMES = ["negative (1-2 stars)", "neutral (3 stars)", "positive (4-5 stars)"]
SEED = 42


def _grouped_balanced_indices(labels: np.ndarray, seed: int) -> list[int]:
    grouped = np.array([GROUP_MAP[int(label)] for label in labels])
    per_class = min(int((grouped == g).sum()) for g in range(3))
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for g in range(3):
        idx = np.where(grouped == g)[0]
        chosen.extend(rng.choice(idx, size=per_class, replace=False).tolist())
    return chosen


def _build_split(split, seed: int):
    labels = np.asarray(split["label"])
    chosen = _grouped_balanced_indices(labels, seed)
    subset = split.select(chosen)
    grouped_labels = [GROUP_MAP[int(labels[i])] for i in chosen]
    subset = subset.remove_columns(["label"]).add_column("label", grouped_labels)
    return subset, chosen


def _grouped_pseudo_logits(logits5: np.ndarray) -> np.ndarray:
    z = logits5 - logits5.max(axis=-1, keepdims=True)
    p5 = np.exp(z)
    p5 /= p5.sum(axis=-1, keepdims=True)
    p3 = np.stack([p5[:, [0, 1]].sum(-1), p5[:, 2], p5[:, [3, 4]].sum(-1)], axis=-1)
    return np.log(np.clip(p3, 1e-12, None))


def build_amazon3_splits(amazon):
    """Group and rebalance Amazon-5 while preserving positional train mapping."""
    train_split, train_chosen = _build_split(amazon.train, SEED)
    test_split, _ = _build_split(amazon.test, SEED + 1)
    return DatasetDict({"train": train_split, "test": test_split}), train_chosen


def amazon3_metadata(dataset: DatasetDict) -> dict:
    """Metadata shared by direct and staged Amazon-3 builds."""
    train_identity = split_identity(dataset["train"], "text", "label")
    test_identity = split_identity(dataset["test"], "text", "label")
    return {
        "source": "grouped from data/amazon (balanced Amazon Review Full, 5 classes)",
        "artifact_version": DATA_ARTIFACT_VERSION,
        "group_map": {"1-2 stars": "negative", "3 stars": "neutral", "4-5 stars": "positive"},
        "num_classes": 3,
        "label_names": GROUP_NAMES,
        "seed": SEED,
        "train_per_class": min(train_identity["class_counts"]),
        "test_per_class": min(test_identity["class_counts"]),
        "train_size": len(dataset["train"]),
        "test_size": len(dataset["test"]),
        "train_identity": train_identity,
        "test_identity": test_identity,
        "note": "Robustness/secondary dataset; Amazon-5 remains the primary reported task.",
    }


def main() -> None:
    amazon = load_amazon()
    amazon3, train_chosen = build_amazon3_splits(amazon)

    AMAZON3_DIR.mkdir(parents=True, exist_ok=True)
    amazon3.save_to_disk(AMAZON3_DIR)

    metadata = amazon3_metadata(amazon3)
    (AMAZON3_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved amazon3 to {AMAZON3_DIR}: train={len(amazon3['train'])} test={len(amazon3['test'])}")

    train_labels_grouped = np.asarray(amazon3["train"]["label"])
    verbalizers = make_verbalizers(3)
    for teacher in GUIDED_TEACHERS:
        logits5 = load_teacher_logits(teacher, "amazon", 5)[train_chosen]
        pseudo_logits = _grouped_pseudo_logits(logits5)
        pred = pseudo_logits.argmax(axis=-1)

        out_path = REPORT_DIR / f"{teacher}_amazon3.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_idx", "true_label_id", "true_label_name",
                    "pred_label_id", "pred_label_name", "correct",
                    *[f"logit_{v}" for v in verbalizers],
                ]
            )
            for i in range(len(train_chosen)):
                true_id = int(train_labels_grouped[i])
                pred_id = int(pred[i])
                writer.writerow(
                    [
                        i, true_id, GROUP_NAMES[true_id], pred_id, GROUP_NAMES[pred_id],
                        int(pred_id == true_id), *pseudo_logits[i].tolist(),
                    ]
                )
        acc = float((pred == train_labels_grouped).mean())
        print(f"  {teacher}: grouped train accuracy = {acc:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
