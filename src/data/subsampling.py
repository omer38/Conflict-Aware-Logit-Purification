"""Low-label subsampling with prompt-inclusive train / calibration / unlabeled splits.

It takes a full training dataset and splits into three datasets. 

labeled_train — a small labeled set used to train the student model.
labeled_cal — a small labeled set used for calibration/router training.
unlabeled — all remaining training examples, treated as unlabeled for distillation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import Dataset

from .loading import TextClassificationDataset


PROMPT_DEMOS_PER_CLASS = 2


def prompt_demo_indices(
    dataset: TextClassificationDataset,
    per_class: int = PROMPT_DEMOS_PER_CLASS,
) -> dict[int, list[int]]:
    """Return the fixed few-shot demonstration indices used for teacher prompts.

    The extraction cache uses the newest ``per_class`` train rows from each class.
    Keeping this selector here makes the declared label set include exactly the
    gold labels that were supplied to the teachers, without changing prompts or
    requiring teacher re-extraction.
    """
    pool: dict[int, list[int]] = {cls: [] for cls in range(dataset.num_classes)}
    train = dataset.train
    for idx in range(len(train) - 1, -1, -1):
        cls = int(train[idx][dataset.label_column])
        if len(pool[cls]) < per_class:
            pool[cls].append(idx)
        if all(len(indices) >= per_class for indices in pool.values()):
            break

    missing = [cls for cls, indices in pool.items() if len(indices) < per_class]
    if missing:
        raise ValueError(
            f"Need {per_class} prompt demonstrations for every class; "
            f"insufficient examples for class ids: {missing}"
        )
    return pool


@dataclass
class DataSplits:
    """Result of low-label subsampling.

    Attributes:
        labeled_train: CE portion when a learned router reserves calibration data.
        labeled_cal: Calibration portion used only for learned-router training.
        unlabeled: Remaining examples (labels held out) for teacher-guided distillation.
        labeled_train_indices: Original ``dataset.train`` row positions of ``labeled_train``.
        labeled_cal_indices: Original ``dataset.train`` row positions of ``labeled_cal``.
        unlabeled_indices: Original ``dataset.train`` row positions of ``unlabeled``.
            These integers match the ``sample_idx`` join key in teacher report CSVs.
        labeled_indices: All declared labelled rows, including the fixed prompt
            demonstrations. ``labeled_train`` and ``labeled_cal`` partition it.
        prompt_demo_indices: Fixed teacher-prompt rows included in every declared
            labelled set, flattened across classes.
        labels_per_class: The L value used.
        seed: Random seed for reproducibility.
        cal_ratio: Fraction of D_L partitioned for possible router calibration.
    """

    labeled_train: Dataset
    labeled_cal: Dataset
    unlabeled: Dataset
    labeled_train_indices: np.ndarray
    labeled_cal_indices: np.ndarray
    unlabeled_indices: np.ndarray
    labeled_indices: np.ndarray
    prompt_demo_indices: np.ndarray
    labels_per_class: int
    seed: int
    cal_ratio: float


def subsample_low_label(
    dataset: TextClassificationDataset,
    labels_per_class: int,
    seed: int,
    cal_ratio: float = 0.3,
) -> DataSplits:
    """Sample prompt-inclusive *labels_per_class* examples per class.

    Parameters
    ----------
    dataset:
        A loaded ``TextClassificationDataset`` (only the *train* split is subsampled).
    labels_per_class:
        Number of labelled examples to keep per class (L in the paper), including
        the two gold-labelled demonstrations used in every teacher prompt.
    seed:
        Random seed for deterministic subsampling.
    cal_ratio:
        Fraction of the labeled set reserved for router calibration (default 0.3).

    Returns
    -------
    DataSplits with ``labeled_train``, ``labeled_cal``, and ``unlabeled`` datasets.

    Raises
    ------
    ValueError
        If any class has fewer examples than *labels_per_class* or if
        *cal_ratio* would leave zero examples in either split for a class.

    Example:
    At L=10, every declared set contains the two fixed prompt demonstrations plus
    eight seed-sampled examples. A router may reserve 3 of these 10 labels for
    calibration; methods without learned calibration use all 10 for CE.
    """
    if labels_per_class < PROMPT_DEMOS_PER_CLASS:
        raise ValueError(
            f"labels_per_class must be at least {PROMPT_DEMOS_PER_CLASS} to include "
            "the fixed teacher-prompt demonstrations."
        )

    train_ds: Dataset = dataset.train
    labels = np.array(train_ds[dataset.label_column])
    rng = np.random.default_rng(seed)
    demos_by_class = prompt_demo_indices(dataset)

    labeled_indices: list[int] = []
    unlabeled_indices: list[int] = []

    for cls in range(dataset.num_classes):
        #Finds all training examples whose label is equal to the current class return list like [True,False,..]
        cls_indices = np.where(labels == cls)[0]
        demo_indices = np.asarray(demos_by_class[cls], dtype=np.int64)
        if len(cls_indices) < labels_per_class:
        # This checks whether the class has enough examples.
            raise ValueError(
                f"Class {cls} has only {len(cls_indices)} examples, "
                f"need {labels_per_class}"
            )
        available = np.setdiff1d(cls_indices, demo_indices)
        sampled_count = labels_per_class - len(demo_indices)
        sampled = rng.choice(available, size=sampled_count, replace=False)
        chosen = np.concatenate([demo_indices, sampled])
        # Every D_L contains the fixed prompt demos; only the remaining L-2
        # examples are seed-sampled.
        rest = np.setdiff1d(cls_indices, chosen) # Put the remaining into the unlabeled set
        labeled_indices.extend(chosen.tolist())
        unlabeled_indices.extend(rest.tolist())

    labeled_indices = np.array(labeled_indices, dtype=np.int64)
    unlabeled_indices = np.array(unlabeled_indices, dtype=np.int64)

    # Stratified split of labeled set into train and calibration portions.
    labeled_labels = labels[labeled_indices]
    train_idx: list[int] = []
    cal_idx: list[int] = []

    for cls in range(dataset.num_classes):
        cls_mask = labeled_labels == cls
        cls_positions = np.where(cls_mask)[0] # Gives the positions of this class inside the labeled set.
        n_cal = max(1, round(len(cls_positions) * cal_ratio)) # To ensure we have at least one calibration example
        n_train = len(cls_positions) - n_cal

        if n_train < 1 or n_cal < 1:
            raise ValueError(
                f"labels_per_class={labels_per_class} with cal_ratio={cal_ratio} "
                f"leaves 0 examples in one split for class {cls}. "
                f"Increase labels_per_class or adjust cal_ratio."
            )

        perm = rng.permutation(cls_positions)
        cal_idx.extend(perm[:n_cal].tolist())
        train_idx.extend(perm[n_cal:].tolist())

    labeled_train_indices = labeled_indices[train_idx]
    labeled_cal_indices = labeled_indices[cal_idx]
    flat_demo_indices = np.array(
        [idx for cls in range(dataset.num_classes) for idx in demos_by_class[cls]],
        dtype=np.int64,
    )

    return DataSplits(
        labeled_train=train_ds.select(labeled_train_indices.tolist()),
        labeled_cal=train_ds.select(labeled_cal_indices.tolist()),
        unlabeled=train_ds.select(unlabeled_indices.tolist()),
        labeled_train_indices=labeled_train_indices,
        labeled_cal_indices=labeled_cal_indices,
        unlabeled_indices=unlabeled_indices,
        labeled_indices=labeled_indices,
        prompt_demo_indices=flat_demo_indices,
        labels_per_class=labels_per_class,
        seed=seed,
        cal_ratio=cal_ratio,
    )
