"""Student training loop: KD + CE distillation and evaluation.

Implements the student objective from the proposal (Eq. 11):

    L = alpha * tau^2 * KL(p_teacher || p_student) + (1 - alpha) * CE(y, z_student)

Supervised baseline is the alpha = 0 special case (CE only). KD distils from
guided teacher soft labels on the unlabeled pool plus CE on the labeled set.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from ..config import StudentConfig
from ..data.loading import TextClassificationDataset
from ..data.subsampling import DataSplits


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _StudentDataset(Dataset):
    """Per-sample text plus optional hard label and teacher soft label."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        soft: np.ndarray,
        has_label: np.ndarray,
        has_soft: np.ndarray,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.soft = soft
        self.has_label = has_label
        self.has_soft = has_soft

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        return {
            "text": self.texts[i],
            "label": int(self.labels[i]),
            "soft": self.soft[i],
            "has_label": bool(self.has_label[i]),
            "has_soft": bool(self.has_soft[i]),
        }


def build_examples(
    dataset: TextClassificationDataset,
    splits: DataSplits,
    mode: str,
    soft_probs: np.ndarray | None,
    reserve_calibration: bool = False,
) -> _StudentDataset:
    """Assemble the training set for ``supervised`` or ``kd`` mode.

    Methods without a learned router use every declared D_L hard label for CE.
    Learned-router methods set ``reserve_calibration=True``: ``labeled_cal`` is
    then reserved for router training and must never leak into CE. In KD mode it
    still contributes to the soft-label distillation pool.
    """
    train = dataset.train
    ce_idx = (
        splits.labeled_train_indices
        if reserve_calibration
        else splits.labeled_indices
    )

    if mode == "supervised":
        return build_supervised_examples(dataset, ce_idx)

    if mode == "kd":
        if soft_probs is None:
            raise ValueError("kd mode requires soft_probs")
        all_idx = np.concatenate(
            [
                ce_idx,
                splits.labeled_cal_indices,
                splits.unlabeled_indices,
            ]
            if reserve_calibration
            else [ce_idx, splits.unlabeled_indices]
        )
        sel = train.select(all_idx.tolist())
        texts = list(sel[dataset.text_column])
        labels = np.full(len(all_idx), -1, dtype=np.int64)
        labels[: len(ce_idx)] = np.asarray(
            train.select(ce_idx.tolist())[dataset.label_column], dtype=np.int64
        )
        soft = soft_probs[all_idx]
        if np.isnan(soft).any():
            raise ValueError(
                "Teacher soft labels contain NaN for some training indices "
                "(incomplete extraction)."
            )
        has_label = np.zeros(len(all_idx), dtype=bool)
        has_label[: len(ce_idx)] = True
        has_soft = np.ones(len(all_idx), dtype=bool)  # train + cal + unlabeled all still get soft-label KD
        return _StudentDataset(texts, labels, soft, has_label, has_soft)

    raise ValueError(f"Unknown mode '{mode}' (expected 'supervised' or 'kd').")


def build_supervised_examples(
    dataset: TextClassificationDataset,
    indices: np.ndarray,
) -> _StudentDataset:
    """Build a CE-only dataset from specified training-pool row indices."""
    selected = dataset.train.select(np.asarray(indices, dtype=np.int64).tolist())
    texts = list(selected[dataset.text_column])
    labels = np.asarray(selected[dataset.label_column], dtype=np.int64)
    soft = np.zeros((len(texts), dataset.num_classes), dtype=np.float32)
    has_label = np.ones(len(texts), dtype=bool)
    has_soft = np.zeros(len(texts), dtype=bool)
    return _StudentDataset(texts, labels, soft, has_label, has_soft)


def _make_collate(tokenizer, max_length: int):
    def collate(batch: list[dict]):
        enc = tokenizer(
            [b["text"] for b in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        soft = torch.tensor(np.stack([b["soft"] for b in batch]), dtype=torch.float32)
        has_label = torch.tensor([b["has_label"] for b in batch], dtype=torch.bool)
        has_soft = torch.tensor([b["has_soft"] for b in batch], dtype=torch.bool)
        return enc, labels, soft, has_label, has_soft

    return collate


def kd_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft: torch.Tensor,
    has_label: torch.Tensor,
    has_soft: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Combined KD + CE loss (proposal Eq. 11)."""
    loss = logits.new_zeros(())
    if alpha < 1.0 and bool(has_label.any()):
        ce = F.cross_entropy(logits[has_label], labels[has_label])
        loss = loss + (1.0 - alpha) * ce
    if alpha > 0.0 and bool(has_soft.any()):
        log_p_student = F.log_softmax(logits[has_soft] / temperature, dim=-1)
        kd = F.kl_div(log_p_student, soft[has_soft], reduction="batchmean")
        loss = loss + alpha * (temperature**2) * kd
    return loss


def fit(
    model,
    tokenizer,
    train_ds: _StudentDataset,
    config: StudentConfig,
    device: torch.device,
    epoch_callback: Callable[[int], None] | None = None,
) -> None:
    """Train the student in place for ``config.epochs`` epochs."""
    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_make_collate(tokenizer, config.max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.to(device)
    model.train()

    for epoch in range(config.epochs):
        running = 0.0
        for enc, labels, soft, has_label, has_soft in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            soft = soft.to(device)
            has_label = has_label.to(device)
            has_soft = has_soft.to(device)

            logits = model(**enc).logits
            loss = kd_ce_loss(
                logits, labels, soft, has_label, has_soft,
                config.alpha, config.temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        print(f"  epoch {epoch + 1}/{config.epochs}  loss {running / len(loader):.4f}")
        if epoch_callback is not None:
            epoch_callback(epoch + 1)
            model.train()


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset: TextClassificationDataset,
    config: StudentConfig,
    device: torch.device,
    return_predictions: bool = False,
) -> dict:
    """Evaluate on the test split, optionally returning paired-test data."""
    model.eval()
    texts = list(dataset.test[dataset.text_column])
    labels = np.asarray(dataset.test[dataset.label_column])

    preds: list[np.ndarray] = []
    for start in range(0, len(texts), config.batch_size):
        enc = tokenizer(
            texts[start : start + config.batch_size],
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        preds.append(logits.argmax(dim=-1).cpu().numpy())

    predictions = np.concatenate(preds)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
    if return_predictions:
        result["test_labels"] = labels.astype(np.int64).tolist()
        result["test_predictions"] = predictions.astype(np.int64).tolist()
    return result


@torch.no_grad()
def evaluate_train_indices(
    model,
    tokenizer,
    dataset: TextClassificationDataset,
    indices: np.ndarray,
    config: StudentConfig,
    device: torch.device,
) -> dict:
    """Evaluate on labelled training-pool rows without touching the test split."""
    model.eval()
    selected = dataset.train.select(np.asarray(indices, dtype=np.int64).tolist())
    texts = list(selected[dataset.text_column])
    labels = np.asarray(selected[dataset.label_column])

    preds: list[np.ndarray] = []
    for start in range(0, len(texts), config.batch_size):
        enc = tokenizer(
            texts[start : start + config.batch_size],
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        preds.append(logits.argmax(dim=-1).cpu().numpy())

    predictions = np.concatenate(preds)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
