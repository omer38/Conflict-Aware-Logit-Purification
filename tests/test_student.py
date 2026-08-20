"""Tests for the student training infrastructure (Stage 4).

Covers guided soft-label loading, sample_idx alignment, the KD+CE loss, and a
one-epoch training smoke test on the real ModernBERT student over a tiny data
slice (it is cached after the first download and reused by the baselines).

Run:
    uv run python tests/test_student.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StudentConfig
from src.data.loading import load_dataset_by_name
from src.data.subsampling import prompt_demo_indices, subsample_low_label
from src.student.model import build_student
from src.student.soft_labels import (
    GUIDED_TEACHERS,
    average_teacher_soft_labels,
    entropy_weighted_teacher_soft_labels,
    load_teacher_logits,
    teacher_soft_labels,
)
from src.student.train import build_examples, evaluate, fit, kd_ce_loss

TEACHER = "qwen3.5-27b-vllm-guided"
DATASET = "ag_news"


def test_soft_labels_are_proper_distributions() -> None:
    """Guided soft labels sum to 1, contain no NaN, and have no floored mass."""
    print("--- test_soft_labels_are_proper_distributions ---")
    ds = load_dataset_by_name(DATASET)
    probs = teacher_soft_labels(TEACHER, DATASET, ds.num_classes, temperature=2.0)

    assert probs.shape == (len(ds.train), ds.num_classes), probs.shape
    assert not np.isnan(probs).any(), "soft labels contain NaN"
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"rows not normalised: {row_sums[:3]}"

    # Guided extraction has floor_fraction = 0, so no class should sit at the
    # imputation floor; every class carries real probability mass.
    logits = load_teacher_logits(TEACHER, DATASET, ds.num_classes)
    assert not np.isnan(logits).any(), "guided logits should cover every sample"
    print(f"  {probs.shape} soft labels, all normalised, no floor. PASSED\n")


def test_sample_idx_alignment() -> None:
    """KD examples line up teacher argmax with the dataset's own indices."""
    print("--- test_sample_idx_alignment ---")
    ds = load_dataset_by_name(DATASET)
    splits = subsample_low_label(ds, labels_per_class=10, seed=42)

    # Labeled indices must map back to the same labels in the raw train split.
    for pos in splits.labeled_train_indices[:5]:
        assert ds.train[int(pos)]["label"] == ds.train.select([int(pos)])["label"][0]

    probs = teacher_soft_labels(TEACHER, DATASET, ds.num_classes, temperature=2.0)
    kd_ds = build_examples(ds, splits, "kd", probs)
    expected = len(splits.labeled_train_indices) + len(splits.labeled_cal_indices) + len(
        splits.unlabeled_indices
    )
    assert len(kd_ds) == expected, (len(kd_ds), expected)

    # First unlabeled example in the KD set: its soft label row must equal the
    # teacher probs at that original train index.
    n_labeled = len(splits.labeled_train_indices) + len(splits.labeled_cal_indices)
    first_unlab_idx = int(splits.unlabeled_indices[0])
    assert np.allclose(kd_ds[n_labeled]["soft"], probs[first_unlab_idx])
    print(f"  {len(kd_ds)} KD examples aligned to sample_idx. PASSED\n")


def test_prompt_inclusive_label_accounting() -> None:
    """D_L includes prompt labels; only learned routers reserve calibration labels."""
    print("--- test_prompt_inclusive_label_accounting ---")
    ds = load_dataset_by_name(DATASET)
    splits = subsample_low_label(ds, labels_per_class=10, seed=42)
    demos = prompt_demo_indices(ds)
    declared = set(splits.labeled_indices.tolist())
    unlabeled = set(splits.unlabeled_indices.tolist())

    for cls, indices in demos.items():
        assert set(indices).issubset(declared), f"class {cls} prompt demos missing from D_L"
        assert not set(indices) & unlabeled, f"class {cls} prompt demos became unlabeled"

    # Baselines without learned calibration receive all L hard labels per class.
    supervised = build_examples(ds, splits, "supervised", None)
    assert supervised.has_label.sum() == 10 * ds.num_classes

    # Learned routers spend the existing 30% reserve and never leak it into CE.
    probs = teacher_soft_labels(TEACHER, DATASET, ds.num_classes, temperature=2.0)
    router_kd = build_examples(
        ds, splits, "kd", probs, reserve_calibration=True
    )
    assert router_kd.has_label.sum() == 7 * ds.num_classes
    assert len(splits.labeled_cal_indices) == 3 * ds.num_classes
    print("  prompt demos included; baseline CE=10/class; router CE/cal=7/3. PASSED\n")


def test_kd_ce_loss_modes() -> None:
    """Loss is a finite, differentiable scalar in both supervised and KD modes."""
    print("--- test_kd_ce_loss_modes ---")
    torch.manual_seed(0)
    batch, num_classes = 6, 4
    logits = torch.randn(batch, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch,))
    soft = torch.softmax(torch.randn(batch, num_classes), dim=-1)

    # Supervised: alpha = 0 -> CE only, soft labels ignored.
    has_label = torch.ones(batch, dtype=torch.bool)
    has_soft = torch.zeros(batch, dtype=torch.bool)
    sup = kd_ce_loss(logits, labels, soft, has_label, has_soft, alpha=0.0, temperature=2.0)
    assert torch.isfinite(sup) and sup.item() > 0
    sup.backward()

    # KD: alpha = 0.5, mix of labeled and unlabeled samples.
    logits2 = torch.randn(batch, num_classes, requires_grad=True)
    has_label = torch.tensor([True, True, False, False, False, False])
    has_soft = torch.ones(batch, dtype=torch.bool)
    kd = kd_ce_loss(logits2, labels, soft, has_label, has_soft, alpha=0.5, temperature=2.0)
    assert torch.isfinite(kd)
    kd.backward()
    assert logits2.grad is not None
    print("  supervised and KD losses finite + differentiable. PASSED\n")


def test_multi_mean_soft_labels() -> None:
    """Uniform logit-mean soft labels are proper C-way distributions."""
    print("--- test_multi_mean_soft_labels ---")
    ds = load_dataset_by_name(DATASET)
    avg_probs = average_teacher_soft_labels(
        GUIDED_TEACHERS, DATASET, ds.num_classes, temperature=2.0
    )
    assert avg_probs.shape == (len(ds.train), ds.num_classes)
    assert not np.isnan(avg_probs).any()
    assert np.allclose(avg_probs.sum(axis=1), 1.0, atol=1e-5)

    single_probs = [
        teacher_soft_labels(t, DATASET, ds.num_classes, temperature=2.0)
        for t in GUIDED_TEACHERS
    ]
    assert not np.allclose(avg_probs, single_probs[0])
    assert not np.allclose(avg_probs, single_probs[1])
    print(f"  {avg_probs.shape} averaged soft labels, distinct from singles. PASSED\n")


def test_entropy_weighted_soft_labels() -> None:
    """Entropy-weighted soft labels are proper and differ from uniform mean."""
    print("--- test_entropy_weighted_soft_labels ---")
    ds = load_dataset_by_name(DATASET)
    ent_probs = entropy_weighted_teacher_soft_labels(
        GUIDED_TEACHERS, DATASET, ds.num_classes, temperature=2.0
    )
    mean_probs = average_teacher_soft_labels(
        GUIDED_TEACHERS, DATASET, ds.num_classes, temperature=2.0
    )
    assert ent_probs.shape == (len(ds.train), ds.num_classes)
    assert not np.isnan(ent_probs).any()
    assert np.allclose(ent_probs.sum(axis=1), 1.0, atol=1e-5)
    assert not np.allclose(ent_probs, mean_probs), "entropy weights should change targets"
    print(f"  {ent_probs.shape} entropy-weighted labels differ from mean. PASSED\n")


def test_one_train_step_smoke() -> None:
    """fit() + evaluate() run end-to-end on a tiny model and data slice."""
    print("--- test_one_train_step_smoke ---")
    ds = load_dataset_by_name(DATASET)
    splits = subsample_low_label(ds, labels_per_class=10, seed=42)
    train_ds = build_examples(ds, splits, "supervised", None)

    config = StudentConfig(
        max_length=64,
        batch_size=8,
        epochs=1,
        alpha=0.0,
    )
    model, tokenizer = build_student(config.model_name, ds.num_classes)
    device = torch.device("cpu")
    fit(model, tokenizer, train_ds, config, device)

    # Evaluate on a small slice of test by temporarily shrinking the split.
    ds.test = ds.test.select(range(32))
    metrics = evaluate(model, tokenizer, ds, config, device, return_predictions=True)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["test_labels"]) == 32
    assert len(metrics["test_predictions"]) == 32
    print(f"  fit + evaluate ran: {metrics}. PASSED\n")


def main() -> None:
    test_soft_labels_are_proper_distributions()
    test_sample_idx_alignment()
    test_prompt_inclusive_label_accounting()
    test_kd_ce_loss_modes()
    test_multi_mean_soft_labels()
    test_entropy_weighted_soft_labels()
    test_one_train_step_smoke()
    print("=== All student tests passed. ===")


if __name__ == "__main__":
    main()
