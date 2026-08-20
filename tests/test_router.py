"""Tests for the router package: oracle upper bound (Phase C) and learned
conflict-aware router (Phase D / v2 shrinkage-to-prior).

Run:
    uv run python tests/test_router.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset

from src.data.loading import TextClassificationDataset
from src.router.model import DEFAULT_HIDDEN_DIM, LinearRouter, MLPRouter
from src.router.oracle import oracle_teacher_soft_labels
from src.router.purify import purified_soft_labels
from src.router.train_router import (
    entropy_prior_log_weights,
    router_loss,
    router_weights,
    select_router_or_fallback,
    train_router,
)


def _write_toy_report(report_dir: Path, teacher: str, dataset_name: str, logits: np.ndarray) -> None:
    path = report_dir / f"{teacher}_{dataset_name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "logit_A", "logit_B"])
        writer.writeheader()
        for idx, row in enumerate(logits):
            writer.writerow({"sample_idx": idx, "logit_A": float(row[0]), "logit_B": float(row[1])})


def test_oracle_upper_bound() -> None:
    """Teacher 0 always right, teacher 1 always wrong: oracle purification recovers truth."""
    print("--- test_oracle_upper_bound ---")
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    n = len(labels)

    right_logits = np.full((n, 2), -5.0)
    right_logits[np.arange(n), labels] = 5.0

    wrong_logits = np.full((n, 2), -5.0)
    wrong_logits[np.arange(n), 1 - labels] = 5.0

    with tempfile.TemporaryDirectory() as tmp:
        report_dir = Path(tmp)
        _write_toy_report(report_dir, "teacher_right", "toy", right_logits)
        _write_toy_report(report_dir, "teacher_wrong", "toy", wrong_logits)

        dataset = TextClassificationDataset(
            name="toy",
            train=Dataset.from_dict({"label": labels.tolist()}),
            test=Dataset.from_dict({"label": []}),
            text_column="text",
            label_column="label",
            num_classes=2,
            label_names=["A", "B"],
        )

        purified = oracle_teacher_soft_labels(
            ("teacher_right", "teacher_wrong"),
            dataset,
            num_classes=2,
            temperature=2.0,
            report_dir=report_dir,
        )

    assert purified.shape == (n, 2)
    assert np.allclose(purified.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(purified.argmax(axis=1), labels), "oracle should recover truth on every sample"
    print("  oracle purified argmax matches truth on every sample. PASSED\n")


def test_router_forward_shapes() -> None:
    """Linear/MLP router corrections have shape (batch, K); combined weights sum to 1."""
    print("--- test_router_forward_shapes ---")
    batch, feature_dim, num_teachers = 5, 4, 3
    phi = torch.randn(batch, feature_dim)
    log_prior = torch.randn(batch, num_teachers)

    linear_router = LinearRouter(feature_dim, num_teachers)
    correction = linear_router(phi)
    assert correction.shape == (batch, num_teachers)

    w = router_weights(linear_router, phi, log_prior)
    assert w.shape == (batch, num_teachers)
    assert torch.allclose(w.sum(dim=-1), torch.ones(batch), atol=1e-5)

    mlp_router = MLPRouter(feature_dim, DEFAULT_HIDDEN_DIM, num_teachers, dropout=0.3)
    mlp_router.eval()
    w_mlp = router_weights(mlp_router, phi, log_prior)
    assert w_mlp.shape == (batch, num_teachers)
    assert torch.allclose(w_mlp.sum(dim=-1), torch.ones(batch), atol=1e-5)
    print("  linear + MLP router outputs are valid (batch, K) distributions. PASSED\n")


def test_zero_init_reproduces_entropy_prior() -> None:
    """Untrained router with zero correction equals entropy prior weights."""
    print("--- test_zero_init_reproduces_entropy_prior ---")
    rng = np.random.default_rng(0)
    n, num_teachers, num_classes = 6, 3, 4
    logits = rng.normal(size=(n, num_teachers, num_classes)).astype(np.float32)
    phi = rng.normal(size=(n, 5)).astype(np.float32)
    log_prior = entropy_prior_log_weights(logits)

    router = LinearRouter(phi.shape[1], num_teachers)
    phi_t = torch.as_tensor(phi, dtype=torch.float32)
    log_prior_t = torch.as_tensor(log_prior, dtype=torch.float32)

    with torch.no_grad():
        w = router_weights(router, phi_t, log_prior_t).numpy()

    expected = np.exp(log_prior - log_prior.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(w, expected, atol=1e-5)
    print("  zero-init router matches entropy prior. PASSED\n")


def test_router_training_reduces_loss() -> None:
    """train_router lowers the Eq. 9 loss vs. an untrained router, for both types."""
    print("--- test_router_training_reduces_loss ---")
    rng = np.random.default_rng(0)
    n, feature_dim, num_teachers, num_classes = 8, 4, 2, 3
    phi = rng.normal(size=(n, feature_dim)).astype(np.float32)
    logits = rng.normal(size=(n, num_teachers, num_classes)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=n).astype(np.int64)
    log_prior = entropy_prior_log_weights(logits)

    phi_t = torch.as_tensor(phi, dtype=torch.float32)
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    log_prior_t = torch.as_tensor(log_prior, dtype=torch.float32)

    for router_type in ["linear", "mlp"]:
        initial_router = train_router(
            phi, logits, labels, log_prior, router_type, temperature=1.0, epochs=0, seed=0
        )
        with torch.no_grad():
            initial_loss = router_loss(
                initial_router, phi_t, logits_t, labels_t, log_prior_t, temperature=1.0
            ).item()

        trained_router = train_router(
            phi, logits, labels, log_prior, router_type, temperature=1.0, epochs=300, lr=1e-2, seed=0
        )
        with torch.no_grad():
            final_loss = router_loss(
                trained_router, phi_t, logits_t, labels_t, log_prior_t, temperature=1.0
            ).item()

        assert final_loss < initial_loss, (router_type, initial_loss, final_loss)
        print(f"  {router_type}: initial={initial_loss:.4f} -> trained={final_loss:.4f}")
    print("  PASSED\n")


def test_purify_matches_manual_mixture() -> None:
    """purified_soft_labels matches a hand-computed weighted mixture on a toy case."""
    print("--- test_purify_matches_manual_mixture ---")
    feature_dim, num_teachers = 3, 2
    router = LinearRouter(feature_dim, num_teachers)
    with torch.no_grad():
        router.linear.weight.fill_(0.0)
        router.linear.bias.copy_(torch.tensor([np.log(2.0), 0.0]))

    phi_all = np.zeros((2, feature_dim), dtype=np.float32)
    logits_all = np.array(
        [
            [[1.0, -1.0], [0.0, 0.0]],
            [[2.0, 0.0], [-1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    log_prior_all = np.zeros((2, num_teachers), dtype=np.float32)

    temperature = 1.0
    purified = purified_soft_labels(router, phi_all, logits_all, log_prior_all, temperature)

    w = np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    manual_z = (w[None, :, None] * logits_all).sum(axis=1)
    manual_probs = np.exp(manual_z - manual_z.max(axis=-1, keepdims=True))
    manual_probs /= manual_probs.sum(axis=-1, keepdims=True)

    assert np.allclose(purified, manual_probs, atol=1e-5), (purified, manual_probs)
    print("  purified output matches hand-computed mixture. PASSED\n")


def test_select_router_or_fallback() -> None:
    """Safety gate uses router only when LOOCV beats both heuristics."""
    print("--- test_select_router_or_fallback ---")
    logits = np.array(
        [
            [[2.0, -2.0], [-2.0, 2.0]],
            [[2.0, -2.0], [-2.0, 2.0]],
            [[-2.0, 2.0], [2.0, -2.0]],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1], dtype=np.int64)

    gate_use = select_router_or_fallback(1.0, logits, labels, temperature=1.0)
    assert gate_use["use_router"] is True
    assert gate_use["fallback_to"] is None

    gate_fallback = select_router_or_fallback(0.0, logits, labels, temperature=1.0)
    assert gate_fallback["use_router"] is False
    assert gate_fallback["fallback_to"] in ("mean", "entropy")
    print("  safety gate selects router or fallback correctly. PASSED\n")


def main() -> None:
    test_oracle_upper_bound()
    test_router_forward_shapes()
    test_zero_init_reproduces_entropy_prior()
    test_router_training_reduces_loss()
    test_purify_matches_manual_mixture()
    test_select_router_or_fallback()
    print("=== All router tests passed. ===")


if __name__ == "__main__":
    main()
