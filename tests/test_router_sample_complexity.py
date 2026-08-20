"""Tests for CAR router training options and sample-complexity helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router.model import LinearRouter
from src.router.sample_complexity import (
    majority_pseudo_labels,
    nstar_from_curve,
    softmax,
    stratified_indices,
)
from src.router.train_router import (
    mixture_probs,
    router_loss,
    router_weights,
    select_router_or_fallback_nll,
    train_router,
    uniform_log_prior,
)


def test_uniform_prior_zero_init() -> None:
    print("--- test_uniform_prior_zero_init ---")
    n, k, d = 6, 3, 4
    phi = torch.zeros(n, d)
    log_prior = torch.as_tensor(uniform_log_prior(n, k))
    router = LinearRouter(d, k)
    w = router_weights(router, phi, log_prior)
    assert torch.allclose(w, torch.full((n, k), 1.0 / k), atol=1e-5)
    print("  PASSED\n")


def test_prob_mixture_rows_sum_to_one() -> None:
    print("--- test_prob_mixture_rows_sum_to_one ---")
    rng = np.random.default_rng(0)
    n, k, c, d = 8, 3, 4, 5
    phi = rng.normal(size=(n, d)).astype(np.float32)
    logits = rng.normal(size=(n, k, c)).astype(np.float32)
    labels = rng.integers(0, c, size=n).astype(np.int64)
    log_prior = uniform_log_prior(n, k)
    router = train_router(
        phi, logits, labels, log_prior, "linear", temperature=1.0,
        epochs=20, mixture_space="probs", entropy_coef=0.1, entropy_floor=0.5, seed=0,
    )
    phi_t = torch.as_tensor(phi)
    logits_t = torch.as_tensor(logits)
    log_prior_t = torch.as_tensor(log_prior)
    with torch.no_grad():
        p, w = mixture_probs(router, phi_t, logits_t, log_prior_t, 1.0, "probs")
        loss = router_loss(
            router, phi_t, logits_t, torch.as_tensor(labels), log_prior_t, 1.0,
            mixture_space="probs", entropy_coef=0.1, entropy_floor=0.5,
        )
    assert p.shape == (n, c)
    assert torch.allclose(p.sum(-1), torch.ones(n), atol=1e-4)
    assert torch.allclose(w.sum(-1), torch.ones(n), atol=1e-4)
    assert torch.isfinite(loss)
    print("  PASSED\n")


def test_nll_gate() -> None:
    print("--- test_nll_gate ---")
    assert select_router_or_fallback_nll(0.4, 0.5)["use_router"] is True
    assert select_router_or_fallback_nll(0.5, 0.4)["use_router"] is False
    print("  PASSED\n")


def test_stratified_and_pseudo() -> None:
    print("--- test_stratified_and_pseudo ---")
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    rng = np.random.default_rng(1)
    idx = stratified_indices(y, 2, rng, num_classes=3)
    assert len(idx) == 6
    assert set(np.bincount(y[idx], minlength=3)) == {2}

    # Two teachers vote class 0, one votes 1 on the first two rows.
    probs = np.zeros((3, 4, 2), dtype=np.float32)
    probs[0, :, 0] = 0.9
    probs[1, :, 0] = 0.9
    probs[2, :, 0] = 0.9
    probs[0, :2, 0] = 0.9
    probs[1, :2, 0] = 0.9
    probs[2, :2, 1] = 0.9
    probs[0, :2, 1] = 0.1
    probs[1, :2, 1] = 0.1
    probs[2, :2, 0] = 0.1
    pidx, py = majority_pseudo_labels(softmax(np.log(np.clip(probs, 1e-6, None)), axis=-1))
    assert len(pidx) == 2
    assert set(py.tolist()) == {0}
    print("  PASSED\n")


def test_nstar_from_curve() -> None:
    print("--- test_nstar_from_curve ---")
    rows = [
        {
            "n_cal": 12,
            "nll_gap_p10": 0.1,
            "entropy_ok_rate": 1.0,
            "oracle_gap_closed_p10": 0.0,
            "gate_pass_rate": 0.0,
        },
        {
            "n_cal": 80,
            "nll_gap_p10": -0.02,
            "entropy_ok_rate": 0.9,
            "oracle_gap_closed_p10": 0.55,
            "gate_pass_rate": 0.85,
        },
    ]
    assert nstar_from_curve(rows) == 80
    rows[1]["oracle_gap_closed_p10"] = 0.1
    assert nstar_from_curve(rows) is None
    from src.router.sample_complexity import nstar_beats_mean

    rows[1]["oracle_gap_closed_p10"] = 0.1
    rows[1]["nll_gap_p10"] = -0.02
    rows[1]["entropy_ok_rate"] = 0.9
    rows[1]["gate_pass_rate"] = 0.85
    assert nstar_beats_mean(rows) == 80
    print("  PASSED\n")


def main() -> None:
    test_uniform_prior_zero_init()
    test_prob_mixture_rows_sum_to_one()
    test_nll_gate()
    test_stratified_and_pseudo()
    test_nstar_from_curve()
    print("=== All sample-complexity router tests passed. ===")


if __name__ == "__main__":
    main()
