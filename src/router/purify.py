"""Apply a trained router to produce purified per-sample teacher soft labels."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .train_router import mixture_probs, router_weights


def purified_soft_labels(
    router: torch.nn.Module,
    phi_all: np.ndarray,
    logits_all: np.ndarray,
    log_prior_all: np.ndarray,
    temperature: float,
    mixture_space: str = "logits",
    disagree_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply *router* to every sample's features, producing ``(N, C)`` purified probs.

    Drop-in compatible with ``build_examples(dataset, splits, "kd", soft_probs)``.
    ``mixture_space='logits'`` is the v2 default.  If ``disagree_mask`` is set,
    agreement samples keep the prior mixture (sparse / disagreement-gated routing).
    """
    router.eval()
    with torch.no_grad():
        phi_t = torch.as_tensor(phi_all, dtype=torch.float32)
        logits_t = torch.as_tensor(logits_all, dtype=torch.float32)
        log_prior_t = torch.as_tensor(log_prior_all, dtype=torch.float32)
        if disagree_mask is None and mixture_space == "logits":
            w = router_weights(router, phi_t, log_prior_t)
            z_bar = (w.unsqueeze(-1) * logits_t).sum(dim=1)
            probs = F.softmax(z_bar / temperature, dim=-1)
            return probs.numpy().astype(np.float32)

        p, w = mixture_probs(
            router, phi_t, logits_t, log_prior_t, temperature, mixture_space
        )
        if disagree_mask is not None:
            prior_w = F.softmax(log_prior_t, dim=-1)
            mask = torch.as_tensor(disagree_mask, dtype=torch.bool)
            w = torch.where(mask.unsqueeze(-1), w, prior_w)
            if mixture_space == "logits":
                z_bar = (w.unsqueeze(-1) * logits_t).sum(dim=1)
                p = F.softmax(z_bar / temperature, dim=-1)
            else:
                teacher_probs = F.softmax(logits_t / temperature, dim=-1)
                p = (w.unsqueeze(-1) * teacher_probs).sum(dim=1)
        return p.numpy().astype(np.float32)
