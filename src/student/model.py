"""Student model construction (ModernBERT + classification head)."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "answerdotai/ModernBERT-base"


def build_student(model_name: str, num_labels: int):
    """Load a pre-trained encoder with a fresh classification head.

    Returns (model, tokenizer). The classifier head is randomly initialised,
    which transformers reports as a warning; this is expected.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer


def select_device() -> torch.device:
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
