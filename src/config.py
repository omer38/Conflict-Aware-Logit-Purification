"""Project-wide experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for dataset loading and low-label subsampling.

    Attributes:
        dataset_name: Which dataset to use ("amazon", "yahoo", or "ag_news").
        labels_per_class: Number of labeled examples per class (L).
        cal_ratio: Fraction of the labeled set reserved for router calibration.
        seed: Random seed for subsampling reproducibility.
    """

    dataset_name: str = "amazon"
    labels_per_class: int = 10
    cal_ratio: float = 0.3
    seed: int = 42


@dataclass
class StudentConfig:
    """Configuration for student model training (Stage 4).

    Attributes:
        model_name: HuggingFace id of the pre-trained student encoder.
        max_length: Maximum tokenized sequence length.
        lr: AdamW learning rate.
        batch_size: Training and evaluation batch size.
        epochs: Number of training epochs.
        temperature: KD softmax temperature (tau), shared by teacher soft labels
            and the student distribution.
        alpha: KD/CE mixing weight; alpha=0 is the supervised (CE-only) baseline.
    """

    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 256
    lr: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    temperature: float = 2.0
    alpha: float = 0.5


@dataclass
class ExperimentGrid:
    """Full grid of dataset / label-budget / seed combinations to sweep.
    """

    dataset_names: list[str] = field(
        default_factory=lambda: ["amazon", "yahoo", "ag_news"]
    )
    labels_per_class_values: list[int] = field(default_factory=lambda: [10, 50])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 2024])
    cal_ratio: float = 0.3

    def iter_configs(self):
        """Yield a ``DataConfig`` for every (dataset, L, seed) combination."""
        for name in self.dataset_names:
            for lpc in self.labels_per_class_values:
                for seed in self.seeds:
                    yield DataConfig(
                        dataset_name=name,
                        labels_per_class=lpc,
                        cal_ratio=self.cal_ratio,
                        seed=seed,
                    )
