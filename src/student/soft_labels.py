"""Load guided teacher soft labels from report CSVs for student distillation.

Guided extraction constrains generation to the class tokens (A, B, ...), so the
report CSV holds clean logprobs over all C classes with ``floor_fraction = 0``.
We read ``logit_A..logit_{C}`` keyed by ``sample_idx`` and turn them into a
proper probability distribution via ``softmax(logits / temperature)``.
"""

from __future__ import annotations

import csv
import string
from pathlib import Path

import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "reports"

GUIDED_TEACHERS = (
    "qwen3.5-27b-vllm-guided",
    "gemma4-26b-vllm-guided",
    "mistral-small-24b-vllm-guided",
)
MULTI_MEAN_TEACHER = "multi-mean-guided"
MULTI_ENTROPY_TEACHER = "multi-entropy-guided"
MULTI_PROBMEAN_TEACHER = "multi-probmean-guided"
MULTI_CALIBRATED_TEACHER = "multi-calibrated-guided"
MULTI_CASCADE_TEACHER = "multi-cascade-guided"
MULTI_UNIFORM_BUDGET_TEACHER = "multi-uniform-budget-guided"
MULTI_DEPTH_MATCHED_RANDOM_TEACHER = "multi-depth-matched-random-guided"
MULTI_MARGINAL_TEACHER = "multi-marginal-guided"
MULTI_ENTROPYEQ_TEACHER = "multi-entropyeq-guided"
QWEN_CALIBRATED_TEACHER = "qwen-calibrated-guided"

# Preferred terminology. The old identifiers remain accepted so historical
# commands and result files stay reproducible.
MULTI_PRIOR_SCALE_ALIGNED_TEACHER = "multi-prior-scale-aligned-guided"
MULTI_PRIOR_MATCHED_TEACHER = "multi-prior-matched-guided"
MULTI_SCALE_ALIGNED_TEACHER = "multi-scale-aligned-guided"
QWEN_PRIOR_MATCHED_TEACHER = "qwen-prior-matched-guided"


def _make_verbalizers(num_classes: int) -> list[str]:
    if num_classes > len(string.ascii_uppercase):
        raise ValueError(f"No single-letter verbalizers for {num_classes} classes")
    return list(string.ascii_uppercase[:num_classes])


def _report_path(teacher: str, dataset_name: str, report_dir: str | Path) -> Path:
    return Path(report_dir) / f"{teacher}_{dataset_name}.csv"


def load_teacher_logits(
    teacher: str,
    dataset_name: str,
    num_classes: int,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Return an ``(N, C)`` logit array aligned to ``dataset.train`` row order.

    Rows are placed at their ``sample_idx`` position; N is ``max(sample_idx) + 1``.
    Any position absent from the CSV stays ``NaN`` (none expected for a full
    guided extraction). Raises if the report file is missing.
    """
    path = _report_path(teacher, dataset_name, report_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No teacher report at {path}. Extract guided logits first, e.g. "
            f"uv run python scripts/extract_logits.py --teacher {teacher} "
            f"--dataset {dataset_name} --report"
        )

    logit_cols = [f"logit_{v}" for v in _make_verbalizers(num_classes)]
    rows: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in logit_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        for row in reader:
            rows[int(row["sample_idx"])] = [float(row[c]) for c in logit_cols]

    n_samples = max(rows) + 1
    logits = np.full((n_samples, num_classes), np.nan, dtype=np.float32)
    for idx, values in rows.items():
        logits[idx] = values
    return logits


def _logits_to_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax over the class dimension (last axis)."""
    scaled = logits / temperature
    scaled = scaled - np.nanmax(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def teacher_soft_labels(
    teacher: str,
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Return ``(N, C)`` temperature-scaled teacher probabilities."""
    logits = load_teacher_logits(teacher, dataset_name, num_classes, report_dir)
    return _logits_to_probs(logits, temperature)


def load_multi_teacher_logits(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Return ``(K, N, C)`` logits stacked for *teachers* (same row order)."""
    arrays = [
        load_teacher_logits(t, dataset_name, num_classes, report_dir) for t in teachers
    ]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Teacher logit shapes differ: {dict(zip(teachers, [a.shape for a in arrays]))}")
    stacked = np.stack(arrays, axis=0)
    if np.isnan(stacked).any():
        raise ValueError(
            "Multi-teacher logits contain NaN; ensure every teacher has a full guided extraction."
        )
    return stacked


def average_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Uniform logit-mean then softmax: proposal baseline #2.

    Computes ``z_bar = mean_i z_i`` then ``softmax(z_bar / temperature)``.
    """
    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    z_bar = stacked.mean(axis=0)
    return _logits_to_probs(z_bar, temperature)


def _probs_to_kd_target(p_bar: np.ndarray, temperature: float) -> np.ndarray:
    """Apply the KD temperature to a probability-space aggregate.

    Probability-space analogue of ``softmax(z_bar / tau)``: treats ``log p_bar``
    as the aggregate logits so all multi-teacher variants soften consistently.
    """
    return _logits_to_probs(np.log(np.clip(p_bar, 1e-12, None)), temperature)


def prob_mean_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Arithmetic probability mean: ``p_bar = mean_i softmax(z_i)``.

    Bounds each teacher's influence on the simplex, so an overconfident
    teacher (large logit magnitudes) cannot dominate the way it does in the
    logit mean. Label-free.
    """
    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    p_bar = _logits_to_probs(stacked, temperature=1.0).mean(axis=0)
    return _probs_to_kd_target(p_bar, temperature)


def uniform_class_prior(num_classes: int) -> np.ndarray:
    """Return an explicit uniform class prior of length ``num_classes``."""
    return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)


def skew_class_prior(
    num_classes: int,
    strength: float,
    dominant_class: int,
) -> np.ndarray:
    """Return the Table A.12 convex-mixture skew of the uniform class prior.

    ``prior = (1 - strength) * uniform_prior``, then ``prior[dominant_class]``
    receives an additional ``+= strength``. Predeclared and label-free: it does
    not inspect labels, predictions, or empirical class frequencies.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    if not 0.0 < strength < 1.0:
        raise ValueError(f"strength must satisfy 0 < strength < 1, got {strength}")
    if not 0 <= dominant_class < num_classes:
        raise ValueError(
            f"dominant_class must satisfy 0 <= dominant_class < {num_classes}, "
            f"got {dominant_class}"
        )
    prior = np.full(num_classes, (1.0 - strength) / num_classes, dtype=np.float64)
    prior[dominant_class] += strength
    return prior


def require_prior_condition_supported(
    mode: str,
    teacher: str,
    prior_condition: str,
) -> None:
    """Raise ``ValueError`` unless a non-"correct" prior condition is well-defined.

    Misspecified assumed priors (``mild``/``strong``) are only meaningful for
    the prior-and-scale-aligned guided teacher in KD mode; every other teacher
    method must keep the true, uniform working-pool prior.
    """
    if prior_condition != "correct" and not (
        mode == "kd" and teacher == MULTI_PRIOR_SCALE_ALIGNED_TEACHER
    ):
        raise ValueError(
            f"--prior-condition {prior_condition} is only defined for --mode kd "
            f"--teacher {MULTI_PRIOR_SCALE_ALIGNED_TEACHER}"
        )


def class_prior_assumption_metadata(
    condition: str,
    assumed_prior: np.ndarray,
    true_prior: np.ndarray,
    dominant_class: int,
    strength: float | None,
) -> dict:
    """Provenance dict recorded alongside a prior-scale-aligned student result.

    ``correct`` keeps the historical schema byte-for-byte (backward
    compatible with existing runs); ``mild``/``strong`` add full A.12-style
    misspecification provenance without touching the correct-condition shape.
    """
    if condition == "correct":
        return {
            "values": assumed_prior.tolist(),
            "source": "known_uniform_prior_from_gold-balanced_working_pool",
            "uses_individual_labels_during_matching": False,
        }
    return {
        "condition": condition,
        "values": assumed_prior.tolist(),
        "true_working_pool_prior": true_prior.tolist(),
        "strength": strength,
        "dominant_class": dominant_class,
        "source": "predeclared_A12_style_misspecification_on_gold_balanced_working_pool",
        "uses_individual_labels_during_matching": False,
    }


def teacher_target_diagnostics(soft_probs: np.ndarray, gold_labels: np.ndarray) -> dict:
    """Offline accuracy/macro-F1/NLL of teacher soft targets against gold labels.

    Diagnostic only: gold labels are used here for post-hoc evaluation, never
    to construct an assumed prior or to alter student training.
    """
    predictions = soft_probs.argmax(axis=1)
    num_classes = soft_probs.shape[1]
    f1_values = []
    for class_id in range(num_classes):
        true_positive = int(np.sum((predictions == class_id) & (gold_labels == class_id)))
        false_positive = int(np.sum((predictions == class_id) & (gold_labels != class_id)))
        false_negative = int(np.sum((predictions != class_id) & (gold_labels == class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    true_class_probs = soft_probs[np.arange(len(gold_labels)), gold_labels]
    return {
        "scope": "offline_full_balanced_working_pool",
        "uses_gold_labels_for_evaluation_only": True,
        "accuracy": float(np.mean(predictions == gold_labels)),
        "macro_f1": float(np.mean(f1_values)),
        "nll": float(-np.mean(np.log(np.clip(true_class_probs, 1e-12, None)))),
        "num_rows": int(len(gold_labels)),
    }


def teacher_target_diagnostics_full(
    soft_probs: np.ndarray, gold_labels: np.ndarray, true_prior: np.ndarray
) -> dict:
    """Extended offline teacher-target diagnostics for a non-uniform pool.

    Adds ECE, Brier score, the predicted class marginal, and its L1 error
    from ``true_prior`` on top of :func:`teacher_target_diagnostics`'s
    accuracy/macro-F1/NLL. Gold labels are used for evaluation only.
    """
    base = teacher_target_diagnostics(soft_probs, gold_labels)
    predictions = soft_probs.argmax(axis=1)
    confidence = soft_probs.max(axis=1)
    correct = predictions == gold_labels
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            confidence <= upper if np.isclose(upper, 1.0) else confidence < upper
        )
        if mask.any():
            bin_accuracy = float(correct[mask].mean())
            bin_confidence = float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)

    one_hot = np.eye(soft_probs.shape[1], dtype=np.float64)[gold_labels]
    brier = float(np.mean(np.sum((soft_probs - one_hot) ** 2, axis=1)))
    predicted_marginal = soft_probs.mean(axis=0)
    l1_error = float(np.abs(predicted_marginal - true_prior).sum())

    base.update(
        {
            "ece": ece,
            "brier": brier,
            "predicted_marginal": predicted_marginal.tolist(),
            "l1_error_from_true_prior": l1_error,
        }
    )
    return base


def _validate_class_prior(target_prior: np.ndarray, num_classes: int) -> np.ndarray:
    prior = np.asarray(target_prior, dtype=np.float64)
    if prior.shape != (num_classes,):
        raise ValueError(
            f"target_prior must have shape ({num_classes},), got {prior.shape}"
        )
    if not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ValueError("target_prior must contain finite, strictly positive values")
    if not np.isclose(prior.sum(), 1.0, atol=1e-8):
        raise ValueError(f"target_prior must sum to 1, got {prior.sum():.8f}")
    return prior


def _validate_sample_weights(
    sample_weights: np.ndarray | None, n_samples: int
) -> np.ndarray | None:
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.shape != (n_samples,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(
            "sample_weights must be non-negative with shape (N,) and positive sum"
        )
    return weights


def fit_prior_bias(
    stacked: np.ndarray,
    target_prior: np.ndarray,
    n_iter: int = 100,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit per-teacher class-bias vectors ``b_k`` on *stacked* logits only.

    Returns ``(K, C)`` biases such that, on the *fitting* batch,
    ``mean_x softmax(z_k(x) + b_k) ~= target_prior``. No individual labels
    are consumed; ``target_prior`` must be supplied or estimated externally.
    Pass a held-out (e.g. reference) ``stacked`` here and apply the returned
    bias to a different batch via :func:`apply_prior_bias` for out-of-sample
    (transductive-free) prior matching.
    """
    num_classes = stacked.shape[-1]
    prior = _validate_class_prior(target_prior, num_classes)
    weights = _validate_sample_weights(sample_weights, stacked.shape[1])
    bias = np.zeros((stacked.shape[0], num_classes), dtype=np.float64)
    for k in range(stacked.shape[0]):
        b = np.zeros(num_classes, dtype=np.float64)
        for _ in range(n_iter):
            probabilities = _logits_to_probs(stacked[k] + b, temperature=1.0)
            p_mean = np.average(probabilities, axis=0, weights=weights)
            b += np.log(prior) - np.log(np.clip(p_mean, 1e-12, None))
        bias[k] = b
    return bias


def apply_prior_bias(stacked: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Add fitted per-teacher class-bias vectors to (possibly different) logits."""
    bias = np.asarray(bias, dtype=np.float64)
    expected_shape = (stacked.shape[0], stacked.shape[-1])
    if bias.shape != expected_shape:
        raise ValueError(f"bias shape {bias.shape} != expected {expected_shape}")
    return stacked + bias[:, None, :]


def fit_confidence_scale(
    stacked: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit per-teacher confidence-scale temperatures on *stacked* logits only.

    Returns ``(K,)`` temperatures that equalise mean prediction entropy to
    the least-confident teacher's entropy on the *fitting* batch. Label-free.
    Pass a held-out (e.g. reference) ``stacked`` here and apply the returned
    scales to a different batch via :func:`apply_confidence_scale`.
    """
    weights = _validate_sample_weights(sample_weights, stacked.shape[1])

    def mean_entropy(logits: np.ndarray, t: float) -> float:
        p = _logits_to_probs(logits / t, temperature=1.0)
        per_sample = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)
        return float(np.average(per_sample, weights=weights))

    target = max(mean_entropy(stacked[k], 1.0) for k in range(stacked.shape[0]))
    scales = np.empty(stacked.shape[0], dtype=np.float64)
    for k in range(stacked.shape[0]):
        lo, hi = 0.25, 16.0
        for _ in range(50):
            t = 0.5 * (lo + hi)
            if mean_entropy(stacked[k], t) < target:
                lo = t
            else:
                hi = t
        scales[k] = 0.5 * (lo + hi)
    return scales


def apply_confidence_scale(stacked: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Divide (possibly different) logits by fitted per-teacher scales."""
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (stacked.shape[0],):
        raise ValueError(f"scales shape {scales.shape} != expected {(stacked.shape[0],)}")
    return stacked / scales[:, None, None]


# Historical private name kept as an alias for existing callers
# (scripts/report_scale_diagnostics.py) that import it directly.
_fit_confidence_scales = fit_confidence_scale


def probability_mean_kd_target(stacked: np.ndarray, temperature: float) -> np.ndarray:
    """Arithmetic probability mean of *stacked* logits, then the KD-temperature transform."""
    p_bar = _logits_to_probs(stacked, temperature=1.0).mean(axis=0)
    return _probs_to_kd_target(p_bar, temperature)


def _match_class_prior_logits(
    stacked: np.ndarray,
    target_prior: np.ndarray,
    n_iter: int = 100,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Add per-class logit biases to match a supplied pool-level class prior.

    Transductive (fits and applies on the same ``stacked`` logits); delegates
    to :func:`fit_prior_bias` / :func:`apply_prior_bias` so this historical
    entry point stays numerically identical to before the fit/apply split
    (see ``tests/test_prior_alignment.py``).
    """
    bias = fit_prior_bias(stacked, target_prior, n_iter=n_iter, sample_weights=sample_weights)
    return apply_prior_bias(stacked, bias)


def _align_confidence_scales_logits(
    stacked: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Align teachers to the same mean prediction entropy.

    Transductive (fits and applies on the same ``stacked`` logits); delegates
    to :func:`fit_confidence_scale` / :func:`apply_confidence_scale` so this
    historical entry point stays numerically identical to before the
    fit/apply split. This label-independent scale alignment softens teachers
    to the entropy of the least-confident teacher; it does not guarantee
    conventional probability calibration or confidence reliability.
    """
    scales = fit_confidence_scale(stacked, sample_weights)
    return apply_confidence_scale(stacked, scales)


def _marginal_calibrate_logits(stacked: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """Backward-compatible uniform-prior wrapper; prefer class-prior matching."""
    return _match_class_prior_logits(
        stacked, uniform_class_prior(stacked.shape[-1]), n_iter=n_iter
    )


def _entropy_equalize_logits(stacked: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper; prefer confidence-scale alignment."""
    return _align_confidence_scales_logits(stacked)


def prior_scale_aligned_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    target_prior: np.ndarray,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Class-prior matching plus confidence-scale alignment, then prob-mean.

    This uses no individual labels during correction, but it assumes that
    ``target_prior`` is known. It is not conventional reliability calibration.
    """
    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    aligned = _align_confidence_scales_logits(
        _match_class_prior_logits(stacked, target_prior)
    )
    p_bar = _logits_to_probs(aligned, temperature=1.0).mean(axis=0)
    return _probs_to_kd_target(p_bar, temperature)


def prior_matched_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    target_prior: np.ndarray,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Class-prior matching only, then arithmetic probability mean."""
    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    matched = _match_class_prior_logits(stacked, target_prior)
    p_bar = _logits_to_probs(matched, temperature=1.0).mean(axis=0)
    return _probs_to_kd_target(p_bar, temperature)


def scale_aligned_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Confidence-scale alignment only, then arithmetic probability mean."""
    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    aligned = _align_confidence_scales_logits(stacked)
    p_bar = _logits_to_probs(aligned, temperature=1.0).mean(axis=0)
    return _probs_to_kd_target(p_bar, temperature)


def prior_matched_single_teacher_soft_labels(
    teacher: str,
    dataset_name: str,
    num_classes: int,
    temperature: float,
    target_prior: np.ndarray,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Class-prior-match one teacher to an explicitly supplied target prior."""
    logits = load_teacher_logits(teacher, dataset_name, num_classes, report_dir)
    matched = _match_class_prior_logits(logits[None], target_prior)[0]
    return _logits_to_probs(matched, temperature)


# Historical API wrappers. They deliberately make their old implicit uniform
# assumption explicit when delegating to the preferred functions.
def calibrated_mean_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    return prior_scale_aligned_teacher_soft_labels(
        teachers,
        dataset_name,
        num_classes,
        temperature,
        uniform_class_prior(num_classes),
        report_dir,
    )


def marginal_only_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    return prior_matched_teacher_soft_labels(
        teachers,
        dataset_name,
        num_classes,
        temperature,
        uniform_class_prior(num_classes),
        report_dir,
    )


def entropy_only_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    return scale_aligned_teacher_soft_labels(
        teachers, dataset_name, num_classes, temperature, report_dir
    )


def calibrated_single_teacher_soft_labels(
    teacher: str,
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    return prior_matched_single_teacher_soft_labels(
        teacher,
        dataset_name,
        num_classes,
        temperature,
        uniform_class_prior(num_classes),
        report_dir,
    )


def _cascade_teacher_probabilities(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    report_dir: str | Path,
    target_prior: np.ndarray | None,
) -> np.ndarray:
    """Return the prior-matched, scale-aligned probabilities used by the cascade."""
    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    if target_prior is None:
        target_prior = uniform_class_prior(num_classes)
    aligned = _align_confidence_scales_logits(
        _match_class_prior_logits(stacked, target_prior)
    )
    return _logits_to_probs(aligned, temperature=1.0)


def _cascade_allocation_masks(
    probabilities: np.ndarray,
    target_queries: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the cascade's second/third-teacher masks and realised query rate."""
    ent1 = -(
        probabilities[0] * np.log(np.clip(probabilities[0], 1e-12, None))
    ).sum(axis=-1)
    votes1 = probabilities[0].argmax(axis=-1)
    votes2 = probabilities[1].argmax(axis=-1)

    def queries_at(h: float) -> tuple[np.ndarray, np.ndarray, float]:
        ask2 = ent1 > h
        ask3 = ask2 & (votes1 != votes2)
        return ask2, ask3, 1.0 + float(ask2.mean()) + float(ask3.mean())

    lo, hi = float(ent1.min()) - 1e-6, float(ent1.max()) + 1e-6
    best = min(
        (queries_at(lo), queries_at(hi)),
        key=lambda result: abs(result[2] - target_queries),
    )
    for _ in range(60):
        h = 0.5 * (lo + hi)
        candidate = queries_at(h)
        _, _, queries = candidate
        if abs(queries - target_queries) < abs(best[2] - target_queries):
            best = candidate
        if queries > target_queries:
            lo = h
        else:
            hi = h
    return best


def cascade_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    target_queries: float = 1.4,
    report_dir: str | Path = REPORT_DIR,
    target_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Entropy cascade over prior-matched, confidence-scale-aligned teachers.

    No individual labels are used during correction. ``target_prior`` supplies
    the assumed pool prior; omitted values retain the historical uniform-prior
    behaviour for backward compatibility.

    The escalation threshold is bisected (on entropy of ``teachers[0]``) so the
    realised mean queries/sample matches ``target_queries``. Returns the
    ``(N, C)`` KD targets and the realised queries/sample (for cost reporting).
    """
    if len(teachers) != 3:
        raise ValueError("cascade_teacher_soft_labels requires exactly three teachers")
    if not np.isfinite(target_queries) or not 1.0 <= target_queries <= 3.0:
        raise ValueError("target_queries must be between 1 and 3")

    p = _cascade_teacher_probabilities(
        teachers, dataset_name, num_classes, report_dir, target_prior
    )
    ask2, ask3, realised_queries = _cascade_allocation_masks(p, target_queries)

    p_bar = p[0].copy()
    two_only = ask2 & ~ask3
    p_bar[two_only] = 0.5 * (p[0][two_only] + p[1][two_only])
    p_bar[ask3] = (p[0][ask3] + p[1][ask3] + p[2][ask3]) / 3.0

    return _probs_to_kd_target(p_bar, temperature), realised_queries


def depth_matched_random_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    target_queries: float = 1.4,
    allocation_seed: int = 42,
    report_dir: str | Path = REPORT_DIR,
    target_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Randomly assign the exact query-depth counts produced by the cascade."""
    if len(teachers) != 3:
        raise ValueError(
            "depth_matched_random_teacher_soft_labels requires exactly three teachers"
        )
    if not np.isfinite(target_queries) or not 1.0 <= target_queries <= 3.0:
        raise ValueError("target_queries must be between 1 and 3")

    p = _cascade_teacher_probabilities(
        teachers, dataset_name, num_classes, report_dir, target_prior
    )
    ask2, ask3, cascade_realised = _cascade_allocation_masks(p, target_queries)
    n_samples = int(p.shape[1])
    n_depth_1 = int((~ask2).sum())
    n_depth_2 = int((ask2 & ~ask3).sum())
    n_depth_3 = int(ask3.sum())
    if n_depth_1 + n_depth_2 + n_depth_3 != n_samples:
        raise RuntimeError("cascade depth counts do not sum to the sample count")

    count_realised = (
        n_depth_1 + 2 * n_depth_2 + 3 * n_depth_3
    ) / n_samples
    if not np.isclose(count_realised, cascade_realised, rtol=0.0, atol=1e-12):
        raise RuntimeError("depth-count query rate does not match the cascade")
    realised = cascade_realised

    rng = np.random.default_rng(allocation_seed)
    permutation = rng.permutation(n_samples)
    depth_3_indices = permutation[:n_depth_3]
    depth_2_indices = permutation[n_depth_3:n_depth_3 + n_depth_2]

    p_bar = p[0].copy()
    p_bar[depth_2_indices] = (
        p[0][depth_2_indices] + p[1][depth_2_indices]
    ) / 2.0
    p_bar[depth_3_indices] = (
        p[0][depth_3_indices]
        + p[1][depth_3_indices]
        + p[2][depth_3_indices]
    ) / 3.0

    metadata = {
        "policy": "depth_matched_random",
        "source_policy": "entropy_cascade",
        "target_queries_per_sample": float(target_queries),
        "realised_queries_per_sample": float(realised),
        "allocation_seed": int(allocation_seed),
        "n_samples": n_samples,
        "depth_1_count": n_depth_1,
        "depth_2_count": n_depth_2,
        "depth_3_count": n_depth_3,
        "depth_1_fraction": float(n_depth_1 / n_samples),
        "depth_2_fraction": float(n_depth_2 / n_samples),
        "depth_3_fraction": float(n_depth_3 / n_samples),
        "primary_teacher": teachers[0],
        "teachers": list(teachers),
    }
    return _probs_to_kd_target(p_bar, temperature), metadata


def uniform_budget_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    target_queries: float = 1.4,
    allocation_seed: int = 42,
    report_dir: str | Path = REPORT_DIR,
    target_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Matched-budget uniform allocation over prior-matched, scale-aligned teachers.

    Queries ``teachers[0]`` on every row. On a deterministic random subset of
    ``m = round(N * (target_queries - 1) / (K - 1))`` rows, replaces the primary
    target with the arithmetic mean of all ``K`` teacher probability vectors.
    Unselected rows stay primary-only. Allocation does not use gold labels.

    Returns ``(N, C)`` KD targets and allocation metadata including the realised
    queries/sample computed from the integer subset size.
    """
    k = len(teachers)
    if k < 2:
        raise ValueError(f"uniform budget requires at least 2 teachers, got {k}")
    if not (1.0 <= target_queries <= float(k)):
        raise ValueError(
            f"target_queries must satisfy 1 <= q <= K={k}, got {target_queries}"
        )

    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    if target_prior is None:
        target_prior = uniform_class_prior(num_classes)
    aligned = _align_confidence_scales_logits(
        _match_class_prior_logits(stacked, target_prior)
    )
    p = _logits_to_probs(aligned, temperature=1.0)
    n_samples = int(p.shape[1])
    selected_count = int(round(n_samples * (target_queries - 1.0) / (k - 1)))
    if not (0 <= selected_count <= n_samples):
        raise ValueError(
            f"selected_count={selected_count} out of range for N={n_samples}"
        )

    rng = np.random.default_rng(allocation_seed)
    selected = (
        rng.choice(n_samples, size=selected_count, replace=False)
        if selected_count > 0
        else np.array([], dtype=np.int64)
    )
    selected = np.sort(selected.astype(np.int64))

    p_bar = p[0].copy()
    if selected_count > 0:
        p_bar[selected] = p[:, selected, :].mean(axis=0)

    realised = 1.0 + (k - 1) * selected_count / n_samples
    metadata = {
        "policy": "uniform_budget",
        "target_queries_per_sample": float(target_queries),
        "realised_queries_per_sample": float(realised),
        "allocation_seed": int(allocation_seed),
        "selected_count": int(selected_count),
        "selected_fraction": float(selected_count / n_samples),
        "primary_teacher": teachers[0],
        "teachers": list(teachers),
    }
    return _probs_to_kd_target(p_bar, temperature), metadata


def _entropy_weights(probs: np.ndarray) -> np.ndarray:
    """Per-teacher sample weights ``w ∝ exp(-H(p))``, normalised over teachers.

    *probs* has shape ``(K, N, C)``; returns ``(K, N)`` weights summing to 1
    over axis 0 for each sample.
    """
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)
    weights = np.exp(-entropy)
    weights /= weights.sum(axis=0, keepdims=True)
    return weights.astype(np.float32)


def entropy_weighted_teacher_soft_labels(
    teachers: tuple[str, ...] | list[str],
    dataset_name: str,
    num_classes: int,
    temperature: float,
    report_dir: str | Path = REPORT_DIR,
) -> np.ndarray:
    """Entropy-weighted logit combination then softmax: proposal baseline #3 (DE-MKD style).

    Lower-entropy (more confident) teachers receive higher weight per sample:
    ``w_i ∝ exp(-H(p_i))``, ``z_bar = sum_i w_i z_i``, then ``softmax(z_bar / τ)``.
    """
    stacked = load_multi_teacher_logits(teachers, dataset_name, num_classes, report_dir)
    probs = _logits_to_probs(stacked, temperature)
    weights = _entropy_weights(probs)
    z_bar = (weights[:, :, None] * stacked).sum(axis=0)
    return _logits_to_probs(z_bar, temperature)
