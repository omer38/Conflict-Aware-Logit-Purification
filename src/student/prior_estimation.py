"""Class-prior estimators for the Amazon-3 source-prior experiment.

Implements constrained soft black-box shift estimation (soft-BBSE) and
bias-corrected temperature scaling followed by maximum-likelihood label-shift
estimation (BCTS+MLLS). Both use the fixed committee predictor ``g(x)`` and
the same declared label budget; reference-batch labels are never accepted.

BBSE solves a confusion-moment system. BCTS+MLLS first calibrates ``g(x)`` on
the deliberately class-balanced declared labels, then estimates the prior of
an unlabelled reference batch by EM under label shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

_PRIOR_FLOOR = 1e-6
_PROB_FLOOR = 1e-12


def committee_probs(stacked_logits: np.ndarray) -> np.ndarray:
    """Fixed committee predictor ``g(x) = mean_k softmax(z_k(x))`` (temperature 1).

    ``stacked_logits`` has shape ``(K, N, C)``. No prior matching, no
    confidence-scale fitting, and no labels are used here; keeping this
    predictor fixed avoids making the prior estimator depend on the
    correction it is evaluating.
    """
    z = stacked_logits - stacked_logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs.mean(axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass
class BiasCorrectedCalibration:
    """Fitted bias-corrected temperature scaling parameters."""

    temperature: float
    bias: np.ndarray
    success: bool
    message: str
    n_iterations: int
    labelled_nll_before: float
    labelled_nll_after: float

    def to_json(self) -> dict:
        return {
            "temperature": float(self.temperature),
            "bias": self.bias.tolist(),
            "solver_success": bool(self.success),
            "solver_message": self.message,
            "solver_iterations": int(self.n_iterations),
            "labelled_nll_before": float(self.labelled_nll_before),
            "labelled_nll_after": float(self.labelled_nll_after),
        }


def apply_bias_corrected_calibration(
    predicted_probs: np.ndarray,
    calibration: BiasCorrectedCalibration,
) -> np.ndarray:
    """Apply ``softmax(log(p) / T + b)`` to classifier probabilities."""
    probs = np.asarray(predicted_probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"predicted_probs must be 2-D, got shape {probs.shape}")
    if calibration.temperature <= 0 or not np.isfinite(calibration.temperature):
        raise ValueError("calibration temperature must be finite and positive")
    if calibration.bias.shape != (probs.shape[1],):
        raise ValueError(
            f"calibration bias must have shape ({probs.shape[1]},), got {calibration.bias.shape}"
        )
    logits = np.log(np.clip(probs, _PROB_FLOOR, 1.0))
    return _softmax(logits / calibration.temperature + calibration.bias)


def fit_bias_corrected_calibration(
    predicted_probs: np.ndarray,
    declared_labels: np.ndarray,
) -> BiasCorrectedCalibration:
    """Fit bias-corrected temperature scaling by labelled NLL.

    The parameter vector contains log-temperature and ``C-1`` free biases;
    the final bias is the negative sum of the free values, fixing the
    softmax shift non-identifiability. The fit is deterministic.
    """
    probs = np.asarray(predicted_probs, dtype=np.float64)
    labels = np.asarray(declared_labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"predicted_probs must be 2-D, got shape {probs.shape}")
    if labels.shape != (len(probs),):
        raise ValueError(f"declared_labels must have shape ({len(probs)},), got {labels.shape}")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("predicted_probs must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("predicted_probs rows must sum to 1")
    num_classes = probs.shape[1]
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("declared_labels contain an out-of-range class")
    if any(not np.any(labels == c) for c in range(num_classes)):
        raise ValueError("declared_labels must contain every class")

    logits = np.log(np.clip(probs, _PROB_FLOOR, 1.0))

    def unpack(params: np.ndarray) -> tuple[float, np.ndarray]:
        temperature = float(np.exp(params[0]))
        free_bias = params[1:]
        bias = np.concatenate([free_bias, [-float(free_bias.sum())]])
        return temperature, bias

    def nll(params: np.ndarray) -> float:
        temperature, bias = unpack(params)
        calibrated = _softmax(logits / temperature + bias)
        return float(
            -np.log(np.clip(calibrated[np.arange(len(labels)), labels], _PROB_FLOOR, 1.0)).mean()
        )

    initial = np.zeros(num_classes, dtype=np.float64)
    before = nll(initial)
    result = minimize(
        nll,
        initial,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    temperature, bias = unpack(result.x)
    after = nll(result.x)
    return BiasCorrectedCalibration(
        temperature=temperature,
        bias=bias,
        success=bool(result.success and np.isfinite(after) and after <= before + 1e-10),
        message=str(result.message),
        n_iterations=int(getattr(result, "nit", 0)),
        labelled_nll_before=before,
        labelled_nll_after=after,
    )


@dataclass
class MLLSPriorEstimate:
    """Maximum-likelihood label-shift estimate and convergence diagnostics."""

    estimated_prior: np.ndarray
    source_prior: np.ndarray
    success: bool
    message: str
    n_iterations: int
    max_update: float
    mean_log_likelihood: float

    def to_json(self) -> dict:
        return {
            "estimated_prior": self.estimated_prior.tolist(),
            "source_prior": self.source_prior.tolist(),
            "solver_success": bool(self.success),
            "solver_message": self.message,
            "solver_iterations": int(self.n_iterations),
            "max_update": float(self.max_update),
            "mean_log_likelihood": float(self.mean_log_likelihood),
        }


def estimate_prior_mlls(
    calibrated_reference_probs: np.ndarray,
    source_prior: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> MLLSPriorEstimate:
    """Estimate an unlabelled reference prior by Saerens-style EM."""
    probs = np.asarray(calibrated_reference_probs, dtype=np.float64)
    source = np.asarray(source_prior, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"calibrated_reference_probs must be 2-D, got shape {probs.shape}")
    num_classes = probs.shape[1]
    if source.shape != (num_classes,):
        raise ValueError(f"source_prior must have shape ({num_classes},), got {source.shape}")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("calibrated_reference_probs must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("calibrated_reference_probs rows must sum to 1")
    if not np.isfinite(source).all() or np.any(source <= 0) or not np.isclose(source.sum(), 1.0):
        raise ValueError("source_prior must be finite, strictly positive, and sum to 1")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")

    estimate = source.copy()
    max_update = float("inf")
    for iteration in range(1, max_iterations + 1):
        responsibilities = probs * (estimate / source)[None, :]
        responsibilities /= np.clip(
            responsibilities.sum(axis=1, keepdims=True), _PROB_FLOOR, None
        )
        updated = responsibilities.mean(axis=0)
        updated = np.clip(updated, _PRIOR_FLOOR, None)
        updated /= updated.sum()
        max_update = float(np.max(np.abs(updated - estimate)))
        estimate = updated
        if max_update < tolerance:
            break

    mixture = (probs * (estimate / source)[None, :]).sum(axis=1)
    mean_log_likelihood = float(np.log(np.clip(mixture, _PROB_FLOOR, None)).mean())
    success = max_update < tolerance
    message = "converged" if success else "maximum iterations reached"
    return MLLSPriorEstimate(
        estimated_prior=estimate,
        source_prior=source,
        success=success,
        message=message,
        n_iterations=iteration,
        max_update=max_update,
        mean_log_likelihood=mean_log_likelihood,
    )


def confusion_moment_matrix(
    predicted_probs: np.ndarray,
    declared_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Soft confusion-moment matrix ``M[i, j] = mean_{x: y=j} g_i(x)``.

    Columns correspond to true classes ``j`` (from *declared_labels* only,
    never reference-batch labels); rows correspond to predicted-probability
    components ``i``. Each column sums to 1 because ``predicted_probs`` rows
    are proper probability distributions.
    """
    if predicted_probs.shape[0] != declared_labels.shape[0]:
        raise ValueError("predicted_probs and declared_labels must have the same length")
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for j in range(num_classes):
        mask = declared_labels == j
        if not np.any(mask):
            raise ValueError(f"no declared-label rows for class {j}; cannot estimate column {j}")
        matrix[:, j] = predicted_probs[mask].mean(axis=0)
    return matrix


def reference_marginal(predicted_probs_reference: np.ndarray) -> np.ndarray:
    """Predicted marginal ``mu_i`` over an unlabelled reference batch.

    Accepts only predicted probabilities; there is no parameter through
    which reference-batch labels could be supplied.
    """
    return predicted_probs_reference.mean(axis=0)


@dataclass
class PriorEstimate:
    """Full diagnostics for one constrained soft-BBSE solve."""

    estimated_prior: np.ndarray
    confusion_matrix: np.ndarray
    reference_marginal: np.ndarray
    success: bool
    message: str
    residual: float
    rank: int
    singular_values: np.ndarray
    condition_number: float

    def to_json(self) -> dict:
        return {
            "estimated_prior": self.estimated_prior.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "reference_marginal": self.reference_marginal.tolist(),
            "solver_success": bool(self.success),
            "solver_message": self.message,
            "residual": float(self.residual),
            "matrix_rank": int(self.rank),
            "singular_values": self.singular_values.tolist(),
            "condition_number": float(self.condition_number),
        }


def estimate_prior_bbse(matrix: np.ndarray, mu: np.ndarray) -> PriorEstimate:
    """Solve ``min_pi ||M @ pi - mu||_2^2`` s.t. ``pi_c >= 1e-6``, ``sum(pi) = 1``.

    Uses SciPy's SLSQP with a fixed, non-random initial point, so the result
    is deterministic given ``matrix`` and ``mu``. Reports solver
    success/message, residual, and moment-matrix rank/singular
    values/condition number so callers can detect ill-posed problems and
    stop rather than silently substituting a different prior.
    """
    num_classes = matrix.shape[1]
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    if mu.shape != (num_classes,):
        raise ValueError(f"mu must have shape ({num_classes},), got {mu.shape}")

    x0 = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)

    def objective(p: np.ndarray) -> float:
        r = matrix @ p - mu
        return float(r @ r)

    def objective_grad(p: np.ndarray) -> np.ndarray:
        r = matrix @ p - mu
        return 2.0 * (matrix.T @ r)

    constraints = [{"type": "eq", "fun": lambda p: float(np.sum(p) - 1.0), "jac": lambda p: np.ones_like(p)}]
    bounds = [(_PRIOR_FLOOR, 1.0) for _ in range(num_classes)]

    result = minimize(
        objective,
        x0,
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )

    pi_hat = np.clip(result.x, _PRIOR_FLOOR, None)
    pi_hat = pi_hat / pi_hat.sum()

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0
        else float("inf")
    )
    residual = float(np.linalg.norm(matrix @ pi_hat - mu))

    return PriorEstimate(
        estimated_prior=pi_hat,
        confusion_matrix=matrix,
        reference_marginal=mu,
        success=bool(result.success),
        message=str(result.message),
        residual=residual,
        rank=rank,
        singular_values=singular_values,
        condition_number=condition_number,
    )


def prior_error_metrics(estimated_prior: np.ndarray, true_prior: np.ndarray) -> dict:
    """Component-wise / L1 / total-variation error against a true prior.

    Post-hoc evaluation only; never used to construct or adjust the estimate.
    """
    diff = np.asarray(estimated_prior, dtype=np.float64) - np.asarray(true_prior, dtype=np.float64)
    l1 = float(np.abs(diff).sum())
    return {
        "component_error": diff.tolist(),
        "l1_error": l1,
        "total_variation_error": l1 / 2.0,
    }


@dataclass
class BootstrapResult:
    lower_2_5: np.ndarray
    upper_97_5: np.ndarray
    n_replicates: int
    seed: int
    replicate_priors: np.ndarray = field(repr=False)

    def to_json(self) -> dict:
        return {
            "lower_2_5": self.lower_2_5.tolist(),
            "upper_97_5": self.upper_97_5.tolist(),
            "n_replicates": int(self.n_replicates),
            "seed": int(self.seed),
        }


def bootstrap_prior_ci(
    declared_predicted_probs: np.ndarray,
    declared_labels: np.ndarray,
    reference_predicted_probs: np.ndarray,
    num_classes: int,
    *,
    n_replicates: int = 1000,
    seed: int,
) -> BootstrapResult:
    """Percentile bootstrap CI for each estimated-prior component.

    Each replicate resamples declared predictions within each true class and
    resamples unlabelled reference predictions independently, reconstructs
    ``M`` and ``mu``, and re-solves the same constrained problem. This
    function never receives reference-batch labels. It is a CPU diagnostic
    only and must not influence the prior actually used for student
    training.
    """
    if n_replicates < 1000:
        raise ValueError(f"n_replicates must be >= 1000, got {n_replicates}")
    rng = np.random.default_rng(seed)
    class_indices = {
        j: np.where(declared_labels == j)[0] for j in range(num_classes)
    }
    for j, idx in class_indices.items():
        if len(idx) == 0:
            raise ValueError(f"no declared-label rows for class {j}; cannot bootstrap")
    n_reference = reference_predicted_probs.shape[0]

    replicate_priors = np.empty((n_replicates, num_classes), dtype=np.float64)
    for b in range(n_replicates):
        matrix_b = np.empty((num_classes, num_classes), dtype=np.float64)
        for j in range(num_classes):
            resampled = rng.choice(class_indices[j], size=len(class_indices[j]), replace=True)
            matrix_b[:, j] = declared_predicted_probs[resampled].mean(axis=0)
        ref_resampled = rng.choice(n_reference, size=n_reference, replace=True)
        mu_b = reference_predicted_probs[ref_resampled].mean(axis=0)
        estimate_b = estimate_prior_bbse(matrix_b, mu_b)
        replicate_priors[b] = estimate_b.estimated_prior

    lower = np.percentile(replicate_priors, 2.5, axis=0)
    upper = np.percentile(replicate_priors, 97.5, axis=0)
    return BootstrapResult(
        lower_2_5=lower,
        upper_97_5=upper,
        n_replicates=n_replicates,
        seed=seed,
        replicate_priors=replicate_priors,
    )
