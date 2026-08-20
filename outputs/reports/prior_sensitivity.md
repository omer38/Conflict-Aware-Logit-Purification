# Class-prior sensitivity of prior-and-scale-aligned teacher aggregation

This is a CPU-only cached-logit **label-shift simulation**. Gold labels are used
only to construct and evaluate simulated class priors; the correction itself sees
teacher logits, unlabeled pool frequencies, and a supplied target prior. It does
not test covariate shift or a new deployment domain.
Like the deployed pool-level method, matching is transductive: offsets are fitted
and evaluated on the same simulated unlabeled pool.

Reported cells are accuracy / NLL / 10-bin ECE at teacher-target level (tau=1).
Mild and strong synthetic results average over every possible dominant class.

## Source priors

| Dataset | Un-rebalanced source prior | Distinct from uniform? |
|---|---|---|
| ag_news | `[0.250, 0.250, 0.250, 0.250]` | no |
| yahoo | `[0.103, 0.103, 0.102, 0.098, 0.097, 0.099, 0.099, 0.096, 0.100, 0.103]` | yes |
| amazon | `[0.200, 0.200, 0.200, 0.200, 0.200]` | no |
| amazon3 | `[0.400, 0.200, 0.400]` | yes |

## Balanced working pool: assumed-prior sensitivity

| Method | AG News | Yahoo | Amazon | Amazon-3 |
|---|---:|---:|---:|---:|
| probability mean | 0.879 / 0.479 / 0.073 | 0.658 / 1.456 / 0.168 | 0.634 / 1.035 / 0.159 | 0.759 / 0.593 / 0.108 |
| scale alignment only | 0.879 / 0.427 / 0.065 | 0.658 / 1.269 / 0.110 | 0.632 / 0.909 / 0.084 | 0.763 / 0.536 / 0.036 |
| correct uniform prior | 0.899 / 0.343 / 0.022 | 0.663 / 1.184 / 0.088 | 0.659 / 0.851 / 0.026 | 0.801 / 0.473 / 0.026 |
| mildly misspecified prior | 0.869 / 0.533 / 0.068 | 0.642 / 1.339 / 0.081 | 0.635 / 0.990 / 0.059 | 0.784 / 0.553 / 0.035 |
| strongly misspecified prior | 0.650 / 1.032 / 0.137 | 0.458 / 1.941 / 0.212 | 0.439 / 1.413 / 0.206 | 0.618 / 0.945 / 0.154 |

## Natural source prior under label shift

| Method | AG News | Yahoo | Amazon | Amazon-3 |
|---|---:|---:|---:|---:|
| probability mean | 0.879 / 0.479 / 0.073 | 0.658 / 1.450 / 0.167 | 0.634 / 1.035 / 0.159 | 0.820 / 0.460 / 0.068 |
| scale alignment only | 0.879 / 0.427 / 0.065 | 0.658 / 1.265 / 0.109 | 0.632 / 0.909 / 0.084 | 0.824 / 0.426 / 0.018 |
| correct natural prior | 0.899 / 0.343 / 0.022 | 0.663 / 1.184 / 0.087 | 0.659 / 0.851 / 0.026 | 0.827 / 0.419 / 0.025 |
| forced uniform prior | 0.899 / 0.343 / 0.022 | 0.663 / 1.184 / 0.089 | 0.659 / 0.851 / 0.026 | 0.794 / 0.518 / 0.024 |

## Strong synthetic skew under label shift

| Method | AG News | Yahoo | Amazon | Amazon-3 |
|---|---:|---:|---:|---:|
| scale alignment only | 0.879 / 0.430 / 0.066 | 0.658 / 1.269 / 0.111 | 0.631 / 0.906 / 0.095 | 0.763 / 0.534 / 0.050 |
| correct skewed prior | 0.918 / 0.278 / 0.023 | 0.742 / 0.869 / 0.050 | 0.756 / 0.642 / 0.028 | 0.846 / 0.381 / 0.021 |
| forced uniform prior | 0.737 / 1.260 / 0.232 | 0.520 / 2.396 / 0.271 | 0.514 / 1.493 / 0.169 | 0.655 / 0.983 / 0.092 |
| wrong dominant class | 0.339 / 1.636 / 0.420 | 0.271 / 2.728 / 0.439 | 0.280 / 1.837 / 0.402 | 0.385 / 1.512 / 0.349 |

## Interpretation guardrails

- Class-prior matching is unsupervised conditional on a supplied prior; it is
  not label-free in an end-to-end setup where labels established that prior.
- Confidence-scale alignment equalises mean entropy; it does not guarantee
  empirical probability calibration.
- The natural-source row is not an imbalance stress test for datasets whose
  raw benchmark source is already uniform.
- Importance weighting reuses cached class-conditional examples and therefore
  supports claims about prior shift only, not general deployment shift.
