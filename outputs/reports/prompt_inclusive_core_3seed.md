# Prompt-inclusive core student results (3-seed)

Mandatory like-for-like grid under `prompt-inclusive-v1` label accounting and `large-test-v1` evaluation.

Seeds: 42, 123, 456. Report every seed value; mean ± sample SD is secondary.

Paired McNemar tests are computed **per training seed** on the fixed test rows. Holm-adjusted p-values correct the three seed-level tests within each dataset × L × comparison cell. Seed-level SE of paired deltas is descriptive only.

This is a three-seed pilot, not an established population-level claim.

## ag_news · L=10

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 65.09% ± 5.77 | 67.37%, 58.53%, 69.38% | 64.13% ± 5.05 | 65.31%, 58.60%, 68.50% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 87.14% ± 0.49 | 87.50%, 86.58%, 87.33% | 87.02% ± 0.55 | 87.43%, 86.40%, 87.24% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 88.80% ± 0.28 | 89.07%, 88.83%, 88.51% | 88.78% ± 0.28 | 89.05%, 88.81%, 88.49% |
| logit_mean | `multi-mean-guided` | 85.70% ± 0.34 | 85.59%, 85.43%, 86.08% | 85.44% ± 0.35 | 85.35%, 85.15%, 85.83% |
| prob_mean | `multi-probmean-guided` | 86.10% ± 0.41 | 85.67%, 86.14%, 86.49% | 85.89% ± 0.44 | 85.44%, 85.91%, 86.32% |
| prior_matched | `multi-prior-matched-guided` | 88.36% ± 0.23 | 88.54%, 88.09%, 88.43% | 88.32% ± 0.24 | 88.50%, 88.05%, 88.41% |
| scale_aligned | `multi-scale-aligned-guided` | 86.22% ± 0.41 | 86.70%, 85.96%, 86.00% | 86.02% ± 0.46 | 86.55%, 85.71%, 85.80% |
| combined | `multi-prior-scale-aligned-guided` | 88.47% ± 0.18 | 88.59%, 88.26%, 88.55% | 88.44% ± 0.19 | 88.57%, 88.22%, 88.54% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +22.04 ± 5.32 (descriptive SE 3.07)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 87.50% | 67.37% | +20.13 | 1757 | 227 | 1.017e-292 | 2.035e-292 | raw_qwen |
| 123 | 86.58% | 58.53% | +28.05 | 2515 | 383 | 0 | 0 | raw_qwen |
| 456 | 87.33% | 69.38% | +17.95 | 1794 | 430 | 3.457e-197 | 3.457e-197 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: +1.67 ± 0.54 (descriptive SE 0.31)
- Seed wins (accuracy): prior_matched_qwen=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 89.07% | 87.50% | +1.57 | 257 | 138 | 2.21e-09 | 4.42e-09 | prior_matched_qwen |
| 123 | 88.83% | 86.58% | +2.25 | 305 | 134 | 2.046e-16 | 6.138e-16 | prior_matched_qwen |
| 456 | 88.51% | 87.33% | +1.18 | 224 | 134 | 2.273e-06 | 2.273e-06 | prior_matched_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: -1.43 ± 0.41 (descriptive SE 0.24)
- Seed wins (accuracy): logit_mean=0, raw_qwen=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 85.59% | 87.50% | -1.91 | 64 | 209 | 3.959e-19 | 1.188e-18 | raw_qwen |
| 123 | 85.43% | 86.58% | -1.14 | 70 | 157 | 7.564e-09 | 1.513e-08 | raw_qwen |
| 456 | 86.08% | 87.33% | -1.25 | 93 | 188 | 1.51e-08 | 1.513e-08 | raw_qwen |

#### probability mean − logit mean

- Mean paired Δacc: +0.40 ± 0.32 (descriptive SE 0.18)
- Seed wins (accuracy): prob_mean=3, logit_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 85.67% | 85.59% | +0.08 | 76 | 70 | 0.6792 | 0.6792 | prob_mean |
| 123 | 86.14% | 85.43% | +0.71 | 125 | 71 | 0.0001402 | 0.0004207 | prob_mean |
| 456 | 86.49% | 86.08% | +0.41 | 129 | 98 | 0.04622 | 0.09245 | prob_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: +2.25 ± 0.53 (descriptive SE 0.31)
- Seed wins (accuracy): prior_matched=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.54% | 85.67% | +2.87 | 340 | 122 | 7.343e-25 | 2.203e-24 | prior_matched |
| 123 | 88.09% | 86.14% | +1.95 | 276 | 128 | 1.395e-13 | 2.791e-13 | prior_matched |
| 456 | 88.43% | 86.49% | +1.95 | 305 | 157 | 5.285e-12 | 5.285e-12 | prior_matched |

#### scale-aligned mean − probability mean

- Mean paired Δacc: +0.12 ± 0.80 (descriptive SE 0.46)
- Seed wins (accuracy): scale_aligned=1, prob_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 86.70% | 85.67% | +1.03 | 114 | 36 | 1.223e-10 | 3.669e-10 | scale_aligned |
| 123 | 85.96% | 86.14% | -0.18 | 47 | 61 | 0.2108 | 0.2108 | prob_mean |
| 456 | 86.00% | 86.49% | -0.49 | 61 | 98 | 0.004158 | 0.008317 | prob_mean |

#### combined alignment − probability mean

- Mean paired Δacc: +2.37 ± 0.48 (descriptive SE 0.28)
- Seed wins (accuracy): combined=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.59% | 85.67% | +2.92 | 368 | 146 | 3.865e-23 | 1.159e-22 | combined |
| 123 | 88.26% | 86.14% | +2.12 | 299 | 138 | 9.768e-15 | 1.954e-14 | combined |
| 456 | 88.55% | 86.49% | +2.07 | 291 | 134 | 1.934e-14 | 1.954e-14 | combined |

#### combined alignment − logit mean

- Mean paired Δacc: +2.77 ± 0.27 (descriptive SE 0.16)
- Seed wins (accuracy): combined=3, logit_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.59% | 85.59% | +3.00 | 382 | 154 | 2.275e-23 | 6.824e-23 | combined |
| 123 | 88.26% | 85.43% | +2.83 | 356 | 141 | 1.726e-22 | 3.452e-22 | combined |
| 456 | 88.55% | 86.08% | +2.47 | 359 | 171 | 2.217e-16 | 2.217e-16 | combined |

#### combined alignment − prior-matched mean

- Mean paired Δacc: +0.11 ± 0.06 (descriptive SE 0.03)
- Seed wins (accuracy): combined=3, prior_matched=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.59% | 88.54% | +0.05 | 57 | 53 | 0.775 | 1 | combined |
| 123 | 88.26% | 88.09% | +0.17 | 87 | 74 | 0.3443 | 1 | combined |
| 456 | 88.55% | 88.43% | +0.12 | 69 | 60 | 0.4814 | 1 | combined |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: -0.33 ± 0.33 (descriptive SE 0.19)
- Seed wins (accuracy): combined=1, prior_matched_qwen=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.59% | 89.07% | -0.47 | 69 | 105 | 0.007788 | 0.01558 | prior_matched_qwen |
| 123 | 88.26% | 88.83% | -0.57 | 67 | 110 | 0.001522 | 0.004565 | prior_matched_qwen |
| 456 | 88.55% | 88.51% | +0.04 | 75 | 72 | 0.869 | 0.869 | combined |

## ag_news · L=50

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 84.42% ± 1.30 | 85.57%, 83.01%, 84.67% | 84.32% ± 1.34 | 85.55%, 82.89%, 84.51% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 87.08% ± 0.10 | 87.01%, 87.04%, 87.20% | 86.96% ± 0.10 | 86.89%, 86.91%, 87.08% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 88.81% ± 0.07 | 88.75%, 88.88%, 88.79% | 88.79% ± 0.06 | 88.73%, 88.86%, 88.78% |
| logit_mean | `multi-mean-guided` | 86.39% ± 1.28 | 86.89%, 84.93%, 87.33% | 86.16% ± 1.42 | 86.72%, 84.54%, 87.21% |
| prob_mean | `multi-probmean-guided` | 86.66% ± 0.50 | 86.62%, 86.18%, 87.18% | 86.50% ± 0.54 | 86.44%, 85.99%, 87.07% |
| prior_matched | `multi-prior-matched-guided` | 88.46% ± 0.04 | 88.43%, 88.50%, 88.43% | 88.43% ± 0.05 | 88.39%, 88.48%, 88.41% |
| scale_aligned | `multi-scale-aligned-guided` | 86.79% ± 0.42 | 87.12%, 86.32%, 86.95% | 86.65% ± 0.47 | 87.01%, 86.12%, 86.81% |
| combined | `multi-prior-scale-aligned-guided` | 88.30% ± 0.32 | 88.22%, 88.03%, 88.64% | 88.27% ± 0.33 | 88.19%, 87.99%, 88.63% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +2.67 ± 1.30 (descriptive SE 0.75)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 87.01% | 85.57% | +1.45 | 512 | 402 | 0.0003069 | 0.0003069 | raw_qwen |
| 123 | 87.04% | 83.01% | +4.03 | 625 | 319 | 1.299e-23 | 3.898e-23 | raw_qwen |
| 456 | 87.20% | 84.67% | +2.53 | 513 | 321 | 3.082e-11 | 6.165e-11 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: +1.72 ± 0.13 (descriptive SE 0.07)
- Seed wins (accuracy): prior_matched_qwen=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.75% | 87.01% | +1.74 | 301 | 169 | 1.189e-09 | 2.378e-09 | prior_matched_qwen |
| 123 | 88.88% | 87.04% | +1.84 | 263 | 123 | 8.453e-13 | 2.536e-12 | prior_matched_qwen |
| 456 | 88.79% | 87.20% | +1.59 | 269 | 148 | 3.278e-09 | 3.278e-09 | prior_matched_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: -0.70 ± 1.23 (descriptive SE 0.71)
- Seed wins (accuracy): logit_mean=1, raw_qwen=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 86.89% | 87.01% | -0.12 | 138 | 147 | 0.6357 | 1 | raw_qwen |
| 123 | 84.93% | 87.04% | -2.11 | 74 | 234 | 1.682e-20 | 5.046e-20 | raw_qwen |
| 456 | 87.33% | 87.20% | +0.13 | 110 | 100 | 0.5347 | 1 | logit_mean |

#### probability mean − logit mean

- Mean paired Δacc: +0.28 ± 0.85 (descriptive SE 0.49)
- Seed wins (accuracy): prob_mean=1, logit_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 86.62% | 86.89% | -0.28 | 93 | 114 | 0.1643 | 0.3287 | logit_mean |
| 123 | 86.18% | 84.93% | +1.25 | 162 | 67 | 2.932e-10 | 8.797e-10 | prob_mean |
| 456 | 87.18% | 87.33% | -0.14 | 94 | 105 | 0.4785 | 0.4785 | logit_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: +1.79 ± 0.53 (descriptive SE 0.31)
- Seed wins (accuracy): prior_matched=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.43% | 86.62% | +1.82 | 241 | 103 | 6.974e-14 | 1.395e-13 | prior_matched |
| 123 | 88.50% | 86.18% | +2.32 | 319 | 143 | 1.68e-16 | 5.04e-16 | prior_matched |
| 456 | 88.43% | 87.18% | +1.25 | 216 | 121 | 2.558e-07 | 2.558e-07 | prior_matched |

#### scale-aligned mean − probability mean

- Mean paired Δacc: +0.13 ± 0.37 (descriptive SE 0.21)
- Seed wins (accuracy): scale_aligned=2, prob_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 87.12% | 86.62% | +0.50 | 109 | 71 | 0.005668 | 0.017 | scale_aligned |
| 123 | 86.32% | 86.18% | +0.13 | 77 | 67 | 0.4534 | 0.4534 | scale_aligned |
| 456 | 86.95% | 87.18% | -0.24 | 23 | 41 | 0.03277 | 0.06553 | prob_mean |

#### combined alignment − probability mean

- Mean paired Δacc: +1.64 ± 0.19 (descriptive SE 0.11)
- Seed wins (accuracy): combined=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.22% | 86.62% | +1.61 | 296 | 174 | 2.001e-08 | 3.04e-08 | combined |
| 123 | 88.03% | 86.18% | +1.84 | 285 | 145 | 1.356e-11 | 4.068e-11 | combined |
| 456 | 88.64% | 87.18% | +1.46 | 247 | 136 | 1.52e-08 | 3.04e-08 | combined |

#### combined alignment − logit mean

- Mean paired Δacc: +1.91 ± 1.02 (descriptive SE 0.59)
- Seed wins (accuracy): combined=3, logit_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.22% | 86.89% | +1.33 | 266 | 165 | 1.312e-06 | 1.312e-06 | combined |
| 123 | 88.03% | 84.93% | +3.09 | 397 | 162 | 9.176e-24 | 2.753e-23 | combined |
| 456 | 88.64% | 87.33% | +1.32 | 248 | 148 | 5.722e-07 | 1.144e-06 | combined |

#### combined alignment − prior-matched mean

- Mean paired Δacc: -0.16 ± 0.35 (descriptive SE 0.20)
- Seed wins (accuracy): combined=1, prior_matched=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.22% | 88.43% | -0.21 | 77 | 93 | 0.2499 | 0.2972 | prior_matched |
| 123 | 88.03% | 88.50% | -0.47 | 61 | 97 | 0.005197 | 0.01559 | prior_matched |
| 456 | 88.64% | 88.43% | +0.21 | 62 | 46 | 0.1486 | 0.2972 | combined |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: -0.51 ± 0.36 (descriptive SE 0.21)
- Seed wins (accuracy): combined=0, prior_matched_qwen=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.22% | 88.75% | -0.53 | 59 | 99 | 0.001826 | 0.003653 | prior_matched_qwen |
| 123 | 88.03% | 88.88% | -0.86 | 70 | 135 | 6.615e-06 | 1.984e-05 | prior_matched_qwen |
| 456 | 88.64% | 88.79% | -0.14 | 53 | 64 | 0.3553 | 0.3553 | prior_matched_qwen |

## yahoo · L=10

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 26.76% ± 2.66 | 29.50%, 26.61%, 24.18% | 24.92% ± 2.19 | 27.17%, 24.80%, 22.79% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 59.20% ± 0.57 | 59.82%, 59.09%, 58.70% | 59.98% ± 0.39 | 60.29%, 60.11%, 59.55% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 60.92% ± 0.21 | 61.16%, 60.80%, 60.80% | 60.84% ± 0.24 | 61.06%, 60.86%, 60.59% |
| logit_mean | `multi-mean-guided` | 60.77% ± 0.43 | 60.92%, 61.10%, 60.28% | 61.02% ± 0.40 | 61.10%, 61.38%, 60.59% |
| prob_mean | `multi-probmean-guided` | 60.48% ± 0.18 | 60.29%, 60.49%, 60.65% | 60.73% ± 0.26 | 60.47%, 60.98%, 60.72% |
| prior_matched | `multi-prior-matched-guided` | 61.62% ± 0.44 | 62.10%, 61.23%, 61.52% | 61.50% ± 0.38 | 61.93%, 61.21%, 61.35% |
| scale_aligned | `multi-scale-aligned-guided` | 60.00% ± 0.14 | 60.12%, 60.03%, 59.85% | 60.13% ± 0.23 | 60.24%, 60.29%, 59.87% |
| combined | `multi-prior-scale-aligned-guided` | 61.60% ± 0.13 | 61.72%, 61.47%, 61.60% | 61.40% ± 0.19 | 61.63%, 61.29%, 61.30% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +32.44 ± 2.10 (descriptive SE 1.21)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 59.82% | 29.50% | +30.32 | 3518 | 486 | 0 | 0 | raw_qwen |
| 123 | 59.09% | 26.61% | +32.48 | 3794 | 546 | 0 | 0 | raw_qwen |
| 456 | 58.70% | 24.18% | +34.52 | 3983 | 531 | 0 | 0 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: +1.72 ± 0.38 (descriptive SE 0.22)
- Seed wins (accuracy): prior_matched_qwen=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.16% | 59.82% | +1.34 | 536 | 402 | 1.364e-05 | 1.364e-05 | prior_matched_qwen |
| 123 | 60.80% | 59.09% | +1.71 | 574 | 403 | 4.975e-08 | 9.951e-08 | prior_matched_qwen |
| 456 | 60.80% | 58.70% | +2.10 | 638 | 428 | 1.349e-10 | 4.046e-10 | prior_matched_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: +1.56 ± 0.46 (descriptive SE 0.26)
- Seed wins (accuracy): logit_mean=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 60.92% | 59.82% | +1.10 | 419 | 309 | 5.189e-05 | 5.189e-05 | logit_mean |
| 123 | 61.10% | 59.09% | +2.01 | 498 | 297 | 9.979e-13 | 2.994e-12 | logit_mean |
| 456 | 60.28% | 58.70% | +1.58 | 458 | 300 | 1.049e-08 | 2.099e-08 | logit_mean |

#### probability mean − logit mean

- Mean paired Δacc: -0.29 ± 0.57 (descriptive SE 0.33)
- Seed wins (accuracy): prob_mean=1, logit_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 60.29% | 60.92% | -0.63 | 292 | 355 | 0.01473 | 0.03114 | logit_mean |
| 123 | 60.49% | 61.10% | -0.61 | 244 | 305 | 0.01038 | 0.03114 | logit_mean |
| 456 | 60.65% | 60.28% | +0.37 | 319 | 282 | 0.1419 | 0.1419 | prob_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: +1.14 ± 0.58 (descriptive SE 0.34)
- Seed wins (accuracy): prior_matched=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 62.10% | 60.29% | +1.81 | 520 | 339 | 7.104e-10 | 2.131e-09 | prior_matched |
| 123 | 61.23% | 60.49% | +0.74 | 490 | 416 | 0.01525 | 0.01525 | prior_matched |
| 456 | 61.52% | 60.65% | +0.87 | 473 | 386 | 0.00332 | 0.00664 | prior_matched |

#### scale-aligned mean − probability mean

- Mean paired Δacc: -0.48 ± 0.32 (descriptive SE 0.18)
- Seed wins (accuracy): scale_aligned=0, prob_mean=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 60.12% | 60.29% | -0.17 | 252 | 269 | 0.4834 | 0.4834 | prob_mean |
| 123 | 60.03% | 60.49% | -0.46 | 227 | 273 | 0.04406 | 0.08813 | prob_mean |
| 456 | 59.85% | 60.65% | -0.80 | 209 | 289 | 0.0003897 | 0.001169 | prob_mean |

#### combined alignment − probability mean

- Mean paired Δacc: +1.12 ± 0.27 (descriptive SE 0.16)
- Seed wins (accuracy): combined=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.72% | 60.29% | +1.43 | 497 | 354 | 1.068e-06 | 3.205e-06 | combined |
| 123 | 61.47% | 60.49% | +0.98 | 481 | 383 | 0.000956 | 0.001912 | combined |
| 456 | 61.60% | 60.65% | +0.95 | 511 | 416 | 0.002003 | 0.002003 | combined |

#### combined alignment − logit mean

- Mean paired Δacc: +0.83 ± 0.48 (descriptive SE 0.27)
- Seed wins (accuracy): combined=3, logit_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.72% | 60.92% | +0.80 | 506 | 426 | 0.009624 | 0.01925 | combined |
| 123 | 61.47% | 61.10% | +0.37 | 428 | 391 | 0.2084 | 0.2084 | combined |
| 456 | 61.60% | 60.28% | +1.32 | 622 | 490 | 8.4e-05 | 0.000252 | combined |

#### combined alignment − prior-matched mean

- Mean paired Δacc: -0.02 ± 0.32 (descriptive SE 0.19)
- Seed wins (accuracy): combined=2, prior_matched=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.72% | 62.10% | -0.38 | 203 | 241 | 0.07898 | 0.2369 | prior_matched |
| 123 | 61.47% | 61.23% | +0.24 | 244 | 220 | 0.2856 | 0.5712 | combined |
| 456 | 61.60% | 61.52% | +0.08 | 278 | 270 | 0.765 | 0.765 | combined |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: +0.68 ± 0.12 (descriptive SE 0.07)
- Seed wins (accuracy): combined=3, prior_matched_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.72% | 61.16% | +0.56 | 326 | 270 | 0.02419 | 0.02419 | combined |
| 123 | 61.47% | 60.80% | +0.67 | 320 | 253 | 0.005783 | 0.01393 | combined |
| 456 | 61.60% | 60.80% | +0.80 | 430 | 350 | 0.004644 | 0.01393 | combined |

## yahoo · L=50

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 52.14% ± 1.41 | 53.77%, 51.32%, 51.33% | 52.30% ± 1.33 | 53.82%, 51.73%, 51.36% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 58.70% ± 1.45 | 59.40%, 57.04%, 59.67% | 59.05% ± 1.25 | 59.64%, 57.61%, 59.89% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 60.55% ± 0.81 | 61.38%, 59.77%, 60.49% | 60.58% ± 0.72 | 61.36%, 59.93%, 60.46% |
| logit_mean | `multi-mean-guided` | 60.55% ± 0.96 | 60.46%, 59.64%, 61.56% | 60.56% ± 0.93 | 60.30%, 59.78%, 61.59% |
| prob_mean | `multi-probmean-guided` | 59.84% ± 1.10 | 60.96%, 58.76%, 59.81% | 59.79% ± 0.77 | 60.68%, 59.25%, 59.45% |
| prior_matched | `multi-prior-matched-guided` | 61.32% ± 0.34 | 61.64%, 61.34%, 60.97% | 61.24% ± 0.36 | 61.55%, 61.32%, 60.85% |
| scale_aligned | `multi-scale-aligned-guided` | 60.37% ± 0.94 | 60.90%, 59.28%, 60.93% | 60.23% ± 0.60 | 60.31%, 59.60%, 60.78% |
| combined | `multi-prior-scale-aligned-guided` | 60.57% ± 0.64 | 61.31%, 60.15%, 60.25% | 60.50% ± 0.66 | 61.26%, 60.23%, 60.02% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +6.56 ± 1.54 (descriptive SE 0.89)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 59.40% | 53.77% | +5.63 | 1156 | 593 | 7.134e-42 | 1.427e-41 | raw_qwen |
| 123 | 57.04% | 51.32% | +5.72 | 1395 | 823 | 3.412e-34 | 3.412e-34 | raw_qwen |
| 456 | 59.67% | 51.33% | +8.34 | 1432 | 598 | 1.485e-78 | 4.455e-78 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: +1.84 ± 0.96 (descriptive SE 0.56)
- Seed wins (accuracy): prior_matched_qwen=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.38% | 59.40% | +1.98 | 665 | 467 | 4.365e-09 | 8.73e-09 | prior_matched_qwen |
| 123 | 59.77% | 57.04% | +2.73 | 633 | 360 | 3.741e-18 | 1.122e-17 | prior_matched_qwen |
| 456 | 60.49% | 59.67% | +0.82 | 554 | 472 | 0.01141 | 0.01141 | prior_matched_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: +1.85 ± 0.77 (descriptive SE 0.45)
- Seed wins (accuracy): logit_mean=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 60.46% | 59.40% | +1.06 | 538 | 432 | 0.0007399 | 0.0007399 | logit_mean |
| 123 | 59.64% | 57.04% | +2.60 | 712 | 452 | 2.483e-14 | 7.449e-14 | logit_mean |
| 456 | 61.56% | 59.67% | +1.89 | 509 | 320 | 5.477e-11 | 1.095e-10 | logit_mean |

#### probability mean − logit mean

- Mean paired Δacc: -0.71 ± 1.13 (descriptive SE 0.66)
- Seed wins (accuracy): prob_mean=1, logit_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 60.96% | 60.46% | +0.50 | 354 | 304 | 0.05602 | 0.05602 | prob_mean |
| 123 | 58.76% | 59.64% | -0.88 | 420 | 508 | 0.004266 | 0.008533 | logit_mean |
| 456 | 59.81% | 61.56% | -1.75 | 365 | 540 | 6.573e-09 | 1.972e-08 | logit_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: +1.47 ± 0.99 (descriptive SE 0.57)
- Seed wins (accuracy): prior_matched=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.64% | 60.96% | +0.68 | 443 | 375 | 0.01909 | 0.01909 | prior_matched |
| 123 | 61.34% | 58.76% | +2.58 | 581 | 323 | 7.552e-18 | 2.266e-17 | prior_matched |
| 456 | 60.97% | 59.81% | +1.16 | 528 | 412 | 0.0001732 | 0.0003463 | prior_matched |

#### scale-aligned mean − probability mean

- Mean paired Δacc: +0.53 ± 0.59 (descriptive SE 0.34)
- Seed wins (accuracy): scale_aligned=2, prob_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 60.90% | 60.96% | -0.06 | 283 | 289 | 0.8344 | 0.8344 | prob_mean |
| 123 | 59.28% | 58.76% | +0.52 | 303 | 251 | 0.03016 | 0.06032 | scale_aligned |
| 456 | 60.93% | 59.81% | +1.12 | 415 | 303 | 3.32e-05 | 9.96e-05 | scale_aligned |

#### combined alignment − probability mean

- Mean paired Δacc: +0.73 ± 0.58 (descriptive SE 0.33)
- Seed wins (accuracy): combined=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.31% | 60.96% | +0.35 | 464 | 429 | 0.2552 | 0.3408 | combined |
| 123 | 60.15% | 58.76% | +1.39 | 533 | 394 | 5.611e-06 | 1.683e-05 | combined |
| 456 | 60.25% | 59.81% | +0.44 | 514 | 470 | 0.1704 | 0.3408 | combined |

#### combined alignment − logit mean

- Mean paired Δacc: +0.02 ± 1.16 (descriptive SE 0.67)
- Seed wins (accuracy): combined=2, logit_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.31% | 60.46% | +0.85 | 545 | 460 | 0.008025 | 0.01605 | combined |
| 123 | 60.15% | 59.64% | +0.51 | 596 | 545 | 0.1388 | 0.1388 | combined |
| 456 | 60.25% | 61.56% | -1.31 | 452 | 583 | 5.213e-05 | 0.0001564 | logit_mean |

#### combined alignment − prior-matched mean

- Mean paired Δacc: -0.75 ± 0.43 (descriptive SE 0.25)
- Seed wins (accuracy): combined=0, prior_matched=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.31% | 61.64% | -0.33 | 254 | 287 | 0.1688 | 0.1688 | prior_matched |
| 123 | 60.15% | 61.34% | -1.19 | 285 | 404 | 6.606e-06 | 1.982e-05 | prior_matched |
| 456 | 60.25% | 60.97% | -0.72 | 301 | 373 | 0.0062 | 0.0124 | prior_matched |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: +0.02 ± 0.32 (descriptive SE 0.18)
- Seed wins (accuracy): combined=1, prior_matched_qwen=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.31% | 61.38% | -0.07 | 289 | 296 | 0.8041 | 0.8192 | prior_matched_qwen |
| 123 | 60.15% | 59.77% | +0.38 | 385 | 347 | 0.1714 | 0.5142 | combined |
| 456 | 60.25% | 60.49% | -0.24 | 377 | 401 | 0.4096 | 0.8192 | prior_matched_qwen |

## amazon · L=10

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 22.00% ± 1.18 | 20.83%, 23.18%, 22.00% | 16.50% ± 3.47 | 13.78%, 20.41%, 15.31% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 57.35% ± 0.82 | 56.41%, 57.81%, 57.84% | 54.86% ± 1.94 | 52.71%, 55.34%, 56.51% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 58.94% ± 0.49 | 58.74%, 59.49%, 58.58% | 58.75% ± 0.69 | 58.10%, 59.48%, 58.66% |
| logit_mean | `multi-mean-guided` | 58.19% ± 1.09 | 57.13%, 58.13%, 59.31% | 56.60% ± 1.72 | 54.86%, 56.65%, 58.29% |
| prob_mean | `multi-probmean-guided` | 57.85% ± 1.75 | 55.83%, 58.75%, 58.97% | 56.10% ± 2.67 | 53.02%, 57.53%, 57.76% |
| prior_matched | `multi-prior-matched-guided` | 58.68% ± 0.67 | 57.91%, 59.16%, 58.96% | 58.17% ± 1.26 | 56.76%, 59.17%, 58.59% |
| scale_aligned | `multi-scale-aligned-guided` | 56.70% ± 0.78 | 55.80%, 57.20%, 57.11% | 54.54% ± 1.27 | 53.08%, 55.21%, 55.33% |
| combined | `multi-prior-scale-aligned-guided` | 58.18% ± 0.32 | 57.98%, 58.54%, 58.01% | 58.02% ± 0.61 | 57.32%, 58.44%, 58.30% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +35.35 ± 0.64 (descriptive SE 0.37)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 56.41% | 20.83% | +35.58 | 4867 | 1309 | 0 | 0 | raw_qwen |
| 123 | 57.81% | 23.18% | +34.63 | 4409 | 946 | 0 | 0 | raw_qwen |
| 456 | 57.84% | 22.00% | +35.84 | 4846 | 1262 | 0 | 0 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: +1.58 ± 0.80 (descriptive SE 0.46)
- Seed wins (accuracy): prior_matched_qwen=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 58.74% | 56.41% | +2.33 | 918 | 685 | 6.457e-09 | 1.937e-08 | prior_matched_qwen |
| 123 | 59.49% | 57.81% | +1.68 | 971 | 803 | 7.256e-05 | 0.0001451 | prior_matched_qwen |
| 456 | 58.58% | 57.84% | +0.74 | 925 | 851 | 0.08321 | 0.08321 | prior_matched_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: +0.84 ± 0.58 (descriptive SE 0.34)
- Seed wins (accuracy): logit_mean=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 57.13% | 56.41% | +0.72 | 870 | 798 | 0.0821 | 0.1642 | logit_mean |
| 123 | 58.13% | 57.81% | +0.32 | 919 | 887 | 0.4657 | 0.4657 | logit_mean |
| 456 | 59.31% | 57.84% | +1.47 | 971 | 824 | 0.0005652 | 0.001696 | logit_mean |

#### probability mean − logit mean

- Mean paired Δacc: -0.34 ± 0.96 (descriptive SE 0.55)
- Seed wins (accuracy): prob_mean=1, logit_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 55.83% | 57.13% | -1.30 | 418 | 548 | 3.234e-05 | 9.701e-05 | logit_mean |
| 123 | 58.75% | 58.13% | +0.62 | 663 | 601 | 0.08617 | 0.1723 | prob_mean |
| 456 | 58.97% | 59.31% | -0.34 | 481 | 515 | 0.2957 | 0.2957 | logit_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: +0.83 ± 1.11 (descriptive SE 0.64)
- Seed wins (accuracy): prior_matched=2, prob_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 57.91% | 55.83% | +2.08 | 723 | 515 | 3.711e-09 | 1.113e-08 | prior_matched |
| 123 | 59.16% | 58.75% | +0.41 | 807 | 766 | 0.3132 | 0.6264 | prior_matched |
| 456 | 58.96% | 58.97% | -0.01 | 642 | 643 | 1 | 1 | prob_mean |

#### scale-aligned mean − probability mean

- Mean paired Δacc: -1.15 ± 0.98 (descriptive SE 0.57)
- Seed wins (accuracy): scale_aligned=0, prob_mean=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 55.80% | 55.83% | -0.03 | 347 | 350 | 0.9396 | 0.9396 | prob_mean |
| 123 | 57.20% | 58.75% | -1.55 | 499 | 654 | 5.579e-06 | 1.116e-05 | prob_mean |
| 456 | 57.11% | 58.97% | -1.86 | 384 | 570 | 1.877e-09 | 5.631e-09 | prob_mean |

#### combined alignment − probability mean

- Mean paired Δacc: +0.33 ± 1.62 (descriptive SE 0.94)
- Seed wins (accuracy): combined=1, prob_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 57.98% | 55.83% | +2.15 | 946 | 731 | 1.671e-07 | 5.014e-07 | combined |
| 123 | 58.54% | 58.75% | -0.21 | 710 | 731 | 0.5983 | 0.5983 | prob_mean |
| 456 | 58.01% | 58.97% | -0.96 | 882 | 978 | 0.02758 | 0.05517 | prob_mean |

#### combined alignment − logit mean

- Mean paired Δacc: -0.01 ± 1.14 (descriptive SE 0.66)
- Seed wins (accuracy): combined=2, logit_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 57.98% | 57.13% | +0.85 | 811 | 726 | 0.03211 | 0.06422 | combined |
| 123 | 58.54% | 58.13% | +0.41 | 857 | 816 | 0.3281 | 0.3281 | combined |
| 456 | 58.01% | 59.31% | -1.30 | 896 | 1026 | 0.003246 | 0.009737 | logit_mean |

#### combined alignment − prior-matched mean

- Mean paired Δacc: -0.50 ± 0.52 (descriptive SE 0.30)
- Seed wins (accuracy): combined=1, prior_matched=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 57.98% | 57.91% | +0.07 | 493 | 486 | 0.8479 | 0.8479 | combined |
| 123 | 58.54% | 59.16% | -0.62 | 511 | 573 | 0.06387 | 0.1277 | prior_matched |
| 456 | 58.01% | 58.96% | -0.95 | 655 | 750 | 0.01212 | 0.03637 | prior_matched |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: -0.76 ± 0.19 (descriptive SE 0.11)
- Seed wins (accuracy): combined=0, prior_matched_qwen=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 57.98% | 58.74% | -0.76 | 511 | 587 | 0.02357 | 0.04714 | prior_matched_qwen |
| 123 | 58.54% | 59.49% | -0.95 | 537 | 632 | 0.005949 | 0.01785 | prior_matched_qwen |
| 456 | 58.01% | 58.58% | -0.57 | 674 | 731 | 0.1351 | 0.1351 | prior_matched_qwen |

## amazon · L=50

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 33.76% ± 2.35 | 36.37%, 31.80%, 33.12% | 30.38% ± 1.21 | 30.64%, 31.44%, 29.07% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 57.32% ± 1.05 | 58.53%, 56.86%, 56.58% | 55.26% ± 2.80 | 58.49%, 53.74%, 53.56% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 57.33% ± 1.23 | 56.01%, 57.54%, 58.44% | 57.36% ± 0.81 | 56.44%, 57.65%, 57.97% |
| logit_mean | `multi-mean-guided` | 56.77% ± 0.67 | 56.91%, 57.36%, 56.05% | 55.48% ± 0.86 | 55.74%, 56.19%, 54.53% |
| prob_mean | `multi-probmean-guided` | 56.45% ± 0.26 | 56.45%, 56.71%, 56.20% | 55.13% ± 1.04 | 56.04%, 55.35%, 53.99% |
| prior_matched | `multi-prior-matched-guided` | 57.24% ± 1.30 | 55.95%, 57.22%, 58.54% | 57.02% ± 1.11 | 55.95%, 56.95%, 58.17% |
| scale_aligned | `multi-scale-aligned-guided` | 56.06% ± 0.11 | 56.03%, 55.96%, 56.18% | 54.70% ± 0.72 | 55.48%, 54.56%, 54.06% |
| combined | `multi-prior-scale-aligned-guided` | 53.20% ± 1.51 | 51.50%, 53.70%, 54.40% | 52.27% ± 1.44 | 50.69%, 53.52%, 52.60% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +23.56 ± 1.45 (descriptive SE 0.84)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 58.53% | 36.37% | +22.16 | 3412 | 1196 | 2.195e-243 | 2.195e-243 | raw_qwen |
| 123 | 56.86% | 31.80% | +25.06 | 3444 | 938 | 0 | 0 | raw_qwen |
| 456 | 56.58% | 33.12% | +23.46 | 3356 | 1010 | 7.902e-291 | 1.58e-290 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: +0.01 ± 2.27 (descriptive SE 1.31)
- Seed wins (accuracy): prior_matched_qwen=2, raw_qwen=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 56.01% | 58.53% | -2.52 | 628 | 880 | 9.277e-11 | 2.783e-10 | raw_qwen |
| 123 | 57.54% | 56.86% | +0.68 | 965 | 897 | 0.1205 | 0.1205 | prior_matched_qwen |
| 456 | 58.44% | 56.58% | +1.86 | 915 | 729 | 4.941e-06 | 9.881e-06 | prior_matched_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: -0.55 ± 1.06 (descriptive SE 0.61)
- Seed wins (accuracy): logit_mean=1, raw_qwen=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 56.91% | 58.53% | -1.62 | 656 | 818 | 2.699e-05 | 8.098e-05 | raw_qwen |
| 123 | 57.36% | 56.86% | +0.50 | 924 | 874 | 0.2478 | 0.3734 | logit_mean |
| 456 | 56.05% | 56.58% | -0.53 | 749 | 802 | 0.1867 | 0.3734 | raw_qwen |

#### probability mean − logit mean

- Mean paired Δacc: -0.32 ± 0.42 (descriptive SE 0.24)
- Seed wins (accuracy): prob_mean=1, logit_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 56.45% | 56.91% | -0.46 | 672 | 718 | 0.2274 | 0.4548 | logit_mean |
| 123 | 56.71% | 57.36% | -0.65 | 512 | 577 | 0.0524 | 0.1572 | logit_mean |
| 456 | 56.20% | 56.05% | +0.15 | 638 | 623 | 0.6934 | 0.6934 | prob_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: +0.78 ± 1.44 (descriptive SE 0.83)
- Seed wins (accuracy): prior_matched=2, prob_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 55.95% | 56.45% | -0.50 | 697 | 747 | 0.1972 | 0.3944 | prob_mean |
| 123 | 57.22% | 56.71% | +0.51 | 793 | 742 | 0.2019 | 0.3944 | prior_matched |
| 456 | 58.54% | 56.20% | +2.34 | 887 | 653 | 2.706e-09 | 8.119e-09 | prior_matched |

#### scale-aligned mean − probability mean

- Mean paired Δacc: -0.40 ± 0.37 (descriptive SE 0.21)
- Seed wins (accuracy): scale_aligned=0, prob_mean=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 56.03% | 56.45% | -0.42 | 486 | 528 | 0.1979 | 0.3958 | prob_mean |
| 123 | 55.96% | 56.71% | -0.75 | 493 | 568 | 0.02305 | 0.06916 | prob_mean |
| 456 | 56.18% | 56.20% | -0.02 | 410 | 412 | 0.9722 | 0.9722 | prob_mean |

#### combined alignment − probability mean

- Mean paired Δacc: -3.25 ± 1.59 (descriptive SE 0.92)
- Seed wins (accuracy): combined=0, prob_mean=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 51.50% | 56.45% | -4.95 | 674 | 1169 | 5.373e-31 | 1.612e-30 | prob_mean |
| 123 | 53.70% | 56.71% | -3.01 | 1021 | 1322 | 5.433e-10 | 1.087e-09 | prob_mean |
| 456 | 54.40% | 56.20% | -1.80 | 649 | 829 | 3.139e-06 | 3.139e-06 | prob_mean |

#### combined alignment − logit mean

- Mean paired Δacc: -3.57 ± 1.88 (descriptive SE 1.09)
- Seed wins (accuracy): combined=0, logit_mean=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 51.50% | 56.91% | -5.41 | 788 | 1329 | 3.854e-32 | 1.156e-31 | logit_mean |
| 123 | 53.70% | 57.36% | -3.66 | 1093 | 1459 | 4.573e-13 | 9.146e-13 | logit_mean |
| 456 | 54.40% | 56.05% | -1.65 | 605 | 770 | 9.522e-06 | 9.522e-06 | logit_mean |

#### combined alignment − prior-matched mean

- Mean paired Δacc: -4.04 ± 0.47 (descriptive SE 0.27)
- Seed wins (accuracy): combined=0, prior_matched=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 51.50% | 55.95% | -4.45 | 502 | 947 | 6.436e-32 | 1.931e-31 | prior_matched |
| 123 | 53.70% | 57.22% | -3.52 | 550 | 902 | 2.109e-20 | 2.109e-20 | prior_matched |
| 456 | 54.40% | 58.54% | -4.14 | 443 | 857 | 7.061e-31 | 1.412e-30 | prior_matched |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: -4.13 ± 0.34 (descriptive SE 0.20)
- Seed wins (accuracy): combined=0, prior_matched_qwen=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 51.50% | 56.01% | -4.51 | 611 | 1062 | 1.766e-28 | 5.299e-28 | prior_matched_qwen |
| 123 | 53.70% | 57.54% | -3.84 | 585 | 969 | 1.584e-22 | 1.584e-22 | prior_matched_qwen |
| 456 | 54.40% | 58.44% | -4.04 | 493 | 897 | 1.333e-27 | 2.666e-27 | prior_matched_qwen |

## amazon3 · L=10

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 38.88% ± 1.67 | 40.77%, 38.27%, 37.60% | 37.46% ± 1.69 | 39.39%, 36.27%, 36.71% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 72.11% ± 0.99 | 71.22%, 71.93%, 73.17% | 70.30% ± 1.34 | 69.55%, 69.50%, 71.85% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 74.06% ± 1.17 | 74.68%, 72.72%, 74.78% | 73.53% ± 1.84 | 74.55%, 71.40%, 74.63% |
| logit_mean | `multi-mean-guided` | 71.92% ± 1.42 | 72.08%, 70.42%, 73.25% | 69.74% ± 2.50 | 70.39%, 66.98%, 71.86% |
| prob_mean | `multi-probmean-guided` | 72.24% ± 0.75 | 72.22%, 71.50%, 73.00% | 70.17% ± 1.41 | 70.37%, 68.68%, 71.47% |
| prior_matched | `multi-prior-matched-guided` | 73.90% ± 0.70 | 74.53%, 73.15%, 74.02% | 73.28% ± 1.24 | 74.13%, 71.86%, 73.85% |
| scale_aligned | `multi-scale-aligned-guided` | 70.41% ± 1.55 | 69.95%, 69.13%, 72.13% | 67.68% ± 2.45 | 67.47%, 65.34%, 70.23% |
| combined | `multi-prior-scale-aligned-guided` | 73.79% ± 0.47 | 74.08%, 73.25%, 74.05% | 73.48% ± 0.86 | 73.86%, 72.50%, 74.08% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +33.23 ± 2.59 (descriptive SE 1.49)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 71.22% | 40.77% | +30.45 | 2296 | 469 | 1.184e-287 | 1.184e-287 | raw_qwen |
| 123 | 71.93% | 38.27% | +33.67 | 2354 | 334 | 0 | 0 | raw_qwen |
| 456 | 73.17% | 37.60% | +35.57 | 2801 | 667 | 1.149e-308 | 2.299e-308 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: +1.96 ± 1.37 (descriptive SE 0.79)
- Seed wins (accuracy): prior_matched_qwen=3, raw_qwen=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.68% | 71.22% | +3.47 | 549 | 341 | 3.172e-12 | 9.515e-12 | prior_matched_qwen |
| 123 | 72.72% | 71.93% | +0.78 | 219 | 172 | 0.01988 | 0.01988 | prior_matched_qwen |
| 456 | 74.78% | 73.17% | +1.62 | 364 | 267 | 0.0001289 | 0.0002577 | prior_matched_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: -0.19 ± 1.21 (descriptive SE 0.70)
- Seed wins (accuracy): logit_mean=2, raw_qwen=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 72.08% | 71.22% | +0.87 | 258 | 206 | 0.01781 | 0.03562 | logit_mean |
| 123 | 70.42% | 71.93% | -1.52 | 203 | 294 | 5.176e-05 | 0.0001553 | raw_qwen |
| 456 | 73.25% | 73.17% | +0.08 | 237 | 232 | 0.8535 | 0.8535 | logit_mean |

#### probability mean − logit mean

- Mean paired Δacc: +0.32 ± 0.69 (descriptive SE 0.40)
- Seed wins (accuracy): prob_mean=2, logit_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 72.22% | 72.08% | +0.13 | 229 | 221 | 0.7415 | 0.9073 | prob_mean |
| 123 | 71.50% | 70.42% | +1.08 | 215 | 150 | 0.0007856 | 0.002357 | prob_mean |
| 456 | 73.00% | 73.25% | -0.25 | 167 | 182 | 0.4537 | 0.9073 | logit_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: +1.66 ± 0.65 (descriptive SE 0.38)
- Seed wins (accuracy): prior_matched=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.53% | 72.22% | +2.32 | 401 | 262 | 7.513e-08 | 2.254e-07 | prior_matched |
| 123 | 73.15% | 71.50% | +1.65 | 311 | 212 | 1.729e-05 | 3.459e-05 | prior_matched |
| 456 | 74.02% | 73.00% | +1.02 | 369 | 308 | 0.02104 | 0.02104 | prior_matched |

#### scale-aligned mean − probability mean

- Mean paired Δacc: -1.83 ± 0.84 (descriptive SE 0.48)
- Seed wins (accuracy): scale_aligned=0, prob_mean=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 69.95% | 72.22% | -2.27 | 107 | 243 | 2.693e-13 | 8.08e-13 | prob_mean |
| 123 | 69.13% | 71.50% | -2.37 | 135 | 277 | 2.284e-12 | 4.568e-12 | prob_mean |
| 456 | 72.13% | 73.00% | -0.87 | 100 | 152 | 0.00127 | 0.00127 | prob_mean |

#### combined alignment − probability mean

- Mean paired Δacc: +1.56 ± 0.44 (descriptive SE 0.26)
- Seed wins (accuracy): combined=3, prob_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.08% | 72.22% | +1.87 | 423 | 311 | 4.052e-05 | 0.0001216 | combined |
| 123 | 73.25% | 71.50% | +1.75 | 422 | 317 | 0.0001273 | 0.0002546 | combined |
| 456 | 74.05% | 73.00% | +1.05 | 427 | 364 | 0.02743 | 0.02743 | combined |

#### combined alignment − logit mean

- Mean paired Δacc: +1.88 ± 1.02 (descriptive SE 0.59)
- Seed wins (accuracy): combined=3, logit_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.08% | 72.08% | +2.00 | 419 | 299 | 8.556e-06 | 1.711e-05 | combined |
| 123 | 73.25% | 70.42% | +2.83 | 513 | 343 | 6.842e-09 | 2.053e-08 | combined |
| 456 | 74.05% | 73.25% | +0.80 | 434 | 386 | 0.1007 | 0.1007 | combined |

#### combined alignment − prior-matched mean

- Mean paired Δacc: -0.11 ± 0.30 (descriptive SE 0.17)
- Seed wins (accuracy): combined=2, prior_matched=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.08% | 74.53% | -0.45 | 138 | 165 | 0.1351 | 0.4054 | prior_matched |
| 123 | 73.25% | 73.15% | +0.10 | 208 | 202 | 0.805 | 1 | combined |
| 456 | 74.05% | 74.02% | +0.03 | 225 | 223 | 0.9623 | 1 | combined |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: -0.27 ± 0.70 (descriptive SE 0.40)
- Seed wins (accuracy): combined=1, prior_matched_qwen=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.08% | 74.68% | -0.60 | 177 | 213 | 0.07621 | 0.1524 | prior_matched_qwen |
| 123 | 73.25% | 72.72% | +0.53 | 222 | 190 | 0.1266 | 0.1524 | combined |
| 456 | 74.05% | 74.78% | -0.73 | 166 | 210 | 0.02645 | 0.07936 | prior_matched_qwen |

## amazon3 · L=50

### Method summary (accuracy / macro-F1)

| method | teacher | acc mean±SD | acc seeds | F1 mean±SD | F1 seeds |
|---|---|---:|---|---:|---|
| supervised | `none` | 47.66% ± 2.81 | 50.68%, 45.13%, 47.17% | 46.40% ± 3.78 | 50.76%, 44.10%, 44.34% |
| raw_qwen | `qwen3.5-27b-vllm-guided` | 73.76% ± 0.24 | 73.50%, 73.97%, 73.80% | 73.58% ± 0.24 | 73.51%, 73.38%, 73.84% |
| prior_matched_qwen | `qwen-prior-matched-guided` | 73.24% ± 2.71 | 74.38%, 75.20%, 70.15% | 73.28% ± 2.53 | 74.21%, 75.21%, 70.42% |
| logit_mean | `multi-mean-guided` | 72.33% ± 1.06 | 71.17%, 72.57%, 73.25% | 71.26% ± 1.23 | 70.09%, 71.16%, 72.55% |
| prob_mean | `multi-probmean-guided` | 72.99% ± 0.72 | 72.35%, 72.85%, 73.77% | 72.12% ± 1.06 | 71.18%, 71.91%, 73.26% |
| prior_matched | `multi-prior-matched-guided` | 72.69% ± 1.29 | 73.15%, 73.70%, 71.23% | 72.63% ± 1.37 | 72.59%, 74.02%, 71.29% |
| scale_aligned | `multi-scale-aligned-guided` | 72.72% ± 0.76 | 71.87%, 72.95%, 73.33% | 71.60% ± 1.32 | 70.11%, 72.06%, 72.62% |
| combined | `multi-prior-scale-aligned-guided` | 72.84% ± 1.57 | 74.02%, 73.45%, 71.07% | 72.93% ± 1.52 | 73.79%, 73.84%, 71.17% |

### Pairwise comparisons (A − B)

#### raw Qwen − supervised

- Mean paired Δacc: +26.09 ± 3.04 (descriptive SE 1.76)
- Seed wins (accuracy): raw_qwen=3, supervised=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 73.50% | 50.68% | +22.82 | 1838 | 469 | 1.482e-190 | 1.482e-190 | raw_qwen |
| 123 | 73.97% | 45.13% | +28.83 | 2372 | 642 | 2.046e-231 | 4.091e-231 | raw_qwen |
| 456 | 73.80% | 47.17% | +26.63 | 2012 | 414 | 6.959e-251 | 2.088e-250 | raw_qwen |

#### prior-matched Qwen − raw Qwen

- Mean paired Δacc: -0.51 ± 2.72 (descriptive SE 1.57)
- Seed wins (accuracy): prior_matched_qwen=2, raw_qwen=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.38% | 73.50% | +0.88 | 264 | 211 | 0.01695 | 0.01695 | prior_matched_qwen |
| 123 | 75.20% | 73.97% | +1.23 | 273 | 199 | 0.000762 | 0.001524 | prior_matched_qwen |
| 456 | 70.15% | 73.80% | -3.65 | 326 | 545 | 1.125e-13 | 3.374e-13 | raw_qwen |

#### logit mean − raw Qwen

- Mean paired Δacc: -1.43 ± 0.89 (descriptive SE 0.51)
- Seed wins (accuracy): logit_mean=0, raw_qwen=3, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 71.17% | 73.50% | -2.33 | 275 | 415 | 1.101e-07 | 3.304e-07 | raw_qwen |
| 123 | 72.57% | 73.97% | -1.40 | 196 | 280 | 0.0001371 | 0.0002742 | raw_qwen |
| 456 | 73.25% | 73.80% | -0.55 | 283 | 316 | 0.191 | 0.191 | raw_qwen |

#### probability mean − logit mean

- Mean paired Δacc: +0.66 ± 0.47 (descriptive SE 0.27)
- Seed wins (accuracy): prob_mean=3, logit_mean=0, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 72.35% | 71.17% | +1.18 | 339 | 268 | 0.004455 | 0.01337 | prob_mean |
| 123 | 72.85% | 72.57% | +0.28 | 217 | 200 | 0.4334 | 0.4334 | prob_mean |
| 456 | 73.77% | 73.25% | +0.52 | 218 | 187 | 0.1359 | 0.2719 | prob_mean |

#### prior-matched mean − probability mean

- Mean paired Δacc: -0.29 ± 1.94 (descriptive SE 1.12)
- Seed wins (accuracy): prior_matched=2, prob_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 73.15% | 72.35% | +0.80 | 270 | 222 | 0.03399 | 0.06798 | prior_matched |
| 123 | 73.70% | 72.85% | +0.85 | 481 | 430 | 0.09755 | 0.09755 | prior_matched |
| 456 | 71.23% | 73.77% | -2.53 | 414 | 566 | 1.347e-06 | 4.04e-06 | prob_mean |

#### scale-aligned mean − probability mean

- Mean paired Δacc: -0.27 ± 0.32 (descriptive SE 0.19)
- Seed wins (accuracy): scale_aligned=1, prob_mean=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 71.87% | 72.35% | -0.48 | 224 | 253 | 0.1998 | 0.4825 | prob_mean |
| 123 | 72.95% | 72.85% | +0.10 | 226 | 220 | 0.8129 | 0.8129 | scale_aligned |
| 456 | 73.33% | 73.77% | -0.43 | 146 | 172 | 0.1608 | 0.4825 | prob_mean |

#### combined alignment − probability mean

- Mean paired Δacc: -0.14 ± 2.28 (descriptive SE 1.31)
- Seed wins (accuracy): combined=2, prob_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.02% | 72.35% | +1.67 | 329 | 229 | 2.652e-05 | 5.303e-05 | combined |
| 123 | 73.45% | 72.85% | +0.60 | 460 | 424 | 0.2391 | 0.2391 | combined |
| 456 | 71.07% | 73.77% | -2.70 | 384 | 546 | 1.208e-07 | 3.625e-07 | prob_mean |

#### combined alignment − logit mean

- Mean paired Δacc: +0.52 ± 2.54 (descriptive SE 1.46)
- Seed wins (accuracy): combined=2, logit_mean=1, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.02% | 71.17% | +2.85 | 403 | 232 | 1.147e-11 | 3.442e-11 | combined |
| 123 | 73.45% | 72.57% | +0.88 | 525 | 472 | 0.09954 | 0.09954 | combined |
| 456 | 71.07% | 73.25% | -2.18 | 468 | 599 | 6.764e-05 | 0.0001353 | logit_mean |

#### combined alignment − prior-matched mean

- Mean paired Δacc: +0.15 ± 0.62 (descriptive SE 0.36)
- Seed wins (accuracy): combined=1, prior_matched=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.02% | 73.15% | +0.87 | 210 | 158 | 0.007762 | 0.02329 | combined |
| 123 | 73.45% | 73.70% | -0.25 | 185 | 200 | 0.4756 | 0.9512 | prior_matched |
| 456 | 71.07% | 71.23% | -0.17 | 147 | 157 | 0.6058 | 0.9512 | prior_matched |

#### combined alignment − prior-matched Qwen

- Mean paired Δacc: -0.40 ± 1.33 (descriptive SE 0.77)
- Seed wins (accuracy): combined=1, prior_matched_qwen=2, tie=0

| seed | A acc | B acc | Δacc | only_A | only_B | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.02% | 74.38% | -0.37 | 163 | 185 | 0.2603 | 0.2603 | prior_matched_qwen |
| 123 | 73.45% | 75.20% | -1.75 | 207 | 312 | 4.653e-06 | 1.396e-05 | prior_matched_qwen |
| 456 | 71.07% | 70.15% | +0.92 | 276 | 221 | 0.01534 | 0.03068 | combined |

