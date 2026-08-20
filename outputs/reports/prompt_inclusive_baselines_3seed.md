# Prompt-inclusive baseline student results (Phase 2, 3 seeds)

Gemma single, Mistral single and the entropy-weighted committee under `prompt-inclusive-v1` accounting on the `large-test-v1` fixed tests, joined against the Phase 1 core grid.

Seeds: 42, 123, 456. Paired exact McNemar tests are computed per seed on identical test rows; Holm adjustment is applied across the three seed-level tests within each comparison.

## ag_news · L=10

Best single teacher: **raw_qwen**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 84.80% ± 0.41 | 84.43% ± 0.48 | 85.26%, 84.62%, 84.51% |
| mistral | 86.42% ± 0.38 | 86.29% ± 0.42 | 86.57%, 85.99%, 86.70% |
| entropy | 85.80% ± 0.38 | 85.53% ± 0.40 | 85.76%, 85.43%, 86.20% |
| raw_qwen | 87.14% ± 0.49 | 87.02% ± 0.55 | 87.50%, 86.58%, 87.33% |
| prior_matched_qwen | 88.80% ± 0.28 | 88.78% ± 0.28 | 89.07%, 88.83%, 88.51% |
| prob_mean | 86.10% ± 0.41 | 85.89% ± 0.44 | 85.67%, 86.14%, 86.49% |
| combined | 88.47% ± 0.18 | 88.44% ± 0.19 | 88.59%, 88.26%, 88.55% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | -2.34 ± 0.44 | 0/3 | yes | 3 |
| mistral − raw_qwen | -0.72 ± 0.19 | 0/3 | yes | 3 |
| entropy − prob_mean | -0.30 ± 0.40 | 1/2 | no | 1 |
| entropy − combined | -2.67 ± 0.27 | 0/3 | yes | 3 |
| combined − best_single:raw_qwen | +1.33 ± 0.31 | 3/0 | yes | 3 |
| prior_matched_qwen − best_single:raw_qwen | +1.67 ± 0.54 | 3/0 | yes | 3 |

## ag_news · L=50

Best single teacher: **raw_qwen**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 85.35% ± 2.25 | 84.99% ± 2.58 | 85.22%, 83.17%, 87.66% |
| mistral | 86.58% ± 0.50 | 86.45% ± 0.52 | 86.89%, 86.00%, 86.84% |
| entropy | 86.50% ± 0.71 | 86.30% ± 0.79 | 86.45%, 85.83%, 87.24% |
| raw_qwen | 87.08% ± 0.10 | 86.96% ± 0.10 | 87.01%, 87.04%, 87.20% |
| prior_matched_qwen | 88.81% ± 0.07 | 88.79% ± 0.06 | 88.75%, 88.88%, 88.79% |
| prob_mean | 86.66% ± 0.50 | 86.50% ± 0.54 | 86.62%, 86.18%, 87.18% |
| combined | 88.30% ± 0.32 | 88.27% ± 0.33 | 88.22%, 88.03%, 88.64% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | -1.73 ± 2.17 | 1/2 | no | 3 |
| mistral − raw_qwen | -0.50 ± 0.48 | 0/3 | yes | 1 |
| entropy − prob_mean | -0.16 ± 0.20 | 1/2 | no | 0 |
| entropy − combined | -1.79 ± 0.40 | 0/3 | yes | 3 |
| combined − best_single:raw_qwen | +1.21 ± 0.23 | 3/0 | yes | 3 |
| prior_matched_qwen − best_single:raw_qwen | +1.72 ± 0.13 | 3/0 | yes | 3 |

## yahoo · L=10

Best single teacher: **gemma**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 60.30% ± 0.25 | 60.65% ± 0.36 | 60.48%, 60.40%, 60.02% |
| mistral | 59.78% ± 0.41 | 59.46% ± 0.62 | 59.58%, 60.25%, 59.51% |
| entropy | 60.45% ± 1.07 | 60.83% ± 0.97 | 61.07%, 61.07%, 59.21% |
| raw_qwen | 59.20% ± 0.57 | 59.98% ± 0.39 | 59.82%, 59.09%, 58.70% |
| prior_matched_qwen | 60.92% ± 0.21 | 60.84% ± 0.24 | 61.16%, 60.80%, 60.80% |
| prob_mean | 60.48% ± 0.18 | 60.73% ± 0.26 | 60.29%, 60.49%, 60.65% |
| combined | 61.60% ± 0.13 | 61.40% ± 0.19 | 61.72%, 61.47%, 61.60% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | +1.10 ± 0.38 | 3/0 | yes | 3 |
| mistral − raw_qwen | +0.58 ± 0.73 | 2/1 | no | 2 |
| entropy − prob_mean | -0.03 ± 1.23 | 2/1 | no | 3 |
| entropy − combined | -1.15 ± 1.08 | 0/3 | yes | 1 |
| combined − best_single:gemma | +1.30 ± 0.26 | 3/0 | yes | 3 |
| prior_matched_qwen − best_single:gemma | +0.62 ± 0.20 | 3/0 | yes | 0 |

## yahoo · L=50

Best single teacher: **mistral**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 59.60% ± 1.50 | 59.82% ± 1.43 | 61.32%, 58.53%, 58.96% |
| mistral | 60.23% ± 0.93 | 59.71% ± 0.73 | 61.21%, 59.35%, 60.13% |
| entropy | 60.06% ± 0.66 | 60.46% ± 0.60 | 60.77%, 59.47%, 59.95% |
| raw_qwen | 58.70% ± 1.45 | 59.05% ± 1.25 | 59.40%, 57.04%, 59.67% |
| prior_matched_qwen | 60.55% ± 0.81 | 60.58% ± 0.72 | 61.38%, 59.77%, 60.49% |
| prob_mean | 59.84% ± 1.10 | 59.79% ± 0.77 | 60.96%, 58.76%, 59.81% |
| combined | 60.57% ± 0.64 | 60.50% ± 0.66 | 61.31%, 60.15%, 60.25% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | +0.90 ± 1.41 | 2/1 | no | 3 |
| mistral − raw_qwen | +1.53 ± 0.96 | 3/0 | yes | 2 |
| entropy − prob_mean | +0.22 ± 0.46 | 2/1 | no | 1 |
| entropy − combined | -0.51 ± 0.19 | 0/3 | yes | 0 |
| combined − best_single:mistral | +0.34 ± 0.40 | 3/0 | yes | 1 |
| prior_matched_qwen − best_single:mistral | +0.32 ± 0.13 | 3/0 | yes | 0 |

## amazon · L=10

Best single teacher: **raw_qwen**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 57.15% ± 0.74 | 55.59% ± 1.37 | 56.38%, 57.86%, 57.22% |
| mistral | 52.41% ± 2.80 | 47.79% ± 5.12 | 49.35%, 54.85%, 53.02% |
| entropy | 58.49% ± 0.46 | 57.56% ± 1.16 | 57.97%, 58.80%, 58.71% |
| raw_qwen | 57.35% ± 0.82 | 54.86% ± 1.94 | 56.41%, 57.81%, 57.84% |
| prior_matched_qwen | 58.94% ± 0.49 | 58.75% ± 0.69 | 58.74%, 59.49%, 58.58% |
| prob_mean | 57.85% ± 1.75 | 56.10% ± 2.67 | 55.83%, 58.75%, 58.97% |
| combined | 58.18% ± 0.32 | 58.02% ± 0.61 | 57.98%, 58.54%, 58.01% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | -0.20 ± 0.37 | 1/2 | no | 0 |
| mistral − raw_qwen | -4.95 ± 2.05 | 0/3 | yes | 3 |
| entropy − prob_mean | +0.64 ± 1.31 | 2/1 | no | 1 |
| entropy − combined | +0.32 ± 0.36 | 2/1 | no | 0 |
| combined − best_single:raw_qwen | +0.82 ± 0.70 | 3/0 | yes | 1 |
| prior_matched_qwen − best_single:raw_qwen | +1.58 ± 0.80 | 3/0 | yes | 2 |

## amazon · L=50

Best single teacher: **raw_qwen**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 55.81% ± 0.92 | 54.28% ± 0.85 | 54.76%, 56.15%, 56.51% |
| mistral | 53.73% ± 0.58 | 51.10% ± 1.67 | 54.12%, 54.01%, 53.06% |
| entropy | 56.20% ± 1.21 | 54.57% ± 2.23 | 56.20%, 57.40%, 54.99% |
| raw_qwen | 57.32% ± 1.05 | 55.26% ± 2.80 | 58.53%, 56.86%, 56.58% |
| prior_matched_qwen | 57.33% ± 1.23 | 57.36% ± 0.81 | 56.01%, 57.54%, 58.44% |
| prob_mean | 56.45% ± 0.26 | 55.13% ± 1.04 | 56.45%, 56.71%, 56.20% |
| combined | 53.20% ± 1.51 | 52.27% ± 1.44 | 51.50%, 53.70%, 54.40% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | -1.52 ± 1.98 | 0/3 | yes | 1 |
| mistral − raw_qwen | -3.59 ± 0.78 | 0/3 | yes | 3 |
| entropy − prob_mean | -0.26 ± 0.95 | 1/2 | no | 1 |
| entropy − combined | +3.00 ± 2.14 | 3/0 | yes | 2 |
| combined − best_single:raw_qwen | -4.12 ± 2.56 | 0/3 | yes | 3 |
| prior_matched_qwen − best_single:raw_qwen | +0.01 ± 2.27 | 2/1 | no | 2 |

## amazon3 · L=10

Best single teacher: **raw_qwen**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 69.43% ± 2.88 | 66.03% ± 4.01 | 66.17%, 70.50%, 71.62% |
| mistral | 67.97% ± 0.68 | 63.68% ± 1.82 | 67.93%, 67.32%, 68.67% |
| entropy | 70.37% ± 1.70 | 67.23% ± 3.01 | 69.03%, 69.80%, 72.28% |
| raw_qwen | 72.11% ± 0.99 | 70.30% ± 1.34 | 71.22%, 71.93%, 73.17% |
| prior_matched_qwen | 74.06% ± 1.17 | 73.53% ± 1.84 | 74.68%, 72.72%, 74.78% |
| prob_mean | 72.24% ± 0.75 | 70.17% ± 1.41 | 72.22%, 71.50%, 73.00% |
| combined | 73.79% ± 0.47 | 73.48% ± 0.86 | 74.08%, 73.25%, 74.05% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | -2.68 ± 2.06 | 0/3 | yes | 3 |
| mistral − raw_qwen | -4.13 ± 0.74 | 0/3 | yes | 3 |
| entropy − prob_mean | -1.87 ± 1.24 | 0/3 | yes | 3 |
| entropy − combined | -3.42 ± 1.64 | 0/3 | yes | 3 |
| combined − best_single:raw_qwen | +1.69 ± 1.04 | 3/0 | yes | 2 |
| prior_matched_qwen − best_single:raw_qwen | +1.96 ± 1.37 | 3/0 | yes | 3 |

## amazon3 · L=50

Best single teacher: **raw_qwen**

| Method | Accuracy | Macro-F1 | Seed accuracies |
|---|---:|---:|---|
| gemma | 68.81% ± 2.87 | 66.49% ± 3.16 | 71.45%, 69.22%, 65.75% |
| mistral | 72.36% ± 0.22 | 71.31% ± 0.17 | 72.58%, 72.15%, 72.33% |
| entropy | 72.37% ± 1.07 | 71.25% ± 0.98 | 73.42%, 72.42%, 71.28% |
| raw_qwen | 73.76% ± 0.24 | 73.58% ± 0.24 | 73.50%, 73.97%, 73.80% |
| prior_matched_qwen | 73.24% ± 2.71 | 73.28% ± 2.53 | 74.38%, 75.20%, 70.15% |
| prob_mean | 72.99% ± 0.72 | 72.12% ± 1.06 | 72.35%, 72.85%, 73.77% |
| combined | 72.84% ± 1.57 | 72.93% ± 1.52 | 74.02%, 73.45%, 71.07% |

| Comparison | Mean Δacc (pp) | Seed wins | All same sign | Holm-sig seeds |
|---|---:|---:|---|---:|
| gemma − raw_qwen | -4.95 ± 3.00 | 0/3 | yes | 3 |
| mistral − raw_qwen | -1.40 ± 0.45 | 0/3 | yes | 2 |
| entropy − prob_mean | -0.62 ± 1.78 | 1/2 | no | 2 |
| entropy − combined | -0.47 ± 0.63 | 1/2 | no | 0 |
| combined − best_single:raw_qwen | -0.91 ± 1.66 | 1/2 | no | 1 |
| prior_matched_qwen − best_single:raw_qwen | -0.51 ± 2.72 | 2/1 | no | 3 |

