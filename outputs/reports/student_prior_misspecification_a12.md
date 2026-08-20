# Student-level A.12 prior-misspecification experiment

We retrained students using the mild and strong misspecification strengths from Table A.12, with class 0 predeclared as the dominant assumed class. The actual training pool and the fixed balanced test set are identical across conditions; only the prior supplied to prior-and-scale-aligned matching changes. This is an assumed-prior sensitivity experiment on balanced pools, not a label-shift, unbalanced-pool, covariate-shift, or domain-shift experiment.

> The student experiment uses one predeclared misspecification direction on three balanced datasets; it does not average over all possible dominant classes.

Seeds: 42, 123, 456. Grid: 3 datasets x 2 label budgets x 3 seeds x 3 conditions = 54 cells (18 pre-existing correct-prior baselines + 36 new mild/strong cells).

Primary comparisons: mild − correct, strong − correct. Secondary: strong − mild. With three seeds, effect sizes/direction/dataset- and budget-dependence are emphasised over seed-level significance.

## ag_news · L=10

### Condition summary (accuracy / macro-F1, student + teacher-target)

| condition | student acc mean±SD | student F1 mean±SD | teacher-target acc | teacher-target NLL |
|---|---:|---:|---:|---:|
| correct | 88.47% ± 0.18% | 88.44% ± 0.19% | 89.93% | 0.4073 |
| mild | 84.42% ± 1.12% | 84.47% ± 1.09% | 84.83% | 0.5121 |
| strong | 64.20% ± 5.17% | 64.77% ± 4.96% | 63.58% | 0.8727 |

### Paired comparisons (per seed)

#### mild − correct

- Mean paired Δacc: -4.05 ± 1.03 (descriptive SE 0.59); per-seed: -4.55, -4.72, -2.87
- Wins/ties/losses (mild vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 84.04% | 88.59% | -4.55 | 572 | 113 | 459 | 2.071e-50 | 4.143e-50 |
| 123 | 83.54% | 88.26% | -4.72 | 565 | 103 | 462 | 3.064e-55 | 9.193e-55 |
| 456 | 85.68% | 88.55% | -2.87 | 428 | 105 | 323 | 6.931e-27 | 6.931e-27 |

#### strong − correct

- Mean paired Δacc: -24.27 ± 5.08 (descriptive SE 2.94); per-seed: -26.95, -27.46, -18.41
- Wins/ties/losses (strong vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 61.64% | 88.59% | -26.95 | 2412 | 182 | 2230 | 0 | 0 |
| 123 | 60.80% | 88.26% | -27.46 | 2433 | 173 | 2260 | 0 | 0 |
| 456 | 70.14% | 88.55% | -18.41 | 1755 | 178 | 1577 | 4.955e-280 | 4.955e-280 |

#### strong − mild (secondary)

- Mean paired Δacc: -20.22 ± 4.06 (descriptive SE 2.34); per-seed: -22.39, -22.74, -15.54
- Wins/ties/losses (strong vs mild): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 61.64% | 84.04% | -22.39 | 1882 | 90 | 1792 | 0 | 0 |
| 123 | 60.80% | 83.54% | -22.74 | 1900 | 86 | 1814 | 0 | 0 |
| 456 | 70.14% | 85.68% | -15.54 | 1355 | 87 | 1268 | 2.321e-269 | 2.321e-269 |

## ag_news · L=50

### Condition summary (accuracy / macro-F1, student + teacher-target)

| condition | student acc mean±SD | student F1 mean±SD | teacher-target acc | teacher-target NLL |
|---|---:|---:|---:|---:|
| correct | 88.30% ± 0.32% | 88.27% ± 0.33% | 89.93% | 0.4073 |
| mild | 84.48% ± 0.35% | 84.53% ± 0.36% | 84.83% | 0.5121 |
| strong | 67.89% ± 3.03% | 68.37% ± 2.87% | 63.58% | 0.8727 |

### Paired comparisons (per seed)

#### mild − correct

- Mean paired Δacc: -3.82 ± 0.09 (descriptive SE 0.05); per-seed: -3.92, -3.78, -3.76
- Wins/ties/losses (mild vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 84.30% | 88.22% | -3.92 | 550 | 126 | 424 | 1.104e-38 | 2.208e-38 |
| 123 | 84.25% | 88.03% | -3.78 | 557 | 135 | 422 | 2.257e-35 | 2.257e-35 |
| 456 | 84.88% | 88.64% | -3.76 | 506 | 110 | 396 | 6.498e-39 | 1.949e-38 |

#### strong − correct

- Mean paired Δacc: -20.41 ± 3.25 (descriptive SE 1.87); per-seed: -22.50, -16.67, -22.07
- Wins/ties/losses (strong vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 65.72% | 88.22% | -22.50 | 2024 | 157 | 1867 | 0 | 0 |
| 123 | 71.36% | 88.03% | -16.67 | 1575 | 154 | 1421 | 5.758e-257 | 5.758e-257 |
| 456 | 66.58% | 88.64% | -22.07 | 2027 | 175 | 1852 | 0 | 0 |

#### strong − mild (secondary)

- Mean paired Δacc: -16.59 ± 3.20 (descriptive SE 1.85); per-seed: -18.58, -12.89, -18.30
- Wins/ties/losses (strong vs mild): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 65.72% | 84.30% | -18.58 | 1544 | 66 | 1478 | 0 | 0 |
| 123 | 71.36% | 84.25% | -12.89 | 1102 | 61 | 1041 | 5.308e-231 | 5.308e-231 |
| 456 | 66.58% | 84.88% | -18.30 | 1555 | 82 | 1473 | 0 | 0 |

## amazon · L=10

### Condition summary (accuracy / macro-F1, student + teacher-target)

| condition | student acc mean±SD | student F1 mean±SD | teacher-target acc | teacher-target NLL |
|---|---:|---:|---:|---:|
| correct | 58.18% ± 0.32% | 58.02% ± 0.61% | 65.88% | 0.9908 |
| mild | 54.27% ± 1.09% | 52.81% ± 1.74% | 62.80% | 1.0468 |
| strong | 35.51% ± 3.21% | 31.33% ± 4.43% | 43.08% | 1.2295 |

### Paired comparisons (per seed)

#### mild − correct

- Mean paired Δacc: -3.91 ± 1.37 (descriptive SE 0.79); per-seed: -3.63, -5.40, -2.70
- Wins/ties/losses (mild vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 54.35% | 57.98% | -3.63 | 1663 | 650 | 1013 | 5.006e-19 | 1.001e-18 |
| 123 | 53.14% | 58.54% | -5.40 | 2188 | 824 | 1364 | 5.077e-31 | 1.523e-30 |
| 456 | 55.31% | 58.01% | -2.70 | 1954 | 842 | 1112 | 1.096e-09 | 1.096e-09 |

#### strong − correct

- Mean paired Δacc: -22.66 ± 3.48 (descriptive SE 2.01); per-seed: -22.50, -26.22, -19.27
- Wins/ties/losses (strong vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 35.48% | 57.98% | -22.50 | 3408 | 579 | 2829 | 0 | 0 |
| 123 | 32.32% | 58.54% | -26.22 | 3924 | 651 | 3273 | 0 | 0 |
| 456 | 38.74% | 58.01% | -19.27 | 3699 | 886 | 2813 | 3.507e-231 | 3.507e-231 |

#### strong − mild (secondary)

- Mean paired Δacc: -18.75 ± 2.13 (descriptive SE 1.23); per-seed: -18.87, -20.82, -16.57
- Wins/ties/losses (strong vs mild): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 35.48% | 54.35% | -18.87 | 2315 | 214 | 2101 | 0 | 0 |
| 123 | 32.32% | 53.14% | -20.82 | 2630 | 274 | 2356 | 0 | 0 |
| 456 | 38.74% | 55.31% | -16.57 | 2545 | 444 | 2101 | 1.666e-256 | 1.666e-256 |

## amazon · L=50

### Condition summary (accuracy / macro-F1, student + teacher-target)

| condition | student acc mean±SD | student F1 mean±SD | teacher-target acc | teacher-target NLL |
|---|---:|---:|---:|---:|
| correct | 53.20% ± 1.51% | 52.27% ± 1.44% | 65.88% | 0.9908 |
| mild | 53.24% ± 3.06% | 52.11% ± 3.50% | 62.80% | 1.0468 |
| strong | 45.01% ± 5.54% | 41.72% ± 7.19% | 43.08% | 1.2295 |

### Paired comparisons (per seed)

#### mild − correct

- Mean paired Δacc: +0.04 ± 3.11 (descriptive SE 1.80); per-seed: +1.81, -3.55, +1.87
- Wins/ties/losses (mild vs correct): 2/0/1

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 53.31% | 51.50% | +1.81 | 1687 | 934 | 753 | 1.152e-05 | 2.305e-05 |
| 123 | 50.15% | 53.70% | -3.55 | 1995 | 820 | 1175 | 1.922e-15 | 5.765e-15 |
| 456 | 56.27% | 54.40% | +1.87 | 1915 | 1051 | 864 | 2.104e-05 | 2.305e-05 |

#### strong − correct

- Mean paired Δacc: -8.19 ± 6.90 (descriptive SE 3.98); per-seed: -0.33, -13.26, -10.98
- Wins/ties/losses (strong vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 51.17% | 51.50% | -0.33 | 2655 | 1311 | 1344 | 0.5346 | 0.5346 |
| 123 | 40.44% | 53.70% | -13.26 | 2950 | 812 | 2138 | 3.504e-136 | 1.051e-135 |
| 456 | 43.42% | 54.40% | -10.98 | 3100 | 1001 | 2099 | 2.862e-88 | 5.724e-88 |

#### strong − mild (secondary)

- Mean paired Δacc: -8.23 ± 5.51 (descriptive SE 3.18); per-seed: -2.14, -9.71, -12.85
- Wins/ties/losses (strong vs mild): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 51.17% | 53.31% | -2.14 | 1604 | 695 | 909 | 1.004e-07 | 1.004e-07 |
| 123 | 40.44% | 50.15% | -9.71 | 1845 | 437 | 1408 | 1.1e-118 | 2.2e-118 |
| 456 | 43.42% | 56.27% | -12.85 | 2243 | 479 | 1764 | 5.582e-172 | 1.675e-171 |

## yahoo · L=10

### Condition summary (accuracy / macro-F1, student + teacher-target)

| condition | student acc mean±SD | student F1 mean±SD | teacher-target acc | teacher-target NLL |
|---|---:|---:|---:|---:|
| correct | 61.60% ± 0.13% | 61.40% ± 0.19% | 66.34% | 1.2621 |
| mild | 58.04% ± 0.75% | 59.73% ± 0.64% | 64.14% | 1.3169 |
| strong | 39.52% ± 3.02% | 45.09% ± 2.93% | 48.14% | 1.5698 |

### Paired comparisons (per seed)

#### mild − correct

- Mean paired Δacc: -3.56 ± 0.76 (descriptive SE 0.44); per-seed: -3.20, -3.04, -4.43
- Wins/ties/losses (mild vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 58.52% | 61.72% | -3.20 | 988 | 334 | 654 | 1.319e-24 | 2.638e-24 |
| 123 | 58.43% | 61.47% | -3.04 | 1002 | 349 | 653 | 5.073e-22 | 5.073e-22 |
| 456 | 57.17% | 61.60% | -4.43 | 1313 | 435 | 878 | 7.301e-35 | 2.19e-34 |

#### strong − correct

- Mean paired Δacc: -22.08 ± 3.06 (descriptive SE 1.77); per-seed: -21.23, -19.53, -25.47
- Wins/ties/losses (strong vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 40.49% | 61.72% | -21.23 | 3079 | 478 | 2601 | 0 | 0 |
| 123 | 41.94% | 61.47% | -19.53 | 2929 | 488 | 2441 | 9.916e-311 | 9.916e-311 |
| 456 | 36.13% | 61.60% | -25.47 | 3573 | 513 | 3060 | 0 | 0 |

#### strong − mild (secondary)

- Mean paired Δacc: -18.52 ± 2.31 (descriptive SE 1.34); per-seed: -18.03, -16.49, -21.04
- Wins/ties/losses (strong vs mild): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 40.49% | 58.52% | -18.03 | 2267 | 232 | 2035 | 0 | 0 |
| 123 | 41.94% | 58.43% | -16.49 | 2065 | 208 | 1857 | 0 | 0 |
| 456 | 36.13% | 57.17% | -21.04 | 2494 | 195 | 2299 | 0 | 0 |

## yahoo · L=50

### Condition summary (accuracy / macro-F1, student + teacher-target)

| condition | student acc mean±SD | student F1 mean±SD | teacher-target acc | teacher-target NLL |
|---|---:|---:|---:|---:|
| correct | 60.57% ± 0.64% | 60.50% ± 0.66% | 66.34% | 1.2621 |
| mild | 58.97% ± 1.59% | 59.92% ± 1.10% | 64.14% | 1.3169 |
| strong | 47.98% ± 5.92% | 52.23% ± 5.72% | 48.14% | 1.5698 |

### Paired comparisons (per seed)

#### mild − correct

- Mean paired Δacc: -1.60 ± 1.31 (descriptive SE 0.75); per-seed: -1.34, -3.02, -0.45
- Wins/ties/losses (mild vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 59.97% | 61.31% | -1.34 | 924 | 395 | 529 | 1.173e-05 | 2.346e-05 |
| 123 | 57.13% | 60.15% | -3.02 | 1142 | 420 | 722 | 3.268e-19 | 9.805e-19 |
| 456 | 59.80% | 60.25% | -0.45 | 987 | 471 | 516 | 0.1613 | 0.1613 |

#### strong − correct

- Mean paired Δacc: -12.59 ± 5.64 (descriptive SE 3.26); per-seed: -10.56, -18.96, -8.24
- Wins/ties/losses (strong vs correct): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 50.75% | 61.31% | -10.56 | 2064 | 504 | 1560 | 5.2e-125 | 1.04e-124 |
| 123 | 41.19% | 60.15% | -18.96 | 2788 | 446 | 2342 | 5.438e-309 | 1.631e-308 |
| 456 | 52.01% | 60.25% | -8.24 | 1858 | 517 | 1341 | 4.399e-84 | 4.399e-84 |

#### strong − mild (secondary)

- Mean paired Δacc: -10.98 ± 4.35 (descriptive SE 2.51); per-seed: -9.22, -15.94, -7.79
- Wins/ties/losses (strong vs mild): 0/0/3

| seed | A acc | B acc | Δacc | discordant | only_A | only_B | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 50.75% | 59.97% | -9.22 | 1508 | 293 | 1215 | 2.283e-133 | 4.565e-133 |
| 123 | 41.19% | 57.13% | -15.94 | 2100 | 253 | 1847 | 1.367e-298 | 4.1e-298 |
| 456 | 52.01% | 59.80% | -7.79 | 1345 | 283 | 1062 | 3.219e-106 | 3.219e-106 |

## Completeness / provenance

- Cells found matching this experiment's teacher: 60
- Missing cells: 0
- Duplicate cells: 0
- Misplaced cells (wrong source directory for their condition): 0
- Unexpected cells (out-of-scope dataset/budget/seed/condition, excluded from the 54-cell grid): 6
  - ('amazon3', 10, 123, 'correct', 0), ('amazon3', 10, 42, 'correct', 0), ('amazon3', 10, 456, 'correct', 0), ('amazon3', 50, 123, 'correct', 0), ('amazon3', 50, 42, 'correct', 0), ('amazon3', 50, 456, 'correct', 0)
- Baseline source: `outputs/student/prompt_inclusive_core_3seed_v1` (18 correct-prior cells, not retrained)
- New-cell source: `outputs/student/prior_misspecification_a12_v1` (36 mild/strong cells)
- Predeclared misspecification direction: class 0 (fixed before examining results)
- No result here is an average over dominant classes; this is a single predeclared direction.
