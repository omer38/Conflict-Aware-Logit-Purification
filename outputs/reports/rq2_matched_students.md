# RQ2 matched-budget student results (q/N = 1.4)

Cascade (`multi-cascade-guided`) versus uniform allocation (`multi-uniform-budget-guided`) at matched teacher-query budget.

Paired McNemar tests are computed **per training seed** on the fixed `large-test-v1` test rows. Holm-adjusted p-values are reported within each dataset×L cell (5 seeds). With only five seeds, seed-level deltas are the primary evidence.

## ag_news · L=10

- Cascade accuracy: 88.34% ± 0.13
- Uniform accuracy: 88.62% ± 0.16
- Cascade macro-F1: 88.32% ± 0.13
- Uniform macro-F1: 88.61% ± 0.16
- Mean paired Δacc (cascade−uniform): -0.28 ± 0.18 (SE 0.08)
- Seed wins (accuracy): cascade=0, uniform=5, tie=0
- Realised q/N: cascade≈1.4002, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.53% | 88.79% | -0.26 | 0.1519 | 0.4339 | uniform |
| 123 | 88.26% | 88.70% | -0.43 | 0.01304 | 0.05216 | uniform |
| 456 | 88.28% | 88.72% | -0.45 | 0.002662 | 0.01331 | uniform |
| 789 | 88.43% | 88.45% | -0.01 | 1 | 1 | uniform |
| 2024 | 88.22% | 88.46% | -0.24 | 0.1446 | 0.4339 | uniform |

## ag_news · L=50

- Cascade accuracy: 88.56% ± 0.24
- Uniform accuracy: 88.75% ± 0.15
- Cascade macro-F1: 88.54% ± 0.24
- Uniform macro-F1: 88.73% ± 0.16
- Mean paired Δacc (cascade−uniform): -0.19 ± 0.19 (SE 0.09)
- Seed wins (accuracy): cascade=1, uniform=4, tie=0
- Realised q/N: cascade≈1.4002, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.62% | 88.61% | +0.01 | 1 | 1 | cascade |
| 123 | 88.43% | 88.61% | -0.17 | 0.3156 | 0.9469 | uniform |
| 456 | 88.34% | 88.78% | -0.43 | 0.0125 | 0.06248 | uniform |
| 789 | 88.47% | 88.80% | -0.33 | 0.03703 | 0.1481 | uniform |
| 2024 | 88.95% | 88.96% | -0.01 | 1 | 1 | uniform |

## yahoo · L=10

- Cascade accuracy: 60.86% ± 0.43
- Uniform accuracy: 60.86% ± 0.34
- Cascade macro-F1: 60.72% ± 0.44
- Uniform macro-F1: 60.71% ± 0.56
- Mean paired Δacc (cascade−uniform): -0.00 ± 0.50 (SE 0.22)
- Seed wins (accuracy): cascade=2, uniform=3, tie=0
- Realised q/N: cascade≈1.4000, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.57% | 61.11% | +0.46 | 0.0295 | 0.0885 | cascade |
| 123 | 60.53% | 61.19% | -0.66 | 0.004321 | 0.02161 | uniform |
| 456 | 60.81% | 60.95% | -0.14 | 0.5801 | 0.791 | uniform |
| 789 | 60.89% | 60.36% | +0.53 | 0.02058 | 0.08231 | cascade |
| 2024 | 60.51% | 60.71% | -0.20 | 0.3955 | 0.791 | uniform |

## yahoo · L=50

- Cascade accuracy: 60.46% ± 0.56
- Uniform accuracy: 60.32% ± 0.46
- Cascade macro-F1: 60.36% ± 0.59
- Uniform macro-F1: 60.23% ± 0.50
- Mean paired Δacc (cascade−uniform): +0.13 ± 0.29 (SE 0.13)
- Seed wins (accuracy): cascade=3, uniform=2, tie=0
- Realised q/N: cascade≈1.4000, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 61.25% | 60.69% | +0.56 | 0.01765 | 0.08824 | cascade |
| 123 | 60.06% | 60.21% | -0.15 | 0.585 | 1 | uniform |
| 456 | 60.15% | 60.22% | -0.07 | 0.8267 | 1 | uniform |
| 789 | 60.84% | 60.82% | +0.02 | 0.9694 | 1 | cascade |
| 2024 | 59.98% | 59.67% | +0.31 | 0.2301 | 0.9204 | cascade |

## amazon · L=10

- Cascade accuracy: 57.69% ± 1.01
- Uniform accuracy: 57.43% ± 1.56
- Cascade macro-F1: 57.58% ± 1.60
- Uniform macro-F1: 57.36% ± 1.83
- Mean paired Δacc (cascade−uniform): +0.26 ± 0.65 (SE 0.29)
- Seed wins (accuracy): cascade=4, uniform=1, tie=0
- Realised q/N: cascade≈1.4004, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 58.21% | 58.13% | +0.08 | 0.806 | 1 | cascade |
| 123 | 58.77% | 58.69% | +0.08 | 0.8196 | 1 | cascade |
| 456 | 57.82% | 58.36% | -0.54 | 0.0904 | 0.3616 | uniform |
| 789 | 56.07% | 54.84% | +1.23 | 0.001312 | 0.006559 | cascade |
| 2024 | 57.58% | 57.12% | +0.46 | 0.1522 | 0.4566 | cascade |

## amazon · L=50

- Cascade accuracy: 54.97% ± 1.13
- Uniform accuracy: 55.11% ± 1.81
- Cascade macro-F1: 54.10% ± 1.32
- Uniform macro-F1: 54.48% ± 1.57
- Mean paired Δacc (cascade−uniform): -0.14 ± 1.28 (SE 0.57)
- Seed wins (accuracy): cascade=2, uniform=3, tie=0
- Realised q/N: cascade≈1.4004, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 53.61% | 53.40% | +0.21 | 0.5441 | 0.7581 | cascade |
| 123 | 54.70% | 52.92% | +1.78 | 1.255e-07 | 6.273e-07 | cascade |
| 456 | 56.02% | 56.74% | -0.72 | 0.03592 | 0.1078 | uniform |
| 789 | 56.24% | 56.55% | -0.31 | 0.379 | 0.7581 | uniform |
| 2024 | 54.26% | 55.94% | -1.68 | 9.369e-05 | 0.0003748 | uniform |

## amazon3 · L=10

- Cascade accuracy: 74.03% ± 0.87
- Uniform accuracy: 74.23% ± 0.78
- Cascade macro-F1: 73.74% ± 1.42
- Uniform macro-F1: 73.98% ± 1.25
- Mean paired Δacc (cascade−uniform): -0.20 ± 0.44 (SE 0.20)
- Seed wins (accuracy): cascade=2, uniform=3, tie=0
- Realised q/N: cascade≈1.4000, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.12% | 73.70% | +0.42 | 0.1975 | 0.5926 | cascade |
| 123 | 72.62% | 73.35% | -0.73 | 0.01032 | 0.05161 | uniform |
| 456 | 73.97% | 74.22% | -0.25 | 0.443 | 0.886 | uniform |
| 789 | 74.55% | 74.53% | +0.02 | 1 | 1 | cascade |
| 2024 | 74.90% | 75.37% | -0.47 | 0.0786 | 0.3144 | uniform |

## amazon3 · L=50

- Cascade accuracy: 71.58% ± 1.34
- Uniform accuracy: 71.59% ± 2.12
- Cascade macro-F1: 71.24% ± 1.67
- Uniform macro-F1: 71.08% ± 3.21
- Mean paired Δacc (cascade−uniform): -0.01 ± 1.52 (SE 0.68)
- Seed wins (accuracy): cascade=3, uniform=2, tie=0
- Realised q/N: cascade≈1.4000, uniform=1.4000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 73.87% | 73.73% | +0.13 | 0.6779 | 1 | cascade |
| 123 | 71.37% | 73.75% | -2.38 | 8.413e-08 | 4.206e-07 | uniform |
| 456 | 71.33% | 71.38% | -0.05 | 0.9094 | 1 | uniform |
| 789 | 71.00% | 69.17% | +1.83 | 2.996e-06 | 1.198e-05 | cascade |
| 2024 | 70.35% | 69.92% | +0.43 | 0.1863 | 0.559 | cascade |

