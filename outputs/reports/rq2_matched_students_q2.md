# RQ2 matched-budget student results (q/N = 2.0)

Cascade (`multi-cascade-guided`) versus uniform allocation (`multi-uniform-budget-guided`) at matched teacher-query budget.

Paired McNemar tests are computed **per training seed** on the fixed `large-test-v1` test rows. Holm-adjusted p-values are reported within each dataset×L cell (5 seeds). With only five seeds, seed-level deltas are the primary evidence.

## ag_news · L=10

- Cascade accuracy: 88.38% ± 0.30
- Uniform accuracy: 88.65% ± 0.13
- Cascade macro-F1: 88.36% ± 0.31
- Uniform macro-F1: 88.63% ± 0.14
- Mean paired Δacc (cascade−uniform): -0.28 ± 0.19 (SE 0.11)
- Seed wins (accuracy): cascade=0, uniform=3, tie=0
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.64% | 88.72% | -0.08 | 0.6398 | 0.6398 | uniform |
| 123 | 88.05% | 88.50% | -0.45 | 0.004481 | 0.01344 | uniform |
| 456 | 88.43% | 88.74% | -0.30 | 0.05232 | 0.1046 | uniform |

## ag_news · L=50

- Cascade accuracy: 88.43% ± 0.32
- Uniform accuracy: 88.46% ± 0.20
- Cascade macro-F1: 88.40% ± 0.34
- Uniform macro-F1: 88.44% ± 0.20
- Mean paired Δacc (cascade−uniform): -0.03 ± 0.12 (SE 0.07)
- Seed wins (accuracy): cascade=1, uniform=2, tie=0
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 88.45% | 88.46% | -0.01 | 1 | 1 | uniform |
| 123 | 88.09% | 88.25% | -0.16 | 0.4703 | 1 | uniform |
| 456 | 88.74% | 88.66% | +0.08 | 0.6241 | 1 | cascade |

## yahoo · L=10

- Cascade accuracy: 61.36% ± 0.61
- Uniform accuracy: 60.88% ± 0.48
- Cascade macro-F1: 61.15% ± 0.64
- Uniform macro-F1: 60.76% ± 0.52
- Mean paired Δacc (cascade−uniform): +0.48 ± 0.13 (SE 0.08)
- Seed wins (accuracy): cascade=3, uniform=0, tie=0
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 62.06% | 61.43% | +0.63 | 0.003178 | 0.009533 | cascade |
| 123 | 60.95% | 60.56% | +0.39 | 0.1148 | 0.1794 | cascade |
| 456 | 61.06% | 60.64% | +0.42 | 0.08969 | 0.1794 | cascade |

## yahoo · L=50

- Cascade accuracy: 60.15% ± 0.39
- Uniform accuracy: 60.42% ± 0.26
- Cascade macro-F1: 60.21% ± 0.28
- Uniform macro-F1: 60.24% ± 0.34
- Mean paired Δacc (cascade−uniform): -0.26 ± 0.62 (SE 0.36)
- Seed wins (accuracy): cascade=1, uniform=2, tie=0
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 60.60% | 60.16% | +0.44 | 0.0876 | 0.1752 | cascade |
| 123 | 59.90% | 60.41% | -0.51 | 0.09387 | 0.1752 | uniform |
| 456 | 59.96% | 60.68% | -0.72 | 0.01117 | 0.03352 | uniform |

## amazon · L=10

- Cascade accuracy: 58.20% ± 0.38
- Uniform accuracy: 58.48% ± 0.42
- Cascade macro-F1: 58.34% ± 0.49
- Uniform macro-F1: 58.36% ± 0.65
- Mean paired Δacc (cascade−uniform): -0.28 ± 0.15 (SE 0.09)
- Seed wins (accuracy): cascade=0, uniform=3, tie=0
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 57.76% | 58.06% | -0.30 | 0.3758 | 0.7515 | uniform |
| 123 | 58.37% | 58.49% | -0.12 | 0.7293 | 0.7515 | uniform |
| 456 | 58.47% | 58.89% | -0.42 | 0.1502 | 0.4505 | uniform |

## amazon · L=50

- Cascade accuracy: 53.77% ± 2.13
- Uniform accuracy: 55.06% ± 2.04
- Cascade macro-F1: 53.22% ± 1.79
- Uniform macro-F1: 54.88% ± 1.60
- Mean paired Δacc (cascade−uniform): -1.29 ± 0.43 (SE 0.25)
- Seed wins (accuracy): cascade=0, uniform=3, tie=0
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 52.37% | 53.36% | -0.99 | 0.005725 | 0.005725 | uniform |
| 123 | 52.71% | 54.49% | -1.78 | 8.151e-07 | 2.445e-06 | uniform |
| 456 | 56.22% | 57.32% | -1.10 | 0.0002049 | 0.0004099 | uniform |

## amazon3 · L=10

- Cascade accuracy: 73.46% ± 0.37
- Uniform accuracy: 73.54% ± 0.65
- Cascade macro-F1: 73.24% ± 0.80
- Uniform macro-F1: 73.00% ± 1.29
- Mean paired Δacc (cascade−uniform): -0.08 ± 0.45 (SE 0.26)
- Seed wins (accuracy): cascade=1, uniform=1, tie=1
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 73.85% | 73.85% | +0.00 | 1 | 1 | tie |
| 123 | 73.12% | 72.80% | +0.32 | 0.3074 | 0.6148 | cascade |
| 456 | 73.42% | 73.98% | -0.57 | 0.05988 | 0.1797 | uniform |

## amazon3 · L=50

- Cascade accuracy: 71.71% ± 3.32
- Uniform accuracy: 72.61% ± 1.78
- Cascade macro-F1: 71.69% ± 3.57
- Uniform macro-F1: 72.62% ± 1.77
- Mean paired Δacc (cascade−uniform): -0.90 ± 1.62 (SE 0.93)
- Seed wins (accuracy): cascade=1, uniform=2, tie=0
- Realised q/N: cascade≈2.0000, uniform=2.0000

| seed | cascade acc | uniform acc | Δacc | McNemar p | Holm p | winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 74.08% | 73.52% | +0.57 | 0.09035 | 0.1807 | cascade |
| 123 | 73.12% | 73.75% | -0.63 | 0.09048 | 0.1807 | uniform |
| 456 | 67.92% | 70.55% | -2.63 | 1.067e-15 | 3.2e-15 | uniform |

