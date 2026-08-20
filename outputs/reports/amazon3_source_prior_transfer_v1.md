# Amazon-3 source-prior transfer experiment

**Student-level replication under a source-induced non-uniform prior.** The working pool has a 40/20/40 class prior induced by grouping the balanced, gold-constructed Amazon-5 benchmark (1-2 stars -> negative, 3 stars -> neutral, 4-5 stars -> positive); it is **not** natural deployment prevalence. The prior is estimated with a constrained soft-BBSE estimator from a fixed committee predictor on a separate 500-row unlabelled reference batch, then alignment parameters (class-prior bias, confidence scale) are fit on that reference batch and transferred out of sample to a disjoint 2,000-row student working pool. No additional teacher queries and no labels beyond the declared per-class budget were used.

Grid: 4 methods x 2 label budgets x 3 seeds = 24 cells. Seeds: 42, 123, 456.

Primary comparison: `estimated_prior_bbse - scale_only` (macro-F1 primary, accuracy secondary). Secondary: estimated vs uniform, oracle vs estimated (prior-estimation gap), uniform vs scale_only (cost of a wrong prior). With three seeds, effect direction/size/consistency are emphasised over seed-level statistical significance.

## L=10

### Method summary (mean +/- SD across 3 seeds)

| method | accuracy | macro-F1 |
|---|---:|---:|
| scale_only | 78.56% +/- 0.34% | 70.99% +/- 0.71% |
| uniform_prior | 74.53% +/- 1.60% | 72.49% +/- 0.99% |
| estimated_prior_bbse | 77.57% +/- 2.07% | 65.24% +/- 8.25% |
| oracle_prior | 79.00% +/- 0.31% | 73.66% +/- 0.35% |

### Prior estimates by seed (estimated_prior_bbse)

| seed | estimated prior | L1 error vs true | working-marginal L1 error |
|---:|---|---:|---:|
| 42 | [0.439, 0.135, 0.426] | 0.1301 | 0.1023 |
| 123 | [0.509, 0.000, 0.491] | 0.4000 | 0.3978 |
| 456 | [0.410, 0.134, 0.457] | 0.1327 | 0.1172 |

### Paired comparisons (per seed)

#### estimated_prior_bbse - scale_only

- Mean paired delta-macroF1: -5.76 +/- 8.17; per-seed: -0.74, -15.18, -1.34
- Mean paired delta-acc: -0.98 +/- 1.72; per-seed: +0.08, -2.97, -0.06
- Wins/ties/losses (accuracy, estimated_prior_bbse vs scale_only): 1/0/2

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | -0.74 | [-1.36, -0.13] | +0.08 | 0.7224 | 1 |
| 123 | -15.18 | [-16.10, -14.32] | -2.97 | 1.246e-17 | 3.738e-17 |
| 456 | -1.34 | [-1.93, -0.79] | -0.06 | 0.8199 | 1 |

#### estimated_prior_bbse - uniform_prior

- Mean paired delta-macroF1: -7.25 +/- 7.40; per-seed: -3.86, -15.74, -2.17
- Mean paired delta-acc: +3.04 +/- 0.98; per-seed: +2.73, +2.25, +4.14
- Wins/ties/losses (accuracy, estimated_prior_bbse vs uniform_prior): 3/0/0

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | -3.86 | [-4.87, -2.92] | +2.73 | 2.622e-10 | 5.245e-10 |
| 123 | -15.74 | [-16.51, -14.96] | +2.25 | 8.28e-05 | 8.28e-05 |
| 456 | -2.17 | [-3.14, -1.26] | +4.14 | 8.771e-20 | 2.631e-19 |

#### oracle_prior - estimated_prior_bbse

- Mean paired delta-macroF1: +8.42 +/- 8.16; per-seed: +3.77, +17.85, +3.65
- Mean paired delta-acc: +1.43 +/- 1.86; per-seed: +0.49, +3.57, +0.22
- Wins/ties/losses (accuracy, oracle_prior vs estimated_prior_bbse): 3/0/0

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | +3.77 | [+3.04, +4.50] | +0.49 | 0.0575 | 0.115 |
| 123 | +17.85 | [+16.98, +18.73] | +3.57 | 4.786e-19 | 1.436e-18 |
| 456 | +3.65 | [+2.90, +4.39] | +0.22 | 0.4745 | 0.4745 |

#### uniform_prior - scale_only

- Mean paired delta-macroF1: +1.50 +/- 1.41; per-seed: +3.12, +0.56, +0.83
- Mean paired delta-acc: -4.02 +/- 1.29; per-seed: -2.65, -5.22, -4.20
- Wins/ties/losses (accuracy, uniform_prior vs scale_only): 0/0/3

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | +3.12 | [+2.22, +4.11] | -2.65 | 1.569e-10 | 1.569e-10 |
| 123 | +0.56 | [-0.37, +1.50] | -5.22 | 9.362e-30 | 2.809e-29 |
| 456 | +0.83 | [-0.08, +1.79] | -4.20 | 1.256e-22 | 2.513e-22 |

## L=50

### Method summary (mean +/- SD across 3 seeds)

| method | accuracy | macro-F1 |
|---|---:|---:|
| scale_only | 78.32% +/- 0.29% | 72.48% +/- 2.13% |
| uniform_prior | 75.24% +/- 2.36% | 72.75% +/- 1.29% |
| estimated_prior_bbse | 77.57% +/- 0.15% | 71.28% +/- 4.86% |
| oracle_prior | 78.15% +/- 0.95% | 73.49% +/- 0.87% |

### Prior estimates by seed (estimated_prior_bbse)

| seed | estimated prior | L1 error vs true | working-marginal L1 error |
|---:|---|---:|---:|
| 42 | [0.438, 0.068, 0.494] | 0.2639 | 0.2466 |
| 123 | [0.407, 0.186, 0.408] | 0.0287 | 0.0214 |
| 456 | [0.351, 0.247, 0.402] | 0.0989 | 0.1293 |

### Paired comparisons (per seed)

#### estimated_prior_bbse - scale_only

- Mean paired delta-macroF1: -1.19 +/- 2.87; per-seed: -4.44, -0.12, +0.98
- Mean paired delta-acc: -0.75 +/- 0.16; per-seed: -0.71, -0.61, -0.93
- Wins/ties/losses (accuracy, estimated_prior_bbse vs scale_only): 0/0/3

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | -4.44 | [-5.27, -3.76] | -0.71 | 0.004071 | 0.008141 |
| 123 | -0.12 | [-0.62, +0.39] | -0.61 | 0.006827 | 0.008141 |
| 456 | +0.98 | [+0.30, +1.65] | -0.93 | 0.001274 | 0.003822 |

#### estimated_prior_bbse - uniform_prior

- Mean paired delta-macroF1: -1.46 +/- 5.85; per-seed: -8.13, +2.80, +0.94
- Mean paired delta-acc: +2.32 +/- 2.22; per-seed: +0.21, +4.64, +2.12
- Wins/ties/losses (accuracy, estimated_prior_bbse vs uniform_prior): 3/0/0

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | -8.13 | [-9.17, -7.23] | +0.21 | 0.6238 | 0.6238 |
| 123 | +2.80 | [+2.19, +3.42] | +4.64 | 1.676e-47 | 5.029e-47 |
| 456 | +0.94 | [+0.37, +1.51] | +2.12 | 9.523e-15 | 1.905e-14 |

#### oracle_prior - estimated_prior_bbse

- Mean paired delta-macroF1: +2.21 +/- 4.00; per-seed: +6.83, -0.01, -0.18
- Mean paired delta-acc: +0.59 +/- 0.80; per-seed: +1.14, -0.33, +0.95
- Wins/ties/losses (accuracy, oracle_prior vs estimated_prior_bbse): 2/0/1

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | +6.83 | [+5.99, +7.72] | +1.14 | 0.0001367 | 0.00041 |
| 123 | -0.01 | [-0.43, +0.37] | -0.33 | 0.08387 | 0.08387 |
| 456 | -0.18 | [-0.73, +0.37] | +0.95 | 0.0001909 | 0.00041 |

#### uniform_prior - scale_only

- Mean paired delta-macroF1: +0.27 +/- 3.31; per-seed: +3.69, -2.92, +0.04
- Mean paired delta-acc: -3.07 +/- 2.17; per-seed: -0.92, -5.25, -3.05
- Wins/ties/losses (accuracy, uniform_prior vs scale_only): 0/0/3

| seed | delta-macroF1 | F1 boot 95% CI | delta-acc | McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|
| 42 | +3.69 | [+2.87, +4.49] | -0.92 | 0.009732 | 0.009732 |
| 123 | -2.92 | [-3.57, -2.26] | -5.25 | 9.029e-50 | 2.709e-49 |
| 456 | +0.04 | [-0.75, +0.84] | -3.05 | 1.117e-17 | 2.235e-17 |

## Completeness / provenance

- Cells found: 24 / 24
- Missing cells: 0
- Unexpected cells: 0
- Source: `outputs/student/amazon3_source_prior_transfer_v1`
- This is a controlled label-shift study (source-induced non-uniform prior via grouping a balanced benchmark), not a test of covariate shift, domain shift, or natural deployment prevalence.
