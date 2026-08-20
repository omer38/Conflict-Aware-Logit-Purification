# Amazon-3 calibrated maximum-likelihood prior follow-up

This isolated follow-up replaces confusion-moment inversion with bias-corrected temperature scaling plus maximum-likelihood label-shift estimation (BCTS+MLLS). It was motivated after the original BBSE estimator showed small-sample instability and is therefore reported as a follow-up robustness analysis, not a predeclared primary estimator.

The same 500-row unlabelled reference batch, 2,000-row working pool, declared labels, teacher cache, student configuration, and seeds are used. Existing baseline and oracle cells are reused without retraining.

## L=10

| method | accuracy | macro-F1 |
|---|---:|---:|
| scale_only | 78.56% +/- 0.34% | 70.99% +/- 0.71% |
| uniform_prior | 74.53% +/- 1.60% | 72.49% +/- 0.99% |
| estimated_prior_mlls_bcts | 78.02% +/- 1.57% | 69.18% +/- 4.73% |
| oracle_prior | 79.00% +/- 0.31% | 73.66% +/- 0.35% |

### BCTS+MLLS prior estimates

| seed | estimated prior | L1 error |
|---:|---|---:|
| 42 | [0.456, 0.135, 0.408] | 0.1296 |
| 123 | [0.523, 0.077, 0.400] | 0.2465 |
| 456 | [0.394, 0.169, 0.438] | 0.0758 |

### estimated_prior_mlls_bcts - scale_only

- Mean delta macro-F1: -1.81 +/- 4.47 pp
- Mean delta accuracy: -0.53 +/- 1.25 pp
- Accuracy wins/ties/losses: 1/0/2

### estimated_prior_mlls_bcts - uniform_prior

- Mean delta macro-F1: -3.31 +/- 4.11 pp
- Mean delta accuracy: +3.49 +/- 1.17 pp
- Accuracy wins/ties/losses: 3/0/0

### oracle_prior - estimated_prior_mlls_bcts

- Mean delta macro-F1: +4.48 +/- 4.56 pp
- Mean delta accuracy: +0.98 +/- 1.45 pp
- Accuracy wins/ties/losses: 2/0/1

## L=50

| method | accuracy | macro-F1 |
|---|---:|---:|
| scale_only | 78.32% +/- 0.29% | 72.48% +/- 2.13% |
| uniform_prior | 75.24% +/- 2.36% | 72.75% +/- 1.29% |
| estimated_prior_mlls_bcts | 76.85% +/- 1.68% | 67.41% +/- 4.20% |
| oracle_prior | 78.15% +/- 0.95% | 73.49% +/- 0.87% |

### BCTS+MLLS prior estimates

| seed | estimated prior | L1 error |
|---:|---|---:|
| 42 | [0.407, 0.101, 0.492] | 0.1987 |
| 123 | [0.386, 0.205, 0.409] | 0.0285 |
| 456 | [0.348, 0.259, 0.392] | 0.1188 |

### estimated_prior_mlls_bcts - scale_only

- Mean delta macro-F1: -5.07 +/- 3.74 pp
- Mean delta accuracy: -1.46 +/- 1.39 pp
- Accuracy wins/ties/losses: 0/1/2

### estimated_prior_mlls_bcts - uniform_prior

- Mean delta macro-F1: -5.34 +/- 4.36 pp
- Mean delta accuracy: +1.61 +/- 2.02 pp
- Accuracy wins/ties/losses: 2/0/1

### oracle_prior - estimated_prior_mlls_bcts

- Mean delta macro-F1: +6.08 +/- 3.76 pp
- Mean delta accuracy: +1.30 +/- 1.11 pp
- Accuracy wins/ties/losses: 3/0/0

## Provenance

- New valid cells: 6/6
- New-cell directory: `outputs/student/amazon3_source_prior_transfer_mlls_v1`
- Reused baseline directory: `outputs/student/amazon3_source_prior_transfer_v1`
- No new teacher queries were made.
