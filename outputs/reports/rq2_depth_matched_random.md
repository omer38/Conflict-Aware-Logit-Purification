# RQ2 depth-matched random control

Cascade versus random allocation with identical cascade-derived query-depth counts. Exact paired McNemar tests are Holm-adjusted across the three seeds within each dataset–budget–query-rate cell.

| Dataset | L | q/N | Cascade accuracy | Random accuracy | Mean difference | Seed directions | Maximum Holm p | Robust winner |
|---|---:|---:|---:|---:|---:|---|---:|---|
| ag_news | 10 | 1.4 | 88.36% | 88.54% | -0.19 pp | depth_matched_random, depth_matched_random, depth_matched_random | 0.8228 | none |
| ag_news | 50 | 1.4 | 88.46% | 88.67% | -0.21 pp | cascade, depth_matched_random, depth_matched_random | 0.6728 | none |
| yahoo | 10 | 1.4 | 60.97% | 60.95% | +0.02 pp | cascade, depth_matched_random, depth_matched_random | 1 | none |
| yahoo | 50 | 1.4 | 60.49% | 60.71% | -0.23 pp | cascade, depth_matched_random, depth_matched_random | 0.4709 | none |
| amazon | 10 | 1.4 | 58.27% | 58.29% | -0.02 pp | depth_matched_random, cascade, depth_matched_random | 1 | none |
| amazon | 50 | 1.4 | 54.78% | 54.07% | +0.70 pp | cascade, cascade, depth_matched_random | 0.6641 | none |
| amazon3 | 10 | 1.4 | 73.57% | 73.99% | -0.42 pp | cascade, depth_matched_random, depth_matched_random | 0.8652 | none |
| amazon3 | 50 | 1.4 | 72.19% | 72.99% | -0.80 pp | depth_matched_random, depth_matched_random, cascade | 1 | none |
| ag_news | 10 | 2 | 88.38% | 88.26% | +0.12 pp | cascade, depth_matched_random, cascade | 1 | none |
| ag_news | 50 | 2 | 88.43% | 88.47% | -0.04 pp | depth_matched_random, depth_matched_random, cascade | 0.3291 | none |
| yahoo | 10 | 2 | 61.36% | 61.29% | +0.07 pp | cascade, depth_matched_random, depth_matched_random | 0.6987 | none |
| yahoo | 50 | 2 | 60.15% | 60.38% | -0.23 pp | cascade, depth_matched_random, depth_matched_random | 0.7451 | none |
| amazon | 10 | 2 | 58.20% | 58.48% | -0.28 pp | depth_matched_random, depth_matched_random, depth_matched_random | 1 | none |
| amazon | 50 | 2 | 53.77% | 53.92% | -0.16 pp | depth_matched_random, cascade, cascade | 0.0741 | none |
| amazon3 | 10 | 2 | 73.46% | 73.32% | +0.14 pp | cascade, cascade, depth_matched_random | 0.534 | none |
| amazon3 | 50 | 2 | 71.71% | 72.42% | -0.72 pp | cascade, depth_matched_random, depth_matched_random | 0.2899 | none |

## Query-rate audit

The new control exactly matches the cascade rule recomputed from the current tracked teacher logits in every cell. Pre-existing cascade JSONs differ from that recomputation by one query in the following cells, while remaining within the established ±0.001 tolerance:

- ag_news, L=10, q=1.4: stored 1.4001667, recomputed/random 1.4000000
- ag_news, L=50, q=1.4: stored 1.4001667, recomputed/random 1.4000000
- amazon, L=10, q=1.4: stored 1.4004000, recomputed/random 1.4000000
- amazon, L=50, q=1.4: stored 1.4004000, recomputed/random 1.4000000
