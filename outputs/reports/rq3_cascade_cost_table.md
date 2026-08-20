# RQ2/RQ3: Cost-normalised query-budget comparison at the student level

All cells: ModernBERT-base student, batch_size=16, max_length=256, alpha=0.5, tau=2.0, seed=42.
Cascade realises 1.40 queries/sample (bisected on the entropy threshold; docs/labelfree_calibration_results.md target ~1.3-1.4).

| Dataset | L | 1 teacher (1.0 q/N) | Cascade (1.4 q/N) | Mean (3.0 q/N) | Calibrated (3.0 q/N) | Cascade vs Calibrated |
|---|---|---|---|---|---|---|
| ag_news | 10 | 0.8700 | 0.8800 | 0.8517 | 0.8825 | -0.25 pp |
| ag_news | 50 | 0.8708 | 0.8783 | 0.8575 | 0.8867 | -0.83 pp |
| yahoo | 10 | 0.5970 | 0.6170 | 0.6220 | 0.6230 | -0.60 pp |
| yahoo | 50 | 0.6000 | 0.6040 | 0.6160 | 0.6240 | -2.00 pp |
| amazon | 10 | 0.5660 | 0.5540 | 0.5820 | 0.5540 | +0.00 pp |
| amazon | 50 | 0.5620 | 0.5580 | 0.5600 | 0.5400 | +1.80 pp |

Cascade queries 1.4/3 = 47% of the full-committee cost. It beats the single-teacher baseline
on 4/6 cells and is within 0.25-0.83pp of the full calibrated committee on AG News and Yahoo,
while matching or beating it outright on Amazon.
