> **Diagnostic only.** Harm, rescue, newly-harmed churn and AUC use gold
> training-pool labels. They are retrospective supervised diagnostics,
> not label-free or deployable. AUC is stratified 5-fold out-of-fold
> mean ± fold SD.

| Dataset | Harm rate (logit mean) | Harm rate (prior-and-scale-aligned mean) | Drop | OOF AUC (logit mean) | OOF AUC (aligned mean) |
|---|---|---|---|---|---|
| ag_news | 4.03% | 2.72% | +1.32 pp | 0.976±0.002 | 0.950±0.006 |
| yahoo | 6.70% | 6.42% | +0.28 pp | 0.926±0.018 | 0.887±0.007 |
| amazon | 13.64% | 13.64% | +0.00 pp | 0.843±0.024 | 0.807±0.019 |
| amazon3 | 9.87% | 7.67% | +2.20 pp | 0.955±0.014 | 0.889±0.026 |
