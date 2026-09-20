# Final station-specific nested 10 x 10 SVM results

These outputs were generated from the 28,022-row
`data/mfdfa_features.csv` using 15 scalar MFDFA descriptors. Nine SVM models
were trained and evaluated independently, one per station. No data were pooled
across stations.

## Headline results

- Station-macro accuracy: **94.28%**
- Station-macro balanced accuracy: **94.28%**
- Station-macro precision: **93.56%**
- Station-macro F1: **93.87%**
- Out-of-fold predictions: **28,022**
- Outer folds: **90** (10 per station)
- Inner folds per outer-training partition: **10**

The headline values are unweighted arithmetic means of the nine station-level
metrics. The waveform-weighted OOF accuracy is supplied only as a descriptive
quantity and is not the result of a pooled global model.

`nested_oof_predictions.csv` contains exactly one held-out prediction per
record. `nested_fold_results.csv` and `nested_grid_candidates.csv` provide the
outer-fold and inner-search audit trails. `validation_report.json` records the
structural consistency checks.
