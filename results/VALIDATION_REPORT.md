# Validation report

The saved outputs pass the following checks:

- 28,022 unique records received exactly one out-of-fold prediction.
- Nine stations were modeled independently with no cross-station pooling.
- Each station has ten outer folds; each inner search uses ten folds.
- Ninety outer folds and 810 grid candidates are recorded.
- Confusion counts reconcile with all predictions and class totals.
- Recomputed station-macro metrics match `aggregate_metrics.json`.
- The input SHA-256 corresponds to `data/mfdfa_features.csv`.

The summed confusion matrix is descriptive only; it combines predictions after
the nine independent models were evaluated and does not represent a pooled
classifier.
