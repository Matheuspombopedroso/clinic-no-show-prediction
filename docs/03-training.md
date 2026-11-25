# 3. Training and evaluation

**Code:** [`src/noshow/train.py`](../src/noshow/train.py),
[`src/noshow/model.py`](../src/noshow/model.py) ·
**Tests:** [`tests/test_model.py`](../tests/test_model.py)

## What problem it solves

The goal is a model whose numbers you can trust: good at ranking risky
appointments, honest about its probabilities, and evaluated the way it will
be used, which means predicting the future from the past.

## How it works

1. **Drop cancelled appointments.** A cancelled slot is neither a show nor a
   no-show. Cancellations only feed the history features.
2. **Time-based split.** Rows are sorted by `scheduled_at`. The most recent
   20% are the test set, and the last 10% of the remaining training period is
   used for early stopping:

   ```python
   def time_split(df, test_fraction):
       ordered = df.sort_values("scheduled_at", kind="stable").reset_index(drop=True)
       cut = int(len(ordered) * (1.0 - test_fraction))
       return ordered.iloc[:cut], ordered.iloc[cut:]
   ```

   A random split would let the model learn from appointments that happen
   *after* the ones it is tested on, which overstates real performance.
3. **XGBoost.** `XGBClassifier` with shallow trees (`max_depth=4`), a low
   learning rate (0.03), row and column subsampling, and early stopping after
   50 rounds without improvement in validation log-loss. With fewer than 50k
   rows of tabular data, a regularised configuration beats a deep one.
4. **Metrics on the test set:**
   - **ROC-AUC**: does the model rank no-shows above shows?
   - **PR-AUC** (average precision): ranking quality focused on the
     minority class. Compare it with the base rate, which is what a random
     model would score.
   - **Brier score**: mean squared error of the probabilities, which measures
     calibration and sharpness. It is compared with the Brier score of always
     predicting the base rate.
   - **Reliability table**: mean predicted probability against observed
     no-show rate, per decile of predictions.
5. **Versioned artifact.** The model and its metadata are written to
   `artifacts/no-show-xgb/<version>/`:
   - `model.json`: XGBoost's portable JSON format.
   - `metadata.json`: version, feature list, train and test periods, best
     iteration, parameters, test metrics, gain-based feature importance, and
     Python/XGBoost versions.

## Run just this part

```bash
python -m noshow.train --data data/appointments.csv --version 1.0.0
```

## Output from the run on the benchmark machine

```
saved model 1.0.0 to artifacts
o-show-xgb.0.0
{
  "n": 9978,
  "no_show_rate": 0.2171,
  "roc_auc": 0.7727,
  "pr_auc": 0.4889,
  "brier": 0.1413,
  "brier_baseline": 0.17
}
reliability (test set, by predicted-probability decile):
  n  mean_predicted  observed_rate
998          0.0439         0.0301
998          0.0624         0.0571
998          0.0819         0.0762
997          0.1083         0.0943
998          0.1492         0.1393
998          0.2059         0.2194
997          0.2676         0.2518
998          0.3304         0.2986
998          0.4150         0.4409
998          0.6025         0.5631
```

## What to look at

- **ROC-AUC 0.773 / PR-AUC 0.489** against a base rate of 0.217. The model
  ranks clearly better than chance, but it is far from perfect. That is
  expected, because the generator adds noise and hides each patient's true
  reliability.
- **Brier 0.141 vs 0.170 baseline.** The probabilities carry real
  information beyond the average rate.
- **Training details.** 39,912 training rows (2024-01-06 to 2025-09-22),
  9,978 test rows (2025-09-22 to 2025-12-30), early stopping at iteration
  223.
- **Calibration.** Predicted and observed rates match within a few points in
  most deciles. The top decile is somewhat over-confident (0.60 predicted vs
  0.56 observed). If downstream decisions depended on exact probabilities,
  post-hoc calibration such as isotonic regression on a held-out slice would
  be the next step.
- **Feature importance (gain).** `confirmed` dominates, followed by
  `prior_no_show_rate`, `days_since_last_visit` and `lead_time_days`. This
  matches the generator's coefficients. `is_new_patient` gets no gain
  because it is fully determined by `prior_appointments == 0`. It is kept for
  readability, and dropping it would not change predictions.
- **Caveat.** These metrics describe how well the model recovers a process
  this project designed. They say nothing about how a model would perform on
  real clinic data.
