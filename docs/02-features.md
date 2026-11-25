# 2. Feature engineering

**Code:** [`src/noshow/features.py`](../src/noshow/features.py) ·
**Tests:** [`tests/test_features.py`](../tests/test_features.py)

## What problem it solves

Training-serving skew happens when features are computed one way during
training and another way in the API. Predictions then silently drift. All
feature logic lives in one module with two entry points, and a parity test
keeps them identical:

- `build_features(df)`: vectorised pandas version, used by training on tens
  of thousands of rows.
- `appointments_to_matrix(appointments)`: plain Python/NumPy version used by
  the online service. For the 1-50 rows of a live request, building a pandas
  DataFrame and parsing datetimes added about 11 ms per call, several times
  the model call itself (see the micro-benchmark in [benchmarks.md](benchmarks.md)).

## How it works

Input: raw fields (`booked_at`, `scheduled_at`, `confirmed`,
`prior_*`, `days_since_last_visit`). Output: the columns in `FEATURE_COLUMNS`,
in a fixed order.

| feature | how it is computed | why |
|---|---|---|
| `lead_time_days` | `(scheduled_at - booked_at)` in fractional days | long lead time → plans change |
| `weekday`, `hour` | from `scheduled_at` | slot effects; trees split integers directly, so no one-hot encoding |
| `confirmed` | 0/1 | strongest protective signal |
| `prior_appointments` | raw count | how much history backs the rates |
| `prior_no_show_rate` | smoothed rate (below) | proxy for the hidden reliability |
| `prior_cancellation_rate` | smoothed rate | habitual cancellers |
| `days_since_last_visit` | raw; `NaN` for new patients | XGBoost learns a default branch for missing values |
| `is_new_patient` | `prior_appointments == 0` | first visits behave differently |

### Smoothed rates

A patient with 1 visit and 1 no-show does not have a "100% no-show rate". The
rate is shrunk toward a prior with a small Beta prior:

```python
def smoothed_rate(events, total, prior_rate):
    return (events + PRIOR_STRENGTH * prior_rate) / (total + PRIOR_STRENGTH)
```

With `PRIOR_STRENGTH = 2` and `PRIOR_NO_SHOW_RATE = 0.2`, a new patient gets
0.2. After 1 no-show in 1 visit the rate is 0.47. After 100 no-shows in 100
visits it is about 0.98.

### Input validation

`Appointment` (in [`schemas.py`](../src/noshow/schemas.py)) rejects impossible
inputs before they reach the model. The API returns 422 when:

- `scheduled_at` is earlier than `booked_at`;
- `prior_no_shows + prior_cancellations > prior_appointments`;
- a count is negative.

### Serving fast path and parity test

```python
def appointments_to_matrix(appointments):
    out = np.empty((len(appointments), len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, a in enumerate(appointments):
        prior = float(a.prior_appointments)
        out[i] = ((a.scheduled_at - a.booked_at).total_seconds() / 86_400.0, a.scheduled_at.weekday(), ...)
    return out
```

`test_fast_path_matches_pandas_features` builds features both ways for 3,000
generated appointments and asserts that the two matrices are equal (NaNs
included). If someone changes one implementation and not the other, CI
fails.

## Safety net in the artifact

The feature list is saved in the model's `metadata.json`. When the server
loads a model, it compares that list with `FEATURE_COLUMNS` and refuses to
start if they differ (`test_feature_mismatch_is_refused`).

## Run just this part

```bash
pytest tests/test_features.py -q
```
