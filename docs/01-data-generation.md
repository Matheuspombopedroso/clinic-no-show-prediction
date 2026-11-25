# 1. Synthetic data generation

**Code:** [`src/noshow/data.py`](../src/noshow/data.py)

## What problem it solves

The project is public, so it cannot use real patient data. But a model trained
on random noise would make every metric and benchmark meaningless. The
generator therefore simulates a clinic in which no-shows depend on the
features in a *plausible* way, with noise, so that there is something real to
learn and the model can be evaluated honestly.

## How it works

The generator is a small simulation, one synthetic patient at a time:

1. **Hidden traits.** Each patient gets a `reliability` score (standard
   normal) and a `cancel_propensity`. They are never written to the CSV. The
   model has to infer reliability from the patient's past behaviour, as a
   real clinic would.
2. **Bookings over time.** A patient books, attends (or not), then books again
   after a random gap (20, 45 or 90 days on average, depending on the
   patient type), from 2024-01-01 to 2025-12-31.
3. **Slot and lead time.** About 25% of bookings are short-notice (0–2 days).
   The rest are roughly exponential with a 14-day mean, capped at 90 days.
   Slots fall Monday to Saturday, 07:00–18:30.
4. **Outcome.** First the appointment may be cancelled (more likely for
   cancel-prone patients and long lead times). Otherwise the no-show
   probability is a logistic function:

```python
logit = (
    -2.5
    - 0.9 * reliability  # hidden trait, visible via past no-shows
    + 0.35 * np.log1p(lead_days)  # long lead time -> forgets / plans change
    + WEEKDAY_EFFECT[weekday]  # Mon/Fri/Sat worse
    + _hour_effect(hour)  # early morning / end of day worse
    - 1.2 * confirmed  # strongest protective signal
    + 1.2 * cancel_rate  # habitual cancellers also skip more
    + 0.003 * min(days_since, 365)  # long absence -> weaker link to the clinic
    + (0.35 if n_prior == 0 else 0.0)  # first visit
    + rng.normal(0.0, 0.5)  # irreducible noise
)
```

5. **History without leakage.** The `prior_*` counters and
   `days_since_last_visit` are written *before* the current outcome is added
   to the running totals. A test (`test_generated_history_has_no_leakage`)
   checks that `prior_no_shows` equals the cumulative count of earlier
   no-shows for each patient.

Two details keep every generated row a valid API request: the next booking
is clamped to happen after the previous appointment (so `days_since_last_visit`
is never negative), and timestamps are whole minutes. A test
(`test_every_generated_row_is_a_valid_api_request`) validates every generated
row against the API schema.

A fixed seed (`SEED = 42`) makes the output identical on every run. The
Docker build relies on this to train the same model inside the image.

## Output columns

| column | meaning |
|---|---|
| `appointment_id`, `patient_id` | synthetic identifiers |
| `booked_at`, `scheduled_at` | booking time and appointment start |
| `confirmed` | patient confirmed the reminder |
| `prior_appointments`, `prior_no_shows`, `prior_cancellations` | history as of booking |
| `days_since_last_visit` | days since last *attended* visit (empty for new patients) |
| `status` | `attended` / `no_show` / `cancelled` |
| `no_show` | label: 1 if `status == "no_show"` |

## Run just this part

```bash
python -m noshow.data --out data/appointments.csv
```

## What to look at in the output

This is the output from the run on the benchmark machine:

```
wrote 55433 appointments (49890 non-cancelled) to data\appointments.csv
status
attended     0.695
no_show      0.205
cancelled    0.100
no-show rate among non-cancelled: 0.228
```

A no-show rate of about 23% among non-cancelled appointments is inside the
range usually reported for outpatient clinics. The coefficients were set to
produce a rate in that range, so this figure is a design choice, not a
finding.
