# Detecting structural change

A fitted vine always moves a little from day to day, mostly because of estimation
noise. The project therefore asks a statistical question: *is the dependence in the
latest window different from the dependence in the window before it by more than noise
alone would explain?*

```bash
vine-risk run --steps changes      # or: python scripts/detect_changes.py
```

- `structural_change_scores.parquet`: for every date, `structural_change_score` (the
  Frobenius distance between the Kendall-tau matrix now and one window ago, the "S_t"
  of the project brief) plus a model-based distance and some diagnostics.
- `change_scan.parquet`: every 5th day, a **permutation test** comparing the latest
  window with the one before it. Observations (in blocks of 10 days) are randomly
  shuffled between the two windows many times to learn how large the difference looks
  when nothing has changed. Statistics: the whole tau matrix, and the lower and upper
  tail dependence, each as an "any pair changed" and an "average over pairs" version. The
  file also has effect sizes (`stat_*` relative to `null_mean_*`) and Benjamini-Hochberg
  corrected flags, because many dates are tested.

The [validation study](validation.md) shows how well these detectors work on simulated data.

## What the detectors say about the real data

On the real 8-asset data (window 250, 2006-2026), with the calibrated permutation test:

- The Kendall-tau matrix differs significantly from the year before on about 39% of
  scan dates (30% after multiple-testing correction). Dependence is clearly not
  constant, but effect sizes are modest: typically 1.6 times the no-change noise level,
  at most 3.4.
- Much of this is the whole market becoming more or less correlated together (the tau
  distance correlates 0.86 with the change in average dependence).
- Tail-dependence changes are rarely significant (3-10% of dates raw, none after
  correction). The [validation study](validation.md) shows why this is weak evidence of "no change": even a
  strong tail change (C2) is missed in about half of the simulated histories.
- None of this establishes *why* dependence changed. These are associations with
  periods in the calendar, not causal statements.
