# Live monitor

!!! info "Replay versus the real daily update"
    This page is about the monitor and its **replay** of history. To keep a run up to date with new prices every day, see [Daily updates](daily-update.md); it uses the same monitor.

The monitor is the "live" mode of the project: it receives **one observation at a time**
and, after each, reports updated dependence metrics, a comparison with the previous model, a
change score and an alert state. The *replay* feeds history through it day by day, which makes the whole system testable and reproducible without any data
connection; with real data the same monitor is run by `vine-risk update`.

```bash
vine-risk replay --days 120                               # replay the last 120 days
vine-risk replay --start 2023-11-20 --end 2024-02-15
```

What happens for every new price (`src/vine_risk/monitor.py`, class `LiveMonitor`):

```
new price -> log return -> update the rolling window -> refit if due (refit_frequency)
          -> dependence metrics (D_t, tails, vine structure changes vs the previous fit)
          -> distance to the model one window earlier
          -> permutation test of the last two windows (every 5th day)
          -> alert state -> record (console, JSON log, table)
```

The script first *primes* the monitor with the history before the start date and the fits
already saved in `checkpoint.jsonl`, exactly as a real system would resume from saved state,
so a replay of 120 days costs 120 refits (about a second each) and not 5,000.

**Alerts** come from the permutation test of [structural change detection](methodology.md), because it is calibrated and does not
depend on the number of assets. An alert *starts* when the Kendall-tau matrix of the latest
window differs significantly (p <= 1%) from the one before **and** the difference is at
least 2 times what noise alone produces; it *ends* when that factor falls below 1.5. Using two
levels stops the alert from flickering around a single cut-off. On the real data this gives 11
alert episodes in 20 years (one every two years or so), against about 40% of scan dates that are
individually significant, because with a year of data even small differences are significant.
These thresholds (2.0 and 1.5) are conventions, not estimated quantities; change them in
`MonitorConfig`.

**No look-ahead, verified.** After the replay, the script compares the monitor's output with
the batch results stored in the run folder (`--no-verify` skips this). On the real data the
replays match to rounding error (differences of 0 to 1e-18) for the metrics, the change scores,
the permutation tests and the VaR/ES numbers, for example:

```
alert state at the start: normal
2023-12-14 [alert] D_t=0.207 ES=2.22% ** ALERT ** dependence differs from the previous window
    (tau-matrix distance 2.2x the no-change level, p=0.002)
60 observations, 60 refits in 75s; 1 new alert(s); final state: alert
  OK   m_d_t         vs dependence_metrics.parquet            60 dates, max abs diff 0.00e+00
  OK   scan_p_tau    vs change_scan.parquet                   12 dates, max abs diff 0.00e+00
  OK   risk_es_vine  vs portfolio_risk.parquet                12 dates, max abs diff 0.00e+00
PASS: the live replay reproduces the batch results
```

The automated tests (`tests/test_monitor.py`) also check that running on a prefix of the data
gives the same records as the full run, and that resuming from a checkpoint gives the same
records as running continuously. Outputs go to `<run>/live/` (`live_log.jsonl` with every
record as it happens, `live_log.parquet`, `live_alerts.csv`), and panel 9 of the dashboard
shows them.

Things to know: the monitor is not a trading signal; an alert says that the dependence of
the last year differs from the year before, not why; resuming from saved fits needs the
alert state as well (the script derives it from the stored scans, otherwise an ongoing alert
would be reported as new); and the first alert can only come after `2 x window` observations.
