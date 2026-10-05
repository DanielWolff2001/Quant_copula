# Command line

Everything is available through one command, `vine-risk` (also `python -m vine_risk`). Each command takes
`--config` (default `configs/default.yaml`) and `--help`.

| Command | What it does |
|---------|--------------|
| [`vine-risk run`](#run) | run or extend the whole pipeline |
| [`vine-risk update`](#update) | daily update: fetch new prices, process them, extend the run |
| [`vine-risk schedule`](#schedule) | print a ready-made cron / launchd / systemd entry for the daily update |
| [`vine-risk benchmark`](#benchmark) | compare the vine copula with standard risk models, backtested |
| [`vine-risk replay`](#replay) | simulated live monitoring: replay recent history day by day |
| [`vine-risk info`](#info) | show what produced a run folder |
| [`vine-risk dashboard`](#dashboard) | open the Streamlit dashboard |

Exit codes: `0` success, `2` a problem that was explained on screen (bad settings, revised data, a run folder in use, ...),
`130` interrupted with Ctrl+C (progress is kept).

## run

```bash
vine-risk run [--run-dir DIR] [--steps rolling,metrics,changes,risk] [--last N] [--window W]
              [--refit-frequency K] [--n-jobs J] [--scan-step S] [--n-perm P] [--risk-step R]
              [--refresh] [--force]
```

Runs four steps in order, each reading what the previous one wrote into the **run folder** (default `data/results/w<window>`):

| Step | Does | Writes |
|------|------|--------|
| `rolling` | fits a vine on every window (in parallel) | `checkpoint.jsonl`, `fits.parquet`, `pair_copulas.parquet`, `pairwise.parquet` |
| `metrics` | dependence metrics per fit | `dependence_metrics.parquet`, `pairwise_*.parquet` |
| `changes` | change scores and the permutation-test scan (`--filtered-scan` adds a scan on GARCH-filtered residuals) | `structural_change_scores.parquet`, `change_scan.parquet` (`change_scan_garch.parquet`) |
| `risk` | rolling VaR and Expected Shortfall, VaR backtest | `portfolio_risk.parquet`, `var_backtest.csv` |

Useful options: `--last 700` uses only the last 700 days (a quick test); `--steps rolling,metrics` runs a subset;
`--refresh` downloads the prices again instead of using the local copy; `--window`, `--refit-frequency`, `--marginal` and
`--marginal-lookback` override the config (for example `--marginal garch_t --run-dir data/results/w250_garch` for a GARCH-filtered run).

**Every step is incremental.** Running the same command again, or after new prices arrived, only does the new work: new fits, new
scan dates, new risk dates. Extending a run gives exactly the same result as running everything at once (this is tested).

**Overrides must be repeated.** Settings you change on the command line (`--window`, `--refit-frequency`, `--marginal`, `--marginal-lookback`) belong to the run. `update`
and `replay` accept the same options and must be given the same values; otherwise the safety check below stops them. Putting the
values in the config file avoids the problem.

**Safety checks.** A run folder remembers how it was made (`manifest.json`). `run` refuses to continue if the fit settings differ
(for example another window), or if the prices behind the stored fits have changed (a data vendor revised history). The message says
what differs; use a new `--run-dir` or delete the old one. `--force` skips the checks and is not recommended. A lock file stops two
commands from writing to the same folder at once.

## update

```bash
vine-risk update [--run-dir DIR] [--dry-run] [--threads 4] [--alpha 0.01]
                 [--enter-ratio 2.0] [--exit-ratio 1.5] [--force]
```

The daily job; see [Daily updates](daily-update.md). `--dry-run` fetches and checks the prices and reports what it would do,
without changing anything.

## schedule

```bash
vine-risk schedule --kind cron|launchd|systemd [--at 23:30] [--run-dir DIR]
```

Prints the text of a scheduler entry that runs `vine-risk update` on weekdays at the given local time, with this machine's paths
filled in. Nothing is installed; see [Daily updates](daily-update.md#scheduling) for where to put it.

## benchmark

```bash
vine-risk benchmark --garch-run-dir data/results/w250_garch [--run-dir data/results/w250] [--out data/results/benchmark]
                    [--alphas 0.975 0.99] [--first DATE] [--step 1] [--reference vine_garch] [--no-evaluate]
```

Compares ten risk models on the same days and portfolios: for every date, each model forecasts tomorrow's VaR and Expected Shortfall
from the data up to that date; the forecasts are then backtested against what happened (Kupiec, Christoffersen, Acerbi-Székely, a
joint VaR/ES score and Diebold-Mariano tests). Needs the fits of two run folders, one with rank marginals (`--run-dir`) and one made
with `--marginal garch_t` (`--garch-run-dir`). Without the second, the GARCH copula models are left out. Writes `forecasts.parquet`,
`realized.parquet` and `report.csv`; it is incremental like the other steps. See [Risk model comparison](benchmarks.md).

## replay

```bash
vine-risk replay [--days 120 | --start 2023-11-20] [--end DATE] [--delay SECONDS] [--no-verify]
```

Feeds history through the live monitor one day at a time, resuming from the fits stored in the run folder, and (unless
`--no-verify`) checks that the result equals the stored batch results exactly. Writes a log to `<run>/live/`. See
[Live monitor](live-monitor.md).

## info

```bash
vine-risk info [RUN_DIR]
```

Shows what is in a run folder and what made it: code version and git commit, Python and library versions, fit settings and
their hash, the data range, which steps ran and how long they took, and whether the current environment differs.

## dashboard

```bash
vine-risk dashboard [--run-dir DIR] [--port 8501]
```

Starts the [dashboard](dashboard.md) on the chosen run. It needs the repository checkout (the app file lives in `dashboard/`).

## The scripts

`scripts/*.py` do the same as the commands above (`run_rolling.py`, `compute_metrics.py`, `detect_changes.py`, `compute_risk.py`,
`run_live.py`) with their original options; they are thin wrappers around the same functions. `run_validation.py` (the 30-minute
[validation study](validation.md)), `plot_validation.py` and `make_doc_figures.py` have no command-line equivalent.
