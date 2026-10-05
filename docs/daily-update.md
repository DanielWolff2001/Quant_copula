# Daily updates

`vine-risk update` keeps a run folder current. Run it once a day (by hand or from a scheduler): it fetches the latest prices,
processes every new day the way a live monitor would, and brings all tables up to date.

```bash
vine-risk update --dry-run      # look first: what would be processed?
vine-risk update                # do it
```

## What happens

1. **Fetch** the prices from the configured source and **check them** (table below). A failed check stops everything before anything is written.
2. **Check the history.** The prices behind the latest stored fit are compared with the fresh ones. Vendors occasionally revise
   history (after a split or dividend adjustment); stored fits would then be stale, so the update refuses and says so.
3. **Process each new day** through the [live monitor](live-monitor.md), which resumes from the stored fits: it refits when due,
   appends each new fit to `checkpoint.jsonl`, runs the permutation test on scan days, and updates the alert state. A new alert is
   printed and logged.
4. **Bring the tables up to date** (metrics, change scores and scan, portfolio risk, backtest) incrementally.
5. **Remember** the prices that were used (the local price copy is replaced only now).

The result is identical to what `vine-risk run` on the full data would have produced; the test suite checks this.

## Where prices come from

Set the source in `configs/default.yaml`:

```yaml
data:
  source: yahoo            # the default: Yahoo Finance through yfinance
```

or use your own files, so any vendor can be used:

```yaml
data:
  source: csv
  csv_path: data/my_prices.csv     # one wide CSV (Date column + one column per ticker) ...
  # csv_path: data/prices/         # ... or a folder with one <TICKER>.csv each ("Adj Close" or "Close" column)
```

Prices must be **adjusted** for splits and dividends. CSV files are read afresh every time, so an update picks up whatever you
have appended to them.

## Checks on the new prices

| Check | Result |
|-------|--------|
| no data, a ticker missing or empty, non-positive prices on the new days | **error**: the update stops, nothing is changed |
| a ticker without a price on a new day | warning: the last price is carried forward (up to the configured limit), otherwise the day is dropped |
| a one-day move above 25 % (log) on a new day | warning: possibly an unadjusted split |
| the last price is more than 5 business days old | warning: is the feed stalled? |

Warnings are printed and stored in `live/last_update.json`.

## Files an update writes (in `<run>/live/`)

| File | Content |
|------|---------|
| `updates.jsonl` | one line per processed day: metrics, scores, scan results, alert state (appended) |
| `alerts.csv` | every new alert with its date and message (appended) |
| `last_update.json` | status of the latest check: time, new days, fits added, alerts, warnings. Handy for a health check: if it is old, the job did not run. |

The result tables (`dependence_metrics.parquet`, `change_scan.parquet`, ...) are updated in place, and `manifest.json` records the
update.

## Scheduling

`vine-risk schedule` prints a ready-made entry with this machine's paths filled in. It installs nothing; you decide where it goes.

```bash
vine-risk schedule --kind cron --at 23:30       # then: crontab -e, and paste the line
vine-risk schedule --kind launchd --at 23:30    # macOS: save as ~/Library/LaunchAgents/com.vinerisk.update.plist
                                                #        then: launchctl load ~/Library/LaunchAgents/com.vinerisk.update.plist
vine-risk schedule --kind systemd --at 23:30    # Linux: two unit files and a timer, instructions in the output
```

The entry runs on weekdays at the time you give, in the machine's local time. Choose a time after the market's close when your
data vendor has the day's prices (US markets close at 16:00 New York time). Output goes to `data/update.log`. A run lock prevents two
updates from overlapping; if one is still running, the next one stops with an explanation.

## When something goes wrong

| Message | Meaning and what to do |
|---------|------------------------|
| `... failed validation: no prices for: [...]` | the source returned nothing for a ticker; check the ticker symbol and the feed |
| `The data behind the stored fit of <date> changed` | the vendor revised old prices; refit the affected period in a new run folder (`vine-risk run --run-dir ...`) |
| `... was made with different fit settings` | the config differs from the one that made the run (window, vine options, assets, seed); undo the change or use a new folder |
| `... is in use by process N` | another command is working on the folder; wait for it, or delete `<run>/.lock` if you are sure it is gone |
| `checkpoint.jsonl not found` | no run exists yet; create it with `vine-risk run` |

## What it does not do

It does not send notifications (read `alerts.csv` or `last_update.json`, or wrap the command in your own mailer), does not trade, and
does not repair bad data beyond the carry-forward above. An alert says that the dependence of the last year differs from the year
before; it does not say why.
