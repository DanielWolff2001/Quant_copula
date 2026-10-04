"""Simulated live monitoring: replay history one observation at a time.

Usage: python scripts/run_live.py [--run data/results/w250] [--days 120 | --start 2026-04-01]
       [--end 2026-10-02] [--delay 0] [--no-verify] [--out <run>/live]

The monitor is primed with the history before ``--start`` and the fits already stored in
``<run>/checkpoint.jsonl`` (as a live system resuming from saved state would be), then each
later price is fed in turn: new price -> return -> window update -> refit -> metrics ->
comparison with the previous model -> change score -> alert. Nothing after the current date
is ever visible to it.

Writes ``live_log.jsonl`` (every record, as it happens), ``live_log.parquet`` and
``live_alerts.csv`` to ``--out``. Unless ``--no-verify`` is given, the replay is compared
with the batch results stored in the run folder, which must agree to rounding error.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vine_risk.config import load_config
from vine_risk.data import clean_prices, download_prices
from vine_risk.monitor import (
    ConsoleSink, JsonlSink, LiveMonitor, MonitorConfig, alert_state_from_scans, records_to_frame,
)
from vine_risk.returns import log_returns
from vine_risk.rolling import load_results


def verify(live: pd.DataFrame, run: Path, tol: float = 1e-9) -> bool:
    """Compare the replay with the batch tables stored in the run folder."""
    checks = [("dependence_metrics.parquet", "d_t", "m_d_t"), ("dependence_metrics.parquet", "bic", "m_bic"),
              ("structural_change_scores.parquet", "s_tau", "s_s_tau"),
              ("structural_change_scores.parquet", "dist_model", "s_dist_model"),
              ("change_scan.parquet", "stat_tau", "scan_stat_tau"), ("change_scan.parquet", "p_tau", "scan_p_tau"),
              ("portfolio_risk.parquet", "es_vine", "risk_es_vine"), ("portfolio_risk.parquet", "var_hist", "risk_var_hist")]
    ok = True
    for fname, col, lcol in checks:
        path = run / fname
        if not path.exists() or lcol not in live:
            print(f"  skipped {lcol}: {fname} not available")
            continue
        batch = pd.read_parquet(path)
        batch = batch.set_index("timestamp") if "timestamp" in batch else batch
        batch.index = pd.DatetimeIndex(batch.index)
        mine = live[lcol].dropna()
        common = mine.index.intersection(batch[col].dropna().index)
        if len(common) == 0:
            print(f"  skipped {lcol}: no common dates")
            continue
        diff = float(np.abs(mine.loc[common] - batch.loc[common, col]).max())
        good = diff <= tol
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {lcol:<18} vs {fname:<34} {len(common):4d} dates, max abs diff {diff:.2e}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="data/results/w250")
    ap.add_argument("--days", type=int, default=120, help="replay the last N observations")
    ap.add_argument("--start", help="first replayed date (overrides --days)")
    ap.add_argument("--end", help="last replayed date")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between observations")
    ap.add_argument("--scan-step", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--risk-every", type=int, default=5, help="VaR/ES at every N-th fit (0 = never)")
    ap.add_argument("--threads", type=int, default=4, help="threads of the vine fitter")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg, run = load_config(a.config), Path(a.run)
    d = cfg.data
    prices = clean_prices(download_prices(cfg.assets, d.start, d.end, d.cache_dir), d.max_missing_frac, d.max_ffill_days)
    returns = log_returns(prices)
    if a.end:
        prices, returns = prices.loc[: a.end], returns.loc[: a.end]
    start = pd.Timestamp(a.start) if a.start else returns.index[-a.days]
    history = returns.loc[returns.index < start]
    feed = prices.loc[prices.index >= history.index[-1]]  # the last historical price starts the feed
    window = cfg.rolling.window
    if len(history) < 2 * window:
        raise SystemExit(f"Need at least {2 * window} observations before {start.date()}, have {len(history)}.")

    mcfg = MonitorConfig.from_config(cfg, scan_step=a.scan_step, n_perm=a.n_perm, risk_every_fits=a.risk_every)
    mcfg = MonitorConfig(**{**mcfg.__dict__, "vine_kwargs": {**mcfg.vine_kwargs, "num_threads": a.threads}})
    out = Path(a.out or run / "live")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "live_log.jsonl"
    log.unlink(missing_ok=True)

    prior = [r for r in load_results(run / "checkpoint.jsonl") if pd.Timestamp(r.timestamp) <= history.index[-1]]
    monitor = LiveMonitor(list(prices.columns), mcfg, [JsonlSink(log), ConsoleSink(every=20)])
    state = None
    scan_file = run / "change_scan.parquet"
    if scan_file.exists() and a.n_perm == 499:  # stored scans use 499 permutations
        scan = pd.read_parquet(scan_file)
        scan.index = pd.DatetimeIndex(scan.index)
        state = alert_state_from_scans(scan, mcfg, history.index[-1])
    else:
        print("note: no matching stored scans; starting in the 'normal' state, so an ongoing alert "
              "episode may be reported as new")
    monitor.prime(history, prior, alert_state=state)
    print(f"alert state at the start: {monitor.alert_state}")
    monitor.set_last_price(feed.iloc[0])
    print(f"primed with {len(history)} observations and {len(prior)} stored fits up to {history.index[-1].date()}; "
          f"replaying {len(feed) - 1} observations from {feed.index[1].date()} to {feed.index[-1].date()}")

    t0 = time.time()
    for date, row in feed.iloc[1:].iterrows():
        monitor.on_prices(date, row)
        if a.delay:
            time.sleep(a.delay)
    df = records_to_frame(monitor.records)
    df.to_parquet(out / "live_log.parquet")
    alerts = df[df["new_alert"]][["alert_state", "message"]]
    alerts.to_csv(out / "live_alerts.csv")
    print(f"\n{len(df)} observations, {int(df['refit'].sum())} refits in {time.time() - t0:.0f}s; "
          f"{len(alerts)} new alert(s); final state: {df['alert_state'].iloc[-1]}")
    if not a.no_verify:
        print("verification against the stored batch run:")
        print("PASS: the live replay reproduces the batch results" if verify(df, run) else "FAIL: replay differs from batch")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
