"""Simulated live monitoring: replay history through a `vine_risk.monitor.LiveMonitor`."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from vine_risk.config import Config
from vine_risk.monitor import (
    ConsoleSink, JsonlSink, LiveMonitor, MonitorConfig, alert_state_from_scans, records_to_frame,
)
from vine_risk.pipeline import load_prices_and_returns
from vine_risk.rolling import load_results

CHECKS = [("dependence_metrics.parquet", "d_t", "m_d_t"), ("dependence_metrics.parquet", "bic", "m_bic"),
          ("structural_change_scores.parquet", "s_tau", "s_s_tau"),
          ("structural_change_scores.parquet", "dist_model", "s_dist_model"),
          ("change_scan.parquet", "stat_tau", "scan_stat_tau"), ("change_scan.parquet", "p_tau", "scan_p_tau"),
          ("portfolio_risk.parquet", "es_vine", "risk_es_vine"), ("portfolio_risk.parquet", "var_hist", "risk_var_hist")]


def verify_against_batch(live: pd.DataFrame, run: str | Path, tol: float = 1e-9,
                         out: Callable[[str], None] = print) -> bool:
    """Compare a replay with the batch tables stored in the run folder; ``True`` if all agree."""
    run, ok = Path(run), True
    for fname, col, lcol in CHECKS:
        path = run / fname
        if not path.exists() or lcol not in live:
            out(f"  skipped {lcol}: {fname} not available")
            continue
        batch = pd.read_parquet(path)
        batch = batch.set_index("timestamp") if "timestamp" in batch else batch
        batch.index = pd.DatetimeIndex(batch.index)
        mine = live[lcol].dropna()
        common = mine.index.intersection(batch[col].dropna().index)
        if len(common) == 0:
            out(f"  skipped {lcol}: no common dates")
            continue
        diff = float(np.abs(mine.loc[common] - batch.loc[common, col]).max())
        good = diff <= tol
        ok &= good
        out(f"  {'OK  ' if good else 'FAIL'} {lcol:<18} vs {fname:<34} {len(common):4d} dates, max abs diff {diff:.2e}")
    return ok


def run_replay(cfg: Config, run: str | Path, *, days: int = 120, start: str | None = None, end: str | None = None,
               delay: float = 0.0, scan_step: int = 5, n_perm: int = 499, risk_every: int = 5, threads: int = 4,
               verify: bool = True, out_dir: str | Path | None = None, out: Callable[[str], None] = print) -> pd.DataFrame:
    """Replay recent history through a monitor primed from the saved run; see ``scripts/run_live.py``."""
    run = Path(run)
    prices, returns = load_prices_and_returns(cfg)
    if end:
        prices, returns = prices.loc[:end], returns.loc[:end]
    first = pd.Timestamp(start) if start else returns.index[-days]
    history = returns.loc[returns.index < first]
    feed = prices.loc[prices.index >= history.index[-1]]  # the last historical price starts the feed
    window = cfg.rolling.window
    if len(history) < 2 * window:
        raise ValueError(f"Need at least {2 * window} observations before {first.date()}, have {len(history)}.")

    mcfg = MonitorConfig.from_config(cfg, scan_step=scan_step, n_perm=n_perm, risk_every_fits=risk_every)
    mcfg = replace(mcfg, vine_kwargs={**mcfg.vine_kwargs, "num_threads": threads})
    target = Path(out_dir or run / "live")
    target.mkdir(parents=True, exist_ok=True)
    log = target / "live_log.jsonl"
    log.unlink(missing_ok=True)

    prior = [r for r in load_results(run / "checkpoint.jsonl") if pd.Timestamp(r.timestamp) <= history.index[-1]]
    monitor = LiveMonitor(list(prices.columns), mcfg, [JsonlSink(log), ConsoleSink(every=20, out=out)])
    state = None
    scan_file = run / "change_scan.parquet"
    if scan_file.exists() and n_perm == 499:  # stored scans use 499 permutations
        scan = pd.read_parquet(scan_file)
        scan.index = pd.DatetimeIndex(scan.index)
        state = alert_state_from_scans(scan, mcfg, history.index[-1])
    else:
        out("note: no matching stored scans; starting in the 'normal' state, so an ongoing alert "
            "episode may be reported as new")
    monitor.prime(history, prior, alert_state=state)
    monitor.set_last_price(feed.iloc[0])
    out(f"alert state at the start: {monitor.alert_state}")
    out(f"primed with {len(history)} observations and {len(prior)} stored fits up to {history.index[-1].date()}; "
        f"replaying {len(feed) - 1} observations from {feed.index[1].date()} to {feed.index[-1].date()}")

    t0 = time.time()
    for date, row in feed.iloc[1:].iterrows():
        monitor.on_prices(date, row)
        if delay:
            time.sleep(delay)
    df = records_to_frame(monitor.records)
    df.to_parquet(target / "live_log.parquet")
    alerts = df[df["new_alert"]][["alert_state", "message"]]
    alerts.to_csv(target / "live_alerts.csv")
    out(f"\n{len(df)} observations, {int(df['refit'].sum())} refits in {time.time() - t0:.0f}s; "
        f"{len(alerts)} new alert(s); final state: {df['alert_state'].iloc[-1]}")
    if verify:
        out("verification against the stored batch run:")
        out("PASS: the live replay reproduces the batch results" if verify_against_batch(df, run, out=out)
            else "FAIL: replay differs from batch")
    out(f"-> {target}")
    return df
