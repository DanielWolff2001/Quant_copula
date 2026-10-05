"""The daily update: fetch new prices, process them as a live monitor would, extend the run folder.

``vine-risk update`` (run once a day, for instance from a scheduler; see ``vine-risk schedule``) does:

1. fetch the latest prices from the configured source and sanity-check them;
2. check that the history behind the stored fits is unchanged (prices can be revised by the vendor);
3. feed every new day through a `vine_risk.monitor.LiveMonitor` that resumes from the saved
   fits - each new fit is appended to the checkpoint, alerts are logged;
4. bring the result tables (metrics, change scan, risk, backtest) up to date, incrementally.

Nothing is changed if a check fails, and a lock prevents two updates from running at once.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from vine_risk.change_detection import default_lag
from vine_risk.config import Config
from vine_risk.data import DataIssue, clean_prices, price_cache_path, validate_prices
from vine_risk.locking import run_lock
from vine_risk.manifest import ManifestMismatch, check_resume, read_manifest, verify_history_unchanged
from vine_risk.monitor import ConsoleSink, JsonlSink, LiveMonitor, MonitorConfig, alert_state_from_scans
from vine_risk.returns import log_returns
from vine_risk.rolling import CheckpointIndex, RollingVineModel
from vine_risk.runner import run_pipeline
from vine_risk.sources import PriceSource, make_source


@dataclass
class UpdateReport:
    """What an update did."""

    status: str  # "updated" | "up_to_date" | "dry_run"
    last_fit: str  # date of the latest stored fit (after the update)
    last_observation: str  # date of the latest price processed
    new_days: list[str] = field(default_factory=list)
    fits_added: int = 0
    alerts: list[dict[str, str]] = field(default_factory=list)  # alerts raised during this update
    alert_state: str = ""
    issues: list[DataIssue] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "issues": [i.__dict__ for i in self.issues]}


def _write_status(live_dir: Path, report: UpdateReport) -> None:
    live_dir.mkdir(parents=True, exist_ok=True)
    payload = {"checked": datetime.now(timezone.utc).isoformat(timespec="seconds"), **report.to_dict()}
    (live_dir / "last_update.json").write_text(json.dumps(payload, indent=1))


def update_run(
    cfg: Config, run_dir: str | Path, *, source: PriceSource | None = None, threads: int = 4,
    n_jobs: int | None = None, scan_step: int = 5, n_perm: int = 499, risk_step: int = 5,
    monitor_overrides: dict[str, Any] | None = None, dry_run: bool = False, force: bool = False,
    today: pd.Timestamp | None = None, out: Callable[[str], None] = print,
) -> UpdateReport:
    """Bring ``run_dir`` up to date with the latest prices (see the module docstring).

    ``dry_run`` fetches and checks but writes nothing. ``force`` skips the settings/history checks
    (not recommended). ``monitor_overrides`` replaces fields of `MonitorConfig` (``alpha``,
    ``enter_ratio``, ``exit_ratio``, ...). Raises ``ManifestMismatch`` if the stored run is
    incompatible with the settings or the data, ``ValueError`` if the new prices fail validation.
    """
    run_dir = Path(run_dir)
    ckpt = run_dir / "checkpoint.jsonl"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found; create the run first with `vine-risk run`.")
    if dry_run:  # read-only: no lock needed
        return _update(cfg, run_dir, source, threads, n_jobs, scan_step, n_perm, risk_step, monitor_overrides,
                       True, force, today, out)
    with run_lock(run_dir):
        return _update(cfg, run_dir, source, threads, n_jobs, scan_step, n_perm, risk_step, monitor_overrides,
                       False, force, today, out)


def _update(cfg, run_dir: Path, source, threads, n_jobs, scan_step, n_perm, risk_step, overrides, dry_run, force,
            today, out) -> UpdateReport:
    t0 = time.time()
    d = cfg.data
    source = source or make_source(d)
    if not force:
        check_resume(run_dir, cfg)
    idx = CheckpointIndex(run_dir / "checkpoint.jsonl")
    latest = idx.latest()
    if latest is None:
        raise ValueError(f"{run_dir}/checkpoint.jsonl holds no fits.")
    last_fit = pd.Timestamp(latest.timestamp)

    # 1. fetch and sanity-check
    raw = source.fetch(cfg.assets, d.start, d.end)
    issues = validate_prices(raw, cfg.assets, new_from=last_fit + pd.Timedelta(days=1), today=today)
    for i in issues:
        out(f"{'ERROR' if i.level == 'error' else 'warning'}: {i.message}")
    if any(i.level == "error" for i in issues):
        raise ValueError("The fetched prices failed validation: " + "; ".join(i.message for i in issues if i.level == "error"))
    prices = clean_prices(raw, d.max_missing_frac, d.max_ffill_days)
    returns = log_returns(prices)

    # 2. is the history behind the stored fits unchanged?
    if not force:
        model = RollingVineModel.from_config(cfg)
        verify_history_unchanged(latest, returns, marginal_factory=model.marginal_factory, lookback=model.lookback)
    # Resume after the last price the pipeline processed. With a refit frequency above 1 that is later
    # than the last fit; days in between were seen (and logged) before and must not be handled again.
    manifest = read_manifest(run_dir)
    seen = pd.Timestamp(manifest["data"]["last_date"]) if manifest and manifest.get("data") else last_fit
    processed_until = max(last_fit, seen)
    if processed_until not in returns.index:
        raise ManifestMismatch(f"The last processed day {processed_until.date()} is missing from the fetched prices.")
    new_days = returns.index[returns.index > processed_until]
    report = UpdateReport("dry_run" if dry_run else "up_to_date", str(last_fit.date()), str(returns.index[-1].date()),
                          [str(t.date()) for t in new_days], issues=issues)
    if dry_run or new_days.empty:
        out(f"{'would process' if dry_run else 'nothing to do:'} {len(new_days)} new day(s) after {processed_until.date()}"
            + (f" ({new_days[0].date()} to {new_days[-1].date()})" if len(new_days) else ""))
        report.seconds = time.time() - t0
        if not dry_run:
            _write_status(run_dir / "live", report)
        return report

    # 3. feed the new days through the monitor, resuming from the saved fits
    window, freq = cfg.rolling.window, cfg.rolling.refit_frequency
    mcfg = MonitorConfig.from_config(cfg, scan_step=scan_step, n_perm=n_perm, risk_every_fits=0)
    mcfg = replace(mcfg, vine_kwargs={**mcfg.vine_kwargs, "num_threads": threads}, **(overrides or {}))
    prior = [idx.get(t) for t in idx.timestamps[-(default_lag(window, freq) + 2):]]
    history = returns.loc[:processed_until]
    state = None
    scan_file = run_dir / "change_scan.parquet"
    if scan_file.exists():
        scan = pd.read_parquet(scan_file)
        scan.index = pd.DatetimeIndex(scan.index)
        state = alert_state_from_scans(scan, mcfg, processed_until)
    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    def append_fit(result) -> None:
        with open(run_dir / "checkpoint.jsonl", "a") as f:
            f.write(result.to_json() + "\n")

    monitor = LiveMonitor(list(prices.columns), mcfg, [JsonlSink(live_dir / "updates.jsonl"), ConsoleSink(every=1, out=out)],
                          on_fit=append_fit)
    monitor.prime(history, prior, alert_state=state, n_fits=len(idx))
    monitor.set_last_price(prices.loc[processed_until])
    for date in new_days:
        monitor.on_prices(date, prices.loc[date])
    new_alerts = [{"date": r.timestamp, "message": r.message} for r in monitor.records if r.new_alert]

    # 4. bring the tables up to date (incremental); then remember the prices we used
    run_pipeline(cfg, returns, run_dir, n_jobs=n_jobs, scan_step=scan_step, n_perm=n_perm, risk_step=risk_step,
                 command=["vine-risk", "update"])
    if source.cacheable and d.cache_dir:
        Path(d.cache_dir).mkdir(parents=True, exist_ok=True)
        raw.to_parquet(price_cache_path(cfg.assets, d.start, d.end, d.cache_dir))
    if new_alerts:
        alerts_file = live_dir / "alerts.csv"
        pd.DataFrame(new_alerts).to_csv(alerts_file, mode="a", header=not alerts_file.exists(), index=False)

    fit_dates = [r.timestamp for r in monitor.records if r.refit]
    report.status, report.fits_added = "updated", len(fit_dates)
    report.alerts, report.alert_state = new_alerts, monitor.alert_state
    report.last_fit = fit_dates[-1] if fit_dates else report.last_fit
    report.seconds = time.time() - t0
    _write_status(live_dir, report)
    out(f"updated: {len(new_days)} new day(s) up to {report.last_observation}, {report.fits_added} new fit(s), "
        f"{len(new_alerts)} new alert(s), alert state {report.alert_state}, {report.seconds:.0f}s")
    return report
