"""Live monitoring: feed observations one at a time and get a monitoring record back.

Workflow per observation (PDF section 14)::

    new price -> new return -> update rolling window -> refit if due
        -> dependence metrics -> comparison with the previous model
        -> structural-change scores (+ permutation test) -> alert state -> output

The monitor only ever sees data up to the current date. It is built from the same pieces
as the batch pipeline (`vine_risk.rolling.RollingVineModel`,
`vine_risk.dependence.dependence_metrics`, `vine_risk.change_detection.change_scan`'s
test, `vine_risk.portfolio.window_risk`), so replaying history through it reproduces
the batch results exactly; ``tests/test_monitor.py`` checks this.

Alerts come from the permutation test between the last two windows (calibrated by
construction and independent of the number of assets, unlike a raw distance threshold). An
alert starts when the test is significant *and* the effect is at least ``enter_ratio`` times
the typical no-change distance, and ends when the effect falls below ``exit_ratio``; the
two thresholds (hysteresis) avoid flickering around a single cut-off.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from vine_risk.change_detection import default_lag, frobenius_change, model_distance, two_window_change_test
from vine_risk.config import Config
from vine_risk.copula import VineFitResult
from vine_risk.dependence import dependence_metrics, pairwise_series
from vine_risk.portfolio import window_risk
from vine_risk.marginals import MarginalSpec
from vine_risk.rolling import RollingVineModel

logger = logging.getLogger(__name__)

NORMAL, ALERT, WARMING_UP = "normal", "alert", "warming_up"


@dataclass(frozen=True)
class MonitorConfig:
    """Settings of the live monitor (all with sensible defaults)."""

    window: int = 250
    refit_frequency: int = 1
    vine_kwargs: dict = field(default_factory=dict)
    marginal: str = "empirical"  # see MarginalSpec
    marginal_lookback: int | None = None  # None: window
    scan_step: int = 5  # run the permutation test every N observations
    n_perm: int = 199
    scan_q: float = 0.1  # level of the empirical tail coefficients in the test
    block: int = 10  # permute blocks of this many days (keeps volatility clustering)
    alpha: float = 0.01
    enter_ratio: float = 2.0  # alert starts: significant and distance >= enter_ratio x no-change level
    exit_ratio: float = 1.5  # alert ends: distance < exit_ratio x no-change level
    risk_every_fits: int = 5  # compute VaR/ES at every N-th fit (0 = never)
    risk_alpha: float = 0.99
    risk_sims: int = 2 ** 14
    weights: tuple[float, ...] | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.window < 2 or self.refit_frequency < 1 or self.scan_step < 1:
            raise ValueError("window >= 2, refit_frequency >= 1 and scan_step >= 1 are required.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if self.exit_ratio > self.enter_ratio:
            raise ValueError("exit_ratio must not exceed enter_ratio.")
        if self.risk_every_fits < 0:
            raise ValueError("risk_every_fits must be >= 0.")

    @classmethod
    def from_config(cls, cfg: Config, **overrides: Any) -> "MonitorConfig":
        r = cfg.rolling
        kw: dict[str, Any] = dict(
            window=r.window, refit_frequency=r.refit_frequency, marginal=r.marginal, marginal_lookback=r.lookback,
            vine_kwargs=dict(selection_criterion=r.selection_criterion, truncation_level=r.truncation_level,
                             tail_simulations=r.tail_simulations, tail_level=r.tail_level, seed=cfg.risk.seed),
            risk_alpha=cfg.risk.confidence_level, risk_sims=cfg.risk.simulations, seed=cfg.risk.seed,
            weights=None if cfg.risk.weights is None else tuple(cfg.risk.weights),
        )
        kw.update(overrides)
        return cls(**kw)


@dataclass
class MonitorRecord:
    """What the monitor reports after one observation."""

    timestamp: str
    position: int  # row index of the observation in the return history
    refit: bool
    fit_status: str  # "ok" | "failed" | "none" (no refit on this date)
    metrics: dict[str, float] = field(default_factory=dict)  # dependence + structure vs previous fit
    scores: dict[str, float] = field(default_factory=dict)  # distance to the fit one window earlier
    scan: dict[str, float] | None = None  # permutation test (on scan dates)
    risk: dict[str, float] | None = None  # VaR / ES (on risk dates)
    alert_state: str = WARMING_UP
    new_alert: bool = False
    message: str = ""

    def to_row(self) -> dict[str, Any]:
        """Flat dictionary (one column per quantity) for tables and logs."""
        row: dict[str, Any] = {"timestamp": self.timestamp, "position": self.position, "refit": self.refit,
                               "fit_status": self.fit_status, "alert_state": self.alert_state,
                               "new_alert": self.new_alert, "message": self.message}
        for prefix, d in (("m_", self.metrics), ("s_", self.scores), ("scan_", self.scan), ("risk_", self.risk)):
            row.update({f"{prefix}{k}": v for k, v in (d or {}).items()})
        return row


Sink = Callable[[MonitorRecord], None]


def next_alert_state(state: str, p_tau: float, ratio: float, cfg: MonitorConfig) -> tuple[str, bool, str]:
    """One step of the alert rule (with hysteresis) after a permutation test.

    Returns ``(new_state, new_alert, message)``. An alert starts when the test is significant
    and the effect is at least ``enter_ratio`` times the no-change level; it ends when the
    effect falls below ``exit_ratio`` (significance is not needed to *stay* in alert).
    """
    if state != ALERT and p_tau <= cfg.alpha and ratio >= cfg.enter_ratio:
        return ALERT, True, (f"dependence differs from the previous window (tau-matrix distance {ratio:.1f}x "
                             f"the no-change level, p={p_tau:.3f})")
    if state == ALERT and not ratio >= cfg.exit_ratio:
        return NORMAL, False, "alert ended"
    return (NORMAL if state == WARMING_UP else state), False, ""


def alert_state_from_scans(scan: pd.DataFrame, cfg: MonitorConfig, until: pd.Timestamp | None = None) -> str:
    """Alert state after replaying stored permutation-test results (``change_scan.parquet``)
    up to ``until``; use it to resume a monitor in the right state. The scans must have been
    made with the same settings as the monitor (the p-values depend on ``n_perm``)."""
    state = WARMING_UP
    rows = scan if until is None else scan.loc[: pd.Timestamp(until)]
    for _, r in rows.sort_index().iterrows():
        ratio = r["stat_tau"] / r["null_mean_tau"] if r["null_mean_tau"] > 0 else float("nan")
        state, _, _ = next_alert_state(state, r["p_tau"], ratio, cfg)
    return state


class JsonlSink:
    """Appends every record as one JSON line (the monitoring output / audit log)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: MonitorRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_row(), default=float) + "\n")


class LiveMonitor:
    """Sequential dependence monitor.

    Args:
        assets: asset names (column order of every observation).
        config: `MonitorConfig`.
        sinks: callables invoked with every `MonitorRecord`.
        on_fit: optional callable invoked with every new `vine_risk.copula.VineFitResult` right
            after it is made (e.g. to append it to a checkpoint file).
    """

    def __init__(self, assets: Sequence[str], config: MonitorConfig, sinks: Sequence[Sink] = (),
                 on_fit: Callable[[VineFitResult], None] | None = None) -> None:
        self.assets = list(assets)
        self.cfg = config
        self.sinks = list(sinks)
        self.on_fit = on_fit
        self.model = RollingVineModel(config.window, config.refit_frequency, config.vine_kwargs,
                                      MarginalSpec(config.marginal), config.marginal_lookback)
        self.lag = default_lag(config.window, config.refit_frequency)  # in fits
        self.position = -1  # row index of the latest observation
        self.alert_state = WARMING_UP
        self._n_fits = 0  # fits made so far (full history, results are trimmed)
        self._returns = pd.DataFrame(columns=self.assets, index=pd.DatetimeIndex([]), dtype=float)  # last 2 windows
        self._last_price: pd.Series | None = None
        self._last_scan_state: dict[str, float] | None = None
        self.records: list[MonitorRecord] = []

    # ---- resuming ---------------------------------------------------------------------
    def prime(self, history: pd.DataFrame, results: Sequence[VineFitResult] = (),
              alert_state: str | None = None, n_fits: int | None = None) -> None:
        """Resume after ``history`` (a full return history, oldest first) without refitting.

        ``results`` are the fits already made, e.g. read from a checkpoint; only those on or
        before the last history date are allowed. ``alert_state`` restores the alert flag
        (default: warming up / normal). ``n_fits`` is the total number of fits made so far when
        ``results`` holds only the most recent ones (it keeps the VaR/ES schedule aligned).
        """
        if list(history.columns) != self.assets:
            raise ValueError("History columns do not match the monitor's assets.")
        self.model.prime(history, results)
        self.position = len(history) - 1
        self._returns = history.iloc[-2 * self.cfg.window:].copy()
        self._n_fits = len(results) if n_fits is None else n_fits
        self.model.results = self.model.results[-(self.lag + 2):]
        self.alert_state = alert_state or (NORMAL if len(history) >= 2 * self.cfg.window else WARMING_UP)

    # ---- observations -----------------------------------------------------------------
    def on_prices(self, date: pd.Timestamp, prices: pd.Series) -> MonitorRecord | None:
        """New (adjusted) price vector: compute the log return and process it.

        The first price only initialises the previous-price state and returns ``None``.
        """
        if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any():
            raise ValueError(f"Invalid prices at {date}.")
        prev, self._last_price = self._last_price, prices.copy()
        if prev is None:
            return None
        return self.on_return(date, np.log(prices / prev))

    def set_last_price(self, prices: pd.Series) -> None:
        """Declare the most recent price (needed before the first `on_prices` after `prime`)."""
        self._last_price = prices.copy()

    def on_return(self, date: pd.Timestamp, row: pd.Series) -> MonitorRecord:
        """Process one new log-return vector."""
        date = pd.Timestamp(date)
        row = row[self.assets]
        self.model.push(date, row)  # data update
        self.position += 1
        self._returns = pd.concat([self._returns, row.to_frame().T.set_axis([date])]).iloc[-2 * self.cfg.window:]
        rec = MonitorRecord(str(date.date()), self.position, False, "none", alert_state=self.alert_state)
        if self.model.should_refit():  # model refit, separate from the data update
            self._on_refit(self.model.refit(), rec)
        self._maybe_scan(rec)
        rec.alert_state = self.alert_state
        self.records.append(rec)
        for sink in self.sinks:
            sink(rec)
        return rec

    # ---- internals --------------------------------------------------------------------
    def _on_refit(self, cur: VineFitResult, rec: MonitorRecord) -> None:
        self._n_fits += 1
        if self.on_fit is not None:
            self.on_fit(cur)
        rec.refit, rec.fit_status = True, cur.status
        res = self.model.results
        if cur.status != "ok":
            rec.message = f"fit failed: {cur.message}"
        else:
            prev = res[-2] if len(res) >= 2 else None
            ref = [prev, cur] if prev is not None else [cur]
            rec.metrics = {k: float(v) for k, v in dependence_metrics(ref).iloc[-1].items()
                           if k != "status" and pd.notna(v)}
            if len(res) > self.lag:  # fit one window earlier
                old = res[-1 - self.lag]
                pair = [old, cur]
                dist = model_distance(pair, lag=1).iloc[-1]
                rec.scores = {"s_tau": float(frobenius_change(pairwise_series(pair, "tau"), 1).iloc[-1]),
                              **{k: float(v) for k, v in dist.items()}}
            k = self.cfg.risk_every_fits
            if k and (self._n_fits - 1) % k == 0:
                w = None if self.cfg.weights is None else np.asarray(self.cfg.weights, dtype=float)
                buf = self.model._buffer  # the marginal's history; the vine's window is its last n_obs rows
                marg = None if self.cfg.marginal == "empirical" else self.model.marginal_factory().fit(buf)
                rec.risk = window_risk(cur, buf.iloc[-cur.n_obs:], w, self.cfg.risk_alpha, self.cfg.risk_sims,
                                       self.cfg.seed, marginal=marg)
        del res[: max(0, len(res) - (self.lag + 2))]  # keep only what the comparisons need

    def _maybe_scan(self, rec: MonitorRecord) -> None:
        W, c = self.cfg.window, self.cfg
        if len(self._returns) < 2 * W or (self.position - (2 * W - 1)) % c.scan_step != 0:
            return
        out = two_window_change_test(self._returns.to_numpy(dtype=float), W, c.n_perm, c.scan_q, c.seed, c.block)
        ratio = out["stat_tau"] / out["null_mean_tau"] if out["null_mean_tau"] > 0 else float("nan")
        out["ratio_tau"] = float(ratio)
        rec.scan = out
        self.alert_state, rec.new_alert, msg = next_alert_state(self.alert_state, out["p_tau"], ratio, c)
        rec.message = msg or rec.message


def records_to_frame(records: Sequence[MonitorRecord]) -> pd.DataFrame:
    """Records as a table indexed by timestamp (one row per observation)."""
    df = pd.DataFrame([r.to_row() for r in records])
    if df.empty:
        return df
    lead = ["position", "refit", "fit_status", "alert_state", "new_alert", "message"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    rest = sorted(c for c in df.columns if c not in lead and c != "timestamp")
    return df.set_index("timestamp")[lead + rest]  # stable column order, whatever came first


class ConsoleSink:
    """Prints a one-line summary on refit dates every ``every`` observations, and always on alerts."""

    def __init__(self, every: int = 20, out: Callable[[str], None] = print) -> None:
        self.every, self.out, self._n = every, out, 0

    def __call__(self, r: MonitorRecord) -> None:
        self._n += 1
        if r.new_alert or r.message or self._n % self.every == 0:
            d = r.metrics.get("d_t")
            tail = f" D_t={d:.3f}" if d is not None else ""
            es = f" ES={100 * r.risk['es_vine']:.2f}%" if r.risk else ""
            flag = " ** ALERT **" if r.new_alert else ""
            self.out(f"{r.timestamp} [{r.alert_state}]{tail}{es}{flag} {r.message}".rstrip())
