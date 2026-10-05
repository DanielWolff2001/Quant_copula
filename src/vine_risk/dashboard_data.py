"""Data access for the dashboard: load a finished run and answer the questions the panels ask.

Kept free of Streamlit so it can be tested. A "run" is the folder written by
``run_rolling.py`` and the scripts that follow it (``compute_metrics``, ``detect_changes``,
``compute_risk``); the panels that need a missing file are simply skipped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vine_risk.copula import VineFitResult  # noqa: F401  (re-exported for callers)
from vine_risk.rolling import CheckpointIndex  # noqa: F401  (moved to vine_risk.rolling)

SCRIPT_HINT = {
    "dependence_metrics.parquet": "python scripts/compute_metrics.py",
    "pairwise_tau.parquet": "python scripts/compute_metrics.py",
    "structural_change_scores.parquet": "python scripts/detect_changes.py",
    "change_scan.parquet": "python scripts/detect_changes.py",
    "portfolio_risk.parquet": "python scripts/compute_risk.py",
}


@dataclass
class RunData:
    """Everything the dashboard shows. Optional tables are ``None`` when not computed."""

    run_dir: Path
    assets: list[str]
    metrics: pd.DataFrame  # dependence_metrics.parquet, indexed by timestamp
    tau: pd.DataFrame  # empirical Kendall's tau, one column per pair "A-B"
    lower: pd.DataFrame | None = None  # model-implied lower-tail coefficient per pair
    upper: pd.DataFrame | None = None
    scores: pd.DataFrame | None = None
    scan: pd.DataFrame | None = None
    risk: pd.DataFrame | None = None
    backtest: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None
    window: int = 0  # observations per fit
    live: pd.DataFrame | None = None  # log of a simulated live replay (scripts/run_live.py)


def _read(run_dir: Path, name: str, required: bool = False) -> pd.DataFrame | None:
    path = run_dir / name
    if not path.exists():
        if required:
            hint = SCRIPT_HINT.get(name, "")
            raise FileNotFoundError(f"{path} not found" + (f"; create it with `{hint}`." if hint else "."))
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def read_first_fit(checkpoint: str | Path) -> dict:
    """The first stored fit as a dict (asset names, window length, ...)."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; create it with `python scripts/run_rolling.py`.")
    with open(path) as f:
        return json.loads(f.readline())


def read_assets(checkpoint: str | Path) -> list[str]:
    """Asset names, from the first stored fit (reliable even for tickers containing ``-``)."""
    return list(read_first_fit(checkpoint)["assets"])


def load_run(
    run_dir: str | Path, prices: pd.DataFrame | None = None, returns: pd.DataFrame | None = None,
) -> RunData:
    """Load the tables of a run folder. Raises ``FileNotFoundError`` with a hint if the
    two required tables (metrics and pairwise tau) are missing."""
    d = Path(run_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Run folder {d} does not exist; create it with `python scripts/run_rolling.py`.")
    bt = d / "var_backtest.csv"
    first = read_first_fit(d / "checkpoint.jsonl")
    return RunData(
        run_dir=d,
        assets=list(first["assets"]), window=int(first["n_obs"]),
        metrics=_read(d, "dependence_metrics.parquet", required=True),
        tau=_read(d, "pairwise_tau.parquet", required=True),
        lower=_read(d, "pairwise_lower_tail_q.parquet"),
        upper=_read(d, "pairwise_upper_tail_q.parquet"),
        scores=_read(d, "structural_change_scores.parquet"),
        scan=_read(d, "change_scan.parquet"),
        risk=_read(d, "portfolio_risk.parquet"),
        backtest=pd.read_csv(bt, index_col=0) if bt.exists() else None,
        prices=prices, returns=returns, live=_read(d, "live/live_log.parquet"),
    )


def nearest_fit_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    """The latest fit date on or before ``date`` (the model in force on that day)."""
    pos = index.searchsorted(pd.Timestamp(date), side="right") - 1
    if pos < 0:
        raise ValueError(f"No fit on or before {date}.")
    return index[pos]


def pair_column(frame: pd.DataFrame, a: str, b: str, assets: list[str]) -> str:
    """Name of the ``"A-B"`` column for two assets (in the order given by ``assets``)."""
    if a == b:
        raise ValueError("Choose two different assets.")
    first, second = sorted((a, b), key=assets.index)
    col = f"{first}-{second}"
    if col not in frame.columns:
        raise KeyError(f"Pair {col} not found.")
    return col


def tau_matrix_at(tau: pd.DataFrame, date: pd.Timestamp, assets: list[str]) -> pd.DataFrame:
    """Symmetric Kendall-tau matrix (diagonal 1) of the fit in force on ``date``."""
    row = tau.loc[nearest_fit_date(tau.index, date)]
    m = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    for i, a in enumerate(assets):
        for b in assets[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = row[f"{a}-{b}"]
    return m


def alert_periods(score: pd.Series, threshold: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Maximal runs of consecutive observations whose score exceeds ``threshold``."""
    flag = (score > threshold).fillna(False).to_numpy()
    out, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((score.index[start], score.index[i - 1]))
            start = None
    if start is not None:
        out.append((score.index[start], score.index[-1]))
    return out


def prices_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a price-like panel (start = 1) from log returns, for runs without prices."""
    return np.exp(returns.cumsum())
