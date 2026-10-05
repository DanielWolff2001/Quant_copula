"""Price download (with on-disk cache) and cleaning."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vine_risk.sources import PriceSource, YahooSource

logger = logging.getLogger(__name__)


def price_cache_path(tickers: list[str], start: str, end: str | None, cache_dir: str | Path) -> Path:
    """The cache file for a set of tickers and dates (``end=None`` means "up to the latest day")."""
    key = "_".join(sorted(tickers)) + f"_{start}_{end or 'latest'}"
    return Path(cache_dir) / f"prices_{key}.parquet"


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path | None = None,
    *,
    source: PriceSource | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted daily close prices from a source (default: Yahoo Finance), cached as parquet.

    Cached results make repeated runs reproducible and offline-capable. The cache is **not** updated
    by itself: pass ``refresh=True`` to download again and overwrite it (a daily update does this).
    Sources that read local files (:class:`~vine_risk.sources.CsvSource`) are never cached.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    source = source or YahooSource()
    cache_file = None
    if cache_dir is not None and source.cacheable:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = price_cache_path(tickers, start, end, cache_dir)
        if cache_file.is_file() and not refresh:
            logger.info("Loading cached prices from %s", cache_file)
            return pd.read_parquet(cache_file)
    prices = source.fetch(tickers, start, end)
    if cache_file is not None:
        prices.to_parquet(cache_file)
    return prices


@dataclass(frozen=True)
class DataIssue:
    """A problem found in a price panel. ``level`` is ``"error"`` (do not use) or ``"warning"``."""

    level: str
    message: str


def validate_prices(
    prices: pd.DataFrame,
    tickers: list[str],
    *,
    new_from: pd.Timestamp | None = None,
    max_jump: float = 0.25,
    max_stale_bdays: int = 5,
    today: pd.Timestamp | None = None,
) -> list[DataIssue]:
    """Sanity checks on freshly fetched prices.

    Errors: no data, a missing or entirely empty ticker, non-positive prices on the new days.
    Warnings: a ticker without a price on some new day (cleaning will carry the last price forward),
    a one-day move of more than ``max_jump`` in log terms on a new day (possibly an unadjusted split),
    and a last date more than ``max_stale_bdays`` business days before ``today`` (stalled feed).
    ``new_from`` limits the day-level checks to dates on or after it.
    """
    issues: list[DataIssue] = []
    if prices is None or prices.empty:
        return [DataIssue("error", "no price data")]
    absent = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if absent:
        issues.append(DataIssue("error", f"no prices for: {absent}"))
    present = [t for t in tickers if t not in absent]
    px = prices[present].sort_index()
    recent = px if new_from is None else px.loc[px.index >= new_from]
    if (recent <= 0).any().any():
        bad = recent.columns[(recent <= 0).any()].tolist()
        issues.append(DataIssue("error", f"non-positive prices on new days for: {bad}"))
    gaps = recent.isna().any(axis=1)
    if gaps.any():
        issues.append(DataIssue("warning", f"{int(gaps.sum())} new day(s) lack a price for some ticker "
                                           f"(first: {recent.index[gaps][0].date()}); the last price is carried forward"))
    jumps = np.log(px.where(px > 0) / px.where(px > 0).shift(1)).abs()
    jumps = jumps if new_from is None else jumps.loc[jumps.index >= new_from]
    big = jumps.stack()[lambda s: s > max_jump] if not jumps.empty else pd.Series(dtype=float)
    for (date, ticker), v in big.items():
        issues.append(DataIssue("warning", f"{ticker} moved {v:.0%} (log) on {date.date()}: check for an unadjusted split"))
    if today is not None:
        late = len(pd.bdate_range(px.index[-1], pd.Timestamp(today))) - 1
        if late > max_stale_bdays:
            issues.append(DataIssue("warning", f"last price is from {px.index[-1].date()}, {late} business days "
                                               f"before {pd.Timestamp(today).date()}: is the feed stalled?"))
    return issues


def clean_prices(
    prices: pd.DataFrame,
    max_missing_frac: float = 0.05,
    max_ffill_days: int = 3,
) -> pd.DataFrame:
    """Clean a price panel.

    Sorts and de-duplicates the index, treats non-positive prices as missing, drops
    tickers with too many missing values, forward-fills short gaps, and keeps only
    dates where every remaining ticker has a price (so the history starts once the
    youngest asset exists).
    """
    if prices.empty:
        raise ValueError("Empty price frame.")
    df = prices.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.where(df > 0)
    df = df.dropna(how="all")

    missing = df.isna().mean()
    bad = missing[missing > max_missing_frac].index.tolist()
    if bad:
        logger.warning("Dropping tickers with >%.0f%% missing prices: %s", max_missing_frac * 100, bad)
        df = df.drop(columns=bad)
    if df.shape[1] == 0:
        raise ValueError("All tickers were dropped during cleaning.")

    df = df.ffill(limit=max_ffill_days).dropna(how="any")
    return df
