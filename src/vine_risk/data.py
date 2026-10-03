"""Price download (with on-disk cache) and cleaning."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download split- and dividend-adjusted daily close prices.

    Results are cached as parquet so repeated runs are reproducible and offline-capable.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    import yfinance as yf

    cache_file = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key = "_".join(sorted(tickers)) + f"_{start}_{end or 'latest'}"
        cache_file = Path(cache_dir) / f"prices_{key}.parquet"
        if cache_file.is_file():
            logger.info("Loading cached prices from %s", cache_file)
            return pd.read_parquet(cache_file)

    logger.info("Downloading %d tickers from Yahoo Finance", len(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {tickers}")
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise RuntimeError(f"Download returned no data for: {missing}")
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    if cache_file is not None:
        prices.to_parquet(cache_file)
    return prices


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
