"""Where prices come from.

A *source* is anything with ``fetch(tickers, start, end)`` returning adjusted daily close prices
(index = dates, one column per ticker). Two are built in:

* `YahooSource` - Yahoo Finance through ``yfinance`` (the default);
* `CsvSource` - your own files, so any vendor's data can be used: a wide CSV (``Date`` column
  plus one column per ticker) or a folder with one ``<TICKER>.csv`` per ticker.

Select one in the config (``data: source: csv`` and ``csv_path: ...``) or pass an object to the
functions that take a ``source``. Prices must be **adjusted** for splits and dividends.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)
PRICE_COLUMNS = ("Adj Close", "adj_close", "Adj_Close", "Close", "close")


@runtime_checkable
class PriceSource(Protocol):
    """Anything that can deliver adjusted daily close prices."""

    name: str
    cacheable: bool  # True if repeated calls may be served from the local price cache

    def fetch(self, tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame: ...


class YahooSource:
    """Adjusted daily closes from Yahoo Finance (``yfinance``)."""

    name, cacheable = "yahoo", True

    def fetch(self, tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        tickers = list(tickers)
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
        return prices


class CsvSource:
    """Prices from local CSV files (never cached: the files are read every time).

    ``path`` is either one wide CSV (first column = date, one column per ticker) or a folder with
    one ``<TICKER>.csv`` per ticker having a ``Date`` column and an adjusted-close column
    (``Adj Close`` is preferred over ``Close``).
    """

    name, cacheable = "csv", False

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
        tickers = list(tickers)
        if not self.path.exists():
            raise FileNotFoundError(f"CSV price source {self.path} does not exist.")
        prices = self._read_folder(tickers) if self.path.is_dir() else self._read_wide(tickers)
        prices = prices.sort_index().loc[pd.Timestamp(start): (pd.Timestamp(end) if end else None)]
        prices.index.name = "Date"  # same as the Yahoo source
        if prices.empty:
            raise RuntimeError(f"{self.path} has no prices between {start} and {end or 'today'}.")
        return prices

    def _read_wide(self, tickers: list[str]) -> pd.DataFrame:
        df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise RuntimeError(f"{self.path} has no column for: {missing}")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz else pd.DatetimeIndex(df.index)
        return df[tickers].astype(float)

    def _read_folder(self, tickers: list[str]) -> pd.DataFrame:
        cols = {}
        for t in tickers:
            f = self.path / f"{t}.csv"
            if not f.is_file():
                raise RuntimeError(f"{self.path} has no file {t}.csv")
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            col = next((c for c in PRICE_COLUMNS if c in df.columns), None)
            if col is None:
                raise RuntimeError(f"{f} has none of the price columns {PRICE_COLUMNS}")
            cols[t] = df[col].astype(float)
        return pd.DataFrame(cols)


def make_source(data_cfg) -> PriceSource:
    """The source selected by a `vine_risk.config.DataConfig`."""
    kind = getattr(data_cfg, "source", "yahoo")
    if kind == "yahoo":
        return YahooSource()
    if kind == "csv":
        if not getattr(data_cfg, "csv_path", None):
            raise ValueError("data.source is 'csv' but data.csv_path is not set.")
        return CsvSource(data_cfg.csv_path)
    raise ValueError(f"Unknown data.source {kind!r}; choose 'yahoo' or 'csv'.")
