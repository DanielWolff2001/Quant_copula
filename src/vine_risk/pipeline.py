"""Convenience functions that chain the data steps (used by the notebooks)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vine_risk.config import Config
from vine_risk.data import clean_prices, download_prices
from vine_risk.returns import log_returns
from vine_risk.sources import PriceSource, make_source


def load_prices_and_returns(cfg: Config, *, refresh: bool = False,
                            source: PriceSource | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or read from the cache), clean and convert to log returns.

    ``refresh=True`` downloads again and overwrites the price cache; ``source`` overrides the source
    selected in the config.

    Returns ``(prices, returns)``: the cleaned adjusted prices and the model-ready log-return
    matrix (one row fewer than ``prices``).
    """
    d = cfg.data
    raw = download_prices(cfg.assets, d.start, d.end, d.cache_dir, source=source or make_source(d), refresh=refresh)
    prices = clean_prices(raw, d.max_missing_frac, d.max_ffill_days)
    return prices, log_returns(prices)


def find_run(root: str | Path = "data/results", prefer: tuple[str, ...] = ("w250", "demo")) -> Path:
    """The first existing run folder (one containing ``checkpoint.jsonl``) among ``prefer``.

    Raises ``FileNotFoundError`` with instructions if none exists.
    """
    base = Path(root)
    for name in prefer:
        if (base / name / "checkpoint.jsonl").is_file():
            return base / name
    raise FileNotFoundError(
        f"No finished run found in {base}/{{{','.join(prefer)}}}. Create one with "
        "`python scripts/run_rolling.py` (about an hour for 20 years) or run notebook 03, "
        "which makes a small demo run in a minute or two.")
