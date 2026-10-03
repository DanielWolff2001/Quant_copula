"""Log returns and the model-ready return matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = log(P_t / P_{t-1}); the first row (undefined) is dropped."""
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive to compute log returns.")
    return np.log(prices / prices.shift(1)).iloc[1:]


def build_return_matrix(
    prices: pd.DataFrame,
    max_missing_frac: float = 0.05,
    max_ffill_days: int = 3,
) -> pd.DataFrame:
    """Full pipeline: cleaning -> log returns -> missing-data handling.

    Returns a finite T x d return matrix (rows = dates, columns = assets).
    """
    from vine_risk.data import clean_prices

    clean = clean_prices(prices, max_missing_frac, max_ffill_days)
    rets = log_returns(clean).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if rets.empty:
        raise ValueError("No complete return observations after cleaning.")
    return rets
