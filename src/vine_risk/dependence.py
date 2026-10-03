"""Pairwise dependence measures."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def pairwise_dependence(u: pd.DataFrame, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Kendall's tau, Spearman's rho and Pearson correlation for every asset pair.

    ``tau`` and ``spearman`` are rank-based, so they are identical on returns and on
    the pseudo-observations ``u``; they are computed from ``u``. ``pearson`` is a
    linear correlation of the *returns* and is NaN when ``returns`` is not given.

    Returns a long frame with columns ``asset_i, asset_j, tau, spearman, pearson``
    (one row per pair ``i < j`` in column order).
    """
    if returns is not None and not returns.columns.equals(u.columns):
        raise ValueError("returns and u must have the same columns.")
    rows = []
    for a, b in combinations(u.columns, 2):
        tau = stats.kendalltau(u[a], u[b]).statistic
        rho = stats.spearmanr(u[a], u[b]).statistic
        pear = np.corrcoef(returns[a], returns[b])[0, 1] if returns is not None else np.nan
        rows.append((a, b, tau, rho, pear))
    return pd.DataFrame(rows, columns=["asset_i", "asset_j", "tau", "spearman", "pearson"])
