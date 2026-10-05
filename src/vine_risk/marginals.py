"""Marginal models: map returns to (approximately) uniform pseudo-observations.

The copula engine only consumes the uniform output, so a parametric marginal
(Student-t, skewed, GARCH, ...) can be added by subclassing `Marginal`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats


class Marginal(ABC):
    """Per-asset probability integral transform ``u = F(r)`` and its inverse."""

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "Marginal":
        """Estimate the marginal of each column of ``returns``."""

    @abstractmethod
    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Map returns to uniforms in (0, 1) using the fitted marginals."""

    @abstractmethod
    def inverse_transform(self, u: pd.DataFrame) -> pd.DataFrame:
        """Map uniforms in (0, 1) back to returns (needed for simulation)."""

    def fit_transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        return self.fit(returns).transform(returns)


class EmpiricalMarginal(Marginal):
    """Rank-based empirical CDF, ``u = rank(r) / (T + 1)``.

    On the fitting sample this is exactly the rank transform (ties get average
    ranks), which keeps ``u`` strictly inside (0, 1). New observations are mapped
    through the same sample ECDF, so the transform uses no information beyond the
    fitting window.
    """

    def __init__(self) -> None:
        self._sorted: dict[str, np.ndarray] = {}
        self._columns: list[str] = []

    def fit(self, returns: pd.DataFrame) -> "EmpiricalMarginal":
        _check_finite(returns)
        self._columns = list(returns.columns)
        self._sorted = {c: np.sort(returns[c].to_numpy()) for c in self._columns}
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted(returns.columns)
        _check_finite(returns)
        out = {}
        for c in self._columns:
            s = self._sorted[c]
            x = returns[c].to_numpy()
            # average rank among the sample: (#<x + #<=x + 1) / 2 for in-sample values
            lo = np.searchsorted(s, x, side="left")
            hi = np.searchsorted(s, x, side="right")
            rank = (lo + hi + 1) / 2.0
            out[c] = np.clip(rank, 1.0, len(s)) / (len(s) + 1)
        return pd.DataFrame(out, index=returns.index, columns=self._columns)

    def inverse_transform(self, u: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted(u.columns)
        out = {}
        for c in self._columns:
            s = self._sorted[c]
            n = len(s)
            grid = np.arange(1, n + 1) / (n + 1)
            out[c] = np.interp(u[c].to_numpy(), grid, s)  # clamps in the tails
        return pd.DataFrame(out, index=u.index, columns=self._columns)

    def _check_fitted(self, columns: pd.Index) -> None:
        if not self._columns:
            raise RuntimeError("Marginal is not fitted.")
        if list(columns) != self._columns:
            raise ValueError(f"Columns {list(columns)} do not match fitted {self._columns}.")


def _check_finite(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Empty return frame.")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Returns contain NaN or infinite values.")


def uniformity_report(u: pd.DataFrame) -> pd.DataFrame:
    """Kolmogorov-Smirnov test of U(0,1) per column (a sanity check, not a model test)."""
    rows = {c: stats.kstest(u[c].to_numpy(), "uniform") for c in u.columns}
    return pd.DataFrame(
        {"ks_stat": {c: r.statistic for c, r in rows.items()},
         "p_value": {c: r.pvalue for c, r in rows.items()}}
    )
