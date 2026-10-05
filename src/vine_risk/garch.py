"""GARCH-filtered marginals.

Daily returns have volatility that clusters, so a copula fitted to *raw* returns mixes two things: how
volatile the market is, and how assets depend on each other. Filtering each asset with a GARCH(1,1)
model removes the first: the standardised residuals ``(r_t - mu) / sigma_t`` have roughly constant
variance, and their probability integral transform is what the copula is fitted to.

:class:`GarchMarginal` implements the :class:`~vine_risk.marginals.Marginal` interface, so it can replace
the rank transform without any change to the copula code. It uses the ``arch`` package for the
univariate fits. Two things differ from :class:`~vine_risk.marginals.EmpiricalMarginal`:

* it is fitted on a (long) history and may transform any sub-window of it, because GARCH parameters are
  poorly determined by 250 observations;
* ``inverse_transform`` returns the *conditional* one-day-ahead distribution, ``mu + sigma_{T+1} * z(u)``,
  so scenarios drawn through it are forecasts for tomorrow given today's volatility.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from vine_risk.marginals import Marginal

logger = logging.getLogger(__name__)

SCALE = 100.0  # arch works best with returns in percent
EWMA_LAMBDA = 0.94


def _ewma_filter(y: np.ndarray, lam: float = EWMA_LAMBDA) -> tuple[np.ndarray, float]:
    """Conditional volatility from a RiskMetrics EWMA; returns the in-sample path and the next-day value."""
    var = np.empty(len(y) + 1)
    var[0] = np.var(y[: min(len(y), 30)]) or 1e-8
    for t in range(len(y)):
        var[t + 1] = lam * var[t] + (1 - lam) * y[t] ** 2
    return np.sqrt(var[:-1]), float(np.sqrt(var[-1]))


class GarchMarginal(Marginal):
    """Per-asset GARCH(1,1) with a constant mean; transform = PIT of the standardised residuals.

    Args:
        innovations: ``"t"`` (standardised Student-t, default; its degrees of freedom are estimated) or
            ``"empirical"`` (the empirical distribution of the residuals, which is rank-based).
        min_obs: minimum number of observations per asset.

    If a fit fails (no convergence, non-finite parameters) that asset falls back to a RiskMetrics EWMA
    volatility with normal innovations; ``fallbacks`` lists the affected assets.
    """

    def __init__(self, innovations: str = "t", min_obs: int = 100) -> None:
        if innovations not in ("t", "empirical"):
            raise ValueError("innovations must be 't' or 'empirical'.")
        self.innovations, self.min_obs = innovations, min_obs
        self._columns: list[str] = []
        self.index: pd.DatetimeIndex | None = None
        self.mu: dict[str, float] = {}
        self.sigma: dict[str, np.ndarray] = {}  # in-sample conditional volatility (return units)
        self.sigma_next: dict[str, float] = {}  # one-day-ahead volatility (return units)
        self.resid: dict[str, np.ndarray] = {}  # standardised residuals
        self.nu: dict[str, float] = {}  # degrees of freedom (inf for the EWMA fallback)
        self.params: dict[str, dict[str, float]] = {}
        self.fallbacks: list[str] = []

    # ---- fitting --------------------------------------------------------------
    def fit(self, returns: pd.DataFrame) -> "GarchMarginal":
        from arch import arch_model

        if returns.empty or not np.isfinite(returns.to_numpy()).all():
            raise ValueError("Returns must be a non-empty frame without NaN or infinite values.")
        if len(returns) < self.min_obs:
            raise ValueError(f"Need at least {self.min_obs} observations, got {len(returns)}.")
        self._columns, self.index = list(returns.columns), returns.index
        self.fallbacks = []
        for c in self._columns:
            y = returns[c].to_numpy(dtype=float) * SCALE
            try:
                res = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1,
                                 dist="t" if self.innovations == "t" else "normal").fit(disp="off", show_warning=False)
                p = res.params
                ok = (res.convergence_flag == 0 and np.isfinite(p).all() and p["omega"] > 0
                      and p["alpha[1]"] >= 0 and p["beta[1]"] >= 0 and p["alpha[1]"] + p["beta[1]"] < 1.0)
                if not ok:
                    raise RuntimeError("non-converged or non-stationary fit")
                sigma = np.asarray(res.conditional_volatility)
                mu = float(p["mu"])
                nxt = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]))
                nu = float(p["nu"]) if self.innovations == "t" else float("inf")
                self.params[c] = {k: float(v) for k, v in p.items()}
            except Exception as e:  # arch raises a variety of errors; the fallback keeps the pipeline alive
                logger.warning("GARCH fit failed for %s (%s); using an EWMA volatility.", c, e)
                self.fallbacks.append(c)
                mu = float(y.mean())
                sigma, nxt = _ewma_filter(y - mu)
                nu, self.params[c] = float("inf"), {"mu": mu}
            self.mu[c] = mu / SCALE
            self.sigma[c] = sigma / SCALE
            self.sigma_next[c] = nxt / SCALE
            self.nu[c] = nu
            self.resid[c] = (y - mu) / sigma
        return self

    # ---- standardised-innovation distribution ----------------------------------------
    def _cdf(self, c: str, z: np.ndarray) -> np.ndarray:
        nu = self.nu[c]
        if self.innovations == "t" and np.isfinite(nu):
            return stats.t.cdf(z * np.sqrt(nu / (nu - 2)), nu)
        if self.innovations == "empirical":
            s = np.sort(self.resid[c])
            lo, hi = np.searchsorted(s, z, side="left"), np.searchsorted(s, z, side="right")
            return np.clip((lo + hi + 1) / 2.0, 1.0, len(s)) / (len(s) + 1)
        return stats.norm.cdf(z)

    def _ppf(self, c: str, u: np.ndarray) -> np.ndarray:
        nu = self.nu[c]
        if self.innovations == "t" and np.isfinite(nu):
            return stats.t.ppf(u, nu) / np.sqrt(nu / (nu - 2))
        if self.innovations == "empirical":
            s = np.sort(self.resid[c])
            return np.interp(u, np.arange(1, len(s) + 1) / (len(s) + 1), s)
        return stats.norm.ppf(u)

    # ---- Marginal interface -----------------------------------------------------------
    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """PIT of the standardised residuals for dates in the fitted history (any sub-window of it)."""
        self._check(returns.columns)
        if self.index is None or not returns.index.isin(self.index).all():
            raise ValueError("transform() only accepts dates that were in the fitted history.")
        pos = self.index.get_indexer(returns.index)
        u = {c: np.clip(self._cdf(c, self.resid[c][pos]), 1e-6, 1 - 1e-6) for c in self._columns}
        return pd.DataFrame(u, index=returns.index, columns=self._columns)

    def inverse_transform(self, u: pd.DataFrame) -> pd.DataFrame:
        """One-day-ahead *conditional* returns: ``mu + sigma_{T+1} * z(u)`` (not the in-sample distribution)."""
        self._check(u.columns)
        out = {c: self.mu[c] + self.sigma_next[c] * self._ppf(c, np.clip(u[c].to_numpy(), 1e-9, 1 - 1e-9))
               for c in self._columns}
        return pd.DataFrame(out, index=u.index, columns=self._columns)

    def standardised_residuals(self) -> pd.DataFrame:
        """The fitted history's standardised residuals (rows = dates), e.g. for filtered historical simulation."""
        return pd.DataFrame({c: self.resid[c] for c in self._columns}, index=self.index)

    def next_day(self) -> pd.DataFrame:
        """One-day-ahead mean and volatility per asset (rows ``mu``, ``sigma``)."""
        return pd.DataFrame({"mu": self.mu, "sigma": self.sigma_next}).T[self._columns]

    def _check(self, columns: pd.Index) -> None:
        if not self._columns:
            raise RuntimeError("Marginal is not fitted.")
        if list(columns) != self._columns:
            raise ValueError(f"Columns {list(columns)} do not match fitted {self._columns}.")
