"""Synthetic return data with known dependence regimes.

Used to validate the change detectors: the true copula, and the time at which it
changes, are known. Each regime is an equicorrelated elliptical copula, either
Gaussian or Student-t, so ordinary dependence (Kendall's tau) and tail dependence can
be changed independently of each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Regime:
    """``n`` observations from an equicorrelated copula.

    Attributes:
        n: number of observations.
        rho: copula correlation parameter. Kendall's tau is ``2/pi * arcsin(rho)``.
        df: degrees of freedom of a Student-t copula; ``None`` for a Gaussian copula.
    """

    n: int
    rho: float
    df: float | None = None

    @property
    def tau(self) -> float:
        return float(2 / np.pi * np.arcsin(self.rho))

    @property
    def tail_dependence(self) -> float:
        """Asymptotic lower (= upper) tail dependence coefficient."""
        if self.df is None:
            return 0.0
        nu, r = self.df, self.rho
        return float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1))


def simulate_regimes(
    regimes: list[Regime], n_assets: int = 4, seed: int = 0, start: str = "2010-01-04",
    vol: float = 0.01, vol_phi: float | None = None, vol_eta: float = 0.15,
) -> pd.DataFrame:
    """Concatenate the regimes into one T x d return matrix (N(0, vol^2) margins).

    With ``vol_phi`` set, all assets are multiplied by a common stochastic volatility
    ``exp(h_t)`` with ``h_t = vol_phi * h_{t-1} + vol_eta * eps_t``, which mimics
    volatility clustering. The copula regimes (and hence the dependence structure
    being tested) are unchanged by it; it is used as a harder "no change" null.
    """
    if n_assets < 2:
        raise ValueError("n_assets must be >= 2.")
    rng = np.random.default_rng(seed)
    parts = []
    for reg in regimes:
        if not -1 < reg.rho < 1:
            raise ValueError("rho must be in (-1, 1).")
        corr = np.full((n_assets, n_assets), reg.rho) + (1 - reg.rho) * np.eye(n_assets)
        z = rng.multivariate_normal(np.zeros(n_assets), corr, size=reg.n)
        if reg.df is None:
            u = stats.norm.cdf(z)
        else:
            w = rng.chisquare(reg.df, size=(reg.n, 1)) / reg.df
            u = stats.t.cdf(z / np.sqrt(w), reg.df)
        parts.append(stats.norm.ppf(u) * vol)
    x = np.vstack(parts)
    if vol_phi is not None:
        h = np.zeros(len(x))
        eps = rng.standard_normal(len(x))
        for t in range(1, len(x)):
            h[t] = vol_phi * h[t - 1] + vol_eta * eps[t]
        x = x * np.exp(h)[:, None]
    idx = pd.bdate_range(start, periods=len(x))
    return pd.DataFrame(x, index=idx, columns=[f"X{i + 1}" for i in range(n_assets)])


def simulate_garch(
    regimes: list[Regime], n_assets: int = 4, seed: int = 0, start: str = "2010-01-04", omega: float = 0.02,
    alpha: float = 0.08, beta: float = 0.90, nu: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns with GARCH(1,1) volatility and copula-linked Student-t innovations; known truth.

    Each asset follows ``r_t = sigma_t * e_t``, ``sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2``
    (``omega`` in percent-squared; the default gives a 1 % daily unconditional volatility), where the
    innovations ``e_t`` have unit variance, a Student-t distribution with ``nu`` degrees of freedom, and
    the dependence structure given by ``regimes`` (the same equicorrelated copulas as
    :func:`simulate_regimes`).

    Returns ``(returns, sigma)``: the return matrix (decimal) and the true conditional volatility, whose
    row ``t`` is the volatility *of* ``r_t`` (known one step earlier).
    """
    if not (alpha >= 0 and beta >= 0 and alpha + beta < 1 and omega > 0 and nu > 2):
        raise ValueError("Need omega > 0, alpha, beta >= 0, alpha + beta < 1 and nu > 2.")
    u = stats.norm.cdf(simulate_regimes(regimes, n_assets, seed, start, vol=1.0).to_numpy())
    e = stats.t.ppf(u, nu) / np.sqrt(nu / (nu - 2))
    n = len(e)
    var = np.empty((n, n_assets))
    r = np.empty((n, n_assets))
    var[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t:
            var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = np.sqrt(var[t]) * e[t]
    idx = pd.bdate_range(start, periods=n)
    cols = [f"X{i + 1}" for i in range(n_assets)]
    return pd.DataFrame(r / 100, index=idx, columns=cols), pd.DataFrame(np.sqrt(var) / 100, index=idx, columns=cols)
