"""Standard risk models to compare the vine copula against.

Every model turns the return history up to a date into one-day-ahead **VaR and Expected Shortfall
forecasts** for a set of portfolios and confidence levels, using only that history:

* :class:`HistoricalSimulation` - empirical quantiles of the last ``window`` portfolio returns;
* :class:`EwmaNormal` - RiskMetrics: an exponentially weighted covariance, normal returns;
* :class:`EwmaStudentT` - the same covariance with Student-t returns (fat tails), tail thickness estimated;
* :class:`FilteredHistoricalSimulation` - GARCH(1,1) volatility per asset, with the standardised residuals
  resampled *jointly* (rows), so cross-asset dependence and joint tail events are kept.

The copula models (vine, Gaussian, independence, with rank or GARCH marginals) live in
:mod:`vine_risk.benchmark`, because they need the fitted vines. All forecasts are returned as
:class:`RiskForecast` records, with ``tail`` a few quantiles of the loss tail (see
:func:`vine_risk.portfolio.tail_grid`) for the Expected Shortfall backtests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from vine_risk.garch import GarchMarginal
from vine_risk.portfolio import loss_summary, portfolio_loss, tail_grid, var_es

TAIL_POINTS = 20


@dataclass(frozen=True)
class RiskForecast:
    """A one-day-ahead VaR / Expected Shortfall forecast (losses are positive, in return units)."""

    model: str
    portfolio: str
    alpha: float
    var: float
    es: float
    tail: np.ndarray


def _check_inputs(history: pd.DataFrame, portfolios: Mapping[str, np.ndarray], alphas: Sequence[float]) -> None:
    if history.empty or not np.isfinite(history.to_numpy()).all():
        raise ValueError("history must be a non-empty return frame without NaN or infinite values.")
    for name, w in portfolios.items():
        if np.asarray(w).shape != (history.shape[1],):
            raise ValueError(f"portfolio {name!r} has {np.asarray(w).size} weights for {history.shape[1]} assets.")
    if not alphas or any(not 0 < a < 1 for a in alphas):
        raise ValueError("alphas must be a non-empty sequence of levels in (0, 1).")


class RiskModel(ABC):
    """Interface of a one-day-ahead risk model."""

    name: str

    @abstractmethod
    def forecast(self, history: pd.DataFrame, portfolios: Mapping[str, np.ndarray], alphas: Sequence[float],
                 ) -> list[RiskForecast]:
        """Forecasts for the day after the last row of ``history``, for every portfolio and level."""


def _from_losses(model: str, portfolios, alphas, losses_by_portfolio: Mapping[str, np.ndarray]) -> list[RiskForecast]:
    out = []
    for pname in portfolios:
        for a, s in loss_summary(losses_by_portfolio[pname], alphas, TAIL_POINTS).items():
            out.append(RiskForecast(model, pname, a, s["var"], s["es"], s["tail"]))
    return out


class HistoricalSimulation(RiskModel):
    """Empirical VaR/ES of the portfolio's last ``window`` returns (no model, slow to react)."""

    def __init__(self, window: int = 250) -> None:
        self.name, self.window = "hist", window

    def forecast(self, history, portfolios, alphas):
        _check_inputs(history, portfolios, alphas)
        win = history.iloc[-self.window:].to_numpy()
        return _from_losses(self.name, portfolios, alphas, {p: portfolio_loss(np.asarray(w), win) for p, w in portfolios.items()})


# ---- EWMA (RiskMetrics) -------------------------------------------------------------------
def ewma_covariances(returns: np.ndarray, lam: float = 0.94, init: int = 60) -> np.ndarray:
    """Exponentially weighted covariance forecasts.

    Returns an array ``S`` of shape ``(T + 1, d, d)``: ``S[t]`` is the covariance forecast *for* row ``t``
    (made from rows before ``t``) and ``S[T]`` the forecast for the day after the data. The recursion is
    ``S[t+1] = lam * S[t] + (1 - lam) * r_t r_t'`` (zero mean, as in RiskMetrics) started from the sample
    covariance of the first ``init`` rows.
    """
    T, d = returns.shape
    S = np.empty((T + 1, d, d))
    first = returns[: min(init, T)]
    S[0] = first.T @ first / len(first)
    for t in range(T):
        S[t + 1] = lam * S[t] + (1 - lam) * np.outer(returns[t], returns[t])
    return S


class EwmaNormal(RiskModel):
    """RiskMetrics: EWMA covariance (``lam`` = 0.94), normally distributed returns.

    ``VaR = sigma * z_alpha`` and ``ES = sigma * phi(z_alpha) / (1 - alpha)`` with ``sigma`` the forecast
    portfolio volatility. Reacts quickly to volatility, but normal tails are thin.
    """

    def __init__(self, lam: float = 0.94, lookback: int = 500) -> None:
        if not 0 < lam < 1:
            raise ValueError("lam must be in (0, 1).")
        self.name, self.lam, self.lookback = "ewma_n", lam, lookback

    def forecast(self, history, portfolios, alphas):
        _check_inputs(history, portfolios, alphas)
        S = ewma_covariances(history.iloc[-self.lookback:].to_numpy(), self.lam)[-1]
        out = []
        for pname, w in portfolios.items():
            sigma = float(np.sqrt(np.asarray(w) @ S @ np.asarray(w)))
            for a in alphas:
                z = stats.norm.ppf(a)
                probs = a + (1 - a) * (np.arange(TAIL_POINTS) + 0.5) / TAIL_POINTS
                out.append(RiskForecast(self.name, pname, a, sigma * z, sigma * stats.norm.pdf(z) / (1 - a),
                                        sigma * stats.norm.ppf(probs)))
        return out


def _t_unit_loglik(nu: float, z: np.ndarray) -> float:
    """Log-likelihood of unit-variance Student-t data ``z`` with ``nu`` degrees of freedom."""
    c = np.sqrt(nu / (nu - 2))
    return float(np.sum(stats.t.logpdf(z * c, nu) + np.log(c)))


def fit_t_dof(z: np.ndarray) -> float:
    """Degrees of freedom (2.1 to 60) of the unit-variance Student-t that best fits standardised data."""
    res = optimize.minimize_scalar(lambda nu: -_t_unit_loglik(nu, z), bounds=(2.1, 60.0), method="bounded")
    return float(res.x)


def t_var_es(sigma: float, nu: float, alpha: float) -> tuple[float, float]:
    """VaR and ES of a Student-t loss with standard deviation ``sigma`` and ``nu`` degrees of freedom."""
    scale = sigma * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    return float(scale * q), float(scale * stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1) * (1 - alpha)))


class EwmaStudentT(RiskModel):
    """EWMA covariance with Student-t returns: fat tails, tail thickness estimated per portfolio.

    The degrees of freedom are estimated by maximum likelihood on the portfolio's EWMA-standardised
    returns over the lookback period.
    """

    def __init__(self, lam: float = 0.94, lookback: int = 500) -> None:
        if not 0 < lam < 1:
            raise ValueError("lam must be in (0, 1).")
        self.name, self.lam, self.lookback = "ewma_t", lam, lookback

    def forecast(self, history, portfolios, alphas):
        _check_inputs(history, portfolios, alphas)
        r = history.iloc[-self.lookback:].to_numpy()
        S = ewma_covariances(r, self.lam)
        out = []
        for pname, w in portfolios.items():
            w = np.asarray(w)
            sig = np.sqrt(np.einsum("i,tij,j->t", w, S, w))  # sig[t]: forecast sd for row t; sig[-1] for tomorrow
            nu = fit_t_dof((r @ w) / sig[:-1])
            for a in alphas:
                var, es = t_var_es(float(sig[-1]), nu, a)
                probs = a + (1 - a) * (np.arange(TAIL_POINTS) + 0.5) / TAIL_POINTS
                tail = float(sig[-1]) * np.sqrt((nu - 2) / nu) * stats.t.ppf(probs, nu)
                out.append(RiskForecast(self.name, pname, a, var, es, tail))
        return out


class FilteredHistoricalSimulation(RiskModel):
    """GARCH(1,1) volatility per asset, standardised residuals used jointly.

    Fits a GARCH(1,1) to every asset over ``lookback`` days and rescales each *row* of the
    standardised-residual matrix by tomorrow's forecast volatility; every row is one equally likely
    scenario, so the cross-asset dependence, including joint tail events, is kept as observed.

    By default each row is used once (``n_boot=None``), which is the standard method. Resampling rows with
    replacement (``n_boot`` draws) is possible but not advisable: it turns the tail into a staircase of
    ``lookback`` atoms, and a high quantile can then jump to the next atom (in one test with 500 residuals
    the 99 % VaR came out 30 % too high).
    """

    def __init__(self, lookback: int = 500, n_boot: int | None = None, seed: int = 0) -> None:
        self.name, self.lookback, self.n_boot, self.seed = "fhs", lookback, n_boot, seed

    def forecast(self, history, portfolios, alphas, garch: GarchMarginal | None = None):
        """``garch`` may be a `GarchMarginal` already fitted to the same history (to share the fit)."""
        _check_inputs(history, portfolios, alphas)
        hist = history.iloc[-self.lookback:]
        g = garch if garch is not None else GarchMarginal(innovations="empirical").fit(hist)
        z = g.standardised_residuals().to_numpy()
        nd = g.next_day()
        if self.n_boot is not None:
            z = z[np.random.default_rng(self.seed).integers(0, len(z), self.n_boot)]
        scen = nd.loc["mu"].to_numpy() + nd.loc["sigma"].to_numpy() * z
        return _from_losses(self.name, portfolios, alphas, {p: portfolio_loss(np.asarray(w), scen) for p, w in portfolios.items()})
