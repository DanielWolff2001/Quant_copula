"""Portfolio risk (VaR, Expected Shortfall) from fitted vine copulas.

For a window ending at ``t`` the pipeline is: rebuild the fitted vine from its stored
`vine_risk.copula.VineFitResult`, draw uniform scenarios from it, map them back
to returns through the window's empirical marginals, form portfolio losses
``L = -w'r`` and read off ``VaR_alpha`` and ``ES_alpha = E[L | L >= VaR_alpha]``.

To isolate the effect of dependence, the same scenarios (common Sobol points) and the
same marginals are also pushed through three benchmark copulas: independence, a
Gaussian copula with the window's normal-scores correlation (no tail dependence), and
the window's own historical returns.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

from vine_risk.copula import VineFitResult
from vine_risk.marginals import EmpiricalMarginal, Marginal

logger = logging.getLogger(__name__)

_FAMILY = {"independence": pv.BicopFamily.indep}


def _family(name: str) -> pv.BicopFamily:
    if name in _FAMILY:
        return _FAMILY[name]
    try:
        return getattr(pv.BicopFamily, name)
    except AttributeError as e:
        raise ValueError(f"Unknown copula family {name!r}") from e


def rebuild_vinecop(result: VineFitResult) -> pv.Vinecop:
    """Reconstruct the fitted ``pyvinecopulib`` vine from a stored result.

    Works from the pair-copulas' conditioned sets, the variable order, families,
    rotations and parameters, so no refit is needed.
    """
    if result.status != "ok" or not result.pair_copulas:
        raise ValueError("Cannot rebuild a vine from a failed fit.")
    d = result.n_assets
    col = {a: i + 1 for i, a in enumerate(result.assets)}  # 1-based variable labels
    order = result.order
    n_trees = max(p.tree for p in result.pair_copulas)
    mat = np.zeros((d, d), dtype=np.uint64)
    for e in range(d):
        mat[d - 1 - e, e] = order[e]  # diagonal: the variable order
    trees: list[list[pv.Bicop]] = [[] for _ in range(n_trees)]
    for p in sorted(result.pair_copulas, key=lambda p: (p.tree, p.edge)):
        e = p.edge - 1
        a, b = (col[x] for x in p.conditioned)
        partner = b if a == order[e] else a
        mat[p.tree - 1, e] = partner
        params = np.asarray(p.parameters, dtype=float).reshape(-1, 1) if p.parameters else np.empty((0, 0))
        trees[p.tree - 1].append(pv.Bicop(_family(p.family), p.rotation, params))
    return pv.Vinecop.from_structure(matrix=mat, pair_copulas=trees)


def portfolio_loss(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Loss ``L = -w'r`` for each row of ``returns`` (positive = loss)."""
    w = np.asarray(weights, dtype=float)
    r = np.asarray(returns, dtype=float)
    if r.shape[-1] != w.shape[0]:
        raise ValueError(f"{w.shape[0]} weights for {r.shape[-1]} assets.")
    return -(r @ w)


def var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Empirical ``VaR_alpha`` (the alpha-quantile of the losses) and ``ES_alpha``
    (mean of the losses at or above it)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    x = np.asarray(losses, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("losses must be a non-empty finite array.")
    var = float(np.quantile(x, alpha))
    return var, float(x[x >= var].mean())


def equal_weights(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def _gaussian_copula_uniforms(u_window: np.ndarray, sobol: np.ndarray) -> np.ndarray:
    """Gaussian copula with the window's normal-scores correlation, from Sobol points."""
    z = stats.norm.ppf(u_window)
    corr = np.corrcoef(z, rowvar=False)
    chol = np.linalg.cholesky(corr + 1e-10 * np.eye(corr.shape[0]))
    return stats.norm.cdf(stats.norm.ppf(sobol) @ chol.T)


def scenario_returns(
    result: VineFitResult,
    window_returns: pd.DataFrame,
    *,
    marginal: Marginal | None = None,
    n_sims: int = 2 ** 14,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Asset-return scenarios (``n_sims`` x assets) under the vine, a Gaussian copula and independence.

    The scenarios of all three models are built from the *same* Sobol points and mapped to returns
    through the same ``marginal``, so differences between models come from dependence only.
    ``marginal`` is a fitted `Marginal` whose ``transform`` accepts ``window_returns`` (default: rank
    marginals fitted on the window). With a GARCH marginal the scenarios are forecasts for the next day
    given today's volatility; with rank marginals they follow the window's own distribution.
    """
    if list(window_returns.columns) != list(result.assets):
        raise ValueError("window_returns columns do not match the fit.")
    marg = marginal if marginal is not None else EmpiricalMarginal().fit(window_returns)
    u_win = marg.transform(window_returns).to_numpy()
    sobol = pv.utils.sobol(n_sims, result.n_assets, [seed])
    scen_u = {
        "vine": rebuild_vinecop(result).inverse_rosenblatt(sobol),
        "gauss": _gaussian_copula_uniforms(u_win, sobol),
        "indep": sobol,
    }
    cols = window_returns.columns
    return {name: marg.inverse_transform(pd.DataFrame(np.clip(u, 1e-9, 1 - 1e-9), columns=cols)).to_numpy()
            for name, u in scen_u.items()}


def tail_grid(losses: np.ndarray, alpha: float, k: int = 20) -> np.ndarray:
    """``k`` quantiles that summarise the loss tail beyond the ``alpha``-quantile.

    The quantiles are taken at the midpoints of ``k`` equal-probability cells of the tail
    (``alpha + (1 - alpha) * (i + 0.5) / k``). Their mean is close to the Expected Shortfall, and drawing
    one of them at random mimics a loss given that VaR was exceeded (used by the ES backtests).
    """
    probs = alpha + (1 - alpha) * (np.arange(k) + 0.5) / k
    return np.quantile(np.asarray(losses, dtype=float), probs)


def loss_summary(losses: np.ndarray, alphas: Sequence[float], k: int = 20) -> dict[float, dict]:
    """VaR, ES and the tail grid of a loss sample for several confidence levels."""
    out = {}
    for a in alphas:
        var, es = var_es(losses, a)
        out[a] = {"var": var, "es": es, "tail": tail_grid(losses, a, k)}
    return out


def window_risk(
    result: VineFitResult,
    window_returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    alpha: float = 0.99,
    n_sims: int = 2 ** 14,
    seed: int = 0,
    *,
    marginal: Marginal | None = None,
) -> dict[str, float]:
    """VaR and ES for one window under the vine and the benchmark models.

    ``window_returns`` must be the data the vine was fitted on. The same Sobol points
    are used for every model (and, with the same ``seed``, for every window), so
    differences between models or dates are not simulation noise. ``marginal`` is a fitted
    marginal model (default: ranks on the window; see `scenario_returns`).

    Returns ``var_*``/``es_*`` for ``vine``, ``gauss``, ``indep`` and ``hist``. Losses
    are in return units (0.02 = 2% of portfolio value).
    """
    w = equal_weights(result.n_assets) if weights is None else np.asarray(weights, dtype=float)
    out: dict[str, float] = {}
    for name, r in scenario_returns(result, window_returns, marginal=marginal, n_sims=n_sims, seed=seed).items():
        out[f"var_{name}"], out[f"es_{name}"] = var_es(portfolio_loss(w, r), alpha)
    out["var_hist"], out["es_hist"] = var_es(portfolio_loss(w, window_returns.to_numpy()), alpha)
    return out


def _risk_task(args: tuple) -> dict[str, float]:  # top-level so it can be pickled
    result, window, weights, alpha, n_sims, seed, marginal = args
    return window_risk(result, window, weights, alpha, n_sims, seed, marginal=marginal)


def rolling_risk(
    results: Sequence[VineFitResult],
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    alpha: float = 0.99,
    n_sims: int = 2 ** 14,
    seed: int = 0,
    step: int = 1,
    n_jobs: int = 1,
    after: pd.Timestamp | None = None,
    marginal_factory: Callable[[], Marginal] | None = None,
    lookback: int | None = None,
) -> pd.DataFrame:
    """VaR/ES at every ``step``-th fit (``timestamp`` -> risk measures).

    The fits are picked from the first one on (the 0th, ``step``-th, ...); with ``after`` only
    those later than that date are computed, to extend an earlier table on the same grid.
    ``marginal_factory`` and ``lookback`` give the marginal model of the run (default: ranks on the
    window); a GARCH marginal is fitted on the ``lookback`` observations ending at each fit.

    Each fit's window is taken from ``returns`` (the ``n_obs`` observations ending at the
    fit's timestamp), so a row only uses data available at its timestamp. Also adds
    ``es_dependence_ratio = es_vine / es_indep`` (how much dependence inflates tail
    risk, with marginals held fixed) and ``es_non_gaussian = es_vine - es_gauss``.
    """
    if step < 1:
        raise ValueError("step must be >= 1.")
    picked = [r for r in results if r.status == "ok"][::step]
    if after is not None:
        picked = [r for r in picked if pd.Timestamp(r.timestamp) > after]
    tasks, stamps = [], []
    for r in picked:
        end = returns.index.get_loc(pd.Timestamp(r.timestamp))
        hist = returns.iloc[end - (lookback or r.n_obs) + 1: end + 1]
        tasks.append((r, hist.iloc[-r.n_obs:], weights, alpha, n_sims, seed,
                      None if marginal_factory is None else marginal_factory().fit(hist)))
        stamps.append(pd.Timestamp(r.timestamp))
    if n_jobs > 1 and tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_risk_task, tasks, chunksize=max(1, len(tasks) // (n_jobs * 8))))
    else:
        rows = [_risk_task(t) for t in tasks]
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(stamps, name="timestamp"))
    if out.empty:
        return out.reindex(columns=[*(f"{m}_{k}" for m in ("var", "es") for k in ("vine", "gauss", "indep", "hist")),
                                    "es_dependence_ratio", "es_non_gaussian"])
    out["es_dependence_ratio"] = out["es_vine"] / out["es_indep"]
    out["es_non_gaussian"] = out["es_vine"] - out["es_gauss"]
    return out


def backtest_var(
    risk: pd.DataFrame, returns: pd.DataFrame, weights: np.ndarray | None = None,
    alpha: float = 0.99, models: Sequence[str] = ("vine", "gauss", "indep", "hist"),
) -> pd.DataFrame:
    """Out-of-sample VaR check: does next day's loss exceed the VaR computed at ``t``?

    Reports the exceedance rate per model (should be about ``1 - alpha``) and the p-value
    of Kupiec's unconditional-coverage likelihood-ratio test. Dates where the next
    observation is unavailable are dropped.
    """
    w = equal_weights(returns.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    loss = pd.Series(portfolio_loss(w, returns.to_numpy()), index=returns.index)
    nxt = loss.shift(-1).reindex(risk.index).dropna()
    rows = {}
    for m in models:
        hit = nxt > risk.loc[nxt.index, f"var_{m}"]
        n, x = len(hit), int(hit.sum())
        rows[m] = {"n": n, "exceedances": x, "rate": x / n, "expected_rate": 1 - alpha,
                   "kupiec_p": _kupiec_p(n, x, 1 - alpha)}
    return pd.DataFrame(rows).T


def _kupiec_p(n: int, x: int, p: float) -> float:
    """p-value of Kupiec's proportion-of-failures test (chi-square, 1 dof)."""
    def ll(q: float) -> float:
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)
    lr = -2.0 * (ll(p) - ll(x / n))
    return float(stats.chi2.sf(max(lr, 0.0), 1))
