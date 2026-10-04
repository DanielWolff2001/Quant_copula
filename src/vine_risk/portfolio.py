"""Portfolio risk (VaR, Expected Shortfall) from fitted vine copulas.

For a window ending at ``t`` the pipeline is: rebuild the fitted vine from its stored
:class:`~vine_risk.copula.VineFitResult`, draw uniform scenarios from it, map them back
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
from typing import Sequence

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

from vine_risk.copula import VineFitResult
from vine_risk.marginals import EmpiricalMarginal

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


def window_risk(
    result: VineFitResult,
    window_returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    alpha: float = 0.99,
    n_sims: int = 2 ** 14,
    seed: int = 0,
) -> dict[str, float]:
    """VaR and ES for one window under the vine and the benchmark models.

    ``window_returns`` must be the data the vine was fitted on. The same Sobol points
    are used for every model (and, with the same ``seed``, for every window), so
    differences between models or dates are not simulation noise.

    Returns ``var_*``/``es_*`` for ``vine``, ``gauss``, ``indep`` and ``hist``. Losses
    are in return units (0.02 = 2% of portfolio value).
    """
    if list(window_returns.columns) != list(result.assets):
        raise ValueError("window_returns columns do not match the fit.")
    w = equal_weights(result.n_assets) if weights is None else np.asarray(weights, dtype=float)
    marg = EmpiricalMarginal().fit(window_returns)
    u_win = marg.transform(window_returns).to_numpy()
    sobol = pv.utils.sobol(n_sims, result.n_assets, [seed])
    scen = {
        "vine": rebuild_vinecop(result).inverse_rosenblatt(sobol),
        "gauss": _gaussian_copula_uniforms(u_win, sobol),
        "indep": sobol,
    }
    out: dict[str, float] = {}
    cols = window_returns.columns
    for name, u in scen.items():
        r = marg.inverse_transform(pd.DataFrame(np.clip(u, 1e-9, 1 - 1e-9), columns=cols)).to_numpy()
        out[f"var_{name}"], out[f"es_{name}"] = var_es(portfolio_loss(w, r), alpha)
    out["var_hist"], out["es_hist"] = var_es(portfolio_loss(w, window_returns.to_numpy()), alpha)
    return out


def _risk_task(args: tuple) -> dict[str, float]:  # top-level so it can be pickled
    return window_risk(*args)


def rolling_risk(
    results: Sequence[VineFitResult],
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    alpha: float = 0.99,
    n_sims: int = 2 ** 14,
    seed: int = 0,
    step: int = 1,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """VaR/ES at every ``step``-th fit (``timestamp`` -> risk measures).

    Each fit's window is taken from ``returns`` (the ``n_obs`` observations ending at the
    fit's timestamp), so a row only uses data available at its timestamp. Also adds
    ``es_dependence_ratio = es_vine / es_indep`` (how much dependence inflates tail
    risk, with marginals held fixed) and ``es_non_gaussian = es_vine - es_gauss``.
    """
    if step < 1:
        raise ValueError("step must be >= 1.")
    picked = [r for r in results if r.status == "ok"][::step]
    tasks, stamps = [], []
    for r in picked:
        end = returns.index.get_loc(pd.Timestamp(r.timestamp))
        win = returns.iloc[end - r.n_obs + 1: end + 1]
        tasks.append((r, win, weights, alpha, n_sims, seed))
        stamps.append(pd.Timestamp(r.timestamp))
    if n_jobs > 1 and tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_risk_task, tasks, chunksize=max(1, len(tasks) // (n_jobs * 8))))
    else:
        rows = [_risk_task(t) for t in tasks]
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(stamps, name="timestamp"))
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
