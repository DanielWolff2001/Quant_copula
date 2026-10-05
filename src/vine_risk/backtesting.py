"""Backtests for VaR and Expected Shortfall forecasts, and tests for comparing models.

Conventions: losses are positive numbers; a forecast is made at date ``t`` for the loss on the next day,
and ``var``/``es`` are the Value at Risk and Expected Shortfall at confidence level ``alpha`` (for
example 0.99, so a loss exceeds the VaR with probability ``q = 1 - alpha`` = 1 %).

* **Coverage** - :func:`kupiec_pof` (is the exceedance rate right?), :func:`christoffersen_independence`
  (do exceedances cluster?) and :func:`christoffersen_cc` (both at once).
* **Expected Shortfall** - :func:`acerbi_szekely` (the Z1 and Z2 tests: is the loss on exceedance days as
  large as the ES forecast says?). Their p-values are simulated under the null hypothesis that the
  forecast distribution is right, using each date's tail summary.
* **Comparison** - :func:`fz0_loss`, a scoring rule that is minimised in expectation by the true
  (VaR, ES) pair and so ranks models by both together, and :func:`diebold_mariano`, which tests whether
  the average score of two models differs significantly (with an autocorrelation-robust variance).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def hits(losses: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Boolean array: the realised loss exceeded the forecast VaR."""
    return np.asarray(losses, dtype=float) > np.asarray(var, dtype=float)


def _bernoulli_ll(n: int, x: int, p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return (n - x) * np.log(1 - p) + x * np.log(p)


def kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """Kupiec's proportion-of-failures test: ``x`` exceedances in ``n`` days where ``p`` was expected.

    Returns ``(LR statistic, p-value)`` (chi-square, 1 degree of freedom).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    lr = max(-2.0 * (_bernoulli_ll(n, x, p) - _bernoulli_ll(n, x, x / n)), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(h: np.ndarray) -> tuple[float, float]:
    """Christoffersen's test that exceedances are independent over time (first-order Markov chain).

    Returns ``(LR statistic, p-value)`` (chi-square, 1 degree of freedom); ``(0, 1)`` if there is nothing
    to test (no exceedances, or they never follow each other / never occur after a quiet day).
    """
    h = np.asarray(h, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00, n01 = int(np.sum((prev == 0) & (cur == 0))), int(np.sum((prev == 0) & (cur == 1)))
    n10, n11 = int(np.sum((prev == 1) & (cur == 0))), int(np.sum((prev == 1) & (cur == 1)))
    if n01 + n11 == 0 or (n00 + n01) == 0:
        return 0.0, 1.0
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll1 = _bernoulli_ll(n00 + n01, n01, pi01) + _bernoulli_ll(n10 + n11, n11, pi11)
    ll0 = _bernoulli_ll(n00 + n01 + n10 + n11, n01 + n11, pi)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_cc(h: np.ndarray, p: float) -> tuple[float, float]:
    """Conditional coverage: correct exceedance rate *and* independence (chi-square, 2 degrees of freedom)."""
    h = np.asarray(h, dtype=int)
    lr = kupiec_pof(len(h), int(h.sum()), p)[0] + christoffersen_independence(h)[0]
    return float(lr), float(stats.chi2.sf(lr, 2))


def acerbi_szekely(losses, var, es, tails, alpha: float, n_sim: int = 2000, seed: int = 0) -> dict[str, float]:
    """The Acerbi-Székely Expected Shortfall backtests Z1 and Z2.

    * ``Z1 = mean over exceedance days of (loss / ES) - 1``: is the loss, given a VaR exceedance, as large
      as forecast? (``NaN`` without exceedances.)
    * ``Z2 = sum_t (loss_t * 1{exceeded} / ES_t) / (n q) - 1``: also checks the number of exceedances.

    Both are zero if the forecasts are right and **positive if risk was underestimated**. The reported
    ``p1``/``p2`` are one-sided simulated p-values ``P(Z_sim >= Z)`` under the hypothesis that each day's
    forecast distribution is correct, where an exceedance happens with probability ``1 - alpha`` and its
    size is drawn from ``tails`` (one row of tail quantiles per day, see `vine_risk.portfolio.tail_grid`).
    Small p-values reject the model.
    """
    losses, var, es = (np.asarray(a, dtype=float) for a in (losses, var, es))
    tails = np.asarray(tails, dtype=float)
    n, q = len(losses), 1 - alpha
    if tails.shape[0] != n:
        raise ValueError("tails needs one row per day.")
    exc = losses > var
    ratio = np.where(exc, losses / es, 0.0)
    z1 = float(ratio[exc].mean() - 1) if exc.any() else float("nan")
    z2 = float(ratio.sum() / (n * q) - 1)
    rng = np.random.default_rng(seed)
    k = tails.shape[1]
    sim_exc = rng.random((n_sim, n)) < q
    sim_loss = tails[np.arange(n), rng.integers(0, k, (n_sim, n))]
    sim_ratio = np.where(sim_exc, sim_loss / es, 0.0)
    z2_sim = sim_ratio.sum(axis=1) / (n * q) - 1
    cnt = sim_exc.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z1_sim = sim_ratio.sum(axis=1) / cnt - 1
    p2 = float((1 + np.sum(z2_sim >= z2)) / (n_sim + 1))
    p1 = float((1 + np.nansum(z1_sim[cnt > 0] >= z1)) / (n_sim + 1)) if exc.any() else float("nan")
    return {"n_exceed": int(exc.sum()), "z1": z1, "z2": z2, "p1": p1, "p2": p2}


def fz0_loss(losses, var, es, alpha: float) -> np.ndarray:
    """Fissler-Ziegel / Patton-Ziegel-Chen "FZ0" score of (VaR, ES) forecasts, per day (lower is better).

    ``S = 1{loss >= VaR} (loss - VaR) / ((1 - alpha) ES) + VaR / ES + log(ES) - 1`` for positive losses.
    It is strictly consistent for the pair (VaR, ES): its expected value is smallest at the true values.
    Requires ``ES > 0``.
    """
    losses, var, es = (np.asarray(a, dtype=float) for a in (losses, var, es))
    if (es <= 0).any():
        raise ValueError("FZ0 needs positive Expected Shortfall forecasts.")
    return np.where(losses >= var, (losses - var) / ((1 - alpha) * es), 0.0) + var / es + np.log(es) - 1.0


def pinball_loss(losses, var, alpha: float) -> np.ndarray:
    """Quantile (pinball) loss of VaR forecasts at level ``alpha`` (lower is better)."""
    losses, var = np.asarray(losses, dtype=float), np.asarray(var, dtype=float)
    return np.where(losses >= var, alpha * (losses - var), (1 - alpha) * (var - losses))


def diebold_mariano(d: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    """Diebold-Mariano test that a series of loss differences ``d = score_A - score_B`` has mean zero.

    The variance uses a Bartlett kernel with ``lag`` autocovariances (default ``n ** (1/3)``) because daily
    forecast scores are autocorrelated. Returns ``(t statistic, two-sided p-value)``; a **negative**
    statistic means model A scores lower (better) than B.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 10:
        raise ValueError("Need at least 10 observations.")
    lag = int(n ** (1 / 3)) if lag is None else lag
    dc = d - d.mean()
    var = float(dc @ dc / n)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * float(dc[k:] @ dc[:-k] / n)
    if var <= 0:
        return 0.0, 1.0
    stat = float(d.mean() / np.sqrt(var / n))
    return stat, float(2 * stats.norm.sf(abs(stat)))


def evaluate_forecasts(
    forecasts: pd.DataFrame, realized: pd.DataFrame, reference: str | None = None, dm_lag: int | None = None,
    n_sim: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """Backtest every (portfolio, level, model) in a table of forecasts.

    ``forecasts`` has one row per forecast date, model, portfolio and level, with columns ``date``,
    ``model``, ``portfolio``, ``alpha``, ``var``, ``es`` and ``tail`` (an array). ``realized`` is a
    frame indexed by forecast date with one column per portfolio holding the loss on the *next* day.

    Returns one row per (portfolio, alpha, model): number of days, exceedances and rate, the p-values of
    Kupiec (``kupiec_p``), Christoffersen's independence (``ind_p``) and conditional coverage (``cc_p``)
    tests, the Acerbi-Székely ``z1``/``z2`` with p-values ``z1_p``/``z2_p``, the mean ``fz0`` and ``pinball``
    scores, and, if ``reference`` names a model, the Diebold-Mariano statistic and p-value of each model's
    FZ0 score against it (``dm_stat`` < 0: better than the reference).
    """
    rows = []
    for (pname, alpha), g in forecasts.groupby(["portfolio", "alpha"], sort=True):
        scores = {}
        for model, m in g.groupby("model", sort=False):
            m = m.sort_values("date")
            loss = realized.loc[m["date"], pname].to_numpy()
            var, es = m["var"].to_numpy(), m["es"].to_numpy()
            h = hits(loss, var)
            n, x = len(h), int(h.sum())
            kp, ip, cp = kupiec_pof(n, x, 1 - alpha)[1], christoffersen_independence(h)[1], christoffersen_cc(h, 1 - alpha)[1]
            a_s = acerbi_szekely(loss, var, es, np.vstack(m["tail"].to_numpy()), alpha, n_sim, seed)
            fz = fz0_loss(loss, var, es, alpha)
            scores[model] = pd.Series(fz, index=m["date"].to_numpy())
            rows.append({"portfolio": pname, "alpha": alpha, "model": model, "n": n, "exceedances": x, "rate": x / n,
                         "kupiec_p": kp, "ind_p": ip, "cc_p": cp, "z1": a_s["z1"], "z1_p": a_s["p1"],
                         "z2": a_s["z2"], "z2_p": a_s["p2"], "fz0": float(fz.mean()),
                         "pinball": float(pinball_loss(loss, var, alpha).mean())})
        if reference is not None and reference in scores:
            for r in rows[-len(scores):]:
                s = scores[r["model"]]
                common = s.index.intersection(scores[reference].index)
                r["dm_stat"], r["dm_p"] = (0.0, 1.0) if r["model"] == reference else \
                    diebold_mariano(s.loc[common].to_numpy() - scores[reference].loc[common].to_numpy(), dm_lag)
    return pd.DataFrame(rows).set_index(["portfolio", "alpha", "model"])
