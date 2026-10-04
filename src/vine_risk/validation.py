"""Synthetic validation study: do the methods work when the truth is known?

Detection experiments (PDF section 20), each replicated over many seeds:

* ``A_const``  - constant dependence: any alert is a false alarm;
* ``A_vol``    - constant dependence, clustered volatility: a harder null;
* ``B_corr``   - correlation jumps from 0.3 to 0.7 at ``change_at``;
* ``C_tail_moderate`` / ``C_tail_strong`` - Gaussian -> Student-t copula at the *same*
  Kendall's tau (tail dependence changes, ordinary dependence does not).

For a scan date ``t`` (permutation test) or fit timestamp ``t`` (distance scores) two
windows of ``window`` days are compared: ``(t-2W, t-W]`` and ``(t-W, t]``. Dates fall into
zones relative to the change point ``c``:

* ``pre``: ``t < c`` - neither window contains post-change data;
* ``change``: ``c + W/2 <= t <= c + 3W/2`` - the windows straddle the change (the
  difference is largest at ``t = c + W``);
* ``post``: ``t >= c + 2W`` - both windows are entirely post-change;
* ``transition``: everything else.

Alerts in ``pre`` and ``post`` (and everywhere in the no-change experiments) are false
alarms; alerts in ``change`` are detections.

Also provided: :func:`estimation_accuracy`, how well a fitted vine recovers tail
dependence and portfolio VaR/ES when the true copula is known.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from vine_risk.change_detection import change_scan, default_lag, structural_change_scores
from vine_risk.copula import VineCopula
from vine_risk.marginals import EmpiricalMarginal
from vine_risk.portfolio import equal_weights, portfolio_loss, var_es, window_risk
from vine_risk.rolling import RollingVineModel
from vine_risk.synthetic import Regime, simulate_regimes

logger = logging.getLogger(__name__)

NULL_ZONES = ("null", "pre", "post")


@dataclass(frozen=True)
class Experiment:
    name: str
    regimes: tuple[Regime, ...]
    vol_phi: float | None = None
    description: str = ""

    @property
    def n_obs(self) -> int:
        return sum(r.n for r in self.regimes)

    @property
    def change_at(self) -> int | None:
        """Index of the first post-change observation (``None`` if there is no change)."""
        return self.regimes[0].n if len(self.regimes) > 1 else None


def standard_experiments(n_obs: int = 1700, change_at: int = 900) -> dict[str, Experiment]:
    """The experiments of the validation study."""
    def jump(a: Regime, b: Regime) -> tuple[Regime, Regime]:
        return replace_n(a, change_at), replace_n(b, n_obs - change_at)

    def replace_n(r: Regime, n: int) -> Regime:
        return Regime(n, r.rho, r.df)

    exps = [
        Experiment("A_const", (Regime(n_obs, 0.5),), description="constant Gaussian copula, rho=0.5"),
        Experiment("A_vol", (Regime(n_obs, 0.5),), vol_phi=0.98,
                   description="constant copula, clustered volatility"),
        Experiment("B_corr", jump(Regime(1, 0.3), Regime(1, 0.7)),
                   description="Gaussian copula, rho 0.3 -> 0.7"),
        Experiment("C_tail_moderate", jump(Regime(1, 0.5), Regime(1, 0.5, 3.0)),
                   description="Gaussian -> Student-t(3), same rho (tau)"),
        Experiment("C_tail_strong", jump(Regime(1, 0.5), Regime(1, 0.5, 1.0)),
                   description="Gaussian -> Student-t(1), same rho (tau)"),
    ]
    return {e.name: e for e in exps}


def zone_of(pos: int, change_at: int | None, window: int) -> str:
    """Zone of a date at integer position ``pos`` (see the module docstring)."""
    if change_at is None:
        return "null"
    if pos < change_at:
        return "pre"
    if change_at + window // 2 <= pos <= change_at + 3 * window // 2:
        return "change"
    if pos >= change_at + 2 * window:
        return "post"
    return "transition"


def simulate_experiment(exp: Experiment, seed: int, n_assets: int = 4) -> pd.DataFrame:
    return simulate_regimes(list(exp.regimes), n_assets, seed, vol_phi=exp.vol_phi)


def _label(index: pd.DatetimeIndex, dates: pd.DatetimeIndex, exp: Experiment, window: int) -> pd.DataFrame:
    pos = index.get_indexer(dates)
    return pd.DataFrame({"pos": pos, "zone": [zone_of(int(p), exp.change_at, window) for p in pos]},
                        index=dates)


def _scan_rep(args: tuple) -> pd.DataFrame:  # top-level so it can be pickled
    exp, rep, seed, window, step, n_perm, block, q, n_assets = args
    r = simulate_experiment(exp, seed, n_assets)
    scan = change_scan(r, window, step, n_perm, q, seed=seed, block=block)
    return scan.join(_label(r.index, scan.index, exp, window)).assign(rep=rep, experiment=exp.name)


def run_scan_experiment(
    exp: Experiment, seeds: Sequence[int], window: int = 250, step: int = 10, n_perm: int = 199,
    block: int = 10, q: float = 0.1, n_assets: int = 4, n_jobs: int = 1,
) -> pd.DataFrame:
    """Permutation-test scan on each replication; one row per (rep, scan date)."""
    tasks = [(exp, rep, seed, window, step, n_perm, block, q, n_assets) for rep, seed in enumerate(seeds)]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            parts = list(ex.map(_scan_rep, tasks))
    else:
        parts = [_scan_rep(t) for t in tasks]
    return pd.concat(parts).reset_index()


def run_score_experiment(
    exp: Experiment, seeds: Sequence[int], window: int = 250, refit_frequency: int = 10,
    n_assets: int = 4, vine_kwargs: Mapping | None = None, n_jobs: int = 1,
) -> pd.DataFrame:
    """Distance scores from rolling vine fits on each replication."""
    kw = dict(truncation_level=3, tail_simulations=2 ** 12)
    kw.update(vine_kwargs or {})
    lag = default_lag(window, refit_frequency)
    out = []
    for rep, seed in enumerate(seeds):
        r = simulate_experiment(exp, seed, n_assets)
        res = RollingVineModel(window, refit_frequency, kw).run(r, n_jobs=n_jobs)
        sc = structural_change_scores(res, lag=lag, baseline=500 // refit_frequency)
        one = structural_change_scores(res, lag=1, baseline=500 // refit_frequency)
        sc["s_tau_lag1"] = one["s_tau"]
        out.append(sc.join(_label(r.index, sc.index, exp, window)).assign(rep=rep, experiment=exp.name))
    return pd.concat(out).reset_index()


def add_flags(df: pd.DataFrame, flags: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Attach boolean alert columns (``flag_<name>``) to a result table."""
    out = df.copy()
    for name, f in flags.items():
        out[f"flag_{name}"] = f.to_numpy()
    return out


def null_threshold(df: pd.DataFrame, column: str, quantile: float = 0.99) -> float:
    """The ``quantile`` of a score over the no-change dates of ``df`` (used to calibrate
    thresholds of the distance scores, which have no built-in null distribution)."""
    return float(df.loc[df["zone"].isin(NULL_ZONES), column].quantile(quantile))


def detection_summary(df: pd.DataFrame, change_at: int | None, window: int) -> pd.DataFrame:
    """Alert statistics per ``flag_*`` column of one experiment.

    * ``false_alarm``: share of no-change dates (zones null/pre/post) with an alert;
    * ``power``: share of ``change``-zone dates with an alert;
    * ``detected``: share of replications with at least one alert in the change zone;
    * ``median_delay``: median over replications of the days between the change point
      and the first alert in ``[c, c + 2W]`` (NaN if never alerted).
    """
    rows = {}
    for col in [c for c in df.columns if c.startswith("flag_")]:
        d = df[["rep", "pos", "zone", col]].dropna()
        d[col] = d[col].astype(bool)
        row = {"false_alarm": d.loc[d.zone.isin(NULL_ZONES), col].mean()}
        if change_at is not None:
            ch = d[d.zone == "change"]
            win = d[(d.pos >= change_at) & (d.pos <= change_at + 2 * window) & d[col]]
            first = win.groupby("rep").pos.min() - change_at
            row.update(power=ch[col].mean(), detected=ch.groupby("rep")[col].any().mean(),
                       median_delay=float(first.median()) if len(first) else np.nan)
        rows[col.removeprefix("flag_")] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Accuracy of the fitted vine when the true copula is known
# ---------------------------------------------------------------------------

def _truth(regime: Regime, n_assets: int, q: float, alpha: float, n_truth: int, seed: int) -> dict[str, float]:
    x = simulate_regimes([Regime(n_truth, regime.rho, regime.df)], n_assets, seed)
    u = x.rank().to_numpy() / (n_truth + 1)
    lows = [np.mean((u[:, i] < q) & (u[:, j] < q)) / q for i in range(n_assets) for j in range(i + 1, n_assets)]
    var, es = var_es(portfolio_loss(equal_weights(n_assets), x.to_numpy()), alpha)
    return {"lower_tail_q": float(np.mean(lows)), "var": var, "es": es}


def estimation_accuracy(
    regime: Regime, windows: Sequence[int] = (125, 250, 500), n_reps: int = 40, n_assets: int = 4,
    q: float = 0.05, alpha: float = 0.99, n_sims: int = 2 ** 14, n_truth: int = 1_000_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate tail dependence and portfolio VaR/ES from windows of a known copula.

    One row per (window, replication) with the vine's average lower-tail coefficient and
    the 99% VaR/ES under the vine, Gaussian-copula and historical estimators, and the
    truth (from ``n_truth`` simulated observations) repeated in ``true_*`` columns.
    """
    truth = _truth(regime, n_assets, q, alpha, n_truth, seed + 10_000_000)
    rows = []
    for w in windows:
        for rep in range(n_reps):
            r = simulate_regimes([Regime(w, regime.rho, regime.df)], n_assets, seed + 1000 * w + rep)
            u = EmpiricalMarginal().fit_transform(r)
            vc = VineCopula(tail_simulations=2 ** 13, tail_level=q).fit(u, r)
            res = vc.summary()
            risk = window_risk(res, r, alpha=alpha, n_sims=n_sims, seed=seed)
            rows.append({
                "window": w, "rep": rep, "lower_tail_q": res.pairwise_frame()["lower_tail_q"].mean(),
                **{k: risk[k] for k in ("var_vine", "es_vine", "var_gauss", "es_gauss", "var_hist", "es_hist")},
                **{f"true_{k}": v for k, v in truth.items()},
            })
    return pd.DataFrame(rows)


def accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Bias and RMSE (as % of the truth) per window and estimator."""
    pairs = {"lower_tail_q (vine)": ("lower_tail_q", "true_lower_tail_q"),
             "VaR vine": ("var_vine", "true_var"), "VaR gauss": ("var_gauss", "true_var"),
             "VaR hist": ("var_hist", "true_var"), "ES vine": ("es_vine", "true_es"),
             "ES gauss": ("es_gauss", "true_es"), "ES hist": ("es_hist", "true_es")}
    rows = []
    for w, g in df.groupby("window"):
        for label, (est, tru) in pairs.items():
            err = (g[est] - g[tru]) / g[tru]
            rows.append({"window": w, "estimator": label, "truth": g[tru].iloc[0],
                         "mean_estimate": g[est].mean(), "bias_pct": 100 * err.mean(),
                         "rmse_pct": 100 * np.sqrt((err ** 2).mean())})
    return pd.DataFrame(rows)
