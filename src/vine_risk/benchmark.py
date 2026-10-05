"""Compare the vine copula's risk forecasts with standard models, on the same footing.

For every forecast date (an *origin*), all models produce a one-day-ahead VaR and Expected Shortfall
for every portfolio and confidence level, using only the data up to that date:

=============  ==========================================================================================
``hist``       historical simulation, last 250 days
``ewma_n``     RiskMetrics: EWMA covariance, normal returns
``ewma_t``     EWMA covariance, Student-t returns
``fhs``        filtered historical simulation (GARCH volatility, jointly resampled residuals)
``vine_emp``   vine copula with rank marginals (the window's own distribution)
``gauss_emp``  Gaussian copula, same marginals;  ``indep_emp``: independence
``vine_garch`` vine copula on GARCH-filtered marginals, forecasts through today's volatility
``gauss_garch``, ``indep_garch``  the same benchmarks with GARCH marginals
=============  ==========================================================================================

The copula models need fitted vines, which come from two run folders made with
:mod:`vine_risk.runner` (one with rank marginals, one with ``rolling.marginal: garch_t``).
:func:`evaluate` backtests everything with :mod:`vine_risk.backtesting`.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from vine_risk.backtesting import evaluate_forecasts
from vine_risk.config import Config
from vine_risk.copula import VineFitResult
from vine_risk.garch import GarchMarginal
from vine_risk.manifest import read_manifest, stable_hash
from vine_risk.portfolio import loss_summary, portfolio_loss, scenario_returns
from vine_risk.riskmodels import (
    EwmaNormal, EwmaStudentT, FilteredHistoricalSimulation, HistoricalSimulation, RiskForecast,
)
from vine_risk.rolling import CheckpointIndex

MODELS = ("hist", "ewma_n", "ewma_t", "fhs", "vine_emp", "gauss_emp", "indep_emp", "vine_garch", "gauss_garch", "indep_garch")
DEFAULT_ALPHAS = (0.975, 0.99)


def resolve_portfolios(cfg: Config) -> dict[str, np.ndarray]:
    """Named weight vectors from ``risk.portfolios`` (weights per ticker, normalised to sum to 1).

    ``None`` as a weight specification means equal weights; without any configuration a single
    equal-weight portfolio called ``equal`` is used.
    """
    spec = cfg.risk.portfolios or {"equal": None}
    out = {}
    for name, w in spec.items():
        if w is None:
            vec = np.ones(len(cfg.assets))
        else:
            unknown = set(w) - set(cfg.assets)
            if unknown:
                raise ValueError(f"portfolio {name!r} uses tickers that are not in `assets`: {sorted(unknown)}")
            vec = np.array([float(w.get(a, 0.0)) for a in cfg.assets])
        if (vec < 0).any() or vec.sum() <= 0:
            raise ValueError(f"portfolio {name!r} needs non-negative weights that are not all zero.")
        out[name] = vec / vec.sum()
    return out


@dataclass(frozen=True)
class BenchmarkSpec:
    portfolios: Mapping[str, np.ndarray]
    alphas: Sequence[float] = DEFAULT_ALPHAS
    window: int = 250  # historical simulation window
    ewma_lambda: float = 0.94
    ewma_lookback: int = 500
    garch_lookback: int = 500  # history of the GARCH marginals; must equal the GARCH run's lookback
    n_sims: int = 2 ** 14
    seed: int = 42

    def fingerprint(self) -> str:
        return stable_hash({"portfolios": {k: list(map(float, v)) for k, v in self.portfolios.items()},
                            "alphas": list(self.alphas), "window": self.window, "lam": self.ewma_lambda,
                            "ewma_lookback": self.ewma_lookback, "garch_lookback": self.garch_lookback,
                            "n_sims": self.n_sims, "seed": self.seed})

    @property
    def history_needed(self) -> int:
        return max(self.window, self.ewma_lookback, self.garch_lookback)


def _copula_forecasts(prefix: str, scen: Mapping[str, np.ndarray], spec: BenchmarkSpec) -> list[RiskForecast]:
    out = []
    for kind, r in scen.items():
        for pname, w in spec.portfolios.items():
            for a, s in loss_summary(portfolio_loss(np.asarray(w), r), spec.alphas).items():
                out.append(RiskForecast(f"{kind}_{prefix}", pname, a, s["var"], s["es"], s["tail"]))
    return out


def forecast_origin(spec: BenchmarkSpec, history: pd.DataFrame, emp_fit: VineFitResult | None = None,
                    garch_fit: VineFitResult | None = None) -> list[RiskForecast]:
    """All models' forecasts for the day after the last row of ``history`` (which holds enough past data).

    Copula models are included for the fits that are given.
    """
    pf, al = dict(spec.portfolios), list(spec.alphas)
    out: list[RiskForecast] = []
    out += HistoricalSimulation(spec.window).forecast(history, pf, al)
    out += EwmaNormal(spec.ewma_lambda, spec.ewma_lookback).forecast(history, pf, al)
    out += EwmaStudentT(spec.ewma_lambda, spec.ewma_lookback).forecast(history, pf, al)
    ghist = history.iloc[-spec.garch_lookback:]
    garch = GarchMarginal(innovations="t").fit(ghist)  # shared by the filtered models
    out += FilteredHistoricalSimulation(spec.garch_lookback).forecast(history, pf, al, garch=garch)
    if emp_fit is not None:
        win = history.iloc[-emp_fit.n_obs:]
        out += _copula_forecasts("emp", scenario_returns(emp_fit, win, n_sims=spec.n_sims, seed=spec.seed), spec)
    if garch_fit is not None:
        win = history.iloc[-garch_fit.n_obs:]
        out += _copula_forecasts("garch", scenario_returns(garch_fit, win, marginal=garch, n_sims=spec.n_sims, seed=spec.seed), spec)
    return out


def _origin_task(args: tuple) -> list[dict]:  # top-level so it can be pickled
    spec, date, history, emp_fit, garch_fit = args
    return [{"date": date, "model": f.model, "portfolio": f.portfolio, "alpha": f.alpha, "var": f.var, "es": f.es,
             "tail": f.tail} for f in forecast_origin(spec, history, emp_fit, garch_fit)]


def realized_losses(returns: pd.DataFrame, portfolios: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """The loss each portfolio suffers on a day, indexed by that day's *previous* date (the forecast origin)."""
    loss = pd.DataFrame({p: portfolio_loss(np.asarray(w), returns.to_numpy()) for p, w in portfolios.items()}, index=returns.index)
    nxt = loss.shift(-1).dropna()
    nxt.index.name = "date"
    return nxt


def origin_dates(emp_index: CheckpointIndex | None, garch_index: CheckpointIndex | None, returns: pd.DataFrame,
                 spec: BenchmarkSpec, step: int = 1, first: str | None = None) -> pd.DatetimeIndex:
    """Forecast origins: dates with every required fit, enough history, and a next day to be judged against."""
    sets = [set(ix.timestamps) for ix in (emp_index, garch_index) if ix is not None]
    dates = sorted(set.intersection(*sets)) if sets else list(returns.index)
    pos = returns.index.get_indexer(pd.DatetimeIndex(dates))
    ok = (pos >= spec.history_needed - 1) & (pos < len(returns) - 1)
    out = pd.DatetimeIndex(np.asarray(dates)[ok])[::step]
    return out[out >= pd.Timestamp(first)] if first else out


def run_benchmark(
    cfg: Config, returns: pd.DataFrame, out_dir: str | Path, *, emp_run: str | Path | None = None,
    garch_run: str | Path | None = None, spec: BenchmarkSpec | None = None, step: int = 1, first: str | None = None,
    n_jobs: int = 1, progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Compute (or extend) the table of one-day-ahead forecasts of every model; returns it.

    Writes ``forecasts.parquet`` (long table: ``date, model, portfolio, alpha, var, es, tail``),
    ``realized.parquet`` and ``benchmark.json`` into ``out_dir``. Dates already in the table are not
    recomputed when the settings are unchanged, so an interrupted or extended run resumes cheaply.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = spec or BenchmarkSpec(resolve_portfolios(cfg), garch_lookback=cfg.rolling.lookback if cfg.rolling.marginal != "empirical" else 500,
                                 seed=cfg.risk.seed, n_sims=cfg.risk.simulations)
    emp_ix = CheckpointIndex(Path(emp_run) / "checkpoint.jsonl") if emp_run else None
    gar_ix = CheckpointIndex(Path(garch_run) / "checkpoint.jsonl") if garch_run else None
    origins = origin_dates(emp_ix, gar_ix, returns, spec, step, first)
    if len(origins) == 0:
        raise ValueError(
            f"No forecast dates: every model needs {spec.history_needed} past observations (the longest lookback) and, "
            f"where fits are used, a stored fit; {len(returns)} observations are available. Shorten the lookbacks "
            "(--hist-window, --ewma-lookback, the run's marginal lookback) or use more data.")
    meta = {"fingerprint": spec.fingerprint(), "step": step, "emp_run": str(emp_run), "garch_run": str(garch_run)}
    path = out_dir / "forecasts.parquet"
    old = None
    prev = out_dir / "benchmark.json"
    if path.is_file() and prev.is_file() and _load_meta(prev).get("fingerprint") == meta["fingerprint"] \
            and _load_meta(prev).get("step") == step:
        old = pd.read_parquet(path)
    done = set(pd.DatetimeIndex(old["date"]).unique()) if old is not None else set()
    todo = [d for d in origins if d not in done]
    t0 = time.time()
    tasks = []
    for d in todo:
        end = returns.index.get_loc(d)
        tasks.append((spec, d, returns.iloc[max(0, end - spec.history_needed + 1): end + 1],
                      emp_ix.get(d) if emp_ix else None, gar_ix.get(d) if gar_ix else None))
    if n_jobs > 1 and tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            parts = []
            for i, rows in enumerate(ex.map(_origin_task, tasks, chunksize=max(1, len(tasks) // (n_jobs * 8))), start=1):
                parts.append(rows)
                if progress:
                    progress(i, len(tasks))
    else:
        parts = []
        for i, t in enumerate(tasks, start=1):
            parts.append(_origin_task(t))
            if progress:
                progress(i, len(tasks))
    new = pd.DataFrame([r for rows in parts for r in rows])
    table = pd.concat([old, new], ignore_index=True) if old is not None and not new.empty else (old if old is not None else new)
    table = table.sort_values(["date", "portfolio", "alpha", "model"], kind="stable").reset_index(drop=True)
    table.to_parquet(path)
    realized_losses(returns, spec.portfolios).to_parquet(out_dir / "realized.parquet")
    _save_meta(prev, {**meta, "origins": int(table["date"].nunique()), "new_origins": len(todo),
                      "seconds": round(time.time() - t0, 1), "models": sorted(table["model"].unique())})
    return table


def _load_meta(path: Path) -> dict:
    import json
    return json.loads(path.read_text())


def _save_meta(path: Path, meta: dict) -> None:
    import json
    path.write_text(json.dumps(meta, indent=1))


def evaluate(out_dir: str | Path, reference: str | None = "vine_garch", n_sim: int = 2000, seed: int = 0,
             dm_lag: int | None = None) -> pd.DataFrame:
    """Backtest the forecast table in ``out_dir`` and write ``report.csv``; see `evaluate_forecasts`.

    Only dates on which *all* models have forecasts are used, so the models are compared on the same days.
    """
    out_dir = Path(out_dir)
    table = pd.read_parquet(out_dir / "forecasts.parquet")
    realized = pd.read_parquet(out_dir / "realized.parquet")
    n_models = table["model"].nunique()
    counts = table.groupby("date")["model"].nunique()
    common = counts.index[counts == counts.max()]
    table = table[table["date"].isin(common)]
    if reference is not None and reference not in set(table["model"]):
        reference = None
    report = evaluate_forecasts(table, realized, reference, dm_lag, n_sim, seed)
    report.to_csv(out_dir / "report.csv")
    return report


def format_report(report: pd.DataFrame) -> str:
    """A compact text version of `evaluate`'s table, one block per portfolio and level."""
    cols = [c for c in ("exceedances", "rate", "kupiec_p", "cc_p", "z2_p", "fz0", "dm_p") if c in report.columns]
    lines = []
    for (pname, alpha), g in report.groupby(level=["portfolio", "alpha"], sort=True):
        g = g.droplevel(["portfolio", "alpha"])
        n = int(g["n"].iloc[0])
        lines.append(f"\nportfolio {pname}, {alpha:.1%} level ({n} days; expected exceedance rate {1 - alpha:.1%})")
        show = g[cols].copy()
        show["rate"] = (100 * show["rate"]).round(2)
        lines.append(show.rename(columns={"rate": "rate %"}).round(3).to_string())
    return "\n".join(lines)
