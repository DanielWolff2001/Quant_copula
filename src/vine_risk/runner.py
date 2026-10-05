"""The pipeline steps, as library functions shared by the command line and the scripts.

Steps, in order (each reads what the previous ones wrote into the *run folder*):

1. `fit_rolling`     - rolling vine fits           -> ``checkpoint.jsonl`` + result tables
2. `compute_metrics` - dependence metrics          -> ``dependence_metrics.parquet``, ``pairwise_*.parquet``
3. `detect_changes`  - change scores + scan        -> ``structural_change_scores.parquet``, ``change_scan.parquet``
4. `compute_risk`    - rolling VaR / ES + backtest -> ``portfolio_risk.parquet``, ``var_backtest.csv``

Every step is **incremental**: it extends what is already in the run folder (new fits, new scan
dates, new risk dates) instead of recomputing, which makes a daily update cheap. A
``manifest.json`` records the code, settings and data behind the results; resuming with
different fit settings, or after the vendor revised old prices, is refused.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from vine_risk.change_detection import benjamini_hochberg, change_scan, default_lag, structural_change_scores
from vine_risk.config import Config
from vine_risk.copula import VineFitResult
from vine_risk.dependence import dependence_metrics, pairwise_series
from vine_risk.manifest import build_manifest, check_resume, read_manifest, record_step, verify_history_unchanged, write_manifest
from vine_risk.portfolio import backtest_var, rolling_risk
from vine_risk.rolling import CheckpointIndex, RollingVineModel, load_results, save_results

logger = logging.getLogger(__name__)
Progress = Callable[[int, int], None]

STEPS = ("rolling", "metrics", "changes", "risk")
PAIRWISE_COLUMNS = ("tau", "spearman", "model_tau", "lower_tail_q", "upper_tail_q")


def default_run_dir(cfg: Config, root: str | Path = "data/results") -> Path:
    """``data/results/w<window>`` (the folder name used throughout the documentation)."""
    return Path(root) / f"w{cfg.rolling.window}"


def with_rolling(cfg: Config, **overrides) -> Config:
    """``cfg`` with some ``rolling`` settings replaced (``None`` values are ignored)."""
    over = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, rolling=replace(cfg.rolling, **over)) if over else cfg


def _timed(run_dir: Path, name: str, info: dict, t0: float) -> None:
    record_step(run_dir, name, {**info, "seconds": round(time.time() - t0, 1)})


# ------------------------------------------------------------------ 1. rolling fits
def fit_rolling(cfg: Config, returns: pd.DataFrame, run_dir: str | Path, *, n_jobs: int | None = None,
                progress: Progress | None = None, force: bool = False,
                command: Sequence[str] | None = None) -> list[VineFitResult]:
    """Fit (or extend) the rolling vines on ``returns`` and store them in ``run_dir``.

    Unless ``force`` is given, refuses to extend a run folder that was made with different fit
    settings, or whose stored fits no longer match the data (revised prices).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoint.jsonl"
    model = RollingVineModel.from_config(cfg)
    if not force:
        check_resume(run_dir, cfg)
        if ckpt.is_file():
            latest = CheckpointIndex(ckpt).latest()
            if latest is not None:
                verify_history_unchanged(latest, returns, marginal_factory=model.marginal_factory, lookback=model.lookback)
    write_manifest(run_dir, build_manifest(cfg, returns, command))
    t0 = time.time()
    results = model.run(returns, n_jobs=n_jobs or cfg.rolling.n_jobs, checkpoint=ckpt, progress=progress)
    save_results(results, run_dir)
    failed = sum(r.status != "ok" for r in results)
    _timed(run_dir, "rolling", {"fits": len(results), "failed": failed,
                                "first": results[0].timestamp, "last": results[-1].timestamp}, t0)
    return results


# ------------------------------------------------------------------ 2. metrics
def compute_metrics(run_dir: str | Path, results: Sequence[VineFitResult] | None = None) -> pd.DataFrame:
    """Dependence metrics for every fit; writes ``dependence_metrics.parquet`` and ``pairwise_*.parquet``."""
    run_dir = Path(run_dir)
    t0 = time.time()
    results = list(results) if results is not None else load_results(run_dir / "checkpoint.jsonl")
    metrics = dependence_metrics(results)
    metrics.to_parquet(run_dir / "dependence_metrics.parquet")
    for col in PAIRWISE_COLUMNS:
        pairwise_series(results, col).to_parquet(run_dir / f"pairwise_{col}.parquet")
    _timed(run_dir, "metrics", {"fits": len(results)}, t0)
    return metrics


# ------------------------------------------------------------------ 3. change detection
def add_fdr_flags(scan: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    """Add a Benjamini-Hochberg flag ``<p column>_fdr<alpha>`` for every p-value column of ``scan``."""
    out = scan.copy()
    for c in [c for c in out.columns if c.startswith("p_") and "_fdr" not in c]:
        out[f"{c}_fdr{alpha}"] = benjamini_hochberg(out[c], alpha)
    return out


def detect_changes(cfg: Config, returns: pd.DataFrame, run_dir: str | Path, *, step: int = 5, n_perm: int = 499,
                   q: float = 0.1, block: int = 10, alpha: float = 0.01, progress: Progress | None = None,
                   results: Sequence[VineFitResult] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change scores (from the fits) and the permutation-test scan (from the returns).

    The scores are recomputed (cheap); the scan is **extended** from its last stored date when the
    stored one was made with the same settings, otherwise recomputed. Returns ``(scores, scan)``.
    """
    run_dir = Path(run_dir)
    t0 = time.time()
    results = list(results) if results is not None else load_results(run_dir / "checkpoint.jsonl")
    window = cfg.rolling.window
    freq = _fit_spacing(results, returns)
    scores = structural_change_scores(results, lag=default_lag(window, freq), baseline=max(20, 500 // freq))
    scores.to_parquet(run_dir / "structural_change_scores.parquet")

    if len(returns) < 2 * window:  # the permutation test compares two full windows
        logger.warning("Only %d observations; the permutation scan needs %d (two windows). Skipping the scan.",
                       len(returns), 2 * window)
        _timed(run_dir, "changes", {"scan_dates": 0, "skipped": f"needs {2 * window} observations"}, t0)
        return scores, pd.DataFrame()
    settings = {"window": window, "step": step, "n_perm": n_perm, "q": q, "block": block, "seed": cfg.risk.seed,
                "first_return": str(returns.index[0].date())}
    path = run_dir / "change_scan.parquet"
    old = read_manifest(run_dir)
    same = old is not None and old.get("steps", {}).get("changes", {}).get("settings") == settings
    stored = _read_table(path) if same and path.is_file() else None
    after = None if stored is None else stored.index.max()
    new = change_scan(returns, window, step, n_perm, q, cfg.risk.seed, block, after=after, progress=progress)
    base = stored[[c for c in stored.columns if "_fdr" not in c]] if stored is not None else None
    scan = pd.concat([base, new]) if base is not None and not new.empty else (base if base is not None else new)
    scan = add_fdr_flags(scan, alpha)
    scan.to_parquet(path)
    _timed(run_dir, "changes", {"settings": settings, "scan_dates": len(scan), "new_scan_dates": len(new)}, t0)
    return scores, scan


def _fit_spacing(results: Sequence[VineFitResult], returns: pd.DataFrame) -> int:
    """Median number of observations between consecutive fits (the refit frequency in effect)."""
    if len(results) < 2:
        return 1
    pos = returns.index.get_indexer(pd.DatetimeIndex([r.timestamp for r in results]))
    return max(1, int(np.median(np.diff(pos))))


# ------------------------------------------------------------------ 4. risk
def compute_risk(cfg: Config, returns: pd.DataFrame, run_dir: str | Path, *, step: int = 5,
                 n_jobs: int | None = None, results: Sequence[VineFitResult] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling VaR/ES at every ``step``-th fit, extended from the last stored date; plus the VaR backtest."""
    run_dir = Path(run_dir)
    t0 = time.time()
    results = list(results) if results is not None else load_results(run_dir / "checkpoint.jsonl")
    w = None if cfg.risk.weights is None else np.asarray(cfg.risk.weights, dtype=float)
    if w is not None and len(w) != returns.shape[1]:
        raise ValueError(f"risk.weights has {len(w)} entries for {returns.shape[1]} assets.")
    settings = {"alpha": cfg.risk.confidence_level, "sims": cfg.risk.simulations, "seed": cfg.risk.seed,
                "weights": None if w is None else list(map(float, w)), "step": step,
                "first_fit": results[0].timestamp}
    path = run_dir / "portfolio_risk.parquet"
    old = read_manifest(run_dir)
    same = old is not None and old.get("steps", {}).get("risk", {}).get("settings") == settings
    stored = _read_table(path) if same and path.is_file() else None
    new = rolling_risk(results, returns, w, cfg.risk.confidence_level, cfg.risk.simulations, cfg.risk.seed, step,
                       n_jobs or cfg.rolling.n_jobs, after=None if stored is None else stored.index.max())
    risk = pd.concat([stored, new]) if stored is not None and not new.empty else (stored if stored is not None else new)
    risk.to_parquet(path)
    backtest = backtest_var(risk, returns, w, cfg.risk.confidence_level)
    backtest.to_csv(run_dir / "var_backtest.csv")
    _timed(run_dir, "risk", {"settings": settings, "dates": len(risk), "new_dates": len(new)}, t0)
    return risk, backtest


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    return df.sort_index()


# ------------------------------------------------------------------ everything
def run_pipeline(cfg: Config, returns: pd.DataFrame, run_dir: str | Path, steps: Sequence[str] = STEPS, *,
                 n_jobs: int | None = None, scan_step: int = 5, n_perm: int = 499, risk_step: int = 5,
                 progress: Callable[[str], Progress | None] | None = None, force: bool = False,
                 command: Sequence[str] | None = None) -> dict[str, float]:
    """Run the chosen steps in order and return the seconds each took.

    ``progress(step_name)`` may return a ``progress(done, total)`` callback for that step.
    """
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown step(s) {sorted(unknown)}; choose from {STEPS}.")
    run_dir = Path(run_dir)
    ordered = [s for s in STEPS if s in steps]
    cb = (lambda name: progress(name)) if progress else (lambda name: None)
    timings: dict[str, float] = {}
    results: list[VineFitResult] | None = None
    for name in ordered:
        t0 = time.time()
        if name == "rolling":
            results = fit_rolling(cfg, returns, run_dir, n_jobs=n_jobs, progress=cb(name), force=force, command=command)
        else:
            if results is None:
                results = load_results(run_dir / "checkpoint.jsonl")
            if name == "metrics":
                compute_metrics(run_dir, results)
            elif name == "changes":
                detect_changes(cfg, returns, run_dir, step=scan_step, n_perm=n_perm, progress=cb(name), results=results)
            elif name == "risk":
                compute_risk(cfg, returns, run_dir, step=risk_step, n_jobs=n_jobs, results=results)
        timings[name] = time.time() - t0
    return timings
