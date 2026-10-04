"""Portfolio VaR / Expected Shortfall along a finished rolling run.

Usage: python scripts/compute_risk.py [--config configs/default.yaml]
       [--run data/results/w250] [--step 5]

Rebuilds the fitted vine of every ``step``-th window from the stored results (no
refitting), simulates portfolio losses, and writes ``portfolio_risk.parquet`` and
``var_backtest.csv`` into the run folder.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from vine_risk.config import load_config
from vine_risk.data import download_prices
from vine_risk.portfolio import backtest_var, rolling_risk
from vine_risk.returns import build_return_matrix
from vine_risk.rolling import load_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="data/results/w250")
    ap.add_argument("--step", type=int, default=5, help="compute risk for every N-th fit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(a.config)
    run = Path(a.run)
    d = cfg.data
    returns = build_return_matrix(download_prices(cfg.assets, d.start, d.end, d.cache_dir),
                                  d.max_missing_frac, d.max_ffill_days)
    w = None if cfg.risk.weights is None else np.asarray(cfg.risk.weights, dtype=float)
    if w is not None and len(w) != returns.shape[1]:
        raise ValueError(f"risk.weights has {len(w)} entries for {returns.shape[1]} assets.")

    results = load_results(run / "checkpoint.jsonl")
    risk = rolling_risk(results, returns, w, cfg.risk.confidence_level, cfg.risk.simulations,
                        cfg.risk.seed, a.step, cfg.rolling.n_jobs)
    risk.to_parquet(run / "portfolio_risk.parquet")
    bt = backtest_var(risk, returns, w, cfg.risk.confidence_level)
    bt.to_csv(run / "var_backtest.csv")
    print(bt.round(4).to_string())
    print(f"{len(risk)} dates -> {run}/portfolio_risk.parquet")


if __name__ == "__main__":
    main()
