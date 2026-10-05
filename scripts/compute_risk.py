"""Portfolio VaR / Expected Shortfall along a finished rolling run.

Usage: python scripts/compute_risk.py [--config configs/default.yaml] [--run data/results/w250] [--step 5]

Same as the ``risk`` step of ``vine-risk run``. Rebuilds the fitted vine of every ``step``-th window from the
stored results (no refitting), simulates portfolio losses, and writes ``portfolio_risk.parquet`` and
``var_backtest.csv`` into the run folder. Extends earlier results when the settings are unchanged.
"""
from __future__ import annotations

import argparse
import logging

from vine_risk.config import load_config
from vine_risk.pipeline import load_prices_and_returns
from vine_risk.runner import compute_risk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="data/results/w250")
    ap.add_argument("--step", type=int, default=5, help="compute risk for every N-th fit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    _, returns = load_prices_and_returns(cfg)
    risk, bt = compute_risk(cfg, returns, a.run, step=a.step)
    print(bt.round(4).to_string())
    print(f"{len(risk)} dates -> {a.run}/portfolio_risk.parquet")


if __name__ == "__main__":
    main()
