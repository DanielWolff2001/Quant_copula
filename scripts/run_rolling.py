"""Run the rolling vine fit on the configured universe.

Usage: python scripts/run_rolling.py [--config configs/default.yaml] [--window 250]
       [--refit-frequency 1] [--last N] [--out data/results/w250]

Same as the first step of ``vine-risk run``. Results are checkpointed to ``<out>/checkpoint.jsonl`` (re-run
to resume) and written as parquet tables to ``<out>``. Refuses to extend a folder made with different fit
settings; see ``vine-risk run --help`` for the whole pipeline.
"""
from __future__ import annotations

import argparse
import logging

from vine_risk.cli import TextProgress
from vine_risk.config import load_config
from vine_risk.pipeline import load_prices_and_returns
from vine_risk.runner import default_run_dir, fit_rolling, with_rolling


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--window", type=int)
    ap.add_argument("--refit-frequency", type=int)
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--last", type=int, help="only use the last N return observations")
    ap.add_argument("--out", help="output directory (default data/results/w<window>)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = with_rolling(load_config(a.config), window=a.window, refit_frequency=a.refit_frequency, n_jobs=a.n_jobs)
    _, returns = load_prices_and_returns(cfg)
    if a.last:
        returns = returns.iloc[-a.last:]
    out = a.out or default_run_dir(cfg)
    results = fit_rolling(cfg, returns, out, progress=TextProgress("rolling fits"))
    failed = sum(r.status != "ok" for r in results)
    print(f"{len(results)} fits ({failed} failed), {results[0].timestamp} .. {results[-1].timestamp} -> {out}")


if __name__ == "__main__":
    main()
