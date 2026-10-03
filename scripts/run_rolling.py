"""Run the rolling vine fit on the configured universe.

Usage: python scripts/run_rolling.py [--config configs/default.yaml] [--window 250]
       [--refit-frequency 1] [--last N] [--out data/results/w250]

Results are checkpointed to ``<out>/checkpoint.jsonl`` (re-run to resume) and written
as parquet tables to ``<out>``.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from vine_risk.config import load_config
from vine_risk.data import download_prices
from vine_risk.returns import build_return_matrix
from vine_risk.rolling import RollingVineModel, save_results


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
    cfg = load_config(a.config)
    over = {k: v for k, v in dict(window=a.window, refit_frequency=a.refit_frequency,
                                  n_jobs=a.n_jobs).items() if v is not None}
    cfg = dataclasses.replace(cfg, rolling=dataclasses.replace(cfg.rolling, **over))

    d = cfg.data
    returns = build_return_matrix(download_prices(cfg.assets, d.start, d.end, d.cache_dir),
                                  d.max_missing_frac, d.max_ffill_days)
    if a.last:
        returns = returns.iloc[-a.last:]
    out = Path(a.out or f"data/results/w{cfg.rolling.window}")
    out.mkdir(parents=True, exist_ok=True)

    model = RollingVineModel.from_config(cfg)
    results = model.run(returns, n_jobs=cfg.rolling.n_jobs, checkpoint=out / "checkpoint.jsonl")
    save_results(results, out)
    failed = sum(r.status != "ok" for r in results)
    print(f"{len(results)} fits ({failed} failed), {results[0].timestamp} .. {results[-1].timestamp} -> {out}")


if __name__ == "__main__":
    main()
