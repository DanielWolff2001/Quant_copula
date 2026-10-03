"""Structural change detection on a finished rolling run.

Usage: python scripts/detect_changes.py [--config configs/default.yaml]
       [--run data/results/w250] [--step 5] [--n-perm 499] [--block 10] [--alpha 0.01]

Writes into the run folder:
  structural_change_scores.parquet  timestamp -> structural_change_score (+ diagnostics)
  change_scan.parquet               timestamp -> permutation-test statistics and p-values
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vine_risk.change_detection import benjamini_hochberg, change_scan, default_lag, structural_change_scores
from vine_risk.config import load_config
from vine_risk.data import download_prices
from vine_risk.returns import build_return_matrix
from vine_risk.rolling import load_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="data/results/w250")
    ap.add_argument("--step", type=int, default=5, help="scan every N observations")
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--q", type=float, default=0.1, help="level of the empirical tail coefficients")
    ap.add_argument("--block", type=int, default=10, help="permute blocks of this many days (keeps volatility clustering)")
    ap.add_argument("--alpha", type=float, default=0.01, help="FDR level for the reported alerts")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(a.config)
    run = Path(a.run)
    window, freq = cfg.rolling.window, cfg.rolling.refit_frequency

    results = load_results(run / "checkpoint.jsonl")
    scores = structural_change_scores(results, lag=default_lag(window, freq), baseline=500 // freq)
    scores.to_parquet(run / "structural_change_scores.parquet")

    d = cfg.data
    returns = build_return_matrix(download_prices(cfg.assets, d.start, d.end, d.cache_dir),
                                  d.max_missing_frac, d.max_ffill_days)
    scan = change_scan(returns, window, a.step, a.n_perm, a.q, cfg.risk.seed, a.block)
    for c in [c for c in scan if c.startswith("p_")]:
        scan[f"{c}_fdr{a.alpha}"] = benjamini_hochberg(scan[c], a.alpha)
    scan.to_parquet(run / "change_scan.parquet")
    print(f"{len(scores)} score rows, {len(scan)} scan dates -> {run}")


if __name__ == "__main__":
    main()
