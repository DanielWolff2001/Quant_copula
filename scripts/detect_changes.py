"""Structural change detection on a finished rolling run.

Usage: python scripts/detect_changes.py [--config configs/default.yaml] [--run data/results/w250]
       [--step 5] [--n-perm 499] [--block 10] [--alpha 0.01]

Same as the ``changes`` step of ``vine-risk run``. Writes into the run folder:
  structural_change_scores.parquet  timestamp -> structural_change_score (+ diagnostics)
  change_scan.parquet               timestamp -> permutation-test statistics and p-values
The scan is extended from its last stored date when it was made with the same settings.
"""
from __future__ import annotations

import argparse
import logging

from vine_risk.cli import TextProgress
from vine_risk.config import load_config
from vine_risk.pipeline import load_prices_and_returns
from vine_risk.runner import detect_changes


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
    _, returns = load_prices_and_returns(cfg)
    scores, scan = detect_changes(cfg, returns, a.run, step=a.step, n_perm=a.n_perm, q=a.q, block=a.block,
                                  alpha=a.alpha, progress=TextProgress("permutation scan"))
    print(f"{len(scores)} score rows, {len(scan)} scan dates -> {a.run}")


if __name__ == "__main__":
    main()
