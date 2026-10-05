"""Compute the dependence monitoring metrics from a finished (or partial) rolling run.

Usage: python scripts/compute_metrics.py [--run data/results/w250]

Same as the ``metrics`` step of ``vine-risk run``. Writes ``dependence_metrics.parquet`` plus wide pairwise
tables (``pairwise_tau.parquet``, ``pairwise_lower_tail_q.parquet``, ...) into the run folder.
"""
from __future__ import annotations

import argparse

from vine_risk.rolling import load_results
from vine_risk.runner import compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="data/results/w250")
    run = ap.parse_args().run
    n = len(compute_metrics(run))
    print(f"{n} fits -> {run}/dependence_metrics.parquet (+ pairwise_*.parquet)")


if __name__ == "__main__":
    main()
