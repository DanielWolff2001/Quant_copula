"""Compute the dependence monitoring metrics from a finished (or partial) rolling run.

Usage: python scripts/compute_metrics.py [--run data/results/w250]

Reads ``<run>/checkpoint.jsonl`` and writes ``dependence_metrics.parquet`` plus wide
pairwise tables (``pairwise_tau.parquet``, ``pairwise_lower_tail_q.parquet``, ...) into
the same folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vine_risk.dependence import dependence_metrics, pairwise_series
from vine_risk.rolling import load_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="data/results/w250")
    run = Path(ap.parse_args().run)
    results = load_results(run / "checkpoint.jsonl")
    dependence_metrics(results).to_parquet(run / "dependence_metrics.parquet")
    for col in ("tau", "spearman", "model_tau", "lower_tail_q", "upper_tail_q"):
        pairwise_series(results, col).to_parquet(run / f"pairwise_{col}.parquet")
    print(f"{len(results)} fits -> {run}/dependence_metrics.parquet (+ pairwise_*.parquet)")


if __name__ == "__main__":
    main()
