"""Simulated live monitoring: replay history one observation at a time.

Usage: python scripts/run_live.py [--run data/results/w250] [--days 120 | --start 2026-04-01]
       [--end 2026-10-02] [--delay 0] [--no-verify] [--out <run>/live]

Same as ``vine-risk replay``. The monitor is primed with the history before ``--start`` and the fits
already stored in ``<run>/checkpoint.jsonl`` (as a live system resuming from saved state would be), then
each later price is fed in turn. Nothing after the current date is ever visible to it. Writes
``live_log.jsonl``, ``live_log.parquet`` and ``live_alerts.csv`` to ``--out``; unless ``--no-verify`` is
given, the replay is compared with the batch results stored in the run folder.
"""
from __future__ import annotations

import argparse
import logging

from vine_risk.config import load_config
from vine_risk.replay import run_replay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="data/results/w250")
    ap.add_argument("--days", type=int, default=120, help="replay the last N observations")
    ap.add_argument("--start", help="first replayed date (overrides --days)")
    ap.add_argument("--end", help="last replayed date")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between observations")
    ap.add_argument("--scan-step", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--risk-every", type=int, default=5, help="VaR/ES at every N-th fit (0 = never)")
    ap.add_argument("--threads", type=int, default=4, help="threads of the vine fitter")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_replay(load_config(a.config), a.run, days=a.days, start=a.start, end=a.end, delay=a.delay,
               scan_step=a.scan_step, n_perm=a.n_perm, risk_every=a.risk_every, threads=a.threads,
               verify=not a.no_verify, out_dir=a.out)


if __name__ == "__main__":
    main()
