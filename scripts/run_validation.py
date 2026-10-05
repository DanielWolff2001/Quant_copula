"""Run the synthetic validation study (PDF section 20) and write summary tables.

Usage: python scripts/run_validation.py [--reps-scan 20] [--reps-fit 10] [--reps-acc 40]
       [--window 250] [--n-jobs 4] [--out reports/validation] [--raw data/results/validation]

Takes roughly half an hour. Everything is seeded, so reruns give identical tables.
Writes ``detection_scan.csv``, ``detection_scores.csv``, ``detection_scores_volcal.csv``,
``estimation_accuracy.csv`` and ``conditional_accuracy.csv`` to ``--out`` (small, kept in git) and the raw per-date
tables as parquet to ``--raw``. With ``--reuse`` the raw tables already in ``--raw`` are
loaded instead of simulated again (only the summary tables are recomputed).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from vine_risk.synthetic import Regime
from vine_risk.validation import (
    accuracy_summary, add_flags, conditional_accuracy, conditional_accuracy_summary, detection_summary,
    estimation_accuracy, null_threshold,
    run_scan_experiment, run_score_experiment, standard_experiments,
)

log = logging.getLogger("validation")
SCAN_STATS = ("tau", "mean_tau", "lower", "mean_lower", "upper", "mean_upper")
SCORES = ("s_tau", "dist_model", "dist_lower", "dist_upper", "s_tau_lag1", "cusum_d_t", "z_d_t")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps-scan", type=int, default=20)
    ap.add_argument("--reps-fit", type=int, default=10)
    ap.add_argument("--reps-acc", type=int, default=40)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="reuse raw tables found in --raw")
    ap.add_argument("--out", default="reports/validation")
    ap.add_argument("--raw", default="data/results/validation")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out, raw = Path(a.out), Path(a.raw)
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    exps = standard_experiments()
    W = a.window

    def cached(path: Path, make):
        if a.reuse and path.exists():
            return pd.read_parquet(path)
        df = make()
        df.to_parquet(path)
        return df

    # 1. permutation-test scan (calibrated by construction; alert = p <= alpha)
    scan_tables = {}
    for name, exp in exps.items():
        log.info("scan %s", name)
        df = cached(raw / f"scan_{name}.parquet", lambda: run_scan_experiment(
            exp, [1000 + i for i in range(a.reps_scan)], W, n_jobs=a.n_jobs))
        df = add_flags(df, {s: df[f"p_{s}"] <= a.alpha for s in SCAN_STATS})
        scan_tables[name] = detection_summary(df, exp.change_at, W)
    pd.concat(scan_tables, names=["experiment", "statistic"]).to_csv(out / "detection_scan.csv")

    # 2. distance scores from rolling vine fits; thresholds calibrated on A_const only
    score_dfs = {}
    for name, exp in exps.items():
        log.info("scores %s", name)
        score_dfs[name] = cached(raw / f"scores_{name}.parquet", lambda: run_score_experiment(
            exp, [2000 + i for i in range(a.reps_fit)], W, n_jobs=a.n_jobs))
    # thresholds = 99th percentile of each score on a no-change experiment; the distance
    # scores have no built-in null distribution, so this calibration is part of the method
    for null_name, fname in (("A_const", "detection_scores"), ("A_vol", "detection_scores_volcal")):
        thresholds = {c: null_threshold(score_dfs[null_name], c, 0.99) for c in SCORES}
        (out / f"score_thresholds_{null_name}.json").write_text(json.dumps(thresholds, indent=1))
        score_tables = {}
        for name, exp in exps.items():
            df = score_dfs[name].drop(columns=[c for c in score_dfs[name] if c.startswith("flag_")])
            flags = {c: df[c] > thresholds[c] for c in SCORES}
            flags["z_s_tau_gt_3"] = df["z_s_tau"] > 3.0  # rule of thumb, no calibration
            score_tables[name] = detection_summary(add_flags(df, flags), exp.change_at, W)
        pd.concat(score_tables, names=["experiment", "score"]).to_csv(out / f"{fname}.csv")

    # 3. accuracy of the fitted vine when the truth is known
    acc = []
    for label, regime in (("gaussian_rho0.5", Regime(1, 0.5)), ("student3_rho0.5", Regime(1, 0.5, 3.0))):
        log.info("accuracy %s", label)
        df = cached(raw / f"accuracy_v2_{label}.parquet", lambda: estimation_accuracy(regime, n_reps=a.reps_acc))
        acc.append(accuracy_summary(df).assign(true_copula=label))
    pd.concat(acc).to_csv(out / "estimation_accuracy.csv", index=False)

    # 4. conditional risk when the truth is a GARCH process: which model forecasts tomorrow's VaR/ES best?
    cond = []
    for label, regime in (("gaussian_rho0.5", Regime(1, 0.5)), ("student3_rho0.5", Regime(1, 0.5, 3.0))):
        log.info("conditional accuracy %s", label)
        df = cached(raw / f"conditional_{label}.parquet", lambda: conditional_accuracy(
            regime, n_origins=10, n_series=8, seed=0))
        cond.append(conditional_accuracy_summary(df).assign(true_copula=label))
    pd.concat(cond).to_csv(out / "conditional_accuracy.csv")
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
