import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vine_risk.change_detection import change_scan
from vine_risk.config import Config, RiskConfig, RollingConfig
from vine_risk.manifest import ManifestMismatch, read_manifest
from vine_risk.portfolio import rolling_risk
from vine_risk.rolling import CheckpointIndex, RollingVineModel, load_results
from vine_risk.runner import (
    STEPS, add_fdr_flags, compute_metrics, compute_risk, default_run_dir, detect_changes, fit_rolling, run_pipeline,
    with_rolling,
)
from vine_risk.synthetic import Regime, simulate_regimes

CFG = Config(assets=["X1", "X2", "X3"],
             rolling=RollingConfig(window=60, refit_frequency=5, truncation_level=2, n_jobs=1, tail_simulations=2 ** 10),
             risk=RiskConfig(confidence_level=0.99, simulations=2 ** 9, seed=3))


@pytest.fixture(scope="module")
def returns():
    return simulate_regimes([Regime(150, 0.2), Regime(130, 0.85)], n_assets=3, seed=9)


def test_default_run_dir_and_overrides():
    assert default_run_dir(CFG).as_posix() == "data/results/w60"
    c = with_rolling(CFG, window=125, n_jobs=None)
    assert c.rolling.window == 125 and c.rolling.n_jobs == 1 and with_rolling(CFG) is CFG


def test_checkpoint_index_latest_and_progress(tmp_path, returns):
    ticks = []
    fit_rolling(CFG, returns.iloc[:120], tmp_path, progress=lambda i, n: ticks.append((i, n)))
    assert ticks[0][0] == 1 and ticks[-1][0] == ticks[-1][1] == len(ticks)  # 1..N of N
    idx = CheckpointIndex(tmp_path / "checkpoint.jsonl")
    assert idx.latest().timestamp == str(returns.index[119].date()) and len(idx) == len(ticks)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory, returns):
    d = tmp_path_factory.mktemp("run") / "w60"
    timings = run_pipeline(CFG, returns, d, scan_step=10, n_perm=49, risk_step=3, command=["test"])
    return d, timings


def test_pipeline_writes_every_artifact_and_a_manifest(full_run):
    d, timings = full_run
    assert list(timings) == list(STEPS)
    for name in ("checkpoint.jsonl", "fits.parquet", "dependence_metrics.parquet", "pairwise_tau.parquet",
                 "structural_change_scores.parquet", "change_scan.parquet", "portfolio_risk.parquet",
                 "var_backtest.csv", "manifest.json"):
        assert (d / name).is_file(), name
    m = read_manifest(d)
    assert set(m["steps"]) == set(STEPS) and m["command"] == ["test"] and m["steps"]["rolling"]["failed"] == 0
    assert m["fit_hash"] and m["data"]["n_obs"] == 280


def test_incremental_extension_equals_a_full_run(full_run, returns, tmp_path):
    full_dir, _ = full_run
    part = tmp_path / "w60"
    cut = 220
    run_pipeline(CFG, returns.iloc[:cut], part, scan_step=10, n_perm=49, risk_step=3)
    first_scan = pd.read_parquet(part / "change_scan.parquet")
    run_pipeline(CFG, returns, part, scan_step=10, n_perm=49, risk_step=3)  # extend with the remaining days
    m = read_manifest(part)["steps"]
    assert m["changes"]["new_scan_dates"] == len(pd.read_parquet(part / "change_scan.parquet")) - len(first_scan)
    assert 0 < m["risk"]["new_dates"] < m["risk"]["dates"]
    for name in ("change_scan.parquet", "portfolio_risk.parquet", "dependence_metrics.parquet",
                 "structural_change_scores.parquet"):
        a, b = pd.read_parquet(part / name), pd.read_parquet(full_dir / name)
        if a.index.name != "timestamp" and "timestamp" in a:
            a, b = a.set_index("timestamp"), b.set_index("timestamp")
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, check_freq=False, obj=name)
    assert [r.to_json() for r in load_results(part / "checkpoint.jsonl")] == \
           [r.to_json() for r in load_results(full_dir / "checkpoint.jsonl")]


def test_changed_scan_settings_trigger_a_full_recompute(full_run, returns, tmp_path):
    d, _ = full_run
    work = tmp_path / "copy"
    import shutil
    shutil.copytree(d, work)
    detect_changes(CFG, returns, work, step=20, n_perm=49)  # different step
    m = read_manifest(work)["steps"]["changes"]
    assert m["new_scan_dates"] == m["scan_dates"]  # nothing reused


def test_resume_is_refused_with_different_fit_settings_or_revised_data(full_run, returns, tmp_path):
    d, _ = full_run
    import shutil
    work = tmp_path / "w"
    shutil.copytree(d, work)
    with pytest.raises(ManifestMismatch, match="window"):
        fit_rolling(with_rolling(CFG, window=80), returns, work)
    revised = returns.copy()
    revised.iloc[200:230, 0] = revised.iloc[200:230, 0].to_numpy()[::-1]
    with pytest.raises(ManifestMismatch, match="changed"):
        fit_rolling(CFG, revised, work)
    fit_rolling(with_rolling(CFG, window=80), returns, tmp_path / "other")  # a new folder is fine


def test_scan_and_risk_after_option_matches_full_computation(returns):
    full = change_scan(returns, 60, 10, 49, 0.1, 3, 10)
    cut = full.index[4]
    tail = change_scan(returns, 60, 10, 49, 0.1, 3, 10, after=cut)
    pd.testing.assert_frame_equal(full.loc[full.index > cut], tail)
    res = RollingVineModel.from_config(CFG).run(returns)
    all_risk = rolling_risk(res, returns, n_sims=2 ** 9, seed=3, step=3)
    part = rolling_risk(res, returns, n_sims=2 ** 9, seed=3, step=3, after=all_risk.index[5])
    pd.testing.assert_frame_equal(all_risk.iloc[6:], part)
    assert rolling_risk(res, returns, n_sims=2 ** 9, step=3, after=returns.index[-1]).empty


def test_add_fdr_flags_and_unknown_steps(returns):
    scan = pd.DataFrame({"p_tau": [0.001, 0.5, 0.002], "stat_tau": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
    out = add_fdr_flags(scan, 0.05)
    assert "p_tau_fdr0.05" in out and out["p_tau_fdr0.05"].tolist() == [True, False, True]
    assert list(add_fdr_flags(out, 0.05).columns) == list(out.columns)  # idempotent
    with pytest.raises(ValueError, match="Unknown step"):
        run_pipeline(CFG, returns, "x", steps=["rolling", "nope"])


def test_risk_weights_must_match_assets(returns, tmp_path):
    bad = replace(CFG, risk=replace(CFG.risk, weights=[0.5, 0.5]))
    fit_rolling(CFG, returns.iloc[:130], tmp_path)
    with pytest.raises(ValueError, match="weights"):
        compute_risk(bad, returns.iloc[:130], tmp_path)


def test_scan_is_skipped_gracefully_on_short_history(returns, tmp_path):
    short = returns.iloc[:110]  # fewer than two windows (120)
    run_pipeline(CFG, short, tmp_path, steps=["rolling", "changes"], scan_step=10, n_perm=49)
    assert (tmp_path / "structural_change_scores.parquet").is_file() and not (tmp_path / "change_scan.parquet").exists()
    assert "skipped" in read_manifest(tmp_path)["steps"]["changes"]


def test_filtered_scan_option_writes_and_extends_a_second_scan(tmp_path):
    from dataclasses import replace as rep_
    from vine_risk.synthetic import simulate_garch
    # data with real volatility clustering: GARCH fits are then well determined and reproducible
    g, _ = simulate_garch([Regime(160, 0.2), Regime(160, 0.85)], n_assets=3, seed=9)
    cfg = rep_(CFG, rolling=rep_(CFG.rolling, window=80))
    run_pipeline(cfg, g.iloc[:260], tmp_path, steps=["rolling", "changes"], scan_step=20, n_perm=29, filtered_scan=True)
    first = pd.read_parquet(tmp_path / "change_scan_garch.parquet")
    assert len(first) >= 1
    run_pipeline(cfg, g, tmp_path, steps=["rolling", "changes"], scan_step=20, n_perm=29, filtered_scan=True)
    m = read_manifest(tmp_path)["steps"]["changes"]
    both = pd.read_parquet(tmp_path / "change_scan_garch.parquet")
    assert len(both) > len(first) and m["filtered_new_dates"] == len(both) - len(first)
    full = tmp_path.parent / "full_filtered"
    run_pipeline(cfg, g, full, steps=["rolling", "changes"], scan_step=20, n_perm=29, filtered_scan=True)
    pd.testing.assert_frame_equal(pd.read_parquet(full / "change_scan_garch.parquet"), both)  # extending == all at once
    assert "p_tau_fdr0.01" in both.columns
