import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vine_risk.cli import main
from vine_risk.locking import RunLockError, run_lock
from vine_risk.manifest import ManifestMismatch
from vine_risk.rolling import load_results
from vine_risk.synthetic import Regime, simulate_regimes
from vine_risk.update import update_run
from vine_risk.config import load_config

W, N_ALL, N_FIRST = 60, 340, 180
RUN_ARGS = ["--n-perm", "99", "--scan-step", "10", "--risk-step", "3"]
UPD_ARGS = RUN_ARGS + ["--alpha", "0.05"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    r = simulate_regimes([Regime(150, 0.2), Regime(190, 0.85)], n_assets=3, seed=21)
    prices = 100 * np.exp(r.cumsum())
    csv = root / "prices.csv"
    cfg = root / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "assets": ["X1", "X2", "X3"], "data": {"start": "2010-01-01", "source": "csv", "csv_path": str(csv)},
        "rolling": {"window": W, "refit_frequency": 5, "truncation_level": 2, "n_jobs": 1, "tail_simulations": 1024},
        "risk": {"simulations": 512, "seed": 3}}))
    return dict(root=root, prices=prices, csv=csv, cfg=str(cfg))


def _digest(run: Path):
    out = {}
    for name in ("dependence_metrics.parquet", "change_scan.parquet", "portfolio_risk.parquet",
                 "structural_change_scores.parquet"):
        df = pd.read_parquet(run / name)
        out[name] = df.round(10).to_csv()
    out["fits"] = [r.to_json() for r in load_results(run / "checkpoint.jsonl")]
    return out


@pytest.fixture(scope="module")
def updated(world):
    """An initial run on the first 180 days, then a daily update with the whole history."""
    w = world
    w["prices"].iloc[:N_FIRST].to_csv(w["csv"])
    run = w["root"] / "inc" / "w60"
    assert main(["run", "--config", w["cfg"], "--run-dir", str(run), *RUN_ARGS]) == 0
    w["prices"].to_csv(w["csv"])  # "new data arrives"
    report = update_run(load_config(w["cfg"]), run, threads=1, scan_step=10, n_perm=99, risk_step=3,
                        monitor_overrides=dict(alpha=0.05), out=lambda s: None)
    return run, report


def test_update_equals_one_full_run(world, updated):
    run, report = updated
    full = world["root"] / "full" / "w60"
    assert main(["run", "--config", world["cfg"], "--run-dir", str(full), *RUN_ARGS]) == 0
    a, b = _digest(run), _digest(full)
    assert a["fits"] == b["fits"], "fits made by the live update differ from the batch fits"
    for k in a:
        assert a[k] == b[k], k


def test_update_report_log_and_alert_files(world, updated):
    run, report = updated
    assert report.status == "updated" and len(report.new_days) > 100 and report.fits_added > 20
    assert report.last_observation == str(world["prices"].index[-1].date())
    status = json.loads((run / "live" / "last_update.json").read_text())
    assert status["status"] == "updated" and status["alerts"] == report.alerts
    # the known correlation jump (day 150) raises an alert during the update, never before it
    assert len(report.alerts) >= 1 and pd.Timestamp(report.alerts[0]["date"]) > world["prices"].index[150]
    alerts = pd.read_csv(run / "live" / "alerts.csv")
    assert list(alerts.columns) == ["date", "message"] and len(alerts) == len(report.alerts)
    lines = [json.loads(l) for l in (run / "live" / "updates.jsonl").read_text().splitlines()]
    assert len(lines) == len(report.new_days) and sum(l["new_alert"] for l in lines) == len(report.alerts)
    assert json.loads((run / "manifest.json").read_text())["steps"]["rolling"]["fits"] == len(load_results(run / "checkpoint.jsonl"))


def test_second_update_is_a_no_op(world, updated):
    run, _ = updated
    before = _digest(run)
    r2 = update_run(load_config(world["cfg"]), run, threads=1, scan_step=10, n_perm=99, risk_step=3, out=lambda s: None)
    assert r2.status == "up_to_date" and r2.new_days == [] and r2.fits_added == 0
    assert _digest(run) == before


def _tree_hash(run: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(run.rglob("*")):
        if p.is_file() and p.name != "last_update.json":
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def test_dry_run_changes_nothing_and_reports_the_work(world, tmp_path):
    w = world
    w["prices"].iloc[:N_FIRST].to_csv(w["csv"])
    run = tmp_path / "w60"
    assert main(["run", "--config", w["cfg"], "--run-dir", str(run), *RUN_ARGS]) == 0
    w["prices"].to_csv(w["csv"])
    before, lines = _tree_hash(run), []
    r = update_run(load_config(w["cfg"]), run, dry_run=True, out=lines.append)
    assert r.status == "dry_run" and len(r.new_days) > 100 and _tree_hash(run) == before
    assert any("would process" in l for l in lines)


def test_update_refuses_revised_history_and_changed_settings(world, tmp_path):
    w = world
    w["prices"].iloc[:N_FIRST].to_csv(w["csv"])
    run = tmp_path / "w60"
    assert main(["run", "--config", w["cfg"], "--run-dir", str(run), *RUN_ARGS]) == 0
    before = _tree_hash(run)
    revised = w["prices"].copy()
    revised.iloc[100:130, 0] = revised.iloc[100:130, 0].to_numpy()[::-1]  # the vendor "corrected" old prices
    revised.to_csv(w["csv"])
    with pytest.raises(ManifestMismatch, match="changed"):
        update_run(load_config(w["cfg"]), run, out=lambda s: None)
    assert main(["update", "--config", w["cfg"], "--run-dir", str(run)]) == 2
    other = yaml.safe_load(Path(w["cfg"]).read_text())
    other["rolling"]["window"] = 80
    cfg2 = tmp_path / "c2.yaml"; cfg2.write_text(yaml.safe_dump(other))
    w["prices"].to_csv(w["csv"])
    with pytest.raises(ManifestMismatch, match="window"):
        update_run(load_config(str(cfg2)), run, out=lambda s: None)
    assert _tree_hash(run) == before  # nothing was changed by the refused updates


def test_update_rejects_invalid_new_prices(world, tmp_path):
    w = world
    w["prices"].iloc[:N_FIRST].to_csv(w["csv"])
    run = tmp_path / "w60"
    assert main(["run", "--config", w["cfg"], "--run-dir", str(run), *RUN_ARGS]) == 0
    bad = w["prices"].copy(); bad.iloc[-1, 1] = -5.0
    bad.to_csv(w["csv"])
    before = _tree_hash(run)
    with pytest.raises(ValueError, match="failed validation"):
        update_run(load_config(w["cfg"]), run, out=lambda s: None)
    assert _tree_hash(run) == before
    with pytest.raises(FileNotFoundError, match="vine-risk run"):
        update_run(load_config(w["cfg"]), tmp_path / "nothing", out=lambda s: None)


def test_update_waits_for_a_running_job(world, updated):
    run, _ = updated
    import subprocess, sys
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (run / ".lock").write_text(str(other.pid))
        with pytest.raises(RunLockError):
            update_run(load_config(world["cfg"]), run, out=lambda s: None)
        assert main(["update", "--config", world["cfg"], "--run-dir", str(run)]) == 2
    finally:
        other.kill(); other.wait(); (run / ".lock").unlink(missing_ok=True)


def test_days_after_the_last_fit_are_not_reprocessed(world, tmp_path):
    w = world
    cut = 183  # 3 days after the last fit (position 174): seen, but not yet refit
    w["prices"].iloc[:cut].to_csv(w["csv"])
    run = tmp_path / "w60"
    assert main(["run", "--config", w["cfg"], "--run-dir", str(run), *RUN_ARGS]) == 0
    last_fit = load_results(run / "checkpoint.jsonl")[-1].timestamp
    assert pd.Timestamp(last_fit) < w["prices"].index[cut - 2]
    # nothing new arrived: the unfit days after the last fit are already accounted for
    r = update_run(load_config(w["cfg"]), run, threads=1, scan_step=10, n_perm=99, risk_step=3, out=lambda s: None)
    assert r.status == "up_to_date" and r.new_days == []
    w["prices"].iloc[:cut + 12].to_csv(w["csv"])
    r = update_run(load_config(w["cfg"]), run, threads=1, scan_step=10, n_perm=99, risk_step=3, out=lambda s: None)
    assert len(r.new_days) == 12 and r.new_days[0] == str(w["prices"].index[cut].date())  # only the genuinely new days
    w["prices"].to_csv(w["csv"])
