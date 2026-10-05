import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vine_risk import __version__
from vine_risk.cli import TextProgress, build_parser, find_dashboard_app, format_info, main
from vine_risk.synthetic import Regime, simulate_regimes


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A tiny project: a price cache for 3 synthetic assets and a config pointing at it."""
    root = tmp_path_factory.mktemp("proj")
    r = simulate_regimes([Regime(140, 0.2), Regime(120, 0.85)], n_assets=3, seed=4)
    cache = root / "cache"
    cache.mkdir()
    (100 * np.exp(r.cumsum())).to_parquet(cache / "prices_X1_X2_X3_2010-01-01_latest.parquet")
    cfg = root / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "assets": ["X1", "X2", "X3"], "data": {"start": "2010-01-01", "cache_dir": str(cache)},
        "rolling": {"window": 60, "refit_frequency": 5, "truncation_level": 2, "n_jobs": 1, "tail_simulations": 1024},
        "risk": {"simulations": 512, "seed": 3}}))
    return root, cfg


@pytest.fixture(scope="module")
def run_dir(project):
    root, cfg = project
    d = root / "results" / "w60"
    code = main(["run", "--config", str(cfg), "--run-dir", str(d), "--n-perm", "49", "--scan-step", "10", "--risk-step", "3"])
    assert code == 0
    return d


def test_run_command_creates_the_whole_run(run_dir):
    for name in ("checkpoint.jsonl", "dependence_metrics.parquet", "change_scan.parquet", "portfolio_risk.parquet",
                 "var_backtest.csv", "manifest.json"):
        assert (run_dir / name).is_file(), name


def test_run_again_resumes_and_only_a_subset_of_steps_can_be_chosen(project, run_dir, capsys):
    _, cfg = project
    before = (run_dir / "checkpoint.jsonl").read_text()
    assert main(["run", "--config", str(cfg), "--run-dir", str(run_dir), "--steps", "rolling,metrics", "--n-perm", "49"]) == 0
    assert (run_dir / "checkpoint.jsonl").read_text() == before  # nothing refitted or duplicated
    out = capsys.readouterr().out
    assert "rolling" in out and "changes" not in out.split("run folder")[1].split("done")[0]


def test_unknown_step_and_mismatched_settings_give_clean_errors(project, run_dir, capsys):
    _, cfg = project
    assert main(["run", "--config", str(cfg), "--run-dir", str(run_dir), "--steps", "rolling,nope"]) == 2
    assert "Unknown step" in capsys.readouterr().err
    assert main(["run", "--config", str(cfg), "--run-dir", str(run_dir), "--window", "80"]) == 2
    err = capsys.readouterr().err
    assert "different fit settings" in err and "window: stored 60, now 80" in err
    assert main(["info", str(run_dir) + "-missing"]) == 2


def test_info_describes_the_run(run_dir):
    text = format_info(run_dir)
    for needle in ("window 60", "3 assets", f"vine-risk {__version__}", "rolling", "metrics", "changes", "risk", "checkpoint.jsonl"):
        assert needle in text, needle
    assert "not run" not in text and "differences from the current environment: none" in text
    assert "no manifest" in format_info(run_dir.parent)  # a folder that is not a run


def test_main_info_prints(project, run_dir, capsys):
    _, cfg = project
    assert main(["info", "--config", str(cfg), str(run_dir)]) == 0
    assert "fit settings" in capsys.readouterr().out


def test_text_progress_non_tty_prints_every_ten_percent():
    buf = io.StringIO()
    p = TextProgress("fits", stream=buf)
    for i in range(1, 101):
        p(i, 100)
    lines = buf.getvalue().splitlines()
    assert 10 <= len(lines) <= 12 and lines[-1].startswith("fits: 100/100 (100%)")


def test_text_progress_tty_redraws_one_line():
    class Tty(io.StringIO):
        def isatty(self):
            return True
    buf = Tty()
    p = TextProgress("fits", stream=buf, min_interval=0)
    for i in range(1, 11):
        p(i, 10)
    assert buf.getvalue().count("\r") == 10 and buf.getvalue().endswith("\n") and "100%" in buf.getvalue()


def test_parser_and_entry_points():
    p = build_parser()
    assert p.parse_args(["run", "--last", "300", "--steps", "rolling"]).last == 300
    assert p.parse_args(["replay", "--days", "30", "--no-verify"]).no_verify
    with pytest.raises(SystemExit):
        p.parse_args([])  # a command is required
    assert find_dashboard_app().name == "app.py"
    out = subprocess.run([sys.executable, "-m", "vine_risk", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and __version__ in out.stdout
