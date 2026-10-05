import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vine_risk.config import Config, RollingConfig, RiskConfig
from vine_risk.copula import VineCopula
from vine_risk.manifest import (
    ManifestMismatch, build_manifest, check_resume, collect_environment, data_fingerprint, fit_fingerprint,
    read_manifest, record_step, stable_hash, verify_history_unchanged, write_manifest,
)
from vine_risk.marginals import EmpiricalMarginal
from vine_risk.synthetic import Regime, simulate_regimes

CFG = Config(assets=["X1", "X2", "X3"], rolling=RollingConfig(window=60, refit_frequency=5))


@pytest.fixture(scope="module")
def returns():
    return simulate_regimes([Regime(120, 0.5)], n_assets=3, seed=1)


def test_stable_hash_ignores_key_order_and_detects_changes():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_fit_fingerprint_depends_on_fit_settings_only():
    base = fit_fingerprint(CFG)
    assert fit_fingerprint(replace(CFG, risk=replace(CFG.risk, confidence_level=0.95, simulations=2048))) == base  # risk settings
    assert fit_fingerprint(replace(CFG, rolling=replace(CFG.rolling, n_jobs=8))) == base  # parallelism
    for changed in (replace(CFG, rolling=replace(CFG.rolling, window=125)),
                    replace(CFG, rolling=replace(CFG.rolling, truncation_level=2)),
                    replace(CFG, assets=["X1", "X2"]),
                    replace(CFG, risk=replace(CFG.risk, seed=7))):  # the seed drives the tail simulation
        assert fit_fingerprint(changed) != base


def test_data_fingerprint(returns):
    f = data_fingerprint(returns)
    assert f["n_obs"] == 120 and f["first_date"] == str(returns.index[0].date()) and f["assets"] == ["X1", "X2", "X3"]
    changed = returns.copy(); changed.iloc[5, 0] += 1e-9
    assert data_fingerprint(changed)["returns_sha256"] != f["returns_sha256"]
    assert data_fingerprint(returns)["returns_sha256"] == f["returns_sha256"]


def test_environment_records_versions():
    env = collect_environment()
    assert env["python"].count(".") == 2 and "numpy" in env["libraries"] and "pyvinecopulib" in env["libraries"]
    assert env["git"] is None or len(env["git"]["commit"]) == 40


def test_write_read_roundtrip_and_step_records(tmp_path, returns):
    m = build_manifest(CFG, returns, command=["vine-risk", "run"])
    assert m["config_hash"] == stable_hash(m["config"]) and m["command"] == ["vine-risk", "run"]
    write_manifest(tmp_path, m)
    back = read_manifest(tmp_path)
    assert back["fit_hash"] == m["fit_hash"] and back["data"]["n_obs"] == 120 and back["steps"] == {}
    record_step(tmp_path, "rolling", {"windows": 12})
    assert read_manifest(tmp_path)["steps"]["rolling"]["windows"] == 12
    # rewriting keeps the creation time and earlier step records
    created = read_manifest(tmp_path)["created"]
    write_manifest(tmp_path, build_manifest(CFG, returns))
    again = read_manifest(tmp_path)
    assert again["created"] == created and again["steps"]["rolling"]["windows"] == 12
    json.loads((tmp_path / "manifest.json").read_text())
    assert read_manifest(tmp_path / "nowhere") is None
    record_step(tmp_path / "nowhere", "x", {})  # silently does nothing


def test_check_resume(tmp_path, returns):
    assert check_resume(tmp_path, CFG) == []  # nothing stored yet
    write_manifest(tmp_path, build_manifest(CFG, returns))
    assert check_resume(tmp_path, replace(CFG, risk=RiskConfig(confidence_level=0.95))) == []  # irrelevant change
    with pytest.raises(ManifestMismatch, match=r"window: stored 60, now 125") as e:
        check_resume(tmp_path, replace(CFG, rolling=replace(CFG.rolling, window=125)))
    assert "different run folder" in str(e.value)


def test_verify_history_detects_revised_prices(returns):
    u = EmpiricalMarginal().fit_transform(returns.iloc[:60])
    res = VineCopula(families=["indep", "gaussian"], truncation_level=1, tail_simulations=0).fit(u, returns.iloc[:60]).summary()
    verify_history_unchanged(res, returns)  # same data: fine
    revised = returns.copy()
    revised.iloc[10:20, 0] = revised.iloc[10:20, 0].to_numpy()[::-1]  # reorder some days -> different ranks
    with pytest.raises(ManifestMismatch, match="changed"):
        verify_history_unchanged(res, revised)
    with pytest.raises(ManifestMismatch, match="no matching"):
        verify_history_unchanged(res, returns.iloc[:50])
