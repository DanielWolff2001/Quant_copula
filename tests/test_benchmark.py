from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vine_risk.backtesting import evaluate_forecasts
from vine_risk.benchmark import (
    MODELS, BenchmarkSpec, evaluate, forecast_origin, format_report, origin_dates, realized_losses, resolve_portfolios,
    run_benchmark,
)
from vine_risk.cli import main
from vine_risk.config import Config, RiskConfig, RollingConfig
from vine_risk.portfolio import window_risk
from vine_risk.rolling import CheckpointIndex
from vine_risk.runner import fit_rolling
from vine_risk.synthetic import Regime, simulate_garch

W, LB = 60, 150
SMALL = dict(truncation_level=2, n_jobs=1, tail_simulations=2 ** 9)
EMP = Config(assets=["X1", "X2", "X3"], rolling=RollingConfig(window=W, refit_frequency=5, **SMALL),
             risk=RiskConfig(simulations=2 ** 9, seed=3, portfolios={"equal": None, "first": {"X1": 1}, "mix": {"X1": 0.7, "X3": 0.3}}))
GAR = replace(EMP, rolling=replace(EMP.rolling, marginal="garch_t", marginal_lookback=LB))
SPEC = BenchmarkSpec(resolve_portfolios(EMP), alphas=(0.9, 0.95), window=W, ewma_lookback=120, garch_lookback=LB,
                     n_sims=2 ** 9, seed=3)  # 99 % would be capped by the sample maximum of a 60-day window


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    r, _ = simulate_garch([Regime(180, 0.2), Regime(160, 0.7)], n_assets=3, seed=31)
    fit_rolling(EMP, r, root / "emp")
    fit_rolling(GAR, r, root / "garch")
    return dict(root=root, returns=r, emp=CheckpointIndex(root / "emp" / "checkpoint.jsonl"),
                gar=CheckpointIndex(root / "garch" / "checkpoint.jsonl"))


def test_resolve_portfolios():
    pf = resolve_portfolios(EMP)
    assert set(pf) == {"equal", "first", "mix"} and pf["equal"].tolist() == pytest.approx([1 / 3] * 3)
    assert pf["first"].tolist() == [1, 0, 0] and pf["mix"].tolist() == pytest.approx([0.7, 0.0, 0.3])
    assert list(resolve_portfolios(replace(EMP, risk=RiskConfig()))) == ["equal"]
    for bad in ({"x": {"ZZZ": 1}}, {"x": {"X1": -1, "X2": 2}}, {"x": {"X1": 0}}):
        with pytest.raises(ValueError):
            resolve_portfolios(replace(EMP, risk=RiskConfig(portfolios=bad)))


def test_forecast_origin_has_every_model_and_is_consistent_with_window_risk(world):
    r, d = world["returns"], world["gar"].timestamps[3]
    end = r.index.get_loc(d)
    hist = r.iloc[end - LB + 1: end + 1]
    emp, gar = world["emp"].get(d), world["gar"].get(d)
    fc = forecast_origin(SPEC, hist, emp, gar)
    assert {f.model for f in fc} == set(MODELS) and len(fc) == len(MODELS) * 3 * 2
    assert all(0 < f.var < f.es and len(f.tail) == 20 for f in fc)
    ref = window_risk(emp, hist.iloc[-W:], SPEC.portfolios["equal"], 0.95, SPEC.n_sims, SPEC.seed)
    f = next(f for f in fc if f.model == "vine_emp" and f.portfolio == "equal" and f.alpha == 0.95)
    assert (f.var, f.es) == (ref["var_vine"], ref["es_vine"])
    again = forecast_origin(SPEC, hist, emp, gar)
    assert all(a.var == b.var and a.es == b.es for a, b in zip(fc, again))  # seeded
    assert {f.model for f in forecast_origin(SPEC, hist, emp, None)} == {"hist", "ewma_n", "ewma_t", "fhs", "vine_emp", "gauss_emp", "indep_emp"}
    assert {f.model for f in forecast_origin(SPEC, hist)} == {"hist", "ewma_n", "ewma_t", "fhs"}


def test_origin_dates_and_realized_losses(world):
    r = world["returns"]
    o = origin_dates(world["emp"], world["gar"], r, SPEC)
    assert o[0] == world["gar"].timestamps[0] and o[-1] < r.index[-1] and set(o) <= set(world["emp"].timestamps)
    assert len(origin_dates(world["emp"], world["gar"], r, SPEC, step=2)) == (len(o) + 1) // 2
    assert origin_dates(world["emp"], world["gar"], r, SPEC, first=str(o[5].date()))[0] == o[5]
    shorter = replace(SPEC, garch_lookback=60, ewma_lookback=60)  # a smaller history requirement allows earlier origins
    assert origin_dates(world["emp"], None, r, shorter)[0] < o[0]
    real = realized_losses(r, SPEC.portfolios)
    d = o[2]
    nxt = r.index[r.index.get_loc(d) + 1]
    assert real.loc[d, "first"] == pytest.approx(-r.loc[nxt, "X1"])
    assert real.loc[d, "mix"] == pytest.approx(-(0.7 * r.loc[nxt, "X1"] + 0.3 * r.loc[nxt, "X3"]))
    assert real.index[-1] == r.index[-2]


@pytest.fixture(scope="module")
def table(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    t = run_benchmark(EMP, world["returns"], out, emp_run=world["root"] / "emp", garch_run=world["root"] / "garch", spec=SPEC)
    return out, t


def test_benchmark_table_shape_and_files(world, table):
    out, t = table
    n = len(origin_dates(world["emp"], world["gar"], world["returns"], SPEC))
    assert len(t) == n * len(MODELS) * 3 * 2 and set(t["model"]) == set(MODELS)
    assert (out / "realized.parquet").is_file() and (out / "benchmark.json").is_file()
    assert t.groupby(["date", "portfolio", "alpha"]).size().eq(len(MODELS)).all()


def test_extending_equals_a_full_run(world, table, tmp_path):
    _, full = table
    r = world["returns"]
    part = run_benchmark(EMP, r.iloc[:300], tmp_path, emp_run=world["root"] / "emp", garch_run=world["root"] / "garch", spec=SPEC)
    assert part["date"].nunique() < full["date"].nunique()
    ext = run_benchmark(EMP, r, tmp_path, emp_run=world["root"] / "emp", garch_run=world["root"] / "garch", spec=SPEC)
    key = ["date", "portfolio", "alpha", "model"]
    a, b = ext.sort_values(key).reset_index(drop=True), full.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns="tail"), b.drop(columns="tail"))
    assert np.allclose(np.vstack(a["tail"]), np.vstack(b["tail"]))
    import json
    assert json.loads((tmp_path / "benchmark.json").read_text())["new_origins"] == full["date"].nunique() - part["date"].nunique()
    changed = run_benchmark(EMP, r, tmp_path, emp_run=world["root"] / "emp", garch_run=world["root"] / "garch",
                            spec=replace(SPEC, n_sims=2 ** 8))  # other settings: everything is recomputed
    assert json.loads((tmp_path / "benchmark.json").read_text())["new_origins"] == changed["date"].nunique()


def test_parallel_matches_serial(world, table, tmp_path):
    _, serial = table
    par = run_benchmark(EMP, world["returns"], tmp_path, emp_run=world["root"] / "emp", garch_run=world["root"] / "garch",
                        spec=SPEC, n_jobs=2)
    key = ["date", "portfolio", "alpha", "model"]
    pd.testing.assert_frame_equal(par.sort_values(key).reset_index(drop=True).drop(columns="tail"),
                                  serial.sort_values(key).reset_index(drop=True).drop(columns="tail"))


def test_evaluate_uses_common_dates_and_writes_the_report(world, table):
    out, t = table
    rep = evaluate(out, reference="vine_garch", n_sim=200)
    assert (out / "report.csv").is_file()
    assert set(rep.index.get_level_values("model")) == set(MODELS)
    assert {"rate", "kupiec_p", "cc_p", "z1_p", "z2_p", "fz0", "dm_p", "dm_stat"} <= set(rep.columns)
    assert (rep["n"] == t["date"].nunique()).all()
    direct = evaluate_forecasts(t, pd.read_parquet(out / "realized.parquet"), "vine_garch", None, 200, 0)
    pd.testing.assert_frame_equal(rep, direct)
    assert evaluate(out, reference="no_such_model", n_sim=100).columns.tolist().count("dm_p") == 0
    text = format_report(rep)
    assert "portfolio equal, 95.0% level" in text and "vine_garch" in text


def test_benchmark_command(world, tmp_path, capsys):
    import yaml
    csv = tmp_path / "prices.csv"
    (100 * np.exp(world["returns"].cumsum())).to_csv(csv)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "assets": ["X1", "X2", "X3"], "data": {"start": "2010-01-01", "source": "csv", "csv_path": str(csv)},
        "rolling": {"window": W, "refit_frequency": 5, "truncation_level": 2, "n_jobs": 1, "tail_simulations": 512},
        "risk": {"simulations": 512, "seed": 3, "portfolios": {"equal": None}}}))
    out = tmp_path / "bench"
    assert main(["benchmark", "--config", str(cfg), "--run-dir", str(world["root"] / "emp"), "--garch-run-dir",
                 str(world["root"] / "garch"), "--out", str(out), "--alphas", "0.95", "--step", "3", "--n-sim", "100", "--hist-window", "60",
                 "--ewma-lookback", "120"]) == 0
    text = capsys.readouterr().out
    assert "portfolio equal, 95.0% level" in text and "vine_garch" in text and (out / "report.csv").is_file()
    assert main(["benchmark", "--config", str(cfg), "--run-dir", str(world["root"] / "emp"), "--garch-run-dir",
                 str(tmp_path / "missing"), "--out", str(out)]) == 2


def test_clear_error_when_there_is_too_little_history(world, tmp_path):
    with pytest.raises(ValueError, match="No forecast dates"):
        run_benchmark(EMP, world["returns"], tmp_path, emp_run=world["root"] / "emp", spec=replace(SPEC, ewma_lookback=5000))
