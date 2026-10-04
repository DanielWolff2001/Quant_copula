from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vine_risk import dashboard_figures as figs
from vine_risk.change_detection import benjamini_hochberg, change_scan, default_lag, structural_change_scores
from vine_risk.copula import PairCopulaInfo, VineFitResult
from vine_risk.dashboard_data import (
    CheckpointIndex, alert_periods, load_run, nearest_fit_date, pair_column, prices_from_returns,
    read_assets, tau_matrix_at,
)
from vine_risk.dependence import dependence_metrics, pairwise_series
from vine_risk.monitor import LiveMonitor, MonitorConfig, records_to_frame
from vine_risk.portfolio import backtest_var, rolling_risk
from vine_risk.rolling import RollingVineModel
from vine_risk.synthetic import Regime, simulate_regimes
from vine_risk.visualization import vine_tree_graph


# ---- pure data helpers ---------------------------------------------------------
def test_alert_periods_finds_maximal_runs():
    s = pd.Series([0, 3, 4, 0, 0, 5, 0, 6, 7], index=pd.date_range("2020-01-01", periods=9), dtype=float)
    p = alert_periods(s, 2.0)
    assert [(a.day, b.day) for a, b in p] == [(2, 3), (6, 6), (8, 9)]
    assert alert_periods(s, 10) == []
    assert alert_periods(s.where(s > 2), 2.0)[0][0].day == 2  # NaN never counts as an alert


def test_nearest_fit_date_and_pair_column_and_prices():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-10", "2020-01-17"])
    assert nearest_fit_date(idx, pd.Timestamp("2020-01-12")) == pd.Timestamp("2020-01-10")
    assert nearest_fit_date(idx, pd.Timestamp("2020-01-10")) == pd.Timestamp("2020-01-10")
    with pytest.raises(ValueError):
        nearest_fit_date(idx, pd.Timestamp("2020-01-01"))
    f = pd.DataFrame(columns=["A-B", "A-C", "B-C"])
    assert pair_column(f, "C", "A", ["A", "B", "C"]) == "A-C"
    with pytest.raises(ValueError):
        pair_column(f, "A", "A", ["A", "B", "C"])
    r = pd.DataFrame({"x": [np.log(2), np.log(1.5)]})
    assert prices_from_returns(r)["x"].tolist() == pytest.approx([2.0, 3.0])


def test_tau_matrix_at_is_symmetric_and_uses_fit_in_force():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-10"])
    tau = pd.DataFrame({"A-B": [0.1, 0.5], "A-C": [0.2, 0.6], "B-C": [0.3, 0.7]}, index=idx)
    m = tau_matrix_at(tau, pd.Timestamp("2020-01-12"), ["A", "B", "C"])
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1)
    assert m.loc["A", "C"] == 0.6 and m.loc["C", "B"] == 0.7


def _result(ts, assets=("A", "B", "C")):
    pcs = [PairCopulaInfo(1, 1, ("A", "B"), (), "gaussian", 0, [0.5], 0.33, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
           PairCopulaInfo(1, 2, ("B", "C"), (), "clayton", 0, [2.0], 0.5, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0),
           PairCopulaInfo(2, 1, ("A", "C"), ("B",), "frank", 0, [1.0], 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
    return VineFitResult(assets=list(assets), n_obs=100, n_assets=3, timestamp=ts, window_end=ts, pair_copulas=pcs,
                         order=[3, 2, 1])


def test_checkpoint_index_random_access(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    stamps = ["2020-01-03", "2020-01-10", "2020-01-17"]
    path.write_text("".join(_result(t).to_json() + "\n" for t in stamps))
    idx = CheckpointIndex(path)
    assert list(idx.timestamps.strftime("%Y-%m-%d")) == stamps
    assert idx.get("2020-01-10").timestamp == "2020-01-10"
    assert idx.get(pd.Timestamp("2020-01-17")).to_json() == _result("2020-01-17").to_json()  # NaN != NaN
    with pytest.raises(KeyError):
        idx.get("2020-02-01")
    assert read_assets(path) == ["A", "B", "C"]


def test_load_run_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_rolling"):
        load_run(tmp_path / "nope")
    (tmp_path / "checkpoint.jsonl").write_text(_result("2020-01-03").to_json() + "\n")
    with pytest.raises(FileNotFoundError, match="compute_metrics"):
        load_run(tmp_path)


def test_vine_tree_graph():
    nodes, edges = vine_tree_graph(_result("2020-01-03"), 1)
    assert nodes == ["A", "B", "C"] and [(u, v) for u, v, _ in edges] == [("A", "B"), ("B", "C")]
    nodes2, edges2 = vine_tree_graph(_result("2020-01-03"), 2)
    assert nodes2 == ["A,B", "B,C"] and [(u, v) for u, v, _ in edges2] == [("A,B", "B,C")]
    with pytest.raises(ValueError):
        vine_tree_graph(_result("2020-01-03"), 3)


# ---- a small but complete run folder ---------------------------------------------
@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    run = root / "w80"
    run.mkdir()
    r = simulate_regimes([Regime(220, 0.2), Regime(180, 0.8)], n_assets=3, seed=5)
    kw = dict(families=["indep", "gaussian", "clayton"], truncation_level=2, tail_simulations=2 ** 10)
    res = RollingVineModel(80, 5, kw).run(r, checkpoint=run / "checkpoint.jsonl")
    dependence_metrics(res).to_parquet(run / "dependence_metrics.parquet")
    for col in ("tau", "lower_tail_q", "upper_tail_q"):
        pairwise_series(res, col).to_parquet(run / f"pairwise_{col}.parquet")
    structural_change_scores(res, lag=default_lag(80, 5), baseline=20).to_parquet(run / "structural_change_scores.parquet")
    scan = change_scan(r, 80, step=10, n_perm=49, block=10)
    scan["p_tau_fdr0.01"] = benjamini_hochberg(scan["p_tau"], 0.01)
    scan.to_parquet(run / "change_scan.parquet")
    risk = rolling_risk(res, r, n_sims=2 ** 9, step=2)
    risk.to_parquet(run / "portfolio_risk.parquet")
    backtest_var(risk, r).to_csv(run / "var_backtest.csv")
    # a short live replay, primed with the stored fits (as scripts/run_live.py does)
    mcfg = MonitorConfig(window=80, refit_frequency=5, vine_kwargs=kw, scan_step=10, n_perm=49, block=10,
                         alpha=0.05, risk_every_fits=0)
    mon = LiveMonitor(list(r.columns), mcfg)
    cut = 340
    mon.prime(r.iloc[:cut], [x for x in res if pd.Timestamp(x.timestamp) <= r.index[cut - 1]])
    for ts, row in r.iloc[cut:].iterrows():
        mon.on_return(ts, row)
    (run / "live").mkdir()
    records_to_frame(mon.records).to_parquet(run / "live" / "live_log.parquet")
    cache = root / "cache"
    cache.mkdir()
    prices = 100 * np.exp(r.cumsum())
    prices.to_parquet(cache / "prices_X1_X2_X3_2010-01-01_latest.parquet")
    cfg = root / "config.yaml"
    cfg.write_text(yaml.safe_dump({"assets": ["X1", "X2", "X3"],
                                   "data": {"start": "2010-01-01", "cache_dir": str(cache)},
                                   "rolling": {"window": 80, "refit_frequency": 5}}))
    return dict(run=run, config=cfg, returns=r, results=res)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_figure_builds_in_both_themes(small_run, theme):
    d = load_run(small_run["run"], returns=small_run["returns"])
    m = d.metrics
    assert len(figs.dependence_figure(m, theme).data) == 1 and len(figs.dependence_figure(m, theme, True).data) == 3
    mat = tau_matrix_at(d.tau, m.index[-1], d.assets)
    hm = figs.tau_heatmap(mat, m.index[-1], theme)
    assert hm.data[0].zmin == -hm.data[0].zmax and np.allclose(hm.data[0].z, mat.to_numpy())
    assert len(figs.pair_figure(d.tau["X1-X2"], "X1-X2", theme).data) == 1
    assert len(figs.tail_figure(d.lower["X1-X2"], d.upper["X1-X2"], "X1-X2", theme).data) == 2
    for col in (["X1"], ["X1", "X2", "X3"]):
        fig = figs.prices_figure(prices_from_returns(small_run["returns"][col]), theme, "start = 1")
        assert len(fig.data) == len(col)
    four = prices_from_returns(pd.concat([small_run["returns"]] * 2, axis=1).set_axis(list("ABCDEF"), axis=1))
    assert len(figs.prices_figure(four, theme, "x").data) == 6  # small multiples beyond three assets
    s = d.scores["structural_change_score"].dropna()
    thr = float(s.quantile(0.8))
    fig = figs.change_score_figure(s, thr, "S_t", theme)
    assert len(fig.layout.shapes) == len(alert_periods(s, thr)) + 1  # one band per period + the threshold line
    risk = figs.risk_figure(d.risk, "es", theme)
    assert [t.name for t in risk.data] == ["vine copula", "Gaussian copula", "independence", "historical"]
    assert np.allclose(risk.data[0].y, 100 * d.risk["es_vine"])
    with pytest.raises(ValueError):
        figs.risk_figure(d.risk, "sharpe", theme)
    fit = small_run["results"][-1]
    for tree in (1, 2):
        assert len(figs.vine_tree_figure(fit, tree, theme).data) == 3
    live = figs.live_figure(d.live, theme)
    assert d.live is not None and len(live.data) >= 3  # ratio line + two threshold legend entries (+ alerts)
    assert len(live.layout.shapes) >= 2  # the two threshold lines (+ shaded alert periods)


def test_log_axis_labels_use_log_units():
    idx = pd.date_range("2020-01-01", periods=50)
    frame = pd.DataFrame({"big": np.linspace(1, 60, 50), "small": np.linspace(1, 3, 50)}, index=idx)
    fig = figs.prices_figure(frame, "light", "start = 1", log=True)
    assert fig.layout.yaxis.type == "log"
    ys = {a.text: a.y for a in fig.layout.annotations}
    assert 1.5 < ys["big"] < 2.1 and 0.3 < ys["small"] < 0.8  # log10(60) = 1.78, log10(3) = 0.48
    lin = figs.prices_figure(frame, "light", "start = 1")
    assert {a.text: a.y for a in lin.layout.annotations}["big"] == pytest.approx(60, abs=6)


def test_end_labels_do_not_overlap():
    idx = pd.date_range("2020-01-01", periods=20)
    risk = pd.DataFrame({f"{m}_{k}": np.linspace(0.01, 0.05, 20) + 0.0005 * i
                         for i, k in enumerate(["vine", "gauss", "indep", "hist"]) for m in ("es", "var")}, index=idx)
    ys = sorted(a.y for a in figs.risk_figure(risk, "es", "light").layout.annotations)
    assert len(ys) == 4 and min(np.diff(ys)) > 0.04 * 100 * (0.05 - 0.01) * 0.9  # >= 5% of the plotted range


def test_series_colors_are_the_validated_palette():
    assert figs.tokens("light")["series"][:3] == ["#2a78d6", "#eb6834", "#1baf7a"]
    assert figs.tokens("dark")["series"][:3] == ["#3987e5", "#d95926", "#199e70"]
    assert figs.tokens("unknown") == figs.tokens("light")


# ---- the real app, driven headlessly -----------------------------------------------
def _app(small_run, monkeypatch, run_dir=None):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("VINE_RISK_RUN_DIR", str(run_dir or small_run["run"]))
    monkeypatch.setenv("VINE_RISK_CONFIG", str(small_run["config"]))
    return AppTest.from_file(str(Path(__file__).parent.parent / "dashboard" / "app.py"), default_timeout=120)


def test_app_renders_all_panels_and_responds_to_widgets(small_run, monkeypatch):
    at = _app(small_run, monkeypatch).run()
    assert not at.exception
    assert [s.value[:2] for s in at.subheader] == ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
    assert len(at.get("plotly_chart")) == 9
    assert len(at.metric) == 8  # 4 key figures + 4 in the live panel
    at.selectbox(key=None)  # widgets exist
    pair_b = [s for s in at.selectbox if s.label == "Asset B"][0]
    pair_b.select("X3").run()
    assert not at.exception
    pair_b.select("X1").run()  # same asset twice -> a warning, not a crash
    assert not at.exception and any("different assets" in w.value for w in at.warning)
    [r for r in at.radio if r.label == "Measure"][0].set_value("Value at Risk").run()
    assert not at.exception
    at.sidebar.checkbox[0].check().run()  # data tables
    assert not at.exception and len(at.dataframe) >= 6


def test_app_degrades_gracefully_without_optional_files(small_run, monkeypatch, tmp_path):
    import shutil
    partial = tmp_path / "partial"
    shutil.copytree(small_run["run"], partial)
    for name in ("portfolio_risk.parquet", "change_scan.parquet", "structural_change_scores.parquet"):
        (partial / name).unlink()
    shutil.rmtree(partial / "live")
    at = _app(small_run, monkeypatch, partial).run()
    assert not at.exception
    texts = " ".join(i.value for i in at.info)
    assert "compute_risk.py" in texts and "detect_changes.py" in texts and "run_live.py" in texts


def test_app_shows_error_for_missing_run(small_run, monkeypatch, tmp_path):
    at = _app(small_run, monkeypatch, tmp_path / "does-not-exist").run()
    assert not at.exception and any("run_rolling" in e.value for e in at.error)


def test_matplotlib_vine_tree_draws_small_and_larger_trees():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vine_risk.visualization import plot_vine_tree
    res = _result("2020-01-03")
    for tree in (1, 2):
        fig = plot_vine_tree(res, tree)
        assert len(fig.axes) == 1
        plt.close(fig)
