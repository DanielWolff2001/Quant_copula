import numpy as np
import pandas as pd
import pyvinecopulib as pv
import pytest
from scipy import stats

from vine_risk.config import load_config
from vine_risk.copula import VineCopula, VineFitResult
from vine_risk.marginals import EmpiricalMarginal
from vine_risk.portfolio import (
    backtest_var, equal_weights, portfolio_loss, rebuild_vinecop, rolling_risk, var_es, window_risk,
)
from vine_risk.rolling import RollingVineModel
from vine_risk.synthetic import Regime, simulate_regimes

FAST = dict(truncation_level=2, tail_simulations=0)


# ---- loss, VaR, ES against hand calculations -------------------------------
def test_portfolio_loss_matches_manual_example():
    r = np.array([[0.01, -0.02, 0.03], [-0.05, 0.00, 0.01]])
    w = np.array([0.5, 0.3, 0.2])
    manual = [-(0.5 * 0.01 + 0.3 * -0.02 + 0.2 * 0.03), -(0.5 * -0.05 + 0.3 * 0.0 + 0.2 * 0.01)]
    assert portfolio_loss(w, r) == pytest.approx(manual)
    assert portfolio_loss(w, r[0]) == pytest.approx(manual[0])  # a single scenario
    with pytest.raises(ValueError):
        portfolio_loss(w[:2], r)


def test_var_es_known_sample():
    losses = np.arange(1.0, 101.0)
    var, es = var_es(losses, 0.95)
    assert var == pytest.approx(95.05)
    assert es == pytest.approx(98.0)  # mean of 96..100
    assert es >= var


def test_var_es_normal_distribution():
    x = np.random.default_rng(0).standard_normal(400_000)
    var, es = var_es(x, 0.99)
    assert var == pytest.approx(stats.norm.ppf(0.99), abs=0.03)
    assert es == pytest.approx(stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, abs=0.04)  # 2.665


def test_var_es_validation():
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            var_es(np.arange(10.0), bad)
    with pytest.raises(ValueError):
        var_es(np.array([1.0, np.nan]))
    assert equal_weights(4).tolist() == [0.25] * 4


# ---- rebuilding a fitted vine from its stored result ------------------------
@pytest.mark.parametrize("trunc", [None, 2])
def test_rebuilt_vine_is_identical_to_the_fitted_one(trunc):
    r = simulate_regimes([Regime(300, 0.5, df=4)], n_assets=4, seed=1)
    u = EmpiricalMarginal().fit_transform(r)
    v = VineCopula(truncation_level=trunc, tail_simulations=0).fit(u, r)
    rb = rebuild_vinecop(v.summary())
    assert np.abs(v._model.logpdf(u.to_numpy()) - rb.logpdf(u.to_numpy())).max() == 0.0
    s = pv.utils.sobol(512, 4, [3])
    assert np.abs(v._model.inverse_rosenblatt(s) - rb.inverse_rosenblatt(s)).max() == 0.0


def test_rebuild_survives_json_roundtrip_and_rejects_failed_fit():
    r = simulate_regimes([Regime(200, 0.5)], n_assets=3, seed=2)
    res = VineCopula(**FAST).fit(EmpiricalMarginal().fit_transform(r)).summary()
    rebuild_vinecop(VineFitResult.from_json(res.to_json()))
    with pytest.raises(ValueError):
        rebuild_vinecop(VineFitResult.failed(["A", "B"], 100, "boom"))


# ---- risk of a window ---------------------------------------------------------
def _fit(regime, n_assets=3, seed=0):
    r = simulate_regimes([regime], n_assets=n_assets, seed=seed)
    res = VineCopula(**FAST).fit(EmpiricalMarginal().fit_transform(r), r).summary()
    return res, r


def test_independence_benchmark_matches_analytic_normal_portfolio():
    res, r = _fit(Regime(2000, 0.0))
    out = window_risk(res, r, alpha=0.99, n_sims=2 ** 14)
    sd = np.sqrt(3 * (1 / 3) ** 2) * 0.01  # three independent N(0, 1%) assets, equal weights
    assert out["var_indep"] == pytest.approx(stats.norm.ppf(0.99) * sd, rel=0.05)
    assert out["es_indep"] == pytest.approx(stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01 * sd, rel=0.06)


def test_dependence_raises_tail_risk_and_models_agree_when_dependence_is_gaussian():
    res, r = _fit(Regime(1500, 0.8))
    out = window_risk(res, r, n_sims=2 ** 14)
    assert out["es_vine"] > 1.3 * out["es_indep"]
    assert out["es_vine"] == pytest.approx(out["es_gauss"], rel=0.08)  # truth is a Gaussian copula
    assert out["es_vine"] == pytest.approx(out["es_hist"], rel=0.15)
    assert out["es_vine"] >= out["var_vine"]


def test_t_copula_has_fatter_tail_than_gaussian_with_same_correlation():
    res, r = _fit(Regime(1500, 0.5, df=3), n_assets=4)
    out = window_risk(res, r, n_sims=2 ** 14)
    assert out["es_vine"] > out["es_gauss"]


def test_window_risk_is_reproducible_and_checks_columns():
    res, r = _fit(Regime(300, 0.5))
    assert window_risk(res, r, n_sims=2 ** 10) == window_risk(res, r, n_sims=2 ** 10)
    with pytest.raises(ValueError):
        window_risk(res, r.iloc[:, ::-1])


# ---- rolling risk and backtest --------------------------------------------------
@pytest.fixture(scope="module")
def rolling_setup():
    r = simulate_regimes([Regime(220, 0.2), Regime(180, 0.8)], n_assets=3, seed=5)
    res = RollingVineModel(80, 5, FAST).run(r)
    return res, r


def test_rolling_risk_shape_columns_and_no_lookahead(rolling_setup):
    res, r = rolling_setup
    risk = rolling_risk(res, r, n_sims=2 ** 10, step=2)
    assert len(risk) == len(res[::2])
    assert {"es_vine", "es_dependence_ratio", "es_non_gaussian", "var_hist"} <= set(risk.columns)
    assert (risk.es_vine >= risk.var_vine).all()
    cut = rolling_risk(RollingVineModel(80, 5, FAST).run(r.iloc[:300]), r.iloc[:300], n_sims=2 ** 10, step=2)
    pd.testing.assert_frame_equal(risk.loc[cut.index], cut)  # earlier rows ignore later data


def test_dependence_ratio_rises_with_the_correlation_jump(rolling_setup):
    res, r = rolling_setup
    risk = rolling_risk(res, r, n_sims=2 ** 12)
    early, late = risk.iloc[:3], risk.iloc[-3:]
    assert late.es_dependence_ratio.mean() > early.es_dependence_ratio.mean() + 0.1


def test_parallel_matches_serial(rolling_setup):
    res, r = rolling_setup
    a = rolling_risk(res[:6], r, n_sims=2 ** 10, n_jobs=1)
    b = rolling_risk(res[:6], r, n_sims=2 ** 10, n_jobs=2)
    pd.testing.assert_frame_equal(a, b)


def test_backtest_counts_exceedances_and_kupiec():
    idx = pd.bdate_range("2020-01-01", periods=6)
    rets = pd.DataFrame({"A": [0.0, -0.02, 0.0, -0.05, 0.01, 0.0]}, index=idx)  # losses 0, .02, 0, .05, -.01, 0
    risk = pd.DataFrame({"var_vine": 0.03}, index=idx[:5])
    bt = backtest_var(risk, rets, weights=np.array([1.0]), alpha=0.99, models=("vine",))
    row = bt.loc["vine"]
    assert row["n"] == 5 and row["exceedances"] == 1 and row["rate"] == pytest.approx(0.2)
    assert 0 < row["kupiec_p"] < 0.05  # 1 hit in 5 days vs 1% expected is implausible


def test_config_defaults():
    cfg = load_config("configs/default.yaml")
    assert cfg.risk.simulations == 16384 and cfg.risk.weights is None and cfg.risk.confidence_level == 0.99


# ---- generalised scenarios: marginals, tail grids, summaries ------------------------------
from vine_risk.garch import GarchMarginal
from vine_risk.marginals import MarginalSpec
from vine_risk.portfolio import loss_summary, scenario_returns, tail_grid
from vine_risk.synthetic import simulate_garch


def test_tail_grid_and_loss_summary():
    losses = np.random.default_rng(0).standard_normal(200_000)
    grid = tail_grid(losses, 0.99, k=20)
    assert len(grid) == 20 and (np.diff(grid) > 0).all() and grid[0] > stats.norm.ppf(0.99)
    var, es = var_es(losses, 0.99)
    assert grid.mean() == pytest.approx(es, rel=0.02)  # the grid reproduces the expected shortfall
    summ = loss_summary(losses, [0.975, 0.99], k=10)
    assert set(summ) == {0.975, 0.99} and summ[0.99]["var"] == pytest.approx(var) and summ[0.99]["es"] == pytest.approx(es)
    assert summ[0.975]["var"] < summ[0.99]["var"] and len(summ[0.975]["tail"]) == 10


def test_scenario_returns_default_matches_window_risk_and_shares_random_numbers():
    res, r = _fit(Regime(400, 0.5))
    sc = scenario_returns(res, r, n_sims=2 ** 10, seed=1)
    assert set(sc) == {"vine", "gauss", "indep"} and all(v.shape == (2 ** 10, 3) for v in sc.values())
    out = window_risk(res, r, n_sims=2 ** 10, seed=1)
    var, es = var_es(portfolio_loss(equal_weights(3), sc["vine"]), 0.99)
    assert out["var_vine"] == var and out["es_vine"] == es
    again = scenario_returns(res, r, n_sims=2 ** 10, seed=1)
    assert all(np.array_equal(sc[k], again[k]) for k in sc)


def test_garch_marginal_makes_risk_a_conditional_forecast():
    calm, _ = simulate_garch([Regime(700, 0.5)], n_assets=3, seed=8)
    wild = calm.copy()
    wild.iloc[-8:] *= 4.0  # a volatility burst just before the forecast date
    res = {}
    for name, data in (("calm", calm), ("wild", wild)):
        marg = GarchMarginal().fit(data)
        win = data.iloc[-250:]
        fit = VineCopula(**FAST).fit(marg.transform(win), win).summary()
        res[name] = (window_risk(fit, win, n_sims=2 ** 12, marginal=marg), window_risk(fit, win, n_sims=2 ** 12))
    garch_calm, emp_calm = res["calm"]
    garch_wild, emp_wild = res["wild"]
    assert garch_wild["es_vine"] > 1.8 * garch_calm["es_vine"]  # conditional: reacts to today's volatility
    assert emp_wild["es_vine"] < 1.8 * emp_calm["es_vine"]  # rank marginals describe the whole window
    assert garch_wild["es_indep"] > 1.5 * garch_calm["es_indep"]


def test_rolling_risk_with_a_garch_marginal_equals_manual_computation():
    r, _ = simulate_garch([Regime(330, 0.6)], n_assets=3, seed=2)
    model = RollingVineModel(60, 20, dict(FAST, tail_simulations=2 ** 9), MarginalSpec("garch_t"), 150)
    fits = model.run(r)
    risk = rolling_risk(fits, r, n_sims=2 ** 9, seed=0, step=2, marginal_factory=MarginalSpec("garch_t"), lookback=150)
    f = fits[2]
    end = r.index.get_loc(pd.Timestamp(f.timestamp))
    hist = r.iloc[end - 149: end + 1]
    manual = window_risk(f, hist.iloc[-60:], n_sims=2 ** 9, seed=0, marginal=GarchMarginal().fit(hist))
    got = risk.loc[pd.Timestamp(f.timestamp)]
    assert got["es_vine"] == pytest.approx(manual["es_vine"]) and got["var_indep"] == pytest.approx(manual["var_indep"])
    plain = rolling_risk(fits, r, n_sims=2 ** 9, seed=0, step=2)
    assert not np.isclose(plain.loc[pd.Timestamp(f.timestamp), "es_vine"], got["es_vine"])  # the marginal matters
