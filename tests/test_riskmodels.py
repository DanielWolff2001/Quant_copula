import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vine_risk.garch import GarchMarginal
from vine_risk.portfolio import portfolio_loss, var_es
from vine_risk.riskmodels import (
    EwmaNormal, EwmaStudentT, FilteredHistoricalSimulation, HistoricalSimulation, RiskForecast, ewma_covariances,
    fit_t_dof, t_var_es,
)
from vine_risk.synthetic import Regime, simulate_garch

W3 = {"eq": np.array([1 / 3] * 3), "first": np.array([1.0, 0.0, 0.0])}
ALPHAS = [0.975, 0.99]


@pytest.fixture(scope="module")
def iid():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((3000, 3)) * 0.01, index=pd.bdate_range("2010-01-04", periods=3000), columns=list("ABC"))


def _by(fc, portfolio, alpha):
    return next(f for f in fc if f.portfolio == portfolio and f.alpha == alpha)


def test_forecast_records_are_complete_for_every_model(iid):
    models = [HistoricalSimulation(250), EwmaNormal(), EwmaStudentT(), FilteredHistoricalSimulation(500)]
    for m in models:
        fc = m.forecast(iid.iloc[:700], W3, ALPHAS)
        assert len(fc) == 4 and all(isinstance(f, RiskForecast) and f.model == m.name for f in fc)
        for f in fc:
            assert 0 < f.var < f.es and len(f.tail) == 20 and np.all(np.diff(f.tail) >= -1e-12)
            assert f.tail.mean() == pytest.approx(f.es, rel=0.15)  # the tail grid summarises the ES
        assert _by(fc, "eq", 0.99).var > _by(fc, "eq", 0.975).var


def test_historical_simulation_matches_the_definition(iid):
    h = iid.iloc[:400]
    f = _by(HistoricalSimulation(250).forecast(h, W3, [0.99]), "eq", 0.99)
    var, es = var_es(portfolio_loss(W3["eq"], h.iloc[-250:].to_numpy()), 0.99)
    assert (f.var, f.es) == (var, es)


def test_ewma_normal_matches_the_closed_form_and_analytic_truth(iid):
    f = _by(EwmaNormal(lam=0.999, lookback=3000).forecast(iid, W3, [0.99]), "first", 0.99)
    assert f.var == pytest.approx(0.01 * stats.norm.ppf(0.99), rel=0.1)  # iid N(0, 1%) data
    assert f.es == pytest.approx(0.01 * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, rel=0.1)
    S = ewma_covariances(iid.iloc[:200].to_numpy(), 0.94)
    assert S.shape == (201, 3, 3) and np.allclose(S[-1], S[-1].T)
    manual = 0.94 * S[199] + 0.06 * np.outer(iid.iloc[199], iid.iloc[199])
    assert np.allclose(S[200], manual)
    with pytest.raises(ValueError):
        EwmaNormal(lam=1.0)


def test_ewma_reacts_to_a_volatility_burst_historical_simulation_does_not(iid):
    h = iid.iloc[:600].copy()
    h.iloc[-10:] *= 4.0
    calm = iid.iloc[:600]
    r_e = _by(EwmaNormal().forecast(h, W3, [0.99]), "eq", 0.99).var / _by(EwmaNormal().forecast(calm, W3, [0.99]), "eq", 0.99).var
    r_h = _by(HistoricalSimulation().forecast(h, W3, [0.99]), "eq", 0.99).var / _by(HistoricalSimulation().forecast(calm, W3, [0.99]), "eq", 0.99).var
    assert r_e > 2.0 and r_h < r_e


def test_student_t_formulas_and_dof_estimation():
    rng = np.random.default_rng(1)
    nu = 5.0
    z = stats.t.rvs(nu, size=200_000, random_state=rng) * np.sqrt((nu - 2) / nu)  # unit variance
    var, es = t_var_es(1.0, nu, 0.99)
    assert var == pytest.approx(np.quantile(z, 0.99), rel=0.02) and es == pytest.approx(z[z >= var].mean(), rel=0.03)
    assert fit_t_dof(z[:20_000]) == pytest.approx(5.0, abs=0.5)
    assert fit_t_dof(rng.standard_normal(20_000)) > 30  # normal data: very large degrees of freedom
    assert t_var_es(1.0, 4.0, 0.99)[1] > t_var_es(1.0, 30.0, 0.99)[1]  # fatter tails, larger ES at equal volatility


def test_ewma_student_t_has_fatter_tails_than_normal_on_fat_tailed_data():
    r, _ = simulate_garch([Regime(1500, 0.5)], n_assets=3, seed=4, nu=4.0, alpha=0.0001, beta=0.0001, omega=1.0)  # ~iid t(4)
    n = _by(EwmaNormal(lam=0.99, lookback=1500).forecast(r, W3, [0.99]), "first", 0.99)
    t = _by(EwmaStudentT(lam=0.99, lookback=1500).forecast(r, W3, [0.99]), "first", 0.99)
    assert t.es > n.es and t.es / t.var > n.es / n.var  # heavier tail beyond the VaR


def test_filtered_historical_simulation_recovers_the_true_conditional_quantile():
    omega, alpha, beta, nu = 0.05, 0.08, 0.90, 6.0
    r, sig = simulate_garch([Regime(1400, 0.5)], n_assets=3, seed=6, omega=omega, alpha=alpha, beta=beta, nu=nu)
    for cut in (1200, 1300, 1400):  # several forecast dates
        h, s = r.iloc[:cut], sig.iloc[:cut]
        true_next = np.sqrt(omega + alpha * (h.iloc[-1, 0] * 100) ** 2 + beta * (s.iloc[-1, 0] * 100) ** 2) / 100
        true_var = true_next * stats.t.ppf(0.99, nu) / np.sqrt(nu / (nu - 2))
        f = _by(FilteredHistoricalSimulation(1000).forecast(h, {"first": W3["first"]}, [0.99]), "first", 0.99)
        assert f.var == pytest.approx(true_var, rel=0.25)  # estimation error of a 1000-day GARCH fit and a 1% tail
    # an unfiltered model with the same history misses the conditional volatility
    h = r.iloc[:1400]
    assert abs(np.log(f.var / true_var)) < abs(np.log(_by(HistoricalSimulation(250).forecast(h, W3, [0.99]), "first", 0.99).var / true_var)) + 0.1


def test_shared_garch_fit_gives_identical_results(iid):
    h = iid.iloc[:600]
    g = GarchMarginal(innovations="empirical").fit(h)
    fhs = FilteredHistoricalSimulation(600)
    a = fhs.forecast(h, W3, [0.99])
    b = fhs.forecast(h, W3, [0.99], garch=g)
    assert all(x.var == y.var and x.es == y.es for x, y in zip(a, b))


def test_input_validation(iid):
    h = iid.iloc[:300]
    holes = h.copy()
    holes.iloc[5, 0] = np.nan
    for model in (HistoricalSimulation(), EwmaNormal(), EwmaStudentT()):
        with pytest.raises(ValueError, match="weights"):
            model.forecast(h, {"bad": np.ones(2)}, [0.99])
        with pytest.raises(ValueError, match="alphas"):
            model.forecast(h, W3, [1.5])
        with pytest.raises(ValueError, match="NaN"):
            model.forecast(holes, W3, [0.99])


def test_fhs_uses_each_residual_once():
    r, _ = simulate_garch([Regime(1000, 0.5)], n_assets=3, seed=9)
    h = r.iloc[:800]
    once = _by(FilteredHistoricalSimulation(500).forecast(h, W3, [0.99]), "eq", 0.99)
    again = _by(FilteredHistoricalSimulation(500).forecast(h, W3, [0.99]), "eq", 0.99)
    assert (once.var, once.es) == (again.var, again.es)  # deterministic: no random numbers
    g = GarchMarginal(innovations="empirical").fit(h.iloc[-500:])
    z = g.standardised_residuals().to_numpy()
    nd = g.next_day()
    direct = np.quantile(portfolio_loss(W3["eq"], nd.loc["mu"].to_numpy() + nd.loc["sigma"].to_numpy() * z), 0.99)
    assert once.var == pytest.approx(direct)  # exactly the empirical quantile of the 500 scenarios
    boot = _by(FilteredHistoricalSimulation(500, n_boot=2 ** 12, seed=1).forecast(h, W3, [0.99]), "eq", 0.99)
    assert boot.var != once.var and 0.5 < boot.var / once.var < 2  # the resampling option exists and is in the same range
