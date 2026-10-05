import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vine_risk.garch import GarchMarginal
from vine_risk.marginals import MARGINAL_KINDS, EmpiricalMarginal, MarginalSpec
from vine_risk.synthetic import Regime, simulate_garch


@pytest.fixture(scope="module")
def data():
    return simulate_garch([Regime(1500, 0.5)], n_assets=3, seed=3, omega=0.05, alpha=0.08, beta=0.90, nu=6.0)


@pytest.fixture(scope="module")
def fitted(data):
    r, _ = data
    return GarchMarginal().fit(r)


def test_simulator_matches_its_definition(data):
    r, s = data
    assert r.shape == s.shape == (1500, 3)
    z = (r / s).to_numpy()
    assert z.std() == pytest.approx(1.0, abs=0.05) and abs(z.mean()) < 0.05
    assert stats.kurtosis(z.ravel()) > 1.0  # fat-tailed innovations
    sq = (r ** 2).iloc[:, 0]
    assert sq.autocorr(1) > 0.1  # volatility clusters
    resid = s.iloc[1:].to_numpy() ** 2 * 1e4 - 0.05 - 0.08 * (r.iloc[:-1].to_numpy() * 100) ** 2 - 0.90 * s.iloc[:-1].to_numpy() ** 2 * 1e4
    assert np.abs(resid).max() < 1e-9  # sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2
    with pytest.raises(ValueError):
        simulate_garch([Regime(10, 0.5)], alpha=0.6, beta=0.5)


def test_recovers_garch_parameters(fitted):
    for c, p in fitted.params.items():
        assert p["alpha[1]"] + p["beta[1]"] == pytest.approx(0.98, abs=0.05)
        assert 0.02 < p["alpha[1]"] < 0.18 and 0.8 < p["beta[1]"] < 0.97
        assert 4 < p["nu"] < 10
    assert fitted.fallbacks == []


def test_pit_is_uniform_and_removes_volatility_clustering(data, fitted):
    r, _ = data
    u = fitted.transform(r)
    assert ((u > 0) & (u < 1)).all().all()
    for c in u:
        assert stats.kstest(u[c], "uniform").pvalue > 0.01
    raw = r.iloc[:, 0].abs().autocorr(1)
    filt = pd.Series(stats.norm.ppf(u.iloc[:, 0])).abs().autocorr(1)
    assert raw > 0.15 and filt < 0.5 * raw  # the filter removes most of the clustering


def test_transform_accepts_subwindows_only(data, fitted):
    r, _ = data
    sub = fitted.transform(r.iloc[-250:])
    pd.testing.assert_frame_equal(sub, fitted.transform(r).iloc[-250:])  # the same u, whatever the sub-window
    with pytest.raises(ValueError, match="fitted history"):
        fitted.transform(r.iloc[:5].set_axis(pd.bdate_range("2030-01-01", periods=5)))
    with pytest.raises(ValueError, match="do not match"):
        fitted.transform(r[["X2", "X1", "X3"]])
    with pytest.raises(RuntimeError):
        GarchMarginal().transform(r)


def test_inverse_gives_the_conditional_one_day_ahead_distribution(data, fitted):
    r, _ = data
    nd = fitted.next_day()
    u = pd.DataFrame({c: [0.5, 0.99, 0.01] for c in r.columns})
    q = fitted.inverse_transform(u)
    for c in r.columns:
        assert q[c].iloc[0] == pytest.approx(nd.loc["mu", c], abs=1e-9)  # the median is the mean
        nu = fitted.nu[c]
        z99 = stats.t.ppf(0.99, nu) / np.sqrt(nu / (nu - 2))
        assert q[c].iloc[1] == pytest.approx(nd.loc["mu", c] + nd.loc["sigma", c] * z99)
        assert q[c].iloc[2] < nd.loc["mu", c] < q[c].iloc[1]


def test_conditional_scale_follows_recent_volatility():
    calm, _ = simulate_garch([Regime(800, 0.5)], n_assets=2, seed=5)
    wild = calm.copy()
    wild.iloc[-5:] *= 5.0  # a volatility burst on the last five days
    a, b = GarchMarginal().fit(calm), GarchMarginal().fit(wild)
    assert (b.next_day().loc["sigma"] > 1.5 * a.next_day().loc["sigma"]).all()


def test_empirical_innovations(data):
    r, _ = data
    m = GarchMarginal(innovations="empirical").fit(r)
    u = m.transform(r)
    assert all(stats.kstest(u[c], "uniform").pvalue > 0.5 for c in u)  # rank-based: almost exactly uniform
    q = m.inverse_transform(pd.DataFrame({c: [0.5] for c in r.columns}))
    assert np.allclose(q.iloc[0], m.next_day().loc["mu"], atol=5e-4)  # the median residual is about zero


def test_failed_fit_falls_back_to_ewma_and_stays_finite():
    flat = pd.DataFrame({"A": 0.01 * np.sign(np.sin(np.arange(300))), "B": np.random.default_rng(0).normal(0, 0.01, 300)},
                        index=pd.bdate_range("2020-01-01", periods=300))
    m = GarchMarginal().fit(flat)
    u = m.transform(flat)
    assert np.isfinite(u.to_numpy()).all() and ((u > 0) & (u < 1)).all().all()
    assert np.isfinite(m.next_day().to_numpy()).all() and (m.next_day().loc["sigma"] > 0).all()


def test_input_validation(data):
    r, _ = data
    with pytest.raises(ValueError):
        GarchMarginal(innovations="normal")
    with pytest.raises(ValueError, match="at least"):
        GarchMarginal().fit(r.iloc[:50])
    bad = r.copy(); bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        GarchMarginal().fit(bad)
    assert isinstance(GarchMarginal().fit(r).standardised_residuals(), pd.DataFrame)


def test_marginal_spec_is_picklable_and_validated():
    assert MARGINAL_KINDS == ("empirical", "garch_t", "garch_empirical")
    assert isinstance(MarginalSpec()(), EmpiricalMarginal)
    assert isinstance(MarginalSpec("garch_t")(), GarchMarginal) and MarginalSpec("garch_t")().innovations == "t"
    assert MarginalSpec("garch_empirical")().innovations == "empirical"
    spec = pickle.loads(pickle.dumps(MarginalSpec("garch_t")))  # needed for worker processes
    assert spec == MarginalSpec("garch_t") and spec != MarginalSpec() and hash(spec) == hash(MarginalSpec("garch_t"))
    with pytest.raises(ValueError, match="Unknown marginal"):
        MarginalSpec("kernel")
