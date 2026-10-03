import json

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import pytest
from scipy import stats

from vine_risk.copula import VineCopula, VineFitError, VineFitResult
from vine_risk.marginals import EmpiricalMarginal


def _clayton_gaussian(n=1500, seed=3):
    """A, B ~ Clayton(theta=2, tau=0.5); C ~ Gaussian(rho=0.6) with B (tau~0.41)."""
    ab = pv.Bicop(pv.BicopFamily.clayton, 0, np.array([[2.0]])).sample(n, seeds=[seed])
    rng = np.random.default_rng(seed)
    z = stats.norm.ppf(ab[:, 1])
    c = stats.norm.cdf(0.6 * z + 0.8 * rng.standard_normal(n))
    r = pd.DataFrame({"A": ab[:, 0], "B": ab[:, 1], "C": c},
                     index=pd.bdate_range("2020-01-01", periods=n))
    return EmpiricalMarginal().fit_transform(r), r


@pytest.fixture(scope="module")
def fitted():
    u, r = _clayton_gaussian()
    v = VineCopula().fit(u, r)
    return v, v.summary(), u


def test_recovers_tree1_structure_and_dependence(fitted):
    _, res, u = fitted
    t1 = {frozenset(p.conditioned): p for p in res.pair_copulas if p.tree == 1}
    assert len(t1) == 2
    ab = t1[frozenset({"A", "B"})]
    assert ab.family == "clayton" and ab.rotation == 0
    assert ab.parameters[0] == pytest.approx(2.0, rel=0.2)
    assert ab.tau == pytest.approx(0.5, abs=0.05)
    assert ab.lower_tail == pytest.approx(2 ** -0.5, abs=0.1) and ab.upper_tail == 0.0
    assert frozenset({"B", "C"}) in t1


def test_number_of_pair_copulas_and_diagnostics(fitted):
    _, res, u = fitted
    d = 3
    assert len(res.pair_copulas) == d * (d - 1) // 2
    assert res.n_obs == len(u) and res.n_assets == d and res.status == "ok"
    assert res.bic == pytest.approx(-2 * res.loglik + np.log(res.n_obs) * res.n_params)
    assert res.aic == pytest.approx(-2 * res.loglik + 2 * res.n_params)
    assert res.window_start == "2020-01-01"


def test_empirical_tau_matches_scipy(fitted):
    _, res, u = fitted
    m = res.tau_matrix()
    assert m.loc["A", "B"] == pytest.approx(stats.kendalltau(u["A"], u["B"]).statistic)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1)
    assert res.pairwise_frame()["pearson"].notna().all()


def test_json_roundtrip(fitted):
    _, res, _ = fitted
    back = VineFitResult.from_json(res.to_json())
    assert back == res
    json.loads(res.to_json())  # valid JSON


def test_simulation_is_seeded_and_uniform(fitted):
    v, _, u = fitted
    s1, s2 = v.simulate(2000, seed=7), v.simulate(2000, seed=7)
    pd.testing.assert_frame_equal(s1, s2)
    assert ((s1 > 0) & (s1 < 1)).all().all()
    assert stats.kendalltau(s1["A"], s1["B"]).statistic == pytest.approx(0.5, abs=0.06)


def test_input_validation():
    u, _ = _clayton_gaussian(200)
    with pytest.raises(ValueError):
        VineCopula().fit(u.iloc[:, :1])
    with pytest.raises(ValueError):
        VineCopula().fit(u.iloc[:20])
    bad = u.copy()
    bad.iloc[0, 0] = 1.0
    with pytest.raises(ValueError):
        VineCopula().fit(bad)
    with pytest.raises(ValueError):
        VineCopula(selection_criterion="mbic")
    with pytest.raises(ValueError):
        VineCopula(families=["nonsense"])
    with pytest.raises(RuntimeError):
        VineCopula().summary()
    assert issubclass(VineFitError, RuntimeError)


def test_restricted_family_set():
    u, _ = _clayton_gaussian(500)
    res = VineCopula(families=["indep", "gaussian"]).fit(u).summary()
    assert {p.family for p in res.pair_copulas} <= {"independence", "gaussian"}
