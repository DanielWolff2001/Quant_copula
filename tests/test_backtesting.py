import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vine_risk.backtesting import (
    acerbi_szekely, christoffersen_cc, christoffersen_independence, diebold_mariano, evaluate_forecasts, fz0_loss,
    hits, kupiec_pof, pinball_loss,
)
from vine_risk.portfolio import tail_grid
from vine_risk.riskmodels import t_var_es


# ---- coverage tests --------------------------------------------------------------
def test_kupiec_known_values():
    lr, p = kupiec_pof(1000, 10, 0.01)  # exactly the expected number of exceedances
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    lr, p = kupiec_pof(250, 7, 0.01)  # hand computation of the likelihood ratio
    manual = -2 * ((243 * np.log(0.99) + 7 * np.log(0.01)) - (243 * np.log(243 / 250) + 7 * np.log(7 / 250)))
    assert lr == pytest.approx(manual) and p == pytest.approx(stats.chi2.sf(manual, 1))
    assert kupiec_pof(500, 0, 0.01)[1] < 0.05  # 5 exceedances expected, none seen
    assert kupiec_pof(500, 0, 0.001)[1] == pytest.approx(stats.chi2.sf(-2 * 500 * np.log(0.999), 1))  # 0.5 expected: no evidence
    assert kupiec_pof(250, 20, 0.01)[1] < 1e-6
    with pytest.raises(ValueError):
        kupiec_pof(0, 0, 0.01)


def test_christoffersen_independence_by_hand_and_by_simulation():
    h = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    n00, n01, n10, n11 = 4, 2, 2, 1  # transitions of the series above
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 9
    ll = lambda n, x, p: (n - x) * np.log(1 - p) + x * np.log(p)
    manual = -2 * (ll(9, 3, pi) - (ll(n00 + n01, n01, pi01) + ll(n10 + n11, n11, pi11)))
    assert christoffersen_independence(h)[0] == pytest.approx(manual)
    rng = np.random.default_rng(0)
    iid = rng.random(5000) < 0.02
    clustered = np.zeros(5000, bool)
    for s in rng.integers(0, 5000 - 5, 25):
        clustered[s: s + 4] = True  # bursts of exceedances
    assert christoffersen_independence(iid)[1] > 0.01 and christoffersen_independence(clustered)[1] < 1e-6
    assert christoffersen_independence(np.zeros(100))[1] == 1.0
    lr, p = christoffersen_cc(clustered, 0.02)
    assert p < 1e-6 and lr > kupiec_pof(5000, int(clustered.sum()), 0.02)[0]
    assert hits(np.array([1.0, 3.0]), np.array([2.0, 2.0])).tolist() == [False, True]


# ---- Expected Shortfall tests -----------------------------------------------------------
def _scenario(n, alpha, es_scale, seed, vol=None):
    """Realised losses from N(0, vol_t) and ES forecasts that are right (es_scale = 1) or too small."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.5, 2.0, n) if vol is None else vol
    losses = vol * rng.standard_normal(n)
    z = stats.norm.ppf(alpha)
    var = vol * z
    es_true = vol * stats.norm.pdf(z) / (1 - alpha)
    probs = alpha + (1 - alpha) * (np.arange(20) + 0.5) / 20
    tails = vol[:, None] * stats.norm.ppf(probs)[None, :]
    return losses, var * es_scale, es_true * es_scale, tails * es_scale


def test_acerbi_szekely_has_the_right_size_and_detects_underestimation():
    alpha, rej_true, rej_wrong = 0.975, 0, 0
    for seed in range(60):
        l, v, e, t = _scenario(1500, alpha, 1.0, seed)
        rej_true += acerbi_szekely(l, v, e, t, alpha, 500, seed)["p2"] < 0.05
        l, v, e, t = _scenario(1500, alpha, 0.65, seed)  # the whole forecast distribution is 35 % too narrow
        rej_wrong += acerbi_szekely(l, v, e, t, alpha, 500, seed)["p2"] < 0.05
    assert rej_true / 60 <= 0.15  # nominal 5 % (the tail grid is an approximation)
    assert rej_wrong / 60 >= 0.9


def test_acerbi_szekely_statistics_and_edge_cases():
    l, v, e, t = _scenario(2000, 0.99, 1.0, 1)
    out = acerbi_szekely(l, v, e, t, 0.99, 200, 0)
    assert abs(out["z2"]) < 0.5 and out["n_exceed"] == int((l > v).sum())
    only = acerbi_szekely(np.array([0.1] * 50), np.array([1.0] * 50), np.array([1.5] * 50), np.tile([2, 3.0], (50, 1)), 0.9, 100)
    assert np.isnan(only["z1"]) and np.isnan(only["p1"]) and only["z2"] == pytest.approx(-1.0)  # no exceedances at all
    big = acerbi_szekely(np.array([5.0, 6.0] * 25), np.array([1.0] * 50), np.array([2.0] * 50), np.tile([2, 3.0], (50, 1)), 0.9, 200)
    assert big["z1"] == pytest.approx(1.75) and big["p1"] < 0.01 and big["p2"] < 0.01  # losses far beyond the forecast ES
    with pytest.raises(ValueError, match="one row per day"):
        acerbi_szekely(l, v, e, t[:10], 0.99)


# ---- scoring rules and model comparison ---------------------------------------------------
def test_fz0_is_minimised_at_the_true_var_es_pair():
    rng = np.random.default_rng(2)
    nu, alpha = 5.0, 0.975
    z = stats.t.rvs(nu, size=400_000, random_state=rng) * np.sqrt((nu - 2) / nu)
    var, es = t_var_es(1.0, nu, alpha)
    best = fz0_loss(z, var, es, alpha).mean()
    for dv, de in ((1.15, 1.0), (0.85, 1.0), (1.0, 1.2), (1.0, 0.85), (1.1, 1.1), (0.9, 0.9)):
        assert fz0_loss(z, var * dv, es * de, alpha).mean() > best  # any other pair scores worse
    assert pinball_loss(z, var, alpha).mean() < pinball_loss(z, var * 1.2, alpha).mean()
    with pytest.raises(ValueError, match="positive"):
        fz0_loss(z[:5], np.ones(5), np.zeros(5), alpha)


def test_diebold_mariano():
    rng = np.random.default_rng(3)
    stat, p = diebold_mariano(rng.standard_normal(2000))
    assert p > 0.01
    stat, p = diebold_mariano(rng.standard_normal(2000) - 0.3)  # A clearly better (lower score)
    assert stat < -5 and p < 1e-6
    ar = np.zeros(3000)  # strongly autocorrelated differences with a tiny mean
    e = rng.standard_normal(3000)
    for t in range(1, 3000):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    assert diebold_mariano(ar, lag=0)[1] < diebold_mariano(ar, lag=40)[1]  # the robust variance is larger
    assert diebold_mariano(np.zeros(50)) == (0.0, 1.0)
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(5))


def test_evaluate_forecasts_ranks_models_and_reports_tests():
    n, alpha = 2500, 0.99
    rng = np.random.default_rng(4)
    vol = np.exp(0.5 * np.cumsum(rng.standard_normal(n)) * 0.05) * 0.01  # slowly varying volatility
    nu = 4.0
    loss = vol * stats.t.rvs(nu, size=n, random_state=rng) * np.sqrt((nu - 2) / nu)
    dates = pd.bdate_range("2015-01-01", periods=n)
    realized = pd.DataFrame({"p": loss}, index=dates)
    rows = []
    z = stats.norm.ppf(alpha)
    probs = alpha + (1 - alpha) * (np.arange(20) + 0.5) / 20
    for model in ("right", "thin", "flat"):
        for i, d in enumerate(dates):
            if model == "right":  # the true Student-t forecast
                v, e = t_var_es(vol[i], nu, alpha)
                tail = vol[i] * np.sqrt((nu - 2) / nu) * stats.t.ppf(probs, nu)
            elif model == "thin":  # normal tails: understates the ES
                v, e = vol[i] * z, vol[i] * stats.norm.pdf(z) / (1 - alpha)
                tail = vol[i] * stats.norm.ppf(probs)
            else:  # constant volatility: ignores the volatility dynamics
                v, e = t_var_es(vol.mean(), nu, alpha)
                tail = vol.mean() * np.sqrt((nu - 2) / nu) * stats.t.ppf(probs, nu)
            rows.append({"date": d, "model": model, "portfolio": "p", "alpha": alpha, "var": v, "es": e, "tail": tail})
    table = evaluate_forecasts(pd.DataFrame(rows), realized, reference="right", n_sim=500, seed=1)
    r = table.loc[("p", alpha)]
    assert set(r.index) == {"right", "thin", "flat"} and (r.n == n).all()
    assert r.loc["right", "kupiec_p"] > 0.05 and r.loc["right", "z2_p"] > 0.01  # the true model passes
    assert r.loc["thin", "z2_p"] < 0.05 and r.loc["thin", "z2"] > 0  # thin tails: ES underestimated
    assert r.loc["right", "fz0"] < r.loc["thin", "fz0"] and r.loc["right", "fz0"] < r.loc["flat", "fz0"]
    assert r.loc["right", "dm_p"] == 1.0 and r.loc["flat", "dm_stat"] > 0 and r.loc["flat", "dm_p"] < 0.05  # worse than the reference
