import numpy as np
import pytest
from scipy import stats

from vine_risk.synthetic import Regime, simulate_regimes


def test_shape_index_and_reproducibility():
    a = simulate_regimes([Regime(100, 0.5), Regime(50, 0.8)], n_assets=3, seed=1)
    b = simulate_regimes([Regime(100, 0.5), Regime(50, 0.8)], n_assets=3, seed=1)
    assert a.shape == (150, 3) and a.equals(b) and a.index.is_monotonic_increasing


def test_gaussian_and_t_regimes_have_requested_tau():
    for df in (None, 4):
        x = simulate_regimes([Regime(6000, 0.6, df)], n_assets=2, seed=2)
        tau = stats.kendalltau(x["X1"], x["X2"]).statistic
        assert tau == pytest.approx(Regime(1, 0.6).tau, abs=0.03)


def test_tail_dependence_truth():
    assert Regime(1, 0.5).tail_dependence == 0.0
    assert Regime(1, 0.5, df=3).tail_dependence == pytest.approx(0.3125, abs=0.02)  # known t_3 value
    # empirical check: the t copula puts more joint mass in the corner than the Gaussian
    q = 0.02
    def corner(df):
        x = simulate_regimes([Regime(60000, 0.5, df)], n_assets=2, seed=3).rank() / 60001
        return np.mean((x["X1"] < q) & (x["X2"] < q))
    assert corner(3) > 1.5 * corner(None)


def test_invalid_input():
    with pytest.raises(ValueError):
        simulate_regimes([Regime(10, 1.0)])
    with pytest.raises(ValueError):
        simulate_regimes([Regime(10, 0.5)], n_assets=1)


def test_volatility_clustering_option():
    base = simulate_regimes([Regime(2000, 0.5)], n_assets=3, seed=4)
    vc = simulate_regimes([Regime(2000, 0.5)], n_assets=3, seed=4, vol_phi=0.98)
    assert base.shape == vc.shape
    ratio = vc / base  # the common volatility factor is shared by all assets and days
    assert np.allclose(ratio.iloc[:, 0], ratio.iloc[:, 1]) and ratio.iloc[:, 0].std() > 0.2
    sq = (vc ** 2).iloc[:, 0]
    assert sq.autocorr(1) > 0.15 and abs((base ** 2).iloc[:, 0].autocorr(1)) < 0.1  # clustering
