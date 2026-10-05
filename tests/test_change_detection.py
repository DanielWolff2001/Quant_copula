import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vine_risk.change_detection import (
    aspect_flags, benjamini_hochberg, change_scan, cusum, default_lag, frobenius_change,
    rolling_zscore, structural_change_scores, two_window_change_test,
)
from vine_risk.rolling import RollingVineModel
from vine_risk.synthetic import Regime, simulate_regimes


# ---- simple building blocks -------------------------------------------------
def test_frobenius_change_hand_computed():
    pairs = pd.DataFrame({"A-B": [0.5, 0.2, 0.2], "A-C": [0.1, 0.1, 0.4]},
                         index=pd.date_range("2020-01-01", periods=3))
    s = frobenius_change(pairs, lag=1)
    assert np.isnan(s.iloc[0])
    assert s.iloc[1] == pytest.approx(np.sqrt(2 * 0.3 ** 2))  # full symmetric matrix counts each pair twice
    assert s.iloc[2] == pytest.approx(np.sqrt(2 * 0.3 ** 2))
    assert frobenius_change(pairs, lag=2).iloc[2] == pytest.approx(np.sqrt(2 * (0.3 ** 2 + 0.3 ** 2)))
    with pytest.raises(ValueError):
        frobenius_change(pairs, lag=0)


def test_default_lag():
    assert default_lag(250, 1) == 250 and default_lag(250, 5) == 50 and default_lag(3, 10) == 1


def test_rolling_zscore_uses_only_the_past():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    z = rolling_zscore(x, baseline=4, min_periods=4)
    assert z.iloc[:4].isna().all()
    past = [1.0, 2.0, 3.0, 4.0]
    assert z.iloc[4] == pytest.approx((100 - np.mean(past)) / np.std(past, ddof=1))
    assert rolling_zscore(pd.Series([1.0] * 6), baseline=3, min_periods=3).isna().all()  # zero variance


def test_cusum_known_sequence():
    c = cusum(pd.Series([0.0, 1.5, 1.5, -2.0, np.nan, 1.0]), k=0.5)
    assert c.tolist() == pytest.approx([0.0, 1.0, 2.0, 0.0, 0.0, 0.5])


def test_aspect_flags_exceed_past_quantile():
    a = pd.DataFrame({"x": [0.0, 1.0] * 5 + [5.0]})
    f = aspect_flags(a, baseline=10, quantile=0.9, min_periods=5)
    assert bool(f["x"].iloc[-1]) and not f["x"].iloc[:5].any()


def test_benjamini_hochberg_textbook_example():
    p = pd.Series([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216], name="p")
    assert benjamini_hochberg(p, 0.05).sum() == 2  # k=2: 0.008 <= 0.05*2/10, 0.039 > 0.05*3/10
    assert benjamini_hochberg(pd.Series([np.nan, 0.5]), 0.05).tolist() == [False, False]
    with pytest.raises(ValueError):
        benjamini_hochberg(p, 1.5)


# ---- permutation test --------------------------------------------------------
def _pooled(rho_old, rho_new, n=150, d=3, seed=0, df_new=None):
    x = simulate_regimes([Regime(n, rho_old), Regime(n, rho_new, df_new)], d, seed).to_numpy()
    return x, n


def test_statistics_match_direct_computation():
    x, n = _pooled(0.2, 0.7, seed=4)
    out = two_window_change_test(x, n, n_perm=19, q=0.1, seed=1)
    d = x.shape[1]
    tau = lambda w: np.array([[stats.kendalltau(w[:, i], w[:, j]).statistic for j in range(d)] for i in range(d)])
    np.fill_diagonal(t_old := tau(x[:n]), 0), np.fill_diagonal(t_new := tau(x[n:]), 0)
    assert out["stat_tau"] == pytest.approx(np.linalg.norm(t_old - t_new))
    # lower-tail coefficient with ranks taken within each window
    def lam(sl):
        u = stats.rankdata(x[sl], axis=0) / (len(x[sl]) + 1)
        m = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    joint = ((u[:, i] < .1) & (u[:, j] < .1)).sum()
                    m[i, j] = joint / (0.5 * ((u[:, i] < .1).sum() + (u[:, j] < .1).sum()))
        return m
    assert out["stat_lower"] == pytest.approx(np.linalg.norm(lam(slice(0, n)) - lam(slice(n, None))))


def test_no_change_gives_unremarkable_pvalues_and_change_is_detected():
    null_p = [two_window_change_test(*_pooled(0.5, 0.5, seed=s), n_perm=99)["p_tau"] for s in range(12)]
    assert np.mean(null_p) > 0.25 and sum(p < 0.05 for p in null_p) <= 2
    out = two_window_change_test(*_pooled(0.2, 0.8, seed=1), n_perm=99)
    assert out["p_tau"] == pytest.approx(0.01) and out["p_mean_tau"] == pytest.approx(0.01)  # = 1/(1+99)


def test_large_tail_increase_is_detected_by_tail_statistics():
    # independence -> Cauchy-type copula (df=1): tail coefficient at q=0.1 goes from ~0.1 to ~0.5.
    # (Moderate tail-only changes, e.g. Gaussian -> t(3) at equal tau, are NOT detectable with
    # 250-day windows; see the README.)
    x, n = _pooled(0.0, 0.5, n=250, d=4, seed=2, df_new=1.0)
    out = two_window_change_test(x, n, n_perm=99, q=0.1)
    assert out["p_mean_lower"] <= 0.02 and out["p_mean_upper"] <= 0.02


def test_tail_tests_are_not_fooled_by_volatility_clustering():
    """Constant copula, common stochastic volatility: tail p-values must stay unremarkable."""
    def sim(seed, n=400, d=4):
        rng = np.random.default_rng(seed)
        corr = np.full((d, d), 0.4) + 0.6 * np.eye(d)
        z = rng.multivariate_normal(np.zeros(d), corr, size=n)
        h = np.zeros(n)
        for t in range(1, n):
            h[t] = 0.98 * h[t - 1] + 0.15 * rng.standard_normal()
        return z * np.exp(h)[:, None]
    p = [two_window_change_test(sim(s), 200, n_perm=99, block=10)["p_mean_lower"] for s in range(16)]
    assert np.mean(np.array(p) < 0.05) <= 0.2


def test_block_permutation_and_effect_size_outputs():
    x, n = _pooled(0.2, 0.8, n=100, seed=3)
    out = two_window_change_test(x, n, n_perm=49, block=10)
    assert out["p_tau"] == pytest.approx(0.02) and out["stat_tau"] > 3 * out["null_mean_tau"]
    with pytest.raises(ValueError):
        two_window_change_test(x, n, n_perm=9, block=30)


def test_deterministic_and_validates_input():
    x, n = _pooled(0.3, 0.6)
    assert two_window_change_test(x, n, 29, seed=5) == two_window_change_test(x, n, 29, seed=5)
    with pytest.raises(ValueError):
        two_window_change_test(x, 10)
    with pytest.raises(ValueError):
        change_scan(pd.DataFrame(x[:50]), window=40)


def test_scan_responds_around_known_change_point():
    """Known change point (synthetic): the scan must react in the right place and not before."""
    W, T0 = 100, 400
    r = simulate_regimes([Regime(T0, 0.2), Regime(300, 0.8)], n_assets=3, seed=7)
    sc = change_scan(r, W, step=10, n_perm=99)
    dates = list(r.index)
    pre = sc.loc[: r.index[T0 - 1]]
    zone = sc.loc[r.index[T0 + 3 * W // 4]: r.index[T0 + 5 * W // 4]]  # windows split at the change
    after = sc.loc[r.index[T0 + 2 * W + 10]:]  # both windows entirely post-change
    assert zone["stat_tau"].min() > pre["stat_tau"].max()
    assert (zone["p_tau"] <= 0.02).all() and (pre["p_tau"] > 0.01).mean() > 0.9
    assert after["stat_tau"].max() < zone["stat_tau"].max()
    peak = sc["stat_tau"].idxmax()
    assert abs(dates.index(peak) - (T0 + W)) <= W // 2  # peak when the windows split at the change


# ---- scores from rolling vine fits -------------------------------------------
@pytest.fixture(scope="module")
def fit_results():
    W, T0 = 80, 300
    r = simulate_regimes([Regime(T0, 0.2), Regime(200, 0.8)], n_assets=3, seed=11)
    kw = dict(families=["indep", "gaussian", "clayton", "gumbel"], truncation_level=2, tail_simulations=2 ** 11)
    return RollingVineModel(W, 8, kw).run(r), W, 8, T0, r


def test_scores_react_to_known_change_with_disjoint_lag_only(fit_results):
    res, W, freq, T0, r = fit_results
    change_ts = r.index[T0]
    far = default_lag(W, freq)
    s_far = structural_change_scores(res, lag=far, baseline=20)
    s_one = structural_change_scores(res, lag=1, baseline=20)
    for sc in (s_far, s_one):
        assert {"structural_change_score", "dist_model", "n_aspects_flagged", "z_d_t", "cusum_d_t"} <= set(sc.columns)
    pre, post = s_far.loc[:change_ts - pd.Timedelta(days=1)], s_far.loc[change_ts:]
    assert post["s_tau"].max() > 2 * pre["s_tau"].max()
    assert post["dist_model"].max() > 2 * pre["dist_model"].max()
    # one-step comparisons of overlapping windows are far smaller than the disjoint-window shift
    assert s_one["s_tau"].max() < 0.5 * post["s_tau"].max()
    assert s_far["structural_change_score"].equals(s_far["s_tau"])


# ---- GARCH-filtered scan ---------------------------------------------------------------
from vine_risk.change_detection import change_scan_filtered
from vine_risk.garch import GarchMarginal
from vine_risk.synthetic import simulate_garch


def test_filtered_scan_equals_the_manual_computation_and_extends():
    r, _ = simulate_garch([Regime(180, 0.3), Regime(180, 0.8)], n_assets=3, seed=3)
    W = 100
    scan = change_scan_filtered(r, W, step=40, n_perm=29, block=10, seed=1)
    t = 2 * W - 1
    z = GarchMarginal().fit(r.iloc[t - 2 * W + 1: t + 1]).standardised_residuals().to_numpy()
    assert scan.iloc[0].to_dict() == two_window_change_test(z, W, 29, 0.1, 1, 10)  # the first scan date, by hand
    tail = change_scan_filtered(r, W, step=40, n_perm=29, block=10, seed=1, after=scan.index[1])
    pd.testing.assert_frame_equal(scan.loc[scan.index > scan.index[1]], tail)
    assert scan["p_tau"].iloc[-1] <= 0.1  # the residuals still carry the (large) change of correlation 0.3 -> 0.8
    with pytest.raises(ValueError, match="at least"):
        change_scan_filtered(r.iloc[:150], W)
