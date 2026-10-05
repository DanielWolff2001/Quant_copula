import numpy as np
import pandas as pd
import pytest

from vine_risk.synthetic import Regime
from vine_risk.validation import (
    Experiment, accuracy_summary, add_flags, conditional_accuracy, conditional_accuracy_summary, detection_summary,
    estimation_accuracy, null_threshold,
    run_scan_experiment, run_score_experiment, standard_experiments, zone_of,
)


def test_standard_experiments_definition():
    ex = standard_experiments(1700, 900)
    assert set(ex) == {"A_const", "A_vol", "B_corr", "C_tail_moderate", "C_tail_strong"}
    assert all(e.n_obs == 1700 for e in ex.values())
    assert ex["A_const"].change_at is None and ex["B_corr"].change_at == 900
    assert ex["A_vol"].vol_phi is not None and ex["A_const"].vol_phi is None
    # the tail experiments change the copula family but not rho (hence not Kendall's tau)
    for name in ("C_tail_moderate", "C_tail_strong"):
        a, b = ex[name].regimes
        assert a.rho == b.rho and a.df is None and b.df is not None and b.tail_dependence > 0


def test_zones():
    c, w = 900, 250
    assert zone_of(10, None, w) == "null"
    assert zone_of(899, c, w) == "pre" and zone_of(900, c, w) == "transition"
    assert zone_of(c + 125, c, w) == "change" and zone_of(c + 375, c, w) == "change"
    assert zone_of(c + 376, c, w) == "transition" and zone_of(c + 499, c, w) == "transition"
    assert zone_of(c + 500, c, w) == "post"


def _table():
    # 2 replications; change at 100, window 40: change zone is pos 120..160
    rows = []
    for rep, hits in ((0, {120, 130, 30}), (1, {200})):
        for pos in range(10, 260, 10):
            rows.append({"rep": rep, "pos": pos, "zone": zone_of(pos, 100, 40), "flag_x": pos in hits})
    return pd.DataFrame(rows)


def test_detection_summary_arithmetic():
    s = detection_summary(_table(), change_at=100, window=40).loc["x"]
    # null dates (pos<100 or >=180): rep0 pos 10..90 (9) + 180..250 (8) = 17; rep1 same -> 34; hits: rep0 pos30, rep1 pos200
    assert s["false_alarm"] == pytest.approx(2 / 34)
    # change zone positions 120,130,140,150,160 (5 per rep): rep0 hits 120,130
    assert s["power"] == pytest.approx(2 / 10)
    assert s["detected"] == pytest.approx(0.5)
    assert s["median_delay"] == pytest.approx(20.0)  # rep0: first alert at 120 -> 20; rep1 never in [100, 180]


def test_null_threshold_uses_only_null_dates():
    df = pd.DataFrame({"zone": ["pre", "pre", "change", "post"], "s": [1.0, 3.0, 100.0, 2.0]})
    assert null_threshold(df, "s", 0.5) == pytest.approx(2.0)


def test_add_flags():
    df = pd.DataFrame({"a": [1, 2]})
    out = add_flags(df, {"hi": df["a"] > 1})
    assert out["flag_hi"].tolist() == [False, True] and "flag_hi" not in df


def test_scan_experiment_end_to_end_small():
    W = 80
    small = {
        "none": Experiment("none", (Regime(480, 0.5),)),
        "jump": Experiment("jump", (Regime(240, 0.2), Regime(240, 0.85))),
    }
    summ = {}
    for name, exp in small.items():
        df = run_scan_experiment(exp, [0, 1], window=W, step=10, n_perm=199, block=10, n_assets=3)
        assert {"rep", "pos", "zone", "p_tau", "stat_tau", "experiment"} <= set(df.columns)
        summ[name] = detection_summary(add_flags(df, {"tau": df["p_tau"] <= 0.01}), exp.change_at, W).loc["tau"]
    assert summ["none"]["false_alarm"] <= 0.1
    assert summ["jump"]["power"] >= 0.8 and summ["jump"]["false_alarm"] <= 0.2
    assert 0 <= summ["jump"]["median_delay"] <= 2 * W


def test_score_experiment_end_to_end_small():
    exp = Experiment("jump", (Regime(200, 0.2), Regime(160, 0.85)))
    df = run_score_experiment(exp, [3], window=60, refit_frequency=10, n_assets=3,
                              vine_kwargs=dict(families=["indep", "gaussian", "clayton"], truncation_level=2,
                                               tail_simulations=2 ** 10))
    assert {"s_tau", "s_tau_lag1", "dist_model", "zone"} <= set(df.columns)
    assert df.loc[df.zone == "change", "s_tau"].max() > 2 * df.loc[df.zone == "pre", "s_tau"].max()
    assert df.loc[df.zone == "change", "s_tau_lag1"].max() < df.loc[df.zone == "change", "s_tau"].max()


def test_estimation_accuracy_small():
    df = estimation_accuracy(Regime(1, 0.5, 3.0), windows=(150,), n_reps=3, n_assets=3, n_sims=2 ** 11,
                             n_truth=50_000)
    assert len(df) == 3 and (df["true_es"] > df["true_var"]).all()
    s = accuracy_summary(df)
    assert set(s.window) == {150} and {"ES vine", "VaR hist", "lower_tail_q (vine)"} <= set(s.estimator)
    assert np.isfinite(s[["bias_pct", "rmse_pct"]].to_numpy()).all()
    assert (s.rmse_pct >= s.bias_pct.abs() - 1e-9).all()


def test_accuracy_study_includes_the_standard_models():
    df = estimation_accuracy(Regime(1, 0.5), windows=(150,), n_reps=2, n_assets=3, n_sims=2 ** 10, n_truth=20_000)
    for col in ("var_ewma_n", "es_ewma_n", "var_ewma_t", "es_ewma_t", "var_fhs", "es_fhs"):
        assert col in df.columns and (df[col] > 0).all()
    est = set(accuracy_summary(df).estimator)
    assert {"VaR ewma_n", "ES ewma_t", "ES fhs", "VaR vine"} <= est


def test_conditional_accuracy_small_and_deterministic():
    kw = dict(n_assets=3, n_obs=420, n_origins=3, window=100, lookback=200, n_truth=20_000, n_sims=2 ** 9, seed=2, n_series=2)
    df = conditional_accuracy(Regime(1, 0.5, 3.0), **kw)
    from vine_risk.benchmark import MODELS
    assert len(df) == 2 * 3 * len(MODELS) and set(df["model"]) == set(MODELS) and set(df["series"]) == {0, 1}
    assert (df["true_es"] > df["true_var"]).all() and (df["true_var"] > 0).all() and (df["var"] > 0).all()
    pd.testing.assert_frame_equal(df, conditional_accuracy(Regime(1, 0.5, 3.0), **kw))
    # the truth is the same for every model at an origin, and varies across origins (volatility clustering)
    assert df.groupby(["series", "origin"])["true_var"].nunique().eq(1).all() and df["true_var"].nunique() > 2
    summ = conditional_accuracy_summary(df)
    assert set(summ.index) == set(MODELS) and (summ["n"] == 6).all()
    assert summ["tracking_corr"].between(-1, 1).all()


def test_conditional_models_track_changing_risk_better_than_unconditional_ones():
    df = conditional_accuracy(Regime(1, 0.5), n_assets=3, n_obs=900, n_origins=8, window=200, lookback=400,
                              n_truth=50_000, n_sims=2 ** 10, seed=5, n_series=3)
    track = conditional_accuracy_summary(df)["tracking_corr"]
    conditional = track[["ewma_n", "ewma_t", "fhs", "vine_garch", "gauss_garch"]].mean()
    unconditional = track[["hist", "vine_emp", "gauss_emp"]].mean()
    assert conditional > 0.9 and conditional > unconditional + 0.2
