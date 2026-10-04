import json

import numpy as np
import pandas as pd
import pytest

from vine_risk.change_detection import change_scan, default_lag, structural_change_scores
from vine_risk.config import load_config
from vine_risk.copula import VineCopula, VineFitError
from vine_risk.dependence import dependence_metrics
from vine_risk.monitor import (
    ALERT, NORMAL, WARMING_UP, ConsoleSink, JsonlSink, LiveMonitor, MonitorConfig, alert_state_from_scans,
    next_alert_state, records_to_frame,
)
from vine_risk.portfolio import rolling_risk
from vine_risk.rolling import RollingVineModel
from vine_risk.synthetic import Regime, simulate_regimes

VK = dict(families=["indep", "gaussian", "clayton", "gumbel"], truncation_level=2, tail_simulations=2 ** 10)
# tiny setup (3 assets, 60-day windows): the permutation test is weak here, hence alpha = 5%
CFG = MonitorConfig(window=60, refit_frequency=5, vine_kwargs=VK, scan_step=10, n_perm=99, block=10, alpha=0.05,
                    risk_every_fits=3, risk_sims=2 ** 9)


@pytest.fixture(scope="module")
def data():
    return simulate_regimes([Regime(150, 0.2), Regime(110, 0.85)], n_assets=3, seed=21)


def _run(returns, cfg=CFG, sinks=()):
    m = LiveMonitor(list(returns.columns), cfg, sinks)
    for ts, row in returns.iterrows():
        m.on_return(ts, row)
    return m


@pytest.fixture(scope="module")
def live(data):
    return _run(data)


@pytest.fixture(scope="module")
def batch(data):
    return RollingVineModel(CFG.window, CFG.refit_frequency, VK).run(data)


def test_config_validation_and_from_config():
    with pytest.raises(ValueError):
        MonitorConfig(enter_ratio=1.0, exit_ratio=2.0)
    with pytest.raises(ValueError):
        MonitorConfig(alpha=1.5)
    with pytest.raises(ValueError):
        MonitorConfig(scan_step=0)
    m = MonitorConfig.from_config(load_config("configs/default.yaml"), scan_step=10)
    assert m.window == 250 and m.scan_step == 10 and m.vine_kwargs["truncation_level"] == 3 and m.seed == 42


# ---- the live loop reproduces the batch pipeline exactly (no look-ahead) -------------
def test_refit_schedule_matches_batch(live, batch):
    df = records_to_frame(live.records)
    refits = df.index[df["refit"]]
    assert list(refits.strftime("%Y-%m-%d")) == [r.timestamp for r in batch]
    assert (df.loc[~df["refit"], "fit_status"] == "none").all() and (df.loc[df["refit"], "fit_status"] == "ok").all()


def test_metrics_match_batch(live, batch):
    df = records_to_frame(live.records)
    want = dependence_metrics(batch)
    got = df.loc[df["refit"], [c for c in df if c.startswith("m_")]]
    got.columns = [c[2:] for c in got.columns]
    for col in ("d_t", "mean_lower_tail_q", "mean_upper_tail_q", "tail_asymmetry", "loglik", "bic",
                "mean_abs_dtau", "relationship_change", "family_change_frac"):
        pd.testing.assert_series_equal(got[col].dropna(), want[col].dropna(), check_names=False, rtol=1e-12)


def test_scores_match_batch(live, batch):
    df = records_to_frame(live.records)
    want = structural_change_scores(batch, lag=default_lag(CFG.window, CFG.refit_frequency), baseline=20)
    for mine, theirs in (("s_s_tau", "s_tau"), ("s_dist_model", "dist_model")):
        pd.testing.assert_series_equal(df[mine].dropna(), want[theirs].dropna(), check_names=False, rtol=1e-12)
    assert df["s_s_tau"].dropna().index[0] == want["s_tau"].dropna().index[0]  # not available earlier


def test_permutation_test_matches_batch_scan(live, data):
    df = records_to_frame(live.records)
    scan = change_scan(data, CFG.window, CFG.scan_step, CFG.n_perm, CFG.scan_q, CFG.seed, CFG.block)
    got = df[[c for c in df if c.startswith("scan_")]].dropna(how="all")
    assert list(got.index) == list(scan.index)
    for c in ("stat_tau", "p_tau", "p_mean_lower"):
        np.testing.assert_allclose(got[f"scan_{c}"], scan[c], rtol=1e-12)


def test_risk_matches_batch(live, data, batch):
    df = records_to_frame(live.records)
    want = rolling_risk(batch, data, n_sims=CFG.risk_sims, seed=CFG.seed, step=CFG.risk_every_fits)
    got = df[[c for c in df if c.startswith("risk_")]].dropna(how="all")
    assert list(got.index) == list(want.index)
    np.testing.assert_allclose(got["risk_es_vine"], want["es_vine"], rtol=1e-12)
    np.testing.assert_allclose(got["risk_var_hist"], want["var_hist"], rtol=1e-12)


def test_detects_the_known_change_but_not_before(live, data):
    df = records_to_frame(live.records)
    change = data.index[150]
    assert df.loc[:change, "new_alert"].sum() == 0
    first = df.index[df["new_alert"]]
    assert len(first) >= 1 and change < first[0] <= data.index[150 + CFG.window + 10]
    assert df.loc[first[0], "alert_state"] == ALERT and "differs" in df.loc[first[0], "message"]
    assert (df.loc[: data.index[2 * CFG.window - 2], "alert_state"] == WARMING_UP).all()


def test_no_lookahead_prefix_gives_identical_records(live, data):
    k = 200
    prefix = records_to_frame(_run(data.iloc[:k]).records)
    pd.testing.assert_frame_equal(prefix, records_to_frame(live.records).iloc[:k])


def test_resume_from_checkpoint_equals_continuous_run(live, data, batch):
    p = 175
    state = records_to_frame(live.records).iloc[p]["alert_state"]
    done = [r for r in batch if pd.Timestamp(r.timestamp) <= data.index[p]]
    m = LiveMonitor(list(data.columns), CFG)
    m.prime(data.iloc[: p + 1], done, alert_state=state)
    for ts, row in data.iloc[p + 1:].iterrows():
        m.on_return(ts, row)
    pd.testing.assert_frame_equal(records_to_frame(m.records), records_to_frame(live.records).iloc[p + 1:])


def test_prices_path_gives_same_records_as_returns_path(data):
    prices = 100 * np.exp(data.iloc[:140].cumsum())
    m = LiveMonitor(list(data.columns), CFG)
    outs = [m.on_prices(ts, row) for ts, row in prices.iterrows()]
    assert outs[0] is None and all(o is not None for o in outs[1:])
    rets = np.log(prices / prices.shift(1)).iloc[1:]
    pd.testing.assert_frame_equal(records_to_frame(m.records), records_to_frame(_run(rets).records))
    with pytest.raises(ValueError):
        m.on_prices(pd.Timestamp("2030-01-01"), prices.iloc[0] * -1)


# ---- alert logic -----------------------------------------------------------------------
def test_alert_hysteresis(monkeypatch):
    seq = iter([(0.50, 3.0),   # not significant, large effect  -> no alert
                (0.001, 1.2),  # significant but small effect   -> no alert
                (0.001, 2.5),  # significant and large          -> alert starts
                (0.001, 1.8),  # between exit and enter         -> alert continues
                (0.20, 1.6),   # not significant but >= exit    -> continues (no flicker)
                (0.001, 1.4),  # below exit                      -> ends
                (0.001, 2.1)])  # again                          -> new alert
    import vine_risk.monitor as mon

    def fake(x, n_old, n_perm, q, seed, block):
        p, ratio = next(seq)
        return {"stat_tau": ratio, "null_mean_tau": 1.0, "p_tau": p}

    monkeypatch.setattr(mon, "two_window_change_test", fake)
    cfg = MonitorConfig(window=40, refit_frequency=10_000, vine_kwargs=VK, scan_step=1, n_perm=9, block=1,
                        risk_every_fits=0)
    r = simulate_regimes([Regime(86, 0.5)], n_assets=3, seed=2)  # 2W = 80 -> 7 scans at rows 79..85
    df = records_to_frame(_run(r, cfg).records)
    scans = df.dropna(subset=["scan_ratio_tau"])
    assert list(scans["alert_state"]) == [NORMAL, NORMAL, ALERT, ALERT, ALERT, NORMAL, ALERT]
    assert list(scans["new_alert"]) == [False, False, True, False, False, False, True]
    assert (df.loc[df["scan_ratio_tau"].isna() & (df["position"] < 79), "alert_state"] == WARMING_UP).all()


def test_failed_fit_is_reported_not_raised(monkeypatch, data):
    def boom(self, u, returns=None):
        raise VineFitError("boom")
    monkeypatch.setattr(VineCopula, "fit", boom)
    m = _run(data.iloc[:80])
    df = records_to_frame(m.records)
    assert df["refit"].sum() > 0 and (df.loc[df["refit"], "fit_status"] == "failed").all()
    assert "boom" in df.loc[df["refit"], "message"].iloc[0] and not [c for c in df if c.startswith("m_")]


# ---- inputs and outputs ------------------------------------------------------------------
def test_input_validation(data):
    m = LiveMonitor(list(data.columns), CFG)
    ts, row = data.index[0], data.iloc[0]
    m.on_return(ts, row)
    with pytest.raises(ValueError):
        m.on_return(data.index[1], row.where(row < 0, np.nan))  # missing values
    with pytest.raises(ValueError):
        m.on_return(ts, row)  # not after the previous timestamp
    with pytest.raises(ValueError):
        LiveMonitor(["A", "B"], CFG).prime(data.iloc[:100], [])  # columns do not match
    m2 = LiveMonitor(list(data.columns), CFG)
    with pytest.raises(ValueError):
        m2.prime(data.iloc[:30])  # shorter than the window


def test_prime_rejects_results_from_the_future(batch, data):
    m = LiveMonitor(list(data.columns), CFG)
    with pytest.raises(ValueError, match="look-ahead"):
        m.prime(data.iloc[:100], batch)


def test_sinks(tmp_path, data):
    log, lines = tmp_path / "live.jsonl", []
    _run(data.iloc[:140], sinks=[JsonlSink(log), ConsoleSink(every=1000, out=lines.append)])
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 140 and rows[0]["timestamp"] == str(data.index[0].date())
    assert any(r["new_alert"] for r in rows) == any("ALERT" in l for l in lines)


def test_next_alert_state_rule():
    c = MonitorConfig(alpha=0.01, enter_ratio=2.0, exit_ratio=1.5)
    assert next_alert_state(NORMAL, 0.001, 2.5, c) == (ALERT, True, next_alert_state(NORMAL, 0.001, 2.5, c)[2])
    assert next_alert_state(NORMAL, 0.05, 9.0, c)[:2] == (NORMAL, False)  # large but not significant
    assert next_alert_state(ALERT, 0.5, 1.6, c)[:2] == (ALERT, False)  # stays while above the exit level
    assert next_alert_state(ALERT, 0.001, 1.4, c) == (NORMAL, False, "alert ended")
    assert next_alert_state(WARMING_UP, 0.5, 0.7, c)[0] == NORMAL
    assert next_alert_state(ALERT, 0.001, float("nan"), c)[0] == NORMAL  # undefined effect size ends it


def test_state_restored_from_stored_scans_equals_continuous_state(live, data):
    scan = change_scan(data, CFG.window, CFG.scan_step, CFG.n_perm, CFG.scan_q, CFG.seed, CFG.block)
    df = records_to_frame(live.records)
    for t in (data.index[199], data.index[209], data.index[229], data.index[259]):
        assert alert_state_from_scans(scan, CFG, t) == df.loc[t, "alert_state"]
    assert alert_state_from_scans(scan.iloc[:0], CFG) == WARMING_UP
