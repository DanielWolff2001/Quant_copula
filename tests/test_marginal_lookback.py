import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vine_risk.config import Config, RollingConfig, RiskConfig
from vine_risk.copula import VineCopula
from vine_risk.garch import GarchMarginal
from vine_risk.manifest import ManifestMismatch, check_resume, fit_fingerprint, read_manifest, verify_history_unchanged
from vine_risk.marginals import EmpiricalMarginal, MarginalSpec
from vine_risk.rolling import RollingVineModel, fit_window
from vine_risk.runner import fit_rolling
from vine_risk.synthetic import Regime, simulate_garch

W, L = 60, 150
VK = dict(families=["indep", "gaussian", "clayton", "gumbel"], truncation_level=2, tail_simulations=2 ** 9)
CFG = Config(assets=["X1", "X2", "X3"],
             rolling=RollingConfig(window=W, refit_frequency=10, truncation_level=2, n_jobs=1, tail_simulations=2 ** 9,
                                   marginal="garch_t", marginal_lookback=L),
             risk=RiskConfig(simulations=512, seed=3))


@pytest.fixture(scope="module")
def returns():
    r, _ = simulate_garch([Regime(160, 0.2), Regime(140, 0.8)], n_assets=3, seed=12)
    return r


def _model(freq=10):
    return RollingVineModel(W, freq, VK, MarginalSpec("garch_t"), L)


def test_config_lookback_defaults_and_validation():
    assert RollingConfig(window=250).lookback == 250  # rank marginals: the window itself
    assert RollingConfig(window=250, marginal="garch_t").lookback == 1000
    assert RollingConfig(window=1500, marginal="garch_t").lookback == 1500
    assert RollingConfig(window=250, marginal="garch_t", marginal_lookback=400).lookback == 400
    with pytest.raises(ValueError, match="marginal_lookback"):
        RollingConfig(window=250, marginal_lookback=100)
    with pytest.raises(ValueError, match="rolling.marginal"):
        RollingConfig(marginal="kernel")
    with pytest.raises(ValueError, match="marginal_lookback"):
        RollingVineModel(100, 1, marginal_lookback=50)


def test_fit_window_fits_the_marginal_on_the_long_history_and_the_vine_on_the_window(returns):
    hist = returns.iloc[:L]
    res = fit_window(hist, VK, MarginalSpec("garch_t"), W)
    assert res.n_obs == W and res.window_end == str(hist.index[-1].date()) and res.window_start == str(hist.index[-W].date())
    u = GarchMarginal().fit(hist).transform(hist.iloc[-W:])  # what the function is supposed to do
    manual = VineCopula(**VK).fit(u, hist.iloc[-W:]).summary()
    assert res.to_json() == manual.to_json()
    # a marginal fitted on a shorter history gives different pseudo-observations, hence a different fit
    short = fit_window(hist.iloc[-100:], VK, MarginalSpec("garch_t"), W)
    assert short.to_json() != res.to_json()
    with pytest.raises(ValueError, match="at least"):
        fit_window(hist.iloc[:W - 1], VK, MarginalSpec(), W)


def test_default_behaviour_without_copula_window_is_unchanged(returns):
    w = returns.iloc[:W]
    a = fit_window(w, VK, EmpiricalMarginal)
    b = fit_window(w, VK, EmpiricalMarginal, W)
    assert a.to_json() == b.to_json()


def test_schedule_starts_after_the_lookback(returns):
    m = _model()
    assert m.refit_positions(len(returns))[0] == L - 1
    res = m.run(returns)
    assert res[0].timestamp == str(returns.index[L - 1].date()) and all(r.n_obs == W for r in res)
    with pytest.raises(ValueError, match=str(L)):
        _model().run(returns.iloc[: L - 1])


def test_stepwise_batch_and_parallel_agree_with_a_lookback(returns):
    sub = returns.iloc[:230]
    batch = _model().run(sub)
    parallel = _model().run(sub, n_jobs=2)
    assert [r.to_json() for r in batch] == [r.to_json() for r in parallel]
    live = _model()
    done = [r for r in batch if pd.Timestamp(r.timestamp) <= sub.index[179]]
    live.prime(sub.iloc[:180], done)
    new = [x for ts, row in sub.iloc[180:].iterrows() if (x := live.step(ts, row)) is not None]
    assert [r.to_json() for r in new] == [r.to_json() for r in batch[len(done):]]
    with pytest.raises(ValueError, match="at least"):
        _model().prime(sub.iloc[: L - 1])


def test_no_lookahead_with_garch_marginals(returns):
    full = _model().run(returns.iloc[:260])
    cut = _model().run(returns.iloc[:200])
    assert [r.to_json() for r in full[: len(cut)]] == [r.to_json() for r in cut]
    changed = returns.iloc[:260].copy()
    changed.iloc[230:] *= 4.0  # perturb the future only
    alt = _model().run(changed)
    assert alt[0].to_json() == full[0].to_json() and alt[len(cut) - 1].to_json() == full[len(cut) - 1].to_json()


def test_fit_fingerprint_and_resume_checks_include_the_marginal(returns, tmp_path):
    fp = fit_fingerprint(CFG)
    assert fp["marginal"] == "garch_t" and fp["marginal_lookback"] == L
    assert fit_fingerprint(replace(CFG, rolling=replace(CFG.rolling, marginal="garch_empirical"))) != fp
    assert fit_fingerprint(replace(CFG, rolling=replace(CFG.rolling, marginal_lookback=200))) != fp
    fit_rolling(CFG, returns.iloc[:200], tmp_path)
    with pytest.raises(ManifestMismatch, match="marginal: stored 'garch_t', now 'empirical'"):
        check_resume(tmp_path, replace(CFG, rolling=replace(CFG.rolling, marginal="empirical", marginal_lookback=None)))


def test_runs_made_before_garch_marginals_stay_compatible(returns, tmp_path):
    cfg = replace(CFG, rolling=replace(CFG.rolling, marginal="empirical", marginal_lookback=None))
    fit_rolling(cfg, returns.iloc[:200], tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    for k in ("marginal", "marginal_lookback"):
        m["fit_fingerprint"].pop(k)  # what an old manifest looks like
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert check_resume(tmp_path, cfg) == []
    with pytest.raises(ManifestMismatch, match="marginal"):
        check_resume(tmp_path, CFG)


def test_runner_resumes_and_detects_revisions_with_garch_marginals(returns, tmp_path):
    fit_rolling(CFG, returns.iloc[:220], tmp_path)
    again = fit_rolling(CFG, returns.iloc[:260], tmp_path)  # extends
    assert read_manifest(tmp_path)["steps"]["rolling"]["fits"] == len(again) and len(again) > 8
    revised = returns.iloc[:260].copy()
    revised.iloc[100:130, 0] = revised.iloc[100:130, 0].to_numpy()[::-1]
    with pytest.raises(ManifestMismatch, match="changed"):
        fit_rolling(CFG, revised, tmp_path)
    latest = again[-1]
    verify_history_unchanged(latest, returns, marginal_factory=MarginalSpec("garch_t"), lookback=L)
    with pytest.raises(ManifestMismatch):
        verify_history_unchanged(latest, revised, marginal_factory=MarginalSpec("garch_t"), lookback=L)
