import numpy as np
import pandas as pd
import pytest

from vine_risk.config import load_config
from vine_risk.copula import VineCopula, VineFitError
from vine_risk.rolling import RollingVineModel, fit_window, results_to_frames, save_results

FAST = dict(families=["indep", "gaussian", "clayton"], truncation_level=2)
W = 60


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(5)
    n = 100
    z = rng.standard_normal((n, 3))
    z[:, 1] += 0.8 * z[:, 0]
    return pd.DataFrame(z * 0.01, index=pd.bdate_range("2021-01-01", periods=n), columns=list("ABC"))


def _model(freq=1):
    return RollingVineModel(window=W, refit_frequency=freq, vine_kwargs=FAST)


def test_daily_schedule_and_window_metadata(returns):
    res = _model().run(returns)
    assert len(res) == len(returns) - W + 1
    assert res[0].timestamp == str(returns.index[W - 1].date())
    assert res[-1].timestamp == str(returns.index[-1].date())
    for r in res:
        assert r.n_obs == W and r.n_assets == 3 and r.status == "ok"
    assert res[3].window_start == str(returns.index[3].date())


def test_periodic_refit_schedule(returns):
    res = _model(freq=5).run(returns)
    expected = list(range(W - 1, len(returns), 5))
    assert [r.timestamp for r in res] == [str(returns.index[p].date()) for p in expected]


def test_no_lookahead(returns):
    full = _model().run(returns)
    cut = _model().run(returns.iloc[:80])
    assert [r.to_json() for r in full[: len(cut)]] == [r.to_json() for r in cut]
    changed = returns.copy()
    changed.iloc[85:] *= 5.0  # perturb the future only
    alt = _model().run(changed)
    assert alt[10].to_json() == full[10].to_json()


def test_stepwise_equals_batch(returns):
    for freq in (1, 7):
        batch = _model(freq).run(returns)
        live = _model(freq)
        out = [r for ts, row in returns.iterrows() if (r := live.step(ts, row)) is not None]
        assert [r.to_json() for r in out] == [r.to_json() for r in batch]


def test_parallel_matches_serial(returns):
    sub = returns.iloc[:70]
    serial = _model().run(sub, n_jobs=1)
    par = _model().run(sub, n_jobs=2)
    assert [r.to_json() for r in par] == [r.to_json() for r in serial]


def test_checkpoint_resume(returns, tmp_path, monkeypatch):
    ck = tmp_path / "ck.jsonl"
    first = _model().run(returns.iloc[:75], checkpoint=ck)
    assert len(ck.read_text().splitlines()) == len(first)
    calls = []
    import vine_risk.rolling as rolling
    orig = rolling.fit_window
    monkeypatch.setattr(rolling, "fit_window", lambda *a, **k: calls.append(1) or orig(*a, **k))
    full = _model().run(returns, checkpoint=ck)
    assert len(calls) == len(returns) - 75
    assert len(full) == len(returns) - W + 1
    assert [r.to_json() for r in full[: len(first)]] == [r.to_json() for r in first]


def test_failed_fit_is_recorded_not_raised(returns, monkeypatch):
    def boom(self, u, returns=None):
        raise VineFitError("boom")
    monkeypatch.setattr(VineCopula, "fit", boom)
    r = fit_window(returns.iloc[:W], FAST)
    assert r.status == "failed" and "boom" in r.message and r.timestamp == str(returns.index[W - 1].date())


def test_validation(returns):
    with pytest.raises(ValueError):
        RollingVineModel(window=1)
    with pytest.raises(ValueError):
        RollingVineModel(refit_frequency=0)
    with pytest.raises(ValueError):
        _model().run(returns.iloc[: W - 1])
    bad = returns.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError):
        _model().run(bad)
    m = _model()
    m.push(returns.index[5], returns.iloc[5])
    with pytest.raises(ValueError):
        m.push(returns.index[4], returns.iloc[4])
    with pytest.raises(RuntimeError):
        m.refit()


def test_frames_and_parquet(returns, tmp_path):
    res = _model(freq=10).run(returns)
    frames = results_to_frames(res)
    assert len(frames["fits"]) == len(res)
    assert len(frames["pair_copulas"]) == len(res) * len(res[0].pair_copulas)
    assert len(frames["pairwise"]) == len(res) * 3
    save_results(res, tmp_path)
    back = pd.read_parquet(tmp_path / "pair_copulas.parquet")
    assert len(back) == len(frames["pair_copulas"])
    assert set(back["tree"]) == {1, 2}


def test_from_config_uses_truncation_three():
    m = RollingVineModel.from_config(load_config("configs/default.yaml"))
    assert m.window == 250 and m.refit_frequency == 1
    assert m.vine_kwargs == {"selection_criterion": "bic", "truncation_level": 3,
                             "tail_simulations": 16384, "tail_level": 0.05, "seed": 42}
