import numpy as np
import pandas as pd
import pytest

from vine_risk.config import load_config
from vine_risk.data import clean_prices
from vine_risk.returns import build_return_matrix, log_returns


def _prices():
    idx = pd.bdate_range("2020-01-01", periods=4)
    return pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0], "B": [50.0, 50.0, 55.0, 60.5]}, index=idx)


def test_log_returns_known_values():
    r = log_returns(_prices())
    assert r.shape == (3, 2)
    assert r["A"].iloc[0] == pytest.approx(np.log(1.1))
    assert r["A"].iloc[1] == pytest.approx(np.log(0.9))
    assert r["B"].iloc[0] == 0.0


def test_log_returns_rejects_nonpositive():
    p = _prices()
    p.iloc[1, 0] = 0.0
    with pytest.raises(ValueError):
        log_returns(p)


def test_log_returns_additive():
    p = _prices()
    assert log_returns(p).sum()["A"] == pytest.approx(np.log(p["A"].iloc[-1] / p["A"].iloc[0]))


def test_clean_drops_sparse_ticker_and_fills_short_gap():
    idx = pd.bdate_range("2020-01-01", periods=20)
    df = pd.DataFrame({"A": np.linspace(10, 20, 20), "B": np.linspace(5, 6, 20), "C": np.nan}, index=idx)
    df.loc[idx[5], "B"] = np.nan
    out = clean_prices(df, max_missing_frac=0.1, max_ffill_days=3)
    assert list(out.columns) == ["A", "B"]
    assert len(out) == 20
    assert out["B"].iloc[5] == out["B"].iloc[4]


def test_clean_starts_when_youngest_asset_exists():
    idx = pd.bdate_range("2020-01-01", periods=100)
    df = pd.DataFrame({"A": 10.0, "B": 5.0}, index=idx)
    df.loc[idx[:3], "B"] = np.nan
    out = clean_prices(df)
    assert out.index[0] == idx[3]


def test_build_return_matrix_is_finite():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(100 * np.exp(rng.normal(0, 0.01, (50, 3)).cumsum(0)), index=idx, columns=list("XYZ"))
    r = build_return_matrix(df)
    assert r.shape == (49, 3)
    assert np.isfinite(r.to_numpy()).all()


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.assets[0] == "AAPL" and len(cfg.assets) == 8
    assert cfg.rolling.window == 250
    assert cfg.risk.confidence_level == 0.99
