import numpy as np
import pandas as pd
import pytest

from vine_risk.config import Config, DataConfig
from vine_risk.pipeline import find_run, load_prices_and_returns


def test_load_prices_and_returns_from_cache(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=30)
    prices = pd.DataFrame({"A": 100 * np.exp(np.linspace(0, 0.3, 30)), "B": np.linspace(50, 60, 30)}, index=idx)
    prices.to_parquet(tmp_path / "prices_A_B_2020-01-01_latest.parquet")
    cfg = Config(assets=["A", "B"], data=DataConfig(start="2020-01-01", cache_dir=str(tmp_path)))
    p, r = load_prices_and_returns(cfg)
    assert p.shape == (30, 2) and r.shape == (29, 2)
    assert r["A"].iloc[0] == pytest.approx(0.3 / 29)


def test_find_run_prefers_w250_then_demo(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_rolling"):
        find_run(tmp_path)
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "checkpoint.jsonl").write_text("{}\n")
    assert find_run(tmp_path).name == "demo"
    (tmp_path / "w250").mkdir()
    (tmp_path / "w250" / "checkpoint.jsonl").write_text("{}\n")
    assert find_run(tmp_path).name == "w250"
