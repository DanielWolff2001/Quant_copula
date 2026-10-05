import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from vine_risk.cli import render_schedule
from vine_risk.config import DataConfig
from vine_risk.data import DataIssue, download_prices, price_cache_path, validate_prices
from vine_risk.locking import RunLockError, run_lock
from vine_risk.sources import CsvSource, PriceSource, YahooSource, make_source


def _prices(n=40, cols=("A", "B")):
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for c in cols}, index=idx)


# ---- sources -------------------------------------------------------------------
def test_csv_wide_file(tmp_path):
    p = _prices(cols=("A", "B", "C"))
    p.to_csv(tmp_path / "px.csv")
    src = CsvSource(tmp_path / "px.csv")
    got = src.fetch(["B", "A"], "2021-01-10", "2021-02-12")
    assert list(got.columns) == ["B", "A"] and got.index[0] >= pd.Timestamp("2021-01-10") and got.index[-1] <= pd.Timestamp("2021-02-12")
    pd.testing.assert_frame_equal(got, p.loc[got.index, ["B", "A"]].rename_axis("Date"), check_freq=False)
    with pytest.raises(RuntimeError, match="no column"):
        src.fetch(["A", "Z"], "2021-01-01")
    with pytest.raises(RuntimeError, match="no prices between"):
        src.fetch(["A"], "2030-01-01")


def test_csv_folder_prefers_adjusted_close(tmp_path):
    p = _prices()
    for t in ("A", "B"):
        pd.DataFrame({"Close": p[t] * 2, "Adj Close": p[t]}).rename_axis("Date").to_csv(tmp_path / f"{t}.csv")
    got = CsvSource(tmp_path).fetch(["A", "B"], "2021-01-01")
    pd.testing.assert_frame_equal(got, p.rename_axis("Date"), check_freq=False)
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "C.csv")
    with pytest.raises(RuntimeError, match="none of the price columns"):
        CsvSource(tmp_path).fetch(["C"], "2021-01-01")
    with pytest.raises(RuntimeError, match=r"no file Z\.csv"):
        CsvSource(tmp_path).fetch(["Z"], "2021-01-01")
    with pytest.raises(FileNotFoundError):
        CsvSource(tmp_path / "nowhere").fetch(["A"], "2021-01-01")


def test_make_source():
    assert isinstance(make_source(DataConfig()), YahooSource)
    assert isinstance(make_source(DataConfig(source="csv", csv_path="x.csv")), CsvSource)
    with pytest.raises(ValueError, match="csv_path"):
        make_source(DataConfig(source="csv"))
    with pytest.raises(ValueError, match="data.source"):
        DataConfig(source="bloomberg")
    assert isinstance(YahooSource(), PriceSource) and isinstance(CsvSource("x"), PriceSource)


class FakeSource:
    name, cacheable = "fake", True

    def __init__(self, prices):
        self.prices, self.calls = prices, 0

    def fetch(self, tickers, start, end=None):
        self.calls += 1
        return self.prices[list(tickers)]


def test_download_uses_cache_until_refreshed(tmp_path):
    src = FakeSource(_prices(30))
    a = download_prices(["A", "B"], "2021-01-01", None, tmp_path, source=src)
    b = download_prices(["A", "B"], "2021-01-01", None, tmp_path, source=src)
    assert src.calls == 1 and a.equals(b)
    src.prices = _prices(35)
    c = download_prices(["A", "B"], "2021-01-01", None, tmp_path, source=src)
    assert len(c) == 30 and src.calls == 1  # still the cached version: the cache never refreshes by itself
    d = download_prices(["A", "B"], "2021-01-01", None, tmp_path, source=src, refresh=True)
    assert len(d) == 35 and src.calls == 2
    assert len(pd.read_parquet(price_cache_path(["A", "B"], "2021-01-01", None, tmp_path))) == 35


def test_csv_source_is_never_cached(tmp_path):
    p = _prices(); p.to_csv(tmp_path / "px.csv")
    assert download_prices(["A"], "2021-01-01", None, tmp_path / "cache", source=CsvSource(tmp_path / "px.csv")).shape == (40, 1)
    assert not (tmp_path / "cache").exists()


# ---- validation ----------------------------------------------------------------
def test_validate_prices_checks():
    p = _prices(60)
    assert validate_prices(p, ["A", "B"]) == []
    assert validate_prices(p.iloc[:0], ["A"])[0].level == "error"
    assert "no prices for" in validate_prices(p, ["A", "Z"])[0].message
    bad = p.copy(); bad.iloc[-1, 0] = -1.0
    assert any(i.level == "error" and "non-positive" in i.message for i in validate_prices(bad, ["A", "B"], new_from=p.index[-5]))
    assert validate_prices(bad, ["A", "B"], new_from=p.index[-1] + pd.Timedelta(days=1)) == []  # only new days are judged
    gap = p.copy(); gap.iloc[-2, 1] = np.nan
    assert [i.level for i in validate_prices(gap, ["A", "B"], new_from=p.index[-5])] == ["warning"]
    jump = p.copy(); jump.iloc[-3:, 0] *= 0.5  # a 50 % drop on one day, e.g. an unadjusted split
    msgs = validate_prices(jump, ["A", "B"], new_from=p.index[-5])
    assert len(msgs) == 1 and "unadjusted split" in msgs[0].message and msgs[0].level == "warning"
    stale = validate_prices(p, ["A", "B"], today=p.index[-1] + pd.Timedelta(days=20))
    assert stale and "stalled" in stale[0].message
    assert validate_prices(p, ["A", "B"], today=p.index[-1] + pd.Timedelta(days=3)) == []
    assert isinstance(msgs[0], DataIssue)


# ---- run lock ------------------------------------------------------------------
def test_lock_is_exclusive_across_processes_and_released(tmp_path):
    with run_lock(tmp_path) as lock:
        assert lock.read_text() == str(os.getpid())
    assert not (tmp_path / ".lock").exists()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / ".lock").write_text(str(other.pid))
        with pytest.raises(RunLockError, match=str(other.pid)):
            with run_lock(tmp_path):
                pass
        assert (tmp_path / ".lock").exists()  # the other owner's lock is left alone
    finally:
        other.kill(); other.wait()
    (tmp_path / ".lock").write_text(str(other.pid))  # now a dead process: stale, taken over
    with run_lock(tmp_path):
        assert (tmp_path / ".lock").read_text() == str(os.getpid())
    (tmp_path / ".lock").write_text("garbage")
    with run_lock(tmp_path):
        pass


# ---- scheduler text -------------------------------------------------------------
def test_render_schedule():
    cron = render_schedule("cron", "/venv/bin/vine-risk update", "/proj", "22:15")
    assert "15 22 * * 1-5 cd /proj && /venv/bin/vine-risk update >> data/update.log 2>&1" in cron
    plist = render_schedule("launchd", "/venv/bin/vine-risk update --config /c.yaml", "/proj", "06:05")
    assert plist.count("<key>Weekday</key>") == 5 and "<string>/venv/bin/vine-risk</string>" in plist
    assert "<integer>6</integer>" in plist and "<integer>5</integer>" in plist and "WorkingDirectory" in plist
    assert "OnCalendar=Mon..Fri 23:30" in render_schedule("systemd", "vine-risk update", "/proj")
    for bad in (("cron", "x", "/p", "25:00"), ("cron", "x", "/p", "9"), ("windows", "x", "/p", "09:00")):
        with pytest.raises(ValueError):
            render_schedule(*bad)
