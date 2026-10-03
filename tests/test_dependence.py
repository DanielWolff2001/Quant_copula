import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vine_risk.dependence import pairwise_dependence


def _data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    z[:, 1] += 0.7 * z[:, 0]
    r = pd.DataFrame(z, columns=list("ABC"))
    u = r.rank() / (n + 1)
    return u, r


def test_matches_scipy_and_numpy():
    u, r = _data()
    out = pairwise_dependence(u, r).set_index(["asset_i", "asset_j"])
    assert len(out) == 3
    assert out.loc[("A", "B"), "tau"] == pytest.approx(stats.kendalltau(r["A"], r["B"]).statistic)
    assert out.loc[("A", "B"), "spearman"] == pytest.approx(stats.spearmanr(r["A"], r["B"]).statistic)
    assert out.loc[("A", "B"), "pearson"] == pytest.approx(np.corrcoef(r["A"], r["B"])[0, 1])


def test_pearson_nan_without_returns_and_bounds():
    u, _ = _data()
    out = pairwise_dependence(u)
    assert out["pearson"].isna().all()
    assert out[["tau", "spearman"]].abs().max().max() <= 1


def test_column_mismatch_raises():
    u, r = _data()
    with pytest.raises(ValueError):
        pairwise_dependence(u, r[["B", "A", "C"]])
