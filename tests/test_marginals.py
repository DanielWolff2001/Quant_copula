import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from vine_risk.marginals import EmpiricalMarginal, uniformity_report


def _returns(n=300, d=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.standard_t(4, (n, d)) * 0.01, index=idx, columns=list("ABC")[:d])


def test_ranks_known_values():
    df = pd.DataFrame({"A": [0.3, -0.1, 0.2, 0.0]})
    u = EmpiricalMarginal().fit_transform(df)
    assert u["A"].tolist() == pytest.approx([4 / 5, 1 / 5, 3 / 5, 2 / 5])


def test_strictly_inside_unit_interval():
    u = EmpiricalMarginal().fit_transform(_returns())
    assert ((u > 0) & (u < 1)).all().all()


def test_matches_scipy_rank_with_ties():
    df = pd.DataFrame({"A": [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]})
    u = EmpiricalMarginal().fit_transform(df)
    expected = rankdata(df["A"]) / (len(df) + 1)
    assert u["A"].to_numpy() == pytest.approx(expected)


def test_preserves_order_and_index():
    r = _returns()
    u = EmpiricalMarginal().fit_transform(r)
    assert u.index.equals(r.index)
    for c in r:
        assert (np.argsort(r[c].to_numpy(), kind="stable") == np.argsort(u[c].to_numpy(), kind="stable")).all()


def test_roughly_uniform():
    u = EmpiricalMarginal().fit_transform(_returns(n=2000))
    assert (uniformity_report(u)["p_value"] > 0.99).all()  # rank transform is ~exactly uniform
    assert u.mean().to_numpy() == pytest.approx(0.5, abs=1e-9)


def test_out_of_sample_uses_fitted_ecdf_only():
    train = _returns(n=200)
    m = EmpiricalMarginal().fit(train)
    new = pd.DataFrame({"A": [1.0, -1.0], "B": [0.0, 0.0], "C": [0.0, 0.0]})  # extreme values
    u = m.transform(new)
    assert u.loc[0, "A"] == pytest.approx(200 / 201)
    assert u.loc[1, "A"] == pytest.approx(1 / 201)


def test_inverse_roundtrip():
    r = _returns()
    m = EmpiricalMarginal().fit(r)
    back = m.inverse_transform(m.transform(r))
    assert back.to_numpy() == pytest.approx(r.to_numpy())


def test_rejects_nan_and_unfitted_and_mismatched_columns():
    r = _returns()
    bad = r.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        EmpiricalMarginal().fit(bad)
    with pytest.raises(RuntimeError):
        EmpiricalMarginal().transform(r)
    with pytest.raises(ValueError):
        EmpiricalMarginal().fit(r).transform(r[["A", "B"]])
