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


# ---- monitoring metrics on hand-built results (no model fitting) ----------
from vine_risk.copula import PairCopulaInfo, VineFitResult
from vine_risk.dependence import (
    average_absolute_tau, dependence_metrics, pairwise_changes, pairwise_series, structure_changes,
)

ASSETS = ["A", "B", "C"]


def _pc(tree, cond, given, family, tau, rotation=0):
    return PairCopulaInfo(tree, 1, tuple(cond), tuple(given), family, rotation, [0.5], tau,
                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def _res(ts, taus, pcs, lower=0.2, upper=0.1, status="ok"):
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    pw = [dict(asset_i=i, asset_j=j, tau=t, spearman=1.5 * t, pearson=t, model_tau=t,
               lower_tail_q=lower, upper_tail_q=upper) for (i, j), t in zip(pairs, taus)]
    return VineFitResult(assets=ASSETS, n_obs=100, n_assets=3, timestamp=ts, pair_copulas=pcs,
                         pairwise=pw, loglik=10.0, aic=-1.0, bic=-2.0, n_params=2.0, status=status)


BASE = [_pc(1, "AB", "", "gaussian", 0.5), _pc(1, "BC", "", "clayton", 0.3),
        _pc(2, "AC", "B", "independence", 0.0)]


def test_average_absolute_tau_is_normalised_sum():
    res = [_res("2020-01-01", [0.5, -0.3, 0.1], BASE)]
    d = average_absolute_tau(pairwise_series(res, "tau"))
    assert d.iloc[0] == pytest.approx(2 / (3 * 2) * (0.5 + 0.3 + 0.1))


def test_pairwise_changes_are_consecutive_differences():
    res = [_res("2020-01-01", [0.5, 0.2, 0.1], BASE), _res("2020-01-02", [0.4, 0.3, 0.1], BASE)]
    ch = pairwise_changes(res, "tau")
    assert ch.iloc[0].isna().all()
    assert ch.iloc[1].to_dict() == pytest.approx({"A-B": -0.1, "A-C": 0.1, "B-C": 0.0})


def test_structure_changes_detects_family_and_edge_switches():
    same = _res("2020-01-02", [0.5, 0.2, 0.1], BASE)
    changed = _res("2020-01-03", [0.5, 0.2, 0.1], [
        _pc(1, "AB", "", "gaussian", 0.5), _pc(1, "AC", "", "gumbel", 0.2),  # BC edge replaced by AC
        _pc(2, "BC", "A", "independence", 0.0)])
    sc = structure_changes([_res("2020-01-01", [0.5, 0.2, 0.1], BASE), same, changed])
    assert sc.iloc[0].drop("n_non_independent").isna().all()
    assert sc.iloc[1][["tree1_edge_change", "relationship_change", "family_change_frac",
                       "mean_abs_dtau_pc"]].tolist() == [0.0, 0.0, 0.0, 0.0]
    # tree-1 edges {AB,BC} -> {AB,AC}: |inter|=1, |union|=3
    assert sc.iloc[2]["tree1_edge_change"] == pytest.approx(2 / 3)
    assert sc.iloc[2]["relationship_change"] == pytest.approx(1 - 1 / 5)  # {AB} shared of 5
    assert sc.iloc[2]["family_change_frac"] == 0.0  # only AB is common, family unchanged
    assert sc.iloc[2]["n_non_independent"] == 2


def test_family_change_fraction_and_rotation():
    a = _res("2020-01-01", [0.5, 0.2, 0.1], BASE)
    b = _res("2020-01-02", [0.5, 0.2, 0.1], [
        _pc(1, "AB", "", "gaussian", 0.5), _pc(1, "BC", "", "clayton", 0.3, rotation=180),
        _pc(2, "AC", "B", "independence", 0.0)])
    assert structure_changes([a, b]).iloc[1]["family_change_frac"] == pytest.approx(1 / 3)


def test_dependence_metrics_table_and_failed_fit():
    res = [_res("2020-01-01", [0.5, 0.2, 0.1], BASE),
           VineFitResult.failed(ASSETS, 100, "boom", timestamp="2020-01-02"),
           _res("2020-01-03", [0.4, 0.2, 0.2], BASE, lower=0.3, upper=0.1)]
    m = dependence_metrics(res)
    assert list(m.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert m["d_t"].iloc[0] == pytest.approx((0.5 + 0.2 + 0.1) / 3)
    assert np.isnan(m["d_t"].iloc[1]) and m["status"].iloc[1] == "failed"
    assert m["tail_asymmetry"].iloc[2] == pytest.approx(0.2)
    assert m["max_abs_dtau"].iloc[0] != m["max_abs_dtau"].iloc[0]  # NaN on first row
    with pytest.raises(ValueError):
        dependence_metrics([])
