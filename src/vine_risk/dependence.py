"""Pairwise dependence measures."""
from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:  # avoid a circular import (copula imports this module)
    from vine_risk.copula import VineFitResult


def pairwise_dependence(u: pd.DataFrame, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Kendall's tau, Spearman's rho and Pearson correlation for every asset pair.

    ``tau`` and ``spearman`` are rank-based, so they are identical on returns and on
    the pseudo-observations ``u``; they are computed from ``u``. ``pearson`` is a
    linear correlation of the *returns* and is NaN when ``returns`` is not given.

    Returns a long frame with columns ``asset_i, asset_j, tau, spearman, pearson``
    (one row per pair ``i < j`` in column order).
    """
    if returns is not None and not returns.columns.equals(u.columns):
        raise ValueError("returns and u must have the same columns.")
    rows = []
    for a, b in combinations(u.columns, 2):
        tau = stats.kendalltau(u[a], u[b]).statistic
        rho = stats.spearmanr(u[a], u[b]).statistic
        pear = np.corrcoef(returns[a], returns[b])[0, 1] if returns is not None else np.nan
        rows.append((a, b, tau, rho, pear))
    return pd.DataFrame(rows, columns=["asset_i", "asset_j", "tau", "spearman", "pearson"])


# ---------------------------------------------------------------------------
# Monitoring metrics computed from a sequence of rolling fits
# ---------------------------------------------------------------------------

def pairwise_series(results: Sequence["VineFitResult"], column: str = "tau") -> pd.DataFrame:
    """One pairwise quantity over time: rows = timestamps, columns = ``"A-B"`` pairs.

    ``column`` is any key of ``VineFitResult.pairwise`` (``tau``, ``spearman``,
    ``pearson``, ``model_tau``, ``lower_tail_q``, ``upper_tail_q``). Failed fits give
    NaN rows.
    """
    rows = {}
    for r in results:
        rows[pd.Timestamp(r.timestamp)] = {
            f"{p['asset_i']}-{p['asset_j']}": p.get(column, np.nan) for p in r.pairwise
        }
    out = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    return out.reindex(pd.DatetimeIndex(sorted(rows)))  # keeps timestamps of failed fits


def pairwise_changes(results: Sequence["VineFitResult"], column: str = "tau") -> pd.DataFrame:
    """``x_{ij,t} - x_{ij,t-1}`` for every pair (change between consecutive fits)."""
    return pairwise_series(results, column).diff()


def average_absolute_tau(tau: pd.DataFrame) -> pd.Series:
    """``D_t = 2 / (d (d-1)) * sum_{i<j} |tau_ij,t|`` from a :func:`pairwise_series` frame."""
    return tau.abs().mean(axis=1).rename("d_t")


def _jaccard_distance(a: set, b: set) -> float:
    return 1.0 - len(a & b) / len(a | b) if (a | b) else 0.0


def structure_changes(results: Sequence["VineFitResult"]) -> pd.DataFrame:
    """Compare every fit with the previous one (Level 2 monitoring).

    A pair-copula is identified by its conditioned pair and conditioning set, so
    "the same relationship" can be followed even when the tree changes elsewhere.
    Columns (the first row is NaN):

    * ``tree1_edge_change``: Jaccard distance between the sets of first-tree edges;
    * ``relationship_change``: Jaccard distance between the sets of all (conditional)
      relationships in the vine;
    * ``family_change_frac``: share of relationships present in both fits whose copula
      family or rotation changed;
    * ``mean_abs_dtau_pc``: mean ``|d tau|`` of the pair-copulas present in both fits;
    * ``n_non_independent``: number of pair-copulas that are not independence.
    """
    def keyed(r):
        return {(frozenset(p.conditioned), frozenset(p.conditioning)): p for p in r.pair_copulas}

    nan = dict(tree1_edge_change=np.nan, relationship_change=np.nan,
               family_change_frac=np.nan, mean_abs_dtau_pc=np.nan)
    rows, prev = {}, None
    for r in results:
        cur = keyed(r)
        row = dict(nan, n_non_independent=sum(p.family != "independence" for p in cur.values())
                   if r.status == "ok" else np.nan)
        if prev is not None and cur and prev:
            t1 = lambda d: {k[0] for k, p in d.items() if p.tree == 1}
            row["tree1_edge_change"] = _jaccard_distance(t1(cur), t1(prev))
            row["relationship_change"] = _jaccard_distance(set(cur), set(prev))
            common = set(cur) & set(prev)
            if common:
                row["family_change_frac"] = float(np.mean(
                    [(cur[k].family, cur[k].rotation) != (prev[k].family, prev[k].rotation) for k in common]))
                row["mean_abs_dtau_pc"] = float(np.mean([abs(cur[k].tau - prev[k].tau) for k in common]))
        rows[pd.Timestamp(r.timestamp)] = row
        prev = cur if cur else prev
    return pd.DataFrame.from_dict(rows, orient="index").reindex(pd.DatetimeIndex(sorted(rows)))


def dependence_metrics(results: Sequence["VineFitResult"]) -> pd.DataFrame:
    """All Level 1 and Level 2 monitoring series in one table (index = timestamp).

    Level 1 (pairwise dependence, averaged over pairs): ``d_t`` (mean ``|tau|``),
    ``mean_spearman``, ``mean_pearson``, ``mean_model_tau``, ``mean_lower_tail_q``,
    ``mean_upper_tail_q``, ``tail_asymmetry`` (lower minus upper), and the change
    between consecutive fits ``mean_abs_dtau`` / ``max_abs_dtau``.
    Level 2: the columns of :func:`structure_changes` plus model diagnostics
    (``loglik``, ``aic``, ``bic``, ``n_params``, ``status``).
    """
    if not results:
        raise ValueError("No results.")
    pair = {c: pairwise_series(results, c)
            for c in ("tau", "spearman", "pearson", "model_tau", "lower_tail_q", "upper_tail_q")}
    dtau = pair["tau"].diff().abs()
    out = pd.DataFrame({
        "d_t": average_absolute_tau(pair["tau"]),
        "mean_spearman": pair["spearman"].mean(axis=1),
        "mean_pearson": pair["pearson"].mean(axis=1),
        "mean_model_tau": pair["model_tau"].mean(axis=1),
        "mean_lower_tail_q": pair["lower_tail_q"].mean(axis=1),
        "mean_upper_tail_q": pair["upper_tail_q"].mean(axis=1),
        "mean_abs_dtau": dtau.mean(axis=1),
        "max_abs_dtau": dtau.max(axis=1),
    })
    out["tail_asymmetry"] = out["mean_lower_tail_q"] - out["mean_upper_tail_q"]
    out = out.join(structure_changes(results))
    diag = pd.DataFrame(
        {"loglik": r.loglik, "aic": r.aic, "bic": r.bic, "n_params": r.n_params, "status": r.status}
        for r in results
    )
    diag.index = pd.DatetimeIndex([pd.Timestamp(r.timestamp) for r in results])
    out = out.join(diag.sort_index())
    out.index.name = "timestamp"
    return out
