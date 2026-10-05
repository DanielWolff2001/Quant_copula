"""Structural change detection on a sequence of rolling vine fits.

Every detector turns the fitted models into a time series of *scores*; a score being
large means "the dependence structure looks different from before". A score becomes an
alert only when it exceeds a configurable threshold.

Detectors implemented (simplest first):

1. `frobenius_change` - Frobenius distance between pairwise dependence matrices
   ``||T_t - T_{t-lag}||_F`` (the PDF's ``S_t``);
2. `model_distance` - the same distance on the model-implied Kendall's tau and on
   the lower/upper tail-dependence matrices, so it also sees tail-only changes;
3. `rolling_zscore` - how unusual a series is compared with its own past;
4. `cusum` - a one-sided CUSUM accumulating persistent positive deviations;
5. `change_scan` - a permutation test between two disjoint windows, which gives
   calibrated p-values (the recommended detector).

``lag`` is counted in *fits*. With ``lag = 1`` consecutive overlapping windows are
compared (they share all but one observation, so the score is almost pure estimation
noise and cannot see a regime change); ``lag = window / refit_frequency`` compares two
disjoint windows (see `default_lag`).

Calibration (measured in the validation study, `vine_risk.validation`): the distance
scores have no built-in null distribution, so an alert threshold has to be calibrated by
simulating data without change. A threshold calibrated on constant-dependence data is too
low once volatility clusters (false alarms rise from 1% to 3-10%), so calibrate on the
harder null. With ``lag = 1`` the scores cannot see a regime change at all. The z-score of
the disjoint-window distance has about 1% false alarms above 3 but little power. The
permutation test (5) is calibrated by construction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd

from vine_risk.dependence import pairwise_series, structure_changes

if TYPE_CHECKING:
    from vine_risk.copula import VineFitResult


def default_lag(window: int, refit_frequency: int = 1) -> int:
    """Lag (in fits) after which two windows no longer overlap."""
    return max(1, window // refit_frequency)


def frobenius_change(pairs: pd.DataFrame, lag: int = 1) -> pd.Series:
    """``||M_t - M_{t-lag}||_F`` for symmetric matrices stored as pair columns.

    ``pairs`` has one column per asset pair ``i < j`` (see
    `vine_risk.dependence.pairwise_series`). The matrices have a zero diagonal
    and are symmetric, so ``||.||_F = sqrt(2 * sum_{i<j} diff^2)``.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    diff = pairs - pairs.shift(lag)
    return np.sqrt(2.0 * (diff ** 2).sum(axis=1, min_count=1))


def model_distance(results: Sequence["VineFitResult"], lag: int = 1) -> pd.DataFrame:
    """Distance between fitted models ``D(C_t, C_{t-lag})`` through their implied dependence.

    The vine representation can change completely between fits (other tree, other
    families), so models are compared through quantities every vine implies: the
    pairwise Kendall's tau, lower-tail and upper-tail coefficients. Returns the three
    Frobenius distances (``dist_tau``, ``dist_lower``, ``dist_upper``) and their
    root-sum-of-squares ``dist_model``.
    """
    out = pd.DataFrame({
        name: frobenius_change(pairwise_series(results, col), lag)
        for name, col in (("dist_tau", "model_tau"), ("dist_lower", "lower_tail_q"),
                          ("dist_upper", "upper_tail_q"))
    })
    out["dist_model"] = np.sqrt((out ** 2).sum(axis=1, min_count=3))
    return out


def _min_periods(baseline: int, min_periods: int | None) -> int:
    return max(2, baseline // 5) if min_periods is None else min_periods


def rolling_zscore(x: pd.Series, baseline: int = 500, min_periods: int | None = None) -> pd.Series:
    """``(x_t - mu_t) / sigma_t`` with mean and std taken over the *previous* ``baseline``
    observations only (the current value is excluded, so a shift is not hidden by itself).
    ``min_periods`` defaults to ``baseline // 5``."""
    past = x.shift(1).rolling(baseline, min_periods=_min_periods(baseline, min_periods))
    sd = past.std().replace(0.0, np.nan)
    return (x - past.mean()) / sd


def cusum(z: pd.Series, k: float = 0.5) -> pd.Series:
    """One-sided (upward) CUSUM: ``C_t = max(0, C_{t-1} + z_t - k)``.

    ``k`` is the allowance (in standard deviations): deviations smaller than ``k`` are
    ignored, persistent larger ones accumulate. NaNs are skipped.
    """
    c, out = 0.0, []
    for v in z.to_numpy(dtype=float):
        if np.isfinite(v):
            c = max(0.0, c + v - k)
        out.append(c)
    return pd.Series(out, index=z.index, name="cusum")


def aspect_flags(aspects: pd.DataFrame, baseline: int = 500, quantile: float = 0.95,
                 min_periods: int | None = None) -> pd.DataFrame:
    """Flag each aspect when it exceeds the ``quantile`` of its own past values.

    Quantile flags (not z-scores) are used because several aspects, such as the share of
    pair-copulas that switched family, are zero-inflated.
    """
    thr = aspects.shift(1).rolling(
        baseline, min_periods=_min_periods(baseline, min_periods)).quantile(quantile)
    return (aspects > thr) & thr.notna()


def structural_change_scores(
    results: Sequence["VineFitResult"],
    lag: int = 1,
    baseline: int = 500,
    quantile: float = 0.95,
    cusum_k: float = 0.5,
) -> pd.DataFrame:
    """All structural-change series on one date index (``timestamp`` -> scores).

    Columns:

    * ``s_tau`` - Frobenius change of the empirical Kendall's tau matrix (the PDF's S_t);
    * ``dist_tau``, ``dist_lower``, ``dist_upper``, ``dist_model`` - model distance;
    * ``z_s_tau``, ``z_dist_model`` - the two distances relative to their own past;
    * ``z_d_t`` - the average-dependence level ``D_t`` relative to its own past;
    * ``cusum_d_t`` - CUSUM of ``z_d_t``;
    * ``flag_*`` and ``n_aspects_flagged`` - which of the aspects *tau matrix, tail,
      relationships (vine edges), family* are unusually large, and how many at once;
    * ``structural_change_score`` - alias of ``s_tau``, the primary raw score. Use
      ``lag = default_lag(window, refit_frequency)``; with ``lag = 1`` it is the PDF's
      one-step ``S_t``, which is dominated by estimation noise.

    ``baseline`` is counted in fits. The ``z_*``/``cusum`` columns are secondary
    diagnostics: the z-score is close to calibrated but has little power (the baseline
    absorbs the shift), and the CUSUM never resets, so it keeps alarming after a change.
    Use a null-calibrated threshold on the distances, or `change_scan`, to decide.
    """
    tau = pairwise_series(results, "tau")
    out = pd.DataFrame({"s_tau": frobenius_change(tau, lag)})
    out = out.join(model_distance(results, lag))
    out["z_s_tau"] = rolling_zscore(out["s_tau"], baseline)
    out["z_dist_model"] = rolling_zscore(out["dist_model"], baseline)
    d_t = tau.abs().mean(axis=1)
    out["z_d_t"] = rolling_zscore(d_t, baseline)
    out["cusum_d_t"] = cusum(out["z_d_t"], cusum_k)

    struct = structure_changes(results)
    aspects = pd.DataFrame({
        "tau_matrix": out["s_tau"],
        "tail": out[["dist_lower", "dist_upper"]].max(axis=1, skipna=False),
        "relationships": struct["relationship_change"],
        "family": struct["family_change_frac"],
    })
    flags = aspect_flags(aspects, baseline, quantile).add_prefix("flag_")
    out = out.join(flags)
    out["n_aspects_flagged"] = flags.sum(axis=1)
    out["structural_change_score"] = out["s_tau"]
    out.index.name = "timestamp"
    return out


def alerts(score: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Boolean alert series: ``score > threshold`` (NaN scores never alert)."""
    return (score > threshold).fillna(False)


# ---------------------------------------------------------------------------
# Calibrated detector: permutation test between two disjoint windows
# ---------------------------------------------------------------------------

def _frobenius_from_pairs(delta: np.ndarray) -> np.ndarray:
    """Frobenius norm of symmetric zero-diagonal matrices given their upper triangles (pairs, K)."""
    return np.sqrt(2.0 * (delta ** 2).sum(axis=0))


def _tail_matrices(x: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrised lower/upper tail coefficients (upper triangles) of one window.

    Ranks are taken *within* the window (as the vine does), so differences in marginal
    volatility between windows are removed. The coefficient of a pair is the number of
    joint exceedances divided by the average of the two marginal exceedance counts.
    """
    m, d = x.shape
    u = (np.argsort(np.argsort(x, axis=0), axis=0) + 1) / (m + 1)
    iu = np.triu_indices(d, 1)
    res = []
    for e in ((u < q).astype(float), (u > 1 - q).astype(float)):
        cnt = e.sum(axis=0)
        denom = np.maximum(0.5 * (cnt[:, None] + cnt[None, :]), 1.0)
        res.append(((e.T @ e) / denom)[iu])
    return res[0], res[1]


def two_window_change_test(
    x_old_new: np.ndarray, n_old: int, n_perm: int = 200, q: float = 0.1, seed: int = 0,
    block: int = 1,
) -> dict[str, float]:
    """Test whether dependence differs between an older and a newer window.

    ``x_old_new`` stacks the older window (first ``n_old`` rows) and the newer one. The
    statistics are Frobenius distances between the two windows' matrices of

    * Kendall's tau (``tau``),
    * empirical lower-tail coefficients (``lower``): joint exceedances of the ``q``
      quantile divided by the average of the two marginal exceedance counts (a
      symmetrised ``P(U_j < q | U_i < q)``),
    * empirical upper-tail coefficients (``upper``), the same above ``1 - q``,

    with ranks taken within each window. Each also has a ``mean_*`` version, the
    absolute change of the *average over pairs*: the Frobenius distance reacts to any
    single pair changing, the mean is more powerful against a change shared by all
    pairs (typical for market-wide stress).

    The null distribution comes from randomly re-assigning observations to the two
    windows (``n_perm`` times), which answers: "how large is this distance when there
    is no change and only estimation noise?". With ``block > 1`` blocks of ``block``
    consecutive days are re-assigned instead of single days, so the null keeps local
    serial dependence such as volatility clustering (``block`` must divide both window
    lengths). The p-values are ``(1 + #{null >= observed}) / (1 + n_perm)``. With enough data almost any
    difference becomes significant, so also look at the effect size: ``stat_*`` relative
    to ``null_mean_*`` (the average distance under "no change").

    Assumes exchangeability of the (blocks of) observations. With ``block = 1``,
    volatility clustering leaves the tau tests valid but makes the tail tests too
    liberal; use ``block > 1`` for real returns.
    """
    x = np.asarray(x_old_new, dtype=float)
    n, d = x.shape
    n_new = n - n_old
    if n_old < 20 or n_new < 20 or d < 2:
        raise ValueError("Need at least 20 observations per window and 2 assets.")
    if block < 1 or n_old % block or n_new % block:
        raise ValueError("block must divide both window lengths.")
    rng = np.random.default_rng(seed)
    nb, nb_old = n // block, n_old // block
    # group indicator matrix (n, K+1): column 0 is the observed split, "1" = old window
    g = np.zeros((n, n_perm + 1), dtype=np.float32)
    g[:n_old, 0] = 1.0
    for k in range(1, n_perm + 1):
        chosen = np.zeros(nb, dtype=bool)
        chosen[rng.permutation(nb)[:nb_old]] = True
        g[:, k] = np.repeat(chosen, block)
    h = 1.0 - g
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    # --- Kendall's tau per window via pairwise sign products P_ij(a, b) = s_i(a,b) s_j(a,b)
    sign = [np.sign(x[:, None, i] - x[None, :, i]).astype(np.float32) for i in range(d)]
    tau_old = np.empty((len(pairs), n_perm + 1))
    tau_new = np.empty_like(tau_old)
    for p, (i, j) in enumerate(pairs):
        P = sign[i] * sign[j]
        tau_old[p] = ((P @ g) * g).sum(axis=0) / (n_old * (n_old - 1))
        tau_new[p] = ((P @ h) * h).sum(axis=0) / (n_new * (n_new - 1))
    stats_ = {"tau": _frobenius_from_pairs(tau_old - tau_new),
              "mean_tau": np.abs((tau_old - tau_new).mean(axis=0))}

    # --- tail coefficients with within-window ranks (not linear in g: recomputed per split)
    lower_d = np.empty_like(tau_old)
    upper_d = np.empty_like(tau_old)
    for k in range(n_perm + 1):
        old = g[:, k] > 0.5
        lo_old, up_old = _tail_matrices(x[old], q)
        lo_new, up_new = _tail_matrices(x[~old], q)
        lower_d[:, k], upper_d[:, k] = lo_old - lo_new, up_old - up_new
    for name, delta in (("lower", lower_d), ("upper", upper_d)):
        stats_[name] = _frobenius_from_pairs(delta)
        stats_[f"mean_{name}"] = np.abs(delta.mean(axis=0))

    out: dict[str, float] = {}
    for name, sv in stats_.items():
        out[f"stat_{name}"] = float(sv[0])
        out[f"p_{name}"] = float((1 + np.sum(sv[1:] >= sv[0])) / (1 + n_perm))
        out[f"null_mean_{name}"] = float(sv[1:].mean())  # typical distance under "no change"
    return out


def change_scan(
    returns: pd.DataFrame, window: int, step: int = 5, n_perm: int = 200, q: float = 0.1, seed: int = 0,
    block: int = 1, after: pd.Timestamp | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run `two_window_change_test` along the history.

    At each scan date ``t`` the window ``(t-window, t]`` is compared with the preceding
    window ``(t-2*window, t-window]``; only data up to ``t`` is used. Returns
    ``stat_*`` and ``p_*`` for tau, lower, upper (and their ``mean_*`` versions), indexed by ``t``.

    The scan dates lie on a grid anchored at the first return observation. With ``after`` only
    dates later than that are computed (to extend an earlier scan; the grid stays the same).
    ``progress(done, total)`` is called after every scan date.
    """
    if len(returns) < 2 * window:
        raise ValueError(f"Need at least {2 * window} observations, got {len(returns)}.")
    x = returns.to_numpy()
    grid = [t for t in range(2 * window - 1, len(x), step) if after is None or returns.index[t] > after]
    rows = {}
    for i, t in enumerate(grid, start=1):
        rows[returns.index[t]] = two_window_change_test(x[t - 2 * window + 1: t + 1], window, n_perm, q, seed, block)
        if progress:
            progress(i, len(grid))
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "timestamp"
    return out


def benjamini_hochberg(p: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Hochberg false-discovery-rate control: True where a p-value is rejected.

    Many scan dates are tested, so some small p-values appear by chance. Scan dates
    overlap and are positively dependent, which BH tolerates, but the result is still
    approximate.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    v = p.to_numpy(dtype=float)
    ok = np.isfinite(v)
    out = np.zeros(len(v), dtype=bool)
    pv = v[ok]
    m = len(pv)
    if m:
        order = np.argsort(pv)
        passed = pv[order] <= alpha * np.arange(1, m + 1) / m
        k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
        rej = np.zeros(m, dtype=bool)
        rej[order[:k]] = True
        out[ok] = rej
    return pd.Series(out, index=p.index, name=f"{p.name}_fdr")


def change_scan_filtered(
    returns: pd.DataFrame, window: int, step: int = 5, n_perm: int = 499, q: float = 0.1, seed: int = 0,
    block: int = 10, innovations: str = "t", after: pd.Timestamp | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """The permutation scan of :func:`change_scan`, applied to **GARCH-filtered** residuals.

    A change in volatility can look like a change in dependence (for instance when a market-wide shock
    moves all assets at once). To ask about dependence alone, a GARCH(1,1) is fitted to every asset over
    the last ``2 * window`` days at each scan date; the two windows compared are the halves of the
    resulting standardised-residual matrix. Same grid, columns and conventions as `change_scan`.
    """
    from vine_risk.garch import GarchMarginal

    if len(returns) < 2 * window:
        raise ValueError(f"Need at least {2 * window} observations, got {len(returns)}.")
    grid = [t for t in range(2 * window - 1, len(returns), step) if after is None or returns.index[t] > after]
    rows = {}
    for i, t in enumerate(grid, start=1):
        x = returns.iloc[t - 2 * window + 1: t + 1]
        z = GarchMarginal(innovations=innovations).fit(x).standardised_residuals().to_numpy()
        rows[returns.index[t]] = two_window_change_test(z, window, n_perm, q, seed, block)
        if progress:
            progress(i, len(grid))
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "timestamp"
    return out
