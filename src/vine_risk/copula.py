"""Static vine copula: a thin wrapper around ``pyvinecopulib``.

``VineCopula.fit`` takes pseudo-observations (uniforms) and ``summary`` returns a
plain, serialisable `VineFitResult`; the fitted C++ object is never stored
in results.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

from vine_risk.dependence import pairwise_dependence

logger = logging.getLogger(__name__)


class VineFitError(RuntimeError):
    """Raised when a vine copula cannot be fitted."""


@dataclass
class PairCopulaInfo:
    """One pair-copula of the vine.

    ``tau``, ``lower_tail`` and ``upper_tail`` are properties of the pair-copula
    itself; for ``tree > 1`` they are *conditional* on ``conditioning``.
    """

    tree: int
    edge: int
    conditioned: tuple[str, str]
    conditioning: tuple[str, ...]
    family: str
    rotation: int
    parameters: list[float]
    tau: float
    lower_tail: float
    upper_tail: float
    n_params: float
    loglik: float
    aic: float
    bic: float


@dataclass
class VineFitResult:
    """Structured, serialisable summary of one vine fit."""

    # metadata
    assets: list[str]
    n_obs: int
    n_assets: int
    window_start: str | None = None
    window_end: str | None = None
    timestamp: str | None = None
    # model
    pair_copulas: list[PairCopulaInfo] = field(default_factory=list)
    order: list[int] = field(default_factory=list)  # 1-based variable order of the R-vine
    truncation_level: int | None = None
    # pairwise (unconditional) dependence, empirical
    pairwise: list[dict[str, Any]] = field(default_factory=list)
    # diagnostics
    loglik: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    n_params: float = float("nan")
    selection_criterion: str = ""
    status: str = "ok"  # "ok" | "failed"
    message: str = ""

    # --- convenience views -------------------------------------------------
    def pair_copula_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(p) for p in self.pair_copulas])

    def pairwise_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.pairwise)

    def tau_matrix(self) -> pd.DataFrame:
        """Symmetric matrix of empirical Kendall's tau (diagonal = 1)."""
        m = pd.DataFrame(np.eye(self.n_assets), index=self.assets, columns=self.assets)
        for r in self.pairwise:
            m.loc[r["asset_i"], r["asset_j"]] = m.loc[r["asset_j"], r["asset_i"]] = r["tau"]
        return m

    # --- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VineFitResult":
        d = dict(d)
        d["pair_copulas"] = [
            PairCopulaInfo(**{**p, "conditioned": tuple(p["conditioned"]),
                              "conditioning": tuple(p["conditioning"])})
            for p in d.get("pair_copulas", [])
        ]
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "VineFitResult":
        return cls.from_dict(json.loads(s))

    @classmethod
    def failed(cls, assets: list[str], n_obs: int, message: str, **meta: Any) -> "VineFitResult":
        return cls(assets=list(assets), n_obs=n_obs, n_assets=len(assets),
                   status="failed", message=message, **meta)


class VineCopula:
    """Regular vine copula fitted with ``pyvinecopulib``.

    Args:
        families: names of the pair-copula families to select from (default: all
            parametric families, which include independence).
        selection_criterion: ``"bic"`` (default), ``"aic"`` or ``"loglik"`` for the
            family selection.
        truncation_level: truncate the vine after this tree (``None`` = full vine).
        tree_criterion: edge weight for the tree structure (default Kendall's tau).
        num_threads: threads used by the C++ fitter.
        tail_simulations: number of Sobol points used to compute the model-implied
            pairwise dependence in `summary` (0 disables it; use a power of 2).
        tail_level: ``q`` of the finite-level tail coefficients, see
            `implied_pairwise`.
        seed: seed of the Sobol sequence. It is the same for every fit, so differences
            between windows are not caused by simulation noise.
    """

    def __init__(
        self,
        families: list[str] | None = None,
        selection_criterion: str = "bic",
        truncation_level: int | None = None,
        tree_criterion: str = "tau",
        num_threads: int = 1,
        tail_simulations: int = 2**14,
        tail_level: float = 0.05,
        seed: int = 0,
    ) -> None:
        if not 0 < tail_level < 0.5:
            raise ValueError("tail_level must be in (0, 0.5).")
        if tail_simulations < 0:
            raise ValueError("tail_simulations must be >= 0.")
        self.tail_simulations, self.tail_level, self.seed = tail_simulations, tail_level, seed
        if selection_criterion not in {"aic", "bic", "loglik"}:
            raise ValueError(f"Unsupported selection_criterion: {selection_criterion!r}")
        if families is None:
            fam_set = list(pv.families.parametric)
        else:
            try:
                fam_set = [getattr(pv.BicopFamily, f) for f in families]
            except AttributeError as e:
                raise ValueError(f"Unknown copula family in {families}") from e
        kwargs: dict[str, Any] = dict(
            family_set=fam_set, selection_criterion=selection_criterion,
            tree_criterion=tree_criterion, num_threads=num_threads,
        )
        if truncation_level is not None:
            kwargs["trunc_lvl"] = truncation_level
        self._controls = pv.FitControlsVinecop(**kwargs)
        self.selection_criterion = selection_criterion
        self._model: pv.Vinecop | None = None
        self._u: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def fit(self, u: pd.DataFrame, returns: pd.DataFrame | None = None) -> "VineCopula":
        """Fit to pseudo-observations ``u`` (T x d, strictly inside (0, 1)).

        ``returns`` (same shape/columns) is only used to report Pearson correlation.
        Raises `VineFitError` if the fit fails.
        """
        _validate_uniform(u)
        try:
            model = pv.Vinecop.from_data(u.to_numpy(), controls=self._controls)
        except Exception as e:  # pyvinecopulib raises a variety of C++ errors
            raise VineFitError(f"Vine fit failed: {e}") from e
        self._model, self._u, self._returns = model, u, returns
        logger.debug("Fitted vine on %d x %d data", *u.shape)
        return self

    def summary(self) -> VineFitResult:
        """Return the structured result of the last fit."""
        model, u = self._require_fitted()
        assets = [str(c) for c in u.columns]
        pcs: list[PairCopulaInfo] = []
        for t, tree in enumerate(model.get_trees(), start=1):
            for e, edge in enumerate(tree, start=1):
                bc = edge["pair_copula"]
                td = np.asarray(bc.taildep)
                pcs.append(PairCopulaInfo(
                    tree=t, edge=e,
                    conditioned=tuple(assets[i - 1] for i in edge["conditioned"]),
                    conditioning=tuple(assets[i - 1] for i in edge["conditioning"]),
                    family=bc.family_name.lower(), rotation=int(bc.rotation),
                    parameters=np.asarray(bc.parameters, dtype=float).ravel().tolist(),
                    tau=float(bc.tau), lower_tail=float(td[0, 0]), upper_tail=float(td[1, 1]),
                    n_params=float(bc.npars), loglik=float(bc.loglik()),
                    aic=float(bc.aic()), bic=float(bc.bic()),
                ))
        idx = u.index
        is_dt = isinstance(idx, pd.DatetimeIndex)
        return VineFitResult(
            assets=assets, n_obs=len(u), n_assets=len(assets),
            window_start=str(idx[0].date()) if is_dt else None,
            window_end=str(idx[-1].date()) if is_dt else None,
            timestamp=str(idx[-1].date()) if is_dt else None,
            pair_copulas=pcs, order=[int(x) for x in model.order],
            truncation_level=int(model.trunc_lvl),
            pairwise=self._pairwise_records(u),
            loglik=float(model.loglik(u.to_numpy())), aic=float(model.aic(u.to_numpy())),
            bic=float(model.bic(u.to_numpy())), n_params=float(model.npars),
            selection_criterion=self.selection_criterion,
        )

    def _pairwise_records(self, u: pd.DataFrame) -> list[dict[str, Any]]:
        emp = pairwise_dependence(u, self._returns)
        if self.tail_simulations > 0:
            imp = self.implied_pairwise(self.tail_simulations, self.tail_level, self.seed)
            emp = emp.merge(imp, on=["asset_i", "asset_j"], how="left")
        return emp.to_dict("records")

    def implied_pairwise(self, n: int = 2**14, q: float = 0.05, seed: int = 0) -> pd.DataFrame:
        """Unconditional pairwise dependence implied by the fitted vine.

        Draws ``n`` Sobol points through the vine (inverse Rosenblatt transform) and
        measures, for every asset pair ``(i, j)``:

        * ``model_tau``: Kendall's tau of the simulated sample;
        * ``lower_tail_q``: ``P(U_j < q | U_i < q) = C(q, q) / q``;
        * ``upper_tail_q``: ``P(U_j > 1-q | U_i > 1-q)``.

        These are tail coefficients at the finite level ``q`` (default 5%), not the
        limit ``q -> 0``. They are positive even for the Gaussian copula and (unlike
        the asymptotic coefficients) can be estimated reliably from a simulation.
        Pair-copula tail parameters beyond tree 1 are *conditional*; these are not.
        """
        model, u = self._require_fitted()
        s = model.inverse_rosenblatt(pv.utils.sobol(n, u.shape[1], [seed]))
        rows = []
        for i, j in combinations(range(u.shape[1]), 2):
            a, b = s[:, i], s[:, j]
            rows.append((
                str(u.columns[i]), str(u.columns[j]),
                float(stats.kendalltau(a, b).statistic),
                float(np.mean((a < q) & (b < q)) / q),
                float(np.mean((a > 1 - q) & (b > 1 - q)) / q),
            ))
        return pd.DataFrame(rows, columns=["asset_i", "asset_j", "model_tau", "lower_tail_q", "upper_tail_q"])

    def simulate(self, n: int, seed: int = 0) -> pd.DataFrame:
        """Draw ``n`` uniform samples from the fitted vine (reproducible via ``seed``)."""
        model, u = self._require_fitted()
        s = model.sample(n, seeds=[seed])
        return pd.DataFrame(s, columns=u.columns)

    def _require_fitted(self) -> tuple[pv.Vinecop, pd.DataFrame]:
        if self._model is None or self._u is None:
            raise RuntimeError("VineCopula is not fitted; call fit() first.")
        return self._model, self._u


def _validate_uniform(u: pd.DataFrame) -> None:
    if not isinstance(u, pd.DataFrame):
        raise TypeError("u must be a pandas DataFrame.")
    if u.shape[1] < 2:
        raise ValueError("A vine copula needs at least 2 variables.")
    if u.shape[0] < 10 * u.shape[1]:
        raise ValueError(f"Too few observations ({u.shape[0]}) for {u.shape[1]} variables.")
    x = u.to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("u contains NaN or infinite values.")
    if (x <= 0).any() or (x >= 1).any():
        raise ValueError("u must lie strictly inside (0, 1).")
