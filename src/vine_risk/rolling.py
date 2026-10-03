"""Rolling-window vine copula estimation.

Two layers are kept separate:

* **data update** - :meth:`RollingVineModel.push` appends one observation to the window;
* **model refit** - :meth:`RollingVineModel.refit`, triggered by :meth:`should_refit`
  according to ``refit_frequency`` (1 = refit on every new observation).

``step`` combines them for sequential/"live" use and ``run`` replays a whole history
(optionally in parallel and with a resumable checkpoint file). Both use the same
schedule and give identical results.

Timestamp convention: the result stamped ``t`` is fitted on the ``window``
observations ending at ``t`` inclusive, i.e. it only uses information available at
the close of day ``t``. The marginal transform is re-estimated inside every window.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

from vine_risk.config import Config
from vine_risk.copula import VineCopula, VineFitError, VineFitResult
from vine_risk.marginals import EmpiricalMarginal, Marginal

logger = logging.getLogger(__name__)

MarginalFactory = Callable[[], Marginal]


def fit_window(
    returns: pd.DataFrame,
    vine_kwargs: dict | None = None,
    marginal_factory: MarginalFactory = EmpiricalMarginal,
) -> VineFitResult:
    """Fit marginals and a vine on one window; never raises for model failures.

    A failed fit is returned as a ``VineFitResult`` with ``status == "failed"``.
    Invalid input (e.g. NaNs) still raises ``ValueError``.
    """
    idx = returns.index
    meta = {}
    if isinstance(idx, pd.DatetimeIndex):
        meta = dict(window_start=str(idx[0].date()), window_end=str(idx[-1].date()),
                    timestamp=str(idx[-1].date()))
    try:
        u = marginal_factory().fit_transform(returns)
        return VineCopula(**(vine_kwargs or {})).fit(u, returns).summary()
    except VineFitError as e:
        logger.warning("Vine fit failed at %s: %s", meta.get("timestamp"), e)
        return VineFitResult.failed([str(c) for c in returns.columns], len(returns), str(e), **meta)


def _fit_task(args: tuple) -> VineFitResult:  # top-level so it can be pickled
    return fit_window(*args)


class RollingVineModel:
    """Rolling vine copula estimator.

    Args:
        window: number of observations per fit.
        refit_frequency: refit every this many new observations (1 = daily).
        vine_kwargs: keyword arguments for :class:`VineCopula`.
        marginal_factory: zero-argument callable returning a fresh :class:`Marginal`.
    """

    def __init__(
        self,
        window: int = 250,
        refit_frequency: int = 1,
        vine_kwargs: dict | None = None,
        marginal_factory: MarginalFactory = EmpiricalMarginal,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if refit_frequency < 1:
            raise ValueError("refit_frequency must be >= 1")
        self.window = window
        self.refit_frequency = refit_frequency
        self.vine_kwargs = dict(vine_kwargs or {})
        self.marginal_factory = marginal_factory
        self.results: list[VineFitResult] = []
        self._buffer: pd.DataFrame | None = None
        self._since_fit = 0

    @classmethod
    def from_config(cls, cfg: Config) -> "RollingVineModel":
        r = cfg.rolling
        return cls(
            window=r.window, refit_frequency=r.refit_frequency,
            vine_kwargs=dict(selection_criterion=r.selection_criterion,
                             truncation_level=r.truncation_level),
        )

    # ---- data update -------------------------------------------------------
    def push(self, timestamp: pd.Timestamp, row: pd.Series) -> None:
        """Append one observation (a return vector) and drop data older than the window."""
        if not row.notna().all():
            raise ValueError(f"Observation at {timestamp} contains missing values.")
        new = pd.DataFrame([row.to_numpy()], index=pd.DatetimeIndex([timestamp]), columns=row.index)
        if self._buffer is None:
            self._buffer = new
        else:
            if list(new.columns) != list(self._buffer.columns):
                raise ValueError("Observation columns do not match the window.")
            if timestamp <= self._buffer.index[-1]:
                raise ValueError(f"Timestamp {timestamp} is not after {self._buffer.index[-1]}.")
            self._buffer = pd.concat([self._buffer, new]).iloc[-self.window:]
        self._since_fit += 1

    @property
    def ready(self) -> bool:
        return self._buffer is not None and len(self._buffer) >= self.window

    # ---- model refit -------------------------------------------------------
    def should_refit(self) -> bool:
        """True once the window is full and ``refit_frequency`` observations have passed."""
        if not self.ready:
            return False
        return not self.results or self._since_fit >= self.refit_frequency

    def refit(self) -> VineFitResult:
        """Fit on the current window and record the result."""
        if not self.ready:
            raise RuntimeError(f"Window not full ({0 if self._buffer is None else len(self._buffer)}/{self.window}).")
        res = fit_window(self._buffer, self.vine_kwargs, self.marginal_factory)
        self.results.append(res)
        self._since_fit = 0
        return res

    def step(self, timestamp: pd.Timestamp, row: pd.Series) -> VineFitResult | None:
        """Push one observation; refit if due. Returns the new result or ``None``."""
        self.push(timestamp, row)
        return self.refit() if self.should_refit() else None

    # ---- batch replay ------------------------------------------------------
    def refit_positions(self, n_obs: int) -> list[int]:
        """Row positions (window end) at which a batch run refits."""
        return list(range(self.window - 1, n_obs, self.refit_frequency))

    def run(
        self,
        returns: pd.DataFrame,
        n_jobs: int = 1,
        checkpoint: str | Path | None = None,
    ) -> list[VineFitResult]:
        """Fit the whole history. Equivalent to calling :meth:`step` row by row.

        Args:
            returns: T x d return matrix with a DatetimeIndex.
            n_jobs: worker processes (windows are independent). With ``n_jobs > 1``
                the caller must be import-safe (``if __name__ == "__main__"``).
            checkpoint: JSON-lines file. Results are appended as they complete and
                timestamps already in the file are skipped, so interrupted runs resume.
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must have a DatetimeIndex.")
        if returns.isna().any().any():
            raise ValueError("returns contain missing values.")
        if len(returns) < self.window:
            raise ValueError(f"Need at least {self.window} observations, got {len(returns)}.")

        positions = self.refit_positions(len(returns))
        done = _read_checkpoint(checkpoint) if checkpoint else {}
        todo = [p for p in positions if str(returns.index[p].date()) not in done]
        logger.info("Rolling fit: %d windows, %d already done", len(positions), len(positions) - len(todo))

        tasks = [(returns.iloc[p - self.window + 1: p + 1], self.vine_kwargs, self.marginal_factory)
                 for p in todo]
        if n_jobs > 1 and tasks:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                it = ex.map(_fit_task, tasks, chunksize=max(1, len(tasks) // (n_jobs * 8)))
                new = self._collect(it, checkpoint, len(tasks))
        else:
            new = self._collect((_fit_task(t) for t in tasks), checkpoint, len(tasks))

        for r in new:
            done[r.timestamp] = r
        self.results = [done[str(returns.index[p].date())] for p in positions]
        return self.results

    @staticmethod
    def _collect(it, checkpoint, total: int) -> list[VineFitResult]:
        out = []
        for i, res in enumerate(it, start=1):
            out.append(res)
            if checkpoint:
                with open(checkpoint, "a") as f:
                    f.write(res.to_json() + "\n")
            if i % 100 == 0 or i == total:
                logger.info("Fitted %d/%d windows", i, total)
        return out


def _read_checkpoint(path: str | Path) -> dict[str, VineFitResult]:
    p = Path(path)
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = VineFitResult.from_json(line)
            out[r.timestamp] = r
    return out


def results_to_frames(results: list[VineFitResult]) -> dict[str, pd.DataFrame]:
    """Flatten results into tidy tables keyed by timestamp.

    * ``fits``: one row per timestamp (metadata and model diagnostics);
    * ``pair_copulas``: one row per timestamp x pair-copula;
    * ``pairwise``: one row per timestamp x asset pair (tau, Spearman, Pearson).
    """
    fits, pcs, pw = [], [], []
    for r in results:
        fits.append(dict(
            timestamp=r.timestamp, window_start=r.window_start, window_end=r.window_end,
            n_obs=r.n_obs, n_assets=r.n_assets, loglik=r.loglik, aic=r.aic, bic=r.bic,
            n_params=r.n_params, truncation_level=r.truncation_level, status=r.status,
            message=r.message, order=r.order,
        ))
        for p in r.pair_copulas:
            pcs.append(dict(
                timestamp=r.timestamp, tree=p.tree, edge=p.edge,
                conditioned=list(p.conditioned), conditioning=list(p.conditioning),
                family=p.family, rotation=p.rotation, parameters=p.parameters, tau=p.tau,
                lower_tail=p.lower_tail, upper_tail=p.upper_tail, n_params=p.n_params,
                loglik=p.loglik, aic=p.aic, bic=p.bic,
            ))
        pw.extend({"timestamp": r.timestamp, **row} for row in r.pairwise)
    return {"fits": pd.DataFrame(fits), "pair_copulas": pd.DataFrame(pcs), "pairwise": pd.DataFrame(pw)}


def save_results(results: list[VineFitResult], out_dir: str | Path) -> None:
    """Write the three tidy tables as parquet files into ``out_dir``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in results_to_frames(results).items():
        df.to_parquet(d / f"{name}.parquet", index=False)
