"""Run manifests: a record of what produced the results in a run folder.

Every run folder gets a ``manifest.json`` with the package and library versions, the git
commit, a hash of the pinned dependency list, the full configuration and its hash, and a
fingerprint of the data. It serves two purposes:

* **traceability** - results can be tied to the exact code, settings and data that made them;
* **safety when resuming** - a checkpoint made with a window of 250 days must not be extended
  with a window of 125, or with different prices; :func:`check_resume` refuses to mix them.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from vine_risk import __version__
from vine_risk.config import Config
from vine_risk.copula import VineFitResult
from vine_risk.dependence import pairwise_dependence
from vine_risk.marginals import EmpiricalMarginal

SCHEMA = 1
LIBRARIES = ("numpy", "pandas", "scipy", "pyarrow", "pyvinecopulib", "yfinance", "matplotlib", "networkx",
             "PyYAML", "streamlit", "plotly")
MANIFEST_NAME = "manifest.json"


class ManifestMismatch(RuntimeError):
    """Raised when a run folder was made with settings that differ from the current ones."""


def stable_hash(obj: Any) -> str:
    """SHA-256 of a JSON-serialisable object (keys sorted, so key order does not matter)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def fit_fingerprint(cfg: Config) -> dict[str, Any]:
    """The settings that determine the *fits* (not the risk or monitor settings).

    Two runs with the same fingerprint, on the same data, produce identical fits, so a
    checkpoint may be extended. Changing e.g. the VaR level does not matter; changing the
    window, the vine options or the assets does.
    """
    r, d = cfg.rolling, cfg.data
    return {
        "assets": list(cfg.assets), "window": r.window, "refit_frequency": r.refit_frequency,
        "truncation_level": r.truncation_level, "selection_criterion": r.selection_criterion,
        "tail_simulations": r.tail_simulations, "tail_level": r.tail_level, "seed": cfg.risk.seed,
        "max_missing_frac": d.max_missing_frac, "max_ffill_days": d.max_ffill_days,
    }


def data_fingerprint(returns: pd.DataFrame) -> dict[str, Any]:
    """Shape, date range and a hash of the return matrix."""
    return {
        "assets": [str(c) for c in returns.columns], "n_obs": int(len(returns)),
        "first_date": str(returns.index[0].date()), "last_date": str(returns.index[-1].date()),
        "returns_sha256": hashlib.sha256(np.ascontiguousarray(returns.to_numpy(dtype="float64")).tobytes()).hexdigest(),
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def lock_hash(path: str | Path = "requirements-lock.txt") -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def collect_environment() -> dict[str, Any]:
    """Python, platform, library versions, and the git state of the code."""
    libs = {}
    for name in LIBRARIES:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    commit = _git("rev-parse", "HEAD")
    return {
        "package_version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "libraries": libs,
        "git": None if commit is None else {
            "commit": commit, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        },
        "lock_sha256": lock_hash(),
    }


def build_manifest(cfg: Config, returns: pd.DataFrame, command: Sequence[str] | None = None) -> dict[str, Any]:
    """A fresh manifest for a run on ``returns`` with settings ``cfg``."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fp = fit_fingerprint(cfg)
    return {
        "schema": SCHEMA, "created": now, "updated": now, **collect_environment(),
        "config": asdict(cfg), "config_hash": stable_hash(asdict(cfg)),
        "fit_fingerprint": fp, "fit_hash": stable_hash(fp),
        "data": data_fingerprint(returns), "command": list(command) if command else None, "steps": {},
    }


def read_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    p = Path(run_dir) / MANIFEST_NAME
    return json.loads(p.read_text()) if p.is_file() else None


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest`` into the run folder. If one exists, its creation time and the
    record of earlier steps are kept, so the manifest accumulates over resumed runs."""
    p = Path(run_dir) / MANIFEST_NAME
    old = read_manifest(run_dir)
    if old:
        manifest = {**manifest, "created": old.get("created", manifest["created"]),
                    "steps": {**old.get("steps", {}), **manifest.get("steps", {})}}
    manifest["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1, default=str))
    return p


def record_step(run_dir: str | Path, name: str, info: dict[str, Any]) -> None:
    """Add a record for one pipeline step (what it did, how long it took) to the manifest."""
    m = read_manifest(run_dir)
    if m is None:
        return
    m.setdefault("steps", {})[name] = {**info, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_manifest(run_dir, m)


def check_resume(run_dir: str | Path, cfg: Config) -> list[str]:
    """Check that ``cfg`` is compatible with the fits already stored in ``run_dir``.

    Returns ``[]`` if there is nothing to compare or all fit settings agree; raises
    :class:`ManifestMismatch` listing every difference otherwise.
    """
    old = read_manifest(run_dir)
    if old is None:
        return []
    was, now = old.get("fit_fingerprint", {}), fit_fingerprint(cfg)
    diffs = [f"{k}: stored {was.get(k)!r}, now {now.get(k)!r}" for k in sorted(set(was) | set(now))
             if was.get(k) != now.get(k)]
    if diffs:
        raise ManifestMismatch(
            f"{run_dir} was made with different fit settings; extending it would mix incompatible fits.\n  "
            + "\n  ".join(diffs)
            + "\nUse a different run folder, or delete this one to start over.")
    return []


def verify_history_unchanged(result: VineFitResult, returns: pd.DataFrame, tol: float = 1e-9) -> None:
    """Check that the data behind a stored fit is unchanged.

    Recomputes the empirical Kendall's tau of the stored fit's window from ``returns`` and
    compares with the stored values. Prices from a data vendor can be revised (a split or
    dividend adjustment rescales history), which would make stored fits stale.
    Raises :class:`ManifestMismatch` on a difference.
    """
    ts = pd.Timestamp(result.timestamp)
    if ts not in returns.index:
        raise ManifestMismatch(f"Stored fit dated {ts.date()} has no matching return observation.")
    end = returns.index.get_loc(ts)
    window = returns.iloc[end - result.n_obs + 1: end + 1]
    now = pairwise_dependence(EmpiricalMarginal().fit_transform(window)).set_index(["asset_i", "asset_j"]).tau
    stored = pd.DataFrame(result.pairwise).set_index(["asset_i", "asset_j"]).tau
    err = float((now - stored).abs().max())
    if not err <= tol:
        raise ManifestMismatch(
            f"The data behind the stored fit of {ts.date()} changed (largest difference in Kendall's tau "
            f"{err:.2e}). Prices were probably revised by the data vendor; refit the affected period.")
