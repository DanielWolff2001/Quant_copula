"""Typed configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    frequency: str = "daily"
    start: str = "2005-01-01"
    end: str | None = None
    cache_dir: str = "data/cache"
    max_missing_frac: float = 0.05
    max_ffill_days: int = 3
    source: str = "yahoo"  # "yahoo" or "csv" (see vine_risk.sources)
    csv_path: str | None = None  # file or folder, when source is "csv"

    def __post_init__(self) -> None:
        if self.frequency != "daily":
            raise ValueError("Only daily data is supported.")
        if self.source not in ("yahoo", "csv"):
            raise ValueError("data.source must be 'yahoo' or 'csv'.")


@dataclass(frozen=True)
class RollingConfig:
    window: int = 250
    refit_frequency: int = 1
    truncation_level: int | None = 3  # None = full vine
    selection_criterion: str = "bic"
    n_jobs: int = 1  # worker processes used by RollingVineModel.run
    tail_simulations: int = 16384  # Sobol points for model-implied pairwise dependence
    tail_level: float = 0.05  # q of the finite-level tail coefficients
    marginal: str = "empirical"  # "empirical" (ranks), "garch_t" or "garch_empirical" (GARCH-filtered)
    marginal_lookback: int | None = None  # observations the marginal is fitted on (None: window; GARCH: 1000)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("rolling.window must be >= 2")
        if self.refit_frequency < 1:
            raise ValueError("rolling.refit_frequency must be >= 1")
        if self.n_jobs < 1:
            raise ValueError("rolling.n_jobs must be >= 1")
        if self.marginal not in ("empirical", "garch_t", "garch_empirical"):
            raise ValueError("rolling.marginal must be 'empirical', 'garch_t' or 'garch_empirical'")
        if self.lookback < self.window:
            raise ValueError("rolling.marginal_lookback must be >= rolling.window")

    @property
    def lookback(self) -> int:
        """Observations the marginal model is fitted on. GARCH parameters are poorly determined by a
        short window, so GARCH marginals default to 1000 observations; the copula still uses ``window``."""
        if self.marginal_lookback is not None:
            return self.marginal_lookback
        return self.window if self.marginal == "empirical" else max(self.window, 1000)


@dataclass(frozen=True)
class RiskConfig:
    confidence_level: float = 0.99
    simulations: int = 16384  # Sobol points per window; a power of 2 works best
    seed: int = 42
    weights: list[float] | None = None  # None = equal weights across the assets

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError("risk.confidence_level must be in (0, 1)")
        if self.simulations < 100:
            raise ValueError("risk.simulations must be >= 100")


@dataclass(frozen=True)
class Config:
    assets: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY", "TLT"]
    )
    data: DataConfig = field(default_factory=DataConfig)
    rolling: RollingConfig = field(default_factory=RollingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def load_config(path: str | Path) -> Config:
    """Load a `Config` from a YAML file; missing keys fall back to defaults."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    return Config(
        assets=list(raw.get("assets", Config().assets)),
        data=DataConfig(**raw.get("data", {})),
        rolling=RollingConfig(**raw.get("rolling", {})),
        risk=RiskConfig(**raw.get("risk", {})),
    )
