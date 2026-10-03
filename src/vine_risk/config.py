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

    def __post_init__(self) -> None:
        if self.frequency != "daily":
            raise ValueError("Only daily data is supported.")


@dataclass(frozen=True)
class RollingConfig:
    window: int = 250
    refit_frequency: int = 1


@dataclass(frozen=True)
class RiskConfig:
    confidence_level: float = 0.99
    simulations: int = 10000
    seed: int = 42


@dataclass(frozen=True)
class Config:
    assets: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY", "TLT"]
    )
    data: DataConfig = field(default_factory=DataConfig)
    rolling: RollingConfig = field(default_factory=RollingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file; missing keys fall back to defaults."""
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
