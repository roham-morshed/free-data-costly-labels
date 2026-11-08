"""Typed access to ``configs/experiment.yaml``.

Everything the paper reports as a constant lives in the YAML; this module is a
thin, validated accessor so a typo in a key fails loudly at start-up rather
than silently falling back to a default halfway through a 14,000-cell run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Dict-backed config with attribute-style section access."""

    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __getitem__(self, key: str) -> Any:
        if key not in self.raw:
            raise KeyError(f"Missing config section {key!r} in {self.path}")
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- convenience --------------------------------------------------------
    def path_for(self, name: str) -> Path:
        """Absolute path for one of the entries under ``paths:``."""
        rel = self["paths"][name]
        p = Path(rel)
        return p if p.is_absolute() else repo_root() / p

    @property
    def label_budgets(self) -> list[int]:
        return list(self["grid"]["label_budgets"])

    @property
    def seeds(self) -> list[int]:
        return list(self["grid"]["seeds"])

    @property
    def methods(self) -> list[str]:
        return list(self["grid"]["methods"])


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the experiment configuration."""
    p = Path(path) if path is not None else repo_root() / "configs" / "experiment.yaml"
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    required = {"paths", "grid", "features", "ridge", "safe_ssl", "lfa", "stats"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Config {p} is missing required sections: {sorted(missing)}")
    return Config(raw=raw, path=p)


@lru_cache(maxsize=8)
def load_datasets_yaml(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/datasets.yaml`` (written by scripts/download_datasets.py)."""
    p = Path(path) if path is not None else repo_root() / "configs" / "datasets.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist -- run `python scripts/download_datasets.py` first."
        )
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dataset_names(path: str | Path | None = None) -> list[str]:
    """The final, ordered list of selected dataset names."""
    return list(load_datasets_yaml(path)["datasets"].keys())
