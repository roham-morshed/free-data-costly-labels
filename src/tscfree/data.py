"""Local dataset cache.

``scripts/download_datasets.py`` is the **only** place in the repository that
touches the network.  Everything else reads from ``data/cache/`` through
:func:`load`, so a run is reproducible offline and can never silently pick up a
different version of an archive.

On-disk layout::

    data/cache/<Name>/train.npz   X (n, 1, L) float64, y (n,) unicode
    data/cache/<Name>/test.npz
    data/cache/<Name>/meta.json
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .splits import class_counts
from .utils import read_json, sha256_array, write_json


@dataclass
class Dataset:
    """One dataset with its official train/test split."""

    name: str
    X_train: np.ndarray  # (n_train, 1, length)
    y_train: np.ndarray  # (n_train,)
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def length(self) -> int:
        return int(self.X_train.shape[-1])

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y_train)))


def cache_dir(root: str | Path | None = None) -> Path:
    """Resolve the dataset cache root.

    ``TSCFREE_DATA_CACHE`` overrides the configured path; the test-suite uses it
    to point the whole stack at a temporary directory without touching the real
    cache.
    """
    if root is not None:
        return Path(root)
    env = os.environ.get("TSCFREE_DATA_CACHE")
    if env:
        return Path(env)
    return load_config().path_for("data_cache")


def is_znormalized(X: np.ndarray, mean_tol: float = 0.05, std_tol: float = 0.05) -> bool:
    """Per-series mean ~ 0 and std ~ 1 for (almost) every series.

    UCR series are conventionally z-normalised already; we detect it rather than
    assume it, record the flag in ``meta.json`` and do **not** re-normalise by
    default (config switch ``preprocess.renormalize``).
    """
    Xf = np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)
    m = np.abs(Xf.mean(axis=1))
    s = np.abs(Xf.std(axis=1) - 1.0)
    return bool(np.mean(m < mean_tol) > 0.95 and np.mean(s < std_tol) > 0.95)


def to_aeon_3d(X: np.ndarray) -> np.ndarray:
    """Coerce to aeon's ``(n_cases, n_channels, n_timepoints)`` float64 layout."""
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"expected 2D or 3D array, got shape {X.shape}")
    return np.ascontiguousarray(X, dtype=np.float64)


def save_dataset(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    source: str,
    root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one dataset to the cache and return its metadata dict."""
    d = cache_dir(root) / name
    d.mkdir(parents=True, exist_ok=True)
    X_train = to_aeon_3d(X_train)
    X_test = to_aeon_3d(X_test)
    y_train = np.asarray(y_train).astype(str)
    y_test = np.asarray(y_test).astype(str)

    np.savez_compressed(d / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(d / "test.npz", X=X_test, y=y_test)

    meta: dict[str, Any] = {
        "name": name,
        "source": source,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_channels": int(X_train.shape[1]),
        "length": int(X_train.shape[2]),
        "n_classes": int(len(np.unique(y_train))),
        "classes": sorted(np.unique(y_train).tolist()),
        "class_counts_train": class_counts(y_train),
        "class_counts_test": class_counts(y_test),
        "min_class_count_train": int(min(class_counts(y_train).values())),
        "class_balance_train": float(
            min(class_counts(y_train).values()) / max(class_counts(y_train).values())
        ),
        "znormalized_train": is_znormalized(X_train),
        "znormalized_test": is_znormalized(X_test),
        "sha256_X_train": sha256_array(X_train),
        "sha256_y_train": sha256_array(y_train),
        "sha256_X_test": sha256_array(X_test),
        "sha256_y_test": sha256_array(y_test),
        "has_nan": bool(np.isnan(X_train).any() or np.isnan(X_test).any()),
    }
    if extra:
        meta.update(extra)
    write_json(d / "meta.json", meta)
    return meta


def available(root: str | Path | None = None) -> list[str]:
    """Names present in the local cache (complete triples only)."""
    d = cache_dir(root)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.is_dir() and (p / "train.npz").exists() and (p / "test.npz").exists() and (
            p / "meta.json"
        ).exists():
            out.append(p.name)
    return out


def load_meta(name: str, root: str | Path | None = None) -> dict[str, Any]:
    return read_json(cache_dir(root) / name / "meta.json")


@lru_cache(maxsize=4)
def _load_cached(name: str, root_str: str) -> Dataset:
    d = Path(root_str) / name
    if not d.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} is not in the local cache ({d}). "
            "Run `python scripts/download_datasets.py` first -- load() never uses the network."
        )
    tr = np.load(d / "train.npz", allow_pickle=False)
    te = np.load(d / "test.npz", allow_pickle=False)
    meta = read_json(d / "meta.json")
    return Dataset(
        name=name,
        X_train=np.ascontiguousarray(tr["X"], dtype=np.float64),
        y_train=tr["y"].astype(str),
        X_test=np.ascontiguousarray(te["X"], dtype=np.float64),
        y_test=te["y"].astype(str),
        meta=meta,
    )


def load(name: str, root: str | Path | None = None) -> Dataset:
    """Load a dataset from the local cache.  Never hits the network."""
    return _load_cached(name, str(cache_dir(root)))
