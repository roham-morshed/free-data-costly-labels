"""Shared pytest fixtures and import-path setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep BLAS single-threaded inside tests so results are bit-reproducible.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


@pytest.fixture
def separable_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A trivially separable 2-class time-series problem (sine vs. sawtooth-ish).

    Shape convention is aeon's: ``(n_cases, n_channels, n_timepoints)``.
    """
    rng = np.random.default_rng(0)
    n, length = 120, 64
    t = np.linspace(0, 4 * np.pi, length)

    def make(n_each: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
        a = np.sin(t)[None, :] + 0.10 * rng.standard_normal((n_each, length))
        b = np.sign(np.sin(t))[None, :] + 0.10 * rng.standard_normal((n_each, length))
        X = np.concatenate([a, b], axis=0)[:, None, :].astype(np.float64)
        y = np.array(["A"] * n_each + ["B"] * n_each)
        del offset
        return X, y

    Xtr, ytr = make(n, 0)
    Xte, yte = make(n // 2, 1)
    return Xtr, ytr, Xte, yte


@pytest.fixture
def noise_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure noise with random labels: every method must land at chance."""
    rng = np.random.default_rng(7)
    n, length = 160, 48
    Xtr = rng.standard_normal((n, 1, length))
    ytr = np.array(["A", "B"] * (n // 2))
    rng.shuffle(ytr)
    Xte = rng.standard_normal((200, 1, length))
    yte = np.array(["A", "B"] * 100)
    rng.shuffle(yte)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# End-to-end scaffolding: a synthetic dataset pushed through the real cache
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point the whole stack at a temporary data + feature cache."""
    from tscfree import data as _data

    dcache = tmp_path / "data"
    fcache = tmp_path / "features"
    dcache.mkdir(parents=True, exist_ok=True)
    fcache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TSCFREE_DATA_CACHE", str(dcache))
    monkeypatch.setenv("TSCFREE_FEATURES", str(fcache))
    _data._load_cached.cache_clear()
    yield dcache, fcache
    _data._load_cached.cache_clear()


def build_ctx(
    name,
    Xtr,
    ytr,
    Xte,
    yte,
    k=5,
    seed=0,
    n_kernels=500,
    transform="minirocket",
    transform_seed=0,
    neighbors=(7,),
    guard=True,
):
    """Materialise a real MethodContext from arrays, via the real cache path."""
    from tscfree.config import load_config
    from tscfree.data import load, save_dataset
    from tscfree.features import build_features, build_unsup_cache, load_features, load_unsup
    from tscfree.methods import MethodContext
    from tscfree.splits import labeled_unlabeled_split
    from tscfree.utils import GuardedLabels

    save_dataset(name, Xtr, ytr, Xte, yte, source="synthetic")
    build_features(name, transform, n_kernels, transform_seed)
    build_unsup_cache(name, transform, n_kernels, transform_seed, neighbor_values=tuple(neighbors))

    ds = load(name)
    fb = load_features(name, transform, n_kernels, transform_seed, mmap=False)
    ub = load_unsup(name, transform, n_kernels, transform_seed, tuple(neighbors), mmap=False)
    L, U = labeled_unlabeled_split(ds.y_train, k=k, seed=seed, dataset=name)
    cfg = load_config()
    return MethodContext(
        dataset=name,
        k=k,
        seed=seed,
        X_train=ds.X_train,
        X_test=ds.X_test,
        L_idx=L,
        U_idx=U,
        y_L=ds.y_train[L],
        classes=np.unique(ds.y_train),
        F_train=fb.F_train,
        F_test=fb.F_test,
        Z_train=ub.Z_train,
        Z_test=ub.Z_test,
        graphs=ub.graphs,
        cfg=cfg.raw,
        y_train_guarded=GuardedLabels(ds.y_train, L) if guard else None,
        D_dtw=None,
        transform=transform,
        n_kernels=n_kernels,
        transform_seed=transform_seed,
    ), ds
