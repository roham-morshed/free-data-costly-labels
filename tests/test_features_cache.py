"""Correctness gate 6: the feature cache must be transparent and self-invalidating.

The whole study rests on reusing one cached transform across 14,000 cells.  If
the cache ever returned something different from a fresh computation, or failed
to notice that its configuration changed, every number in the paper would be
wrong in a way no other test would catch.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_ctx  # noqa: F401  (imported for symmetry / future use)

from tscfree.data import load, save_dataset
from tscfree.features import (
    build_features,
    build_unsup_cache,
    cosine_knn_graph,
    knn_connectivity_graph,
    load_features,
    load_unsup,
    make_transform,
    transform_fit_on,
)


def _synth(seed: int = 0, n: int = 40, length: int = 40):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    a = np.sin(t)[None, :] + 0.2 * rng.standard_normal((n, length))
    b = np.cos(2 * t)[None, :] + 0.2 * rng.standard_normal((n, length))
    X = np.concatenate([a, b])[:, None, :]
    y = np.array(["A"] * n + ["B"] * n)
    return X, y


def _install(name: str):
    Xtr, ytr = _synth(0)
    Xte, yte = _synth(1, n=20)
    save_dataset(name, Xtr, ytr, Xte, yte, source="synthetic")
    return Xtr, ytr, Xte, yte


def test_cached_equals_freshly_computed(tmp_cache) -> None:
    name = "CacheEq"
    Xtr, _, Xte, _ = _install(name)
    build_features(name, "minirocket", 400, 0)
    fb = load_features(name, "minirocket", 400, 0, mmap=False)

    tr = make_transform("minirocket", 400, 0)
    tr.fit(Xtr)
    F_train = np.asarray(tr.transform(Xtr), dtype="float32")
    F_test = np.asarray(tr.transform(Xte), dtype="float32")

    assert np.allclose(fb.F_train, F_train, atol=0, rtol=0)
    assert np.allclose(fb.F_test, F_test, atol=0, rtol=0)


def test_cache_hit_is_reported_and_does_not_recompute(tmp_cache) -> None:
    name = "CacheHit"
    _install(name)
    m1 = build_features(name, "minirocket", 400, 0)
    assert m1["cache_hit"] is False
    m2 = build_features(name, "minirocket", 400, 0)
    assert m2["cache_hit"] is True
    assert m1["feature_dim"] == m2["feature_dim"]


def test_changing_n_kernels_invalidates(tmp_cache) -> None:
    name = "CacheKernels"
    _install(name)
    build_features(name, "minirocket", 400, 0)
    m = build_features(name, "minirocket", 800, 0)
    assert m["cache_hit"] is False
    a = load_features(name, "minirocket", 400, 0, mmap=False)
    b = load_features(name, "minirocket", 800, 0, mmap=False)
    assert a.dim != b.dim


def test_changing_transform_seed_invalidates(tmp_cache) -> None:
    name = "CacheSeed"
    _install(name)
    build_features(name, "minirocket", 400, 0)
    build_features(name, "minirocket", 400, 1)
    a = load_features(name, "minirocket", 400, 0, mmap=False)
    b = load_features(name, "minirocket", 400, 1, mmap=False)
    assert not np.array_equal(a.F_train, b.F_train)


def test_stale_cache_is_detected_when_the_data_changes(tmp_cache) -> None:
    """Rewrite the dataset under the same name; loading the old features must fail."""
    from tscfree import data as _data

    name = "CacheStale"
    _install(name)
    build_features(name, "minirocket", 400, 0)

    Xtr2, ytr2 = _synth(seed=5)
    Xte2, yte2 = _synth(seed=6, n=20)
    save_dataset(name, Xtr2, ytr2, Xte2, yte2, source="synthetic-v2")
    _data._load_cached.cache_clear()

    with pytest.raises(ValueError, match="stale"):
        load_features(name, "minirocket", 400, 0, mmap=False)


def test_missing_cache_raises_a_useful_error(tmp_cache) -> None:
    _install("CacheMissing")
    with pytest.raises(FileNotFoundError, match="build_features"):
        load_features("CacheMissing", "minirocket", 400, 0, mmap=False)


def test_unsup_cache_scaler_is_fitted_on_train_only(tmp_cache) -> None:
    name = "UnsupScale"
    _install(name)
    build_features(name, "minirocket", 400, 0)
    build_unsup_cache(name, "minirocket", 400, 0, neighbor_values=(7,))
    ub = load_unsup(name, "minirocket", 400, 0, (7,), mmap=False)
    # standardised training features have (near) zero mean / unit variance;
    # the test features do not, because the scaler never saw them.
    col_ok = np.std(np.asarray(load_features(name, "minirocket", 400, 0, mmap=False).F_train),
                    axis=0) > 1e-8
    assert np.abs(np.mean(np.asarray(ub.Z_train)[:, col_ok], axis=0)).max() < 1e-3
    assert np.abs(np.std(np.asarray(ub.Z_train)[:, col_ok], axis=0) - 1).max() < 1e-3
    assert "knn7" in ub.graphs and "cos7" in ub.graphs
    assert ub.graphs["knn7"].shape == (80, 80)


def test_graphs_are_label_independent_and_symmetric(tmp_cache) -> None:
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 8))
    G = knn_connectivity_graph(Z, 5)
    assert G.shape == (50, 50)
    assert G.nnz == 50 * 5
    S = cosine_knn_graph(Z, 5)
    assert (S != S.T).nnz == 0  # symmetric
    assert S.min() >= 0.0  # non-negative affinities


def test_transform_fit_on_uses_only_the_fit_set(tmp_cache) -> None:
    """Fitting on a subset then transforming two arrays == fitting once, applying twice."""
    Xtr, _ = _synth(0)
    Xte, _ = _synth(1, n=20)
    sub = Xtr[:20]
    (Fa, Fb), _ = transform_fit_on(sub, [sub, Xte], "minirocket", 400, 0)
    tr = make_transform("minirocket", 400, 0)
    tr.fit(sub)
    assert np.allclose(Fa, np.asarray(tr.transform(sub), dtype="float32"))
    assert np.allclose(Fb, np.asarray(tr.transform(Xte), dtype="float32"))


def test_load_never_touches_the_network(tmp_cache) -> None:
    _install("Offline")
    ds = load("Offline")
    assert ds.n_train == 80 and ds.n_test == 40
    with pytest.raises(FileNotFoundError, match="download_datasets"):
        load("DefinitelyNotDownloaded")
