"""Correctness gate 7: LFA behaves as a label-free structure detector."""

from __future__ import annotations

import numpy as np

from tscfree.lfa import (
    LFA_COMPONENTS,
    composite_lfa,
    compute_lfa_components,
    knn_cluster_purity,
    zscore,
)


def _blobs(n_per: int, C: int, dim: int, sep: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((C, dim)) * sep
    return np.vstack([centres[c] + rng.standard_normal((n_per, dim)) for c in range(C)])


def test_lfa_high_on_separated_blobs_low_on_single_blob() -> None:
    good = _blobs(60, 3, 20, sep=12.0, seed=0)
    rng = np.random.default_rng(1)
    bad = rng.standard_normal((180, 20))

    g = compute_lfa_components(good, n_classes=3, seed=0, pca_dims=10, n_bootstrap=5)
    b = compute_lfa_components(bad, n_classes=3, seed=0, pca_dims=10, n_bootstrap=5)

    assert g.sil > b.sil
    assert g.chi > b.chi
    assert g.stab > b.stab
    assert g.knn_purity_proxy > b.knn_purity_proxy
    assert g.sil > 0.5 and b.sil < 0.2


def test_lfa_is_permutation_invariant() -> None:
    X = _blobs(50, 3, 15, sep=8.0, seed=2)
    a = compute_lfa_components(X, n_classes=3, seed=0, pca_dims=10, n_bootstrap=5)
    perm = np.random.default_rng(9).permutation(X.shape[0])
    b = compute_lfa_components(X[perm], n_classes=3, seed=0, pca_dims=10, n_bootstrap=5)
    for name in LFA_COMPONENTS:
        va, vb = getattr(a, name), getattr(b, name)
        assert abs(va - vb) < 5e-2, f"{name}: {va} vs {vb}"


def test_lfa_uses_no_labels() -> None:
    """Same features, different (fake) labels -> identical LFA. Sanity by construction:
    the signature does not accept labels at all, only the class count."""
    import inspect

    params = set(inspect.signature(compute_lfa_components).parameters)
    assert "y" not in params and "labels" not in params and "y_true" not in params
    assert "n_classes" in params


def test_lfa_determinism() -> None:
    X = _blobs(40, 2, 12, sep=6.0, seed=3)
    a = compute_lfa_components(X, n_classes=2, seed=5, pca_dims=8, n_bootstrap=4)
    b = compute_lfa_components(X, n_classes=2, seed=5, pca_dims=8, n_bootstrap=4)
    assert a.as_dict() == b.as_dict()


def test_knn_purity_extremes() -> None:
    X = np.vstack([np.zeros((20, 3)) + 100.0, np.zeros((20, 3)) - 100.0])
    lab = np.array([0] * 20 + [1] * 20)
    assert knn_cluster_purity(X, lab, k=5) == 1.0


def test_zscore_constant_column() -> None:
    assert np.allclose(zscore(np.full(5, 3.0)), 0.0)


def test_composite_lfa_orders_datasets() -> None:
    table = {
        "sil": np.array([0.8, 0.1, 0.4]),
        "chi": np.array([9.0, 1.0, 5.0]),
        "stab": np.array([0.9, 0.1, 0.5]),
        "knn_purity_proxy": np.array([0.95, 0.4, 0.7]),
    }
    lfa = composite_lfa(table)
    assert lfa[0] > lfa[2] > lfa[1]
    assert abs(float(np.mean(lfa))) < 1e-12  # z-scored -> zero mean


def test_composite_lfa_missing_component() -> None:
    import pytest

    with pytest.raises(KeyError):
        composite_lfa({"sil": np.array([1.0, 2.0])})
