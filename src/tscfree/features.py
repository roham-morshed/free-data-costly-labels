"""Random convolutional kernel features, cached.

The single most important performance fact of this study:

    the MiniRocket transform is *unsupervised* -- it never sees a label.

Therefore the transform fitted on ``L u U`` is the same object for **every**
label budget ``k``, **every** evaluation seed and **every** method.  We compute
it once per ``(dataset, transform, n_kernels, transform_seed)`` and reuse it
14,000 times.  That is what turns a multi-day study into a few hours, and it is
also the structural argument of the paper: unlabeled features are free.

What is *not* cached is the ``SUP-strict`` transform, which is fitted on
``X_train[L]`` only and therefore depends on ``(k, seed)``.  It is recomputed
per cell -- cheap, because it is fitted on ``k * C`` series.

Cache layout::

    features/<Name>/<transform>_n<kernels>_s<seed>/F_train.npy   (n_train, D) float32
                                                  /F_test.npy    (n_test,  D) float32
                                                  /manifest.json
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .data import Dataset, load, load_meta
from .utils import Timer, package_versions, read_json, save_npy, sha256_obj, write_json

TRANSFORMS = ("minirocket", "rocket", "multirocket")


def make_transform(name: str, n_kernels: int, seed: int, n_jobs: int = 1) -> Any:
    """Instantiate one of the three random-kernel transforms.

    ``n_kernels`` semantics differ slightly between them (MultiRocket emits
    ``4`` features per kernel), so the resulting feature dimensionality is
    recorded in the manifest rather than assumed.
    """
    from aeon.transformations.collection.convolution_based import (
        MiniRocket,
        MultiRocket,
        Rocket,
    )

    name = name.lower()
    if name == "minirocket":
        return MiniRocket(n_kernels=n_kernels, random_state=seed, n_jobs=n_jobs)
    if name == "rocket":
        return Rocket(n_kernels=n_kernels, random_state=seed, n_jobs=n_jobs)
    if name == "multirocket":
        return MultiRocket(n_kernels=n_kernels, random_state=seed, n_jobs=n_jobs)
    raise ValueError(f"unknown transform {name!r}; expected one of {TRANSFORMS}")


@dataclass
class FeatureBundle:
    """Features of one dataset under one transform configuration."""

    name: str
    F_train: np.ndarray  # (n_train, D)
    F_test: np.ndarray  # (n_test, D)
    manifest: dict[str, Any]

    @property
    def dim(self) -> int:
        return int(self.F_train.shape[1])


def cache_key(transform: str, n_kernels: int, seed: int) -> str:
    return f"{transform}_n{n_kernels}_s{seed}"


def features_root(root: str | Path | None = None) -> Path:
    """Resolve the feature cache root (``TSCFREE_FEATURES`` overrides the config)."""
    if root is not None:
        return Path(root)
    env = os.environ.get("TSCFREE_FEATURES")
    if env:
        return Path(env)
    return load_config().path_for("features")


def feature_dir(
    name: str, transform: str, n_kernels: int, seed: int, root: str | Path | None = None
) -> Path:
    return features_root(root) / name / cache_key(transform, n_kernels, seed)


def _expected_manifest(
    name: str, transform: str, n_kernels: int, seed: int, dtype: str
) -> dict[str, Any]:
    """The part of the manifest that must match for the cache to be valid.

    Includes the *data* hashes, so re-downloading a different version of a
    dataset invalidates its features automatically.
    """
    meta = load_meta(name)
    return {
        "dataset": name,
        "transform": transform,
        "n_kernels": int(n_kernels),
        "transform_seed": int(seed),
        "dtype": dtype,
        "sha256_X_train": meta["sha256_X_train"],
        "sha256_X_test": meta["sha256_X_test"],
    }


def build_features(
    name: str,
    transform: str | None = None,
    n_kernels: int | None = None,
    seed: int | None = None,
    force: bool = False,
    root: str | Path | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Fit the transform on the **whole training set** and cache train+test features.

    Fitting on the whole training set is exactly the ``L u U`` condition of the
    protocol (``L u U == train``), and it is legitimate because no labels are
    involved.  The test set is only ever *transformed*, never fitted on.
    """
    cfg = load_config()
    transform = transform or cfg["features"]["transform"]
    n_kernels = int(n_kernels or cfg["features"]["n_kernels"])
    seed = int(seed if seed is not None else cfg["features"]["transform_seed"])
    dtype = cfg["features"]["dtype"]

    d = feature_dir(name, transform, n_kernels, seed, root)
    man_path = d / "manifest.json"
    expected = _expected_manifest(name, transform, n_kernels, seed, dtype)

    if not force and man_path.exists():
        try:
            existing = read_json(man_path)
            if all(existing.get(k) == v for k, v in expected.items()) and (
                d / "F_train.npy"
            ).exists() and (d / "F_test.npy").exists():
                existing["cache_hit"] = True
                return existing
        except Exception:
            pass  # corrupt manifest -> rebuild

    ds: Dataset = load(name)
    tr = make_transform(transform, n_kernels, seed, n_jobs=n_jobs)

    with Timer() as t_fit:
        tr.fit(ds.X_train)
    with Timer() as t_tr:
        F_train = np.asarray(tr.transform(ds.X_train), dtype=dtype)
    with Timer() as t_te:
        F_test = np.asarray(tr.transform(ds.X_test), dtype=dtype)

    # MiniRocket occasionally emits non-finite values for degenerate constant
    # windows; replace rather than propagate NaNs into every downstream model.
    n_bad = int(np.sum(~np.isfinite(F_train))) + int(np.sum(~np.isfinite(F_test)))
    if n_bad:
        F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
        F_test = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)

    d.mkdir(parents=True, exist_ok=True)
    save_npy(d / "F_train.npy", F_train)
    save_npy(d / "F_test.npy", F_test)

    manifest = dict(expected)
    manifest.update(
        {
            "feature_dim": int(F_train.shape[1]),
            "n_train": int(F_train.shape[0]),
            "n_test": int(F_test.shape[0]),
            "fit_seconds": t_fit.seconds,
            "transform_train_seconds": t_tr.seconds,
            "transform_test_seconds": t_te.seconds,
            "total_seconds": t_fit.seconds + t_tr.seconds + t_te.seconds,
            "n_nonfinite_replaced": n_bad,
            "bytes": int(F_train.nbytes + F_test.nbytes),
            "versions": package_versions(),
            "cache_hit": False,
        }
    )
    manifest["config_hash"] = sha256_obj(expected)
    write_json(man_path, manifest)
    return manifest


def load_features(
    name: str,
    transform: str | None = None,
    n_kernels: int | None = None,
    seed: int | None = None,
    root: str | Path | None = None,
    mmap: bool = True,
) -> FeatureBundle:
    """Load cached features.  Raises if the cache is missing or stale.

    ``mmap=True`` keeps memory flat when many joblib workers open the same
    dataset (the OS page cache is shared).
    """
    cfg = load_config()
    transform = transform or cfg["features"]["transform"]
    n_kernels = int(n_kernels or cfg["features"]["n_kernels"])
    seed = int(seed if seed is not None else cfg["features"]["transform_seed"])
    dtype = cfg["features"]["dtype"]

    d = feature_dir(name, transform, n_kernels, seed, root)
    man_path = d / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(
            f"No feature cache at {d}. Run `python scripts/build_features.py` first."
        )
    manifest = read_json(man_path)
    expected = _expected_manifest(name, transform, n_kernels, seed, dtype)
    stale = {k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v}
    if stale:
        raise ValueError(
            f"Feature cache at {d} is stale (cached vs expected: {stale}). "
            "Rebuild with build_features(..., force=True)."
        )
    mode = "r" if mmap else None
    F_train = np.load(d / "F_train.npy", mmap_mode=mode)
    F_test = np.load(d / "F_test.npy", mmap_mode=mode)
    return FeatureBundle(name=name, F_train=F_train, F_test=F_test, manifest=manifest)


def transform_fit_on(
    X_fit: np.ndarray,
    X_apply: list[np.ndarray],
    transform: str,
    n_kernels: int,
    seed: int,
    dtype: str = "float32",
) -> tuple[list[np.ndarray], float]:
    """Fit a fresh transform on ``X_fit``, apply it to each array in ``X_apply``.

    This is the ``SUP-strict`` path: the transform sees **only** the labeled
    subset.  Never cached, because it depends on which points are labeled.
    """
    tr = make_transform(transform, n_kernels, seed, n_jobs=1)
    with Timer() as t:
        tr.fit(X_fit)
        out = [np.asarray(tr.transform(X), dtype=dtype) for X in X_apply]
    out = [np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0) for F in out]
    return out, t.seconds


# ---------------------------------------------------------------------------
# DTW distance cache (for the 1NN-DTW baseline)
# ---------------------------------------------------------------------------


def dtw_path(name: str, window: float | None, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else load_config().path_for("features")
    tag = "full" if window is None else f"w{window:g}"
    return base / name / f"dtw_test_train_{tag}.npy"


def build_dtw_matrix(
    name: str, window: float | None = None, force: bool = False,
    root: str | Path | None = None, n_jobs: int = 1
) -> dict[str, Any]:
    """Cache the full ``(n_test, n_train)`` DTW distance matrix for one dataset.

    DTW distances do not depend on labels, so one matrix serves every
    ``(k, seed)`` cell: 1NN-DTW then reduces to an ``argmin`` over the columns
    belonging to ``L``.  Without this, 1NN-DTW alone would dominate the runtime
    of the whole study.
    """
    from aeon.distances import pairwise_distance

    p = dtw_path(name, window, root)
    meta_p = p.with_suffix(".json")
    if not force and p.exists() and meta_p.exists():
        m = read_json(meta_p)
        m["cache_hit"] = True
        return m

    ds = load(name)
    kwargs: dict[str, Any] = {}
    if window is not None:
        kwargs["window"] = float(window)
    with Timer() as t:
        D = pairwise_distance(
            ds.X_test, ds.X_train, method="dtw", n_jobs=n_jobs, **kwargs
        ).astype(np.float32)
    save_npy(p, D)
    manifest = {
        "dataset": name,
        "window": window,
        "shape": list(D.shape),
        "seconds": t.seconds,
        "bytes": int(D.nbytes),
        "sha256_X_train": load_meta(name)["sha256_X_train"],
        "sha256_X_test": load_meta(name)["sha256_X_test"],
        "cache_hit": False,
    }
    write_json(meta_p, manifest)
    return manifest


def load_dtw_matrix(
    name: str, window: float | None = None, root: str | Path | None = None, mmap: bool = True
) -> np.ndarray:
    p = dtw_path(name, window, root)
    if not p.exists():
        raise FileNotFoundError(f"No DTW cache at {p}; run scripts/build_features.py")
    return np.load(p, mmap_mode="r" if mmap else None)


# ---------------------------------------------------------------------------
# Label-independent (unsupervised) precomputation: scaler + kNN graphs
# ---------------------------------------------------------------------------
#
# The scaler is fitted on ``L u U`` and the graphs are built on ``L u U``.
# ``L u U`` *is* the whole training set, so none of these objects depends on
# which points are labeled, on ``k``, or on the evaluation seed.  Computing
# them once per dataset instead of once per cell is not a shortcut -- it is the
# operational form of the paper's central claim that the unsupervised half of
# the pipeline is free.
#
# The one-time cost is measured and recorded so the RQ6 Pareto figure can
# charge it honestly.


def unsup_dir(
    name: str, transform: str, n_kernels: int, seed: int, root: str | Path | None = None
) -> Path:
    return feature_dir(name, transform, n_kernels, seed, root) / "unsup"


def knn_connectivity_graph(Z: np.ndarray, n_neighbors: int) -> Any:
    """Exactly sklearn's ``kernel='knn'`` affinity for LabelSpreading.

    sklearn builds ``NearestNeighbors(n_neighbors=k).fit(Z)`` and then takes
    ``kneighbors_graph(Z, k, mode='connectivity')``.  Reproduced here so we can
    precompute it once per dataset and hand it to LabelSpreading through a
    callable kernel.  ``tests/test_methods_sanity.py`` asserts the two paths
    give bit-identical label distributions.
    """
    from sklearn.neighbors import NearestNeighbors

    k = int(min(n_neighbors, max(1, Z.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    return nn.kneighbors_graph(Z, k, mode="connectivity")


def cosine_knn_graph(Z: np.ndarray, n_neighbors: int) -> Any:
    """Symmetrised cosine-similarity kNN affinity used by SAFE-SSL (§6 step 2).

    Weights are ``max(0, cosine_similarity)``; negative similarities are clipped
    because the normalised graph Laplacian used by label spreading is only
    defined for non-negative affinities.  Symmetrisation uses the elementwise
    maximum, i.e. an edge survives if *either* endpoint lists the other among
    its k nearest neighbours.
    """
    from scipy import sparse
    from sklearn.neighbors import kneighbors_graph

    k = int(min(n_neighbors, max(1, Z.shape[0] - 1)))
    D = kneighbors_graph(Z, k, mode="distance", metric="cosine", include_self=False)
    W = D.tocoo()
    sim = np.clip(1.0 - W.data, 0.0, None)
    S = sparse.coo_matrix((sim, (W.row, W.col)), shape=W.shape).tocsr()
    S = S.maximum(S.T)
    S.setdiag(0.0)
    S.eliminate_zeros()
    return S


def build_unsup_cache(
    name: str,
    transform: str | None = None,
    n_kernels: int | None = None,
    seed: int | None = None,
    neighbor_values: tuple[int, ...] = (3, 7, 15),
    force: bool = False,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Standardised features + all kNN graphs needed by the SSL methods.

    Nothing here uses a label.
    """
    from scipy import sparse
    from sklearn.preprocessing import StandardScaler

    cfg = load_config()
    transform = transform or cfg["features"]["transform"]
    n_kernels = int(n_kernels or cfg["features"]["n_kernels"])
    seed = int(seed if seed is not None else cfg["features"]["transform_seed"])

    d = unsup_dir(name, transform, n_kernels, seed, root)
    man_path = d / "manifest.json"
    if not force and man_path.exists():
        try:
            m = read_json(man_path)
            need = set(f"knn{k}" for k in neighbor_values) | set(
                f"cos{k}" for k in neighbor_values
            )
            if need.issubset(set(m.get("graphs", []))) and (d / "Z_train.npy").exists():
                m["cache_hit"] = True
                return m
        except Exception:
            pass

    fb = load_features(name, transform, n_kernels, seed, root, mmap=False)
    d.mkdir(parents=True, exist_ok=True)

    with Timer() as t_scale:
        scaler = StandardScaler().fit(fb.F_train)  # fitted on L u U only
        Z_train = scaler.transform(fb.F_train).astype(np.float32)
        Z_test = scaler.transform(fb.F_test).astype(np.float32)
    Z_train = np.nan_to_num(Z_train, nan=0.0, posinf=0.0, neginf=0.0)
    Z_test = np.nan_to_num(Z_test, nan=0.0, posinf=0.0, neginf=0.0)
    save_npy(d / "Z_train.npy", Z_train)
    save_npy(d / "Z_test.npy", Z_test)

    Z64 = np.ascontiguousarray(Z_train, dtype=np.float64)
    graphs: list[str] = []
    graph_seconds: dict[str, float] = {}
    for k in neighbor_values:
        with Timer() as t1:
            G = knn_connectivity_graph(Z64, k)
        sparse.save_npz(d / f"knn{k}.npz", G.tocsr())
        graphs.append(f"knn{k}")
        graph_seconds[f"knn{k}"] = t1.seconds
        with Timer() as t2:
            S = cosine_knn_graph(Z64, k)
        sparse.save_npz(d / f"cos{k}.npz", S.tocsr())
        graphs.append(f"cos{k}")
        graph_seconds[f"cos{k}"] = t2.seconds

    manifest = {
        "dataset": name,
        "transform": transform,
        "n_kernels": n_kernels,
        "transform_seed": seed,
        "graphs": graphs,
        "scale_seconds": t_scale.seconds,
        "graph_seconds": graph_seconds,
        "unsup_seconds_total": t_scale.seconds + float(sum(graph_seconds.values())),
        "n_train": int(Z_train.shape[0]),
        "feature_dim": int(Z_train.shape[1]),
        "bytes": int(Z_train.nbytes + Z_test.nbytes),
        "cache_hit": False,
    }
    write_json(man_path, manifest)
    return manifest


@dataclass
class UnsupBundle:
    """Cached label-independent artefacts for one dataset."""

    Z_train: np.ndarray
    Z_test: np.ndarray
    graphs: dict[str, Any]
    manifest: dict[str, Any]


def load_unsup(
    name: str,
    transform: str | None = None,
    n_kernels: int | None = None,
    seed: int | None = None,
    neighbor_values: tuple[int, ...] = (7,),
    root: str | Path | None = None,
    mmap: bool = True,
) -> UnsupBundle:
    from scipy import sparse

    cfg = load_config()
    transform = transform or cfg["features"]["transform"]
    n_kernels = int(n_kernels or cfg["features"]["n_kernels"])
    seed = int(seed if seed is not None else cfg["features"]["transform_seed"])
    d = unsup_dir(name, transform, n_kernels, seed, root)
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"No unsupervised cache at {d}; run scripts/build_features.py")
    mode = "r" if mmap else None
    graphs: dict[str, Any] = {}
    for k in neighbor_values:
        for prefix in ("knn", "cos"):
            p = d / f"{prefix}{k}.npz"
            if p.exists():
                graphs[f"{prefix}{k}"] = sparse.load_npz(p)
    return UnsupBundle(
        Z_train=np.load(d / "Z_train.npy", mmap_mode=mode),
        Z_test=np.load(d / "Z_test.npy", mmap_mode=mode),
        graphs=graphs,
        manifest=read_json(d / "manifest.json"),
    )
