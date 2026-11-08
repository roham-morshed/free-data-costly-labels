"""Correctness gate 10: the same cell, run twice, must give bit-identical numbers.

If it does not, no table in the paper is reproducible and no comparison between
methods is trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_ctx

from tscfree.features import build_features, build_unsup_cache, load_features, load_unsup
from tscfree.methods import METHOD_IDS, run_method
from tscfree.splits import labeled_unlabeled_split
from tscfree.utils import rng_for, seed_from, sha256_array

CHEAP = [m for m in METHOD_IDS if m != "1nn_dtw"]


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    length = 48
    t = np.linspace(0, 4 * np.pi, length)
    a = np.sin(t)[None, :] + 0.35 * rng.standard_normal((60, length))
    b = np.cos(2 * t)[None, :] + 0.35 * rng.standard_normal((60, length))
    Xtr = np.concatenate([a, b])[:, None, :]
    ytr = np.array(["A"] * 60 + ["B"] * 60)
    a2 = np.sin(t)[None, :] + 0.35 * rng.standard_normal((25, length))
    b2 = np.cos(2 * t)[None, :] + 0.35 * rng.standard_normal((25, length))
    Xte = np.concatenate([a2, b2])[:, None, :]
    yte = np.array(["A"] * 25 + ["B"] * 25)
    return Xtr, ytr, Xte, yte


@pytest.mark.parametrize("method", CHEAP)
def test_method_is_bit_reproducible(tmp_cache, method: str) -> None:
    ctx, _ = build_ctx("Determinism", *_data(), k=5, seed=3)
    a = run_method(method, ctx)
    b = run_method(method, ctx)
    assert np.array_equal(a.y_pred, b.y_pred)
    assert a.extra["train_signature"] == b.extra["train_signature"]
    assert a.n_pseudo_selected == b.n_pseudo_selected
    assert a.safe_ssl_fired == b.safe_ssl_fired


def test_transform_is_bit_reproducible(tmp_cache) -> None:
    from tscfree.data import save_dataset

    Xtr, ytr, Xte, yte = _data()
    save_dataset("Det2", Xtr, ytr, Xte, yte, source="synthetic")
    build_features("Det2", "minirocket", 400, 0)
    h1 = sha256_array(np.asarray(load_features("Det2", "minirocket", 400, 0, mmap=False).F_train))
    build_features("Det2", "minirocket", 400, 0, force=True)
    h2 = sha256_array(np.asarray(load_features("Det2", "minirocket", 400, 0, mmap=False).F_train))
    assert h1 == h2


def test_unsup_cache_is_bit_reproducible(tmp_cache) -> None:
    from tscfree.data import save_dataset

    Xtr, ytr, Xte, yte = _data()
    save_dataset("Det3", Xtr, ytr, Xte, yte, source="synthetic")
    build_features("Det3", "minirocket", 400, 0)
    build_unsup_cache("Det3", "minirocket", 400, 0, neighbor_values=(7,))
    a = load_unsup("Det3", "minirocket", 400, 0, (7,), mmap=False)
    build_unsup_cache("Det3", "minirocket", 400, 0, neighbor_values=(7,), force=True)
    b = load_unsup("Det3", "minirocket", 400, 0, (7,), mmap=False)
    assert sha256_array(np.asarray(a.Z_train)) == sha256_array(np.asarray(b.Z_train))
    assert (a.graphs["cos7"] != b.graphs["cos7"]).nnz == 0


def test_splits_are_order_independent() -> None:
    y = np.array(["a"] * 40 + ["b"] * 35 + ["c"] * 30)
    first = labeled_unlabeled_split(y, k=5, seed=2, dataset="D")
    for _ in range(3):
        again = labeled_unlabeled_split(y, k=5, seed=2, dataset="D")
        assert np.array_equal(first[0], again[0])
        assert np.array_equal(first[1], again[1])


def test_rng_for_is_keyed_not_sequential() -> None:
    a = rng_for("m", "D", 5, 0).standard_normal(4)
    b = rng_for("m", "D", 5, 0).standard_normal(4)
    c = rng_for("m", "D", 5, 1).standard_normal(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_from_is_stable_and_in_range() -> None:
    s = seed_from("x", 1, 2)
    assert s == seed_from("x", 1, 2)
    assert 0 <= s < 2**31 - 1
    assert seed_from("x", 1, 2) != seed_from("x", 1, 3)


def test_run_cell_is_reproducible(tmp_cache, tmp_path) -> None:
    """Full cell-level determinism, through the real result-file path."""
    from tscfree.config import load_config
    from tscfree.data import save_dataset
    from tscfree.experiment import run_cell
    from tscfree.utils import read_json

    Xtr, ytr, Xte, yte = _data()
    save_dataset("CellDet", Xtr, ytr, Xte, yte, source="synthetic")
    build_features("CellDet", "minirocket", 400, 0)
    build_unsup_cache("CellDet", "minirocket", 400, 0, neighbor_values=(7,))

    cfg = load_config()
    out = tmp_path / "cells"
    # the real config object, with n_kernels redirected through the overrides hook
    r1 = run_cell(
        "CellDet", 5, 0, CHEAP, cfg, 0.9, out, tmp_path / "fail", force=True,
        overrides={"features": {"n_kernels": 400}},
    )
    p1 = read_json(r1["path"])
    r2 = run_cell(
        "CellDet", 5, 0, CHEAP, cfg, 0.9, out, tmp_path / "fail", force=True,
        overrides={"features": {"n_kernels": 400}},
    )
    p2 = read_json(r2["path"])

    def strip_times(payload):
        rows = []
        for r in payload["rows"]:
            rr = {k: v for k, v in r.items() if not k.endswith("seconds")}
            rr.pop("versions", None)
            rows.append(rr)
        return rows

    assert p1["failed_methods"] == [] == p2["failed_methods"]
    assert strip_times(p1) == strip_times(p2)
