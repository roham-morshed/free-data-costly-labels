"""Correctness gate 5: methods must be right on problems whose answer we know.

Two poles:

* **Separable data.**  Every method must reach >= 95% test accuracy with k=5.
  A method that cannot solve a trivially separable problem is broken.
* **Pure noise with random labels.**  Every method must land inside a binomial
  confidence interval of chance.  *A method that beats chance on noise has a
  leak* -- there is no signal to find.

Plus a check that our precomputed-graph shortcut for LabelSpreading is exactly
equivalent to sklearn's own ``kernel='knn'`` path.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_ctx

from tscfree.methods import METHOD_IDS, run_method
from tscfree.metrics import binomial_ci

ALL = list(METHOD_IDS)


# ---------------------------------------------------------------------------
# Separable
# ---------------------------------------------------------------------------


def _separable(seed: int = 0, n: int = 60, n_test: int = 40, length: int = 64):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)

    def gen(m: int, r: np.random.Generator):
        a = np.sin(t)[None, :] + 0.10 * r.standard_normal((m, length))
        b = np.sign(np.sin(t))[None, :] + 0.10 * r.standard_normal((m, length))
        X = np.concatenate([a, b])[:, None, :]
        y = np.array(["A"] * m + ["B"] * m)
        return X, y

    Xtr, ytr = gen(n, rng)
    Xte, yte = gen(n_test, rng)
    return Xtr, ytr, Xte, yte


@pytest.mark.parametrize("method", ALL)
def test_separable_problem_is_solved(tmp_cache, method: str) -> None:
    Xtr, ytr, Xte, yte = _separable()
    ctx, ds = build_ctx("Separable", Xtr, ytr, Xte, yte, k=5, seed=0)
    if method == "1nn_dtw":
        import dataclasses

        from aeon.distances import pairwise_distance

        D = pairwise_distance(ds.X_test, ds.X_train, method="dtw", window=0.1)
        ctx = dataclasses.replace(ctx, D_dtw=D.astype(np.float32))
    res = run_method(method, ctx)
    acc = float(np.mean(res.y_pred == ds.y_test))
    assert acc >= 0.95, f"{method} only reached {acc:.3f} on a separable problem"


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def _noise(seed: int = 11, n: int = 120, n_test: int = 200, length: int = 48):
    rng = np.random.default_rng(seed)
    Xtr = rng.standard_normal((n, 1, length))
    ytr = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    rng.shuffle(ytr)
    Xte = rng.standard_normal((n_test, 1, length))
    yte = np.array(["A"] * (n_test // 2) + ["B"] * (n_test // 2))
    rng.shuffle(yte)
    return Xtr, ytr, Xte, yte


@pytest.mark.parametrize("method", [m for m in ALL if m != "1nn_dtw"])
def test_noise_stays_at_chance(tmp_cache, method: str) -> None:
    Xtr, ytr, Xte, yte = _noise()
    accs = []
    for seed in (0, 1, 2):
        ctx, ds = build_ctx(f"Noise{seed}", Xtr, ytr, Xte, yte, k=5, seed=seed)
        res = run_method(method, ctx)
        accs.append(float(np.mean(res.y_pred == ds.y_test)))
    mean_acc = float(np.mean(accs))
    n_total = len(yte) * len(accs)
    lo, hi = binomial_ci(int(round(mean_acc * n_total)), n_total, alpha=0.001)
    assert lo <= 0.5 <= hi or mean_acc <= 0.5, (
        f"{method} reached {mean_acc:.3f} on pure noise with random labels "
        f"(99.9% CI [{lo:.3f}, {hi:.3f}]) -- that is a leak, not a method."
    )


# ---------------------------------------------------------------------------
# The precomputed-graph shortcut must be exact
# ---------------------------------------------------------------------------


def test_precomputed_graph_matches_sklearn_knn_kernel() -> None:
    from sklearn.semi_supervised import LabelSpreading

    from tscfree.features import knn_connectivity_graph
    from tscfree.methods import run_label_spreading

    rng = np.random.default_rng(0)
    Z = np.vstack(
        [rng.standard_normal((40, 6)) + 5.0, rng.standard_normal((40, 6)) - 5.0]
    )
    y = np.full(80, -1)
    y[[0, 1, 2, 40, 41, 42]] = [0, 0, 0, 1, 1, 1]

    ref = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100, tol=1e-3)
    ref.fit(Z, y)

    W = knn_connectivity_graph(Z, 7)
    dist, classes = run_label_spreading(W, y, alpha=0.2, max_iter=100, tol=1e-3)

    assert np.array_equal(classes, ref.classes_)
    assert np.allclose(dist, ref.label_distributions_, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Structural sanity of the three tiers
# ---------------------------------------------------------------------------


def test_sup_tf_and_sup_strict_use_different_features(tmp_cache) -> None:
    """The two supervised tiers must actually differ, or RQ2 is vacuous."""
    Xtr, ytr, Xte, yte = _separable(seed=4)
    ctx, _ = build_ctx("Tiers", Xtr, ytr, Xte, yte, k=5, seed=0)
    a = run_method("sup_strict", ctx)
    b = run_method("sup_tf", ctx)
    assert a.extra["train_signature"] != b.extra["train_signature"]


def test_ssl_methods_actually_select_pseudo_labels(tmp_cache) -> None:
    Xtr, ytr, Xte, yte = _separable(seed=5)
    ctx, _ = build_ctx("Pseudo", Xtr, ytr, Xte, yte, k=5, seed=0)
    assert run_method("label_spread", ctx).n_pseudo_selected == len(ctx.U_idx)
    safe = run_method("safe_ssl", ctx)
    assert safe.n_pseudo_selected > 0 or safe.safe_ssl_fired
    assert run_method("self_train", ctx).n_pseudo_selected > 0


def test_augment_multiplies_the_labeled_set(tmp_cache) -> None:
    Xtr, ytr, Xte, yte = _separable(seed=6)
    ctx, _ = build_ctx("Aug", Xtr, ytr, Xte, yte, k=5, seed=0)
    res = run_method("augment", ctx)
    n_copies = int(ctx.cfg["augment"]["n_copies"])
    assert res.extra["n_augmented"] == len(ctx.L_idx) * (n_copies + 1)


def test_safe_ssl_valve_reports_a_decision(tmp_cache) -> None:
    Xtr, ytr, Xte, yte = _separable(seed=7)
    ctx, _ = build_ctx("Valve", Xtr, ytr, Xte, yte, k=5, seed=0)
    res = run_method("safe_ssl", ctx)
    assert res.safe_ssl_fired in (True, False)
    assert res.extra["valve"]["status"] in {"ssl_kept", "reverted_to_supervised"}


def test_safe_ssl_valve_abstains_at_k_equals_one(tmp_cache) -> None:
    Xtr, ytr, Xte, yte = _separable(seed=8)
    ctx, _ = build_ctx("ValveK1", Xtr, ytr, Xte, yte, k=1, seed=0)
    res = run_method("safe_ssl", ctx)
    assert res.extra["valve"]["status"] == "abstain_k_too_small"
    assert res.safe_ssl_fired is False
