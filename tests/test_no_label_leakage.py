"""Correctness gate 1 -- THE MOST IMPORTANT TEST.

Two independent proofs that no method uses the labels of the unlabeled pool:

1. **The guard.**  Every method runs with a :class:`GuardedLabels` view of the
   full training label vector installed in the context.  Reading any position
   in ``U`` raises.  A negative control (a deliberately leaky method) proves the
   guard is not vacuous.

2. **The corruption test** -- stronger, because it does not rely on the leak
   going through the guard object at all.  We scramble the true labels of ``U``
   into nonsense and re-run everything.  If a method's fitted training-side
   state or its predictions change by even one bit, it read those labels.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_ctx

from tscfree.methods import METHOD_IDS, run_method
from tscfree.utils import GuardedLabels, LabelLeakageError

CHEAP = [m for m in METHOD_IDS if m != "1nn_dtw"]


def _data(seed: int = 0):
    """Two well-separated classes -- big enough for a real graph, small enough to be fast."""
    rng = np.random.default_rng(seed)
    length = 48
    t = np.linspace(0, 4 * np.pi, length)
    a = np.sin(t)[None, :] + 0.30 * rng.standard_normal((60, length))
    b = np.cos(2 * t)[None, :] + 0.30 * rng.standard_normal((60, length))
    Xtr = np.concatenate([a, b])[:, None, :]
    ytr = np.array(["A"] * 60 + ["B"] * 60)
    a2 = np.sin(t)[None, :] + 0.30 * rng.standard_normal((30, length))
    b2 = np.cos(2 * t)[None, :] + 0.30 * rng.standard_normal((30, length))
    Xte = np.concatenate([a2, b2])[:, None, :]
    yte = np.array(["A"] * 30 + ["B"] * 30)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# 1. The guard itself
# ---------------------------------------------------------------------------


def test_guard_allows_labeled_and_blocks_unlabeled() -> None:
    y = np.array(["a", "b", "a", "b", "a", "b"])
    g = GuardedLabels(y, allowed_idx=[0, 1])
    assert g[0] == "a" and g[1] == "b"
    assert list(g[np.array([0, 1])]) == ["a", "b"]
    with pytest.raises(LabelLeakageError):
        _ = g[2]
    with pytest.raises(LabelLeakageError):
        _ = g[np.array([1, 4])]
    with pytest.raises(LabelLeakageError):
        _ = g[:]
    with pytest.raises(LabelLeakageError):
        _ = np.asarray(g)


def test_guard_is_not_vacuous_negative_control(tmp_cache) -> None:
    """A deliberately leaky 'method' must be caught by the guard."""
    ctx, _ = build_ctx("GuardCtl", *_data(), k=5, seed=0)

    def leaky(ctx_):
        _ = ctx_.y_train_guarded[ctx_.U_idx]  # the crime
        return None

    with pytest.raises(LabelLeakageError):
        leaky(ctx)


# ---------------------------------------------------------------------------
# 2. Every method, end to end, with the guard installed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", CHEAP)
def test_method_runs_clean_under_the_guard(tmp_cache, method: str) -> None:
    ctx, ds = build_ctx("GuardRun", *_data(), k=5, seed=1)
    res = run_method(method, ctx)
    assert res.y_pred.shape == ds.y_test.shape
    assert set(np.unique(res.y_pred)).issubset(set(np.unique(ds.y_train)))
    assert ctx.y_train_guarded is not None


# ---------------------------------------------------------------------------
# 3. Corrupting the hidden labels must change nothing
# ---------------------------------------------------------------------------


def test_context_carries_no_unguarded_full_label_vector() -> None:
    """Structural check: the only full-length label object in the context is the guard.

    ``y_L`` has length ``k * C``; anything else label-shaped would have length
    ``n_train`` and would be a way around the guard.
    """
    import dataclasses

    from tscfree.methods import MethodContext

    fields = {f.name for f in dataclasses.fields(MethodContext)}
    label_ish = {f for f in fields if f.startswith("y_")}
    assert label_ish == {"y_L", "y_train_guarded"}


@pytest.mark.parametrize("method", CHEAP)
def test_corrupting_the_hidden_labels_changes_nothing(tmp_cache, method: str) -> None:
    """Swap the hidden labels for nonsense; every bit of the output must be identical.

    ``y_L`` (the labels the method is entitled to) is byte-identical between the
    two runs; only the *hidden* half of the guarded vector differs.  A method
    that consulted it would produce a different model.
    """
    import dataclasses

    Xtr, ytr, Xte, yte = _data()
    ctx_a, _ = build_ctx("Corrupt", Xtr, ytr, Xte, yte, k=5, seed=2)

    y_bad = ytr.copy()
    y_bad[ctx_a.U_idx] = "A"  # collapse every hidden label onto one class
    assert np.array_equal(y_bad[ctx_a.L_idx], ytr[ctx_a.L_idx])
    assert not np.array_equal(y_bad, ytr)

    ctx_b = dataclasses.replace(
        ctx_a, y_train_guarded=GuardedLabels(y_bad, ctx_a.L_idx)
    )

    res_a = run_method(method, ctx_a)
    res_b = run_method(method, ctx_b)
    assert np.array_equal(res_a.y_pred, res_b.y_pred)
    assert res_a.extra["train_signature"] == res_b.extra["train_signature"]
    assert res_a.n_pseudo_selected == res_b.n_pseudo_selected
    # and nothing touched the guard at all
    assert ctx_a.y_train_guarded.n_touch == 0
    assert ctx_b.y_train_guarded.n_touch == 0
