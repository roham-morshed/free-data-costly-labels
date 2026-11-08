"""Correctness gate 2: nothing may be fitted on the test set.

Protocol: run every method twice -- once with the real test set, once with a
test set whose features are replaced by random noise.  Everything fitted on the
training side (the MiniRocket parameters, the scaler statistics, the graph, the
pseudo-labels, the final linear model) must be **bit-identical** across the two
runs.  If it is not, something looked at the test data.

The comparison is made through ``MethodResult.extra['train_signature']``, a
SHA-256 of the fitted coefficients plus the method's pseudo-label vector, and
through the pseudo-label counts and the safety-valve decision.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import build_ctx

from tscfree.methods import METHOD_IDS, run_method


def _data(seed: int = 3):
    rng = np.random.default_rng(seed)
    length = 48
    t = np.linspace(0, 4 * np.pi, length)
    a = np.sin(t)[None, :] + 0.30 * rng.standard_normal((60, length))
    b = np.cos(2 * t)[None, :] + 0.30 * rng.standard_normal((60, length))
    Xtr = np.concatenate([a, b])[:, None, :]
    ytr = np.array(["A"] * 60 + ["B"] * 60)
    a2 = np.sin(t)[None, :] + 0.30 * rng.standard_normal((25, length))
    b2 = np.cos(2 * t)[None, :] + 0.30 * rng.standard_normal((25, length))
    Xte = np.concatenate([a2, b2])[:, None, :]
    yte = np.array(["A"] * 25 + ["B"] * 25)
    return Xtr, ytr, Xte, yte


def _noise_test_context(ctx, seed: int = 1234):
    """The same context with every test-side array replaced by noise."""
    rng = np.random.default_rng(seed)
    return dataclasses.replace(
        ctx,
        X_test=rng.standard_normal(ctx.X_test.shape),
        F_test=rng.random(ctx.F_test.shape).astype(ctx.F_test.dtype),
        Z_test=rng.standard_normal(ctx.Z_test.shape).astype(ctx.Z_test.dtype),
        D_dtw=None if ctx.D_dtw is None else rng.random(ctx.D_dtw.shape).astype(np.float32),
    )


@pytest.mark.parametrize("method", [m for m in METHOD_IDS if m != "1nn_dtw"])
def test_training_side_is_unchanged_by_a_noise_test_set(tmp_cache, method: str) -> None:
    ctx, _ = build_ctx("NoTestLeak", *_data(), k=5, seed=0)
    ctx_noise = _noise_test_context(ctx)

    a = run_method(method, ctx)
    b = run_method(method, ctx_noise)

    assert a.extra["train_signature"] == b.extra["train_signature"], (
        f"{method}: the fitted training-side state changed when the test set was "
        "replaced by noise -- something was fitted on test data."
    )
    assert a.n_pseudo_selected == b.n_pseudo_selected
    assert a.safe_ssl_fired == b.safe_ssl_fired
    if "valve" in a.extra:
        assert a.extra["valve"] == b.extra["valve"]
    if "round_selection_counts" in a.extra:
        assert a.extra["round_selection_counts"] == b.extra["round_selection_counts"]


def test_1nn_dtw_training_side_is_just_the_labeled_set(tmp_cache) -> None:
    ctx, ds = build_ctx("NoTestLeakDtw", *_data(), k=5, seed=0)
    rng = np.random.default_rng(0)
    ctx = dataclasses.replace(
        ctx, D_dtw=rng.random((ds.n_test, ds.n_train)).astype(np.float32)
    )
    a = run_method("1nn_dtw", ctx)
    b = run_method("1nn_dtw", _noise_test_context(ctx))
    assert a.extra["train_signature"] == b.extra["train_signature"]


def test_predictions_do_change_with_a_noise_test_set(tmp_cache) -> None:
    """Sanity in the other direction: the *predictions* must differ.

    If they did not, the noise substitution would not be exercising anything and
    the test above would be vacuous.
    """
    ctx, _ = build_ctx("NoTestLeakSanity", *_data(), k=5, seed=0)
    a = run_method("sup_tf", ctx)
    b = run_method("sup_tf", _noise_test_context(ctx))
    assert not np.array_equal(a.y_pred, b.y_pred)


def test_safety_valve_never_sees_the_test_set(tmp_cache) -> None:
    """The valve decision is a function of labeled data only."""
    ctx, _ = build_ctx("ValveNoTest", *_data(), k=5, seed=7)
    a = run_method("safe_ssl", ctx)
    b = run_method("safe_ssl", _noise_test_context(ctx))
    assert a.extra["valve"] == b.extra["valve"]
    assert a.safe_ssl_fired == b.safe_ssl_fired
    assert a.extra["valve"]["status"] in {
        "ssl_kept",
        "reverted_to_supervised",
        "abstain_k_too_small",
        "abstain_no_valid_folds",
    }
