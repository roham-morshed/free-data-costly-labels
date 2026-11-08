"""Correctness gate 8: recovery ratio and friends."""

from __future__ import annotations

import numpy as np

from tscfree.metrics import (
    binomial_ci,
    classification_metrics,
    majority_class_accuracy,
    recovery_ratio,
    recovery_ratio_array,
)


def test_recovery_is_zero_for_sup_strict() -> None:
    assert recovery_ratio(0.60, 0.60, 0.90) == 0.0


def test_recovery_is_one_for_sup_full() -> None:
    assert recovery_ratio(0.90, 0.60, 0.90) == 1.0


def test_recovery_half_way() -> None:
    assert recovery_ratio(0.75, 0.60, 0.90) == 0.5


def test_recovery_can_be_negative() -> None:
    assert recovery_ratio(0.50, 0.60, 0.90) < 0


def test_recovery_degenerate_is_nan_not_a_crash() -> None:
    assert np.isnan(recovery_ratio(0.7, 0.8, 0.8))
    assert np.isnan(recovery_ratio(0.8, 0.8, 0.8))


def test_recovery_array_matches_scalar() -> None:
    m = np.array([0.60, 0.75, 0.90, 0.50])
    s = np.array([0.60, 0.60, 0.60, 0.60])
    f = np.array([0.90, 0.90, 0.90, 0.60])
    out = recovery_ratio_array(m, s, f)
    assert np.allclose(out[:3], [0.0, 0.5, 1.0])
    assert np.isnan(out[3])


def test_classification_metrics_perfect() -> None:
    y = np.array(["a", "b", "a", "b"])
    out = classification_metrics(y, y)
    assert out["accuracy"] == 1.0
    assert out["balanced_accuracy"] == 1.0
    assert out["macro_f1"] == 1.0


def test_classification_metrics_balanced_vs_plain() -> None:
    # 90/10 imbalance, predict the majority everywhere
    y_true = np.array(["a"] * 90 + ["b"] * 10)
    y_pred = np.array(["a"] * 100)
    out = classification_metrics(y_true, y_pred)
    assert abs(out["accuracy"] - 0.90) < 1e-12
    assert abs(out["balanced_accuracy"] - 0.50) < 1e-12
    assert out["macro_f1"] < 0.5


def test_majority_class_accuracy() -> None:
    ytr = np.array(["a"] * 8 + ["b"] * 2)
    yte = np.array(["a"] * 7 + ["b"] * 3)
    assert abs(majority_class_accuracy(ytr, yte) - 0.7) < 1e-12


def test_binomial_ci_contains_chance() -> None:
    lo, hi = binomial_ci(50, 100)
    assert lo < 0.5 < hi
    lo2, hi2 = binomial_ci(90, 100)
    assert lo2 > 0.5
