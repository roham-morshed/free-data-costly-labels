"""Evaluation metrics, including the RQ1 recovery ratio.

The recovery ratio answers: *of the gap between the strictly-supervised
low-label baseline and the fully-supervised ceiling, what fraction did this
method close?*

    recovery = (acc_method - acc_sup_strict) / (acc_sup_full - acc_sup_strict)

0 means "no better than sup_strict", 1 means "reached the ceiling", negative
means "worse than doing nothing".  When the denominator is (near) zero the
ratio is undefined -- we return NaN and drop the cell from the aggregate, and
we report how many cells were dropped.  We never clip to hide it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

#: Denominator below this magnitude is treated as degenerate (ceiling == floor).
RECOVERY_EPS = 1e-9


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, balanced accuracy and macro-F1 for one prediction vector."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def recovery_ratio(
    acc_method: float, acc_strict: float, acc_full: float, eps: float = RECOVERY_EPS
) -> float:
    """Fraction of the strict->ceiling gap recovered.  NaN if the gap is degenerate."""
    denom = acc_full - acc_strict
    if not np.isfinite(denom) or abs(denom) < eps:
        return float("nan")
    return float((acc_method - acc_strict) / denom)


def recovery_ratio_array(
    acc_method: np.ndarray, acc_strict: np.ndarray, acc_full: np.ndarray, eps: float = RECOVERY_EPS
) -> np.ndarray:
    """Vectorised :func:`recovery_ratio`; degenerate entries become NaN."""
    acc_method = np.asarray(acc_method, dtype=float)
    acc_strict = np.asarray(acc_strict, dtype=float)
    acc_full = np.asarray(acc_full, dtype=float)
    denom = acc_full - acc_strict
    out = np.full(np.broadcast(acc_method, denom).shape, np.nan, dtype=float)
    ok = np.isfinite(denom) & (np.abs(denom) >= eps)
    out[ok] = (acc_method - acc_strict)[ok] / denom[ok]
    return out


def majority_class_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent *training* class."""
    labels, counts = np.unique(np.asarray(y_train), return_counts=True)
    maj = labels[int(np.argmax(counts))]
    return float(np.mean(np.asarray(y_test) == maj))


def binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact confidence interval for a proportion.

    Used by ``tests/test_methods_sanity.py`` to check that no method beats
    chance on pure noise -- a method that does has a leak.
    """
    from scipy import stats

    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
