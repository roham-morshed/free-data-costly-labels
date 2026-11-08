"""Correctness gate 3: the labeled/unlabeled split must be exact."""

from __future__ import annotations

import numpy as np
import pytest

from tscfree.splits import (
    InfeasibleBudgetError,
    check_budget_feasible,
    class_counts,
    labeled_unlabeled_split,
    n_folds_for_labeled_set,
)


def _y(counts: dict[str, int]) -> np.ndarray:
    return np.array([lab for lab, c in counts.items() for _ in range(c)])


@pytest.mark.parametrize("k", [1, 2, 5, 10, 20])
def test_exactly_k_per_class(k: int) -> None:
    y = _y({"a": 40, "b": 33, "c": 27})
    L, U = labeled_unlabeled_split(y, k=k, seed=0, dataset="synthetic")
    for lab in np.unique(y):
        assert int(np.sum(y[L] == lab)) == k


def test_partition_is_exact() -> None:
    y = _y({"a": 40, "b": 33, "c": 27})
    L, U = labeled_unlabeled_split(y, k=5, seed=3, dataset="synthetic")
    assert np.intersect1d(L, U).size == 0
    assert np.array_equal(np.sort(np.concatenate([L, U])), np.arange(len(y)))
    assert len(L) == 15


def test_indices_are_sorted_and_unique() -> None:
    y = _y({"a": 30, "b": 30})
    L, U = labeled_unlabeled_split(y, k=7, seed=1)
    assert np.all(np.diff(L) > 0)
    assert np.all(np.diff(U) > 0)


def test_same_seed_same_indices() -> None:
    y = _y({"a": 30, "b": 25, "c": 22})
    a1, b1 = labeled_unlabeled_split(y, k=5, seed=11, dataset="D")
    a2, b2 = labeled_unlabeled_split(y, k=5, seed=11, dataset="D")
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)


def test_different_seed_different_indices() -> None:
    y = _y({"a": 60, "b": 60})
    a1, _ = labeled_unlabeled_split(y, k=5, seed=0, dataset="D")
    a2, _ = labeled_unlabeled_split(y, k=5, seed=1, dataset="D")
    assert not np.array_equal(a1, a2)


def test_different_dataset_name_different_indices() -> None:
    y = _y({"a": 60, "b": 60})
    a1, _ = labeled_unlabeled_split(y, k=5, seed=0, dataset="D1")
    a2, _ = labeled_unlabeled_split(y, k=5, seed=0, dataset="D2")
    assert not np.array_equal(a1, a2)


def test_class_smaller_than_k_is_rejected() -> None:
    y = _y({"a": 40, "b": 3})
    with pytest.raises(InfeasibleBudgetError) as exc:
        labeled_unlabeled_split(y, k=5, seed=0)
    assert "b" in str(exc.value)
    assert "5" in str(exc.value)


def test_k_zero_rejected() -> None:
    y = _y({"a": 10, "b": 10})
    with pytest.raises(InfeasibleBudgetError):
        check_budget_feasible(y, 0)


def test_class_counts() -> None:
    y = _y({"a": 4, "b": 6})
    assert class_counts(y) == {"a": 4, "b": 6}


def test_exhaustive_budget_leaves_empty_pool() -> None:
    y = _y({"a": 5, "b": 5})
    L, U = labeled_unlabeled_split(y, k=5, seed=0)
    assert len(L) == 10 and len(U) == 0


def test_n_folds_for_labeled_set() -> None:
    assert n_folds_for_labeled_set(1) == 0  # no honest CV possible
    assert n_folds_for_labeled_set(2) == 2
    assert n_folds_for_labeled_set(5) == 5
    assert n_folds_for_labeled_set(20) == 5
    assert n_folds_for_labeled_set(20, max_folds=10) == 10
