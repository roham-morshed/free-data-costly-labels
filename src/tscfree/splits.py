"""Labeled / unlabeled splits of the official training set.

Protocol (§5 of the specification):

* the official *test* set is never touched here;
* from ``y_train`` we draw exactly ``k`` labeled instances **per class**,
  stratified and seeded;
* everything else in the training set becomes the unlabeled pool ``U``;
* ``L`` and ``U`` partition the training set exactly.
"""

from __future__ import annotations

import numpy as np

from .utils import rng_for


class InfeasibleBudgetError(ValueError):
    """A class has fewer than ``k`` training instances, so the budget is impossible."""


def class_counts(y: np.ndarray) -> dict[str, int]:
    """Per-class instance counts, keyed by ``str(label)`` for JSON friendliness."""
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(lab): int(c) for lab, c in zip(labels, counts, strict=True)}


def check_budget_feasible(y: np.ndarray, k: int) -> None:
    """Raise :class:`InfeasibleBudgetError` if any class has fewer than ``k`` members."""
    if k < 1:
        raise InfeasibleBudgetError(f"k must be >= 1, got {k}")
    counts = class_counts(y)
    bad = {lab: c for lab, c in counts.items() if c < k}
    if bad:
        raise InfeasibleBudgetError(
            f"Cannot draw k={k} labels per class: class(es) {bad} have fewer than {k} "
            "training instances. Reject this (dataset, k) pair rather than silently "
            "drawing fewer labels."
        )


def labeled_unlabeled_split(
    y: np.ndarray, k: int, seed: int, dataset: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(L_idx, U_idx)`` into the training set.

    Parameters
    ----------
    y
        Training labels.  Used *only here* -- the returned ``U_idx`` labels are
        treated as non-existent from this point on (see
        :class:`tscfree.utils.GuardedLabels`).
    k
        Labels per class.
    seed
        Evaluation seed.  Same ``(dataset, k, seed)`` always yields the same
        index sets, regardless of execution order.
    dataset
        Optional name mixed into the RNG stream so that two different datasets
        with identical label vectors still get different draws.

    Returns
    -------
    (L_idx, U_idx)
        Both sorted ascending; ``L`` has exactly ``k`` entries per class;
        ``L`` and ``U`` are disjoint and together cover ``range(len(y))``.
    """
    y = np.asarray(y)
    check_budget_feasible(y, k)
    rng = rng_for("split", dataset, k, seed)

    labels = np.unique(y)
    chosen: list[np.ndarray] = []
    for lab in labels:
        idx = np.flatnonzero(y == lab)
        # permutation of the *sorted* member indices -> order-independent draw
        take = rng.permutation(idx)[:k]
        chosen.append(np.sort(take))
    L_idx = np.sort(np.concatenate(chosen))

    mask = np.ones(len(y), dtype=bool)
    mask[L_idx] = False
    U_idx = np.flatnonzero(mask)

    # -- invariants, asserted every time: this is cheap and the whole paper
    #    rests on them being true.
    assert len(np.intersect1d(L_idx, U_idx)) == 0, "L and U overlap"
    assert len(L_idx) + len(U_idx) == len(y), "L u U does not cover the training set"
    assert len(L_idx) == k * len(labels), "wrong number of labeled instances"
    for lab in labels:
        assert int(np.sum(y[L_idx] == lab)) == k, f"class {lab!r} does not have exactly k labels"

    return L_idx, U_idx


def n_folds_for_labeled_set(k: int, max_folds: int = 5) -> int:
    """Number of stratified CV folds usable on the labeled set alone.

    With ``k`` labels per class at most ``k`` folds are possible (``k``-fold
    with one instance of each class per fold == stratified leave-one-out when
    ``k == max_folds``).  Returns ``0`` when no honest CV is possible, i.e.
    ``k == 1``: with a single labeled instance per class there is no held-out
    labeled data at all, so the safety valve must *abstain* rather than invent
    an estimate.  Callers must handle 0 explicitly.
    """
    n = min(max_folds, k)
    return n if n >= 2 else 0
