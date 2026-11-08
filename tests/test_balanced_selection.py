"""Correctness gate 4: the class-balanced selector must not collapse onto one class.

This is the mechanism that makes SAFE-SSL safe, so it gets a test that proves
both halves of the claim: our selector respects the quota, and the naive
threshold selector -- the thing everyone actually does -- takes almost
everything from the majority class on the same input.
"""

from __future__ import annotations

import numpy as np

from tscfree.methods import class_balanced_select, naive_threshold_select


def biased_posteriors(n: int = 1000, prior: float = 0.9, seed: int = 0) -> np.ndarray:
    """A 90/10 problem where the model is confident on the majority and unsure elsewhere.

    Majority-class points get posteriors near 0.95; minority-class points get
    posteriors barely above 0.5.  This is exactly the situation in which a
    global confidence threshold degenerates.
    """
    rng = np.random.default_rng(seed)
    n_maj = int(round(prior * n))
    n_min = n - n_maj
    p_maj = rng.uniform(0.85, 0.99, n_maj)
    p_min = rng.uniform(0.51, 0.62, n_min)
    P = np.zeros((n, 2))
    P[:n_maj, 0] = p_maj
    P[:n_maj, 1] = 1 - p_maj
    P[n_maj:, 1] = p_min
    P[n_maj:, 0] = 1 - p_min
    return P


def test_balanced_selector_respects_the_quota() -> None:
    """With rho small enough that both classes can fill it, both get exactly q."""
    P = biased_posteriors()  # 900 majority / 100 minority candidates
    C, rho = 2, 0.15
    n_U = P.shape[0]
    q = max(1, int(np.floor(rho * n_U / C)))  # 75
    rows, labs = class_balanced_select(P, rho=rho, n_classes=C)
    counts = np.bincount(labs, minlength=C)
    assert q == 75
    assert counts[0] == q, counts
    assert counts[1] == q, counts
    assert len(np.unique(rows)) == len(rows)


def test_quota_is_capped_by_the_minority_class_supply() -> None:
    """When the quota exceeds the candidates of a class, take all of them.

    Selecting fewer than q from a starved class is the right behaviour: the
    alternative would be to fabricate pseudo-labels for points the propagation
    assigned to a different class.
    """
    P = biased_posteriors()
    rows, labs = class_balanced_select(P, rho=0.25, n_classes=2)
    counts = np.bincount(labs, minlength=2)
    assert counts[0] == 125  # q = floor(0.25 * 1000 / 2)
    assert counts[1] == 100  # only 100 minority candidates exist
    assert len(rows) == 225


def test_naive_threshold_collapses_onto_one_class() -> None:
    P = biased_posteriors()
    rows, labs = naive_threshold_select(P, threshold=0.75)
    assert rows.size > 0
    frac_majority = float(np.mean(labs == 0))
    assert frac_majority >= 0.95, frac_majority


def test_balanced_selector_is_far_more_even_than_naive() -> None:
    P = biased_posteriors()
    _, labs_bal = class_balanced_select(P, rho=0.25, n_classes=2)
    _, labs_naive = naive_threshold_select(P, threshold=0.75)
    imbalance_bal = abs(np.mean(labs_bal == 0) - 0.5)
    imbalance_naive = abs(np.mean(labs_naive == 0) - 0.5)
    # balanced: 125 vs 100 -> 0.056 off parity; naive: ~1000 vs 0 -> 0.5 off parity
    assert imbalance_bal < 0.10
    assert imbalance_naive > 0.4
    assert imbalance_naive > 5 * imbalance_bal


def test_quota_capped_by_available_candidates() -> None:
    """If a class has fewer argmax-c candidates than the quota, take what exists."""
    P = np.zeros((100, 3))
    P[:96, 0] = 0.9
    P[96:98, 1] = 0.9
    P[98:, 2] = 0.9
    P[:, :] += 0.01
    rows, labs = class_balanced_select(P, rho=0.3, n_classes=3)
    counts = np.bincount(labs, minlength=3)
    q = max(1, int(np.floor(0.3 * 100 / 3)))  # 10
    assert counts[0] == q
    assert counts[1] == 2
    assert counts[2] == 2


def test_selector_is_deterministic_and_order_stable() -> None:
    P = biased_posteriors(seed=5)
    a = class_balanced_select(P, 0.25, 2)
    b = class_balanced_select(P, 0.25, 2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.all(np.diff(a[0]) > 0)  # sorted, unique


def test_empty_pool() -> None:
    rows, labs = class_balanced_select(np.zeros((0, 3)), 0.25, 3)
    assert rows.size == 0 and labs.size == 0


def test_rho_scales_selection() -> None:
    P = biased_posteriors(n=800)
    small = class_balanced_select(P, 0.10, 2)[0]
    large = class_balanced_select(P, 0.50, 2)[0]
    assert len(large) > len(small)
    assert len(small) == 2 * max(1, int(np.floor(0.10 * 800 / 2)))
