"""Correctness gate 9: statistics must match scipy and the Demsar hand-worked example."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from tscfree.stats import (
    average_ranks,
    cd_cliques,
    friedman_test,
    holm_correction,
    nemenyi_critical_difference,
    nemenyi_q,
    regression_rate,
    wilcoxon_holm,
    wilcoxon_pair,
    wins_ties_losses,
)

# Demsar (2006), Table 5 -- the published q_alpha values.
DEMSAR_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
              9: 3.102, 10: 3.164}
DEMSAR_Q10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
              9: 2.855, 10: 2.920}


def test_nemenyi_q_matches_demsar_table() -> None:
    # Demsar prints 3 decimals (truncated, e.g. 2.3437 -> 2.343), hence 1.5e-3.
    for k, q in DEMSAR_Q05.items():
        assert abs(nemenyi_q(k, 0.05) - q) < 1.5e-3, k
    for k, q in DEMSAR_Q10.items():
        assert abs(nemenyi_q(k, 0.10) - q) < 1.5e-3, k


def test_critical_difference_matches_demsar_worked_example() -> None:
    # Demsar 2006, section 3.2.2: k = 4 algorithms, N = 14 data sets, alpha = 0.05
    # -> CD = 2.569 * sqrt(4*5 / (6*14)) = 1.25
    cd = nemenyi_critical_difference(n_methods=4, n_datasets=14, alpha=0.05)
    assert abs(cd - 1.25) < 0.01
    # and the closed form, recomputed by hand
    assert abs(cd - 2.569 * np.sqrt(4 * 5 / (6 * 14))) < 1e-3


def test_average_ranks_simple() -> None:
    # method 0 always best, method 2 always worst
    scores = np.array([[0.9, 0.8, 0.7], [0.95, 0.85, 0.6], [0.7, 0.65, 0.5]])
    r = average_ranks(scores)
    assert np.allclose(r, [1.0, 2.0, 3.0])


def test_average_ranks_ties() -> None:
    scores = np.array([[0.8, 0.8, 0.5]])
    r = average_ranks(scores)
    assert np.allclose(r, [1.5, 1.5, 3.0])


def test_friedman_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random((20, 5))
    res = friedman_test(scores)
    ref = sps.friedmanchisquare(*[scores[:, j] for j in range(scores.shape[1])])
    assert abs(res.chi2 - float(ref.statistic)) < 1e-8
    assert abs(res.p_chi2 - float(ref.pvalue)) < 1e-10


def test_friedman_iman_davenport_relation() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random((14, 4))
    res = friedman_test(scores)
    n, k = scores.shape
    expected = (n - 1) * res.chi2 / (n * (k - 1) - res.chi2)
    assert abs(res.f_stat - expected) < 1e-9


def test_holm_matches_manual_stepdown() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = holm_correction(p)
    # m=4; sorted: 0.005(*4=0.02), 0.01(*3=0.03), 0.03(*2=0.06), 0.04(*1=0.04->0.06)
    order = np.argsort(p)
    assert np.allclose(adj[order], [0.02, 0.03, 0.06, 0.06])
    assert np.all(np.diff(adj[order]) >= -1e-15)  # monotone


def test_holm_matches_statsmodels_style_reference() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(30) ** 3
    adj = holm_correction(p)
    # independent reference implementation
    m = len(p)
    order = np.argsort(p)
    ref = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        ref[idx] = min(1.0, run)
    assert np.allclose(adj, ref)


def test_wilcoxon_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    a = rng.random(25)
    b = a + rng.normal(0.02, 0.05, 25)
    p = wilcoxon_pair(a, b)
    ref = sps.wilcoxon(a, b, zero_method="pratt", alternative="two-sided").pvalue
    assert abs(p - float(ref)) < 1e-12


def test_wilcoxon_all_zero_differences() -> None:
    a = np.ones(10)
    assert wilcoxon_pair(a, a.copy()) == 1.0


def test_wins_ties_losses() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 2.0, 4.0])
    assert wins_ties_losses(a, b) == (1, 1, 1)


def test_wilcoxon_holm_family_sizes() -> None:
    rng = np.random.default_rng(4)
    scores = rng.random((20, 4))
    names = ["m0", "m1", "m2", "m3"]
    all_pairs = wilcoxon_holm(scores, names)
    assert len(all_pairs) == 6
    vs_base = wilcoxon_holm(scores, names, baseline="m0")
    assert len(vs_base) == 3
    assert all(r["method_b"] == "m0" for r in vs_base)
    assert all(0.0 <= r["p_holm"] <= 1.0 for r in all_pairs)
    assert all(r["p_holm"] >= r["p_raw"] - 1e-12 for r in all_pairs)


def test_regression_rate() -> None:
    method = np.array([0.50, 0.60, 0.70, 0.80])
    ref = np.array([0.55, 0.60, 0.65, 0.90])
    out = regression_rate(method, ref, threshold=0.01)
    # regressions: -0.05 and -0.10 -> 2 of 4
    assert abs(out["regression_rate"] - 0.5) < 1e-12
    assert abs(out["mean_loss_when_regressed"] - 0.075) < 1e-12
    assert abs(out["worst_loss"] - 0.10) < 1e-12
    assert abs(out["mean_delta"] - (-0.025)) < 1e-12


def test_regression_rate_empty() -> None:
    out = regression_rate(np.array([]), np.array([]))
    assert out["n"] == 0
    assert np.isnan(out["regression_rate"])


def test_cd_cliques_basic() -> None:
    ranks = np.array([1.0, 1.4, 3.0, 3.2])
    spans = cd_cliques(ranks, cd=0.5)
    assert (0, 1) in spans
    assert (2, 3) in spans
    assert len(spans) == 2


def test_cd_cliques_all_connected() -> None:
    ranks = np.array([1.0, 1.2, 1.3])
    spans = cd_cliques(ranks, cd=5.0)
    assert spans == [(0, 2)]


def test_cd_cliques_none() -> None:
    ranks = np.array([1.0, 3.0, 5.0])
    assert cd_cliques(ranks, cd=0.5) == []
