"""Statistical machinery expected by the time-series-classification community.

Everything here operates on a *scores matrix* ``M`` of shape
``(n_datasets, n_methods)`` where entries are already averaged over seeds and
**higher is better**.

Implemented, in the order the paper uses them:

* :func:`average_ranks`            -- Friedman ranks (ties averaged, 1 = best)
* :func:`friedman_test`            -- chi-square and Iman-Davenport F statistics
* :func:`nemenyi_critical_difference`
* :func:`wilcoxon_holm`            -- pairwise Wilcoxon signed-rank + Holm
* :func:`regression_rate`          -- RQ5 headline
* :func:`plot_cd_diagram`          -- the Demsar critical-difference diagram
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Ranks and omnibus tests
# ---------------------------------------------------------------------------


def average_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Per-dataset ranks averaged over datasets.  Rank 1 == best.

    Ties receive the average of the ranks they span, which is what Friedman's
    test assumes.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"expected a (n_datasets, n_methods) matrix, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores matrix contains NaN/inf -- drop those datasets first")
    signed = -scores if higher_is_better else scores
    per_dataset = np.vstack([stats.rankdata(row, method="average") for row in signed])
    return per_dataset.mean(axis=0)


def per_dataset_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """The full ``(n_datasets, n_methods)`` rank matrix (rank 1 == best)."""
    scores = np.asarray(scores, dtype=float)
    signed = -scores if higher_is_better else scores
    return np.vstack([stats.rankdata(row, method="average") for row in signed])


@dataclass
class FriedmanResult:
    chi2: float
    p_chi2: float
    f_stat: float
    p_f: float
    n_datasets: int
    n_methods: int
    avg_ranks: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "chi2": self.chi2,
            "p_chi2": self.p_chi2,
            "iman_davenport_F": self.f_stat,
            "p_F": self.p_f,
            "n_datasets": self.n_datasets,
            "n_methods": self.n_methods,
            "avg_ranks": self.avg_ranks.tolist(),
        }


def friedman_test(scores: np.ndarray, higher_is_better: bool = True) -> FriedmanResult:
    """Friedman chi-square plus the Iman-Davenport F correction.

    Demsar (2006) recommends the F version because the chi-square statistic is
    undesirably conservative.
    """
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    if k < 2:
        raise ValueError("need at least 2 methods")
    if n < 2:
        raise ValueError("need at least 2 datasets")
    ranks = per_dataset_ranks(scores, higher_is_better=higher_is_better)
    r = ranks.mean(axis=0)
    chi2 = 12.0 * n / (k * (k + 1)) * (float(np.sum(r**2)) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = float(stats.chi2.sf(chi2, k - 1))
    denom = n * (k - 1) - chi2
    if denom <= 0:
        f_stat = float("inf")
        p_f = 0.0
    else:
        f_stat = (n - 1) * chi2 / denom
        p_f = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    return FriedmanResult(float(chi2), p_chi2, float(f_stat), p_f, n, k, r)


def nemenyi_q(n_methods: int, alpha: float = 0.05) -> float:
    """Critical value ``q_alpha`` of the Nemenyi test.

    Equal to the studentized range statistic at infinite degrees of freedom
    divided by sqrt(2); reproduces Demsar (2006) Table 5 exactly (verified in
    ``tests/test_stats.py``).
    """
    if n_methods < 2:
        raise ValueError("need at least 2 methods")
    return float(stats.studentized_range.ppf(1 - alpha, n_methods, np.inf)) / np.sqrt(2.0)


def nemenyi_critical_difference(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """CD = q_alpha * sqrt(k(k+1) / (6N)).

    Demsar's worked example (k=4 methods, N=14 datasets, alpha=0.05) gives
    CD = 1.25; asserted in ``tests/test_stats.py``.
    """
    q = nemenyi_q(n_methods, alpha)
    return float(q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(1.0, running)
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def wins_ties_losses(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> tuple[int, int, int]:
    """(#datasets where a > b, #ties, #where a < b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    return int(np.sum(d > tol)), int(np.sum(np.abs(d) <= tol)), int(np.sum(d < -tol))


def wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 if all differences are zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if np.allclose(d, 0.0):
        return 1.0
    # 'pratt' keeps zero-differences in the ranking, which is the conservative
    # choice recommended when many datasets tie (common at k=1).
    res = stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
    return float(res.pvalue)


def wilcoxon_holm(
    scores: np.ndarray,
    method_names: list[str],
    baseline: str | None = None,
) -> list[dict[str, Any]]:
    """Pairwise Wilcoxon signed-rank tests with a Holm correction.

    Parameters
    ----------
    scores
        ``(n_datasets, n_methods)``, higher is better, already seed-averaged.
    method_names
        Column names of ``scores``.
    baseline
        If given, only ``method vs baseline`` comparisons are made and Holm is
        applied over that (smaller) family.  Otherwise all
        ``k(k-1)/2`` pairs form the family.

    Returns
    -------
    A list of dicts, one per comparison, sorted by adjusted p-value.
    """
    scores = np.asarray(scores, dtype=float)
    k = len(method_names)
    if scores.shape[1] != k:
        raise ValueError("method_names length must match the number of columns")

    pairs: list[tuple[int, int]] = []
    if baseline is None:
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    else:
        if baseline not in method_names:
            raise ValueError(f"baseline {baseline!r} not in {method_names}")
        b = method_names.index(baseline)
        pairs = [(i, b) for i in range(k) if i != b]

    rows: list[dict[str, Any]] = []
    praw: list[float] = []
    for i, j in pairs:
        a, bb = scores[:, i], scores[:, j]
        w, t, losses = wins_ties_losses(a, bb)
        p = wilcoxon_pair(a, bb)
        praw.append(p)
        rows.append(
            {
                "method_a": method_names[i],
                "method_b": method_names[j],
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(bb)),
                "median_diff": float(np.median(a - bb)),
                "wins": w,
                "ties": t,
                "losses": losses,
                "p_raw": p,
            }
        )
    padj = holm_correction(np.asarray(praw))
    for row, pa in zip(rows, padj, strict=True):
        row["p_holm"] = float(pa)
    rows.sort(key=lambda r: (r["p_holm"], r["p_raw"]))
    return rows


# ---------------------------------------------------------------------------
# RQ5: regression rate
# ---------------------------------------------------------------------------


def regression_rate(
    method_acc: np.ndarray, reference_acc: np.ndarray, threshold: float = 0.01
) -> dict[str, float]:
    """How often, and how badly, a method is worse than the reference.

    Operates on *unaggregated* (dataset, seed) pairs, per the specification:
    a regression is a case where ``method`` is worse than ``reference`` by more
    than ``threshold`` accuracy.
    """
    m = np.asarray(method_acc, dtype=float)
    r = np.asarray(reference_acc, dtype=float)
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    if m.size == 0:
        return {
            "n": 0,
            "regression_rate": float("nan"),
            "mean_loss_when_regressed": float("nan"),
            "worst_loss": float("nan"),
            "mean_delta": float("nan"),
            "win_rate": float("nan"),
        }
    delta = m - r
    bad = delta < -threshold
    return {
        "n": int(m.size),
        "regression_rate": float(np.mean(bad)),
        "mean_loss_when_regressed": float(-np.mean(delta[bad])) if bad.any() else 0.0,
        "worst_loss": float(-np.min(delta)) if delta.min() < 0 else 0.0,
        "mean_delta": float(np.mean(delta)),
        "win_rate": float(np.mean(delta > threshold)),
    }


# ---------------------------------------------------------------------------
# Critical-difference diagram
# ---------------------------------------------------------------------------


def cd_cliques(avg_ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal groups of consecutive (rank-sorted) methods within ``cd``.

    Returns index pairs ``(i, j)`` into the *rank-sorted* order, each denoting a
    bar covering positions ``i..j``.  Cliques fully contained in another are
    dropped, which is the standard Demsar rendering.
    """
    order = np.argsort(avg_ranks)
    r = np.asarray(avg_ranks)[order]
    n = len(r)
    spans: list[tuple[int, int]] = []
    for i in range(n):
        j = i
        while j + 1 < n and (r[j + 1] - r[i]) <= cd:
            j += 1
        if j > i:
            spans.append((i, j))
    maximal = [s for s in spans if not any(s != o and s[0] >= o[0] and s[1] <= o[1] for o in spans)]
    # de-duplicate while preserving order
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for s in maximal:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def plot_cd_diagram(
    avg_ranks: np.ndarray,
    method_names: list[str],
    cd: float,
    title: str = "",
    ax: Any = None,
    width: float = 7.0,
    text_size: int = 8,
) -> Any:
    """Draw a Demsar critical-difference diagram onto ``ax`` (created if None).

    Layout, top to bottom: title, the CD ruler, the rank axis with its ticks,
    the clique bars, then one row per method with a leader line out to its
    label.  Best (lowest average rank) on the left.

    The y axis is in abstract "rows" and is inverted so that y grows downward,
    which keeps the arithmetic readable.
    """
    import matplotlib.pyplot as plt

    avg_ranks = np.asarray(avg_ranks, dtype=float)
    k = len(method_names)
    order = np.argsort(avg_ranks)
    ranks_sorted = avg_ranks[order]
    names_sorted = [method_names[i] for i in order]

    lo = float(np.floor(min(avg_ranks.min(), 1.0)))
    hi = float(np.ceil(avg_ranks.max()))
    if hi - lo < 1:
        hi = lo + 1

    n_left = (k + 1) // 2
    cliques = cd_cliques(avg_ranks, cd)

    # -- vertical budget, in rows -------------------------------------------
    y_cd = 0.86          # the CD ruler
    y_axis = 1.66        # the rank axis line
    y_clique0 = 1.88     # first clique bar
    clique_gap = 0.20
    y_row0 = y_clique0 + clique_gap * max(len(cliques), 1) + 0.30
    row_gap = 0.55
    y_bottom = y_row0 + row_gap * (n_left - 1) + 0.35
    height_in = 0.30 * y_bottom + 0.35

    if ax is None:
        _fig, ax = plt.subplots(figsize=(width, height_in))
    span = hi - lo
    pad = 0.32 * span  # room for the labels outside the axis
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(y_bottom, 0.0)  # inverted: y grows downward
    ax.set_axis_off()

    # -- rank axis ----------------------------------------------------------
    ax.plot([lo, hi], [y_axis, y_axis], color="black", lw=1.1, zorder=2)
    for tick in np.arange(lo, hi + 1e-9, 0.5):
        big = abs(tick - round(tick)) < 1e-9
        ax.plot([tick, tick], [y_axis, y_axis - (0.11 if big else 0.06)],
                color="black", lw=1.0, zorder=2)
        if big:
            ax.text(tick, y_axis - 0.16, f"{int(round(tick))}", ha="center", va="bottom",
                    fontsize=text_size)

    # -- CD ruler -----------------------------------------------------------
    ax.plot([lo, lo + cd], [y_cd, y_cd], color="black", lw=2.4, solid_capstyle="butt")
    for x in (lo, lo + cd):
        ax.plot([x, x], [y_cd - 0.09, y_cd + 0.09], color="black", lw=1.3)
    ax.text(lo + cd / 2, y_cd - 0.14, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=text_size)

    # -- clique bars --------------------------------------------------------
    for depth, (i, j) in enumerate(cliques):
        y = y_clique0 + clique_gap * depth
        ax.plot([ranks_sorted[i] - 0.02 * span, ranks_sorted[j] + 0.02 * span], [y, y],
                color="black", lw=3.4, solid_capstyle="butt", zorder=3)

    # -- method leaders and labels -----------------------------------------
    for pos in range(k):
        rank = ranks_sorted[pos]
        name = names_sorted[pos]
        left = pos < n_left
        row = pos if left else (k - 1 - pos)
        y = y_row0 + row_gap * row
        x_text = (lo - 0.03 * span) if left else (hi + 0.03 * span)
        ax.plot([rank, rank], [y_axis, y], color="black", lw=0.9)
        ax.plot([rank, x_text], [y, y], color="black", lw=0.9)
        ax.text(
            x_text + (-0.02 * span if left else 0.02 * span),
            y,
            f"{name}  ({rank:.2f})" if left else f"({rank:.2f})  {name}",
            ha="right" if left else "left",
            va="center",
            fontsize=text_size,
        )

    if title:
        ax.text((lo + hi) / 2, 0.02, title, ha="center", va="top",
                fontsize=text_size + 1)
    return ax
