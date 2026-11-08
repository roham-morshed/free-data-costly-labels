"""All classifiers compared in the study, plus SAFE-SSL.

Every method receives a :class:`MethodContext` and returns a
:class:`MethodResult`.  The context deliberately exposes

* ``y_L``                  -- the labels of the ``k * C`` labeled instances,
* ``y_train_guarded``      -- a :class:`~tscfree.utils.GuardedLabels` view that
  *raises* if anything reads a label from the unlabeled pool,

and nothing else label-shaped.  A method that needs the labels of ``U`` cannot
get them by accident; it crashes.

Three tiers (§5):

===========  =========================  ==================================
tier         transform fitted on        classifier fitted on
===========  =========================  ==================================
SUP-strict   ``X_train[L]``             ``L``
SUP-TF       ``X_train[L u U]``         ``L``
SSL-*        ``X_train[L u U]``         ``L`` + selected pseudo-labels
===========  =========================  ==================================

``SUP-TF - SUP-strict`` isolates the value of free unsupervised feature
fitting; ``SSL - SUP-TF`` isolates the value of pseudo-labeling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

from .augment import augment_labeled
from .features import transform_fit_on
from .utils import GuardedLabels, Timer, rng_for, seed_from

METHOD_IDS = (
    "1nn_dtw",
    "sup_strict",
    "sup_tf",
    "self_train",
    "label_spread",
    "augment",
    "safe_ssl",
    "safe_ssl_novalve",
)


# ---------------------------------------------------------------------------
# Context / result containers
# ---------------------------------------------------------------------------


@dataclass
class MethodContext:
    """Everything a method is allowed to see."""

    dataset: str
    k: int
    seed: int

    X_train: np.ndarray  # raw series, full training set (unsupervised use only)
    X_test: np.ndarray

    L_idx: np.ndarray
    U_idx: np.ndarray
    y_L: np.ndarray  # labels of L, in L_idx order
    classes: np.ndarray  # sorted unique labels (all present in y_L by construction)

    F_train: np.ndarray  # transform fitted on L u U
    F_test: np.ndarray
    Z_train: np.ndarray  # standardised version of the above (scaler fitted on L u U)
    Z_test: np.ndarray
    graphs: dict[str, Any]

    cfg: dict[str, Any]
    y_train_guarded: GuardedLabels | None = None
    D_dtw: np.ndarray | None = None  # (n_test, n_train) cached DTW distances

    transform: str = "minirocket"
    n_kernels: int = 10000
    transform_seed: int = 0

    @property
    def n_train(self) -> int:
        return int(self.F_train.shape[0])

    @property
    def n_classes(self) -> int:
        return int(len(self.classes))


@dataclass
class MethodResult:
    """One method's predictions plus everything the result row needs."""

    y_pred: np.ndarray
    feature_seconds: float = 0.0
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    n_pseudo_selected: int = 0
    safe_ssl_fired: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared estimator helpers
# ---------------------------------------------------------------------------


def ridge_alphas(cfg: dict[str, Any]) -> np.ndarray:
    a, b, n = cfg["ridge"]["alphas_logspace"]
    return np.logspace(a, b, int(n))


def fit_ridge(F: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> Any:
    """RidgeClassifierCV, degrading to a fixed alpha when CV is not honest.

    ``RidgeClassifierCV`` selects alpha by closed-form leave-one-out.  With one
    labeled instance per class (k=1) leaving one out removes an entire class,
    so the selection is meaningless; there we fall back to ``alpha=1.0``.  The
    branch taken is recorded so the paper can state it.
    """
    y = np.asarray(y)
    n = len(y)
    n_classes = len(np.unique(y))
    if n >= 4 and n >= 2 * n_classes:
        clf = RidgeClassifierCV(alphas=ridge_alphas(cfg))
        clf._tscfree_cv = True  # type: ignore[attr-defined]
    else:
        clf = RidgeClassifier(alpha=1.0)
        clf._tscfree_cv = False  # type: ignore[attr-defined]
    clf.fit(F, y)
    return clf


def make_logreg(cfg: dict[str, Any], seed: int) -> LogisticRegression:
    """Probabilistic linear model.

    ``RidgeClassifierCV`` has no ``predict_proba``.  Rather than inventing
    probabilities from ``decision_function``, every method that genuinely needs
    class posteriors (self-training) uses logistic regression on the
    standardised features.  This is stated in the paper.
    """
    return LogisticRegression(
        max_iter=int(cfg["logreg"]["max_iter"]),
        C=float(cfg["logreg"]["C"]),
        random_state=seed,
    )


def _encode(classes: np.ndarray, y: np.ndarray) -> tuple[LabelEncoder, np.ndarray]:
    le = LabelEncoder().fit(classes)
    return le, le.transform(y)


def fitted_signature(clf: Any, *arrays: Any) -> str:
    """A content hash of everything that was fitted on the *training* side.

    Coefficients of the final linear model, plus any pseudo-label vector the
    method produced.  Nothing here may depend on the test set, which is what
    ``tests/test_no_test_leakage.py`` checks: replacing the test features with
    noise must leave this string unchanged.
    """
    import hashlib

    h = hashlib.sha256()
    for attr in ("coef_", "intercept_", "alpha_", "classes_"):
        v = getattr(clf, attr, None)
        if v is None and hasattr(clf, "estimator_"):
            v = getattr(clf.estimator_, attr, None)
        if v is not None:
            a = np.asarray(v)
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a.astype(np.float64) if a.dtype.kind in "fiu" else a.astype(str)).tobytes())
    for a in arrays:
        if a is None:
            continue
        arr = np.asarray(a)
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes() if arr.dtype.kind != "U" else arr.astype(str).tobytes())
    return h.hexdigest()


def precomputed_kernel(W: Any) -> Callable[[Any, Any], Any]:
    """A callable kernel for LabelSpreading returning a cached affinity matrix.

    sklearn calls ``kernel(X, X)`` and then applies the normalised graph
    Laplacian itself, so supplying the affinity is mathematically identical to
    letting it rebuild the graph -- and roughly 100x faster, because the graph
    is label-independent and therefore shared across all 14,000 cells.
    ``tests/test_methods_sanity.py`` asserts bit-identical outputs.
    """

    def _k(X: Any, Y: Any = None) -> Any:
        del X, Y
        return W

    return _k


def run_label_spreading(
    W: Any, seed_labels: np.ndarray, alpha: float, max_iter: int, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Label spreading on a precomputed affinity.

    Returns ``(label_distributions (n, n_seen_classes), classes_seen)``.

    ``seed_labels`` uses ``-1`` for unlabeled, integer class ids otherwise.
    """
    n = W.shape[0]
    dummy = np.zeros((n, 1), dtype=np.float64)  # X is unused: the kernel is precomputed
    ls = LabelSpreading(
        kernel=precomputed_kernel(W), alpha=alpha, max_iter=int(max_iter), tol=float(tol)
    )
    ls.fit(dummy, seed_labels)
    return np.asarray(ls.label_distributions_), np.asarray(ls.classes_)


# ---------------------------------------------------------------------------
# Pseudo-label selection
# ---------------------------------------------------------------------------


def class_balanced_select(
    P: np.ndarray, rho: float, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """SAFE-SSL's selector (§6 step 4).

    For each class ``c`` take the top-``q`` unlabeled points by ``P[:, c]``
    among those whose argmax is ``c``, with
    ``q = max(1, floor(rho * n_U / C))``.

    Why per-class quotas rather than a global confidence threshold: a global
    threshold selects wherever the model is already confident, which on an
    imbalanced or easy-majority problem means it selects almost exclusively the
    majority class, reinforces that bias, and collapses.  This is the dominant
    failure mode of naive self-training and the reason SSL sometimes *hurts*.
    ``tests/test_balanced_selection.py`` demonstrates the difference.

    Returns
    -------
    (rows, labels)
        ``rows`` are positions into ``P`` (i.e. into ``U``), ``labels`` the
        integer class ids assigned to them.
    """
    P = np.asarray(P, dtype=float)
    n_U = P.shape[0]
    if n_U == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    q = max(1, int(np.floor(float(rho) * n_U / float(n_classes))))
    hard = np.argmax(P, axis=1)
    rows: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    for c in range(n_classes):
        cand = np.flatnonzero(hard == c)
        if cand.size == 0:
            continue
        # stable ordering: sort by (-confidence, index) so ties break on index
        order = np.lexsort((cand, -P[cand, c]))
        take = cand[order[:q]]
        rows.append(take)
        labs.append(np.full(take.size, c, dtype=int))
    if not rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    r = np.concatenate(rows)
    ll = np.concatenate(labs)
    order = np.argsort(r)
    return r[order], ll[order]


def naive_threshold_select(P: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """The straw man: take every point whose max posterior exceeds ``threshold``.

    Present only so the paper (and ``tests/test_balanced_selection.py``) can
    quantify how badly it skews.
    """
    P = np.asarray(P, dtype=float)
    conf = P.max(axis=1)
    rows = np.flatnonzero(conf >= threshold)
    return rows, np.argmax(P[rows], axis=1) if rows.size else np.empty(0, dtype=int)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def m_1nn_dtw(ctx: MethodContext) -> MethodResult:
    """1-nearest-neighbour DTW on the labeled set only.  No features at all.

    Uses the cached ``(n_test, n_train)`` DTW distance matrix: distances do not
    depend on labels, so the 1NN prediction is an ``argmin`` over the columns
    that happen to be labeled.
    """
    with Timer() as t:
        if ctx.D_dtw is not None:
            sub = np.asarray(ctx.D_dtw[:, ctx.L_idx], dtype=np.float64)
            nn = np.argmin(sub, axis=1)
            y_pred = ctx.y_L[nn]
        else:  # fall back to computing distances on the fly
            from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

            clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
            clf.fit(ctx.X_train[ctx.L_idx], ctx.y_L)
            y_pred = np.asarray(clf.predict(ctx.X_test))
    return MethodResult(
        y_pred=y_pred,
        predict_seconds=t.seconds,
        extra={
            "uses_features": False,
            "train_signature": fitted_signature(None, ctx.L_idx, ctx.y_L.astype(str)),
        },
    )


def m_sup_strict(ctx: MethodContext) -> MethodResult:
    """Tier 1: the transform itself only ever sees the labeled instances."""
    Xl = ctx.X_train[ctx.L_idx]
    (F_l, F_te), feat_s = transform_fit_on(
        Xl, [Xl, ctx.X_test], ctx.transform, ctx.n_kernels, ctx.transform_seed
    )
    with Timer() as t_fit:
        clf = fit_ridge(F_l, ctx.y_L, ctx.cfg)
    with Timer() as t_pred:
        y_pred = np.asarray(clf.predict(F_te))
    return MethodResult(
        y_pred=y_pred,
        feature_seconds=feat_s,
        fit_seconds=t_fit.seconds,
        predict_seconds=t_pred.seconds,
        extra={
            "ridge_cv": bool(getattr(clf, "_tscfree_cv", False)),
            "feature_dim": int(F_l.shape[1]),
            "train_signature": fitted_signature(clf, ctx.L_idx),
        },
    )


def m_sup_tf(ctx: MethodContext) -> MethodResult:
    """Tier 2: transform fitted on L u U (free, label-free), ridge on L only."""
    F_l = np.asarray(ctx.F_train[ctx.L_idx])
    with Timer() as t_fit:
        clf = fit_ridge(F_l, ctx.y_L, ctx.cfg)
    with Timer() as t_pred:
        y_pred = np.asarray(clf.predict(np.asarray(ctx.F_test)))
    return MethodResult(
        y_pred=y_pred,
        fit_seconds=t_fit.seconds,
        predict_seconds=t_pred.seconds,
        extra={
            "ridge_cv": bool(getattr(clf, "_tscfree_cv", False)),
            "train_signature": fitted_signature(clf, ctx.L_idx),
        },
    )


def m_self_train(ctx: MethodContext) -> MethodResult:
    """sklearn's SelfTrainingClassifier around logistic regression.

    Operates on the standardised L u U features; unlabeled entries are -1.
    """
    cfg = ctx.cfg
    le, y_int = _encode(ctx.classes, ctx.y_L)
    y_semi = np.full(ctx.n_train, -1, dtype=int)
    y_semi[ctx.L_idx] = y_int

    Z = np.asarray(ctx.Z_train, dtype=np.float64)
    base = make_logreg(cfg, seed_from("self_train", ctx.dataset, ctx.k, ctx.seed))
    clf = SelfTrainingClassifier(
        base,
        threshold=float(cfg["self_train"]["threshold"]),
        max_iter=int(cfg["self_train"]["max_iter"]),
    )
    with Timer() as t_fit:
        clf.fit(Z, y_semi)
    n_pseudo = int(np.sum(clf.transduction_[ctx.U_idx] != -1))
    with Timer() as t_pred:
        y_pred = le.inverse_transform(
            np.asarray(clf.predict(np.asarray(ctx.Z_test, dtype=np.float64)), dtype=int)
        )
    return MethodResult(
        y_pred=np.asarray(y_pred),
        fit_seconds=t_fit.seconds,
        predict_seconds=t_pred.seconds,
        n_pseudo_selected=n_pseudo,
        extra={
            "n_iter": int(getattr(clf, "n_iter_", -1)),
            "train_signature": fitted_signature(clf, np.asarray(clf.transduction_)),
        },
    )


def m_label_spread(ctx: MethodContext) -> MethodResult:
    """LabelSpreading over the whole training set, then ridge on every label.

    Note the two-stage structure required by the specification: propagation is
    transductive (it labels ``U``), so an inductive model -- the ridge -- must
    be refitted on ``L`` plus *all* propagated labels in order to predict the
    test set.
    """
    cfg = ctx.cfg
    if str(cfg["label_spread"]["kernel"]).lower() != "knn":
        raise ValueError(
            "only the knn kernel is implemented for label_spread; the cached graph "
            f"is a kNN connectivity graph, but the config asks for "
            f"{cfg['label_spread']['kernel']!r}"
        )
    nk = int(cfg["label_spread"]["n_neighbors"])
    W = ctx.graphs[f"knn{nk}"]
    le, y_int = _encode(ctx.classes, ctx.y_L)
    y_semi = np.full(ctx.n_train, -1, dtype=int)
    y_semi[ctx.L_idx] = y_int

    with Timer() as t_fit:
        dist, cls = run_label_spreading(
            W,
            y_semi,
            alpha=float(cfg["label_spread"]["alpha"]),
            max_iter=int(cfg["label_spread"]["max_iter"]),
            tol=float(cfg["label_spread"]["tol"]),
        )
        prop = cls[np.argmax(dist, axis=1)]
        y_all = prop.copy()
        y_all[ctx.L_idx] = y_int  # never overwrite a real label
        clf = fit_ridge(np.asarray(ctx.F_train), le.inverse_transform(y_all), cfg)
    with Timer() as t_pred:
        y_pred = np.asarray(clf.predict(np.asarray(ctx.F_test)))
    return MethodResult(
        y_pred=y_pred,
        fit_seconds=t_fit.seconds,
        predict_seconds=t_pred.seconds,
        n_pseudo_selected=int(len(ctx.U_idx)),
        extra={
            "ridge_cv": bool(getattr(clf, "_tscfree_cv", False)),
            "train_signature": fitted_signature(clf, y_all),
        },
    )


def m_augment(ctx: MethodContext) -> MethodResult:
    """Labeled-data-only competitor: manufacture more labeled series.

    Transform is fitted on ``X_train[L]`` (tier ``SUP-strict``) because this
    baseline is defined to use no unlabeled data whatsoever.
    """
    cfg = ctx.cfg["augment"]
    rng = rng_for("augment", ctx.dataset, ctx.k, ctx.seed)
    Xl = ctx.X_train[ctx.L_idx]
    X_aug, y_aug = augment_labeled(
        Xl,
        ctx.y_L,
        n_copies=int(cfg["n_copies"]),
        rng=rng,
        jitter_sigma=float(cfg["jitter_sigma"]),
        scaling_sigma=float(cfg["scaling_sigma"]),
        warp_n_knots=int(cfg["warp_n_knots"]),
        warp_sigma=float(cfg["warp_sigma"]),
        slice_ratio=float(cfg["slice_ratio"]),
    )
    (F_aug, F_te), feat_s = transform_fit_on(
        Xl, [X_aug, ctx.X_test], ctx.transform, ctx.n_kernels, ctx.transform_seed
    )
    with Timer() as t_fit:
        clf = fit_ridge(F_aug, y_aug, ctx.cfg)
    with Timer() as t_pred:
        y_pred = np.asarray(clf.predict(F_te))
    return MethodResult(
        y_pred=y_pred,
        feature_seconds=feat_s,
        fit_seconds=t_fit.seconds,
        predict_seconds=t_pred.seconds,
        extra={
            "n_augmented": int(X_aug.shape[0]),
            "ridge_cv": bool(getattr(clf, "_tscfree_cv", False)),
            "train_signature": fitted_signature(clf, ctx.L_idx),
        },
    )


# ---------------------------------------------------------------------------
# SAFE-SSL  (§6)
# ---------------------------------------------------------------------------


def _ssl_core(
    ctx: MethodContext,
    L_idx: np.ndarray,
    y_L: np.ndarray,
    W: Any,
    params: dict[str, Any],
) -> tuple[Any, int, list[int]]:
    """Rounds of (spread -> class-balanced select -> refit ridge).

    Returns the final classifier, the number of pseudo-labels used by it, and
    the per-round selection counts.

    Selection restarts from ``L`` every round rather than accumulating.
    Accumulation lets a mistake made in round 1 stay in the training set for
    every later round and be reinforced -- textbook confirmation bias.  Fresh
    selection means a point can be dropped again once the model changes its
    mind about it.
    """
    le, y_int = _encode(ctx.classes, y_L)
    C = ctx.n_classes
    n = ctx.n_train
    mask = np.zeros(n, dtype=bool)
    mask[L_idx] = True
    U_idx = np.flatnonzero(~mask)

    seeds = np.full(n, -1, dtype=int)
    seeds[L_idx] = y_int

    clf = fit_ridge(np.asarray(ctx.F_train[L_idx]), y_L, ctx.cfg)
    counts: list[int] = []
    n_pseudo = 0

    if U_idx.size == 0:
        return clf, 0, counts

    for _ in range(int(params["rounds"])):
        dist, cls = run_label_spreading(
            W,
            seeds,
            alpha=float(params["alpha"]),
            max_iter=int(params["max_iter"]),
            tol=float(params.get("tol", 1e-3)),
        )
        # restrict to the ORIGINAL unlabeled pool, and re-index class columns
        P_U = np.zeros((U_idx.size, C), dtype=float)
        for j, c in enumerate(cls):
            if 0 <= int(c) < C:
                P_U[:, int(c)] = dist[U_idx, j]
        rows, labs = class_balanced_select(P_U, float(params["rho"]), C)
        sel_idx = U_idx[rows]
        counts.append(int(sel_idx.size))

        y_fit = np.concatenate([y_L, le.inverse_transform(labs)]) if sel_idx.size else y_L
        idx_fit = np.concatenate([L_idx, sel_idx]) if sel_idx.size else L_idx
        clf = fit_ridge(np.asarray(ctx.F_train[idx_fit]), y_fit, ctx.cfg)
        n_pseudo = int(sel_idx.size)

        # seeds for the next round: real labels + this round's selection only
        seeds = np.full(n, -1, dtype=int)
        seeds[L_idx] = y_int
        if sel_idx.size:
            seeds[sel_idx] = labs

    return clf, n_pseudo, counts


def _valve_estimates(
    ctx: MethodContext, W: Any, params: dict[str, Any], n_folds: int
) -> tuple[float, float, int]:
    """Cross-validated accuracy of the SSL model vs. sup_tf, on LABELED DATA ONLY.

    Each fold hides a slice of ``L``: those instances join the unlabeled pool
    for the SSL run and are then used as the (only) validation set.  The test
    set is never involved -- choosing the branch with test data would be
    cheating, and ``tests/test_no_test_leakage.py`` enforces that it cannot
    happen.
    """
    y_L = ctx.y_L
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed_from("valve", ctx.dataset, ctx.k, ctx.seed)
    )
    ssl_correct = 0
    sup_correct = 0
    total = 0
    for tr, va in skf.split(np.zeros(len(y_L)), y_L):
        L_sub = ctx.L_idx[tr]
        y_sub = y_L[tr]
        hold = ctx.L_idx[va]
        y_hold = y_L[va]
        if len(np.unique(y_sub)) < 2:
            continue
        ssl_clf, _, _ = _ssl_core(ctx, L_sub, y_sub, W, params)
        sup_clf = fit_ridge(np.asarray(ctx.F_train[L_sub]), y_sub, ctx.cfg)
        Fh = np.asarray(ctx.F_train[hold])
        ssl_correct += int(np.sum(np.asarray(ssl_clf.predict(Fh)) == y_hold))
        sup_correct += int(np.sum(np.asarray(sup_clf.predict(Fh)) == y_hold))
        total += len(y_hold)
    if total == 0:
        return float("nan"), float("nan"), 0
    return ssl_correct / total, sup_correct / total, total


def m_safe_ssl(ctx: MethodContext) -> MethodResult:
    """Our method: class-balanced graph pseudo-labeling with a labeled-only safety valve."""
    from .splits import n_folds_for_labeled_set

    sc = ctx.cfg["safe_ssl"]
    params = {
        "alpha": float(sc["alpha"]),
        "rho": float(sc["rho"]),
        "rounds": int(sc["rounds"]),
        "max_iter": int(sc["max_iter"]),
        "tol": float(sc.get("tol", 1e-3)),
    }
    nk = int(sc["k_neighbors"])
    W = ctx.graphs[f"cos{nk}"]
    delta = float(sc["delta"])

    with Timer() as t_fit:
        ssl_clf, n_pseudo, counts = _ssl_core(ctx, ctx.L_idx, ctx.y_L, W, params)

        n_folds = n_folds_for_labeled_set(ctx.k, int(sc.get("valve_max_folds", 5)))
        if n_folds == 0:
            # k = 1: no held-out labeled data exists, so no honest estimate is
            # possible. The valve abstains and we keep the SSL model; this is a
            # documented policy choice, not an estimate.
            fired = False
            valve = {"status": "abstain_k_too_small", "ssl_cv": None, "sup_cv": None}
            final = ssl_clf
        else:
            ssl_cv, sup_cv, n_val = _valve_estimates(ctx, W, params, n_folds)
            if not np.isfinite(ssl_cv) or not np.isfinite(sup_cv):
                fired = False
                valve = {"status": "abstain_no_valid_folds", "ssl_cv": None, "sup_cv": None}
                final = ssl_clf
            elif ssl_cv > sup_cv + delta:
                fired = False
                valve = {"status": "ssl_kept", "ssl_cv": ssl_cv, "sup_cv": sup_cv, "n_val": n_val}
                final = ssl_clf
            else:
                fired = True
                valve = {
                    "status": "reverted_to_supervised",
                    "ssl_cv": ssl_cv,
                    "sup_cv": sup_cv,
                    "n_val": n_val,
                }
                final = fit_ridge(np.asarray(ctx.F_train[ctx.L_idx]), ctx.y_L, ctx.cfg)
                n_pseudo = 0
    with Timer() as t_pred:
        y_pred = np.asarray(final.predict(np.asarray(ctx.F_test)))

    return MethodResult(
        y_pred=y_pred,
        fit_seconds=t_fit.seconds,
        predict_seconds=t_pred.seconds,
        n_pseudo_selected=int(n_pseudo),
        safe_ssl_fired=bool(fired),
        extra={
            "round_selection_counts": counts,
            "valve": valve,
            "train_signature": fitted_signature(final, ctx.L_idx, np.asarray(counts, dtype=int)),
        },
    )


def m_safe_ssl_novalve(ctx: MethodContext) -> MethodResult:
    """SAFE-SSL with the safety valve removed -- the ablation arm for RQ5.

    Table T4 asks how much damage the valve prevents.  Answering that needs the
    counterfactual: the same class-balanced pseudo-labeling, always applied.
    Reported as its own arm rather than inferred, so the number is measured.
    """
    sc = ctx.cfg["safe_ssl"]
    params = {
        "alpha": float(sc["alpha"]),
        "rho": float(sc["rho"]),
        "rounds": int(sc["rounds"]),
        "max_iter": int(sc["max_iter"]),
        "tol": float(sc.get("tol", 1e-3)),
    }
    W = ctx.graphs[f"cos{int(sc['k_neighbors'])}"]
    with Timer() as t_fit:
        clf, n_pseudo, counts = _ssl_core(ctx, ctx.L_idx, ctx.y_L, W, params)
    with Timer() as t_pred:
        y_pred = np.asarray(clf.predict(np.asarray(ctx.F_test)))
    return MethodResult(
        y_pred=y_pred,
        fit_seconds=t_fit.seconds,
        predict_seconds=t_pred.seconds,
        n_pseudo_selected=int(n_pseudo),
        safe_ssl_fired=False,
        extra={
            "round_selection_counts": counts,
            "valve": {"status": "disabled"},
            "train_signature": fitted_signature(clf, ctx.L_idx, np.asarray(counts, dtype=int)),
        },
    )


# ---------------------------------------------------------------------------
# The ceiling (uses ALL training labels -- never inside the guarded path)
# ---------------------------------------------------------------------------


def run_sup_full(ctx: MethodContext, y_train_full: np.ndarray) -> MethodResult:
    """``SUP-full``: MiniRocket on ``X_train``, ridge on *all* training labels.

    This is the ceiling for the recovery ratio.  It is deliberately not part of
    the guarded method registry: it is scoring infrastructure, not a competitor
    that operates under the label budget.
    """
    with Timer() as t_fit:
        clf = fit_ridge(np.asarray(ctx.F_train), np.asarray(y_train_full), ctx.cfg)
    with Timer() as t_pred:
        y_pred = np.asarray(clf.predict(np.asarray(ctx.F_test)))
    return MethodResult(
        y_pred=y_pred, fit_seconds=t_fit.seconds, predict_seconds=t_pred.seconds
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHODS: dict[str, Callable[[MethodContext], MethodResult]] = {
    "1nn_dtw": m_1nn_dtw,
    "sup_strict": m_sup_strict,
    "sup_tf": m_sup_tf,
    "self_train": m_self_train,
    "label_spread": m_label_spread,
    "augment": m_augment,
    "safe_ssl": m_safe_ssl,
    "safe_ssl_novalve": m_safe_ssl_novalve,
}


def run_method(method_id: str, ctx: MethodContext) -> MethodResult:
    if method_id not in METHODS:
        raise KeyError(f"unknown method {method_id!r}; known: {sorted(METHODS)}")
    return METHODS[method_id](ctx)
