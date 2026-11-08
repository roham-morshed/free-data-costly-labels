"""The cell runner: one ``(dataset, k, seed)`` cell -> one JSON result file.

Design notes
------------

* **One file per cell.**  A cell that already has a result file is skipped, so
  the whole grid is resumable and safe to interrupt.
* **All methods of a cell run together**, because they share the expensive
  per-cell setup (loading cached features, drawing the split) and because the
  three-tier decomposition of RQ2 only makes sense when ``sup_strict``,
  ``sup_tf`` and ``safe_ssl`` saw exactly the same labeled instances.
* **Failures are contained.**  A method that raises is recorded with its full
  traceback in ``results/failures/`` and the other methods still run.  Nothing
  is estimated or filled in.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .data import load
from .features import load_dtw_matrix, load_features, load_unsup
from .methods import METHOD_IDS, MethodContext, run_method, run_sup_full
from .metrics import classification_metrics, majority_class_accuracy, recovery_ratio
from .splits import labeled_unlabeled_split
from .utils import (
    CellKey,
    GuardedLabels,
    package_versions,
    read_json,
    set_all_seeds,
    write_json,
)


def limit_blas_threads(n: int = 1) -> None:
    """Pin BLAS to ``n`` threads.

    With joblib running 14 processes, letting each one spawn 16 BLAS threads
    oversubscribes the CPU by 14x and makes the run *slower*, not faster.  It
    also removes a source of run-to-run numerical nondeterminism in reductions.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)


def build_context(
    dataset: str,
    k: int,
    seed: int,
    cfg: Any,
    transform: str | None = None,
    n_kernels: int | None = None,
    transform_seed: int | None = None,
    neighbor_values: tuple[int, ...] = (7,),
    with_dtw: bool = True,
    guard_labels: bool = True,
) -> tuple[MethodContext, Any]:
    """Assemble everything a cell needs.  Returns ``(ctx, dataset_object)``."""
    fcfg = cfg["features"]
    transform = transform or fcfg["transform"]
    n_kernels = int(n_kernels or fcfg["n_kernels"])
    transform_seed = int(
        transform_seed if transform_seed is not None else fcfg["transform_seed"]
    )

    ds = load(dataset)
    L_idx, U_idx = labeled_unlabeled_split(ds.y_train, k=k, seed=seed, dataset=dataset)
    y_L = ds.y_train[L_idx]
    classes = np.unique(ds.y_train)

    fb = load_features(dataset, transform, n_kernels, transform_seed)
    ub = load_unsup(dataset, transform, n_kernels, transform_seed, neighbor_values)

    D = None
    if with_dtw:
        w = cfg["dtw"]["window"]
        try:
            D = load_dtw_matrix(dataset, None if w is None else float(w))
        except FileNotFoundError:
            D = None

    ctx = MethodContext(
        dataset=dataset,
        k=k,
        seed=seed,
        X_train=ds.X_train,
        X_test=ds.X_test,
        L_idx=L_idx,
        U_idx=U_idx,
        y_L=y_L,
        classes=classes,
        F_train=fb.F_train,
        F_test=fb.F_test,
        Z_train=ub.Z_train,
        Z_test=ub.Z_test,
        graphs=ub.graphs,
        cfg=cfg.raw if hasattr(cfg, "raw") else cfg,
        y_train_guarded=GuardedLabels(ds.y_train, L_idx) if guard_labels else None,
        D_dtw=D,
        transform=transform,
        n_kernels=n_kernels,
        transform_seed=transform_seed,
    )
    ctx.unsup_seconds = float(  # type: ignore[attr-defined]
        fb.manifest.get("total_seconds", 0.0) + ub.manifest.get("unsup_seconds_total", 0.0)
    )
    return ctx, ds


def compute_sup_full(dataset: str, cfg: Any) -> dict[str, Any]:
    """The ceiling: all training labels, transform on ``X_train``.

    Deterministic given the transform seed, so it is computed once per dataset
    rather than once per (k, seed) cell -- there is nothing for a seed to vary.
    """
    ctx, ds = build_context(dataset, k=1, seed=0, cfg=cfg, with_dtw=False, guard_labels=False)
    res = run_sup_full(ctx, ds.y_train)
    m = classification_metrics(ds.y_test, res.y_pred)
    return {
        "dataset": dataset,
        "method": "sup_full",
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "fit_seconds": res.fit_seconds,
        "predict_seconds": res.predict_seconds,
        "majority_baseline": majority_class_accuracy(ds.y_train, ds.y_test),
        "note": "deterministic given the transform seed; identical across evaluation seeds",
    }


def _lfa_for(dataset: str, cfg: Any) -> dict[str, Any] | None:
    p = Path(cfg.path_for("results_lfa")) / f"{dataset}.json"
    if p.exists():
        try:
            return read_json(p)["components"]
        except Exception:
            return None
    return None


def run_cell(
    dataset: str,
    k: int,
    seed: int,
    methods: list[str] | None = None,
    cfg: Any = None,
    acc_full: float | None = None,
    out_dir: str | Path | None = None,
    failures_dir: str | Path | None = None,
    force: bool = False,
    overrides: dict[str, Any] | None = None,
    tag: str = "",
) -> dict[str, Any]:
    """Run every method for one ``(dataset, k, seed)`` and write the result file."""
    limit_blas_threads(1)
    cfg = cfg or load_config()
    methods = list(methods or cfg.methods)
    out_dir = Path(out_dir or cfg.path_for("results_raw"))
    failures_dir = Path(failures_dir or cfg.path_for("failures"))
    key = CellKey(dataset, k, seed)
    fname = (tag + "__" if tag else "") + key.filename()
    out_path = out_dir / fname

    if out_path.exists() and not force:
        return {"status": "skipped", "path": str(out_path)}

    set_all_seeds(seed)

    raw_cfg = dict(cfg.raw)
    if overrides:
        for section, values in overrides.items():
            raw_cfg[section] = {**raw_cfg.get(section, {}), **values}

    class _CfgShim:
        raw = raw_cfg

        def __getitem__(self, key_: str) -> Any:
            return raw_cfg[key_]

        def path_for(self, name: str) -> Path:
            return cfg.path_for(name)

    shim = _CfgShim()

    tcfg = raw_cfg["features"]
    neigh = tuple(
        sorted({int(raw_cfg["label_spread"]["n_neighbors"]), int(raw_cfg["safe_ssl"]["k_neighbors"])})
    )
    ctx, ds = build_context(
        dataset,
        k,
        seed,
        shim,
        transform=tcfg["transform"],
        n_kernels=int(tcfg["n_kernels"]),
        transform_seed=int(tcfg["transform_seed"]),
        neighbor_values=neigh,
    )

    lfa = _lfa_for(dataset, cfg)
    base_row: dict[str, Any] = {
        "dataset": dataset,
        "n_train": ds.n_train,
        "n_test": ds.n_test,
        "length": ds.length,
        "n_classes": ds.n_classes,
        "class_balance_train": ds.meta.get("class_balance_train"),
        "min_class_count_train": ds.meta.get("min_class_count_train"),
        "znormalized": ds.meta.get("znormalized_train"),
        "k": k,
        "seed": seed,
        "n_labeled": int(len(ctx.L_idx)),
        "n_unlabeled": int(len(ctx.U_idx)),
        "transform": ctx.transform,
        "n_kernels": ctx.n_kernels,
        "transform_seed": ctx.transform_seed,
        "feature_dim": int(ctx.F_train.shape[1]),
        "unsup_seconds_amortized": getattr(ctx, "unsup_seconds", float("nan")),
        "majority_baseline": majority_class_accuracy(ds.y_train, ds.y_test),
        "acc_sup_full": acc_full,
        "lfa": lfa,
        "versions": package_versions(),
    }

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    acc_by_method: dict[str, float] = {}

    for mid in methods:
        try:
            res = run_method(mid, ctx)
            met = classification_metrics(ds.y_test, res.y_pred)
            acc_by_method[mid] = met["accuracy"]
            rows.append(
                {
                    **base_row,
                    "method": mid,
                    **met,
                    "n_pseudo_selected": res.n_pseudo_selected,
                    "safe_ssl_fired": res.safe_ssl_fired,
                    "feature_seconds": res.feature_seconds,
                    "fit_seconds": res.fit_seconds,
                    "predict_seconds": res.predict_seconds,
                    "total_seconds": res.feature_seconds + res.fit_seconds + res.predict_seconds,
                    "extra": res.extra,
                }
            )
        except Exception:
            tb = traceback.format_exc()
            failures.append(mid)
            failures_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                failures_dir / f"{fname.replace('.json', '')}__{mid}.json",
                {"dataset": dataset, "k": k, "seed": seed, "method": mid, "traceback": tb},
            )

    # recovery ratio needs sup_strict (this cell) and sup_full (per dataset)
    acc_strict = acc_by_method.get("sup_strict")
    for row in rows:
        if acc_strict is None or acc_full is None:
            row["recovery_ratio"] = None
        else:
            r = recovery_ratio(row["accuracy"], acc_strict, acc_full)
            row["recovery_ratio"] = None if np.isnan(r) else float(r)

    payload = {
        "key": {"dataset": dataset, "k": k, "seed": seed, "tag": tag},
        "overrides": overrides or {},
        "rows": rows,
        "failed_methods": failures,
    }
    write_json(out_path, payload)
    return {"status": "ok", "path": str(out_path), "n_rows": len(rows), "failed": failures}


def iter_grid(
    datasets: list[str], budgets: list[int], seeds: list[int]
) -> list[tuple[str, int, int]]:
    return [(d, k, s) for d in datasets for k in budgets for s in seeds]


__all__ = [
    "METHOD_IDS",
    "build_context",
    "compute_sup_full",
    "iter_grid",
    "limit_blas_threads",
    "run_cell",
]
