"""Main driver: the full grid, and the ablations.

Resumable: one JSON per ``(dataset, k, seed)`` cell; existing files are skipped
unless ``--force``.  Failures never stop the run -- they land in
``results/failures/`` with a full traceback and are counted loudly at the end.

Usage::

    python scripts/run_experiments.py --estimate      # dry run + runtime estimate
    python scripts/run_experiments.py                 # the full grid
    python scripts/run_experiments.py --ablation      # the ablation grid
    python scripts/run_experiments.py --datasets ECG200 --budgets 5 --seeds 0 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from tscfree.config import dataset_names, load_config  # noqa: E402
from tscfree.experiment import (  # noqa: E402
    compute_sup_full,
    iter_grid,
    limit_blas_threads,
    run_cell,
)
from tscfree.features import build_features, build_unsup_cache  # noqa: E402
from tscfree.utils import human_bytes, read_json, write_json  # noqa: E402

# ---------------------------------------------------------------------------
# sup_full ceiling (one per dataset)
# ---------------------------------------------------------------------------


def ensure_sup_full(names: list[str], cfg, force: bool = False, n_jobs: int = -3) -> dict[str, float]:
    """Compute (or load) the fully-supervised ceiling for each dataset."""
    path = ROOT / "results" / "sup_full.json"
    have: dict[str, dict] = {}
    if path.exists() and not force:
        have = read_json(path)
    todo = [n for n in names if n not in have]
    if todo:
        print(f"Computing sup_full ceiling for {len(todo)} dataset(s) ...")

        def _one(n: str) -> dict:
            limit_blas_threads(1)
            return compute_sup_full(n, cfg)

        out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one)(n) for n in todo)
        for row in out:
            have[row["dataset"]] = row
        write_json(path, have)
    return {k: float(v["accuracy"]) for k, v in have.items()}


# ---------------------------------------------------------------------------
# Grid runners
# ---------------------------------------------------------------------------


def run_main_grid(args, cfg) -> int:
    names = args.datasets or dataset_names()
    budgets = args.budgets or cfg.label_budgets
    seeds = args.seeds if args.seeds is not None else cfg.seeds
    methods = args.methods or cfg.methods

    acc_full = ensure_sup_full(names, cfg, force=args.force, n_jobs=args.n_jobs)
    cells = iter_grid(names, budgets, seeds)
    out_dir = cfg.path_for("results_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [c for c in cells if args.force or not (out_dir / f"{c[0]}__k{c[1]}__s{c[2]}.json").exists()]
    print(
        f"grid: {len(names)} datasets x {len(budgets)} budgets x {len(seeds)} seeds "
        f"= {len(cells)} cells ({len(methods)} methods each = {len(cells) * len(methods)} rows)"
    )
    print(f"      {len(cells) - len(todo)} already done, {len(todo)} to run, n_jobs={args.n_jobs}")
    if args.estimate:
        return estimate(todo, methods, cfg, acc_full, args)
    if not todo:
        print("nothing to do")
        return summarise(cfg)

    t0 = time.perf_counter()
    Parallel(n_jobs=args.n_jobs, verbose=cfg["runtime"]["verbose"], batch_size=1)(
        delayed(run_cell)(d, k, s, methods, cfg, acc_full.get(d), out_dir, None, args.force)
        for (d, k, s) in todo
    )
    dt = time.perf_counter() - t0
    print(f"\nfinished {len(todo)} cells in {dt / 60:.1f} min ({dt / max(len(todo), 1):.2f} s/cell)")
    return summarise(cfg)


def estimate(todo, methods, cfg, acc_full, args) -> int:
    """Time a handful of representative cells and extrapolate."""
    if not todo:
        print("nothing to estimate")
        return 0
    rng = np.random.default_rng(0)
    sample = [todo[i] for i in rng.choice(len(todo), size=min(6, len(todo)), replace=False)]
    tmp = ROOT / "results" / "_estimate"
    tmp.mkdir(parents=True, exist_ok=True)
    times = []
    print("\ntiming a sample of cells (sequential, single worker):")
    for d, k, s in sample:
        t0 = time.perf_counter()
        run_cell(d, k, s, methods, cfg, acc_full.get(d), tmp, None, True)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  {d:<32} k={k:<3} seed={s:<3} {dt:8.2f} s")
    med = float(np.median(times))
    workers = args.n_jobs if args.n_jobs > 0 else max(1, (os_cpu() + args.n_jobs + 1))
    total = med * len(todo) / max(workers, 1)
    print(f"\nmedian {med:.2f} s/cell; {len(todo)} cells over ~{workers} workers")
    print(f"ESTIMATED TOTAL: {total / 60:.1f} min  ({total / 3600:.2f} h)")
    return 0


def os_cpu() -> int:
    import os

    return os.cpu_count() or 4


def run_ablation_grid(args, cfg) -> int:
    """Ablate one constant at a time around the default configuration."""
    ab = cfg["ablation"]
    all_names = args.datasets or dataset_names()
    names = all_names[: int(ab["n_datasets"])]
    budgets = args.budgets or list(ab["label_budgets"])
    seeds = args.seeds if args.seeds is not None else list(ab["seeds"])
    acc_full = ensure_sup_full(names, cfg, n_jobs=args.n_jobs)
    out_dir = cfg.path_for("results_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    section_of = {
        "rho": "safe_ssl",
        "rounds": "safe_ssl",
        "k_neighbors": "safe_ssl",
        "alpha": "safe_ssl",
        "delta": "safe_ssl",
        "n_kernels": "features",
        "transform": "features",
    }
    defaults = {
        "rho": cfg["safe_ssl"]["rho"],
        "rounds": cfg["safe_ssl"]["rounds"],
        "k_neighbors": cfg["safe_ssl"]["k_neighbors"],
        "alpha": cfg["safe_ssl"]["alpha"],
        "delta": cfg["safe_ssl"]["delta"],
        "n_kernels": cfg["features"]["n_kernels"],
        "transform": cfg["features"]["transform"],
    }

    # The n_kernels / transform axes need their own feature caches; build them
    # up front (once per configuration) rather than discovering the gap 3,000
    # cells into the run.
    tseed = int(cfg["features"]["transform_seed"])
    needed = {(cfg["features"]["transform"], int(cfg["features"]["n_kernels"]))}
    for val in ab["axes"].get("n_kernels", []):
        needed.add((cfg["features"]["transform"], int(val)))
    for val in ab["axes"].get("transform", []):
        needed.add((str(val), int(cfg["features"]["n_kernels"])))
    for tr, nk in sorted(needed):
        for d in names:
            fm = build_features(d, tr, nk, tseed)
            um = build_unsup_cache(d, tr, nk, tseed, neighbor_values=(3, 7, 15))
            if not (fm.get("cache_hit") and um.get("cache_hit")):
                print(f"  built {tr} n={nk} for {d} "
                      f"(dim={fm.get('feature_dim')}, {fm.get('total_seconds', 0):.1f}s)")

    jobs = []
    for axis, values in ab["axes"].items():
        sec = section_of[axis]
        for val in values:
            tag = f"{axis}={val}"
            for d in names:
                for k in budgets:
                    for s in seeds:
                        p = out_dir / f"{tag}__{d}__k{k}__s{s}.json"
                        if p.exists() and not args.force:
                            continue
                        jobs.append((tag, {sec: {axis: val}}, d, k, s))

    print(f"ablation: {len(jobs)} cells to run over {len(names)} datasets "
          f"(defaults: {defaults})")
    if args.estimate or not jobs:
        return 0

    methods = ["sup_strict", "sup_tf", "safe_ssl"]
    t0 = time.perf_counter()
    Parallel(n_jobs=args.n_jobs, verbose=cfg["runtime"]["verbose"], batch_size=1)(
        delayed(run_cell)(
            d, k, s, methods, cfg, acc_full.get(d), out_dir, None, args.force, ov, tag
        )
        for (tag, ov, d, k, s) in jobs
    )
    dt = time.perf_counter() - t0
    print(f"\nablation finished in {dt / 60:.1f} min")
    return 0


def run_transform_seed_check(args, cfg) -> int:
    """How much of the variance comes from the MiniRocket seed rather than the label draw?

    The main grid caches one transform seed for all evaluation seeds (NOTES.md
    §3).  That is only defensible if the transform seed contributes little
    variance, so we measure it instead of asserting it: a handful of datasets,
    three transform seeds, several evaluation seeds, and a decomposition of the
    accuracy standard deviation into the two sources.
    """
    names = args.datasets or dataset_names()[:5]
    tseeds = [0, 1, 2]
    eval_seeds = args.seeds if args.seeds is not None else [0, 1, 2, 3, 4]
    k = (args.budgets or [5])[0]
    methods = args.methods or ["sup_tf", "safe_ssl"]
    transform = cfg["features"]["transform"]
    n_kernels = int(cfg["features"]["n_kernels"])
    out_dir = ROOT / "results" / "_transform_seed_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"transform-seed sensitivity: {names} x tseeds {tseeds} x eval seeds {eval_seeds}, "
          f"k={k}, methods={methods}")
    for ts in tseeds:
        for d in names:
            build_features(d, transform, n_kernels, ts)
            build_unsup_cache(d, transform, n_kernels, ts, neighbor_values=(3, 7, 15))

    rows = []
    for ts in tseeds:
        for d in names:
            for s in eval_seeds:
                res = run_cell(
                    d, k, s, methods, cfg, None, out_dir, None, True,
                    {"features": {"transform_seed": ts}}, f"ts{ts}",
                )
                for r in read_json(res["path"])["rows"]:
                    rows.append({"dataset": d, "tseed": ts, "seed": s,
                                 "method": r["method"], "accuracy": r["accuracy"]})

    import collections

    by_method = collections.defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    summary = {}
    for m, rs in by_method.items():
        per_ds = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rs:
            per_ds[r["dataset"]][r["tseed"]].append(r["accuracy"])
        sd_t, sd_e = [], []
        for _d, byt in per_ds.items():
            means = [float(np.mean(v)) for v in byt.values()]
            sd_t.append(float(np.std(means, ddof=1)) if len(means) > 1 else 0.0)
            within = [float(np.std(v, ddof=1)) for v in byt.values() if len(v) > 1]
            sd_e.append(float(np.mean(within)) if within else 0.0)
        summary[m] = {
            "sd_across_transform_seeds": float(np.mean(sd_t)),
            "sd_across_evaluation_seeds": float(np.mean(sd_e)),
            "ratio": float(np.mean(sd_t) / max(np.mean(sd_e), 1e-12)),
            "n_datasets": len(per_ds),
        }
        print(f"  {m:<12} sd(transform seed) = {summary[m]['sd_across_transform_seeds']:.4f}  "
              f"sd(evaluation seed) = {summary[m]['sd_across_evaluation_seeds']:.4f}  "
              f"ratio = {summary[m]['ratio']:.2f}")
    write_json(ROOT / "results" / "transform_seed_check.json",
               {"datasets": names, "transform_seeds": tseeds, "eval_seeds": eval_seeds,
                "k": k, "summary": summary, "rows": rows})
    return 0


def summarise(cfg) -> int:
    raw = cfg.path_for("results_raw")
    fails = cfg.path_for("failures")
    files = sorted(raw.glob("*.json"))
    n_rows = 0
    failed_methods: dict[str, int] = {}
    for f in files:
        try:
            payload = read_json(f)
        except Exception:
            continue
        n_rows += len(payload.get("rows", []))
        for m in payload.get("failed_methods", []):
            failed_methods[m] = failed_methods.get(m, 0) + 1
    fail_files = sorted(fails.glob("*.json")) if fails.exists() else []

    print("\n" + "=" * 70)
    print(f"RESULT FILES : {len(files)}   ROWS: {n_rows}")
    print(f"DISK         : {human_bytes(sum(f.stat().st_size for f in files))}")
    if failed_methods or fail_files:
        print(f"!!! FAILURES : {len(fail_files)} traceback file(s) in {fails}")
        for m, c in sorted(failed_methods.items(), key=lambda x: -x[1]):
            print(f"      {m:<20} {c} cell(s)")
        print("!!! Nothing was estimated or interpolated -- those cells are simply missing.")
    else:
        print("FAILURES     : none")
    print("=" * 70)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--transform-seed-check", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if args.n_jobs is None:
        args.n_jobs = int(cfg["runtime"]["n_jobs"])
    limit_blas_threads(1)

    if args.summary_only:
        return summarise(cfg)
    if args.transform_seed_check:
        return run_transform_seed_check(args, cfg)
    if args.ablation:
        return run_ablation_grid(args, cfg)
    return run_main_grid(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
