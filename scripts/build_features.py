"""Build every cached, label-independent artefact.

For each selected dataset:

1. the random-kernel transform fitted on the whole training set (== ``L u U``)
   and applied to train and test;
2. the standardiser (fitted on ``L u U``) and the kNN graphs used by
   ``label_spread`` and ``safe_ssl``;
3. the ``(n_test, n_train)`` DTW distance matrix for the 1NN-DTW baseline.

None of these depends on a label, on ``k`` or on the evaluation seed -- which
is precisely the paper's point -- so they are computed once and reused by all
14,000 cells.

Usage::

    python scripts/build_features.py
    python scripts/build_features.py --transform multirocket --n-kernels 6250
    python scripts/build_features.py --datasets ECG200 Wine --force
    python scripts/build_features.py --no-dtw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscfree.config import dataset_names, load_config  # noqa: E402
from tscfree.experiment import limit_blas_threads  # noqa: E402
from tscfree.features import build_dtw_matrix, build_features, build_unsup_cache  # noqa: E402
from tscfree.utils import dir_size, human_bytes, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--n-kernels", type=int, default=None)
    ap.add_argument("--transform-seed", type=int, default=None)
    ap.add_argument("--neighbors", type=int, nargs="*", default=[3, 7, 15])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-dtw", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=0, help="threads for the transform (0 = all)")
    args = ap.parse_args()

    cfg = load_config()
    names = args.datasets or dataset_names()
    transform = args.transform or cfg["features"]["transform"]
    n_kernels = int(args.n_kernels or cfg["features"]["n_kernels"])
    tseed = int(
        args.transform_seed if args.transform_seed is not None else cfg["features"]["transform_seed"]
    )
    dtw_window = cfg["dtw"]["window"]
    n_jobs = args.n_jobs if args.n_jobs else -1
    if n_jobs != -1:
        limit_blas_threads(n_jobs)

    print(f"Building {transform} (n_kernels={n_kernels}, seed={tseed}) for {len(names)} datasets\n")
    header = (
        f"{'dataset':<34}{'dim':>7}{'feat_s':>9}{'unsup_s':>9}{'dtw_s':>9}{'MB':>8}  cache"
    )
    print(header)
    print("-" * len(header))

    summary = []
    t0 = time.perf_counter()
    for name in names:
        fm = build_features(
            name, transform, n_kernels, tseed, force=args.force, n_jobs=n_jobs
        )
        um = build_unsup_cache(
            name,
            transform,
            n_kernels,
            tseed,
            neighbor_values=tuple(args.neighbors),
            force=args.force,
        )
        dm = {"seconds": float("nan"), "bytes": 0, "cache_hit": None}
        if not args.no_dtw:
            dm = build_dtw_matrix(
                name,
                None if dtw_window is None else float(dtw_window),
                force=args.force,
                n_jobs=n_jobs,
            )
        mb = (fm.get("bytes", 0) + um.get("bytes", 0) + dm.get("bytes", 0)) / 1024**2
        hit = "hit" if (fm.get("cache_hit") and um.get("cache_hit")) else "built"
        print(
            f"{name:<34}{fm.get('feature_dim', 0):>7}"
            f"{fm.get('total_seconds', float('nan')):>9.2f}"
            f"{um.get('unsup_seconds_total', float('nan')):>9.2f}"
            f"{dm.get('seconds', float('nan')):>9.2f}"
            f"{mb:>8.1f}  {hit}"
        )
        summary.append(
            {
                "dataset": name,
                "feature_dim": fm.get("feature_dim"),
                "feature_seconds": fm.get("total_seconds"),
                "unsup_seconds": um.get("unsup_seconds_total"),
                "dtw_seconds": dm.get("seconds"),
                "megabytes": mb,
            }
        )

    total = time.perf_counter() - t0
    size = dir_size(cfg.path_for("features"))
    print("-" * len(header))
    print(f"total wall-clock {total:.1f}s;  feature cache on disk: {human_bytes(size)}")
    write_json(
        ROOT / "results" / f"feature_build_{transform}_n{n_kernels}_s{tseed}.json",
        {
            "transform": transform,
            "n_kernels": n_kernels,
            "transform_seed": tseed,
            "dtw_window": dtw_window,
            "n_datasets": len(names),
            "wall_clock_seconds": total,
            "cache_bytes": size,
            "per_dataset": summary,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
