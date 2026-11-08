"""RQ4: compute the Label-Free Alignment components for every dataset.

LFA is a property of the *unlabeled* feature geometry only, so it is computed
once per (dataset, transform) -- it does not depend on ``k`` or the evaluation
seed.  The composite statistic requires z-scoring across datasets and is
therefore assembled in ``scripts/make_paper_artifacts.py``, not here.

Usage::

    python scripts/run_lfa_study.py
    python scripts/run_lfa_study.py --force --n-jobs 8
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from tscfree.config import dataset_names, load_config  # noqa: E402
from tscfree.data import load_meta  # noqa: E402
from tscfree.experiment import limit_blas_threads  # noqa: E402
from tscfree.features import load_features  # noqa: E402
from tscfree.lfa import compute_lfa_components  # noqa: E402
from tscfree.utils import Timer, package_versions, write_json  # noqa: E402


def one(name: str, cfg_raw: dict, out_dir: Path, force: bool) -> dict:
    limit_blas_threads(1)
    out = out_dir / f"{name}.json"
    if out.exists() and not force:
        return {"dataset": name, "status": "skipped"}
    lc = cfg_raw["lfa"]
    fc = cfg_raw["features"]
    try:
        fb = load_features(name, fc["transform"], int(fc["n_kernels"]), int(fc["transform_seed"]))
        meta = load_meta(name)
        with Timer() as t:
            comp = compute_lfa_components(
                np.asarray(fb.F_train),
                n_classes=int(meta["n_classes"]),
                seed=int(lc["seed"]),
                pca_dims=int(lc["pca_dims"]),
                n_init=int(lc["kmeans_n_init"]),
                n_bootstrap=int(lc["n_bootstrap"]),
                bootstrap_frac=float(lc["bootstrap_frac"]),
                knn_k=int(lc["knn_purity_k"]),
            )
        payload = {
            "dataset": name,
            "n_classes": int(meta["n_classes"]),
            "n_train": int(meta["n_train"]),
            "transform": fc["transform"],
            "n_kernels": int(fc["n_kernels"]),
            "transform_seed": int(fc["transform_seed"]),
            "components": comp.as_dict(),
            "seconds": t.seconds,
            "versions": package_versions(),
        }
        write_json(out, payload)
        return {"dataset": name, "status": "ok", **comp.as_dict(), "seconds": t.seconds}
    except Exception:
        tb = traceback.format_exc()
        write_json(
            Path(cfg_raw["paths"]["failures"]).resolve() / f"lfa__{name}.json",
            {"dataset": name, "stage": "lfa", "traceback": tb},
        )
        return {"dataset": name, "status": "failed", "error": tb.splitlines()[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-3)
    args = ap.parse_args()

    cfg = load_config()
    names = args.datasets or dataset_names()
    out_dir = cfg.path_for("results_lfa")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LFA over {len(names)} datasets (n_jobs={args.n_jobs}) ...")
    t0 = time.perf_counter()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(one)(n, cfg.raw, out_dir, args.force) for n in names
    )
    dt = time.perf_counter() - t0

    ok = [r for r in res if r["status"] == "ok"]
    bad = [r for r in res if r["status"] == "failed"]
    print(f"\ncomputed {len(ok)}, skipped {len(res) - len(ok) - len(bad)}, failed {len(bad)} "
          f"in {dt:.1f}s")
    if ok:
        hdr = f"{'dataset':<34}{'sil':>8}{'chi':>8}{'stab':>8}{'purity':>9}{'s':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(ok, key=lambda r: -r["sil"]):
            print(f"{r['dataset']:<34}{r['sil']:>8.3f}{r['chi']:>8.3f}"
                  f"{r['stab']:>8.3f}{r['knn_purity_proxy']:>9.3f}{r['seconds']:>7.1f}")
    for r in bad:
        print(f"FAILED {r['dataset']}: {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
