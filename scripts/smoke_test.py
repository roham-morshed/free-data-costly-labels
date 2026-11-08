"""A ~2 minute end-to-end sanity run: 3 datasets x k=5 x all cheap methods x 2 seeds.

Prints a readable table.  This is the thing to run after touching anything in
``methods.py``: if a number here moves, you changed the science, not just the
code.

Usage::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --datasets ECG200 Wine Strawberry --seeds 0 1
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tscfree.config import dataset_names, load_config  # noqa: E402
from tscfree.experiment import compute_sup_full, limit_blas_threads, run_cell  # noqa: E402
from tscfree.utils import read_json  # noqa: E402

DEFAULT_METHODS = ["sup_strict", "sup_tf", "self_train", "label_spread", "augment", "safe_ssl"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=[5])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--with-dtw", action="store_true", help="also run the 1NN-DTW baseline")
    args = ap.parse_args()

    limit_blas_threads(4)
    cfg = load_config()
    names = args.datasets or dataset_names()[:3]
    methods = args.methods or (
        DEFAULT_METHODS + ["1nn_dtw"] if args.with_dtw else DEFAULT_METHODS
    )
    out = ROOT / "results" / "_smoke"
    out.mkdir(parents=True, exist_ok=True)

    print(f"SMOKE TEST  datasets={names}  k={args.budgets}  seeds={args.seeds}")
    print(f"            methods={methods}\n")

    t0 = time.perf_counter()
    ceilings = {n: compute_sup_full(n, cfg)["accuracy"] for n in names}

    rows = []
    for d in names:
        for k in args.budgets:
            for s in args.seeds:
                run_cell(d, k, s, methods, cfg, ceilings[d], out, None, force=True)
                payload = read_json(out / f"{d}__k{k}__s{s}.json")
                rows.extend(payload["rows"])
                if payload["failed_methods"]:
                    print(f"  !! failed in {d} k={k} s={s}: {payload['failed_methods']}")
    dt = time.perf_counter() - t0

    acc: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    tim: dict[str, list[float]] = defaultdict(list)
    pseudo: dict[str, list[int]] = defaultdict(list)
    fired: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        acc[(r["dataset"], r["k"], r["method"])].append(r["accuracy"])
        tim[r["method"]].append(r["total_seconds"])
        pseudo[r["method"]].append(r["n_pseudo_selected"])
        if r["safe_ssl_fired"] is not None:
            fired[r["method"]].append(bool(r["safe_ssl_fired"]))

    width = max(len(m) for m in methods) + 2
    hdr = f"{'dataset':<26}{'k':>3}" + "".join(f"{m:>{width}}" for m in methods) + f"{'ceiling':>10}"
    print(hdr)
    print("-" * len(hdr))
    for d in names:
        for k in args.budgets:
            line = f"{d:<26}{k:>3}"
            for m in methods:
                v = acc.get((d, k, m))
                line += f"{np.mean(v):>{width}.3f}" if v else f"{'--':>{width}}"
            line += f"{ceilings[d]:>10.3f}"
            print(line)
    print("-" * len(hdr))
    print(f"{'mean seconds/call':<29}" + "".join(
        f"{np.mean(tim[m]):>{width}.2f}" if tim[m] else f"{'--':>{width}}" for m in methods))
    print(f"{'mean pseudo-labels':<29}" + "".join(
        f"{np.mean(pseudo[m]):>{width}.0f}" if pseudo[m] else f"{'--':>{width}}" for m in methods))
    if fired:
        for m, v in fired.items():
            print(f"{m}: safety valve fired in {sum(v)}/{len(v)} cells")
    print(f"\ntotal wall-clock: {dt:.1f}s for {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
