"""Optional deep semi-supervised baseline (§12) -- the RQ6 Pareto point.

A small 1D-CNN (3 conv blocks, ~50k parameters) trained on the labeled set and
then refined with pseudo-labeling on the unlabeled pool.

Two things make this comparison honest and both are enforced here:

* **A stated wall-clock budget per dataset.**  Training stops when the budget is
  exhausted, and the budget is written into every result row.  If the model
  cannot be trained within the budget, the row says so -- nothing is
  extrapolated.
* **Same device as the rest of the study.**  The core study is CPU-only, so the
  default device is CPU: a GPU-trained arm on a CPU Pareto plot would be
  meaningless.  ``--device cuda`` is available and is recorded in the row, but
  then the plot must be labelled accordingly.

The protocol is otherwise identical to every other method: the transform-free
model sees ``X_train[L]`` with labels and ``X_train[U]`` without, and the test
set only at prediction time.

Usage::

    python scripts/run_deep_baseline.py --budget-seconds 60
    python scripts/run_deep_baseline.py --datasets ECG200 Wine --budgets 5 --seeds 0 1 2
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
from tscfree.data import load  # noqa: E402
from tscfree.experiment import limit_blas_threads  # noqa: E402
from tscfree.metrics import classification_metrics  # noqa: E402
from tscfree.splits import labeled_unlabeled_split  # noqa: E402
from tscfree.utils import Timer, package_versions, read_json, seed_from, write_json  # noqa: E402


def build_model(n_classes: int, length: int, width: int = 80):
    """3 conv blocks + global average pooling.  ~50k parameters."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(1, width, kernel_size=7, padding=3),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.Conv1d(width, width, kernel_size=5, padding=2),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.Conv1d(width, width, kernel_size=3, padding=1),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(width, n_classes),
    )


def _train(model, X, y, epochs, budget_left, device, lr=1e-3, batch=32, seed=0):
    """Train until ``epochs`` or until the wall-clock budget runs out."""
    import torch
    import torch.nn as nn

    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = X.shape[0]
    model.train()
    t0 = time.perf_counter()
    done = 0
    for _ in range(epochs):
        if time.perf_counter() - t0 >= budget_left:
            break
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            out = model(X[idx].to(device))
            loss = lossf(out, y[idx].to(device))
            loss.backward()
            opt.step()
        done += 1
    return done, time.perf_counter() - t0


def run_one(dataset: str, k: int, seed: int, budget: float, device: str,
            epochs: int, threshold: float) -> dict:
    import torch

    torch.manual_seed(seed_from("deep", dataset, k, seed))
    torch.set_num_threads(1)

    ds = load(dataset)
    L, U = labeled_unlabeled_split(ds.y_train, k=k, seed=seed, dataset=dataset)
    classes = np.unique(ds.y_train)
    cls_to_i = {c: i for i, c in enumerate(classes)}
    y_L = np.array([cls_to_i[c] for c in ds.y_train[L]], dtype=np.int64)

    Xtr = torch.tensor(ds.X_train, dtype=torch.float32)
    Xte = torch.tensor(ds.X_test, dtype=torch.float32)
    model = build_model(len(classes), ds.length).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    with Timer() as t_total:
        # stage 1: supervised on L, half the budget
        e1, s1 = _train(
            model, Xtr[L], torch.tensor(y_L), epochs, budget / 2, device,
            seed=seed_from("deep-sup", dataset, k, seed)
        )
        # stage 2: pseudo-label U above the confidence threshold, retrain on the union
        model.eval()
        with torch.no_grad():
            logits = model(Xtr[U].to(device)).cpu()
            probs = torch.softmax(logits, dim=1)
            conf, hard = probs.max(dim=1)
        keep = (conf >= threshold).numpy()
        n_pseudo = int(keep.sum())
        if n_pseudo:
            X2 = torch.cat([Xtr[L], Xtr[U][torch.tensor(keep)]])
            y2 = torch.cat([torch.tensor(y_L), hard[torch.tensor(keep)]])
        else:
            X2, y2 = Xtr[L], torch.tensor(y_L)
        e2, s2 = _train(
            model, X2, y2, epochs, max(budget - s1, 0.0), device,
            seed=seed_from("deep-ssl", dataset, k, seed)
        )
        model.eval()
        with torch.no_grad():
            pred = model(Xte.to(device)).argmax(dim=1).cpu().numpy()

    y_pred = classes[pred]
    met = classification_metrics(ds.y_test, y_pred)
    return {
        "dataset": dataset,
        "method": "deep_ssl",
        "k": k,
        "seed": seed,
        "n_train": ds.n_train,
        "n_test": ds.n_test,
        "length": ds.length,
        "n_classes": ds.n_classes,
        "n_labeled": int(len(L)),
        "n_unlabeled": int(len(U)),
        **met,
        "n_pseudo_selected": n_pseudo,
        "safe_ssl_fired": None,
        "feature_seconds": 0.0,
        "fit_seconds": s1 + s2,
        "predict_seconds": 0.0,
        "total_seconds": t_total.seconds,
        "budget_seconds": budget,
        "budget_exhausted": bool(s1 + s2 >= budget * 0.98),
        "epochs_stage1": e1,
        "epochs_stage2": e2,
        "n_parameters": int(n_params),
        "device": device,
        "pseudo_threshold": threshold,
        "versions": package_versions(),
    }


def _worker(d: str, k: int, s: int, budget: float, device: str, epochs: int,
            threshold: float, out_dir: Path, failures_dir: Path) -> str:
    """One cell, in its own process, pinned to a single CPU thread.

    The single-thread pin matters for RQ6: every other method in the study was
    timed the same way, so "seconds per cell" means the same thing on both axes
    of the Pareto figure.
    """
    limit_blas_threads(1)
    p = out_dir / f"{d}__k{k}__s{s}.json"
    try:
        row = run_one(d, k, s, budget, device, epochs, threshold)
        write_json(p, {"rows": [row], "failed_methods": [],
                       "key": {"dataset": d, "k": k, "seed": s, "tag": "deep"}})
        return "ok"
    except Exception:
        write_json(
            failures_dir / f"deep__{d}__k{k}__s{s}.json",
            {"dataset": d, "k": k, "seed": s, "method": "deep_ssl",
             "traceback": traceback.format_exc()},
        )
        return "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--budget-seconds", type=float, default=60.0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-jobs", type=int, default=-3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not installed; the deep baseline is optional -- skipping.")
        print("The paper must then state that RQ6 has no deep point, not extrapolate one.")
        return 0

    limit_blas_threads(1)
    cfg = load_config()
    names = args.datasets or dataset_names()
    budgets = args.budgets or cfg.label_budgets
    out_dir = ROOT / "results" / "deep"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = cfg.path_for("failures")
    failures_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        (d, k, s)
        for d in names
        for k in budgets
        for s in args.seeds
        if args.force or not (out_dir / f"{d}__k{k}__s{s}.json").exists()
    ]
    total = len(names) * len(budgets) * len(args.seeds)
    print(f"deep_ssl: {len(names)} datasets x {len(budgets)} budgets x {len(args.seeds)} "
          f"seeds = {total} cells ({total - len(todo)} already done), "
          f"budget={args.budget_seconds}s/cell on {args.device}, n_jobs={args.n_jobs}")

    t0 = time.perf_counter()
    res: list[str] = []
    if todo:
        res = Parallel(n_jobs=args.n_jobs, verbose=5, batch_size=1)(
            delayed(_worker)(d, k, s, args.budget_seconds, args.device, args.epochs,
                             args.threshold, out_dir, failures_dir)
            for (d, k, s) in todo
        )
    dt = time.perf_counter() - t0
    n_ok = sum(r == "ok" for r in res)
    n_fail = sum(r == "failed" for r in res)
    print(f"\nfinished in {dt / 60:.1f} min: {n_ok} ok, {total - len(todo)} skipped, "
          f"{n_fail} failed")
    if n_fail:
        print(f"!!! {n_fail} failures -- see {failures_dir}")

    files = sorted(out_dir.glob("*.json"))
    if files:
        rows = [r for f in files for r in read_json(f)["rows"]]
        acc = float(np.mean([r["accuracy"] for r in rows]))
        sec = float(np.mean([r["total_seconds"] for r in rows]))
        exh = float(np.mean([r["budget_exhausted"] for r in rows]))
        par = int(np.median([r["n_parameters"] for r in rows]))
        print(f"mean accuracy {acc:.4f}, mean wall-clock {sec:.1f}s/cell, "
              f"budget exhausted in {exh * 100:.0f}% of cells "
              f"(budget = {args.budget_seconds}s, device = {args.device}, "
              f"median {par} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
