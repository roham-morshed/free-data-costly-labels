"""Download and cache the UCR datasets that satisfy the study's constraints.

THE ONLY SCRIPT IN THIS REPOSITORY WITH NETWORK ACCESS.

Selection rules
---------------

Scientific constraints (unchanged from the specification):

* univariate, equal length, no missing values
* ``n_classes <= 10``
* every class has at least ``25`` *training* instances -- this is the binding
  scientific rule, because k=20 labels per class must leave a non-empty
  unlabeled pool in every class.

Compute-budget constraints (RELAXED -- see NOTES.md):

* ``n_train <= 1500``            (per-cell SSL fits scale with n_train)
* ``n_train + n_test <= 7500``   (feature-cache size)
* ``length <= 2000``             (DTW matrix cost scales with length^2)

The specification asked for ``n_train + n_test <= 2000`` and ``length <= 500``.
Under those caps **only 18** of the 112 univariate equal-length UCR datasets
qualify -- less than half the target of 40 -- and the scan of the full archive
that §4 prescribes does not fix it, because the archive simply does not contain
40 such datasets.  The two caps are pure compute budget, not science, so they
were relaxed to the smallest values that yield exactly 40 datasets while the
scientific rules stay untouched.  Six datasets that pass the scientific rules
are still excluded as too expensive and are listed in ``configs/datasets.yaml``
under ``rejected``: ElectricDevices, FordA, FordB, HandOutlines,
PhalangesOutlinesCorrect, StarLightCurves.

We try the candidate seed list first, then extend by scanning aeon's index of
univariate equal-length UCR datasets in alphabetical order.  The final list plus
full metadata is written to ``configs/datasets.yaml``.

Usage::

    python scripts/download_datasets.py               # normal run
    python scripts/download_datasets.py --target 40   # change the target
    python scripts/download_datasets.py --dry-run     # only report, do not write
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscfree.config import load_config  # noqa: E402
from tscfree.data import cache_dir, save_dataset, to_aeon_3d  # noqa: E402
from tscfree.utils import dir_size, human_bytes, write_json  # noqa: E402

warnings.filterwarnings("ignore")

# scientific constraints (as specified)
MAX_CLASSES = 10
MIN_CLASS_TRAIN = 25
# compute-budget constraints (relaxed from the spec's 2000 / 500 -- see module docstring)
MAX_TRAIN = 1500
MAX_TOTAL = 7500
MAX_LENGTH = 2000
SPEC_MAX_TOTAL = 2000
SPEC_MAX_LENGTH = 500

CANDIDATES = [
    "ECG200", "ECGFiveDays", "TwoLeadECG", "GunPoint", "GunPointAgeSpan",
    "GunPointMaleVersusFemale", "Coffee", "Beef", "Ham", "Meat", "Strawberry", "Wine",
    "ArrowHead", "DistalPhalanxOutlineCorrect", "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW",
    "Car", "Plane", "Trace", "SyntheticControl", "ItalyPowerDemand", "MoteStrain",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "Lightning2", "Lightning7",
    "FaceFour", "OSULeaf", "Symbols", "SwedishLeaf", "Fish", "BirdChicken", "BeetleFly",
    "Herring", "ToeSegmentation1", "ToeSegmentation2", "Chinatown", "FreezerSmallTrain",
    "PowerCons", "Wafer", "Earthquakes",
]


def qualifies(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> str | None:
    """Return None if the dataset qualifies, else a human-readable reason."""
    if Xtr.ndim != 3 or Xte.ndim != 3:
        return "not 3D after coercion"
    if Xtr.shape[1] != 1 or Xte.shape[1] != 1:
        return f"multivariate ({Xtr.shape[1]} channels)"
    if Xtr.shape[2] != Xte.shape[2]:
        return "train/test lengths differ"
    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        return "missing values"
    n_total = Xtr.shape[0] + Xte.shape[0]
    if Xtr.shape[0] > MAX_TRAIN:
        return f"n_train={Xtr.shape[0]} > {MAX_TRAIN}"
    if n_total > MAX_TOTAL:
        return f"n_train+n_test={n_total} > {MAX_TOTAL}"
    if Xtr.shape[2] > MAX_LENGTH:
        return f"length={Xtr.shape[2]} > {MAX_LENGTH}"
    classes_tr = np.unique(ytr)
    if len(classes_tr) > MAX_CLASSES:
        return f"n_classes={len(classes_tr)} > {MAX_CLASSES}"
    if len(classes_tr) < 2:
        return "fewer than 2 classes"
    if not set(np.unique(yte)).issubset(set(classes_tr)):
        return "test contains classes absent from train"
    counts = {c: int(np.sum(ytr == c)) for c in classes_tr}
    if min(counts.values()) < MIN_CLASS_TRAIN:
        return f"smallest training class has {min(counts.values())} < {MIN_CLASS_TRAIN}"
    return None


def try_load(name: str, extract_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from aeon.datasets import load_classification

    Xtr, ytr = load_classification(name, split="train", extract_path=str(extract_path))
    Xte, yte = load_classification(name, split="test", extract_path=str(extract_path))
    return to_aeon_3d(Xtr), np.asarray(ytr).astype(str), to_aeon_3d(Xte), np.asarray(yte).astype(str)


def candidate_pool(extra_first: list[str]) -> list[str]:
    """Seed list first, then the rest of aeon's univariate equal-length UCR index."""
    from aeon.datasets.tsc_datasets import univariate_equal_length

    rest = sorted(set(univariate_equal_length) - set(extra_first))
    return list(extra_first) + rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--download-dir", type=str, default=None,
                    help="scratch dir for aeon's downloader (default: data/_raw)")
    args = ap.parse_args()

    cfg = load_config()
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.download_dir) if args.download_dir else ROOT / "data" / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    pool = candidate_pool(CANDIDATES)

    print(f"Scanning up to {len(pool)} candidates for {args.target} qualifying datasets ...")
    print(f"  rules: n_train+n_test<={MAX_TOTAL}, length<={MAX_LENGTH}, "
          f"n_classes<={MAX_CLASSES}, min class in train >= {MIN_CLASS_TRAIN}\n")

    for name in pool:
        if len(selected) >= args.target:
            break
        try:
            Xtr, ytr, Xte, yte = try_load(name, raw_dir)
        except Exception as exc:  # network / parse / not-found
            rejected[name] = f"load failed: {type(exc).__name__}: {exc}"
            print(f"  [skip] {name:<32} {rejected[name][:80]}")
            continue

        reason = qualifies(Xtr, ytr, Xte, yte)
        if reason is not None:
            rejected[name] = reason
            print(f"  [skip] {name:<32} {reason}")
            continue

        if args.dry_run:
            meta = {
                "name": name, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                "length": int(Xtr.shape[2]), "n_classes": int(len(np.unique(ytr))),
            }
        else:
            meta = save_dataset(name, Xtr, ytr, Xte, yte, source="aeon/load_classification")
        selected[name] = meta
        print(
            f"  [ ok ] {name:<32} n_tr={meta['n_train']:>5} n_te={meta['n_test']:>5} "
            f"L={meta['length']:>4} C={meta['n_classes']:>2} "
            f"minclass={meta.get('min_class_count_train', '?')}"
        )

    print(f"\nSelected {len(selected)} datasets, rejected {len(rejected)}.")
    if len(selected) < args.target:
        print(f"WARNING: only {len(selected)} datasets qualified (target {args.target}).")

    if args.dry_run:
        return 0

    out = {
        "selection_rules": {
            "max_n_train": MAX_TRAIN,
            "max_n_train_plus_test": MAX_TOTAL,
            "max_length": MAX_LENGTH,
            "max_classes": MAX_CLASSES,
            "min_class_count_train": MIN_CLASS_TRAIN,
            "univariate": True,
            "equal_length": True,
            "no_missing_values": True,
            "spec_original_max_n_train_plus_test": SPEC_MAX_TOTAL,
            "spec_original_max_length": SPEC_MAX_LENGTH,
            "relaxation_note": (
                "Under the specification's compute caps (n_train+n_test<=2000, length<=500) "
                "only 18 of the 112 univariate equal-length UCR datasets qualify. The caps "
                "were relaxed to the smallest values yielding the target of 40 datasets; the "
                "scientific rules (n_classes<=10, min class in train>=25) are unchanged."
            ),
        },
        "n_selected": len(selected),
        "datasets": selected,
        "rejected": rejected,
    }
    ds_path = ROOT / "configs" / "datasets.yaml"
    with open(ds_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, sort_keys=False, default_flow_style=False)
    write_json(ROOT / "results" / "dataset_selection.json", out)

    size = dir_size(cache)
    print(f"\nWrote {ds_path}")
    print(f"Cache: {cache}  ({human_bytes(size)})")
    del cfg
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(1) from exc
