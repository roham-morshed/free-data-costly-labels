"""Small, dependency-light helpers: seeding, timing, hashing, IO, label guard."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    """Seed every RNG we can reach.

    ``PYTHONHASHSEED`` only takes effect for *child* processes started after
    this call; the joblib workers spawned by :mod:`tscfree.experiment` inherit
    it, which is why we set it here rather than relying on the shell.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is an optional dependency
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def rng_for(*parts: Any) -> np.random.Generator:
    """A numpy Generator deterministically derived from arbitrary keys.

    Using a hash of the *semantic* cell key (dataset, method, k, seed, ...)
    rather than a global counter means a cell produces the same numbers no
    matter what order the grid is executed in, which is what makes the run
    both resumable and reproducible.
    """
    key = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def seed_from(*parts: Any) -> int:
    """A deterministic 31-bit integer seed derived from arbitrary keys.

    Needed where an API wants an ``int`` random_state rather than a Generator.
    """
    key = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


# ---------------------------------------------------------------------------
# Environment / provenance
# ---------------------------------------------------------------------------


def package_versions() -> dict[str, str]:
    """Version string of every package that can change a number in the paper."""
    out: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("numpy", "scipy", "sklearn", "aeon", "pandas", "numba", "joblib"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "missing"
    return out


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timer:
    """Context manager: ``with Timer() as t: ...`` then read ``t.seconds``."""

    def __init__(self) -> None:
        self.seconds: float = float("nan")
        self._t0: float = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Hashing / IO
# ---------------------------------------------------------------------------


def sha256_array(a: np.ndarray) -> str:
    """Stable content hash of an array (shape + dtype + bytes)."""
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable content hash of a JSON-serialisable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: str | Path, obj: Any) -> None:
    """Atomically write JSON (temp file + os.replace).

    A crash mid-write must never leave a truncated result file, because the
    resume logic treats "file exists" as "cell is done".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_jsonable)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_npy(path: str | Path, a: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    # write through a file handle: np.save(path_str, ...) silently appends
    # ".npy" to any name that lacks it, which would defeat the atomic rename.
    with open(tmp, "wb") as fh:
        np.save(fh, a, allow_pickle=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Label leakage guard  (see tests/test_no_label_leakage.py)
# ---------------------------------------------------------------------------


class LabelLeakageError(AssertionError):
    """Raised when code touches a label it is not allowed to see."""


class GuardedLabels:
    """A read-guarded view of the *full* training label vector.

    The protocol says the labels of the unlabeled pool ``U`` do not exist.
    Rather than trusting ourselves never to index them, methods receive this
    object: reading any position in ``U`` raises immediately, so a leak is a
    hard crash in the test suite instead of an inflated number in a table.

    ``reveal_for_evaluation`` is the single sanctioned escape hatch and is used
    only by diagnostic/oracle code that is reported as such -- never by a
    method.
    """

    __slots__ = ("_y", "_allowed", "_n", "n_touch")

    def __init__(self, y: np.ndarray, allowed_idx: Iterable[int]) -> None:
        self._y = np.asarray(y)
        self._n = int(self._y.shape[0])
        self._allowed = np.zeros(self._n, dtype=bool)
        allowed = np.asarray(list(allowed_idx), dtype=int)
        if allowed.size:
            self._allowed[allowed] = True
        self.n_touch = 0

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[Any]:
        for i in range(self._n):
            yield self[i]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        raise LabelLeakageError(
            "np.asarray() on GuardedLabels would expose the unlabeled pool's labels."
        )

    def __getitem__(self, idx: Any) -> Any:
        self.n_touch += 1
        probe = np.zeros(self._n, dtype=bool)
        probe[idx] = True
        forbidden = probe & (~self._allowed)
        if forbidden.any():
            bad = np.flatnonzero(forbidden)[:5].tolist()
            raise LabelLeakageError(
                f"Attempted to read label(s) at unlabeled-pool index/indices {bad} "
                "(and possibly more). These labels do not exist in this protocol."
            )
        return self._y[idx]

    # -- sanctioned escape hatch -------------------------------------------
    def reveal_for_evaluation(self) -> np.ndarray:
        """Return the hidden labels.  Diagnostics only, never inside a method."""
        return self._y.copy()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellKey:
    """Identity of one experiment cell."""

    dataset: str
    k: int
    seed: int

    def filename(self) -> str:
        return f"{self.dataset}__k{self.k}__s{self.seed}.json"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def dir_size(path: str | Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, f))
    return total
