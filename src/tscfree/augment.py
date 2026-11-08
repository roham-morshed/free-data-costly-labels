"""Time-domain augmentations for the ``augment`` baseline.

This baseline is the honest competitor to semi-supervised learning: instead of
exploiting *real* unlabeled series, it manufactures more labeled ones.  The four
transforms are the standard set from the time-series literature (Le Guennec et
al. 2016; Um et al. 2017):

``jitter``        additive Gaussian noise
``scaling``       multiply the whole series by a random gain
``time_warp``     smoothly stretch/compress the time axis (cubic-spline warp)
``window_slice``  crop a random contiguous window and resample to full length

Augmentation is applied **only to labeled series** -- it never touches ``U`` or
the test set.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def jitter(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return X + rng.normal(loc=0.0, scale=sigma, size=X.shape)


def scaling(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """One random gain per (series, channel)."""
    factor = rng.normal(loc=1.0, scale=sigma, size=(X.shape[0], X.shape[1], 1))
    return X * factor


def time_warp(
    X: np.ndarray, n_knots: int, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Smooth random monotone warp of the time axis, then resample to length L."""
    n, c, length = X.shape
    t = np.arange(length, dtype=float)
    knot_x = np.linspace(0, length - 1, n_knots + 2)
    out = np.empty_like(X)
    for i in range(n):
        knot_y = knot_x * rng.normal(loc=1.0, scale=sigma, size=knot_x.shape)
        knot_y = np.sort(knot_y)
        knot_y = knot_y - knot_y[0]
        span = knot_y[-1]
        if span <= 1e-9:
            out[i] = X[i]
            continue
        knot_y = knot_y / span * (length - 1)
        cs = CubicSpline(knot_x, knot_y)
        warped = np.clip(cs(t), 0, length - 1)
        # enforce monotonicity so the warp stays a valid reparameterisation
        warped = np.maximum.accumulate(warped)
        for ch in range(c):
            out[i, ch] = np.interp(warped, t, X[i, ch])
    return out


def window_slice(X: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Crop a contiguous window covering ``ratio`` of the series, resample to L."""
    n, c, length = X.shape
    win = max(2, int(round(ratio * length)))
    if win >= length:
        return X.copy()
    out = np.empty_like(X)
    src = np.linspace(0, win - 1, length)
    for i in range(n):
        start = int(rng.integers(0, length - win + 1))
        seg = X[i, :, start : start + win]
        for ch in range(c):
            out[i, ch] = np.interp(src, np.arange(win, dtype=float), seg[ch])
    return out


AUGMENTATIONS = ("jitter", "scaling", "time_warp", "window_slice")


def augment_labeled(
    X: np.ndarray,
    y: np.ndarray,
    n_copies: int,
    rng: np.random.Generator,
    jitter_sigma: float = 0.05,
    scaling_sigma: float = 0.1,
    warp_n_knots: int = 4,
    warp_sigma: float = 0.2,
    slice_ratio: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_aug, y_aug)`` = the originals plus ``n_copies`` augmented sets.

    Each copy applies one augmentation, cycling through the four types so the
    mix is balanced and deterministic given ``rng``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    parts_X = [X]
    parts_y = [y]
    for i in range(int(n_copies)):
        kind = AUGMENTATIONS[i % len(AUGMENTATIONS)]
        if kind == "jitter":
            Xa = jitter(X, jitter_sigma, rng)
        elif kind == "scaling":
            Xa = scaling(X, scaling_sigma, rng)
        elif kind == "time_warp":
            Xa = time_warp(X, warp_n_knots, warp_sigma, rng)
        else:
            Xa = window_slice(X, slice_ratio, rng)
        parts_X.append(Xa)
        parts_y.append(y)
    return (
        np.ascontiguousarray(np.concatenate(parts_X, axis=0)),
        np.concatenate(parts_y, axis=0),
    )
