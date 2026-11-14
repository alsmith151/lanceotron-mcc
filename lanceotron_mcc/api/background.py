"""Distance-aware background estimation and array-only peak operations."""

from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression


def regions(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1]).astype(np.int64)
    return edges.reshape(-1, 2)


def moving_average(values, window, valid=None):
    """Centered means with truncated, valid-base denominators at boundaries."""
    values = np.asarray(values)
    if window <= 0:
        raise ValueError("window must be positive")
    n = len(values)
    if not n:
        return np.empty(0, dtype=np.float64)
    if window >= n:
        mean = (
            np.mean(values, dtype=np.float64)
            if valid is None
            else (
                np.mean(values[np.asarray(valid, bool)], dtype=np.float64) if np.any(valid) else 0.0
            )
        )
        return np.full(n, mean)
    window = int(window)
    indices = np.arange(n, dtype=np.int64)
    lo = np.maximum(0, indices - window // 2)
    hi = np.minimum(n, indices + (window - window // 2))
    if valid is None:
        sums = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
        return (sums[hi] - sums[lo]) / (hi - lo)
    valid = np.asarray(valid, dtype=bool)
    sums = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0), dtype=np.float64)]
    counts = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    denom = counts[hi] - counts[lo]
    return np.divide(sums[hi] - sums[lo], denom, out=np.zeros(n), where=denom > 0)


def interval_mask(length, intervals, offset=0):
    valid = np.ones(length, dtype=bool)
    for start, end in intervals:
        lo, hi = max(0, start - offset), min(length, end - offset)
        if lo < hi:
            valid[lo:hi] = False
    return valid


def distance_geometry(positions, viewpoints):
    """Distance to closest oligo boundary; side is relative to that oligo."""
    positions = np.asarray(positions)
    distance = np.full(positions.shape, np.inf)
    side = np.zeros(positions.shape, dtype=np.int8)
    for start, end in sorted(viewpoints):
        current = np.maximum(np.maximum(start - positions, positions - end), 0.0)
        better = current < distance
        side[better] = np.where(positions[better] < start, -1, 1)
        distance[better] = current[better]
    return distance, side


def fit_distance_background(
    signal,
    positions,
    viewpoints,
    eligible=None,
    weights=None,
    iterations=3,
    bins_per_decade=12,
    count_mode=False,
):
    """Robust monotone mean curve fitted independently on each viewpoint arm.

    Log-distance bands pool sparse bins; iterative masking limits domination by
    strong peaks. This is a heuristic expected signal, not a calibrated null.
    """
    y = np.asarray(signal, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if (
        y.ndim != 1
        or positions.shape != y.shape
        or np.any(~np.isfinite(y))
        or np.any(y < 0)
        or np.any(~np.isfinite(positions))
    ):
        raise ValueError("Background requires finite nonnegative signal and viewpoint intervals")
    if not viewpoints:
        raise ValueError("Background requires finite nonnegative signal and viewpoint intervals")
    for interval in viewpoints:
        try:
            interval_size = len(interval)
        except TypeError:
            interval_size = -1
        if interval_size != 2:
            raise ValueError("Viewpoint intervals must be (start, end) pairs")
        start, end = interval
        try:
            finite = np.isfinite(start) and np.isfinite(end)
        except TypeError as exc:
            raise ValueError("Viewpoint intervals must have finite coordinates") from exc
        if not finite or end <= start:
            raise ValueError("Viewpoint intervals must have finite positive width")
    valid = np.ones(y.shape, dtype=bool) if eligible is None else np.asarray(eligible, bool)
    weights = np.ones(y.shape) if weights is None else np.asarray(weights, dtype=float)
    if valid.shape != y.shape:
        raise ValueError("eligible must have the same shape as signal")
    if weights.shape != y.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must match signal and be finite and nonnegative")
    valid = valid & (weights > 0)
    if not np.any(valid):
        raise ValueError("No eligible distance-background observations")
    distance, side = distance_geometry(positions, viewpoints)
    log_distance = np.log10(distance + 1.0)
    bands = np.floor(log_distance * bins_per_decade).astype(np.int64)
    result = np.zeros_like(y)
    for arm in (-1, 1):
        arm_positions = side == arm
        arm_valid = valid & arm_positions
        if np.any(arm_positions) and not np.any(arm_valid):
            raise ValueError(f"No eligible distance-background observations on viewpoint arm {arm}")
        if not np.any(arm_positions):
            continue
        keep = arm_valid.copy()
        positive_signal = y[arm_valid & (y > 0)]
        signal_scale = float(np.median(positive_signal)) if positive_signal.size else 0.0
        fitted = np.zeros_like(y)
        for _ in range(iterations):
            if not np.any(keep):
                break
            band_weights = np.bincount(bands[keep], weights=weights[keep])
            ids = np.flatnonzero(band_weights > 0)
            sizes = band_weights[ids]
            x = np.bincount(bands[keep], weights=log_distance[keep] * weights[keep])[ids] / sizes
            means = np.bincount(bands[keep], weights=y[keep] * weights[keep])[ids] / sizes
            model = isotonic_regression(means, weights=sizes, increasing=False)
            fitted = np.interp(log_distance, x, model.x)
            # Work in event units for counts; a single sparse reporter must not
            # be removed solely because its fold change over a tiny mean is high.
            # Continuous-signal guard scales with the track's units.
            guard = (
                4.0 * np.sqrt(fitted * weights + 1.0) / np.maximum(weights, 1.0)
                if count_mode
                else 4.0 * signal_scale
            )
            cut = (y > 4.0 * fitted) & (y > fitted + guard)
            next_keep = arm_valid & ~cut
            if np.array_equal(keep, next_keep) or not np.any(next_keep):
                break
            keep = next_keep
        result[side == arm] = fitted[side == arm]
    # A small scale-relative floor prevents undefined fold change at sparse tails.
    positive = result[valid & (result > 0)]
    if not len(positive):
        positive = y[valid & (y > 0)]
    floor = (
        max(np.finfo(float).eps, float(np.median(positive)) * 1e-3)
        if len(positive)
        else np.finfo(float).eps
    )
    return np.maximum(result, floor)


def refine_regions(votes, initial, limit, min_size, max_size, chunk_size=1_000_000):
    """Refine broad consensus runs without losing them before threshold escalation."""

    def scan(start, end, threshold):
        pending = None
        for lo in range(start, end, chunk_size):
            hi = min(end, lo + chunk_size)
            for a, b in regions(np.asarray(votes[lo:hi]) >= threshold):
                a, b = int(a + lo), int(b + lo)
                if pending is not None and a == pending[1]:
                    pending = (pending[0], b)
                else:
                    if pending is not None:
                        yield pending
                    pending = (a, b)
            if pending is not None and pending[1] < hi:
                yield pending
                pending = None
        if pending is not None:
            yield pending

    queue = [(a, b, initial) for a, b in scan(0, len(votes), initial)]
    solved = []
    while queue:
        start, end, threshold = queue.pop()
        if end - start < min_size:
            continue
        if end - start <= max_size:
            solved.append((start, end, False))
            continue
        if threshold < limit:
            children = [(a, b) for a, b in scan(start, end, threshold + 1) if b - a >= min_size]
            if children:
                queue.extend((a, b, threshold + 1) for a, b in children)
                continue
        # Flat broad plateaus cannot be split defensibly; retain as unresolved.
        solved.append((start, end, True))
    return sorted(solved)


def rolling_prefix(values, valid=None):
    """Build shared float64 sums and optional valid-base counts once per chunk."""
    values = np.asarray(values)
    if valid is not None and np.all(valid):
        valid = None
    sums = np.empty(len(values) + 1, dtype=np.float64)
    sums[0] = 0.0
    np.cumsum(
        values if valid is None else np.where(valid, values, 0.0), out=sums[1:], dtype=np.float64
    )
    if valid is None:
        return sums, None
    counts = np.empty(len(values) + 1, dtype=np.int64)
    counts[0] = 0
    np.cumsum(valid, out=counts[1:], dtype=np.int64)
    return sums, counts


def average_from_prefix(sums, window, counts=None):
    """Evaluate a centered rolling mean without chromosome-sized index arrays."""
    n = len(sums) - 1
    if window <= 0:
        raise ValueError("window must be positive")
    if n == 0:
        return np.empty(0, dtype=float)
    if window >= n:
        denominator = n if counts is None else counts[-1]
        return np.full(n, sums[-1] / denominator if denominator else 0.0)
    left = window // 2
    right = window - left - 1
    result = np.zeros(n, dtype=np.float64)
    middle = slice(left, n - right)
    np.subtract(sums[window:], sums[:-window], out=result[middle])
    if counts is None:
        result[middle] /= window
        if left:
            result[:left] = sums[right + 1 : window] / np.arange(right + 1, window)
        if right:
            result[-right:] = (sums[-1] - sums[n - right - left : n - left]) / np.arange(
                window - 1, left, -1
            )
    else:
        denominator = counts[window:] - counts[:-window]
        np.divide(result[middle], denominator, out=result[middle], where=denominator > 0)
        if left:
            denominator = counts[right + 1 : window]
            np.divide(
                sums[right + 1 : window], denominator, out=result[:left], where=denominator > 0
            )
        if right:
            denominator = counts[-1] - counts[n - right - left : n - left]
            np.divide(
                sums[-1] - sums[n - right - left : n - left],
                denominator,
                out=result[-right:],
                where=denominator > 0,
            )
    return result
