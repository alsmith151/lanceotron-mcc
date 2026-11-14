"""Opt-in distance-aware BigWig and experimental reporter-count calling."""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.signal import find_peaks

from .background import (
    average_from_prefix,
    fit_distance_background,
    interval_mask,
    moving_average,
    refine_regions,
    rolling_prefix,
)
from .io import ArrayCache, BigWigReader, cache_key, peak_memory_mb, source_identity

# Valley splitting is a statistical refinement with a fixed input bound. It
# must not change when callers tune chunking or memory for execution.
_VALLEY_SPLIT_CAP_BP = 16_384


class AdaptiveParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_sizes: list[int] = Field(default_factory=lambda: [200, 400, 600])
    backgrounds: list[int] = Field(default_factory=lambda: [10_000, 100_000, 1_000_000])
    thresholds: list[float] = Field(default_factory=lambda: [2.0, 4.0, 6.0])
    support_fraction: float = 1 / 3
    min_peak_size: int = 50
    max_peak_size: int = 500
    bin_size: int = 50
    background_bin_size: int = 200
    min_read_count: int = 5
    alpha: float = 0.05
    bait_padding: int = 0
    chunk_size: int = 1_000_000
    memory_budget_mb: float = 512
    n_jobs: int = 1

    @model_validator(mode="after")
    def validate_parameters(self):
        for name in ("window_sizes", "backgrounds", "thresholds"):
            values = getattr(self, name)
            if not values or any(not np.isfinite(x) or x <= 0 for x in values):
                raise ValueError(f"{name} must contain positive finite values")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if self.min_peak_size <= 0 or self.max_peak_size < self.min_peak_size:
            raise ValueError("Require 0 < min_peak_size <= max_peak_size")
        if not 0 < self.support_fraction <= 1 or not 0 < self.alpha < 1:
            raise ValueError("support_fraction must be in (0,1], alpha in (0,1)")
        if (
            min(self.bin_size, self.background_bin_size, self.chunk_size, self.n_jobs) <= 0
            or self.min_read_count < 0
            or self.bait_padding < 0
        ):
            raise ValueError("Sizes/jobs must be positive; padding/read threshold nonnegative")
        if not np.isfinite(self.memory_budget_mb) or self.memory_budget_mb < 16:
            raise ValueError("memory_budget_mb must be at least 16")
        return self


def read_bed(
    path: str | Path,
    group: str | None = None,
    allow_empty: bool = False,
    require_single_group: bool = False,
) -> dict[str, list[tuple[int, int]]]:
    """Load zero-based BED intervals, optionally selecting one named group."""
    try:
        frame = pd.read_csv(path, sep="\t", comment="#", header=None, keep_default_na=False)
    except pd.errors.EmptyDataError:
        if allow_empty:
            return {}
        raise ValueError("BED selection is empty") from None
    if frame.shape[1] < 3:
        raise ValueError("BED must have at least three columns")
    if group is not None:
        if frame.shape[1] < 4:
            raise ValueError("Viewpoint group selection requires BED names")
        names = frame[3].astype(str)
        stripped = names.str.replace(r"-chr[^-]+-\d+-\d+$", "", regex=True)
        frame = frame[(names == group) | (stripped == group)]
    elif require_single_group and frame.shape[1] >= 4:
        names = frame[3].astype(str)
        stripped = names.str.replace(r"-chr[^-]+-\d+-\d+$", "", regex=True)
        if stripped.nunique() > 1:
            raise ValueError("BED contains multiple viewpoint groups; select one by name")
    if frame.empty:
        if allow_empty:
            return {}
        raise ValueError("BED selection is empty")
    result = {}
    for row in frame.itertuples(index=False, name=None):
        start, end = int(row[1]), int(row[2])
        if float(row[1]) != start or float(row[2]) != end or start < 0 or end <= start:
            raise ValueError("BED intervals must be positive-width zero-based integers")
        result.setdefault(str(row[0]), []).append((start, end))
    return {chrom: sorted(intervals) for chrom, intervals in result.items()}


def _viewpoint(viewpoints, chroms):
    if len(viewpoints) != 1:
        raise ValueError("One viewpoint group on one chromosome is required per invocation")
    chrom = next(iter(viewpoints))
    if (
        chrom not in chroms
        or not viewpoints[chrom]
        or any(
            start < 0
            or end <= start
            or end > chroms[chrom]
            or int(start) != start
            or int(end) != end
            for start, end in viewpoints[chrom]
        )
    ):
        raise ValueError("Viewpoint is outside input chromosome bounds")
    return chrom, viewpoints[chrom]


def _validate_exclusions(exclusions, chrom_sizes):
    """Validate direct-API exclusions and return normalized half-open intervals."""
    if exclusions is None:
        return {}
    if not isinstance(exclusions, Mapping):
        raise ValueError("Exclusions must map chromosome names to intervals")
    normalized = {}
    for chrom, intervals in exclusions.items():
        if not isinstance(chrom, str) or not chrom or chrom not in chrom_sizes:
            raise ValueError(f"Exclusion chromosome {chrom!r} is absent from the input")
        size = chrom_sizes[chrom]
        try:
            size = int(size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid chromosome size for {chrom!r}") from exc
        if isinstance(intervals, (str, bytes)):
            raise ValueError(f"Exclusions for {chrom!r} must be interval pairs")
        try:
            intervals = list(intervals)
        except TypeError as exc:
            raise ValueError(f"Exclusions for {chrom!r} must be interval pairs") from exc
        result = []
        for interval in intervals:
            try:
                interval_size = len(interval)
            except TypeError:
                interval_size = -1
            if isinstance(interval, (str, bytes)) or interval_size != 2:
                raise ValueError("Each exclusion must be a (start, end) pair")
            start, end = interval
            try:
                finite = np.isfinite(start) and np.isfinite(end)
                integer = int(start) == start and int(end) == end
            except (TypeError, ValueError, OverflowError):
                finite, integer = False, False
            if (
                isinstance(start, (bool, np.bool_))
                or isinstance(end, (bool, np.bool_))
                or not finite
                or not integer
            ):
                raise ValueError("Exclusion coordinates must be finite integers")
            start, end = int(start), int(end)
            if start < 0 or end <= start or end > size:
                raise ValueError(
                    f"Exclusion interval ({start}, {end}) is outside {chrom} bounds 0..{size}"
                )
            result.append((start, end))
        normalized[chrom] = sorted(result)
    return normalized


def _coarse(values, bin_size):
    starts = np.arange(0, len(values), bin_size, dtype=np.int64)
    widths = np.minimum(bin_size, len(values) - starts)
    means = np.add.reduceat(values, starts, dtype=np.float64) / widths
    return means, starts + widths / 2, widths


def _materialize(reader, chrom, tmp, name, params, sample_count):
    n = reader.chroms[chrom]
    if n * 4 <= params.memory_budget_mb * 1024**2 / (4 * sample_count):
        values = reader.read(chrom)
        if np.any(values < 0):
            raise ValueError(
                "Peak calling requires nonnegative signal; difference tracks are unsupported"
            )
        return values
    values = np.lib.format.open_memmap(
        Path(tmp) / f"{name}.npy", mode="w+", dtype=np.float32, shape=(n,)
    )
    for lo in range(0, n, params.chunk_size):
        hi = min(n, lo + params.chunk_size)
        part = reader.read(chrom, lo, hi)
        if np.any(part < 0):
            raise ValueError("Peak calling requires nonnegative signal")
        values[lo:hi] = part
    values.flush()
    return values


def _signal_depth(values, exclusions, chunk_size):
    total = 0.0
    for lo in range(0, len(values), chunk_size):
        hi = min(len(values), lo + chunk_size)
        valid = interval_mask(hi - lo, exclusions, offset=lo)
        total += np.sum(values[lo:hi][valid], dtype=np.float64)
    return total


def _profile_bin_size(chrom_length, params, unit=1):
    # Explicit statistical resolution, independent of performance settings.
    width = max(unit, math.ceil(params.background_bin_size / unit) * unit)
    fit_mb = math.ceil(chrom_length / width) * 128 / 1024**2
    if fit_mb > params.memory_budget_mb * 0.5:
        raise ValueError(
            f"Distance fitting needs approximately {math.ceil(fit_mb * 2)} MiB; increase memory_budget_mb or explicitly coarsen background_bin_size"
        )
    return width


def _masked_bins(n, bin_size, exclusions):
    count = math.ceil(n / bin_size)
    eligible = np.ones(count, dtype=bool)
    for start, end in exclusions:
        lo = max(0, start // bin_size)
        hi = min(count, math.ceil(end / bin_size))
        eligible[lo:hi] = False
    return eligible


def _split_valleys(calls, mean_slice, params, unit):
    result = []
    for start, end, unresolved in calls:
        if not unresolved:
            result.append((start, end, False))
            continue
        # This fixed cap bounds temporary arrays without coupling biological
        # decisions to chunk_size or memory_budget_mb.
        max_points = max(1, _VALLEY_SPLIT_CAP_BP // unit)
        if end - start > max_points:
            result.append((start, end, True))
            continue
        # Only a substantial internal signal valley justifies an extra split.
        values = moving_average(mean_slice(start, end), max(1, min(params.window_sizes) // unit))
        if len(values) < 3 or values.max() <= 0:
            result.append((start, end, True))
            continue
        valleys, _ = find_peaks(
            -values,
            prominence=0.25 * values.max(),
            distance=max(1, math.ceil(params.min_peak_size / unit)),
        )
        cuts = [start]
        for idx in valleys:
            cut = start + int(idx)
            if (cut - cuts[-1]) * unit >= params.min_peak_size and (
                end - cut
            ) * unit >= params.min_peak_size:
                cuts.append(cut)
        cuts.append(end)
        result.extend(
            (a, b, (b - a) * unit > params.max_peak_size) for a, b in zip(cuts[:-1], cuts[1:])
        )
    return result


def _call_arrays(
    arrays,
    expected,
    exclusions,
    chrom,
    chrom_length,
    unit,
    params,
    tmp,
    identities,
    cache,
    sample_names,
    eligible_bins=None,
    qvalues=None,
    profile_bin_size=None,
):
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("Replicate chromosomes must have matching lengths")
    if len(expected) != len(arrays):
        raise ValueError("Each replicate must have one distance-background profile")
    expected_bins = math.ceil(chrom_length / (profile_bin_size or params.bin_size))
    for background in expected:
        background = np.asarray(background)
        if (
            background.shape != (expected_bins,)
            or np.any(~np.isfinite(background))
            or np.any(background < 0)
        ):
            raise ValueError(
                "Distance-background profiles must match the chromosome and be finite and nonnegative"
            )

    def mean_slice(start, end):
        result = np.zeros(end - start, dtype=np.float64)
        for array in arrays:
            result += array[start:end]
        return result / len(arrays)

    grid_size = len(params.window_sizes) * len(params.backgrounds) * len(params.thresholds)
    votes = np.lib.format.open_memmap(
        Path(tmp) / "votes.npy", mode="w+", dtype=np.int32, shape=(n,)
    )
    chunk = max(1, params.chunk_size // unit)
    windows = sorted(set(max(1, math.ceil(w / unit)) for w in params.window_sizes))
    backgrounds = [max(1, math.ceil(w / unit)) for w in params.backgrounds]
    halo = min(n, max([*windows, *backgrounds]) // 2 + 1)
    profile_bin_size = profile_bin_size or params.bin_size
    coarse_starts = np.arange(len(expected[0]), dtype=np.int64) * profile_bin_size
    expected_positions = (
        coarse_starts + np.minimum(profile_bin_size, chrom_length - coarse_starts) / 2
    )
    base_expected = np.mean(expected, axis=0)
    exclusion_units = [(start // unit, math.ceil(end / unit)) for start, end in exclusions]
    # Reserve half for caching; estimate remaining live smoothing/index buffers.
    bytes_per_position = 96 + 8 * len(windows)
    max_span = int(params.memory_budget_mb * 1024**2 * 0.45 / bytes_per_position)
    if n > max_span and max_span <= 2 * halo:
        required = math.ceil((min(n, 2 * halo + 1)) * bytes_per_position / (0.45 * 1024**2))
        raise ValueError(
            f"Background windows need memory_budget_mb >= {required}; increase budget or shorten backgrounds"
        )
    chunk = min(chunk, max(1, max_span - 2 * halo)) if n > max_span else chunk
    # Calculate valid-base global means without materializing scaled chromosomes.
    global_sums = np.zeros(len(arrays))
    global_count = 0
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        valid = interval_mask(hi - lo, exclusion_units, offset=lo)
        global_count += int(valid.sum())
        for i, array in enumerate(arrays):
            values = np.asarray(array[lo:hi], dtype=np.float64)
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Replicate signal must contain finite nonnegative values")
            global_sums[i] += np.sum(values[valid], dtype=np.float64)
    global_mean = float(global_sums.sum() / (max(1, global_count) * len(arrays)))
    with ThreadPoolExecutor(max_workers=min(params.n_jobs, len(windows))) as pool:
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            # Background alone cannot create a peak. On sparse MCC tracks,
            # skip chunks with no signal within any foreground window.
            probe_lo = max(0, lo - max(windows) // 2 - 1)
            probe_hi = min(n, hi + max(windows) // 2 + 1)
            has_signal = (
                global_mean > 0
                if max(windows) >= n
                else any(
                    np.any(
                        (array.array if isinstance(array, _ScaledArray) else array)[
                            probe_lo:probe_hi
                        ]
                        > 0
                    )
                    for array in arrays
                )
            )
            if not has_signal:
                votes[lo:hi] = 0
                continue
            a, b = max(0, lo - halo), min(n, hi + halo)
            valid = interval_mask(b - a, exclusion_units, offset=a)
            signal = mean_slice(a, b)
            positions = (
                np.arange(a, b) * unit + np.minimum(unit, chrom_length - np.arange(a, b) * unit) / 2
            )
            distance_bg = np.interp(positions, expected_positions, base_expected)
            prefix, valid_counts = rolling_prefix(signal, valid)

            def smooth(window):
                key = cache_key(identities, "smooth", a, b, window, unit, exclusions, chrom_length)
                value = cache.get(key)
                if value is None:
                    value = (
                        np.full(b - a, global_mean)
                        if window >= n
                        else average_from_prefix(prefix, window, valid_counts)
                    )
                    value = cache.put(key, value)
                return value

            # Map avoids repeated smoothing when multiple thresholds share a scale.
            smoothed = dict(zip(windows, pool.map(smooth, windows)))
            local_votes = np.zeros(hi - lo, dtype=np.int32)
            core = slice(lo - a, hi - a)
            # Distinct bp windows may collapse to one count-bin window; retain
            # their multiplicity in the consensus grid for transparent semantics.
            for background in backgrounds:
                baseline = np.maximum(distance_bg, smooth(background))[core]
                for window_bp in params.window_sizes:
                    observed = smoothed[max(1, math.ceil(window_bp / unit))][core]
                    for threshold in params.thresholds:
                        local_votes += (
                            (observed > 0) & (observed >= threshold * baseline) & valid[core]
                        )
            if eligible_bins is not None:
                local_votes[~eligible_bins[lo:hi]] = 0
            votes[lo:hi] = local_votes
    initial = max(1, math.ceil(params.support_fraction * grid_size - 1e-10))
    calls = refine_regions(
        votes,
        initial,
        grid_size,
        math.ceil(params.min_peak_size / unit),
        max(1, params.max_peak_size // unit),
        chunk,
    )
    calls = _split_valleys(calls, mean_slice, params, unit)
    rows = []
    for start, end, unresolved in calls:
        start_bp, end_bp = start * unit, min(chrom_length, end * unit)
        if end_bp - start_bp < params.min_peak_size:
            continue
        positions = (
            np.arange(start, end) * unit
            + np.minimum(unit, chrom_length - np.arange(start, end) * unit) / 2
        )
        means, backgrounds_at_peak, sums, enrichments = [], [], [], []
        for arr, bg in zip(arrays, expected):
            observed = np.asarray(arr[start:end])
            exp = np.interp(positions, expected_positions, bg)
            observed_mean = float(observed.mean())
            bg_mean = float(exp.mean())
            means.append(observed_mean)
            sums.append(float(observed.sum(dtype=np.float64)))
            backgrounds_at_peak.append(bg_mean)
            enrichments.append(observed_mean / max(bg_mean, np.finfo(float).tiny))
        support = sum(x >= min(params.thresholds) for x in enrichments)
        required = len(arrays) // 2 + 1 if len(arrays) > 1 else 1
        values = mean_slice(start, end)
        row = {
            "Chromosome": chrom,
            "Start": start_bp,
            "End": end_bp,
            "Length": end_bp - start_bp,
            "Signal": float(np.mean(means)),
            "Sum": float(np.mean(sums)),
            "Maximum": float(values.max()),
            "Summit": min(end_bp - 1, start_bp + int(np.argmax(values)) * unit),
            "Expected": float(np.mean(backgrounds_at_peak)),
            "Enrichment": float(np.mean(enrichments)),
            "Support": float(np.mean(votes[start:end], dtype=np.float64) / grid_size),
            "ReplicateSupport": support,
            "Replicates": len(arrays),
            "Reproducible": support >= required if len(arrays) > 1 else None,
            "Unresolved": unresolved,
        }
        for name, observed, enrichment in zip(sample_names, means, enrichments):
            row[f"Signal:{name}"] = observed
            row[f"Enrichment:{name}"] = enrichment
        if qvalues is not None:
            # This is a descriptive summary of bin tests, not peak-level FDR.
            row["MinimumBinQ"] = float(np.min(qvalues[start:end]))
            row["SignificantBins"] = int(np.sum(qvalues[start:end] <= params.alpha))
        if support >= required:
            rows.append(row)
    columns = [
        "Chromosome",
        "Start",
        "End",
        "Length",
        "Signal",
        "Sum",
        "Maximum",
        "Summit",
        "Expected",
        "Enrichment",
        "Support",
        "ReplicateSupport",
        "Replicates",
        "Reproducible",
        "Unresolved",
    ]
    columns += [
        column for name in sample_names for column in (f"Signal:{name}", f"Enrichment:{name}")
    ]
    if qvalues is not None:
        columns += ["MinimumBinQ", "SignificantBins", "Count"]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
    return result


def call_adaptive_bigwigs(
    paths: Sequence[str | Path],
    viewpoints: Mapping[str, Sequence[tuple[int, int]]],
    exclusions: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    params: AdaptiveParams | None = None,
    scratch_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call distance-aware peaks and return scored intervals plus run metrics."""
    params = params or AdaptiveParams()
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("At least one BigWig is required")
    if len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError("Biological replicates must use distinct input files")
    file_ids = [(p.stat().st_dev, p.stat().st_ino) for p in paths]
    if len(set(file_ids)) != len(file_ids):
        raise ValueError("Biological replicates must not be hard links to the same input file")
    sample_names = [p.stem for p in paths]
    if len(set(sample_names)) != len(paths):
        raise ValueError("Replicate input filenames must have distinct stems")
    cache = ArrayCache(params.memory_budget_mb / 2, cache_dir)
    t0 = time.perf_counter()
    if scratch_dir:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    with (
        ExitStack() as stack,
        tempfile.TemporaryDirectory(dir=scratch_dir, prefix="lanceotron-") as tmp,
    ):
        readers = [
            stack.enter_context(
                BigWigReader(p, params.memory_budget_mb / (4 * len(paths)), scratch_dir, cache_dir)
            )
            for p in paths
        ]
        chrom, vp = _viewpoint(viewpoints, readers[0].chroms)
        n = readers[0].chroms[chrom]
        if any(r.chroms.get(chrom) != n for r in readers):
            raise ValueError("Replicate chromosomes must have matching lengths")
        normalized_exclusions = _validate_exclusions(exclusions, readers[0].chroms)
        excluded = list(normalized_exclusions.get(chrom, [])) + [
            (max(0, a - params.bait_padding), min(n, b + params.bait_padding)) for a, b in vp
        ]
        arrays = [
            _materialize(r, chrom, tmp, f"sample{i}", params, len(readers))
            for i, r in enumerate(readers)
        ]
        profile_bin_size = _profile_bin_size(n, params)
        eligible = _masked_bins(n, profile_bin_size, excluded)
        if not np.any(eligible):
            raise ValueError("No eligible bins outside exclusions/viewpoint")
        # Equal replicate contribution; a single track retains its original units.
        depths = np.asarray([_signal_depth(a, excluded, params.chunk_size) for a in arrays])
        if np.any(depths <= 0):
            raise ValueError("Each replicate must have positive cis signal")
        scales = np.median(depths) / depths if len(arrays) > 1 else np.ones(1)
        # Scale during local slicing rather than allocating another chromosome.
        scaled = [_ScaledArray(a, scale) for a, scale in zip(arrays, scales)]
        expected = []
        identities = [r.identity | {"scale": float(s)} for r, s in zip(readers, scales)]
        for array, identity, scale in zip(arrays, identities, scales):
            key = cache_key(identity, "distance", vp, excluded, profile_bin_size)
            bg = cache.get(key)
            if bg is None:
                signal, positions, widths = _coarse(array, profile_bin_size)
                bg = cache.put(
                    key, fit_distance_background(signal * scale, positions, vp, eligible, widths)
                )
                del signal, positions, widths
            expected.append(bg)
        result = _call_arrays(
            arrays=scaled,
            expected=expected,
            exclusions=excluded,
            chrom=chrom,
            chrom_length=n,
            unit=1,
            params=params,
            tmp=tmp,
            identities=identities,
            cache=cache,
            sample_names=sample_names,
            profile_bin_size=profile_bin_size,
        )
        metrics = {
            "inputs": [{**r.metrics, "cache": dict(r.cache.metrics)} for r in readers],
            "derived_cache": dict(cache.metrics),
            "normalization_factors": scales.tolist(),
            "background_bin_size": profile_bin_size,
            "elapsed_seconds": time.perf_counter() - t0,
            "peak_memory_mb": peak_memory_mb(),
        }
    return result, metrics


class _ScaledArray:
    def __init__(self, array, scale):
        self.array, self.scale = array, scale

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return np.asarray(self.array[item], dtype=np.float64) * self.scale


def call_count_manifest(
    manifest: str | Path | pd.DataFrame,
    viewpoint: str,
    viewpoints: Mapping[str, Sequence[tuple[int, int]]],
    chrom_sizes: Mapping[str, int],
    exclusions: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    params: AdaptiveParams | None = None,
    scratch_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return experimental count peaks, all bin tests and run metrics.

    A supplied DataFrame is revalidated; its relative pair paths resolve from
    the current directory. Paths loaded by load_manifest are already absolute.
    """
    from .counts import (
        bh_adjust,
        combine_replicate_pvalues,
        count_pairs,
        estimate_dispersion,
        load_manifest,
        negative_binomial_pvalues,
        validate_manifest_frame,
    )

    params = params or AdaptiveParams()
    if isinstance(manifest, pd.DataFrame):
        frame = validate_manifest_frame(manifest, viewpoint, base_dir=Path.cwd())
    else:
        frame = load_manifest(manifest, viewpoint)
    if params.bin_size > params.max_peak_size:
        raise ValueError("Count bin_size cannot exceed max_peak_size")
    chrom, vp = _viewpoint(viewpoints, chrom_sizes)
    n = chrom_sizes[chrom]
    normalized_exclusions = _validate_exclusions(exclusions, chrom_sizes)
    # Count mode retains raw, expected and statistical arrays plus its bin table.
    estimated_mb = math.ceil(n / params.bin_size) * (48 * len(frame) + 128) / 1024**2
    if estimated_mb > params.memory_budget_mb:
        raise ValueError(
            f"Count-mode arrays need approximately {math.ceil(estimated_mb)} MiB; increase memory_budget_mb or bin_size"
        )
    excluded = list(normalized_exclusions.get(chrom, [])) + [
        (max(0, a - params.bait_padding), min(n, b + params.bait_padding)) for a, b in vp
    ]
    eligible = _masked_bins(n, params.bin_size, excluded)
    if not np.any(eligible):
        raise ValueError("No eligible bins outside exclusions/viewpoint")
    cache = ArrayCache(params.memory_budget_mb / 2, cache_dir)
    arrays, expected, pvalues, identities, qc = [], [], [], [], []
    t0 = time.perf_counter()
    if scratch_dir:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_dir, prefix="lanceotron-counts-") as tmp:
        for row in frame.itertuples(index=False):
            identity = source_identity(row.pairs)
            key = cache_key(identity, "counts", viewpoints, chrom_sizes, excluded, params.bin_size)
            counts = cache.get(key)
            if counts is None:
                counts, sample_qc = count_pairs(
                    row.pairs, chrom_sizes, viewpoints, params.bin_size, {chrom: excluded}
                )
                counts = cache.put(key, counts)
                sample_qc = dict(sample_qc)
                cache.put_metadata(key, sample_qc)
            else:
                sample_qc = cache.get_metadata(key) or {"cached_qc_unavailable": True}
                sample_qc["cached"] = True
            if not np.any(counts[eligible]):
                raise ValueError(f"No eligible cis counts for sample {row.sample}")
            widths = np.minimum(params.bin_size, n - np.arange(len(counts)) * params.bin_size)
            positions = np.arange(len(counts)) * params.bin_size + widths / 2
            profile_width = _profile_bin_size(n, params, unit=params.bin_size)
            factor = max(1, profile_width // params.bin_size)
            starts = np.arange(0, len(counts), factor, dtype=np.int64)
            profile_widths = np.minimum(profile_width, n - starts * params.bin_size)
            profile_positions = starts * params.bin_size + profile_widths / 2
            profile_signal = np.add.reduceat(counts, starts, dtype=np.float64) / profile_widths
            profile_eligible = _masked_bins(n, profile_width, excluded)
            profile = fit_distance_background(
                profile_signal,
                profile_positions,
                vp,
                profile_eligible,
                profile_widths,
                count_mode=True,
            )
            bg = np.interp(positions, profile_positions, profile) * widths
            dispersion = estimate_dispersion(
                counts, bg, eligible & (counts <= bg + 4 * np.sqrt(bg + 1))
            )
            p = negative_binomial_pvalues(counts, bg, dispersion)
            sample_qc.update(
                sample=str(row.sample),
                dispersion=float(dispersion),
                eligible_cis_events=int(np.sum(counts[eligible], dtype=np.int64)),
            )
            arrays.append(counts)
            expected.append(bg)
            pvalues.append(p)
            identities.append(identity)
            qc.append(sample_qc)
        combined = combine_replicate_pvalues(np.asarray(pvalues))
        adjusted = bh_adjust(combined, eligible)
        # Normalize candidates/effect sizes for equal replicate contribution;
        # count-mode significance uses the original integer observations above.
        depths = np.asarray([np.sum(a[eligible], dtype=np.float64) for a in arrays])
        scales = np.median(depths) / depths
        scaled = [_ScaledArray(a, s) for a, s in zip(arrays, scales)]
        scaled_expected = [bg * scale for bg, scale in zip(expected, scales)]
        result = _call_arrays(
            arrays=scaled,
            expected=scaled_expected,
            exclusions=excluded,
            chrom=chrom,
            chrom_length=n,
            unit=params.bin_size,
            params=params,
            tmp=tmp,
            identities=[
                identity | {"scale": float(scale)} for identity, scale in zip(identities, scales)
            ],
            cache=cache,
            sample_names=frame["sample"].astype(str).tolist(),
            eligible_bins=eligible,
            qvalues=adjusted,
        )
        # Replace descriptive normalized sum with actual reporter-event counts.
        if not result.empty:
            result["Count"] = [
                int(
                    sum(
                        np.sum(
                            a[
                                int(row.Start) // params.bin_size : math.ceil(
                                    int(row.End) / params.bin_size
                                )
                            ],
                            dtype=np.int64,
                        )
                        for a in arrays
                    )
                )
                for row in result.itertuples()
            ]
            result = result[result.Count >= params.min_read_count].reset_index(drop=True)
        bin_frame = pd.DataFrame(
            {
                "Chromosome": chrom,
                "Start": np.arange(len(combined)) * params.bin_size,
                "End": np.minimum(n, (np.arange(len(combined)) + 1) * params.bin_size),
                "Eligible": eligible,
                "PValue": combined,
                "BinQ": adjusted,
            }
        )
        for name, observed, bg, p in zip(frame["sample"], arrays, expected, pvalues):
            bin_frame[f"Count:{name}"] = observed
            bin_frame[f"Expected:{name}"] = bg
            bin_frame[f"PValue:{name}"] = p
        metrics = {
            "samples": qc,
            "cache": dict(cache.metrics),
            "normalization_factors": scales.tolist(),
            "elapsed_seconds": time.perf_counter() - t0,
            "peak_memory_mb": peak_memory_mb(),
            "experimental": True,
            "count_unit": "reporter event, not independent molecule",
        }
    return result, bin_frame, metrics
