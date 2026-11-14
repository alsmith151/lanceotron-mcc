import pathlib

import numpy as np
import pandas as pd
import pyranges as pr
import torch

from .io import BigWigReader


def average_array(array: torch.Tensor, window: int = 0) -> torch.Tensor:
    """Return a centered moving average, using shorter windows at the edges.

    For even windows, the extra sample is on the left (``left = window // 2``),
    matching the historical interior alignment. A window of zero keeps the
    historical meaning of averaging the whole input.
    """
    if not isinstance(array, torch.Tensor):
        array = torch.as_tensor(array)
    if array.ndim != 1:
        raise ValueError("array must be one-dimensional")
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("window must be a non-negative integer")

    n = array.numel()
    if n == 0:
        return array.clone()
    if not torch.isfinite(array).all():
        raise ValueError("array must contain only finite values")
    if window == 0:
        window = n
    if window >= n:
        if array.is_floating_point():
            return torch.full_like(array, array.mean())
        return array.to(torch.float64).mean().expand(n).clone()

    # Accumulate in float64 on CPU to avoid loss of precision on long genomic
    # signals. Keep the input precision on accelerators, where float64 support
    # can be limited or slow.
    work_dtype = (
        torch.float64
        if array.device.type == "cpu"
        else (array.dtype if array.is_floating_point() else torch.float32)
    )
    values = array.to(work_dtype)
    prefix = torch.cat(
        (torch.zeros(1, dtype=work_dtype, device=array.device), values.cumsum(dim=0))
    )
    indices = torch.arange(n, device=array.device)
    left = window // 2
    right = window - left - 1
    starts = (indices - left).clamp(min=0)
    ends = (indices + right + 1).clamp(max=n)
    result = (prefix[ends] - prefix[starts]) / (ends - starts).to(work_dtype)
    return result.to(array.dtype) if array.is_floating_point() else result


def find_enriched_regions(
    array: torch.Tensor,
    threshold: torch.Tensor | float,
    min_region_size: int = 0,
    max_region_size: int | None = None,
) -> torch.Tensor:
    """Find half-open intervals where values are positive and meet threshold."""
    if not isinstance(array, torch.Tensor):
        array = torch.as_tensor(array)
    if array.ndim != 1:
        raise ValueError("array must be one-dimensional")
    if (
        isinstance(min_region_size, bool)
        or not isinstance(min_region_size, int)
        or min_region_size < 0
    ):
        raise ValueError("min_region_size must be a non-negative integer")
    if max_region_size is not None and (
        isinstance(max_region_size, bool)
        or not isinstance(max_region_size, int)
        or max_region_size < min_region_size
    ):
        raise ValueError("max_region_size must be at least min_region_size")

    original_device = array.device
    # MPS does not support every operation used below consistently; perform
    # the compact boolean and transition operations on CPU in that case.
    work = array.cpu() if original_device.type == "mps" else array
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=work.device)
    else:
        threshold = torch.as_tensor(threshold, device=work.device)
    if threshold.ndim not in (0, 1) or (threshold.ndim == 1 and threshold.shape != work.shape):
        raise ValueError("threshold must be a scalar or match the array shape")
    if not torch.isfinite(threshold).all():
        raise ValueError("threshold must contain only finite values")
    if not torch.isfinite(work).all():
        raise ValueError("array must contain only finite values")

    mask = (work > 0) & (work >= threshold)
    n = mask.numel()
    if n == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=original_device)

    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.bool, device=work.device),
            mask,
            torch.zeros(1, dtype=torch.bool, device=work.device),
        )
    )
    starts = torch.where(padded[1:] & ~padded[:-1])[0]
    ends = torch.where(~padded[1:] & padded[:-1])[0]
    regions = torch.stack((starts, ends), dim=1).to(torch.int64)

    if max_region_size is None:
        max_region_size = n
    sizes = regions[:, 1] - regions[:, 0]
    regions = regions[(sizes >= min_region_size) & (sizes <= max_region_size)]
    return regions.to(original_device)


def add_signal(peaks: pr.PyRanges | pd.DataFrame, bigwig: pathlib.Path | str) -> pr.PyRanges:
    """Add mean BigWig signal to intervals, reading merged ranges per chromosome."""
    if isinstance(peaks, pr.PyRanges):
        peaks = peaks.df.copy()
    else:
        peaks = peaks.copy()
    if "Signal" in peaks.columns:
        return pr.PyRanges(peaks)
    if peaks.empty:
        peaks["Signal"] = pd.Series(dtype="float64")
        return pr.PyRanges(peaks)

    values = np.zeros(len(peaks), dtype=np.float64)
    starts = peaks["Start"].to_numpy(dtype=np.int64)
    ends = peaks["End"].to_numpy(dtype=np.int64)
    chroms = peaks["Chromosome"].to_numpy()

    with BigWigReader(str(bigwig)) as reader:
        # Merge overlapping intervals so each part of the file is fetched only
        # once, then calculate each original interval's mean from that buffer.
        for chrom in pd.unique(chroms):
            if chrom not in reader.chroms:
                raise ValueError(f"Chromosome {chrom!r} is absent from the BigWig")
            row_ids = np.flatnonzero(chroms == chrom)
            valid_ids = row_ids[ends[row_ids] > starts[row_ids]]
            if len(valid_ids) == 0:
                continue
            order = valid_ids[np.argsort(starts[valid_ids], kind="stable")]
            merged = []
            for idx in order:
                start, end = int(starts[idx]), int(ends[idx])
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                    merged[-1][2].append(int(idx))
                else:
                    merged.append([start, end, [int(idx)]])
            for start, end, members in merged:
                signal = np.asarray(reader.read(str(chrom), start=start, end=end), dtype=np.float64)
                for idx in members:
                    left = int(starts[idx]) - start
                    right = int(ends[idx]) - start
                    segment = signal[left:right]
                    finite = segment[np.isfinite(segment)]
                    values[idx] = float(finite.mean()) if finite.size else 0.0

    peaks["Signal"] = values
    return pr.PyRanges(peaks)
