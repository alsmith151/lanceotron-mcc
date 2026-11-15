"""Experimental adapters and statistics for standard ``.pairs`` counts.

Counts produced here are *reporter-end pair events*. They are not deduplicated
and must not be interpreted as independent molecules.

Pair files are streamed as standard pairs rows. Optional pair metadata such as
``pair_type`` is accepted but not used for filtering; upstream deduplication and
quality filtering remain the caller's responsibility.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .pairs import (
    SIDE_CODES,
    PairReader,
    in_intervals,
    merge_intervals,
    new_pairs_qc,
)

__all__ = ["new_pairs_qc"]
_SIDE_NAMES = {code: name for name, code in SIDE_CODES.items()}


def validate_manifest_frame(
    frame: pd.DataFrame, viewpoint: str, base_dir: str | Path = "."
) -> pd.DataFrame:
    """Validate parsed manifest rows and resolve pair paths against ``base_dir``."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Manifest rows must be provided as a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Manifest contains duplicate column names")
    required = {"sample", "viewpoint", "pairs"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, ["sample", "viewpoint", "pairs"]].copy()
    selected = frame["viewpoint"].map(
        lambda value: False if pd.isna(value) else str(value) == str(viewpoint)
    )
    frame = frame.loc[selected].copy()
    for column in ("sample", "viewpoint", "pairs"):
        if frame[column].isna().any():
            raise ValueError(f"Manifest contains an empty {column!r} value")
        frame[column] = frame[column].map(str)
    if frame.empty:
        raise ValueError(f"Manifest contains no rows for viewpoint {viewpoint!r}")
    for column in ("sample", "viewpoint", "pairs"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Manifest contains an empty {column!r} value")
    if frame["sample"].duplicated().any():
        duplicates = sorted(frame.loc[frame["sample"].duplicated(keep=False), "sample"].unique())
        raise ValueError(
            f"Duplicate sample(s) for viewpoint {viewpoint!r}: {', '.join(duplicates)}"
        )
    base = Path(base_dir).expanduser().resolve()
    frame["pairs"] = frame["pairs"].map(
        lambda value: (
            str((base / value).resolve())
            if not Path(value).expanduser().is_absolute()
            else str(Path(value).expanduser().resolve())
        )
    )
    missing_files = [value for value in frame["pairs"] if not Path(value).is_file()]
    if missing_files:
        raise ValueError(
            "Pairs file(s) do not exist or are not regular files: " + ", ".join(missing_files)
        )
    paths_by_file_id: dict[tuple[int, int], list[str]] = {}
    for value in frame["pairs"]:
        stat = Path(value).stat()
        paths_by_file_id.setdefault((stat.st_dev, stat.st_ino), []).append(value)
    repeated_paths = frame["pairs"].duplicated(keep=False)
    duplicate_paths = set(frame.loc[repeated_paths, "pairs"])
    for paths in paths_by_file_id.values():
        if len(paths) > 1:
            duplicate_paths.update(paths)
    if duplicate_paths:
        paths = sorted(duplicate_paths)
        raise ValueError(
            "Each sample must use a distinct pairs file; duplicate resolved path(s): "
            + ", ".join(paths)
        )
    return frame.reset_index(drop=True)


def load_manifest(path: str | Path, viewpoint: str) -> pd.DataFrame:
    """Read a SeqNado TSV once, preserving literal values such as ``NA``.

    The manifest must contain ``sample``, ``viewpoint`` and ``pairs`` columns.
    Relative pair paths are resolved from the manifest's directory.
    """
    manifest_path = Path(path).expanduser().resolve()
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            header = reader.fieldnames or []
            if len(header) != len(set(header)):
                raise ValueError("Manifest contains duplicate column names")
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"Manifest row {line_number} has more fields than the header")
                if any(value is None for value in row.values()):
                    raise ValueError(f"Manifest row {line_number} has fewer fields than the header")
                rows.append(row)
    except (OSError, csv.Error, UnicodeError) as exc:
        raise ValueError(f"Could not read manifest {manifest_path}: {exc}") from exc
    return validate_manifest_frame(
        pd.DataFrame(rows, columns=header), viewpoint, manifest_path.parent
    )


def _normalize_intervals(intervals: Mapping[str, Sequence[Sequence[int]]], label: str):
    if not isinstance(intervals, Mapping) or len(intervals) == 0:
        raise ValueError(f"{label} must be a non-empty mapping of chromosome to intervals")
    result: dict[str, list[tuple[int, int]]] = {}
    for chrom, chrom_intervals in intervals.items():
        if not isinstance(chrom, str) or not chrom:
            raise ValueError(f"{label} chromosome names must be non-empty strings")
        normalized = []
        for interval in chrom_intervals:
            if len(interval) != 2:
                raise ValueError(f"Each {label} interval must be a (start, end) pair")
            start, end = interval
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or int(start) != start
                or int(end) != end
            ):
                raise ValueError(f"{label} interval coordinates must be integers")
            start, end = int(start), int(end)
            if start < 0 or end <= start:
                raise ValueError(f"Invalid {label} interval ({start}, {end})")
            normalized.append((start, end))
        if not normalized:
            raise ValueError(f"{label} contains no intervals for {chrom!r}")
        result[chrom] = normalized
    return result


def count_pairs(
    path: str | Path,
    chrom_sizes: Mapping[str, int],
    viewpoint_intervals: Mapping[str, Sequence[Sequence[int]]],
    bin_size: int = 50,
    excluded: Mapping[str, Sequence[Sequence[int]]] | None = None,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Count reporter-end events in fixed-width cis bins for one viewpoint.

    The standard first seven pairs fields are read ID, chromosome, one-based
    position, chromosome, one-based position, and two strands. Additional
    columns are accepted and ignored. Either end may overlap the viewpoint.
    Duplicate read IDs remain separate events.

    Excluded reporter intervals are zero-based, half-open. ``usable_cis_depth``
    in QC is measured before those intervals are removed.
    """
    if isinstance(bin_size, bool) or int(bin_size) != bin_size or bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")
    bin_size = int(bin_size)
    view = _normalize_intervals(viewpoint_intervals, "viewpoint_intervals")
    if len(view) != 1:
        raise ValueError("A count array requires viewpoint intervals on exactly one chromosome")
    chrom = next(iter(view))
    if chrom not in chrom_sizes:
        raise ValueError(f"No chromosome size supplied for viewpoint chromosome {chrom!r}")
    sizes: dict[str, int] = {}
    for name, size in chrom_sizes.items():
        if isinstance(size, bool) or int(size) != size or int(size) <= 0:
            raise ValueError(f"Invalid chromosome size for {name!r}: {size!r}")
        sizes[str(name)] = int(size)
    chrom_length = sizes[chrom]
    for start, end in view[chrom]:
        if end > chrom_length:
            raise ValueError(
                f"Viewpoint interval ({start}, {end}) exceeds {chrom} length {chrom_length}"
            )
    exclusions = _normalize_intervals(excluded, "excluded") if excluded is not None else {}
    for name, intervals in exclusions.items():
        if name not in sizes:
            raise ValueError(f"No chromosome size supplied for excluded chromosome {name!r}")
        if any(end > sizes[name] for _, end in intervals):
            raise ValueError(f"Excluded interval exceeds {name} length {sizes[name]}")
    counts = np.zeros((chrom_length + bin_size - 1) // bin_size, dtype=np.int64)
    qc = new_pairs_qc(path)
    reader = PairReader(path, sizes, qc=qc)
    code = reader.index.codes[chrom]
    view_starts, view_ends = merge_intervals(view[chrom])
    excluded_starts, excluded_ends = merge_intervals(exclusions.get(chrom, []))
    for chunk in reader:
        frame = chunk.frame
        c1, p1 = frame["chrom1"].to_numpy(), frame["pos1"].to_numpy()
        c2, p2 = frame["chrom2"].to_numpy(), frame["pos2"].to_numpy()
        side = frame["side"].to_numpy()
        meta_chrom = frame["meta_chrom"].to_numpy()
        meta_start, meta_end = frame["meta_start"].to_numpy(), frame["meta_end"].to_numpy()
        annotated = side >= 0
        matches_view = annotated & (meta_chrom == code)
        overlap = np.zeros(len(frame), dtype=bool)
        for start, end in view[chrom]:
            overlap |= (meta_start < end) & (start < meta_end)
        matches_view &= overlap
        meta1 = (c1 == meta_chrom) & (meta_start <= p1) & (p1 <= meta_end)
        meta2 = (c2 == meta_chrom) & (meta_start <= p2) & (p2 <= meta_end)
        unsupported = annotated & (side >= SIDE_CODES["all"])
        failures = unsupported | (matches_view & ~meta1 & ~meta2)
        if failures.any():
            row = int(np.argmax(failures))
            if unsupported[row]:
                raise ValueError(
                    f"MCC read name has unsupported reporter side {_SIDE_NAMES[int(side[row])]!r}"
                )
            raise ValueError(
                f"MCC viewpoint oligo {chrom}:{int(meta_start[row])}-{int(meta_end[row])} "
                "matches neither pair end; check the read-name annotation and alignment"
            )
        overlap1 = np.where(
            annotated, meta1, (c1 == code) & in_intervals(p1, view_starts, view_ends)
        )
        overlap2 = np.where(
            annotated, meta2, (c2 == code) & in_intervals(p2, view_starts, view_ends)
        )
        active = ~annotated | matches_view
        both = active & overlap1 & overlap2
        capture1 = active & overlap1 & ~overlap2
        capture2 = active & ~overlap1 & overlap2
        qc["non_viewpoint_pairs"] += int(np.sum(annotated & ~matches_view))
        qc["non_viewpoint_pairs"] += int(np.sum(active & ~overlap1 & ~overlap2))
        qc["both_viewpoint_pairs"] += int(both.sum())
        qc["self_contact_pairs"] += int(np.sum(both & annotated))
        qc["capture_end_1_pairs"] += int(capture1.sum())
        qc["capture_end_2_pairs"] += int(capture2.sum())
        contact = capture1 | capture2
        reporter_chrom = np.where(capture1, c2, c1)
        reporter_pos = np.where(capture1, p2, p1)
        cis = contact & (reporter_chrom == code)
        qc["trans_pairs"] += int(np.sum(contact & ~cis))
        qc["usable_cis_depth"] += int(cis.sum())
        positions = reporter_pos[cis]
        dropped = in_intervals(positions, excluded_starts, excluded_ends)
        qc["excluded_reporter_events"] += int(dropped.sum())
        positions = positions[~dropped]
        counts += np.bincount(positions // bin_size, minlength=len(counts)).astype(np.int64)
        qc["retained_reporter_events"] += int(len(positions))
    return counts, qc


def _count_arrays(observed: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    if obs.shape != exp.shape:
        raise ValueError("observed and expected must have the same shape")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(exp)):
        raise ValueError("observed and expected values must be finite")
    if np.any(obs < 0) or np.any(exp < 0):
        raise ValueError("observed and expected counts must be non-negative")
    return obs, exp


def estimate_dispersion(
    observed: np.ndarray, expected: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Estimate NB2 dispersion by moments; return zero when depth is inadequate."""
    obs, exp = _count_arrays(observed, expected)
    if mask is not None:
        include = np.asarray(mask, dtype=bool)
        if include.shape != obs.shape:
            raise ValueError("mask must have the same shape as observed")
        obs, exp = obs[include], exp[include]
    # A small expected depth has no stable variance estimate; fall back to Poisson; this is not guaranteed conservative.
    if obs.size < 2 or float(exp.sum()) < 2.0 or float(np.square(exp).sum()) == 0.0:
        return 0.0
    moment = float(np.square(obs - exp).sum() - exp.sum()) / float(np.square(exp).sum())
    return max(0.0, moment) if math.isfinite(moment) else 0.0


def _nbinom_sf():
    """The ufunc behind ``scipy.stats.nbinom.sf`` (private in scipy.special)."""
    try:
        from scipy.special._ufuncs import _nbinom_sf
    except ImportError:  # pragma: no cover - older/newer scipy layouts
        from scipy.stats import nbinom

        return lambda k, n, p: nbinom.sf(k, n, p)
    return _nbinom_sf


def _discrete_sf(ufunc, k: np.ndarray, *params: np.ndarray | float) -> np.ndarray:
    """``scipy.stats`` discrete ``sf`` for integer ``k``: 1 below the support at 0.

    Calls the distribution's underlying ufunc directly; importing
    ``scipy.special`` is much cheaper than ``scipy.stats``.
    """
    result = np.ones(k.shape, dtype=np.float64)
    inside = k >= 0
    if inside.any():
        result[inside] = ufunc(k[inside], *(p[inside] if np.ndim(p) else p for p in params))
    return result


def negative_binomial_pvalues(
    observed: np.ndarray, expected: np.ndarray, dispersion: float
) -> np.ndarray:
    """Return upper-tail NB2 p-values with variance ``mu + dispersion * mu**2``."""
    obs, exp = _count_arrays(observed, expected)
    if np.any(obs != np.floor(obs)):
        raise ValueError("observed counts must be integers")
    if not math.isfinite(float(dispersion)) or dispersion < 0:
        raise ValueError("dispersion must be finite and non-negative")
    from scipy import special  # deferred import

    result = np.ones(obs.shape, dtype=np.float64)
    positive = exp > 0
    # A zero mean assigns no probability to any positive count. Keep p=1 for
    # observed zero, and return the exact upper tail (zero) for observed > 0.
    result[(exp == 0) & (obs > 0)] = 0.0
    if dispersion == 0:
        result[positive] = _discrete_sf(special.pdtrc, obs[positive] - 1, exp[positive])
    else:
        size = 1.0 / float(dispersion)
        prob = size / (size + exp[positive])
        result[positive] = _discrete_sf(_nbinom_sf(), obs[positive] - 1, size, prob)
    return np.clip(np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def combine_replicate_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """Combine independent replicate p-values by Fisher's method (axis 0)."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("pvalues must be a non-empty 2D array shaped (replicates, bins)")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite and between 0 and 1")
    from scipy import special  # deferred import

    statistic = -2.0 * np.log(np.maximum(p, np.finfo(float).tiny)).sum(axis=0)
    # chi2.sf: 1 at or below zero, else the complemented chi-square ufunc.
    combined = np.ones(statistic.shape, dtype=np.float64)
    above = statistic > 0
    combined[above] = special.chdtrc(2 * p.shape[0], statistic[above])
    return np.clip(np.nan_to_num(combined, nan=1.0), 0.0, 1.0)


def bh_adjust(pvalues: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; ineligible positions receive 1."""
    p = np.asarray(pvalues, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite and between 0 and 1")
    include = np.ones(p.shape, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if include.shape != p.shape:
        raise ValueError("eligible must have the same shape as pvalues")
    flat_p, flat_include = p.ravel(), include.ravel()
    indices = np.flatnonzero(flat_include)
    adjusted = np.ones(flat_p.shape, dtype=np.float64)
    if indices.size:
        order = np.argsort(flat_p[indices], kind="stable")
        sorted_indices = indices[order]
        ranks = np.arange(1, indices.size + 1, dtype=np.float64)
        ranked = flat_p[sorted_indices] * indices.size / ranks
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted[sorted_indices] = np.minimum(ranked, 1.0)
    return adjusted.reshape(p.shape)
