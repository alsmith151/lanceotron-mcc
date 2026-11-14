"""Experimental adapters and statistics for standard ``.pairs`` counts.

Counts produced here are *reporter-end pair events*. They are not deduplicated
and must not be interpreted as independent molecules.

Pair files are streamed as standard pairs rows. Optional pair metadata such as
``pair_type`` is accepted but not used for filtering; upstream deduplication and
quality filtering remain the caller's responsibility.
"""

from __future__ import annotations

import csv
import gzip
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, nbinom, poisson

PAIR_COLUMNS = ("readid", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2")
_PAIR_COLUMN_ALIASES = {
    "readID": "readid",
    "read_id": "readid",
    "chrom1": "chr1",
    "chrom2": "chr2",
}
_MCC_SIDES = {"start", "end", "all", "none"}


@dataclass(frozen=True)
class _MCCPairMetadata:
    parent_id: str
    viewpoint: str
    chromosome: str
    start: int
    end: int
    side: str
    read_number: int
    flashed_status: int


@dataclass(frozen=True)
class PairRecord:
    """A validated mapped record from a standard pairs file."""

    read_id: str
    chrom1: str
    pos1: int
    chrom2: str
    pos2: int
    strand1: str
    strand2: str
    metadata: _MCCPairMetadata | None
    shape: str | None = None


@dataclass(frozen=True)
class PairContact:
    """A validated pair oriented from its viewpoint capture to its reporter."""

    read_id: str
    capture_chrom: str
    capture_pos: int
    reporter_chrom: str
    reporter_pos: int
    capture_strand: str
    reporter_strand: str
    metadata: _MCCPairMetadata | None


def new_pairs_qc(path: str | Path) -> dict[str, int | str]:
    """Create a QC mapping suitable for ``iter_viewpoint_contacts`` or ``count_pairs``."""
    return {
        "path": str(Path(path)),
        "reporter_end": "non-viewpoint end",
        "capture_end": "viewpoint-matching end",
        "count_semantics": "reporter-end pair events; not deduplicated molecules",
        "input_pairs": 0,
        "trans_pairs": 0,
        "both_viewpoint_pairs": 0,
        "non_viewpoint_pairs": 0,
        "unmapped_pairs": 0,
        "mcc_annotated_pairs": 0,
        "mcc_start_pairs": 0,
        "mcc_end_pairs": 0,
        "mcc_all_pairs": 0,
        "mcc_none_pairs": 0,
        "self_contact_pairs": 0,
        "capture_end_1_pairs": 0,
        "capture_end_2_pairs": 0,
        "usable_cis_depth": 0,
        "excluded_reporter_events": 0,
        "retained_reporter_events": 0,
    }


def _parse_mcc_read_id(read_id: str, chrom_sizes: Mapping[str, int]) -> _MCCPairMetadata | None:
    """Parse a strict MCCNado SegmentMetadata suffix, if present."""
    parts = read_id.rsplit("__", 4)
    if len(parts) != 5:
        return None
    parent_id, location, side, read_number, flashed_status = parts
    if (
        side not in _MCC_SIDES
        or read_number not in {"1", "2", "3"}
        or flashed_status not in {"0", "1"}
    ):
        return None
    match = re.fullmatch(r"(.+)-(-?\d+)-(-?\d+)", location)
    if match is None:
        raise ValueError(f"Invalid MCC read-name coordinates in {read_id!r}")
    chrom_and_viewpoint, raw_start, raw_end = match.groups()
    start, end = int(raw_start), int(raw_end)
    if start < 0 or start >= end:
        raise ValueError(f"Invalid MCC viewpoint interval in read name {read_id!r}: {start}-{end}")
    resolved: tuple[str, str] | None = None
    for chromosome in sorted(chrom_sizes, key=len, reverse=True):
        suffix = f"-{chromosome}"
        if chrom_and_viewpoint.endswith(suffix):
            viewpoint = chrom_and_viewpoint[: -len(suffix)]
            if viewpoint:
                resolved = viewpoint, chromosome
                break
    if resolved is None:
        raise ValueError(
            f"Could not resolve chromosome in MCC read name {read_id!r} against supplied chromosome sizes"
        )
    viewpoint, chromosome = resolved
    if end > int(chrom_sizes[chromosome]):
        raise ValueError(
            f"MCC viewpoint interval {chromosome}:{start}-{end} in read name {read_id!r} "
            f"exceeds chromosome length {chrom_sizes[chromosome]}"
        )
    return _MCCPairMetadata(
        parent_id, viewpoint, chromosome, start, end, side, int(read_number), int(flashed_status)
    )


def _parse_pair_columns(line: str, path: Path, line_number: int) -> int:
    """Validate a standard ``#columns`` declaration and return its width."""
    columns = line[len("#columns:") :].strip().split()
    if len(columns) < len(PAIR_COLUMNS):
        raise ValueError(f"{path}:{line_number}: #columns header must declare at least 7 columns")
    first_seven = [_PAIR_COLUMN_ALIASES.get(column, column) for column in columns[:7]]
    if first_seven != list(PAIR_COLUMNS):
        raise ValueError(
            f"{path}:{line_number}: #columns first 7 fields must be "
            "readID, chr1, pos1, chr2, pos2, strand1, strand2"
        )
    if len(columns) != len(set(columns)):
        raise ValueError(f"{path}:{line_number}: #columns header contains duplicate columns")
    return len(columns)


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


def _contains(intervals: Sequence[tuple[int, int]], position: int) -> bool:
    return any(start <= position < end for start, end in intervals)


def iter_pairs(
    path: str | Path,
    chrom_sizes: Mapping[str, int],
    *,
    qc: dict[str, int | str] | None = None,
) -> Iterator[PairRecord]:
    """Stream validated mapped records from a standard pairs file.

    The first seven standard fields are required; additional fields are ignored.
    Returned positions are zero-based. Unmapped records are validated, counted
    in QC, and skipped. MCC read-name metadata is parsed when its strict suffix
    schema is present, without using it to infer pair-end capture membership.
    """
    sizes: dict[str, int] = {}
    for name, size in chrom_sizes.items():
        if isinstance(size, bool) or int(size) != size or int(size) <= 0:
            raise ValueError(f"Invalid chromosome size for {name!r}: {size!r}")
        sizes[str(name)] = int(size)
    qc = qc if qc is not None else new_pairs_qc(path)
    for key, value in new_pairs_qc(path).items():
        qc.setdefault(key, value)
    pair_path = Path(path)
    opener = gzip.open if pair_path.suffix.lower() == ".gz" else open
    declared_width: int | None = None
    shape: str | None = None
    try:
        with opener(pair_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("#columns:"):
                    if declared_width is not None:
                        raise ValueError(f"{pair_path}:{line_number}: duplicate #columns header")
                    declared_width = _parse_pair_columns(line, pair_path, line_number)
                    continue
                if line.startswith("#shape:"):
                    if shape is not None:
                        raise ValueError(f"{pair_path}:{line_number}: duplicate #shape header")
                    shape = line[len("#shape:") :].strip().lower()
                    continue
                if line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 7:
                    raise ValueError(
                        f"{pair_path}:{line_number}: expected at least 7 tab-separated columns, found {len(fields)}"
                    )
                if declared_width is not None and len(fields) != declared_width:
                    raise ValueError(
                        f"{pair_path}:{line_number}: row has {len(fields)} fields but "
                        f"#columns header declares {declared_width}"
                    )
                read_id, chr1, raw_pos1, chr2, raw_pos2, strand1, strand2 = fields[:7]
                metadata = _parse_mcc_read_id(read_id, sizes)
                try:
                    pos1, pos2 = int(raw_pos1), int(raw_pos2)
                except ValueError as exc:
                    raise ValueError(
                        f"{pair_path}:{line_number}: pair positions must be integers"
                    ) from exc
                unmapped = []
                for pair_chrom, position in ((chr1, pos1), (chr2, pos2)):
                    if pair_chrom == "!":
                        if position != 0:
                            raise ValueError(
                                f"{pair_path}:{line_number}: unmapped chromosome '!' must have position 0"
                            )
                        unmapped.append(True)
                        continue
                    unmapped.append(False)
                    if pair_chrom not in sizes:
                        raise ValueError(
                            f"{pair_path}:{line_number}: no chromosome size for {pair_chrom!r}"
                        )
                    if position < 1 or position > sizes[pair_chrom]:
                        raise ValueError(
                            f"{pair_path}:{line_number}: one-based position {position} is outside "
                            f"{pair_chrom} bounds 1..{sizes[pair_chrom]}"
                        )
                qc["input_pairs"] += 1
                if metadata is not None:
                    qc["mcc_annotated_pairs"] += 1
                    qc[f"mcc_{metadata.side}_pairs"] += 1
                if any(unmapped):
                    qc["unmapped_pairs"] += 1
                    continue
                yield PairRecord(
                    read_id,
                    chr1,
                    pos1 - 1,
                    chr2,
                    pos2 - 1,
                    strand1,
                    strand2,
                    metadata,
                    shape,
                )
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read pair file {pair_path}: {exc}") from exc


def iter_viewpoint_contacts(
    path: str | Path,
    chrom_sizes: Mapping[str, int],
    viewpoint_intervals: Mapping[str, Sequence[Sequence[int]]],
    *,
    qc: dict[str, int | str] | None = None,
) -> Iterator[PairContact]:
    """Stream valid pair events oriented from one viewpoint end to its reporter.

    Pair positions are zero-based, already converted by :func:`iter_pairs`.
    Cis and trans contacts are yielded; unmapped, self
    viewpoint, and non-viewpoint pairs are recorded in QC and skipped.
    """
    view = _normalize_intervals(viewpoint_intervals, "viewpoint_intervals")
    if len(view) != 1:
        raise ValueError("A viewpoint contact stream requires intervals on exactly one chromosome")
    chrom = next(iter(view))
    if chrom not in chrom_sizes:
        raise ValueError(f"No chromosome size supplied for viewpoint chromosome {chrom!r}")
    sizes = {str(name): int(size) for name, size in chrom_sizes.items()}
    for start, end in view[chrom]:
        if end > sizes[chrom]:
            raise ValueError(
                f"Viewpoint interval ({start}, {end}) exceeds {chrom} length {sizes[chrom]}"
            )
    qc = qc if qc is not None else new_pairs_qc(path)
    defaults = new_pairs_qc(path)
    for key, value in defaults.items():
        qc.setdefault(key, value)
    for record in iter_pairs(path, sizes, qc=qc):
        mcc = record.metadata
        if mcc is not None and mcc.side in {"all", "none"}:
            raise ValueError(f"MCC read name has unsupported reporter side {mcc.side!r}")
        if mcc is not None:
            oligo_matches_view = mcc.chromosome == chrom and any(
                mcc.start < end and start < mcc.end for start, end in view[chrom]
            )
            if not oligo_matches_view:
                qc["non_viewpoint_pairs"] += 1
                continue
            overlap1 = record.chrom1 == mcc.chromosome and mcc.start <= record.pos1 <= mcc.end
            overlap2 = record.chrom2 == mcc.chromosome and mcc.start <= record.pos2 <= mcc.end
            if not overlap1 and not overlap2:
                raise ValueError(
                    f"MCC viewpoint oligo {mcc.chromosome}:{mcc.start}-{mcc.end} "
                    "matches neither pair end; check the read-name annotation and alignment"
                )
        else:
            overlap1 = record.chrom1 == chrom and _contains(view[chrom], record.pos1)
            overlap2 = record.chrom2 == chrom and _contains(view[chrom], record.pos2)
        if overlap1 and overlap2:
            qc["both_viewpoint_pairs"] += 1
            if mcc is not None:
                qc["self_contact_pairs"] += 1
            continue
        if not overlap1 and not overlap2:
            qc["non_viewpoint_pairs"] += 1
            continue
        capture_index = 1 if overlap1 else 2
        if capture_index == 1:
            capture_chrom, capture_pos, capture_strand = (
                record.chrom1,
                record.pos1,
                record.strand1,
            )
            reporter_chrom, reporter_pos, reporter_strand = (
                record.chrom2,
                record.pos2,
                record.strand2,
            )
            qc["capture_end_1_pairs"] += 1
        else:
            capture_chrom, capture_pos, capture_strand = (
                record.chrom2,
                record.pos2,
                record.strand2,
            )
            reporter_chrom, reporter_pos, reporter_strand = (
                record.chrom1,
                record.pos1,
                record.strand1,
            )
            qc["capture_end_2_pairs"] += 1
        if reporter_chrom != chrom:
            qc["trans_pairs"] += 1
        else:
            qc["usable_cis_depth"] += 1
        yield PairContact(
            record.read_id,
            capture_chrom,
            capture_pos,
            reporter_chrom,
            reporter_pos,
            capture_strand,
            reporter_strand,
            mcc,
        )


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
    for contact in iter_viewpoint_contacts(path, chrom_sizes, viewpoint_intervals, qc=qc):
        if contact.reporter_chrom != chrom:
            continue
        if _contains(exclusions.get(chrom, []), contact.reporter_pos):
            qc["excluded_reporter_events"] += 1
            continue
        counts[contact.reporter_pos // bin_size] += 1
        qc["retained_reporter_events"] += 1
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


def negative_binomial_pvalues(
    observed: np.ndarray, expected: np.ndarray, dispersion: float
) -> np.ndarray:
    """Return upper-tail NB2 p-values with variance ``mu + dispersion * mu**2``."""
    obs, exp = _count_arrays(observed, expected)
    if np.any(obs != np.floor(obs)):
        raise ValueError("observed counts must be integers")
    if not math.isfinite(float(dispersion)) or dispersion < 0:
        raise ValueError("dispersion must be finite and non-negative")
    result = np.ones(obs.shape, dtype=np.float64)
    positive = exp > 0
    # A zero mean assigns no probability to any positive count. Keep p=1 for
    # observed zero, and return the exact upper tail (zero) for observed > 0.
    result[(exp == 0) & (obs > 0)] = 0.0
    if dispersion == 0:
        result[positive] = poisson.sf(obs[positive] - 1, exp[positive])
    else:
        size = 1.0 / float(dispersion)
        prob = size / (size + exp[positive])
        result[positive] = nbinom.sf(obs[positive] - 1, size, prob)
    return np.clip(np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def combine_replicate_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """Combine independent replicate p-values by Fisher's method (axis 0)."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("pvalues must be a non-empty 2D array shaped (replicates, bins)")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite and between 0 and 1")
    statistic = -2.0 * np.log(np.maximum(p, np.finfo(float).tiny)).sum(axis=0)
    combined = chi2.sf(statistic, 2 * p.shape[0])
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
