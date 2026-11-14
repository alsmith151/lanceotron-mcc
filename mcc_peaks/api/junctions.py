"""Base-pair junction-end profiles and sparse maps for captured MCC pools.

Capture oligo names identify enrichment provenance in tiled experiments. They
are not substituted for genomic endpoints or required to touch every contact.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from .. import _core
from .adaptive import _validate_exclusions, _viewpoint
from .counts import load_manifest, validate_manifest_frame
from .io import peak_memory_mb
from .pairs import (
    SIDE_CODES,
    ChromIndex,
    PairReader,
    in_chrom_intervals,
    merge_intervals,
    validate_chrom_sizes,
)

PROFILE_COLUMNS = [
    "Sample",
    "Region",
    "Chromosome",
    "Position",
    "RegionStart",
    "RegionEnd",
    "Upstream",
    "Downstream",
    "Unknown",
    "Total",
]
CONTACT_COLUMNS = [
    "Sample",
    "Region",
    "CaptureChromosome",
    "CapturePosition",
    "ReporterChromosome",
    "ReporterPosition",
    "CaptureDirection",
    "ReporterDirection",
    "Count",
]
REGION_COLUMNS = ["Region", "Kind", "Chromosome", "Start", "End"]
# Direction codes are ranked by name so code order matches name order.
DIRECTION_NAMES = ["downstream", "unknown", "upstream"]
DOWNSTREAM, UNKNOWN, UPSTREAM = range(3)
# Profile column (Upstream, Downstream, Unknown) for each direction name.
_STORE = {"upstream": 0, "downstream": 1, "unknown": 2}
STALE_FORMAT_WARNING = (
    "No contact had a junction end inside a capture window, so the capture x reporter map is "
    "empty. Pairs written by older MCCNado releases store the far end of the reporter segment "
    "in pos2 instead of the capture junction; regenerate ligation junctions with current "
    "MCCNado, or widen capture_flank if capture segments extend further from the oligos."
)


@dataclass(frozen=True)
class JunctionRegion:
    name: str
    kind: Literal["capture", "reporter"]
    chrom: str
    start: int
    end: int


_SHAPE_ERROR = "Flipped .pairs do not preserve MCC reporter order; use orientation_source='unknown' or unflipped input"


def junction_directions(
    side: np.ndarray, strand1: np.ndarray, strand2: np.ndarray, source: str = "mccnado"
) -> tuple[np.ndarray, np.ndarray]:
    """Fragment extension from the junction in endpoint 1 / endpoint 2 order.

    MCCNado stores the named reporter as endpoint 1 before optional external
    flipping. `start` is a right reporter segment; `end` is a left segment.
    The second fragment has the complementary ligated endpoint in read order.
    This uses both segment side and strand, not strand alone. Returns
    :data:`DIRECTION_NAMES` codes per endpoint; unannotated reads are unknown.
    """
    if source not in {"mccnado", "unknown"}:
        raise ValueError("orientation_source must be mccnado or unknown")
    unknown = np.full(len(side), UNKNOWN, dtype=np.int64)
    if source == "unknown":
        return unknown, unknown.copy()
    start, end = side == SIDE_CODES["start"], side == SIDE_CODES["end"]

    def direction(strand, positive_downstream):
        plus, minus = strand == "+", strand == "-"
        downstream = np.where(plus, positive_downstream, ~positive_downstream)
        result = np.where(downstream, DOWNSTREAM, UPSTREAM)
        result[~(plus | minus) | ~(start | end)] = UNKNOWN
        return result.astype(np.int64)

    return direction(strand1, start), direction(strand2, end)


def read_peak_regions(path: str | Path, chrom_sizes: Mapping[str, int]) -> pd.DataFrame:
    """Read BED3/BED4 reporter regions (e.g. a peak call) for junction profiling.

    Names come from column 4 when present, otherwise ``peak_{n}`` in file order.
    Coordinate validation happens in ``profile_junctions``.
    """
    rows = []
    try:
        with open(path) as handle:
            for line in handle:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                rows.append(line.rstrip("\r\n").split("\t"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read peak BED {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"Peak BED is empty: {path}")
    if any(len(row) < 3 for row in rows):
        raise ValueError("Peak BED must contain at least three tab-separated columns")
    frame = pd.DataFrame([row[:4] for row in rows])
    for column in (1, 2):
        if not frame[column].str.fullmatch(r"[0-9]+").all():
            raise ValueError("Peak BED start and end must be nonnegative integers")
    names = frame[3].fillna("") if frame.shape[1] >= 4 else [""] * len(frame)
    names = [name or f"peak_{i + 1}" for i, name in enumerate(names)]
    missing = sorted(set(frame[0]) - set(chrom_sizes))
    if missing:
        raise ValueError(
            f"Peak BED chromosomes are absent from chromosome sizes: {', '.join(missing)}"
        )
    return pd.DataFrame(
        {
            "Chromosome": frame[0].to_numpy(),
            "Start": frame[1].astype(np.int64).to_numpy(),
            "End": frame[2].astype(np.int64).to_numpy(),
            "Name": names,
        }
    )


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _regions(viewpoints, chrom_sizes, peaks, reporter_flank, capture_flank):
    chrom, oligos = _viewpoint(viewpoints, chrom_sizes)
    capture = [
        JunctionRegion(f"capture:oligo_{i + 1}", "capture", chrom, start, end)
        for i, (start, end) in enumerate(
            _merge(
                [
                    (
                        max(0, start - capture_flank),
                        min(chrom_sizes[chrom], end + capture_flank + 1),
                    )
                    for start, end in oligos
                ]
            )
        )
    ]
    if peaks is None:
        peaks = pd.DataFrame(
            [(chrom, a, b, f"tile_{i + 1}") for i, (a, b) in enumerate(_merge(oligos))],
            columns=["Chromosome", "Start", "End", "Name"],
        )
    if not {"Chromosome", "Start", "End"}.issubset(peaks):
        raise ValueError("Peak regions require Chromosome, Start and End columns")
    reporter = []
    used = set()
    for i, row in enumerate(peaks.itertuples(index=False)):
        name = str(getattr(row, "Name", f"peak_{i + 1}"))
        identifier = f"reporter:{name}"
        if not name or identifier in used:
            raise ValueError("Reporter region names must be nonempty and unique")
        used.add(identifier)
        start, end = row.Start, row.End
        if row.Chromosome not in chrom_sizes or isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Reporter region chromosome or coordinates are invalid")
        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or int(start) != start
            or int(end) != end
            or not 0 <= start < end <= chrom_sizes[row.Chromosome]
        ):
            raise ValueError(
                "Reporter regions must be positive-width integer BED intervals within chromosome bounds"
            )
        reporter.append(
            JunctionRegion(
                identifier,
                "reporter",
                row.Chromosome,
                max(0, int(start) - reporter_flank),
                min(chrom_sizes[row.Chromosome], int(end) + reporter_flank),
            )
        )
    if not reporter:
        raise ValueError("At least one reporter region is required")
    return capture, reporter


def _region_table(regions, ids, index: ChromIndex):
    """Region lookup arrays for ``_core.footprint_rows``, sorted by chromosome and start."""
    order = sorted(
        range(len(regions)), key=lambda i: (index.codes[regions[i].chrom], regions[i].start)
    )
    chroms = np.asarray([index.codes[regions[i].chrom] for i in order], dtype=np.int64)
    starts = np.asarray([regions[i].start for i in order], dtype=np.int64)
    ends = np.asarray([regions[i].end for i in order], dtype=np.int64)
    max_ends = ends.copy()
    for i in range(1, len(order)):
        if chroms[i] == chroms[i - 1]:
            max_ends[i] = max(max_ends[i], max_ends[i - 1])
    offsets = np.searchsorted(chroms, np.arange(len(index.names) + 1)).astype(np.int64)
    return (
        offsets,
        starts,
        ends,
        max_ends,
        np.asarray([ids[i] for i in order], dtype=np.int64),
    )


def _coded_intervals(intervals, index: ChromIndex):
    return {index.codes[chrom]: merge_intervals(values) for chrom, values in intervals.items()}


def _max_depth(regions) -> int:
    """Largest number of regions covering one base."""
    events = sorted(
        [(r.chrom, r.start, 1) for r in regions] + [(r.chrom, r.end, -1) for r in regions],
        key=lambda event: (event[0], event[1], event[2]),
    )
    depth = best = 0
    for _, _, change in events:
        depth += change
        best = max(best, depth)
    return best


_KEYS = ["read_id", "chrom1", "pos1", "chrom2", "pos2", "strand1", "strand2"]
_CONTACT_KEYS = ["region", "cchrom", "cpos", "rchrom", "rpos", "cdir", "rdir"]
_SEEN_LIMIT = (
    "Distinct selected contacts exceed max_contacts; reduce regions or raise the explicit limit"
)
_CONTACT_LIMIT = (
    "Sparse map exceeds max_contacts; reduce windows/regions or raise the explicit limit"
)


def profile_junctions(
    manifest: str | Path | pd.DataFrame,
    viewpoint: str,
    viewpoints: Mapping[str, Sequence[tuple[int, int]]],
    chrom_sizes: Mapping[str, int],
    peaks: pd.DataFrame | None = None,
    *,
    reporter_flank: int = 100,
    capture_flank: int = 50,
    exclusions: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    allow_unoriented: bool = False,
    deduplicate: bool = True,
    orientation_source: Literal["mccnado", "unknown"] = "mccnado",
    max_profile_bases: int = 2_000_000,
    max_contacts: int = 2_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Profile actual junction ends from a selected capture pool in one pass/sample.

    Both ends contribute to reporter-region profiles when present there. Maps
    require an actual endpoint in a capture window, never an inferred oligo point.
    Capture profiles count each map-eligible event once over the union of reporter
    windows; overlapping reporter windows retain separate descriptive profiles.
    Exact repeated read-ID/contact rows are collapsed by default, not parent IDs.
    """
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0
        for x in (reporter_flank, capture_flank)
    ):
        raise ValueError("Junction flanks must be nonnegative integers")
    if max_profile_bases < 1 or max_contacts < 1:
        raise ValueError("Junction output limits must be positive")
    if orientation_source not in {"mccnado", "unknown"}:
        raise ValueError("orientation_source must be mccnado or unknown")
    if not chrom_sizes or any(
        isinstance(v, bool) or int(v) != v or v <= 0 for v in chrom_sizes.values()
    ):
        raise ValueError("Chromosome lengths must be positive integers")
    samples = (
        validate_manifest_frame(manifest, viewpoint)
        if isinstance(manifest, pd.DataFrame)
        else load_manifest(manifest, viewpoint)
    )
    capture, reporter = _regions(viewpoints, chrom_sizes, peaks, reporter_flank, capture_flank)
    regions = capture + reporter
    bases = sum(r.end - r.start for r in regions) * len(samples)
    if bases > max_profile_bases:
        raise ValueError(
            f"Requested {bases} profile bases exceeds max_profile_bases={max_profile_bases}; reduce windows/regions"
        )
    excluded = _validate_exclusions(exclusions, chrom_sizes)
    sizes = validate_chrom_sizes(chrom_sizes)
    index = ChromIndex(sizes)
    lengths = np.asarray([r.end - r.start for r in regions], dtype=np.int64)
    base = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    region_start = np.asarray([r.start for r in regions], dtype=np.int64)
    ids = list(range(len(regions)))
    capture_table = _region_table(capture, ids[: len(capture)], index)
    reporter_table = _region_table(reporter, ids[len(capture) :], index)
    reporter_union = _coded_intervals(
        {
            chrom: [(r.start, r.end) for r in reporter if r.chrom == chrom]
            for chrom in {r.chrom for r in reporter}
        },
        index,
    )
    excluded_coded = _coded_intervals(excluded, index)
    contacts_per_row = 2 * max(1, _max_depth(reporter))
    store = np.asarray([_STORE[name] for name in DIRECTION_NAMES], dtype=np.int64)
    pool_chrom, pool_oligos = _viewpoint(viewpoints, chrom_sizes)
    pool_code = index.codes[pool_chrom]
    profiles, contact_frames, all_qc = [], [], []
    distinct_contacts = 0
    started = time.perf_counter()
    for sample_index, sample in enumerate(samples.itertuples(index=False)):
        qc = {
            "sample": sample.sample,
            "source_pool_contacts": 0,
            "non_pool_annotation_contacts": 0,
            "duplicate_contacts": 0,
            "selected_contacts": 0,
            "selected_junction_ends": 0,
            "map_eligible_contacts": 0,
            "selected_without_capture_endpoint": 0,
            "unoriented_contacts": 0,
            "excluded_contacts": 0,
        }
        profile = np.zeros(int(lengths.sum()) * 3, dtype=np.int64)
        seen = None
        seen_count = 0
        sample_contacts = None
        parser_qc = {}
        for chunk in PairReader(sample.pairs, chrom_sizes, qc=parser_qc):
            frame = chunk.frame
            side = frame["side"].to_numpy()
            meta_chrom = frame["meta_chrom"].to_numpy()
            meta_start, meta_end = frame["meta_start"].to_numpy(), frame["meta_end"].to_numpy()
            in_pool = np.zeros(len(frame), dtype=bool)
            for start, end in pool_oligos:
                in_pool |= (meta_start < end) & (start < meta_end)
            non_pool = (side >= 0) & ~((meta_chrom == pool_code) & in_pool)
            qc["non_pool_annotation_contacts"] += int(non_pool.sum())
            qc["source_pool_contacts"] += int(len(frame) - non_pool.sum())
            c1, p1 = frame["chrom1"].to_numpy(), frame["pos1"].to_numpy()
            c2, p2 = frame["chrom2"].to_numpy(), frame["pos2"].to_numpy()
            dropped = in_chrom_intervals(c1, p1, excluded_coded) | in_chrom_intervals(
                c2, p2, excluded_coded
            )
            qc["excluded_contacts"] += int(np.sum(dropped & ~non_pool))
            hit = in_chrom_intervals(c1, p1, reporter_union) | in_chrom_intervals(
                c2, p2, reporter_union
            )
            keep = ~non_pool & ~dropped & hit
            if not keep.any():
                continue
            selected = frame.filter(pl.Series(keep))
            n = selected.height
            # Errors, in the order a row-by-row pass would meet them.
            failure, message = n, None
            if deduplicate:
                keys = selected.select(_KEYS)
                fresh = keys.select(pl.struct(_KEYS).is_first_distinct()).to_series().to_numpy()
                if seen is not None:
                    repeated = (
                        keys.with_row_index("row")
                        .join(seen, on=_KEYS, how="semi")["row"]
                        .to_numpy()
                    )
                    fresh[repeated] = False
                if fresh.sum() + seen_count > max_contacts:
                    failure = int(np.flatnonzero(fresh)[max_contacts - seen_count])
                    message = _SEEN_LIMIT
            else:
                fresh = np.ones(n, dtype=bool)
            side = selected["side"].to_numpy()
            dir1, dir2 = junction_directions(
                side,
                selected["strand1"].to_numpy(),
                selected["strand2"].to_numpy(),
                orientation_source,
            )
            if orientation_source == "mccnado" and chunk.shape in {
                "upper triangle",
                "lower triangle",
            }:
                flipped = np.flatnonzero(
                    fresh & ((side == SIDE_CODES["start"]) | (side == SIDE_CODES["end"]))
                )
                if len(flipped) and flipped[0] < failure:
                    failure, message = int(flipped[0]), _SHAPE_ERROR
            unoriented = fresh & ((dir1 == UNKNOWN) | (dir2 == UNKNOWN))
            if not allow_unoriented and orientation_source != "unknown":
                rows = np.flatnonzero(unoriented)
                if len(rows) and rows[0] < failure:
                    failure = int(rows[0])
                    message = (
                        f"Cannot orient {selected['read_id'][failure]!r}; supply MCCNado "
                        "names/strands or explicitly allow unoriented junctions"
                    )
            processed = fresh.copy()
            processed[failure:] = False
            qc["duplicate_contacts"] += int(np.sum(~fresh[:failure]))
            qc["unoriented_contacts"] += int(np.sum(unoriented[:failure]))
            if deduplicate:
                new_keys = keys.filter(pl.Series(processed))
                seen = new_keys if seen is None else pl.concat([seen, new_keys], rechunk=False)
                seen_count += new_keys.height
            rows = np.flatnonzero(processed)
            buffer = np.empty((len(rows) * contacts_per_row, 8), dtype=np.int64)
            written, junction_ends, mapped = _core.footprint_rows(
                c1[keep][rows].astype(np.int64),
                p1[keep][rows].astype(np.int64),
                dir1[rows],
                c2[keep][rows].astype(np.int64),
                p2[keep][rows].astype(np.int64),
                dir2[rows],
                reporter_table,
                capture_table,
                region_start,
                base,
                store,
                profile,
                buffer,
            )
            qc["selected_contacts"] += len(rows)
            qc["selected_junction_ends"] += int(junction_ends)
            qc["map_eligible_contacts"] += int(mapped)
            qc["selected_without_capture_endpoint"] += len(rows) - int(mapped)
            if written:
                emitted = pl.DataFrame(buffer[:written], schema=["row", *_CONTACT_KEYS])
                grouped = emitted.group_by(_CONTACT_KEYS, maintain_order=True).agg(
                    pl.len().alias("count")
                )
                if sample_contacts is not None:
                    grouped_new = grouped.join(sample_contacts, on=_CONTACT_KEYS, how="anti")
                else:
                    grouped_new = grouped
                if distinct_contacts + grouped_new.height > max_contacts:
                    raise ValueError(_CONTACT_LIMIT)
                distinct_contacts += grouped_new.height
                merged = grouped.select([*_CONTACT_KEYS, "count"])
                sample_contacts = (
                    merged
                    if sample_contacts is None
                    else pl.concat([sample_contacts, merged])
                    .group_by(_CONTACT_KEYS)
                    .agg(pl.col("count").sum())
                )
            if message is not None:
                raise ValueError(message)
        counts = profile.reshape(-1, 3)
        names = np.repeat(np.asarray([r.name for r in regions], dtype=object), lengths)
        offsets = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(base, lengths)
        profiles.append(
            pd.DataFrame(
                {
                    "Sample": np.full(len(names), str(sample.sample), dtype=object),
                    "Region": names,
                    "Chromosome": np.repeat(
                        np.asarray([r.chrom for r in regions], dtype=object), lengths
                    ),
                    "Position": offsets + np.repeat(region_start, lengths),
                    "RegionStart": np.repeat(region_start, lengths),
                    "RegionEnd": np.repeat(
                        np.asarray([r.end for r in regions], dtype=np.int64), lengths
                    ),
                    "Upstream": counts[:, 0],
                    "Downstream": counts[:, 1],
                    "Unknown": counts[:, 2],
                    "Total": counts.sum(axis=1),
                }
            )
        )
        if sample_contacts is not None:
            contact_frames.append(
                sample_contacts.with_columns(pl.lit(sample_index).alias("sample"))
            )
        qc["parser"] = parser_qc
        qc["warnings"] = []
        if qc["selected_contacts"] and not qc["map_eligible_contacts"]:
            qc["warnings"].append(STALE_FORMAT_WARNING)
            logger.warning("Sample {}: {}", sample.sample, STALE_FORMAT_WARNING)
        all_qc.append(qc)
    profile_frame = pd.concat(profiles, ignore_index=True).reindex(columns=PROFILE_COLUMNS)
    contact_frame = _contact_frame(contact_frames, samples, regions, index)
    region_frame = pd.DataFrame(
        [(r.name, r.kind, r.chrom, r.start, r.end) for r in regions], columns=REGION_COLUMNS
    )
    metrics = {
        "samples": all_qc,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mb": peak_memory_mb(),
        "coordinate_system": "0-based supplied junction points",
        "profile_unit": "junction ends per region; both contact endpoints may contribute",
        "map_unit": "contacts per reporter region; one deterministic assignment per contact/region",
        "deduplication": "exact read ID and ordered endpoints" if deduplicate else "none",
        "orientation_source": orientation_source,
        "capture_mode": "tiled enrichment pool",
        "footprint_significance": "not estimated; descriptive profiles only",
    }
    return profile_frame, contact_frame, region_frame, metrics


def _contact_frame(frames, samples, regions, index: ChromIndex) -> pd.DataFrame:
    """Decode coded contacts, ordered as sorting the name tuples would order them."""
    if not frames:
        return pd.DataFrame([], columns=CONTACT_COLUMNS)
    table = pl.concat(frames)
    sample_names = np.asarray([str(s) for s in samples["sample"]], dtype=object)
    region_names = np.asarray([r.name for r in regions], dtype=object)
    sample_rank = np.argsort(np.argsort(sample_names.astype(str), kind="stable"))
    region_rank = np.argsort(np.argsort(region_names.astype(str), kind="stable"))
    columns = {name: table[name].to_numpy() for name in table.columns}
    order = np.lexsort(
        (
            columns["rdir"],
            columns["cdir"],
            columns["rpos"],
            columns["rchrom"],
            columns["cpos"],
            columns["cchrom"],
            region_rank[columns["region"]],
            sample_rank[columns["sample"]],
        )
    )
    chroms = np.asarray(index.names, dtype=object)
    directions = np.asarray(DIRECTION_NAMES, dtype=object)
    return pd.DataFrame(
        {
            "Sample": sample_names[columns["sample"][order]],
            "Region": region_names[columns["region"][order]],
            "CaptureChromosome": chroms[columns["cchrom"][order]],
            "CapturePosition": columns["cpos"][order].astype(np.int64),
            "ReporterChromosome": chroms[columns["rchrom"][order]],
            "ReporterPosition": columns["rpos"][order].astype(np.int64),
            "CaptureDirection": directions[columns["cdir"][order]],
            "ReporterDirection": directions[columns["rdir"][order]],
            "Count": columns["count"][order].astype(np.int64),
        },
        columns=CONTACT_COLUMNS,
    )
