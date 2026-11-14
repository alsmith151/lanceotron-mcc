"""Base-pair junction-end profiles and sparse maps for captured MCC pools.

Capture oligo names identify enrichment provenance in tiled experiments. They
are not substituted for genomic endpoints or required to touch every contact.
"""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from .adaptive import _validate_exclusions, _viewpoint
from .counts import PairRecord, iter_pairs, load_manifest, validate_manifest_frame
from .io import peak_memory_mb

PROFILE_COLUMNS = ["Sample", "Region", "Chromosome", "Position", "RegionStart", "RegionEnd",
                   "Upstream", "Downstream", "Unknown", "Total"]
CONTACT_COLUMNS = ["Sample", "Region", "CaptureChromosome", "CapturePosition", "ReporterChromosome",
                   "ReporterPosition", "CaptureDirection", "ReporterDirection", "Count"]
REGION_COLUMNS = ["Region", "Kind", "Chromosome", "Start", "End"]
_DIRECTIONS = {"upstream": 0, "downstream": 1, "unknown": 2}
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


def junction_directions(record: PairRecord, source: str = "mccnado") -> tuple[str, str]:
    """Fragment extension from the junction in endpoint 1 / endpoint 2 order.

    MCCNado stores the named reporter as endpoint 1 before optional external
    flipping. `start` is a right reporter segment; `end` is a left segment.
    The second fragment has the complementary ligated endpoint in read order.
    This uses both segment side and strand, not strand alone.
    """
    if source not in {"mccnado", "unknown"}:
        raise ValueError("orientation_source must be mccnado or unknown")
    if source == "unknown":
        return "unknown", "unknown"
    metadata = record.metadata
    if metadata is None or metadata.side not in {"start", "end"}:
        return "unknown", "unknown"
    if getattr(record, "shape", None) in {"upper triangle", "lower triangle"}:
        raise ValueError("Flipped .pairs do not preserve MCC reporter order; use orientation_source='unknown' or unflipped input")
    def direction(strand, positive_downstream):
        if strand not in {"+", "-"}:
            return "unknown"
        return "downstream" if (strand == "+") == positive_downstream else "upstream"
    return (direction(record.strand1, metadata.side == "start"),
            direction(record.strand2, metadata.side == "end"))


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
        raise ValueError(f"Peak BED chromosomes are absent from chromosome sizes: {', '.join(missing)}")
    return pd.DataFrame({"Chromosome": frame[0].to_numpy(), "Start": frame[1].astype(np.int64).to_numpy(),
                         "End": frame[2].astype(np.int64).to_numpy(), "Name": names})


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
    capture = [JunctionRegion(f"capture:oligo_{i+1}", "capture", chrom, start, end)
               for i, (start, end) in enumerate(_merge([
                   (max(0, start - capture_flank), min(chrom_sizes[chrom], end + capture_flank + 1))
                   for start, end in oligos]))]
    if peaks is None:
        peaks = pd.DataFrame([(chrom, a, b, f"tile_{i+1}") for i, (a, b) in enumerate(_merge(oligos))],
                             columns=["Chromosome", "Start", "End", "Name"])
    if not {"Chromosome", "Start", "End"}.issubset(peaks):
        raise ValueError("Peak regions require Chromosome, Start and End columns")
    reporter = []
    used = set()
    for i, row in enumerate(peaks.itertuples(index=False)):
        name = str(getattr(row, "Name", f"peak_{i+1}"))
        identifier = f"reporter:{name}"
        if not name or identifier in used:
            raise ValueError("Reporter region names must be nonempty and unique")
        used.add(identifier)
        start, end = row.Start, row.End
        if row.Chromosome not in chrom_sizes or isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Reporter region chromosome or coordinates are invalid")
        if (not np.isfinite(start) or not np.isfinite(end) or int(start) != start or int(end) != end
                or not 0 <= start < end <= chrom_sizes[row.Chromosome]):
            raise ValueError("Reporter regions must be positive-width integer BED intervals within chromosome bounds")
        reporter.append(JunctionRegion(identifier, "reporter", row.Chromosome,
                        max(0, int(start) - reporter_flank), min(chrom_sizes[row.Chromosome], int(end) + reporter_flank)))
    if not reporter:
        raise ValueError("At least one reporter region is required")
    return capture, reporter


class _RegionIndex:
    """Point lookup with prefix end maxima; supports overlapping peak windows."""
    def __init__(self, regions):
        self.by_chrom = {}
        for chrom in {r.chrom for r in regions}:
            entries = sorted((r for r in regions if r.chrom == chrom), key=lambda r: r.start)
            starts = np.asarray([r.start for r in entries], dtype=np.int64)
            max_ends = np.maximum.accumulate([r.end for r in entries])
            self.by_chrom[chrom] = entries, starts, max_ends

    def at(self, chrom, position):
        if chrom not in self.by_chrom:
            return []
        entries, starts, max_ends = self.by_chrom[chrom]
        i = int(np.searchsorted(starts, position, side="right")) - 1
        result = []
        while i >= 0 and max_ends[i] > position:
            if position < entries[i].end:
                result.append(entries[i])
            i -= 1
        return result


def profile_junctions(
    manifest: str | Path | pd.DataFrame,
    viewpoint: str,
    viewpoints: Mapping[str, Sequence[tuple[int, int]]],
    chrom_sizes: Mapping[str, int],
    peaks: pd.DataFrame | None = None,
    *, reporter_flank: int = 100, capture_flank: int = 50,
    exclusions: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    allow_unoriented: bool = False, deduplicate: bool = True,
    orientation_source: Literal["mccnado", "unknown"] = "mccnado",
    max_profile_bases: int = 2_000_000, max_contacts: int = 2_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Profile actual junction ends from a selected capture pool in one pass/sample.

    Both ends contribute to reporter-region profiles when present there. Maps
    require an actual endpoint in a capture window, never an inferred oligo point.
    Capture profiles count each map-eligible event once over the union of reporter
    windows; overlapping reporter windows retain separate descriptive profiles.
    Exact repeated read-ID/contact rows are collapsed by default, not parent IDs.
    """
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (reporter_flank, capture_flank)):
        raise ValueError("Junction flanks must be nonnegative integers")
    if max_profile_bases < 1 or max_contacts < 1:
        raise ValueError("Junction output limits must be positive")
    if orientation_source not in {"mccnado", "unknown"}:
        raise ValueError("orientation_source must be mccnado or unknown")
    if not chrom_sizes or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in chrom_sizes.values()):
        raise ValueError("Chromosome lengths must be positive integers")
    samples = (validate_manifest_frame(manifest, viewpoint) if isinstance(manifest, pd.DataFrame)
               else load_manifest(manifest, viewpoint))
    capture, reporter = _regions(viewpoints, chrom_sizes, peaks, reporter_flank, capture_flank)
    regions = capture + reporter
    bases = sum(r.end - r.start for r in regions) * len(samples)
    if bases > max_profile_bases:
        raise ValueError(f"Requested {bases} profile bases exceeds max_profile_bases={max_profile_bases}; reduce windows/regions")
    excluded = _validate_exclusions(exclusions, chrom_sizes)
    capture_index, reporter_index = _RegionIndex(capture), _RegionIndex(reporter)
    profiles, contacts, all_qc = [], Counter(), []
    pool_chrom, pool_oligos = _viewpoint(viewpoints, chrom_sizes)
    started = time.perf_counter()
    for sample in samples.itertuples(index=False):
        qc = {"sample": sample.sample, "source_pool_contacts": 0, "non_pool_annotation_contacts": 0,
              "duplicate_contacts": 0, "selected_contacts": 0, "selected_junction_ends": 0,
              "map_eligible_contacts": 0, "selected_without_capture_endpoint": 0,
              "unoriented_contacts": 0, "excluded_contacts": 0}
        arrays = {r.name: np.zeros((r.end - r.start, 3), dtype=np.int64) for r in regions}
        seen = set()
        parser_qc = {}
        for record in iter_pairs(sample.pairs, chrom_sizes, qc=parser_qc):
            metadata = record.metadata
            if metadata is not None and not (metadata.chromosome == pool_chrom and
                    any(metadata.start < end and start < metadata.end for start, end in pool_oligos)):
                qc["non_pool_annotation_contacts"] += 1
                continue
            qc["source_pool_contacts"] += 1
            ends = [(record.chrom1, record.pos1), (record.chrom2, record.pos2)]
            if any(any(start <= pos < end for start, end in excluded.get(chrom, [])) for chrom, pos in ends):
                qc["excluded_contacts"] += 1
                continue
            hits = [reporter_index.at(*end) for end in ends]
            if not any(hits):
                continue
            key = (record.read_id, record.chrom1, record.pos1, record.chrom2, record.pos2,
                   record.strand1, record.strand2)
            if deduplicate and key in seen:
                qc["duplicate_contacts"] += 1
                continue
            if deduplicate:
                seen.add(key)
                if len(seen) > max_contacts:
                    raise ValueError("Distinct selected contacts exceed max_contacts; reduce regions or raise the explicit limit")
            directions = junction_directions(record, orientation_source)
            if "unknown" in directions:
                qc["unoriented_contacts"] += 1
                if not allow_unoriented and orientation_source != "unknown":
                    raise ValueError(f"Cannot orient {record.read_id!r}; supply MCCNado names/strands or explicitly allow unoriented junctions")
            qc["selected_contacts"] += 1
            for i in (0, 1):
                for region in hits[i]:
                    arrays[region.name][ends[i][1] - region.start, _DIRECTIONS[directions[i]]] += 1
                    qc["selected_junction_ends"] += 1
            capture_hits = [capture_index.at(*end) for end in ends]
            mapped = False
            capture_contributions = set()
            for region in {r.name: r for group in hits for r in group}.values():
                candidates = []
                for ri, ci in ((0, 1), (1, 0)):
                    if region in hits[ri] and capture_hits[ci]:
                        candidates.append((ends[ci][0], ends[ci][1], ends[ri][0], ends[ri][1],
                                           directions[ci], directions[ri], ci))
                if not candidates:
                    continue
                cc, cp, rc, rp, cd, rd, ci = min(candidates)
                contacts[(str(sample.sample), region.name, cc, cp, rc, rp, cd, rd)] += 1
                if len(contacts) > max_contacts:
                    raise ValueError("Sparse map exceeds max_contacts; reduce windows/regions or raise the explicit limit")
                mapped = True
                for cap in capture_hits[ci]:
                    capture_contributions.add((cap.name, ci))
            for name, i in capture_contributions:
                cap = next(r for r in capture if r.name == name)
                arrays[name][ends[i][1] - cap.start, _DIRECTIONS[directions[i]]] += 1
            if mapped:
                qc["map_eligible_contacts"] += 1
            else:
                qc["selected_without_capture_endpoint"] += 1
        for region in regions:
            counts = arrays[region.name]
            profiles.append(pd.DataFrame({
                "Sample": str(sample.sample), "Region": region.name, "Chromosome": region.chrom,
                "Position": np.arange(region.start, region.end, dtype=np.int64),
                "RegionStart": region.start, "RegionEnd": region.end,
                "Upstream": counts[:, 0], "Downstream": counts[:, 1], "Unknown": counts[:, 2],
                "Total": counts.sum(axis=1),
            }))
        qc["parser"] = parser_qc
        qc["warnings"] = []
        if qc["selected_contacts"] and not qc["map_eligible_contacts"]:
            qc["warnings"].append(STALE_FORMAT_WARNING)
            logger.warning("Sample {}: {}", sample.sample, STALE_FORMAT_WARNING)
        all_qc.append(qc)
    profile_frame = pd.concat(profiles, ignore_index=True).reindex(columns=PROFILE_COLUMNS)
    contact_frame = pd.DataFrame([(*key, count) for key, count in sorted(contacts.items())], columns=CONTACT_COLUMNS)
    region_frame = pd.DataFrame([(r.name, r.kind, r.chrom, r.start, r.end) for r in regions], columns=REGION_COLUMNS)
    metrics = {"samples": all_qc, "elapsed_seconds": time.perf_counter() - started,
               "peak_memory_mb": peak_memory_mb(), "coordinate_system": "0-based supplied junction points",
               "profile_unit": "junction ends per region; both contact endpoints may contribute",
               "map_unit": "contacts per reporter region; one deterministic assignment per contact/region",
               "deduplication": "exact read ID and ordered endpoints" if deduplicate else "none",
               "orientation_source": orientation_source, "capture_mode": "tiled enrichment pool",
               "footprint_significance": "not estimated; descriptive profiles only"}
    return profile_frame, contact_frame, region_frame, metrics
