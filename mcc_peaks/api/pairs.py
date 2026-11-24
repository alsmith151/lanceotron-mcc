"""Columnar reader for standard ``.pairs`` files.

Files are decompressed in large newline-aligned blocks on a background thread,
and each block is parsed and validated as columns with polars and numpy. Any
invalid line is located by replaying the per-line rules on that block only, so
error messages and their ordering match a line-by-line reader exactly: the
valid rows before the first bad line are yielded first, then the error is
raised.
"""

from __future__ import annotations

import gzip
import queue
import re
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

PAIR_COLUMNS = ("readid", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2")
_PAIR_COLUMN_ALIASES = {
    "readID": "readid",
    "read_id": "readid",
    "chrom1": "chr1",
    "chrom2": "chr2",
}
_MCC_SIDES = {"start", "end", "all", "none"}
SIDE_CODES = {"start": 0, "end": 1, "all": 2, "none": 3}
NO_SIDE = -1
UNMAPPED = -1
BLOCK_BYTES = 64 << 20
_LOCATION = re.compile(r"(.+)-(-?\d+)-(-?\d+)")
_INT64_MAX = np.iinfo(np.int64).max
_RAW = [f"c{i}" for i in range(7)]


@dataclass(frozen=True)
class MCCMetadata:
    parent_id: str
    viewpoint: str
    chromosome: str
    start: int
    end: int
    side: str
    read_number: int
    flashed_status: int


def new_pairs_qc(path: str | Path) -> dict[str, int | str]:
    """Create a QC mapping filled by :class:`PairReader` and ``count_pairs``."""
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


def validate_chrom_sizes(chrom_sizes: Mapping[str, int]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, size in chrom_sizes.items():
        if isinstance(size, bool) or int(size) != size or int(size) <= 0:
            raise ValueError(f"Invalid chromosome size for {name!r}: {size!r}")
        sizes[str(name)] = int(size)
    return sizes


class ChromIndex:
    """Chromosome codes in lexicographic name order, so code order is name order."""

    def __init__(self, sizes: Mapping[str, int]):
        self.names = sorted(sizes)
        self.codes = {name: i for i, name in enumerate(self.names)}
        self.lengths = np.asarray([sizes[name] for name in self.names], dtype=np.int64)

    def code(self, name: str) -> int:
        return self.codes.get(name, -2)


def _parse_location(location: str, read_id: str, ordered: list[str], sizes: Mapping[str, int]):
    match = _LOCATION.fullmatch(location)
    if match is None:
        raise ValueError(f"Invalid MCC read-name coordinates in {read_id!r}")
    chrom_and_viewpoint, raw_start, raw_end = match.groups()
    start, end = int(raw_start), int(raw_end)
    if start < 0 or start >= end:
        raise ValueError(f"Invalid MCC viewpoint interval in read name {read_id!r}: {start}-{end}")
    resolved: tuple[str, str] | None = None
    for chromosome in ordered:
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
    if end > int(sizes[chromosome]):
        raise ValueError(
            f"MCC viewpoint interval {chromosome}:{start}-{end} in read name {read_id!r} "
            f"exceeds chromosome length {sizes[chromosome]}"
        )
    return viewpoint, chromosome, start, end


def _suffix_order(sizes: Mapping[str, int]) -> list[str]:
    return sorted(sizes, key=len, reverse=True)


def parse_mcc_read_id(read_id: str, chrom_sizes: Mapping[str, int]) -> MCCMetadata | None:
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
    viewpoint, chromosome, start, end = _parse_location(
        location, read_id, _suffix_order(chrom_sizes), chrom_sizes
    )
    return MCCMetadata(
        parent_id, viewpoint, chromosome, start, end, side, int(read_number), int(flashed_status)
    )


def parse_pair_columns(line: str, path: Path, line_number: int) -> int:
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


def _line_error(
    line: str,
    line_number: int,
    path: Path,
    declared_width: int | None,
    sizes: Mapping[str, int],
    ordered: list[str],
) -> ValueError | None:
    """Apply the per-line rules to one data line; return its first error."""
    fields = line.split("\t")
    if len(fields) < 7:
        return ValueError(
            f"{path}:{line_number}: expected at least 7 tab-separated columns, found {len(fields)}"
        )
    if declared_width is not None and len(fields) != declared_width:
        return ValueError(
            f"{path}:{line_number}: row has {len(fields)} fields but "
            f"#columns header declares {declared_width}"
        )
    read_id, chr1, raw_pos1, chr2, raw_pos2 = fields[:5]
    parts = read_id.rsplit("__", 4)
    if (
        len(parts) == 5
        and parts[2] in _MCC_SIDES
        and parts[3] in {"1", "2", "3"}
        and parts[4] in {"0", "1"}
    ):
        try:
            _parse_location(parts[1], read_id, ordered, sizes)
        except ValueError as exc:
            return exc
    try:
        pos1, pos2 = int(raw_pos1), int(raw_pos2)
    except ValueError:
        return ValueError(f"{path}:{line_number}: pair positions must be integers")
    for pair_chrom, position in ((chr1, pos1), (chr2, pos2)):
        if pair_chrom == "!":
            if position != 0:
                return ValueError(
                    f"{path}:{line_number}: unmapped chromosome '!' must have position 0"
                )
            continue
        if pair_chrom not in sizes:
            return ValueError(f"{path}:{line_number}: no chromosome size for {pair_chrom!r}")
        if position < 1 or position > sizes[pair_chrom]:
            return ValueError(
                f"{path}:{line_number}: one-based position {position} is outside "
                f"{pair_chrom} bounds 1..{sizes[pair_chrom]}"
            )
    return None


def _mcc_suffix(read_ids: pl.Series) -> tuple[pl.Series, pl.Series]:
    """Side and location of ``read_id.rsplit("__", 4)`` for valid MCC suffixes, else null.

    The fixed-width flash/read-number tail and side token are checked with plain
    string slices. Without ``___`` runs, ``split("__")`` and ``rsplit`` agree on
    the last piece; the rare IDs containing ``___`` use Python ``rsplit``.
    """
    frame = pl.DataFrame({"id": read_ids})
    ids = pl.col("id")
    head = ids.str.head(-6)
    side = pl.lit(None, dtype=pl.String)
    rest = pl.lit(None, dtype=pl.String)
    for token in ("start", "end", "all", "none"):
        matched = head.str.ends_with(f"__{token}")
        side = pl.when(matched).then(pl.lit(token)).otherwise(side)
        rest = pl.when(matched).then(head.str.head(-(len(token) + 2))).otherwise(rest)
    valid = ids.str.slice(-3).is_in(["__0", "__1"]) & ids.str.slice(-6, 3).is_in(
        ["__1", "__2", "__3"]
    )
    frame = frame.select(
        pl.when(valid).then(side).alias("side"),
        pl.when(valid).then(rest).alias("rest"),
    )
    rest = frame.get_column("rest")
    location = (
        pl.when(rest.str.contains("__", literal=True))
        .then(rest.str.split("__").list.last())
        .otherwise(None)
    )
    location = frame.select(location.alias("location")).get_column("location")
    side = frame.get_column("side").set(location.is_null(), None)
    tricky = rest.str.contains("___", literal=True).fill_null(False)
    if tricky.any():
        values = read_ids.filter(tricky).to_list()
        exact = []
        for value in values:
            parts = value.rsplit("__", 4)
            exact.append(parts[1] if len(parts) == 5 else None)
        index = np.flatnonzero(tricky.to_numpy())
        location = location.scatter(index, pl.Series(exact, dtype=pl.String))
        side = side.scatter(
            index,
            pl.Series(
                [None if loc is None else s for loc, s in zip(exact, side.gather(index))],
                dtype=pl.String,
            ),
        )
    return side, location


def _blocks(path: Path, block_bytes: int) -> Iterator[bytes]:
    """Yield newline-terminated blocks with universal newlines normalised to ``\\n``."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as handle:
        rest = b""
        while True:
            chunk = handle.read(block_bytes)
            if not chunk:
                break
            buffer = rest + chunk
            held = b""
            if b"\r" in buffer:
                if buffer.endswith(b"\r"):
                    buffer, held = buffer[:-1], b"\r"
                buffer = buffer.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            cut = buffer.rfind(b"\n") + 1
            if cut:
                yield buffer[:cut]
            rest = buffer[cut:] + held
        if rest:
            rest = rest.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            yield rest if rest.endswith(b"\n") else rest + b"\n"


def _prefetch(iterator: Iterator[bytes], depth: int = 2) -> Iterator[bytes]:
    """Decompress ahead on a worker thread while the caller parses."""
    slots: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def work():
        try:
            for item in iterator:
                if stop.is_set():
                    return
                slots.put(item)
        except BaseException as exc:  # forwarded to the consumer
            slots.put(exc)
        slots.put(done)

    thread = threading.Thread(target=work, name="pairs-read", daemon=True)
    thread.start()
    try:
        while True:
            item = slots.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while thread.is_alive():
            try:
                slots.get_nowait()
            except queue.Empty:
                thread.join(0.01)


@dataclass
class PairChunk:
    """Mapped records in file order.

    Columns: ``read_id`` (str), ``chrom1``/``chrom2`` (Int32 :class:`ChromIndex`
    codes), zero-based ``pos1``/``pos2`` (Int64), ``strand1``/``strand2`` (str),
    ``side`` (Int8 :data:`SIDE_CODES`, -1 without MCC metadata) and the read-name
    oligo ``meta_chrom`` (Int32 code, -1 if absent), ``meta_start``, ``meta_end``.
    """

    frame: pl.DataFrame
    shape: str | None


class PairReader:
    """Stream validated mapped records from a standard pairs file as columns.

    The first seven standard fields are required; additional fields are ignored.
    Returned positions are zero-based. Unmapped records are validated, counted in
    QC and skipped. MCC read-name metadata is parsed when its strict suffix schema
    is present.
    """

    def __init__(
        self,
        path: str | Path,
        chrom_sizes: Mapping[str, int],
        *,
        qc: dict[str, int | str] | None = None,
        block_bytes: int | None = None,
    ):
        self.sizes = validate_chrom_sizes(chrom_sizes)
        self.index = ChromIndex(self.sizes)
        self.path = Path(path)
        self.qc = qc if qc is not None else new_pairs_qc(path)
        for key, value in new_pairs_qc(path).items():
            self.qc.setdefault(key, value)
        self.block_bytes = int(block_bytes or BLOCK_BYTES)
        self._ordered = _suffix_order(self.sizes)
        self._locations: dict[str, tuple[int, int, int] | None] = {}
        self._declared: int | None = None
        self._shape: str | None = None
        self._line = 0
        chrom_map = dict(self.index.codes)
        chrom_map["!"] = UNMAPPED
        self._chrom_map = chrom_map

    def __iter__(self) -> Iterator[PairChunk]:
        try:
            for block in _prefetch(_blocks(self.path, self.block_bytes)):
                yield from self._block(block)
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Could not read pair file {self.path}: {exc}") from exc

    # -- block structure -------------------------------------------------
    def _block(self, block: bytes) -> Iterator[PairChunk]:
        data = np.frombuffer(block, dtype=np.uint8)
        ends = np.flatnonzero(data == 10)
        starts = np.empty_like(ends)
        starts[0] = 0
        starts[1:] = ends[:-1] + 1
        special = (ends == starts) | (data[np.minimum(starts, len(data) - 1)] == 35)
        special_lines = np.flatnonzero(special)
        first_line = self._line
        self._line += len(ends)
        previous = 0
        for index in [*special_lines.tolist(), len(ends)]:
            if index > previous:
                yield from self._segment(
                    block[starts[previous] : ends[index - 1] + 1],
                    first_line + previous + 1,
                    index - previous,
                )
            if index < len(ends) and ends[index] > starts[index]:
                self._header(
                    block[starts[index] : ends[index]].decode("utf-8"), first_line + index + 1
                )
            previous = index + 1

    def _header(self, line: str, line_number: int) -> None:
        if line.startswith("#columns:"):
            if self._declared is not None:
                raise ValueError(f"{self.path}:{line_number}: duplicate #columns header")
            self._declared = parse_pair_columns(line, self.path, line_number)
        elif line.startswith("#shape:"):
            if self._shape is not None:
                raise ValueError(f"{self.path}:{line_number}: duplicate #shape header")
            self._shape = line[len("#shape:") :].strip().lower()

    # -- data lines ------------------------------------------------------
    def _first_bad_line(self, segment: bytes, first_number: int) -> tuple[int, ValueError]:
        lines = segment.decode("utf-8").split("\n")[:-1]
        for offset, line in enumerate(lines):
            error = _line_error(
                line, first_number + offset, self.path, self._declared, self.sizes, self._ordered
            )
            if error is not None:
                return offset, error
        raise AssertionError("columnar pair validation disagreed with line validation")

    def _fail(self, segment: bytes, first_number: int, n_lines: int) -> Iterator[PairChunk]:
        offset, error = self._first_bad_line(segment, first_number)
        if offset:
            cut = 0
            for _ in range(offset):
                cut = segment.index(b"\n", cut) + 1
            yield from self._segment(segment[:cut], first_number, offset)
        raise error

    def _location(self, location: str) -> tuple[int, int, int] | None:
        if location not in self._locations:
            try:
                _, chromosome, start, end = _parse_location(location, "", self._ordered, self.sizes)
            except ValueError:
                self._locations[location] = None
            else:
                self._locations[location] = (self.index.codes[chromosome], start, end)
        return self._locations[location]

    def _positions(self, raw: pl.Series) -> pl.Series | None:
        parsed = raw.str.to_integer(strict=False)
        if not parsed.null_count():
            return parsed
        # polars accepts a subset of Python int() syntax; resolve the rest exactly.
        values = {}
        for value in raw.filter(parsed.is_null()).unique().to_list():
            try:
                number = int(value)
            except ValueError:
                return None
            if abs(number) > _INT64_MAX:
                return None
            values[value] = number
        fallback = raw.replace_strict(values, default=None, return_dtype=pl.Int64)
        return parsed.fill_null(fallback)

    def _segment(self, segment: bytes, first_number: int, n_lines: int) -> Iterator[PairChunk]:
        data = np.frombuffer(segment, dtype=np.uint8)
        line_ends = np.flatnonzero(data == 10)
        tabs = np.flatnonzero(data == 9)
        fields = np.diff(np.searchsorted(tabs, line_ends), prepend=0) + 1
        if np.any(fields < 7) or (self._declared is not None and np.any(fields != self._declared)):
            yield from self._fail(segment, first_number, n_lines)
            return
        try:
            raw = pl.read_csv(
                segment,
                has_header=False,
                separator="\t",
                quote_char=None,
                schema={name: pl.String for name in _RAW},
                truncate_ragged_lines=True,
            ).fill_null("")
        except Exception:
            yield from self._fail(segment, first_number, n_lines)
            return
        if raw.height != n_lines:
            yield from self._fail(segment, first_number, n_lines)
            return
        side, location = _mcc_suffix(raw.get_column("c0"))
        has_meta = location.is_not_null()
        meta_chrom = pl.Series(np.full(n_lines, -1, dtype=np.int32))
        meta_start = pl.Series(np.zeros(n_lines, dtype=np.int64))
        meta_end = pl.Series(np.zeros(n_lines, dtype=np.int64))
        n_meta = int(has_meta.sum())
        if n_meta:
            uniques = location.drop_nulls().unique().to_list()
            parsed = {value: self._location(value) for value in uniques}
            if any(value is None for value in parsed.values()):
                yield from self._fail(segment, first_number, n_lines)
                return
            meta_chrom = location.replace_strict(
                {k: v[0] for k, v in parsed.items()}, default=-1, return_dtype=pl.Int32
            ).fill_null(-1)
            meta_start = location.replace_strict(
                {k: v[1] for k, v in parsed.items()}, default=0, return_dtype=pl.Int64
            ).fill_null(0)
            meta_end = location.replace_strict(
                {k: v[2] for k, v in parsed.items()}, default=0, return_dtype=pl.Int64
            ).fill_null(0)
        pos1 = self._positions(raw.get_column("c2"))
        pos2 = self._positions(raw.get_column("c4"))
        if pos1 is None or pos2 is None:
            yield from self._fail(segment, first_number, n_lines)
            return
        chrom1 = raw.get_column("c1").replace_strict(
            self._chrom_map, default=None, return_dtype=pl.Int32
        )
        chrom2 = raw.get_column("c3").replace_strict(
            self._chrom_map, default=None, return_dtype=pl.Int32
        )
        if chrom1.null_count() or chrom2.null_count():
            yield from self._fail(segment, first_number, n_lines)
            return
        lengths = self.index.lengths
        c1, c2 = chrom1.to_numpy(), chrom2.to_numpy()
        p1, p2 = pos1.to_numpy(), pos2.to_numpy()
        unmapped1, unmapped2 = c1 == UNMAPPED, c2 == UNMAPPED
        bad = (
            (unmapped1 & (p1 != 0))
            | (unmapped2 & (p2 != 0))
            | (~unmapped1 & ((p1 < 1) | (p1 > lengths[np.maximum(c1, 0)])))
            | (~unmapped2 & ((p2 < 1) | (p2 > lengths[np.maximum(c2, 0)])))
        )
        if np.any(bad):
            yield from self._fail(segment, first_number, n_lines)
            return
        qc = self.qc
        qc["input_pairs"] += n_lines
        side_codes = side.replace_strict(
            SIDE_CODES, default=NO_SIDE, return_dtype=pl.Int8
        ).fill_null(NO_SIDE)
        if n_meta:
            qc["mcc_annotated_pairs"] += n_meta
            counts = np.bincount(side_codes.to_numpy()[has_meta.to_numpy()], minlength=4)
            for name, code in SIDE_CODES.items():
                qc[f"mcc_{name}_pairs"] += int(counts[code])
        unmapped = unmapped1 | unmapped2
        n_unmapped = int(unmapped.sum())
        qc["unmapped_pairs"] += n_unmapped
        frame = pl.DataFrame(
            {
                "read_id": raw.get_column("c0"),
                "chrom1": chrom1,
                "pos1": pos1 - 1,
                "chrom2": chrom2,
                "pos2": pos2 - 1,
                "strand1": raw.get_column("c5"),
                "strand2": raw.get_column("c6"),
                "side": side_codes,
                "meta_chrom": meta_chrom,
                "meta_start": meta_start,
                "meta_end": meta_end,
            }
        )
        if n_unmapped:
            frame = frame.filter(pl.Series(~unmapped))
        if frame.height:
            yield PairChunk(frame, self._shape)


def merge_intervals(intervals) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, merged half-open intervals as start/end arrays."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    if not merged:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    array = np.asarray(merged, dtype=np.int64)
    return array[:, 0].copy(), array[:, 1].copy()


def in_intervals(positions: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Membership of positions in merged half-open intervals."""
    if not len(starts):
        return np.zeros(len(positions), dtype=bool)
    index = np.searchsorted(starts, positions, side="right") - 1
    return (index >= 0) & (positions < ends[np.maximum(index, 0)])


def in_chrom_intervals(
    chroms: np.ndarray,
    positions: np.ndarray,
    intervals: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Membership for positions on coded chromosomes; ``intervals`` keyed by code."""
    result = np.zeros(len(positions), dtype=bool)
    for code, (starts, ends) in intervals.items():
        on = chroms == code
        if on.any():
            result[on] = in_intervals(positions[on], starts, ends)
    return result
