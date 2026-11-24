"""Columnar pairs reader against a line-by-line reference, and chunk invariance."""

import random

import numpy as np
import pandas as pd
import pytest

from mcc_peaks.api import pairs as pairs_api
from mcc_peaks.api.counts import count_pairs
from mcc_peaks.api.junctions import profile_junctions
from mcc_peaks.api.pairs import SIDE_CODES, PairReader, parse_mcc_read_id

SIZES = {"chr1": 1000, "chr2": 500, "chrUn_a": 300, "chr1_alt": 50}
SIDES = {code: name for name, code in SIDE_CODES.items()}


def _reference(path, sizes):
    """Straightforward per-line reading: (records, qc, error message)."""
    text = path.read_bytes().decode("utf-8")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    records, qc, width, shape = [], {"input_pairs": 0, "unmapped_pairs": 0}, None, None
    try:
        for number, line in enumerate(lines, start=1):
            if not line:
                continue
            if line.startswith("#columns:"):
                if width is not None:
                    raise ValueError(f"{path}:{number}: duplicate #columns header")
                width = pairs_api.parse_pair_columns(line, path, number)
                continue
            if line.startswith("#shape:"):
                if shape is not None:
                    raise ValueError(f"{path}:{number}: duplicate #shape header")
                shape = line[len("#shape:") :].strip().lower()
                continue
            if line.startswith("#"):
                continue
            error = pairs_api._line_error(
                line, number, path, width, sizes, pairs_api._suffix_order(sizes)
            )
            if error is not None:
                raise error
            read_id, c1, p1, c2, p2, s1, s2 = line.split("\t")[:7]
            meta = parse_mcc_read_id(read_id, sizes)
            qc["input_pairs"] += 1
            if "!" in (c1, c2):
                qc["unmapped_pairs"] += 1
                continue
            meta = None if meta is None else (meta.chromosome, meta.start, meta.end, meta.side)
            records.append((read_id, c1, int(p1) - 1, c2, int(p2) - 1, s1, s2, meta, shape))
    except ValueError as exc:
        return records, qc, str(exc)
    return records, qc, None


def _columnar(path, sizes, block_bytes):
    qc, records = {}, []
    reader = PairReader(path, sizes, qc=qc, block_bytes=block_bytes)
    names = reader.index.names
    try:
        for chunk in reader:
            for row in chunk.frame.iter_rows(named=True):
                meta = None
                if row["side"] >= 0:
                    meta = (
                        names[row["meta_chrom"]],
                        row["meta_start"],
                        row["meta_end"],
                        SIDES[row["side"]],
                    )
                records.append(
                    (
                        row["read_id"],
                        names[row["chrom1"]],
                        row["pos1"],
                        names[row["chrom2"]],
                        row["pos2"],
                        row["strand1"],
                        row["strand2"],
                        meta,
                        chunk.shape,
                    )
                )
    except ValueError as exc:
        return records, {k: qc[k] for k in ("input_pairs", "unmapped_pairs")}, str(exc)
    return records, {k: qc[k] for k in ("input_pairs", "unmapped_pairs")}, None


TOKENS = ["r", "__", "___", "_", "vp-chr1-100-120", "vp-chr1-0-0", "x-chr2-1-5", "vp-chrUn_a-10-20"]
TOKENS += ["start", "end", "all", "none", "1", "2", "3", "0", "4", "-", "chr1", "a"]


def _random_line(rng):
    read_id = "".join(rng.choice(TOKENS) for _ in range(rng.randint(0, 9)))
    if rng.random() < 0.3:
        side = rng.choice(["start", "end", "all", "none"])
        read_id = f"p{rng.randint(0, 9)}__vp-chr1-100-120__{side}__{rng.choice('123')}__0"
    chroms = [*SIZES, "!"] + (["chrZ"] if rng.random() < 0.02 else [])

    def position(chrom):
        if chrom == "!":
            return "0" if rng.random() < 0.98 else "1"
        if rng.random() < 0.01:
            return rng.choice(["+5", " 7", "1_0", "x", "", "0", str(SIZES.get(chrom, 5) + 1)])
        return str(rng.randint(1, SIZES.get(chrom, 5)))

    c1, c2 = rng.choice(chroms), rng.choice(chroms)
    fields = [read_id, c1, position(c1), c2, position(c2), rng.choice("+-."), rng.choice("+-.")]
    if rng.random() < 0.2:
        fields.append("extra")
    if rng.random() < 0.003:
        fields = fields[:6]
    return "\t".join(fields)


@pytest.mark.parametrize("seed", range(4))
def test_columnar_reader_matches_line_reference(tmp_path, seed):
    rng = random.Random(seed)
    for trial in range(60):
        lines = []
        if rng.random() < 0.3:
            lines.append("#shape: upper triangle")
        if rng.random() < 0.2:
            lines.append("#columns: readID chr1 pos1 chr2 pos2 strand1 strand2")
        for _ in range(rng.randint(0, 50)):
            roll = rng.random()
            if roll < 0.03:
                lines.append("")
            elif roll < 0.05:
                lines.append("# mid-file comment")
            elif roll < 0.052:
                lines.append("#shape: lower triangle")
            else:
                lines.append(_random_line(rng))
        newline = rng.choice(["\n", "\n", "\r\n", "\r"])
        path = tmp_path / f"t{trial}.pairs"
        path.write_text(newline.join(lines) + newline, newline="")
        expected = _reference(path, SIZES)
        for block_bytes in (1 << 20, 37):
            assert _columnar(path, SIZES, block_bytes) == expected, (trial, block_bytes)


def _synthetic_pairs(path, rows, seed, inside=False):
    rng = np.random.default_rng(seed)
    oligos = [(1000 + i * 120, 1120 + i * 120) for i in range(3)]
    lines = ["## pairs format v1.0.0\n"]
    for i in range(rows):
        start, end = oligos[rng.integers(len(oligos))]
        side = ["start", "end"][rng.integers(2)]
        read_id = f"r{i % (rows // 3)}__vp-chr1-{start}-{end}__{side}__1__0"
        capture = int(
            start + 1 + rng.integers(0, 120) if inside else start + rng.integers(-40, 160)
        )
        reporter = int(np.clip(start + rng.normal(0, 3000), 1, 20_000))
        strands = "+-"[rng.integers(2)], "+-"[rng.integers(2)]
        line = f"{read_id}\tchr1\t{reporter}\tchr1\t{capture}\t{strands[0]}\t{strands[1]}\n"
        lines.append(line)
        if rng.random() < 0.2:
            lines.append(line)
    path.write_text("".join(lines))


def test_footprint_and_counts_do_not_depend_on_read_blocks(tmp_path, monkeypatch):
    sizes = {"chr1": 20_000}
    viewpoints = {"chr1": [(1000, 1120), (1120, 1240), (1240, 1360)]}
    peaks = pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr1"],
            "Start": [3000, 3100, 6000],
            "End": [3300, 3400, 6200],
            "Name": ["a", "b", "c"],
        }
    )
    for name, seed in (("s1", 1), ("s2", 2)):
        _synthetic_pairs(tmp_path / f"{name}.pairs", 4000, seed)
    _synthetic_pairs(tmp_path / "counts.pairs", 4000, 3, inside=True)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\ns1\tvp\ts1.pairs\ns2\tvp\ts2.pairs\n")

    def run():
        profiles, contacts, _, metrics = profile_junctions(
            manifest, "vp", viewpoints, sizes, peaks, capture_flank=40
        )
        counts = count_pairs(tmp_path / "counts.pairs", sizes, {"chr1": [(1000, 1360)]})
        return profiles, contacts, [s for s in metrics["samples"]], counts

    whole = run()
    monkeypatch.setattr(pairs_api, "BLOCK_BYTES", 997)
    blocked = run()
    pd.testing.assert_frame_equal(whole[0], blocked[0])
    pd.testing.assert_frame_equal(whole[1], blocked[1])
    assert whole[2] == blocked[2]
    assert whole[3][0].tolist() == blocked[3][0].tolist() and whole[3][1] == blocked[3][1]
    assert not whole[1].empty and whole[2][0]["duplicate_contacts"] > 0
