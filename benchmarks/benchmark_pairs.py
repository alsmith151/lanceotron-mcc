"""Time pairs parsing, count binning and junction footprinting.

Uses a supplied manifest/viewpoint, or generates synthetic MCCNado-style pairs.
"""

import argparse
import gzip
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from mcc_peaks.api.adaptive import read_bed
from mcc_peaks.api.counts import count_pairs, load_manifest
from mcc_peaks.api.junctions import profile_junctions
from mcc_peaks.api.pairs import PairReader

SYNTHETIC_SIZES = {"chr1": 3_000_000, "chr2": 1_500_000}
SYNTHETIC_OLIGOS = [(1_000_000 + i * 120, 1_000_120 + i * 120) for i in range(6)]


def write_synthetic(path: Path, rows: int, seed: int = 2026) -> None:
    """MCCNado layout: pos1 is the reporter junction, pos2 the capture junction."""
    rng = np.random.default_rng(seed)
    oligo = rng.integers(len(SYNTHETIC_OLIGOS), size=rows)
    starts = np.asarray([SYNTHETIC_OLIGOS[i][0] for i in oligo])
    capture = starts + 1 + rng.integers(0, 120, size=rows)
    offsets = rng.exponential(60_000, size=rows) * rng.choice([-1, 1], size=rows)
    reporter = np.clip(starts + offsets.astype(np.int64), 1, SYNTHETIC_SIZES["chr1"])
    sides = np.where(rng.random(rows) < 0.5, "start", "end")
    strands = np.where(rng.random((rows, 2)) < 0.5, "+", "-")
    with gzip.open(path, "wt", compresslevel=1) as handle:
        handle.write("## pairs format v1.0.0\n")
        for i in range(rows):
            s, e = SYNTHETIC_OLIGOS[oligo[i]]
            handle.write(
                f"r{i}__VP-chr1-{s}-{e}__{sides[i]}__1__0\tchr1\t{reporter[i]}\t"
                f"chr1\t{capture[i]}\t{strands[i, 0]}\t{strands[i, 1]}\n"
            )


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, round(time.perf_counter() - started, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--viewpoint")
    parser.add_argument("--viewpoint-bed", type=Path)
    parser.add_argument("--chrom-sizes", type=Path)
    parser.add_argument("--rows", type=int, default=2_000_000, help="Synthetic rows.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mcc-peaks-pairs-") as tmp:
        if args.manifest:
            viewpoint = args.viewpoint
            viewpoints = read_bed(args.viewpoint_bed, viewpoint, require_single_group=True)
            sizes = {
                line.split()[0]: int(line.split()[1])
                for line in args.chrom_sizes.read_text().splitlines()
                if line.strip()
            }
            manifest = args.manifest
        else:
            viewpoint, sizes = "VP", SYNTHETIC_SIZES
            viewpoints = {"chr1": SYNTHETIC_OLIGOS}
            pairs = Path(tmp) / "synthetic.pairs.gz"
            write_synthetic(pairs, args.rows)
            manifest = Path(tmp) / "manifest.tsv"
            manifest.write_text(f"sample\tviewpoint\tpairs\nsynthetic\tVP\t{pairs.name}\n")
        samples = load_manifest(manifest, viewpoint)
        first = samples["pairs"].iloc[0]
        rows, read_seconds = timed(lambda: sum(c.frame.height for c in PairReader(first, sizes)))
        _, count_seconds = timed(lambda: count_pairs(first, sizes, viewpoints))
        (profiles, contacts, _, _), footprint_seconds = timed(
            lambda: profile_junctions(
                samples,
                viewpoint,
                viewpoints,
                sizes,
                allow_unoriented=True,
                max_contacts=10**9,
                max_profile_bases=10**9,
            )
        )
    result = {
        "synthetic": args.manifest is None,
        "mapped_rows_first_sample": rows,
        "read_seconds": read_seconds,
        "rows_per_second": round(rows / max(read_seconds, 1e-9)),
        "count_pairs_seconds": count_seconds,
        "footprint_seconds": footprint_seconds,
        "profile_rows": len(profiles),
        "contacts": len(contacts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
