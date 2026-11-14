"""End-to-end checks for the three supported CLI workflows."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pybigtools
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_bigwig(path: Path, tracks: dict[str, np.ndarray]):
    chroms = {chrom: len(values) for chrom, values in tracks.items()}
    records = (
        (chrom, start, end, float(values[start]))
        for chrom, values in tracks.items()
        for start, end in _nonzero_runs(values)
    )
    writer = pybigtools.open(str(path), "w")
    writer.write(chroms, records)
    writer.close()
    return path


def _nonzero_runs(values):
    nonzero = np.asarray(values) != 0
    edges = np.diff(np.r_[False, nonzero, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "lanceotron_mcc.call_peaks", *map(str, args)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=90,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )


def _assert_success(result):
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _assert_sidecars(outfile, *, bins=False):
    outfile = Path(outfile)
    assert outfile.exists()
    assert Path(str(outfile) + ".tsv").exists()
    assert Path(str(outfile) + ".unresolved.bed").exists()
    assert Path(str(outfile) + ".json").exists()
    if bins:
        assert Path(str(outfile) + ".bins.tsv").exists()


def _signal_track():
    signal = np.ones(500, dtype=np.float32)
    signal[100:120] = 10
    signal[300:320] = 8
    return signal


def _shared_grid():
    return [
        "--thresholds",
        1,
        "--background-windows",
        51,
        "--window-sizes",
        5,
        "--support-fraction",
        1,
    ]


def test_cli_help_groups_and_version_are_meaningful():
    help_result = _invoke("--help")
    _assert_success(help_result)
    for command in ("call-bigwig-peaks", "call-count-peaks", "call-legacy-peaks"):
        assert command in help_result.stdout
    assert "call-mcc-peaks" not in help_result.stdout

    command_help = _invoke("call-bigwig-peaks", "--help")
    _assert_success(command_help)
    for group in ("Inputs", "Peak detection", "Resources"):
        assert group in command_help.stdout
    assert "support-fraction" in command_help.stdout
    assert re.search(r"fraction of grid combinations", command_help.stdout, re.I)
    assert "threshold-initial" not in command_help.stdout

    version_result = _invoke("--version")
    _assert_success(version_result)
    assert re.search(r"\d+\.\d+", version_result.stdout)
    assert version_result.stdout.strip() not in {"", "0.0.0", "unknown"}


def test_cli_legacy_writes_bed_metrics_and_provenance_sidecars(tmp_path):
    bigwig = _write_bigwig(tmp_path / "legacy.bw", {"chr1": _signal_track()})
    outfile = tmp_path / "legacy.bed"
    result = _invoke(
        "call-legacy-peaks",
        "--bigwig",
        bigwig,
        "--output",
        outfile,
        "--cis-chromosome",
        "chr1",
        "--background-chromosomes",
        0,
        "--device",
        "cpu",
        "--no-filter-background",
        "--no-filter-outliers",
        *_shared_grid(),
        "--global-background-floor",
        0,
        "--min-peak-size",
        3,
        "--max-peak-size",
        50,
        "--threads",
        2,
    )
    _assert_success(result)
    _assert_sidecars(outfile)
    peaks = pd.read_csv(str(outfile) + ".tsv", sep="\t")
    assert {"Signal", "Sum", "Maximum", "Summit", "Support Fraction", "Unresolved"} <= set(
        peaks.columns
    )
    metadata = json.loads(Path(str(outfile) + ".json").read_text())
    assert metadata["method"] == "legacy"
    assert metadata["metrics"]["cis_chromosome"] == "chr1"
    assert metadata["parameters"]["support_fraction"] == 1


def test_cli_adaptive_reads_multiple_bigwigs_and_records_original_inputs(tmp_path):
    first = _write_bigwig(tmp_path / "sample-a.bw", {"chr1": _signal_track()})
    second = _write_bigwig(tmp_path / "sample-b.bw", {"chr1": _signal_track()})
    viewpoint = tmp_path / "viewpoint.bed"
    viewpoint.write_text("chr1\t0\t10\tvp\n")
    excluded = tmp_path / "exclude.bed"
    excluded.write_text("chr1\t100\t120\n")
    outfile = tmp_path / "adaptive.bed"
    result = _invoke(
        "call-bigwig-peaks",
        "--bigwigs",
        first,
        second,
        "--output",
        outfile,
        "--viewpoint-bed",
        viewpoint,
        "--viewpoint",
        "vp",
        "--exclude-bed",
        excluded,
        *_shared_grid(),
        "--min-peak-size",
        3,
        "--max-peak-size",
        50,
        "--memory-budget-mb",
        16,
        "--chunk-size",
        97,
    )
    _assert_success(result)
    _assert_sidecars(outfile)
    peaks = pd.read_csv(str(outfile) + ".tsv", sep="\t")
    if not peaks.empty:
        assert not ((peaks.Start < 10) & (peaks.End > 0)).any()
        assert not ((peaks.Start < 120) & (peaks.End > 100)).any()
    metadata = json.loads(Path(str(outfile) + ".json").read_text())
    assert metadata["method"] == "adaptive"
    assert metadata["parameters"]["viewpoints"] == {"chr1": [[0, 10]]}
    input_paths = {item["path"] for item in metadata["inputs"]}
    assert {
        str(first.resolve()),
        str(second.resolve()),
        str(viewpoint.resolve()),
        str(excluded.resolve()),
    } <= input_paths


def _count_inputs(tmp_path):
    pairs = tmp_path / "sample.pairs"
    with pairs.open("w") as handle:
        # Reporter end at 700 supplies background; capture end is inside vp.
        for i in range(20):
            handle.write(f"bg{i}\tchr1\t701\tchr1\t401\t+\t-\n")
        for i in range(10):
            handle.write(f"peak{i}\tchr1\t101\tchr1\t401\t+\t-\n")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(f"sample\tviewpoint\tpairs\nsample1\tvp\t{pairs.name}\n")
    viewpoint = tmp_path / "viewpoint.bed"
    viewpoint.write_text("chr1\t400\t450\tvp\n")
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chr1\t1000\n")
    return manifest, viewpoint, chrom_sizes


def _count_cli(outfile, manifest, viewpoint, sizes, *, support_fraction=1, extra=()):
    return _invoke(
        "call-count-peaks",
        "--manifest",
        manifest,
        "--viewpoint",
        "vp",
        "--viewpoint-bed",
        viewpoint,
        "--chrom-sizes",
        sizes,
        "--output",
        outfile,
        "--bin-size",
        50,
        "--thresholds",
        2,
        "--background-windows",
        500,
        "--window-sizes",
        50,
        "--support-fraction",
        support_fraction,
        "--min-peak-size",
        50,
        "--max-peak-size",
        100,
        "--memory-budget-mb",
        16,
        "--chunk-size",
        100,
        *extra,
    )


def test_cli_counts_writes_bin_q_values_and_experimental_provenance(tmp_path):
    manifest, viewpoint, sizes = _count_inputs(tmp_path)
    outfile = tmp_path / "counts.bed"
    result = _count_cli(outfile, manifest, viewpoint, sizes)
    _assert_success(result)
    _assert_sidecars(outfile, bins=True)
    peaks = pd.read_csv(str(outfile) + ".tsv", sep="\t")
    assert "MinimumBinQ" in peaks.columns
    assert "PeakQ" not in peaks.columns
    assert not peaks.empty
    assert np.isfinite(peaks.MinimumBinQ).all()
    bins = pd.read_csv(str(outfile) + ".bins.tsv", sep="\t")
    assert {"BinQ", "PValue", "Eligible", "Count:sample1"} <= set(bins.columns)
    metadata = json.loads(Path(str(outfile) + ".json").read_text())
    assert metadata["method"] == "counts"
    assert "experimental" in metadata["statistical_status"]
    assert metadata["metrics"]["experimental"] is True


def test_cli_empty_count_results_write_zero_byte_bed(tmp_path):
    manifest, viewpoint, sizes = _count_inputs(tmp_path)
    outfile = tmp_path / "empty.bed"
    result = _count_cli(
        outfile, manifest, viewpoint, sizes, support_fraction=1, extra=("--thresholds", 1e9)
    )
    _assert_success(result)
    _assert_sidecars(outfile, bins=True)
    assert outfile.stat().st_size == 0
    assert pd.read_csv(str(outfile) + ".tsv", sep="\t").empty


def test_cli_lists_replicates_from_manifest(tmp_path):
    manifest, viewpoint, sizes = _count_inputs(tmp_path)
    result = _invoke(
        "call-count-peaks",
        "--manifest",
        manifest,
        "--viewpoint",
        "vp",
        "--viewpoint-bed",
        viewpoint,
        "--chrom-sizes",
        sizes,
        "--help",
    )
    _assert_success(result)
    assert "--manifest" in result.stdout
    assert "pairs" in result.stdout.lower()


@pytest.mark.parametrize(
    ("removed", "extra"),
    [
        ("call-mcc-peaks", ()),
        ("--bigwig", ()),
        ("--outfile", ("legacy.bw",)),
        ("--threshold-initial", (1,)),
        ("--threshold-lst", (1,)),
        ("--threshold-background-lst", (51,)),
        ("--n-jobs", (2,)),
        ("--no-use-gpu", ()),
    ],
)
def test_cli_rejects_removed_commands_and_option_aliases(removed, extra, tmp_path):
    if removed == "call-mcc-peaks":
        result = _invoke(removed)
    elif removed == "--bigwig":
        result = _invoke(
            "call-bigwig-peaks",
            "--output",
            tmp_path / "out.bed",
            "--viewpoint-bed",
            tmp_path / "viewpoint.bed",
            removed,
            tmp_path / "missing.bw",
        )
    else:
        result = _invoke(
            "call-legacy-peaks",
            "--bigwig",
            tmp_path / "missing.bw",
            "--output",
            tmp_path / "out.bed",
            removed,
            *extra,
        )
    assert result.returncode != 0
    assert removed in result.stderr


def test_cli_invalid_chromosome_sizes_has_concise_error_and_debug_traceback(tmp_path):
    manifest, viewpoint, sizes = _count_inputs(tmp_path)
    sizes.write_text("chr1\t0\n")
    args = (
        "call-count-peaks",
        "--manifest",
        manifest,
        "--viewpoint",
        "vp",
        "--viewpoint-bed",
        viewpoint,
        "--chrom-sizes",
        sizes,
        "--output",
        tmp_path / "bad.bed",
    )
    result = _invoke(*args)
    assert result.returncode != 0
    assert "positive integers" in result.stderr
    assert "Traceback" not in result.stderr
    debug = _invoke(*args, "--debug")
    assert debug.returncode != 0
    assert "Traceback" in debug.stderr


def test_cli_invalid_input_is_reported_without_traceback(tmp_path):
    result = _invoke(
        "call-legacy-peaks",
        "--bigwig",
        tmp_path / "missing.bw",
        "--output",
        tmp_path / "x.bed",
        "--cis-chromosome",
        "chr1",
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr


def test_cli_prevents_output_and_sidecar_paths_from_overwriting_inputs(tmp_path):
    source = _write_bigwig(tmp_path / "source.bw", {"chr1": _signal_track()})
    original = source.read_bytes()
    same_path = _invoke(
        "call-legacy-peaks",
        "--bigwig",
        source,
        "--output",
        source,
        "--cis-chromosome",
        "chr1",
        "--background-chromosomes",
        0,
        "--device",
        "cpu",
        "--no-filter-background",
        "--no-filter-outliers",
        *_shared_grid(),
        "--global-background-floor",
        0,
    )
    assert same_path.returncode != 0
    assert source.read_bytes() == original

    valid_bigwig = _write_bigwig(tmp_path / "sidecar.bw", {"chr1": _signal_track()})
    sidecar_input = tmp_path / "sidecar.bed.tsv"
    valid_bigwig.rename(sidecar_input)
    original_sidecar = sidecar_input.read_bytes()
    sidecar = _invoke(
        "call-legacy-peaks",
        "--bigwig",
        sidecar_input,
        "--output",
        tmp_path / "sidecar.bed",
        "--cis-chromosome",
        "chr1",
        "--background-chromosomes",
        0,
        "--device",
        "cpu",
        "--no-filter-background",
        "--no-filter-outliers",
        *_shared_grid(),
        "--global-background-floor",
        0,
    )
    assert sidecar.returncode != 0
    assert sidecar_input.read_bytes() == original_sidecar

