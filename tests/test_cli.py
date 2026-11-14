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
        [sys.executable, "-m", "mcc_peaks.call_peaks", *map(str, args)],
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
    for command in ("call-bigwig-peaks", "call-count-peaks", "footprint"):
        assert command in help_result.stdout
    assert "call-mcc-peaks" not in help_result.stdout
    assert "call-legacy-peaks" not in help_result.stdout

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
        ("call-legacy-peaks", ()),
        ("--bigwig", ()),
        ("--outfile", ("other.bed",)),
        ("--threshold-initial", (1,)),
        ("--threshold-lst", (1,)),
        ("--threshold-background-lst", (51,)),
        ("--n-jobs", (2,)),
        ("--no-use-gpu", ()),
    ],
)
def test_cli_rejects_removed_commands_and_option_aliases(removed, extra, tmp_path):
    if removed in {"call-mcc-peaks", "call-legacy-peaks"}:
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
            "call-bigwig-peaks",
            "--bigwigs",
            tmp_path / "missing.bw",
            "--viewpoint-bed",
            tmp_path / "viewpoint.bed",
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
    viewpoint = tmp_path / "viewpoint.bed"
    viewpoint.write_text("chr1\t0\t10\tvp\n")
    result = _invoke(
        "call-bigwig-peaks",
        "--bigwigs",
        tmp_path / "missing.bw",
        "--viewpoint-bed",
        viewpoint,
        "--output",
        tmp_path / "x.bed",
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr


def test_cli_prevents_output_and_sidecar_paths_from_overwriting_inputs(tmp_path):
    viewpoint = tmp_path / "viewpoint.bed"
    viewpoint.write_text("chr1\t0\t10\tvp\n")

    def call(bigwig, output):
        return _invoke(
            "call-bigwig-peaks",
            "--bigwigs",
            bigwig,
            "--viewpoint-bed",
            viewpoint,
            "--output",
            output,
            *_shared_grid(),
        )

    source = _write_bigwig(tmp_path / "source.bw", {"chr1": _signal_track()})
    original = source.read_bytes()
    assert call(source, source).returncode != 0
    assert source.read_bytes() == original

    valid_bigwig = _write_bigwig(tmp_path / "sidecar.bw", {"chr1": _signal_track()})
    sidecar_input = tmp_path / "sidecar.bed.tsv"
    valid_bigwig.rename(sidecar_input)
    original_sidecar = sidecar_input.read_bytes()
    assert call(sidecar_input, tmp_path / "sidecar.bed").returncode != 0
    assert sidecar_input.read_bytes() == original_sidecar


def _footprint_inputs(tmp_path):
    pairs = tmp_path / "vp.pairs"
    name = "read{}__vp-chr1-1000-1120__{}__1__0"
    pairs.write_text(
        f"{name.format(1, 'start')}\tchr1\t5051\tchr1\t1131\t+\t+\n"
        f"{name.format(2, 'end')}\tchr1\t5061\tchr1\t990\t-\t-\n"
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(f"sample\tviewpoint\tpairs\nrep1\tvp\t{pairs.name}\n")
    viewpoint = tmp_path / "viewpoint.bed"
    viewpoint.write_text("chr1\t1000\t1120\tvp\n")
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t10000\n")
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t5000\t5100\tenhancer\n")
    return manifest, viewpoint, sizes, peaks


def _footprint_cli(prefix, manifest, viewpoint, sizes, *extra):
    return _invoke(
        "footprint",
        "--manifest",
        manifest,
        "--viewpoint",
        "vp",
        "--viewpoint-bed",
        viewpoint,
        "--chrom-sizes",
        sizes,
        "--output",
        prefix,
        *extra,
    )


def test_cli_footprint_writes_profiles_contacts_metrics_and_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    manifest, viewpoint, sizes, peaks = _footprint_inputs(tmp_path)
    prefix = tmp_path / "out" / "vp"
    result = _footprint_cli(prefix, manifest, viewpoint, sizes, "--peaks", peaks)
    _assert_success(result)

    profiles = pd.read_csv(f"{prefix}.profiles.tsv.gz", sep="\t")
    contacts = pd.read_csv(f"{prefix}.contacts.tsv", sep="\t")
    regions = pd.read_csv(f"{prefix}.regions.tsv", sep="\t")
    assert regions["Region"].tolist() == ["capture:oligo_1", "reporter:enhancer"]
    assert contacts[
        ["CapturePosition", "ReporterPosition", "CaptureDirection", "ReporterDirection"]
    ].values.tolist() == [
        [989, 5060, "upstream", "downstream"],
        [1130, 5050, "upstream", "downstream"],
    ]
    reporter = profiles.loc[profiles["Region"] == "reporter:enhancer"].set_index("Position")
    assert reporter.loc[5050, "Downstream"] == 1 and reporter.loc[5060, "Downstream"] == 1
    assert Path(f"{prefix}.pdf").stat().st_size > 0

    provenance = json.loads(Path(f"{prefix}.metrics.json").read_text())
    assert provenance["method"] == "junction-footprint"
    assert provenance["parameters"]["reporter_flank"] == 100
    assert {Path(item["path"]).name for item in provenance["inputs"]} >= {"vp.pairs", "peaks.bed"}
    assert provenance["metrics"]["samples"][0]["map_eligible_contacts"] == 2


def test_cli_footprint_no_plot_and_validation_errors(tmp_path):
    manifest, viewpoint, sizes, _ = _footprint_inputs(tmp_path)
    prefix = tmp_path / "noplot"
    _assert_success(_footprint_cli(prefix, manifest, viewpoint, sizes, "--no-plot"))
    assert Path(f"{prefix}.profiles.tsv.gz").exists()
    assert not Path(f"{prefix}.pdf").exists()

    bad = _footprint_cli(
        tmp_path / "bad", manifest, viewpoint, sizes, "--no-plot", "--reporter-flank", -1
    )
    assert bad.returncode == 2
    assert "flanks" in bad.stderr

    alias = _footprint_cli(tmp_path / "vp", manifest, viewpoint, sizes, "--no-plot")
    assert alias.returncode == 0  # vp.pairs is not an output name
    clash = tmp_path / "clash.regions.tsv"
    clash.write_text("x")
    manifest.write_text(f"sample\tviewpoint\tpairs\nrep1\tvp\t{clash.name}\n")
    collide = _footprint_cli(tmp_path / "clash", manifest, viewpoint, sizes, "--no-plot")
    assert collide.returncode == 2
    assert "overwrite an input" in collide.stderr


def test_chrom_sizes_accept_samtools_fai_index(tmp_path):
    from mcc_peaks.call_peaks import _read_chrom_sizes

    fai = tmp_path / "genome.fa.fai"
    fai.write_text("chr1\t1000\t6\t60\t61\nchr2\t500\t1030\t60\t61\n")
    assert _read_chrom_sizes(fai) == {"chr1": 1000, "chr2": 500}

    bad = tmp_path / "three.sizes"
    bad.write_text("chr1\t1000\textra\n")
    with pytest.raises(ValueError, match="two columns"):
        _read_chrom_sizes(bad)
