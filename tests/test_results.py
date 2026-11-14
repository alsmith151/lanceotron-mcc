import json
import os
from pathlib import Path

import pandas as pd
import pytest

from lanceotron_mcc.api.results import OutputPaths, preflight_outputs, write_outputs


def _peaks(**values):
    fields = {"Chromosome": ["chr1"], "Start": [10], "End": [20], **values}
    return pd.DataFrame(fields)


def _existing_artifacts(outfile, *, include_bins=False):
    paths = OutputPaths.from_bed(outfile)
    for path in paths.artifacts(include_bins):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"old:{path.name}".encode())
    return {path: path.read_bytes() for path in paths.artifacts(include_bins)}


def _identity(path):
    return preflight_outputs(path.with_suffix(".unused"), [path])[0]


def test_preflight_rejects_direct_input_overwrite(tmp_path):
    source = tmp_path / "input.bed"
    source.write_text("source")

    with pytest.raises(ValueError, match="overwrite an input"):
        preflight_outputs(source, [source])


def test_preflight_rejects_symlink_and_hardlink_input_aliases(tmp_path):
    source = tmp_path / "input.bed"
    source.write_text("source")
    symlink = tmp_path / "linked.bed"
    hardlink = tmp_path / "hardlinked.bed"
    symlink.symlink_to(source)
    os.link(source, hardlink)

    with pytest.raises(ValueError, match="overwrite an input"):
        preflight_outputs(symlink, [source])
    with pytest.raises(ValueError, match="aliases an input"):
        preflight_outputs(hardlink, [source])


@pytest.mark.parametrize(
    "suffix",
    [".tsv", ".unresolved.bed", ".json", ".bins.tsv"],
)
def test_preflight_rejects_sidecar_collisions(tmp_path, suffix):
    outfile = tmp_path / "called.bed"
    sidecar_input = Path(f"{outfile}{suffix}")
    sidecar_input.write_text("input data")

    with pytest.raises(ValueError, match="overwrite an input"):
        preflight_outputs(outfile, [sidecar_input])


@pytest.mark.parametrize(
    "bad_provenance",
    [{"score": float("nan")}, {"score": object()}],
    ids=["nonfinite-json", "unsupported-provenance-value"],
)
def test_provenance_failure_leaves_existing_outputs_unchanged(tmp_path, bad_provenance):
    outfile = tmp_path / "called.bed"
    old = _existing_artifacts(outfile)
    source = tmp_path / "input.bed"
    source.write_text("source")

    with pytest.raises((ValueError, TypeError)):
        write_outputs(
            _peaks(),
            outfile,
            method="adaptive",
            parameters={},
            metrics=bad_provenance,
            input_identities=[_identity(source)],
        )

    assert {path: path.read_bytes() for path in old} == old


def test_input_changed_after_preflight_prevents_publish_and_preserves_outputs(tmp_path):
    outfile = tmp_path / "called.bed"
    old = _existing_artifacts(outfile)
    source = tmp_path / "input.bed"
    source.write_text("before")
    identities = preflight_outputs(outfile, [source])
    source.write_text("after, with a different size")

    with pytest.raises(ValueError, match="input changed"):
        write_outputs(
            _peaks(),
            outfile,
            method="adaptive",
            parameters={},
            metrics={},
            input_identities=identities,
        )

    assert {path: path.read_bytes() for path in old} == old


def test_staging_failure_leaves_all_existing_outputs_unchanged(tmp_path, monkeypatch):
    outfile = tmp_path / "called.bed"
    old = _existing_artifacts(outfile, include_bins=True)
    source = tmp_path / "input.bed"
    source.write_text("source")
    original_to_csv = pd.DataFrame.to_csv
    calls = 0

    def fail_when_staging_bins(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated staged bins write failure")
        return original_to_csv(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_when_staging_bins)
    with pytest.raises(OSError, match="staged bins"):
        write_outputs(
            _peaks(),
            outfile,
            method="adaptive",
            parameters={},
            metrics={},
            input_identities=[_identity(source)],
            bins=pd.DataFrame({"Start": [0]}),
        )

    assert {path: path.read_bytes() for path in old} == old


@pytest.mark.parametrize(
    ("start", "end", "unresolved", "message"),
    [
        (float("nan"), 20, False, "coordinates"),
        (-1, 20, False, "coordinates"),
        (10.5, 20, False, "coordinates"),
        (20, 20, False, "positive width"),
        (10, 20, "false", "Unresolved flags must be boolean"),
    ],
)
def test_invalid_coordinates_and_unresolved_values_are_rejected(
    tmp_path, start, end, unresolved, message
):
    outfile = tmp_path / "called.bed"
    source = tmp_path / "input.bed"
    source.write_text("source")
    peaks = pd.DataFrame(
        {"Chromosome": ["chr1"], "Start": [start], "End": [end], "Unresolved": [unresolved]}
    )

    with pytest.raises(ValueError, match=message):
        write_outputs(
            peaks,
            outfile,
            method="adaptive",
            parameters={},
            metrics={},
            input_identities=[_identity(source)],
        )

    assert not any(path.exists() for path in OutputPaths.from_bed(outfile).artifacts())


def test_empty_peak_table_writes_empty_beds_and_metadata(tmp_path):
    outfile = tmp_path / "called.bed"
    source = tmp_path / "input.bed"
    source.write_text("source")

    metadata = write_outputs(
        pd.DataFrame(),
        outfile,
        method="counts",
        parameters={},
        metrics={},
        input_identities=[_identity(source)],
    )

    paths = OutputPaths.from_bed(outfile)
    assert paths.bed.read_bytes() == b""
    assert paths.unresolved.read_bytes() == b""
    table = pd.read_csv(paths.table, sep="\t")
    assert table.empty
    assert list(table.columns) == ["Chromosome", "Start", "End", "Unresolved"]
    disk_metadata = json.loads(paths.provenance.read_text())
    assert disk_metadata == metadata
    assert metadata["resolved_peaks"] == 0
    assert metadata["unresolved_regions"] == 0
    assert metadata["method"] == "counts"


def test_signal_run_removes_previous_count_bin_artifact(tmp_path):
    outfile = tmp_path / "called.bed"
    source = tmp_path / "source.txt"
    source.write_text("input")
    identities = preflight_outputs(outfile, [source])
    common = dict(parameters={}, metrics={}, input_identities=identities)
    write_outputs(_peaks(), outfile, method="counts", bins=pd.DataFrame({"Start": [0]}), **common)
    paths = OutputPaths.from_bed(outfile)
    assert paths.bins.exists()
    write_outputs(_peaks(), outfile, method="adaptive", **common)
    assert not paths.bins.exists()
    metadata = json.loads(paths.provenance.read_text())
    assert paths.bins.name not in metadata["artifacts"]


def test_input_change_during_staging_prevents_publish(tmp_path, monkeypatch):
    outfile = tmp_path / "called.bed"
    old = _existing_artifacts(outfile)
    source = tmp_path / "source.txt"
    source.write_text("before")
    identities = preflight_outputs(outfile, [source])
    original = pd.DataFrame.to_csv

    def change_input(frame, *args, **kwargs):
        source.write_text("changed during output preparation")
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", change_input)
    with pytest.raises(ValueError, match="input changed while staging"):
        write_outputs(
            _peaks(),
            outfile,
            method="adaptive",
            parameters={},
            metrics={},
            input_identities=identities,
        )
    assert {path: path.read_bytes() for path in old} == old
