"""Base-pair junction profiles and sparse capture x reporter maps."""

import pandas as pd
import pytest

from mcc_peaks.api.junctions import (
    STALE_FORMAT_WARNING,
    profile_junctions,
    read_peak_regions,
)

SIZES = {"chr1": 10_000}
VIEWPOINTS = {"chr1": [(1000, 1120)]}
PEAKS = pd.DataFrame({"Chromosome": ["chr1"], "Start": [5000], "End": [5100], "Name": ["peak"]})
CAPTURE = "capture:oligo_1"
REPORTER = "reporter:peak"


def _name(index, side="start", oligo="chr1-1000-1120"):
    return f"read{index}__vp-{oligo}__{side}__1__0"


def _line(read_id, reporter, capture, strand1="+", strand2="+"):
    """MCCNado layout: one-based reporter junction (pos1) then capture junction (pos2)."""
    return f"{read_id}\tchr1\t{reporter}\tchr1\t{capture}\t{strand1}\t{strand2}\n"


def _manifest(tmp_path, *lines, header=""):
    pairs = tmp_path / "vp.pairs"
    pairs.write_text(header + "".join(lines))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(f"sample\tviewpoint\tpairs\nrep1\tvp\t{pairs.name}\n")
    return manifest


def _run(tmp_path, *lines, header="", peaks=PEAKS, **kwargs):
    manifest = _manifest(tmp_path, *lines, header=header)
    return profile_junctions(manifest, "vp", VIEWPOINTS, SIZES, peaks, **kwargs)


def _at(profiles, region, position):
    row = profiles.loc[(profiles["Region"] == region) & (profiles["Position"] == position)]
    assert len(row) == 1
    return row.iloc[0]


# Fragment extends from each junction towards the segment's other end, following
# MCCNado's get_ligation_positions table (start side = right reporter segment).
DIRECTION_CASES = [
    ("start", "+", "+", "downstream", "upstream"),
    ("start", "-", "-", "upstream", "downstream"),
    ("start", "+", "-", "downstream", "downstream"),
    ("start", "-", "+", "upstream", "upstream"),
    ("end", "+", "+", "upstream", "downstream"),
    ("end", "-", "-", "downstream", "upstream"),
    ("end", "+", "-", "upstream", "upstream"),
    ("end", "-", "+", "downstream", "downstream"),
]


@pytest.mark.parametrize(("side", "strand1", "strand2", "reporter", "capture"), DIRECTION_CASES)
def test_mccnado_side_and_strand_give_fragment_directions(
    tmp_path, side, strand1, strand2, reporter, capture
):
    profiles, contacts, _, metrics = _run(
        tmp_path, _line(_name(1, side), 5051, 1131, strand1, strand2)
    )
    assert contacts[["CaptureDirection", "ReporterDirection"]].values.tolist() == [
        [capture, reporter]
    ]
    assert contacts.iloc[0][["CapturePosition", "ReporterPosition", "Count"]].tolist() == [
        1130,
        5050,
        1,
    ]
    reporter_row = _at(profiles, REPORTER, 5050)
    capture_row = _at(profiles, CAPTURE, 1130)
    assert reporter_row[reporter.capitalize()] == 1 and reporter_row["Total"] == 1
    assert capture_row[capture.capitalize()] == 1 and capture_row["Total"] == 1
    qc = metrics["samples"][0]
    assert qc["map_eligible_contacts"] == 1
    assert qc["selected_junction_ends"] == 1
    assert qc["warnings"] == []


def test_regions_include_flanks_and_profile_every_base(tmp_path):
    profiles, _, regions, _ = _run(tmp_path, _line(_name(1), 5051, 1131))
    assert regions.values.tolist() == [
        [CAPTURE, "capture", "chr1", 950, 1171],
        [REPORTER, "reporter", "chr1", 4900, 5200],
    ]
    assert len(profiles) == (1171 - 950) + (5200 - 4900)
    assert profiles["Total"].sum() == 2


def test_overlapping_reporter_windows_count_capture_end_once(tmp_path):
    peaks = pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [5000, 5040],
            "End": [5100, 5200],
            "Name": ["a", "b"],
        }
    )
    profiles, contacts, _, metrics = _run(tmp_path, _line(_name(1), 5051, 1131), peaks=peaks)
    assert sorted(contacts["Region"]) == ["reporter:a", "reporter:b"]
    assert _at(profiles, "reporter:a", 5050)["Total"] == 1
    assert _at(profiles, "reporter:b", 5050)["Total"] == 1
    assert _at(profiles, CAPTURE, 1130)["Total"] == 1
    assert metrics["samples"][0]["map_eligible_contacts"] == 1


def test_map_requires_actual_capture_window_endpoint(tmp_path):
    # Capture junction 200 bp beyond the oligo, outside the 50 bp capture flank.
    profiles, contacts, _, metrics = _run(tmp_path, _line(_name(1), 5051, 1321))
    assert contacts.empty
    assert _at(profiles, REPORTER, 5050)["Total"] == 1
    assert profiles.loc[profiles["Region"] == CAPTURE, "Total"].sum() == 0
    qc = metrics["samples"][0]
    assert qc["selected_without_capture_endpoint"] == 1

    _, contacts, _, _ = _run(tmp_path, _line(_name(1), 5051, 1321), capture_flank=250)
    assert len(contacts) == 1


def test_stale_mccnado_reporter_only_pairs_warn(tmp_path):
    """Older MCCNado wrote the far reporter-segment end into pos2."""
    profiles, contacts, _, metrics = _run(tmp_path, _line(_name(1), 5051, 5161))
    assert contacts.empty
    qc = metrics["samples"][0]
    assert qc["map_eligible_contacts"] == 0
    assert qc["warnings"] == [STALE_FORMAT_WARNING]


def test_exact_duplicates_collapse_unless_disabled(tmp_path):
    line = _line(_name(1), 5051, 1131)
    _, contacts, _, metrics = _run(tmp_path, line, line)
    assert contacts["Count"].tolist() == [1]
    assert metrics["samples"][0]["duplicate_contacts"] == 1

    _, contacts, _, metrics = _run(tmp_path, line, line, deduplicate=False)
    assert contacts["Count"].tolist() == [2]
    assert metrics["deduplication"] == "none"


def test_same_position_different_reads_aggregate(tmp_path):
    _, contacts, _, _ = _run(tmp_path, _line(_name(1), 5051, 1131), _line(_name(2), 5051, 1131))
    assert contacts["Count"].tolist() == [2]


def test_exclusions_and_other_pool_annotations_are_skipped(tmp_path):
    lines = (
        _line(_name(1), 5051, 1131),
        _line(_name(2, oligo="chr1-3000-3120"), 5051, 1131),
    )
    _, contacts, _, metrics = _run(tmp_path, *lines, exclusions={"chr1": [(5040, 5060)]})
    qc = metrics["samples"][0]
    assert contacts.empty
    assert qc["excluded_contacts"] == 1
    assert qc["non_pool_annotation_contacts"] == 1


def test_unoriented_contacts_require_opt_in(tmp_path):
    line = _line("plain-read", 5051, 1131)
    with pytest.raises(ValueError, match="Cannot orient"):
        _run(tmp_path, line)
    profiles, contacts, _, metrics = _run(tmp_path, line, allow_unoriented=True)
    assert contacts[["CaptureDirection", "ReporterDirection"]].values.tolist() == [
        ["unknown", "unknown"]
    ]
    assert _at(profiles, REPORTER, 5050)["Unknown"] == 1
    assert metrics["samples"][0]["unoriented_contacts"] == 1

    _, contacts, _, _ = _run(tmp_path, _line(_name(1), 5051, 1131), orientation_source="unknown")
    assert contacts["ReporterDirection"].tolist() == ["unknown"]


def test_flipped_pairs_cannot_be_oriented(tmp_path):
    with pytest.raises(ValueError, match="Flipped"):
        _run(tmp_path, _line(_name(1), 5051, 1131), header="#shape: upper triangle\n")


def test_limits_and_parameters_are_validated(tmp_path):
    line = _line(_name(1), 5051, 1131)
    with pytest.raises(ValueError, match="max_profile_bases"):
        _run(tmp_path, line, max_profile_bases=10)
    with pytest.raises(ValueError, match="max_contacts"):
        _run(tmp_path, line, _line(_name(2), 5052, 1131), max_contacts=1)
    with pytest.raises(ValueError, match="flanks"):
        _run(tmp_path, line, reporter_flank=-1)
    with pytest.raises(ValueError, match="orientation_source"):
        _run(tmp_path, line, orientation_source="guess")


def test_default_reporter_regions_are_oligo_tiles(tmp_path):
    profiles, contacts, regions, _ = _run(tmp_path, _line(_name(1), 1101, 1131), peaks=None)
    assert regions["Region"].tolist() == [CAPTURE, "reporter:tile_1"]
    assert len(contacts) == 1


def test_read_peak_regions_names_and_validation(tmp_path):
    bed3 = tmp_path / "peaks.bed"
    bed3.write_text("track name=x\nchr1\t10\t20\nchr1\t30\t40\n")
    assert read_peak_regions(bed3, SIZES).values.tolist() == [
        ["chr1", 10, 20, "peak_1"],
        ["chr1", 30, 40, "peak_2"],
    ]
    bed4 = tmp_path / "named.bed"
    bed4.write_text("chr1\t10\t20\tenh\n")
    assert read_peak_regions(bed4, SIZES)["Name"].tolist() == ["enh"]
    bad = tmp_path / "bad.bed"
    bad.write_text("chr9\t10\t20\n")
    with pytest.raises(ValueError, match="absent"):
        read_peak_regions(bad, SIZES)
    empty = tmp_path / "empty.bed"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_peak_regions(empty, SIZES)
