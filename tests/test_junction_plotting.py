import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from lanceotron_mcc.api.junction_plotting import plot_junctions


def _profiles(regions=("reporter:peak1",)):
    rows = []
    for sample in ("rep1", "rep2"):
        for region in ("capture:oligo_1", *regions):
            chrom = "chr1" if region.startswith("capture:") else "chr2"
            for position in (100, 101, 102):
                rows.append(
                    {
                        "Sample": sample,
                        "Region": region,
                        "Chromosome": chrom,
                        "Position": position,
                        "RegionStart": 100,
                        "RegionEnd": 103,
                        "Upstream": position - 99,
                        "Downstream": 4 - (position - 99),
                        "Unknown": 0,
                        "Total": 4,
                    }
                )
    return pd.DataFrame(rows)


def _contacts(regions=("reporter:peak1",)):
    return pd.DataFrame(
        [
            {
                "Sample": "rep1",
                "Region": region,
                "CaptureChromosome": "chr1",
                "CapturePosition": 100 + index,
                "ReporterChromosome": "chr2",
                "ReporterPosition": 200 + index,
                "CaptureDirection": "upstream" if index == 0 else "unknown",
                "ReporterDirection": "downstream",
                "Count": index + 1,
            }
            for index, region in enumerate(regions)
        ]
    ).reindex(
        columns=[
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
    )


def test_writes_one_page_per_reporter_and_discloses_omitted_regions(tmp_path):
    profiles = _profiles(("reporter:peak1", "reporter:peak2", "reporter:peak3"))
    contacts = _contacts(("reporter:peak1", "reporter:peak2", "reporter:peak3"))
    output = tmp_path / "nested" / "junctions.pdf"

    result = plot_junctions(profiles, contacts, output, max_panels=2)

    assert result == output
    pdf = output.read_bytes()
    assert pdf.startswith(b"%PDF-")
    assert pdf.count(b"/Type /Page ") == 3  # two regions and an omission report


def test_empty_profiles_and_contacts_still_write_an_informative_pdf(tmp_path):
    profiles = _profiles(()).iloc[0:0]
    contacts = _contacts(()).iloc[0:0]
    output = tmp_path / "empty.pdf"

    plot_junctions(profiles, contacts, output)

    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 1000


def test_input_frames_are_not_modified(tmp_path):
    profiles = _profiles()
    contacts = _contacts()
    original_profiles = profiles.copy(deep=True)
    original_contacts = contacts.copy(deep=True)

    plot_junctions(profiles, contacts, tmp_path / "junctions.pdf")

    pd.testing.assert_frame_equal(profiles, original_profiles)
    pd.testing.assert_frame_equal(contacts, original_contacts)


def test_rejects_unknown_direction_and_nonpositive_arrow_length(tmp_path):
    contacts = _contacts()
    contacts.loc[0, "CaptureDirection"] = "left"
    with pytest.raises(ValueError, match="CaptureDirection"):
        plot_junctions(_profiles(), contacts, tmp_path / "bad.pdf")

    with pytest.raises(ValueError, match="arrow_length"):
        plot_junctions(_profiles(), _contacts(), tmp_path / "bad.pdf", arrow_length=0)

