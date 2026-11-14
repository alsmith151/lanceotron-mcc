import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from mcc_peaks.api.junction_plotting import plot_junctions


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


def _axes():
    from matplotlib.figure import Figure

    return Figure().add_subplot()


def test_unoriented_profiles_are_labelled_as_unknown_direction():
    from mcc_peaks.api.junction_plotting import _draw_profile

    profiles = _profiles()
    profiles["Unknown"] = profiles["Upstream"] + profiles["Downstream"]
    profiles[["Upstream", "Downstream"]] = 0
    ax = _axes()

    _draw_profile(ax, profiles, title="unoriented")

    labels = [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")]
    assert labels == ["rep1 direction unknown", "rep2 direction unknown"]
    assert "direction unknown" in ax.get_ylabel()


def test_oriented_profiles_label_every_plotted_trace():
    from mcc_peaks.api.junction_plotting import _draw_profile

    profiles = _profiles()
    profiles.loc[0, "Unknown"] = 1
    ax = _axes()

    _draw_profile(ax, profiles, title="oriented")

    labels = [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")]
    assert labels == [
        "rep1 upstream",
        "rep1 downstream",
        "rep1 direction unknown",
        "rep2 upstream",
        "rep2 downstream",
    ]
    assert "upstream +" in ax.get_ylabel()


@pytest.mark.parametrize(
    ("counts", "expected"),
    [((1, 1), ["1"]), ((1, 100), ["1", "10", "100"])],
)
def test_contact_colorbar_labels_are_distinct_plain_counts(counts, expected):
    from mcc_peaks.api.junction_plotting import _draw_contacts

    contacts = pd.concat([_contacts()] * len(counts), ignore_index=True)
    contacts["Count"] = counts
    ax = _axes()

    _draw_contacts(ax, contacts, arrow_length=20)
    colorbar_ax = ax.figure.axes[-1]
    ax.figure.canvas.draw()
    vmin, vmax = colorbar_ax.get_ylim()
    labels = [
        tick.get_text()
        for tick, value in zip(colorbar_ax.get_yticklabels(), colorbar_ax.get_yticks())
        if vmin <= value <= vmax and tick.get_text()
    ]
    assert labels == expected
