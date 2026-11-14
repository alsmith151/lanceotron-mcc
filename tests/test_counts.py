import gzip

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from lanceotron_mcc.api.counts import (
    bh_adjust,
    combine_replicate_pvalues,
    count_pairs,
    estimate_dispersion,
    iter_pairs,
    iter_viewpoint_contacts,
    load_manifest,
    negative_binomial_pvalues,
    validate_manifest_frame,
)


def test_manifest_filters_viewpoint_and_resolves_paths(tmp_path):
    (tmp_path / "a.pairs.gz").touch()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\ns1\tvp1\ta.pairs.gz\ns2\tvp2\tb.pairs.gz\n")

    result = load_manifest(manifest, "vp1")

    assert result[["sample", "viewpoint"]].values.tolist() == [["s1", "vp1"]]
    assert result.loc[0, "pairs"] == str(tmp_path / "a.pairs.gz")


def test_manifest_rejects_duplicate_samples_for_viewpoint(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\ns1\tvp1\ta\ns1\tvp1\tb\n")
    with pytest.raises(ValueError, match="Duplicate sample"):
        load_manifest(manifest, "vp1")


def test_manifest_rejects_duplicate_column_names(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\tpairs\ns1\tvp1\ta\tb\n")
    with pytest.raises(ValueError, match="duplicate column"):
        load_manifest(manifest, "vp1")


def test_manifest_rejects_missing_pairs_file(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\ns1\tvp1\tmissing.pairs\n")
    with pytest.raises(ValueError, match="Pairs file.*do not exist"):
        load_manifest(manifest, "vp1")


def test_manifest_preserves_literal_na_values(tmp_path):
    (tmp_path / "NA").touch()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\nNA\tvp1\tNA\n")

    result = load_manifest(manifest, "vp1")

    assert result.loc[0, "sample"] == "NA"
    assert result.loc[0, "pairs"] == str(tmp_path / "NA")


def test_manifest_rejects_hardlinked_pairs_files(tmp_path):
    first = tmp_path / "a.pairs"
    second = tmp_path / "b.pairs"
    first.write_text("same file")
    second.hardlink_to(first)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\ns1\tvp1\ta.pairs\ns2\tvp1\tb.pairs\n")

    with pytest.raises(ValueError, match="distinct pairs file"):
        load_manifest(manifest, "vp1")


def test_validate_manifest_frame_applies_path_and_row_validation(tmp_path):
    (tmp_path / "NA").touch()
    frame = pd.DataFrame(
        {"sample": ["NA", "other"], "viewpoint": ["vp1", "vp2"], "pairs": ["NA", "missing"]}
    )

    result = validate_manifest_frame(frame, "vp1", base_dir=tmp_path)

    assert result.to_dict("records") == [
        {"sample": "NA", "viewpoint": "vp1", "pairs": str(tmp_path / "NA")}
    ]


def test_manifest_rejects_same_resolved_pairs_path_for_distinct_samples(tmp_path):
    (tmp_path / "a.pairs").touch()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample\tviewpoint\tpairs\ns1\tvp1\ta.pairs\ns2\tvp1\t./a.pairs\n")

    with pytest.raises(ValueError, match="distinct pairs file"):
        load_manifest(manifest, "vp1")


def test_pair_events_are_zero_based_binned_and_exclusions_are_reporter_side(tmp_path):
    pairs = tmp_path / "sample.pairs.gz"
    with gzip.open(pairs, "wt") as out:
        out.write("#pairs format v1\n")
        out.write("r1\tchr1\t1\tchr1\t201\t+\t-\n")
        out.write("r2\tchr1\t51\tchr1\t250\t+\t-\n")
        out.write("r3\tchr1\t101\tchr1\t201\t+\t-\n")
        out.write("r4\tchr2\t1\tchr1\t201\t+\t-\n")

    counts, qc = count_pairs(
        pairs,
        {"chr1": 300, "chr2": 100},
        {"chr1": [(200, 251)]},
        bin_size=50,
        excluded={"chr1": [(50, 100)]},
    )

    assert counts.tolist() == [1, 0, 1, 0, 0, 0]
    assert qc["input_pairs"] == 4
    assert qc["trans_pairs"] == 1
    assert qc["usable_cis_depth"] == 3
    assert qc["excluded_reporter_events"] == 1
    assert qc["retained_reporter_events"] == 2
    assert "not deduplicated molecules" in qc["count_semantics"]


def test_contact_iterator_yields_zero_based_cis_and_trans_contacts_with_qc(tmp_path):
    pairs = tmp_path / "contacts.pairs"
    pairs.write_text("cis\tchr1\t111\tchr1\t1\t+\t-\ntrans\tchr2\t5\tchr1\t112\t-\t+\n")
    qc = {}
    contacts = list(
        iter_viewpoint_contacts(pairs, {"chr1": 300, "chr2": 100}, {"chr1": [(100, 120)]}, qc=qc)
    )
    assert [
        (c.read_id, c.capture_chrom, c.capture_pos, c.reporter_chrom, c.reporter_pos)
        for c in contacts
    ] == [
        ("cis", "chr1", 110, "chr1", 0),
        ("trans", "chr1", 111, "chr2", 4),
    ]
    assert [(c.capture_strand, c.reporter_strand) for c in contacts] == [("+", "-"), ("+", "-")]
    assert qc["input_pairs"] == 2
    assert qc["trans_pairs"] == 1
    assert qc["usable_cis_depth"] == 1


def test_raw_pair_iterator_keeps_tiled_pool_contacts_without_oligo_endpoint(tmp_path):
    pairs = tmp_path / "tiled.pairs"
    pairs.write_text(
        "#shape: upper triangle\n"
        "pool-read__Hba-a1-chr11-32200000-32200020__start__1__0\t"
        "chr1\t3300000\tchr1\t3300150\t+\t-\n"
    )
    qc = {}
    records = list(
        iter_pairs(
            pairs,
            {"chr1": 10_000_000, "chr11": 50_000_000},
            qc=qc,
        )
    )
    assert len(records) == 1
    record = records[0]
    assert (record.chrom1, record.pos1, record.chrom2, record.pos2) == (
        "chr1",
        3_299_999,
        "chr1",
        3_300_149,
    )
    assert record.metadata is not None
    assert (record.metadata.viewpoint, record.metadata.chromosome) == ("Hba-a1", "chr11")
    assert record.shape == "upper triangle"
    assert qc["input_pairs"] == 1
    assert qc["mcc_annotated_pairs"] == 1


def test_pair_parser_rejects_bad_column_count_and_out_of_bounds_coordinate(tmp_path):
    pairs = tmp_path / "bad.pairs"
    pairs.write_text("r1\tchr1\t1\tchr1\t101\t+\n")
    kwargs = ({"chr1": 200}, {"chr1": [(100, 120)]})
    with pytest.raises(ValueError, match="at least 7"):
        count_pairs(pairs, *kwargs)

    pairs.write_text("r1\tchr1\t0\tchr1\t101\t+\t-\n")
    with pytest.raises(ValueError, match="outside"):
        count_pairs(pairs, *kwargs)


def test_pair_parser_skips_pairs_without_viewpoint_overlap(tmp_path):
    pairs = tmp_path / "bad.pairs"
    pairs.write_text("r1\tchr1\t1\tchr1\t101\t+\t-\n")
    counts, qc = count_pairs(pairs, {"chr1": 200}, {"chr1": [(110, 120)]})
    assert counts.sum() == 0
    assert qc["non_viewpoint_pairs"] == 1


def test_standard_pairs_header_optional_columns_and_either_orientation(tmp_path):
    pairs = tmp_path / "sample.pairs"
    pairs.write_text(
        "#pairs format v1.0.0\n"
        "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2 pair_type chrom1\n"
        "r1\tchr1\t1\tchr1\t201\t+\t-\tUU\tchrX\n"
        "r2\tchr1\t251\tchr1\t201\t+\t-\tUU\tchrX\n"
        "r3\tchr2\t5\tchr1\t201\t+\t-\tUU\tchrX\n"
        "r4\tchr1\t201\tchr2\t5\t+\t-\tUU\tchrX\n"
    )
    counts, qc = count_pairs(pairs, {"chr1": 300, "chr2": 100}, {"chr1": [(200, 220)]}, bin_size=50)
    assert counts.tolist() == [1, 0, 0, 0, 0, 1]
    assert qc["input_pairs"] == 4
    assert qc["capture_end_1_pairs"] == 1
    assert qc["capture_end_2_pairs"] == 3
    assert qc["trans_pairs"] == 2


def test_pair_parser_skips_both_viewpoint_unmapped_and_genomewide_pairs(tmp_path):
    pairs = tmp_path / "sample.pairs"
    pairs.write_text(
        "r1\tchr1\t201\tchr1\t205\t+\t-\nr2\tchr1\t1\tchr2\t1\t+\t-\nr3\t!\t0\tchr1\t201\t.\t-\n"
    )
    counts, qc = count_pairs(pairs, {"chr1": 300, "chr2": 100}, {"chr1": [(200, 220)]})
    assert counts.sum() == 0
    assert qc["input_pairs"] == 3
    assert qc["both_viewpoint_pairs"] == 1
    assert qc["non_viewpoint_pairs"] == 1
    assert qc["unmapped_pairs"] == 1


def test_pair_parser_preserves_blank_read_id_and_validates_header_width(tmp_path):
    pairs = tmp_path / "blank_id.pairs"
    pairs.write_text(
        "#columns: read_id chr1 pos1 chr2 pos2 strand1 strand2\n\tchr1\t1\tchr1\t201\t.\t.\n"
    )
    counts, qc = count_pairs(pairs, {"chr1": 300}, {"chr1": [(200, 220)]})
    assert counts.sum() == 1
    assert qc["input_pairs"] == 1

    pairs.write_text(
        "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2 pair_type\n"
        "r1\tchr1\t1\tchr1\t201\t+\t-\n"
    )
    with pytest.raises(ValueError, match="header declares 8"):
        count_pairs(pairs, {"chr1": 300}, {"chr1": [(200, 220)]})


@pytest.mark.parametrize(
    "header, message",
    [
        ("#columns: readID chr1 pos1 chr2 pos2 strand1\n", "at least 7"),
        ("#columns: readID chr2 pos1 chr1 pos2 strand1 strand2\n", "first 7 fields"),
        ("#columns: readID chr1 pos1 chr2 pos2 strand1 strand2 strand2\n", "duplicate"),
    ],
)
def test_pair_parser_rejects_malformed_columns_headers(tmp_path, header, message):
    pairs = tmp_path / "bad.pairs"
    pairs.write_text(header)
    with pytest.raises(ValueError, match=message):
        count_pairs(pairs, {"chr1": 300}, {"chr1": [(200, 220)]})


def test_pair_parser_rejects_invalid_unmapped_and_mapped_zero_positions(tmp_path):
    pairs = tmp_path / "bad.pairs"
    kwargs = ({"chr1": 300}, {"chr1": [(200, 220)]})
    pairs.write_text("r1\t!\t1\tchr1\t201\t.\t-\n")
    with pytest.raises(ValueError, match="must have position 0"):
        count_pairs(pairs, *kwargs)
    pairs.write_text("r1\tchr1\t0\t!\t0\t+\t.\n")
    with pytest.raises(ValueError, match="outside"):
        count_pairs(pairs, *kwargs)


def test_mcc_metadata_orients_capture_and_accepts_oligo_end_boundary(tmp_path):
    pairs = tmp_path / "mcc.pairs"
    pairs.write_text(
        "parent-name__vp-name-chr-long-100-120__start__1__0\tchr-long\t121\tchr-long\t41\t+\t-\n"
    )
    counts, qc = count_pairs(pairs, {"chr-long": 500}, {"chr-long": [(100, 120)]}, bin_size=50)
    assert counts[0] == 1
    assert qc["mcc_annotated_pairs"] == 1
    assert qc["mcc_start_pairs"] == 1
    assert qc["capture_end_1_pairs"] == 1


def test_mcc_metadata_does_not_deduplicate_parent_and_skips_both_oligo_ends(tmp_path):
    pairs = tmp_path / "mcc.pairs"
    read_id = "parent__vp-chr1-100-120__end__2__1"
    pairs.write_text(
        f"{read_id}\tchr1\t41\tchr1\t121\t+\t-\n{read_id}\tchr1\t51\tchr1\t121\t+\t-\n"
    )
    counts, qc = count_pairs(pairs, {"chr1": 500}, {"chr1": [(100, 120)]}, bin_size=50)
    assert counts[0] == 1
    assert counts[1] == 1
    assert qc["mcc_annotated_pairs"] == 2
    assert qc["mcc_end_pairs"] == 2

    pairs.write_text(f"{read_id}\tchr1\t101\tchr1\t121\t+\t-\n")
    counts, qc = count_pairs(pairs, {"chr1": 500}, {"chr1": [(100, 120)]}, bin_size=50)
    assert counts.sum() == 0
    assert qc["self_contact_pairs"] == 1
    assert qc["both_viewpoint_pairs"] == 1


def test_mcc_metadata_mismatch_skips_even_when_standard_end_overlaps(tmp_path):
    pairs = tmp_path / "mcc.pairs"
    pairs.write_text("parent__bait-chr1-100-120__start__1__0\tchr1\t301\tchr1\t41\t+\t-\n")
    counts, qc = count_pairs(pairs, {"chr1": 500}, {"chr1": [(300, 320)]}, bin_size=50)
    assert counts.sum() == 0
    assert qc["non_viewpoint_pairs"] == 1


def test_standard_pairs_keep_half_open_boundary_behavior(tmp_path):
    pairs = tmp_path / "standard.pairs"
    pairs.write_text("plain-id\tchr1\t41\tchr1\t121\t+\t-\n")
    counts, qc = count_pairs(pairs, {"chr1": 500}, {"chr1": [(100, 120)]}, bin_size=50)
    assert counts.sum() == 0
    assert qc["non_viewpoint_pairs"] == 1


@pytest.mark.parametrize(
    "read_id, message",
    [
        ("parent__vp-chr1-0-0__start__1__0", "Invalid MCC viewpoint interval"),
        ("parent__vp-chr2-100-120__start__1__0", "Could not resolve chromosome"),
        ("parent__vp-chr1-100-120__all__1__0", "unsupported reporter side 'all'"),
    ],
)
def test_mcc_metadata_rejects_invalid_annotation(tmp_path, read_id, message):
    pairs = tmp_path / "mcc.pairs"
    pairs.write_text(f"{read_id}\tchr1\t41\tchr1\t121\t+\t-\n")
    with pytest.raises(ValueError, match=message):
        count_pairs(pairs, {"chr1": 500}, {"chr1": [(100, 120)]})


def test_mcc_metadata_oligo_must_match_a_pair_end(tmp_path):
    pairs = tmp_path / "mcc.pairs"
    pairs.write_text("parent__vp-chr1-100-120__start__1__0\tchr1\t41\tchr1\t301\t+\t-\n")
    with pytest.raises(ValueError, match="matches neither pair end"):
        count_pairs(pairs, {"chr1": 500}, {"chr1": [(100, 120)]})


def test_negative_binomial_tail_uses_nb2_parameterization():
    observed = np.array([0, 5, 10])
    expected = np.array([0.0, 4.0, 8.0])
    dispersion = 0.2
    pvalues = negative_binomial_pvalues(observed, expected, dispersion)
    size = 1 / dispersion
    expected_p = nbinom.sf(4, size, size / (size + 4))
    assert pvalues[0] == 1.0
    assert pvalues[1] == pytest.approx(expected_p)
    assert 0 <= pvalues[2] <= 1
    assert estimate_dispersion(np.array([1, 1]), np.array([0.1, 0.1])) == 0.0


def test_fisher_combination_and_bh_adjustment():
    combined = combine_replicate_pvalues(np.array([[0.01, 0.2, 1.0], [0.02, 0.4, 1.0]]))
    assert combined.shape == (3,)
    assert combined[0] < combined[1] < combined[2]
    adjusted = bh_adjust(np.array([0.01, 0.04, 0.03, 0.9]), eligible=np.array([1, 1, 1, 0]))
    assert adjusted.tolist() == pytest.approx([0.03, 0.04, 0.04, 1.0])


def test_stats_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        estimate_dispersion(np.array([1.0, np.nan]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        combine_replicate_pvalues(np.array([[0.1], [np.inf]]))
