import numpy as np
import pandas as pd
import pytest

import lanceotron_mcc.api.counts as counts_api
from lanceotron_mcc.api.adaptive import AdaptiveParams, call_count_manifest
from lanceotron_mcc.api.counts import (
    bh_adjust,
    estimate_dispersion,
    load_manifest,
    negative_binomial_pvalues,
)


def _write_pairs(path, sample, background_events, peak_events=0, peak_position=101):
    with path.open("w") as out:
        for i in range(background_events):
            out.write(f"{sample}-bg{i}\tchr1\t701\tchr1\t401\t+\t-\n")
        for i in range(peak_events):
            out.write(f"{sample}-peak{i}\tchr1\t{peak_position}\tchr1\t401\t+\t-\n")
    return path


def _manifest(tmp_path, samples):
    rows = ["sample\tviewpoint\tpairs"]
    for sample, path in samples:
        rows.append(f"{sample}\tvp\t{path.name}")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(rows) + "\n")
    return manifest


def _params(**kwargs):
    values = dict(
        window_sizes=[50],
        backgrounds=[500],
        thresholds=[2],
        support_fraction=1,
        min_peak_size=50,
        max_peak_size=100,
        bin_size=50,
    )
    values.update(kwargs)
    return AdaptiveParams(**values)


def _call(manifest, **kwargs):
    return call_count_manifest(
        manifest,
        "vp",
        {"chr1": [(400, 450)]},
        {"chr1": kwargs.pop("chrom_size", 1000)},
        params=kwargs.pop("params", _params()),
        **kwargs,
    )


def test_count_pipeline_bins_all_regions_masks_viewpoint_and_reports_pair_events(tmp_path):
    pairs = _write_pairs(tmp_path / "s1.pairs", "s1", background_events=20, peak_events=10)
    manifest = _manifest(tmp_path, [("s1", pairs)])

    peaks, bins, metrics = _call(manifest)

    assert bins.Start.tolist() == list(range(0, 1000, 50))
    assert bins.End.tolist() == list(range(50, 1001, 50))
    assert bins.Eligible.sum() == 19
    assert not bins.loc[bins.Start.eq(400), "Eligible"].item()
    assert peaks.loc[peaks.Start.eq(100), "Count"].item() == 10
    assert metrics["count_unit"] == "reporter event, not independent molecule"
    assert (
        metrics["samples"][0]["count_semantics"]
        == "reporter-end pair events; not deduplicated molecules"
    )


def test_count_pipeline_requires_majority_replicate_support(tmp_path):
    pairs = [
        _write_pairs(tmp_path / "a.pairs", "a", 20, 10),
        _write_pairs(tmp_path / "b.pairs", "b", 20, 10),
        _write_pairs(tmp_path / "c.pairs", "c", 20, 0),
    ]
    manifest = _manifest(tmp_path, [(p.stem, p) for p in pairs])

    peaks, _, _ = _call(manifest)

    target = peaks.loc[peaks.Start.eq(100)].iloc[0]
    assert target.ReplicateSupport == 2
    assert target.Replicates == 3
    assert bool(target.Reproducible)
    assert target.Count == 20


def test_min_read_count_uses_raw_pair_events_after_replicate_normalization(tmp_path):
    pairs = [
        _write_pairs(tmp_path / "low.pairs", "low", 100, 0),
        _write_pairs(tmp_path / "mid.pairs", "mid", 197, 3),
        _write_pairs(tmp_path / "high.pairs", "high", 297, 3),
    ]
    manifest = _manifest(tmp_path, [(p.stem, p) for p in pairs])

    peaks, _, metrics = _call(manifest, params=_params(min_read_count=6))

    target = peaks.loc[peaks.Start.eq(100)].iloc[0]
    assert target.Count == 6
    assert target.ReplicateSupport == 2
    assert metrics["normalization_factors"] == pytest.approx([2, 1, 2 / 3])


def test_count_pipeline_returns_well_formed_empty_output(tmp_path):
    pairs = _write_pairs(tmp_path / "s1.pairs", "s1", background_events=20, peak_events=10)
    manifest = _manifest(tmp_path, [("s1", pairs)])

    peaks, bins, _ = _call(manifest, params=_params(thresholds=[1e9]))

    assert peaks.empty
    assert "Start" in peaks.columns
    assert len(bins) == 20


def test_count_pipeline_accepts_validated_preloaded_manifest_rows(tmp_path, monkeypatch):
    pairs = _write_pairs(tmp_path / "s1.pairs", "s1", background_events=20, peak_events=10)
    manifest = _manifest(tmp_path, [("s1", pairs)])
    rows = load_manifest(manifest, "vp")
    expected_peaks, expected_bins, _ = _call(manifest)

    def fail_if_reloaded(*args, **kwargs):
        raise AssertionError("preloaded manifest rows should avoid parsing the manifest again")

    monkeypatch.setattr(counts_api, "load_manifest", fail_if_reloaded)
    actual_peaks, actual_bins, _ = _call(rows)
    pd.testing.assert_frame_equal(expected_peaks, actual_peaks)
    pd.testing.assert_frame_equal(expected_bins, actual_bins)


def test_count_pipeline_partial_final_bin_bounds_and_count(tmp_path):
    pairs = _write_pairs(
        tmp_path / "s1.pairs", "s1", background_events=20, peak_events=1, peak_position=1001
    )
    manifest = _manifest(tmp_path, [("s1", pairs)])

    _, bins, _ = _call(manifest, chrom_size=1023)

    assert bins.iloc[-1][["Start", "End"]].tolist() == [1000, 1023]
    assert bins.iloc[-1]["Count:s1"] == 1


def test_count_api_rejects_out_of_bounds_exclusions(tmp_path):
    pairs = _write_pairs(tmp_path / "s1.pairs", "s1", background_events=20)
    manifest = _manifest(tmp_path, [("s1", pairs)])
    with pytest.raises(ValueError, match="outside chr1 bounds"):
        _call(manifest, exclusions={"chr1": [(1000, 1001)]})


def test_warm_count_cache_preserves_qc_and_outputs_without_source_read(tmp_path, monkeypatch):
    pairs = _write_pairs(tmp_path / "s1.pairs", "s1", background_events=20, peak_events=10)
    manifest = _manifest(tmp_path, [("s1", pairs)])
    cache_dir = tmp_path / "cache"

    first_peaks, first_bins, first_metrics = _call(manifest, cache_dir=cache_dir)

    def fail_if_read(*args, **kwargs):
        raise AssertionError("warm pair-count cache should avoid rereading the source")

    monkeypatch.setattr(counts_api, "count_pairs", fail_if_read)
    second_peaks, second_bins, second_metrics = _call(manifest, cache_dir=cache_dir)

    pd.testing.assert_frame_equal(first_peaks, second_peaks)
    pd.testing.assert_frame_equal(first_bins, second_bins)
    assert first_metrics["samples"][0]["usable_cis_depth"] == 30
    warm_qc = second_metrics["samples"][0]
    assert warm_qc["usable_cis_depth"] == first_metrics["samples"][0]["usable_cis_depth"]
    assert (
        warm_qc["retained_reporter_events"]
        == first_metrics["samples"][0]["retained_reporter_events"]
    )
    assert warm_qc["cached"] is True


def test_zero_background_assigns_zero_tail_only_for_positive_counts():
    p = negative_binomial_pvalues(np.array([0, 1, 4]), np.zeros(3), dispersion=0.2)
    assert p.tolist() == [1.0, 0.0, 0.0]


def test_overdispersed_null_has_calibrated_tails_and_no_bh_false_positives():
    rng = np.random.default_rng(1337)
    expected = np.full(30_000, 5.0)
    observed = rng.negative_binomial(4, 4 / 9, size=expected.size)
    dispersion = estimate_dispersion(observed, expected)
    pvalues = negative_binomial_pvalues(observed, expected, dispersion)
    qvalues = bh_adjust(pvalues)

    assert dispersion == pytest.approx(0.25, abs=0.04)
    assert pvalues.mean() == pytest.approx(0.5, abs=0.08)
    assert np.count_nonzero(qvalues <= 0.05) <= 1
