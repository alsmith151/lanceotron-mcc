from pathlib import Path

import numpy as np
import pandas as pd
import pybigtools
import pytest

from mcc_peaks.api.adaptive import (
    AdaptiveParams,
    _call_arrays,
    _split_valleys,
    call_adaptive_bigwigs,
    read_bed,
)
from mcc_peaks.api.background import (
    fit_distance_background,
    interval_mask,
    regions,
)
from mcc_peaks.api.io import ArrayCache


def write_bigwig(path: Path, values):
    values = np.asarray(values, dtype=np.float64)
    writer = pybigtools.open(str(path), "w")
    writer.write(
        {"chr1": len(values)},
        (("chr1", i, i + 1, float(value)) for i, value in enumerate(values) if value != 0),
    )


def params(**overrides):
    values = dict(
        window_sizes=[5, 11],
        backgrounds=[51, 101],
        thresholds=[1.5, 2],
        support_fraction=0.25,
        min_peak_size=3,
        max_peak_size=50,
        bin_size=5,
        chunk_size=53,
        memory_budget_mb=16,
    )
    values.update(overrides)
    return AdaptiveParams(**values)


def synthetic_signal(scale=1.0):
    signal = np.ones(500, dtype=np.float64)
    signal[100:120] = 10
    signal[300:320] = 8
    return signal * scale


def test_distance_background_is_monotone_on_both_viewpoint_arms():
    viewpoint = [(100, 110)]
    positions = np.array([99, 95, 90, 80, 60, 30, 0, 111, 115, 120, 130, 150, 180, 210])
    distances = np.maximum(np.maximum(100 - positions, positions - 110), 0)
    signal = 1 + 30 / (distances + 10)
    signal[[3, 10]] += 100  # Isolated peaks should not reverse the fitted trend.

    background = fit_distance_background(signal, positions, viewpoint)
    assert np.all(np.isfinite(background))
    for arm_positions, arm_background in (
        (positions[:7], background[:7]),
        (positions[7:], background[7:]),
    ):
        arm_distances = np.maximum(np.maximum(100 - arm_positions, arm_positions - 110), 0)
        ordered = arm_background[np.argsort(arm_distances)]
        assert np.all(np.diff(ordered) <= 1e-12)


def test_background_remains_finite_for_zero_signal_tail():
    positions = np.arange(0, 200, 10, dtype=np.float64)
    signal = np.zeros_like(positions)
    signal[0:3] = [4, 2, 1]
    background = fit_distance_background(signal, positions, [(0, 10)])
    assert np.all(np.isfinite(background))
    assert np.all(background >= 0)
    assert np.all(background[positions >= 50] > 0)


def test_background_rejects_invalid_fit_weights_and_empty_eligibility():
    signal = np.ones(4)
    positions = np.arange(4, dtype=float)
    with pytest.raises(ValueError, match="weights must match"):
        fit_distance_background(signal, positions, [(0, 1)], weights=np.ones(3))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fit_distance_background(signal, positions, [(0, 1)], weights=[1, 1, np.nan, 1])
    with pytest.raises(ValueError, match="No eligible"):
        fit_distance_background(signal, positions, [(0, 1)], eligible=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="viewpoint arm -1"):
        fit_distance_background(
            np.ones(8),
            np.arange(8, dtype=float),
            [(3, 5)],
            eligible=np.array([False, False, False, False, False, True, True, True]),
        )


def test_read_bed_allows_empty_exclusions_and_requires_one_viewpoint_group(tmp_path):
    empty = tmp_path / "empty.bed"
    empty.write_text("# no exclusions\n")
    assert read_bed(empty, allow_empty=True) == {}
    with pytest.raises(ValueError, match="empty"):
        read_bed(empty)

    viewpoints = tmp_path / "viewpoints.bed"
    viewpoints.write_text("chr1\t10\t20\tA\nchr1\t30\t40\tB\n")
    with pytest.raises(ValueError, match="multiple viewpoint groups"):
        read_bed(viewpoints, require_single_group=True)

    aliases = tmp_path / "aliases.bed"
    aliases.write_text("chr1\t10\t20\tA\nchr1\t30\t40\tA-chr1-30-40\n")
    assert len(read_bed(aliases, require_single_group=True)["chr1"]) == 2


def test_regions_use_int64_coordinates_and_masks_exclude_viewpoint_and_intervals():
    intervals = regions([False, True, True, False, True])
    assert intervals.dtype == np.int64
    np.testing.assert_array_equal(intervals, [[1, 3], [4, 5]])

    valid = interval_mask(12, [(0, 3), (7, 10)])
    np.testing.assert_array_equal(
        valid, [False, False, False, True, True, True, True, False, False, False, True, True]
    )


def test_adaptive_calls_respect_viewpoint_and_exclusion_masks(tmp_path):
    path = tmp_path / "sample.bw"
    write_bigwig(path, synthetic_signal())
    result, _ = call_adaptive_bigwigs(
        [path],
        {"chr1": [(0, 10)]},
        exclusions={"chr1": [(100, 120)]},
        params=params(),
    )

    assert not result.empty
    assert ((result.Start >= 0) & (result.End <= 10)).sum() == 0
    assert ((result.Start < 120) & (result.End > 100)).sum() == 0
    assert result.Start.tolist() == [299]


def test_adaptive_api_rejects_out_of_bounds_exclusions(tmp_path):
    path = tmp_path / "sample.bw"
    write_bigwig(path, synthetic_signal())
    with pytest.raises(ValueError, match="outside chr1 bounds"):
        call_adaptive_bigwigs(
            [path],
            {"chr1": [(0, 10)]},
            exclusions={"chr1": [(500, 501)]},
            params=params(),
        )


def test_adaptive_results_match_across_small_and_whole_chromosome_chunks(tmp_path):
    path = tmp_path / "sample.bw"
    write_bigwig(path, synthetic_signal())
    viewpoints = {"chr1": [(0, 10)]}

    small, _ = call_adaptive_bigwigs([path], viewpoints, params=params(chunk_size=53))
    whole, _ = call_adaptive_bigwigs([path], viewpoints, params=params(chunk_size=1000))
    pd.testing.assert_frame_equal(small, whole, check_exact=True)


def test_even_and_global_windows_are_chunk_stable_and_ignore_excluded_signal(tmp_path):
    baseline = np.ones(300, dtype=np.float64)
    baseline[150:170] = 10
    excluded_peak = baseline.copy()
    excluded_peak[30:50] = 1000
    baseline_path = tmp_path / "baseline" / "sample.bw"
    excluded_path = tmp_path / "excluded" / "sample.bw"
    baseline_path.parent.mkdir()
    excluded_path.parent.mkdir()
    write_bigwig(baseline_path, baseline)
    write_bigwig(excluded_path, excluded_peak)
    viewpoints = {"chr1": [(0, 10)]}
    exclusions = {"chr1": [(30, 50)]}
    common = dict(
        window_sizes=[4, 10],  # Even windows exercise the left-heavy center convention.
        backgrounds=[301],  # Larger than the chromosome, so the global fallback is used.
        thresholds=[1.5, 2],
        support_fraction=0.25,
        min_peak_size=3,
        max_peak_size=50,
        bin_size=5,
        memory_budget_mb=16,
    )

    baseline_chunked, _ = call_adaptive_bigwigs(
        [baseline_path],
        viewpoints,
        exclusions=exclusions,
        params=params(**common, chunk_size=53),
    )
    excluded_chunked, _ = call_adaptive_bigwigs(
        [excluded_path],
        viewpoints,
        exclusions=exclusions,
        params=params(**common, chunk_size=53),
    )
    excluded_whole, _ = call_adaptive_bigwigs(
        [excluded_path],
        viewpoints,
        exclusions=exclusions,
        params=params(**common, chunk_size=1000),
    )

    pd.testing.assert_frame_equal(baseline_chunked, excluded_chunked, check_exact=True)
    pd.testing.assert_frame_equal(excluded_chunked, excluded_whole, check_exact=True)


def test_background_profile_resolution_does_not_depend_on_memory_budget(tmp_path):
    path = tmp_path / "sample.bw"
    write_bigwig(path, synthetic_signal())
    viewpoints = {"chr1": [(0, 10)]}
    outputs = []
    resolutions = []
    for budget in (16, 32):
        result, metrics = call_adaptive_bigwigs(
            [path], viewpoints, params=params(memory_budget_mb=budget)
        )
        outputs.append(result)
        resolutions.append(metrics["background_bin_size"])

    expected_profile = int(np.ceil(200 / 5) * 5)
    assert resolutions == [expected_profile, expected_profile]
    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)


def test_replicates_receive_equal_depth_normalization(tmp_path):
    first = tmp_path / "first.bw"
    second = tmp_path / "second.bw"
    write_bigwig(first, synthetic_signal(1))
    write_bigwig(second, synthetic_signal(3))

    result, metrics = call_adaptive_bigwigs([first, second], {"chr1": [(0, 10)]}, params=params())
    factors = np.asarray(metrics["normalization_factors"])
    depths = np.asarray([synthetic_signal(1).sum(), synthetic_signal(3).sum()])
    np.testing.assert_allclose(depths * factors, np.median(depths))
    assert result.Reproducible.all()
    np.testing.assert_allclose(result["Signal:first"], result["Signal:second"])


def test_replicate_chromosome_length_mismatch_is_rejected(tmp_path):
    first = tmp_path / "first.bw"
    second = tmp_path / "second.bw"
    write_bigwig(first, synthetic_signal())
    write_bigwig(second, synthetic_signal()[:-1])

    with pytest.raises(ValueError, match="matching lengths"):
        call_adaptive_bigwigs([first, second], {"chr1": [(0, 10)]}, params=params())


def test_hardlinked_bigwig_cannot_masquerade_as_two_replicates(tmp_path):
    first = tmp_path / "first.bw"
    alias = tmp_path / "second.bw"
    write_bigwig(first, synthetic_signal())
    alias.hardlink_to(first)
    with pytest.raises(ValueError, match="hard links"):
        call_adaptive_bigwigs([first, alias], {"chr1": [(0, 10)]}, params=params())


def test_continuous_background_is_scale_equivariant():
    positions = np.arange(200, 50_000, 200)
    signal = 0.02 + 0.5 * np.exp(-positions / 5000)
    signal[20:23] += 0.3
    first = fit_distance_background(signal, positions, [(0, 100)])
    second = fit_distance_background(signal * 1000, positions, [(0, 100)])
    np.testing.assert_allclose(second, first * 1000, rtol=1e-12)


def test_sparse_background_floor_does_not_overflow_enrichment():
    positions = np.arange(200, 10_000, 200)
    signal = np.zeros(len(positions))
    signal[5] = 100
    background = fit_distance_background(
        signal, positions, [(0, 100)], count_mode=True, weights=np.full(len(positions), 200)
    )
    assert np.all(np.isfinite(signal / background))


def test_valley_splitting_is_independent_of_chunk_and_memory_settings():
    values = np.zeros(120)
    values[5:45] = 10
    values[75:115] = 10
    calls = [(0, len(values), True)]
    outputs = []
    for chunk_size, budget in ((1, 16), (7, 32), (1000, 16), (1000, 32)):
        outputs.append(
            _split_valleys(
                calls,
                lambda start, end: values[start:end],
                params(
                    window_sizes=[1],
                    chunk_size=chunk_size,
                    memory_budget_mb=budget,
                    min_peak_size=10,
                    max_peak_size=30,
                ),
                unit=1,
            )
        )
    assert all(output == outputs[0] for output in outputs[1:])
    assert len(outputs[0]) == 2


def test_candidate_pipeline_rejects_misaligned_background_profiles(tmp_path):
    with pytest.raises(ValueError, match="match the chromosome"):
        _call_arrays(
            arrays=[np.ones(10)],
            expected=[np.ones(1)],
            exclusions=[],
            chrom="chr1",
            chrom_length=10,
            unit=1,
            params=params(bin_size=5),
            tmp=tmp_path,
            identities=[],
            cache=ArrayCache(1),
            sample_names=["sample"],
        )


def test_candidate_pipeline_rejects_nonfinite_replicate_signal(tmp_path):
    values = np.ones(10)
    values[4] = np.nan
    with pytest.raises(ValueError, match="finite nonnegative"):
        _call_arrays(
            arrays=[values],
            expected=[np.ones(2)],
            exclusions=[],
            chrom="chr1",
            chrom_length=10,
            unit=1,
            params=params(bin_size=5),
            tmp=tmp_path,
            identities=[],
            cache=ArrayCache(1),
            sample_names=["sample"],
        )


def test_sparse_empty_chunks_preserve_foreground_halos(tmp_path):
    path = tmp_path / "sparse.bw"
    signal = np.zeros(500)
    signal[104:108] = 10  # Cross the 53 bp chunk edge after smoothing.
    signal[350:354] = 10
    write_bigwig(path, signal)
    options = dict(window_sizes=[4, 10], backgrounds=[51, 101], background_bin_size=5)
    small, _ = call_adaptive_bigwigs(
        [path], {"chr1": [(0, 10)]}, params=params(**options, chunk_size=53)
    )
    whole, _ = call_adaptive_bigwigs(
        [path], {"chr1": [(0, 10)]}, params=params(**options, chunk_size=1000)
    )
    assert not small.empty
    pd.testing.assert_frame_equal(small, whole, check_exact=True)


@pytest.mark.parametrize("window", [1, 2, 3, 4, 20, 51, 100])
@pytest.mark.parametrize("masked", [False, True])
def test_shared_prefix_smoothing_matches_reference(window, masked):
    from mcc_peaks.api.background import average_from_prefix, moving_average, rolling_prefix

    rng = np.random.default_rng(12)
    values = rng.uniform(0, 20, 51)
    valid = rng.random(51) > 0.3 if masked else None
    sums, counts = rolling_prefix(values, valid)
    np.testing.assert_allclose(
        average_from_prefix(sums, window, counts),
        moving_average(values, window, valid),
        rtol=1e-12,
        atol=1e-12,
    )


def test_bed_preserves_literal_na_chromosome_and_group_names(tmp_path):
    from mcc_peaks.api.adaptive import read_bed

    bed = tmp_path / "viewpoint.bed"
    bed.write_text("NA\t0\t10\tNA\n")
    assert read_bed(bed, group="NA", require_single_group=True) == {"NA": [(0, 10)]}


def test_profile_resolution_only_rounds_to_actual_observation_units():
    from mcc_peaks.api.adaptive import _profile_bin_size

    options = params(background_bin_size=110, bin_size=50)
    assert _profile_bin_size(1000, options) == 110
    assert _profile_bin_size(1000, options, unit=50) == 150
