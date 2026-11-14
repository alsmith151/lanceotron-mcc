import pandas as pd
import pytest
import torch

from lanceotron_mcc.api import candidates


def test_params_reject_invalid_grid_and_keep_numeric_floor():
    with pytest.raises(ValueError):
        candidates.CandidatePeakCallParams(window_sizes=[0])
    with pytest.raises(ValueError):
        candidates.CandidatePeakCallParams(threshold_initial=28)
    with pytest.raises(ValueError):
        candidates.CandidatePeakCallParams(min_peak_size=20, max_peak_size=10)
    params = candidates.CandidatePeakCallParams(
        threshold_background_lst=[10],
        window_sizes=[2],
        threshold_lst=[2],
        threshold_initial=1,
        background_global_min=0.25,
    )
    assert params.background_global_min == 0.25


def test_param_grid_reuses_smoothing_and_counts_votes_as_int32(monkeypatch):
    calls = []

    def moving_average(array, window):
        calls.append(window)
        return array

    monkeypatch.setattr(candidates, "average_array", moving_average)
    params = candidates.EnrichedRegionsParams(
        background_list=[10, 20],
        window_list=[2, 2],
        threshold_list=[1, 2],
        threshold_cumulative=1,
        return_threshold_array=True,
    )
    result = candidates.search_enriched_regions_param_grid(torch.ones(8), params)
    assert calls.count(2) == 1
    assert calls.count(10) == 1
    assert calls.count(20) == 1
    assert result.threshold_array.dtype == torch.int32
    assert result.threshold_array.tolist() == [4] * 8


def test_param_grid_requires_positive_signal_and_inclusive_cutoff(monkeypatch):
    def moving_average(array, window):
        return torch.ones_like(array) if window == 2 else array

    monkeypatch.setattr(candidates, "average_array", moving_average)
    params = candidates.EnrichedRegionsParams(
        background_list=[2],
        window_list=[1],
        threshold_list=[1],
        threshold_cumulative=1,
        background_global_min=0,
        return_threshold_array=True,
    )
    result = candidates.search_enriched_regions_param_grid(torch.tensor([0.0, 1.0, 0.999]), params)
    assert result.threshold_array.tolist() == [0, 1, 0]
    assert result.enriched_regions.tolist() == [[1, 2]]


def test_param_grid_honors_max_region_size():
    params = candidates.EnrichedRegionsParams(
        background_list=[10],
        window_list=[2],
        threshold_list=[1],
        threshold_cumulative=1,
        max_region_size=2,
    )
    result = candidates.search_enriched_regions_param_grid(torch.ones(5), params)
    assert result.enriched_regions.tolist() == []


def test_candidate_call_rejects_nonfinite_coverage():
    params = candidates.CandidatePeakCallParams()
    caller = candidates.ChromosomePeakCaller("chr1", torch.tensor([1.0, float("nan")]), params)
    with pytest.raises(ValueError, match="finite"):
        caller.call_candidates()


def test_peak_refinement_splits_oversized_region_and_reports_signal(monkeypatch):
    votes = torch.tensor([3, 3, 3, 3, 1, 1, 1, 3, 3, 3, 3], dtype=torch.int32)

    def grid(*args, **kwargs):
        return candidates.EnrichedRegionsResult(
            enriched_regions=torch.tensor([[0, 11]], dtype=torch.int64),
            threshold_array=votes,
        )

    monkeypatch.setattr(candidates, "search_enriched_regions_param_grid", grid)
    params = candidates.CandidatePeakCallParams(
        threshold_background_lst=[10],
        window_sizes=[2],
        threshold_lst=[1, 2, 3],
        threshold_initial=1,
        min_peak_size=2,
        max_peak_size=5,
        background_global_min=0,
    )
    result = candidates.ChromosomePeakCaller(
        "chr1", torch.tensor([1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8]), params
    ).call_candidates()
    assert result[["Start", "End"]].values.tolist() == [[0, 4], [7, 11]]
    assert result.Unresolved.tolist() == [False, False]
    assert result.Summit.tolist() == [3, 10]
    assert result.Signal.tolist() == [2.5, 6.5]


def test_peak_refinement_marks_unsplittable_long_region_unresolved(monkeypatch):
    votes = torch.full((10,), 2, dtype=torch.int32)
    monkeypatch.setattr(
        candidates,
        "search_enriched_regions_param_grid",
        lambda *args, **kwargs: candidates.EnrichedRegionsResult(
            enriched_regions=torch.tensor([[0, 10]]), threshold_array=votes
        ),
    )
    params = candidates.CandidatePeakCallParams(
        threshold_background_lst=[10],
        window_sizes=[2],
        threshold_lst=[1, 2],
        threshold_initial=1,
        min_peak_size=1,
        max_peak_size=3,
        background_global_min=0,
    )
    result = candidates.ChromosomePeakCaller("chr2", torch.ones(10), params).call_candidates()
    assert result.Unresolved.tolist() == [True]


def test_refinement_continues_when_first_higher_cut_keeps_parent(monkeypatch):
    votes = torch.tensor([3, 3, 2, 2, 3, 3], dtype=torch.int32)
    monkeypatch.setattr(
        candidates,
        "search_enriched_regions_param_grid",
        lambda *args, **kwargs: candidates.EnrichedRegionsResult(
            enriched_regions=torch.tensor([[0, 6]]), threshold_array=votes
        ),
    )
    params = candidates.CandidatePeakCallParams(
        threshold_background_lst=[10],
        window_sizes=[2],
        threshold_lst=[1, 2, 3],
        threshold_initial=1,
        min_peak_size=1,
        max_peak_size=3,
        background_global_min=0,
    )
    result = candidates.ChromosomePeakCaller("chr3", torch.ones(6), params).call_candidates()
    assert result[["Start", "End"]].values.tolist() == [[0, 2], [4, 6]]
    assert result.Unresolved.tolist() == [False, False]


def test_empty_mcc_filter_is_safe():
    caller = object.__new__(candidates.CandidatePeakCallerMCC)
    caller.cis_chromosome = "chr1"
    filtered = candidates.CandidatePeakCallerMCC.iqr_filter(caller, pd.DataFrame(), threshold=3)
    assert filtered.empty


def test_worker_count_is_limited_by_budget_and_accelerator():
    caller = object.__new__(candidates.CandidatePeakCaller)
    caller.device = "cpu"
    caller.memory_budget_mb = 1

    class Reader:
        chroms = {"chr1": 1000, "chr2": 1000, "chr3": 1000}

    caller.reader = Reader()
    assert caller._effective_n_jobs(["chr1", "chr2", "chr3"], 8) == 3
    caller.device = "cuda"
    assert caller._effective_n_jobs(["chr1", "chr2", "chr3"], 8) == 1
    with pytest.raises(ValueError):
        caller._effective_n_jobs(["chr1", "chr2"], 0)


def test_mcc_filters_use_same_candidates_then_intersect():
    caller = object.__new__(candidates.CandidatePeakCallerMCC)
    caller.cis_chromosome = "chr1"
    caller.n_background = 0
    caller.seed = 0
    caller.apply_background_filter = True
    caller.apply_signal_filter = True
    peaks = pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr1"],
            "Start": [0, 10, 20],
            "End": [5, 15, 25],
            "Signal": [1.0, 2.0, 3.0],
        }
    )
    caller.select_chromosomes = lambda: ["chr1"]
    caller._call_chromosomes = lambda *args, **kwargs: [peaks]
    caller.background_filter = lambda frame: frame.iloc[[0, 2]].copy()
    caller.iqr_filter = lambda frame: frame.iloc[[1, 2]].copy()
    assert caller.call_candidate_peaks()[["Start", "End"]].values.tolist() == [[20, 25]]


def test_mcc_filters_recover_missing_signal_from_bigwig(monkeypatch):
    from types import SimpleNamespace

    caller = object.__new__(candidates.CandidatePeakCallerMCC)
    caller.cis_chromosome = "chr1"
    caller.bigwig = "example.bw"
    peaks = pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr2"],
            "Start": [0, 0],
            "End": [5, 5],
        }
    )
    monkeypatch.setattr(
        candidates,
        "add_signal",
        lambda intervals, bigwig: SimpleNamespace(df=intervals.assign(Signal=[10.0, 2.0])),
    )
    assert candidates.CandidatePeakCallerMCC.background_filter(caller, peaks)[
        "Signal"
    ].tolist() == [10.0]
