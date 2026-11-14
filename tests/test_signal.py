import numpy as np
import pandas as pd
import pytest
import torch

from lanceotron_mcc.api import signal


@pytest.mark.parametrize(
    ("values", "window", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 3, [1.5, 2.0, 3.0, 4.0, 4.5]),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 4, [1.5, 2.0, 2.5, 3.5, 4.0]),
        ([1.0, 2.0, 3.0], 0, [2.0, 2.0, 2.0]),
    ],
)
def test_average_array_centered_with_truncated_edges(values, window, expected):
    result = signal.average_array(torch.tensor(values), window)
    torch.testing.assert_close(result, torch.tensor(expected))


def test_average_array_validates_window_and_empty_input():
    with pytest.raises(ValueError, match="non-negative"):
        signal.average_array(torch.tensor([1.0]), -1)
    assert signal.average_array(torch.empty(0), 3).numel() == 0


def test_signal_helpers_reject_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        signal.average_array(torch.tensor([1.0, float("nan")]), 1)
    with pytest.raises(ValueError, match="finite"):
        signal.find_enriched_regions(torch.tensor([1.0, 2.0]), float("inf"))


@pytest.mark.parametrize(
    ("values", "threshold", "expected"),
    [
        ([0, 2, 3, 0, 1, 1], 1, [[1, 3], [4, 6]]),
        ([2, 2, 2], 1, [[0, 3]]),
        ([0, 0, 0], 0, []),
        ([0, -1, 2, 0], 0, [[2, 3]]),
    ],
)
def test_find_enriched_regions(values, threshold, expected):
    result = signal.find_enriched_regions(torch.tensor(values), threshold)
    assert result.dtype == torch.int64
    assert result.tolist() == expected


def test_find_enriched_regions_empty_and_size_filtering():
    assert signal.find_enriched_regions(torch.empty(0), 0).shape == (0, 2)
    result = signal.find_enriched_regions(torch.tensor([1, 1, 0, 1]), 1, min_region_size=2)
    assert result.tolist() == [[0, 2]]


def test_add_signal_reads_merged_ranges_and_uses_cached_signal(monkeypatch):
    calls = []

    class Reader:
        chroms = {"chr1": 100}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, chrom, start=0, end=None):
            calls.append((chrom, start, end))
            return np.arange(start, end, dtype=np.float64)

    monkeypatch.setattr(signal, "BigWigReader", lambda path: Reader())
    intervals = pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr1"],
            "Start": [0, 2, 8],
            "End": [4, 6, 9],
        }
    )
    result = signal.add_signal(intervals, "example.bw").df
    assert result.Signal.tolist() == [1.5, 3.5, 8.0]
    assert calls == [("chr1", 0, 6), ("chr1", 8, 9)]

    calls.clear()
    cached = intervals.assign(Signal=[7.0, 8.0, 9.0])
    assert signal.add_signal(cached, "example.bw").df.Signal.tolist() == [7.0, 8.0, 9.0]
    assert calls == []


def test_fractional_threshold_is_not_truncated_for_integer_signal():
    assert signal.find_enriched_regions(torch.tensor([1, 2, 1]), 1.5).tolist() == [[1, 2]]
