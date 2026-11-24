import os
from pathlib import Path

import numpy as np
import pybigtools
import pytest

from mcc_peaks.api.io import ArrayCache, BigWigReader, cache_key, source_identity

CHROMS = {"chr1": 8}
RECORDS = [("chr1", 0, 2, 2.0), ("chr1", 4, 7, 1.0)]


def write_bigwig(path: Path, records=RECORDS):
    writer = pybigtools.open(str(path), "w")
    writer.write(CHROMS, records)


def test_reader_fills_uncovered_bases_with_zero_and_caches_reads(tmp_path):
    path = tmp_path / "signal.bw"
    write_bigwig(path)

    with BigWigReader(path) as reader:
        assert reader.chroms == CHROMS
        np.testing.assert_array_equal(reader.read("chr1", 0, 8), [2, 2, 0, 0, 1, 1, 1, 0])
        assert reader.metrics["read_calls"] == 1
        np.testing.assert_array_equal(reader.read("chr1", 0, 8), [2, 2, 0, 0, 1, 1, 1, 0])
        assert reader.metrics["read_calls"] == 1
        assert reader.cache.metrics["memory_hits"] == 1


def test_reader_stats_are_exact_and_include_uncovered_bases(tmp_path):
    path = tmp_path / "signal.bw"
    write_bigwig(path)

    with BigWigReader(path) as reader:
        stats = reader.stats("chr1")
        assert stats["name"] == "chr1"
        assert stats["length"] == 8
        assert stats["sum"] == pytest.approx(7.0)
        assert stats["mean"] == pytest.approx(7.0 / 8.0)
        assert stats["std"] == pytest.approx(np.sqrt(11.0 / 8.0 - (7.0 / 8.0) ** 2))
        assert reader.metrics["stats_scans"] == 1
        assert reader.stats("chr1") == stats
        assert reader.metrics["stats_scans"] == 1


def test_reader_closes_handle_when_context_exits(tmp_path):
    path = tmp_path / "signal.bw"
    write_bigwig(path)

    reader = BigWigReader(path)
    handle = reader._handle
    assert handle is not None
    with reader:
        assert reader._handle is handle
    assert reader._handle is None
    with pytest.raises(ValueError, match="closed"):
        reader.stats("chr1")


def test_file_metadata_change_invalidates_reader_cache_identity(tmp_path):
    path = tmp_path / "signal.bw"
    write_bigwig(path)
    before = source_identity(path)
    cache_dir = tmp_path / "cache"

    with BigWigReader(path, cache_dir=cache_dir) as reader:
        np.testing.assert_array_equal(reader.read("chr1", 0, 8), [2, 2, 0, 0, 1, 1, 1, 0])

    replacement = tmp_path / "replacement.bw"
    write_bigwig(replacement, [("chr1", 0, 2, 4.0), ("chr1", 4, 7, 1.0)])
    os.replace(replacement, path)
    after = source_identity(path)
    assert before != after

    with BigWigReader(path, cache_dir=cache_dir) as reader:
        np.testing.assert_array_equal(reader.read("chr1", 0, 8), [4, 4, 0, 0, 1, 1, 1, 0])
        assert reader.metrics["read_calls"] == 1
        assert reader.cache.metrics["disk_hits"] == 0


def test_reader_staging_reuses_existing_copy(tmp_path):
    path = tmp_path / "signal.bw"
    scratch = tmp_path / "scratch"
    write_bigwig(path)

    with BigWigReader(path, scratch_dir=scratch) as first:
        staged = first.path
        assert staged.parent == scratch
        assert staged.read_bytes() == path.read_bytes()
        staged_mtime = staged.stat().st_mtime_ns
        assert first.metrics["staging_seconds"] >= 0

    with BigWigReader(path, scratch_dir=scratch) as second:
        assert second.path == staged
        assert second.path.stat().st_mtime_ns == staged_mtime


def test_reader_reuses_disk_cached_ranges_across_instances(tmp_path):
    path = tmp_path / "signal.bw"
    cache_dir = tmp_path / "cache"
    write_bigwig(path)

    with BigWigReader(path, cache_dir=cache_dir) as first:
        expected = first.read("chr1", 0, 8)
        assert first.metrics["read_calls"] == 1

    with BigWigReader(path, cache_dir=cache_dir) as second:
        actual = second.read("chr1", 0, 8)
        np.testing.assert_array_equal(actual, expected)
        assert second.metrics["read_calls"] == 0
        assert second.cache.metrics["disk_hits"] == 1


def test_reader_discards_wrong_shape_disk_cache_and_replaces_it(tmp_path):
    path = tmp_path / "signal.bw"
    cache_dir = tmp_path / "cache"
    write_bigwig(path)
    identity = source_identity(path)
    key = cache_key(identity, "signal", "chr1", 0, 8)
    cache_dir.mkdir()
    np.save(cache_dir / f"{key}.npy", np.array([99], dtype=np.float32))

    with BigWigReader(path, cache_dir=cache_dir) as reader:
        actual = reader.read("chr1", 0, 8)
        np.testing.assert_array_equal(actual, [2, 2, 0, 0, 1, 1, 1, 0])
        assert reader.metrics["read_calls"] == 1

    with BigWigReader(path, cache_dir=cache_dir) as reader:
        np.testing.assert_array_equal(reader.read("chr1", 0, 8), actual)
        assert reader.metrics["read_calls"] == 0
        assert reader.cache.metrics["disk_hits"] == 1


def test_reader_replaces_incomplete_staged_copy(tmp_path):
    path = tmp_path / "signal.bw"
    scratch = tmp_path / "scratch"
    write_bigwig(path)
    identity = source_identity(path)
    scratch.mkdir()
    staged = scratch / f"{cache_key(identity)}.bw"
    staged.write_bytes(b"incomplete")

    with BigWigReader(path, scratch_dir=scratch) as reader:
        assert reader.path.read_bytes() == path.read_bytes()
        np.testing.assert_array_equal(reader.read("chr1", 0, 8), [2, 2, 0, 0, 1, 1, 1, 0])


def test_backend_open_errors_include_bigwig_path(tmp_path, monkeypatch):
    import mcc_peaks.api.io as io

    path = tmp_path / "signal.bw"
    write_bigwig(path)

    def fail_open(_path):
        raise RuntimeError("invalid BBI")

    monkeypatch.setattr(io.pybigtools, "open", fail_open)
    with pytest.raises(ValueError, match=r"Could not open BigWig .*signal.bw: invalid BBI"):
        BigWigReader(path)


def test_array_cache_obeys_bounded_lru_eviction():
    cache = ArrayCache(memory_budget_mb=1)
    first = np.zeros(100_000, dtype=np.float32)
    second = np.ones(100_000, dtype=np.float32)
    third = np.full(100_000, 2, dtype=np.float32)

    cache.put("first", first)
    cache.put("second", second)
    assert cache.bytes == 800_000
    cache.get("first")  # Make first most recently used.
    cache.put("third", third)

    assert list(cache.arrays) == ["first", "third"]
    assert cache.bytes == 800_000
    assert cache.bytes <= cache.limit
    assert cache.get("second") is None


def test_array_cache_stores_an_immutable_copy():
    cache = ArrayCache()
    original = np.arange(4, dtype=np.float32)
    cached = cache.put("values", original)

    assert original.flags.writeable
    original[0] = 99
    assert cached[0] == 0
    assert not cached.flags.writeable


def test_reader_rejects_bigbed_instead_of_treating_intervals_as_signal(tmp_path):
    path = tmp_path / "intervals.bb"
    writer = pybigtools.open(str(path), "w")
    writer.write({"chr1": 100}, [("chr1", 10, 20, "feature")])
    with pytest.raises(ValueError, match="Expected a BigWig"):
        BigWigReader(path)
