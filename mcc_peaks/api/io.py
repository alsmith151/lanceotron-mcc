"""Reusable BigWig I/O and bounded memoisation for local/HPC workloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pybigtools

# Bump when extraction/derived-cache contracts or statistical algorithms change.
CACHE_VERSION = 3


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
    }


def cache_key(*parts):
    return hashlib.sha256(
        json.dumps([CACHE_VERSION, *parts], sort_keys=True, default=str).encode()
    ).hexdigest()


class ArrayCache:
    """LRU of immutable arrays; optional atomic .npy cache on explicit local scratch.

    File identities use metadata, not a full hash that would reread remote inputs.
    Inputs must not be edited during a run.
    """

    def __init__(self, memory_budget_mb=512, directory=None):
        if not np.isfinite(memory_budget_mb) or memory_budget_mb <= 0:
            raise ValueError("memory_budget_mb must be positive")
        self.limit = int(memory_budget_mb * 1024**2)
        self.directory = Path(directory).resolve() if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.arrays = OrderedDict()
        self.metadata = {}
        self.bytes = 0
        self.metrics = {"memory_hits": 0, "disk_hits": 0, "misses": 0}
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key in self.arrays:
                self.metrics["memory_hits"] += 1
                self.arrays.move_to_end(key)
                return self.arrays[key]
            if self.directory:
                path = self.directory / f"{key}.npy"
                if path.exists():
                    try:
                        arr = np.load(path, mmap_mode="r", allow_pickle=False)
                    except (EOFError, ValueError, OSError):
                        path.unlink(missing_ok=True)
                    else:
                        self.metrics["disk_hits"] += 1
                        return arr
            self.metrics["misses"] += 1
            return None

    def discard(self, key):
        """Remove a cache entry from memory and disk after validation fails."""
        with self.lock:
            previous = self.arrays.pop(key, None)
            if previous is not None:
                self.bytes -= previous.nbytes
            if self.directory:
                (self.directory / f"{key}.npy").unlink(missing_ok=True)

    def put(self, key, array):
        arr = np.array(array, copy=True, order="C")
        arr.setflags(write=False)
        with self.lock:
            if self.directory:
                path = self.directory / f"{key}.npy"
                fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".npy")
                try:
                    with os.fdopen(fd, "wb") as handle:
                        np.save(handle, arr, allow_pickle=False)
                    os.replace(tmp, path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            previous = self.arrays.pop(key, None)
            if previous is not None:
                self.bytes -= previous.nbytes
            if arr.nbytes <= self.limit:
                while self.bytes + arr.nbytes > self.limit and self.arrays:
                    _, old = self.arrays.popitem(last=False)
                    self.bytes -= old.nbytes
                self.arrays[key] = arr
                self.bytes += arr.nbytes
        return arr

    def put_metadata(self, key, metadata):
        self.metadata[key] = dict(metadata)
        if self.directory:
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".json")
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(metadata, handle, allow_nan=False)
                os.replace(tmp, self.directory / f"{key}.json")
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    def get_metadata(self, key):
        if key in self.metadata:
            return dict(self.metadata[key])
        if self.directory:
            try:
                return json.loads((self.directory / f"{key}.json").read_text())
            except (OSError, ValueError):
                pass
        return None

    def clear(self):
        with self.lock:
            self.arrays.clear()
            self.metadata.clear()
            self.bytes = 0


class BigWigReader:
    def __init__(self, path, memory_budget_mb=512, scratch_dir=None, cache_dir=None):
        self.identity = source_identity(path)
        self.path = Path(self.identity["path"])
        self.cache = ArrayCache(memory_budget_mb, cache_dir)
        self.lock = threading.RLock()
        self.metrics = {
            "read_calls": 0,
            "stats_scans": 0,
            "input_read_seconds": 0.0,
            "staging_seconds": 0.0,
        }
        self._stats = {}
        if scratch_dir:
            start = time.perf_counter()
            scratch = Path(scratch_dir).resolve()
            scratch.mkdir(parents=True, exist_ok=True)
            staged = scratch / f"{cache_key(self.identity)}.bw"
            if staged.exists() and staged.stat().st_size != self.identity["size"]:
                staged.unlink()
            if not staged.exists():
                fd, tmp = tempfile.mkstemp(dir=scratch, suffix=".bw")
                os.close(fd)
                try:
                    shutil.copyfile(self.path, tmp)
                    os.replace(tmp, staged)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            self.path = staged
            self.metrics["staging_seconds"] = time.perf_counter() - start
        try:
            self._handle = pybigtools.open(str(self.path))
            if not self._handle.is_bigwig:
                raise ValueError("Expected a BigWig signal file, received a BigBed")
            self.chroms = dict(self._handle.chroms())
            self.info = self._handle.info()
        except Exception as exc:
            handle = getattr(self, "_handle", None)
            if handle is not None:
                handle.close()
            raise ValueError(f"Could not open BigWig {self.path}: {exc}") from exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self.lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self.cache.clear()

    def read(self, chrom, start=0, end=None):
        if chrom not in self.chroms:
            raise ValueError(f"Chromosome {chrom!r} is absent from the BigWig")
        length = self.chroms[chrom]
        end = length if end is None else int(end)
        start = int(start)
        if not 0 <= start <= end <= length:
            raise ValueError(f"Invalid interval {chrom}:{start}-{end}")
        key = cache_key(self.identity, "signal", chrom, start, end)
        with self.lock:
            if self._handle is None:
                raise ValueError("BigWigReader is closed")
            cached = self.cache.get(key)
            if cached is not None:
                if cached.shape == (end - start,):
                    return cached
                metric = "memory_hits" if key in self.cache.arrays else "disk_hits"
                self.cache.metrics[metric] -= 1
                self.cache.discard(key)
            t0 = time.perf_counter()
            try:
                values = (
                    np.asarray(self._handle.values(chrom, start, end), dtype=np.float32)
                    if end > start
                    else np.empty(0, dtype=np.float32)
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not read BigWig {self.path} interval {chrom}:{start}-{end}: {exc}"
                ) from exc
            if values.shape != (end - start,):
                raise ValueError("Unexpected BigWig extraction length")
            values = np.nan_to_num(values, nan=0.0, posinf=np.inf, neginf=-np.inf)
            if not np.all(np.isfinite(values)):
                raise ValueError("BigWig contains non-finite signal")
            self.metrics["read_calls"] += 1
            self.metrics["input_read_seconds"] += time.perf_counter() - t0
            return self.cache.put(key, values)

    def stats(self, chrom):
        """Exact full-chromosome moments, with missing bases treated as zero."""
        if chrom not in self.chroms:
            raise ValueError(f"Chromosome {chrom!r} is absent from the BigWig")
        with self.lock:
            if self._handle is None:
                raise ValueError("BigWigReader is closed")
            if chrom not in self._stats:
                t0 = time.perf_counter()
                total = square = 0.0
                try:
                    records = self._handle.records(chrom)
                    for start, end, value in records:
                        if not np.isfinite(value):
                            raise ValueError("BigWig contains non-finite signal")
                        width = end - start
                        total += width * float(value)
                        square += width * float(value) ** 2
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError(
                        f"Could not scan BigWig {self.path} chromosome {chrom}: {exc}"
                    ) from exc
                length = self.chroms[chrom]
                mean = total / length if length else 0.0
                self._stats[chrom] = {
                    "name": chrom,
                    "length": length,
                    "mean": mean,
                    "std": np.sqrt(max(0.0, square / max(length, 1) - mean**2)),
                    "sum": total,
                }
                self.metrics["stats_scans"] += 1
                self.metrics["input_read_seconds"] += time.perf_counter() - t0
            return dict(self._stats[chrom])


def peak_memory_mb():
    import resource
    import sys

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)
