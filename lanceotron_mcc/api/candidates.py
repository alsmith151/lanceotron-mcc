"""Candidate peak calling and MCC filtering.

BigWig access is delegated to :class:`BigWigReader`; the caller keeps one
reader open for its lifetime so chromosome metadata and decoded blocks can be
reused across the different steps.
"""

import math
import pathlib
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Literal

import numpy as np
import pandas as pd
import pydantic
import torch
from loguru import logger

from lanceotron_mcc.api.io import BigWigReader
from lanceotron_mcc.api.signal import add_signal, average_array, find_enriched_regions
from lanceotron_mcc.api.utils import Chromosomes, ChromosomeStats, get_device


class EnrichedRegionsParams(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    background_list: list[int] = [int(1e4), int(1e5), int(1e6)]
    window_list: list[int] = [200, 400, 600]
    threshold_list: list[float] = [2.0, 4.0, 6.0]
    threshold_cumulative: float = 9.0
    min_region_size: int = 0
    max_region_size: int | None = None
    background_global_min: float | None = None
    return_threshold_array: bool = False

    @pydantic.model_validator(mode="after")
    def _valid_grid(self):
        for name in ("background_list", "window_list"):
            vals = getattr(self, name)
            if not vals or any(not isinstance(x, int) or x <= 0 for x in vals):
                raise ValueError(f"{name} must contain positive integers")
        if not self.threshold_list or any(
            not math.isfinite(x) or x <= 0 for x in self.threshold_list
        ):
            raise ValueError("threshold_list must contain finite positive values")
        if self.min_region_size < 0 or (
            self.max_region_size is not None and self.max_region_size < self.min_region_size
        ):
            raise ValueError("region size limits must be nonnegative and min <= max")
        if not math.isfinite(self.threshold_cumulative) or self.threshold_cumulative <= 0:
            raise ValueError("threshold_cumulative must be finite and positive")
        if self.background_global_min is not None and (
            not math.isfinite(self.background_global_min) or self.background_global_min < 0
        ):
            raise ValueError("background_global_min must be finite and non-negative")
        return self


class EnrichedRegionsResult(pydantic.BaseModel):
    enriched_regions: torch.Tensor
    threshold_array: torch.Tensor | None = None
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def to_dataframe(self):
        return pd.DataFrame(self.enriched_regions.detach().cpu().numpy(), columns=["Start", "End"])


def _as_numpy(array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(array, dtype=np.float32)


def _runs(mask: np.ndarray):
    """Yield half-open ranges for contiguous true values."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return zip(starts.tolist(), ends.tolist())


def search_enriched_regions(
    array: torch.Tensor,
    background: int,
    window: int,
    threshold: float,
    background_global_min: float | None = None,
    min_region_size: int = 0,
    max_region_size: int | None = None,
) -> torch.Tensor:
    """Find enriched runs for one parameter combination and apply size limits."""
    signal = average_array(array, window)
    background_array = average_array(array, background)
    threshold_array = background_array * threshold
    if background_global_min is not None:
        threshold_array = torch.maximum(
            threshold_array,
            torch.as_tensor(
                background_global_min, device=threshold_array.device, dtype=threshold_array.dtype
            ),
        )
    return find_enriched_regions(signal, threshold_array, min_region_size, max_region_size)


def search_enriched_regions_param_grid(
    array: torch.Tensor, params: EnrichedRegionsParams
) -> EnrichedRegionsResult:
    """Accumulate votes, reusing each window/background moving average."""
    device = array.device if isinstance(array, torch.Tensor) else torch.device("cpu")
    coverage = (
        array
        if isinstance(array, torch.Tensor)
        else torch.as_tensor(array, dtype=torch.float32, device=device)
    )
    threshold_array = torch.zeros(coverage.shape, dtype=torch.int32, device=device)
    windowed = {w: average_array(coverage, w) for w in dict.fromkeys(params.window_list)}
    floor = params.background_global_min
    for background in dict.fromkeys(params.background_list):
        # Keep only one background smoother resident at a time. Windowed
        # arrays are shared across backgrounds, while threshold votes remain
        # a compact int32 array.
        background_array = average_array(coverage, background)
        for window in dict.fromkeys(params.window_list):
            multiplicity = params.background_list.count(background) * params.window_list.count(
                window
            )
            for multiplier in params.threshold_list:
                cutoff = background_array * multiplier
                if floor is not None:
                    cutoff = torch.maximum(
                        cutoff, torch.as_tensor(floor, device=device, dtype=cutoff.dtype)
                    )
                mask = (windowed[window] > 0) & (windowed[window] >= cutoff)
                # Per-combination minimum is retained; no maximum is applied here.
                if params.min_region_size > 1:
                    mask_np = mask.detach().cpu().numpy()
                    keep = np.zeros(mask_np.shape, dtype=bool)
                    for start, end in _runs(mask_np):
                        size = end - start
                        if size >= params.min_region_size and (
                            params.max_region_size is None or size <= params.max_region_size
                        ):
                            keep[start:end] = True
                    mask = torch.as_tensor(keep, device=device)
                elif params.max_region_size is not None:
                    mask_np = mask.detach().cpu().numpy()
                    keep = np.zeros(mask_np.shape, dtype=bool)
                    for start, end in _runs(mask_np):
                        if end - start <= params.max_region_size:
                            keep[start:end] = True
                    mask = torch.as_tensor(keep, device=device)
                threshold_array.add_(mask.to(torch.int32) * multiplicity)
    final_mask = threshold_array >= params.threshold_cumulative
    regions = np.asarray(list(_runs(final_mask.detach().cpu().numpy())), dtype=np.int64).reshape(
        (-1, 2)
    )
    sizes = regions[:, 1] - regions[:, 0]
    if params.max_region_size is not None:
        regions = regions[sizes <= params.max_region_size]
        sizes = regions[:, 1] - regions[:, 0]
    if params.min_region_size:
        regions = regions[sizes >= params.min_region_size]
    enriched = torch.as_tensor(regions, dtype=torch.int64, device=device)
    if params.return_threshold_array:
        return EnrichedRegionsResult(enriched_regions=enriched, threshold_array=threshold_array)
    return EnrichedRegionsResult(enriched_regions=enriched)


class CandidatePeakCallParams(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    threshold_initial: float = 9.0
    threshold_lst: list[float] = [2.0, 4.0, 6.0]
    threshold_background_lst: list[int] = [int(1e4), int(1e5), int(1e6)]
    window_sizes: list[int] = [200, 400, 600]
    min_peak_size: int = 50
    max_peak_size: int = 500
    background_global_min: float | Literal["mean"] | None = "mean"

    @pydantic.model_validator(mode="after")
    def _validate_call_params(self):
        for name in ("threshold_background_lst", "window_sizes"):
            values = getattr(self, name)
            if not values or any(not isinstance(x, int) or x <= 0 for x in values):
                raise ValueError(f"{name} must contain positive integers")
        if not self.threshold_lst or any(
            not math.isfinite(x) or x <= 0 for x in self.threshold_lst
        ):
            raise ValueError("threshold_lst must contain finite positive values")
        grid_size = (
            len(self.threshold_background_lst) * len(self.window_sizes) * len(self.threshold_lst)
        )
        if (
            not math.isfinite(self.threshold_initial)
            or self.threshold_initial <= 0
            or self.threshold_initial > grid_size
        ):
            raise ValueError(
                "threshold_initial must be positive and no greater than the parameter grid size"
            )
        if self.min_peak_size <= 0 or self.max_peak_size < self.min_peak_size:
            raise ValueError("peak sizes must be positive and min_peak_size <= max_peak_size")
        if isinstance(self.background_global_min, (int, float)) and not isinstance(
            self.background_global_min, bool
        ):
            if (
                not math.isfinite(float(self.background_global_min))
                or self.background_global_min < 0
            ):
                raise ValueError("background_global_min must be finite and nonnegative")
        return self


class ChromosomePeakCaller:
    def __init__(
        self, chromosome: str, coverage_array: torch.Tensor, params: CandidatePeakCallParams
    ):
        self.chromosome = chromosome
        self.coverage_array = coverage_array
        self.params = params

    def call_candidates(self):
        coverage = _as_numpy(self.coverage_array)
        if coverage.ndim != 1:
            raise ValueError("coverage_array must be one-dimensional")
        if not np.isfinite(coverage).all():
            raise ValueError("coverage_array must contain only finite values")
        if coverage.size == 0:
            return pd.DataFrame(
                columns=[
                    "Chromosome",
                    "Start",
                    "End",
                    "Length",
                    "Unresolved",
                    "Signal",
                    "Sum",
                    "Maximum",
                    "Summit",
                    "Support Fraction",
                ]
            )
        global_floor = (
            float(coverage.mean())
            if self.params.background_global_min == "mean"
            else self.params.background_global_min
        )
        grid_count = (
            len(self.params.threshold_background_lst)
            * len(self.params.window_sizes)
            * len(self.params.threshold_lst)
        )
        votes_result = search_enriched_regions_param_grid(
            self.coverage_array,
            EnrichedRegionsParams(
                background_list=self.params.threshold_background_lst,
                window_list=self.params.window_sizes,
                threshold_list=self.params.threshold_lst,
                threshold_cumulative=self.params.threshold_initial,
                min_region_size=self.params.min_peak_size,
                background_global_min=global_floor,
                return_threshold_array=True,
            ),
        )
        votes = votes_result.threshold_array.detach().cpu().numpy()
        first_refinement_level = self.params.threshold_initial + 1
        records = []
        pending = [
            (start, end, first_refinement_level)
            for start, end in _runs(votes >= self.params.threshold_initial)
            if end - start >= self.params.min_peak_size
        ]
        # Process each finer threshold at most once per interval. The queue
        # avoids Python recursion limits for unusually large parameter grids.
        while pending:
            start, end, level = pending.pop()
            if end - start <= self.params.max_peak_size:
                records.append((start, end, False))
                continue
            if level > grid_count:
                records.append((start, end, True))
                continue
            split = [
                (a + start, b + start)
                for a, b in _runs(votes[start:end] >= level)
                if b - a >= self.params.min_peak_size
            ]
            if not split or (len(split) == 1 and split[0] == (start, end)):
                pending.append((start, end, level + 1))
            else:
                pending.extend((a, b, level + 1) for a, b in split)
        records.sort(key=lambda record: record[0])
        rows = []
        for start, end, unresolved in records:
            segment = coverage[start:end]
            if not segment.size:
                continue
            local_max = int(np.argmax(segment))
            rows.append(
                {
                    "Chromosome": self.chromosome,
                    "Start": int(start),
                    "End": int(end),
                    "Length": int(end - start),
                    "Unresolved": bool(unresolved),
                    "Signal": float(segment.mean()),
                    "Sum": float(segment.sum(dtype=np.float64)),
                    "Maximum": float(segment[local_max]),
                    "Summit": int(start + local_max),
                    "Support Fraction": float(votes[start:end].mean() / grid_count),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "Chromosome",
                "Start",
                "End",
                "Length",
                "Unresolved",
                "Signal",
                "Sum",
                "Maximum",
                "Summit",
                "Support Fraction",
            ],
        )


class CandidatePeakCaller:
    def __init__(
        self,
        bigwig: pathlib.Path | str,
        params: CandidatePeakCallParams = None,
        only_standard_chromosomes: bool = True,
        device: str = None,
        memory_budget_mb: float = 512,
        scratch_dir=None,
        cache_dir=None,
        seed: int = 0,
    ):
        self.bigwig = str(bigwig)
        self.params = params or CandidatePeakCallParams()
        self.only_standard_chromosomes = only_standard_chromosomes
        self.device = device or get_device()
        self.seed = seed
        self.memory_budget_mb = float(memory_budget_mb)
        if not math.isfinite(self.memory_budget_mb) or self.memory_budget_mb <= 0:
            raise ValueError("memory_budget_mb must be finite and positive")
        self.reader = BigWigReader(
            self.bigwig,
            memory_budget_mb=memory_budget_mb,
            scratch_dir=scratch_dir,
            cache_dir=cache_dir,
        )
        self._stats_cache: dict[str, ChromosomeStats] = {}
        logger.info(f"Using device: {self.device}")

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def total_coverage(self):
        info = self.reader.info or {}
        # pybigtools exposes its file-wide sum under ``summary.sum`` on some
        # versions; its older ``sumData`` field is supported as well.
        summary = info.get("summary", {}) if isinstance(info, dict) else {}
        for value in (info.get("sumData"), info.get("sum"), summary.get("sum")):
            if value is not None and np.isfinite(value):
                return float(value)
        return float(sum(self.reader.stats(chrom)["sum"] for chrom in self.reader.chroms))

    def calculate_chromosome_stats(self, chromosome: str) -> ChromosomeStats:
        if chromosome not in self._stats_cache:
            row = self.reader.stats(chromosome)
            self._stats_cache[chromosome] = ChromosomeStats(
                name=chromosome,
                length=int(row["length"]),
                mean=float(row["mean"]),
                std=float(row["std"]),
            )
        return self._stats_cache[chromosome]

    @property
    def genome_stats(self) -> dict[str, ChromosomeStats]:
        chromosomes = list(self.reader.chroms)
        if self.only_standard_chromosomes:
            chromosomes = Chromosomes(chromosomes).standard_chromosomes.chromosomes
        return {chrom: self.calculate_chromosome_stats(chrom) for chrom in chromosomes}

    def extract_chromosome_signal(
        self, chrom_stats: ChromosomeStats, smoothing: int = None, tensor: bool = False
    ):
        signal = self.reader.read(chrom_stats.name, 0, chrom_stats.length)
        if smoothing is not None:
            signal_tensor = torch.from_numpy(np.array(signal, dtype=np.float32, copy=True))
            signal = _as_numpy(average_array(signal_tensor, smoothing))
        if tensor:
            return torch.tensor(signal, dtype=torch.float32, device=self.device)
        return np.asarray(signal, dtype=np.float32)

    def _chromosomes(self):
        chromosomes = list(self.reader.chroms)
        if self.only_standard_chromosomes:
            chromosomes = list(Chromosomes(chromosomes).standard_chromosomes.chromosomes)
        return sorted(chromosomes)

    def _process_chromosome(self, chrom):
        # Peak calling only needs chromosome length; a full moment scan would
        # reread all records before the actual signal extraction.
        stats = ChromosomeStats(
            name=chrom, length=int(self.reader.chroms[chrom]), mean=0.0, std=0.0
        )
        coverage = self.extract_chromosome_signal(stats, tensor=True)
        return ChromosomePeakCaller(chrom, coverage, self.params).call_candidates()

    def _effective_n_jobs(self, chromosomes, n_jobs):
        if n_jobs is None:
            n_jobs = 1
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        if len(chromosomes) < 2 or n_jobs == 1:
            return 1
        # CUDA/MPS tensors and their temporary moving-average buffers compete
        # for accelerator memory, so keep those runs single-worker. On CPU,
        # reserve a conservative 64 bytes per base for the input, smoothing
        # buffers, masks, and refinement votes.
        if str(self.device).startswith(("cuda", "mps")):
            if n_jobs > 1:
                logger.info("Using one peak-calling worker for accelerator memory")
            return 1
        longest = max(int(self.reader.chroms[chrom]) for chrom in chromosomes)
        estimated_worker_bytes = max(1, longest) * 64
        budget_bytes = int(self.memory_budget_mb * 1024**2)
        memory_workers = max(1, budget_bytes // estimated_worker_bytes)
        workers = min(n_jobs, len(chromosomes), memory_workers)
        if workers < n_jobs:
            logger.info(
                "Capping peak-calling workers from {} to {} using the {} MiB memory budget",
                n_jobs,
                workers,
                self.memory_budget_mb,
            )
        if memory_workers == 1 and estimated_worker_bytes > budget_bytes:
            logger.warning(
                "A chromosome needs an estimated {:.1f} MiB for candidate calling, above the {:.1f} MiB budget; using one worker",
                estimated_worker_bytes / 1024**2,
                self.memory_budget_mb,
            )
        return workers

    def _call_chromosomes(self, chromosomes, n_jobs=1):
        """Read serially, then compute with a memory-bounded worker pool."""
        chromosomes = list(chromosomes)
        workers = self._effective_n_jobs(chromosomes, n_jobs)
        if workers == 1:
            return [self._process_chromosome(chrom) for chrom in chromosomes]

        def call(chrom, coverage):
            return ChromosomePeakCaller(chrom, coverage, self.params).call_candidates()

        frames = []
        pending = set()
        # Only the dispatching thread touches BigWigReader. Keep at most one
        # chromosome array per worker alive while compute threads are active.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="peak-call") as pool:
            for chrom in chromosomes:
                stats = ChromosomeStats(
                    name=chrom, length=int(self.reader.chroms[chrom]), mean=0.0, std=0.0
                )
                coverage = self.extract_chromosome_signal(stats, tensor=True)
                pending.add(pool.submit(call, chrom, coverage))
                if len(pending) >= workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    frames.extend(future.result() for future in done)
            for future in pending:
                frames.append(future.result())
        return frames

    def call_candidate_peaks(self, n_jobs: int = 1):
        chromosomes = self._chromosomes()
        frames = self._call_chromosomes(chromosomes, n_jobs=n_jobs)
        peaks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not peaks.empty:
            peaks = peaks.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
        return peaks


class CandidatePeakCallerMCC(CandidatePeakCaller):
    def __init__(
        self,
        bigwig: pathlib.Path | str,
        params: CandidatePeakCallParams = None,
        cis_chromosome: str | Literal["estimate"] | None = None,
        n_background: int = 1,
        apply_background_filter: bool = True,
        apply_signal_filter: bool = True,
        device: str = None,
        memory_budget_mb: float = 512,
        scratch_dir=None,
        cache_dir=None,
        seed: int = 0,
    ):
        super().__init__(
            bigwig,
            params,
            device=device,
            memory_budget_mb=memory_budget_mb,
            scratch_dir=scratch_dir,
            cache_dir=cache_dir,
            seed=seed,
        )
        try:
            self.cis_chromosome = (
                self.estimate_cis_chromosome()
                if cis_chromosome in (None, "estimate")
                else cis_chromosome
            )
        except Exception:
            self.close()
            raise
        self.n_background = n_background
        if isinstance(n_background, bool) or not isinstance(n_background, int) or n_background < 0:
            self.close()
            raise ValueError("n_background must be a non-negative integer")
        self.apply_background_filter = apply_background_filter
        self.apply_signal_filter = apply_signal_filter

    def estimate_cis_chromosome(self):
        stats = self.genome_stats
        if not stats:
            raise ValueError("No chromosomes are available to estimate the cis chromosome")
        total = sum(s.length * s.mean for s in stats.values())
        if total <= 0:
            warnings.warn(
                "No positive coverage found; using the first standard chromosome", RuntimeWarning
            )
            return sorted(stats)[0]
        return max(sorted(stats), key=lambda chrom: stats[chrom].length * stats[chrom].mean)

    def set_cis_chromosome(self, chromosome: str | None = None):
        self.cis_chromosome = self.estimate_cis_chromosome() if chromosome is None else chromosome

    def select_chromosomes(self):
        chromosomes = self._chromosomes()
        if self.cis_chromosome not in chromosomes:
            raise ValueError(
                f"cis chromosome {self.cis_chromosome!r} is absent from the selected chromosomes"
            )
        others = [chrom for chrom in chromosomes if chrom != self.cis_chromosome]
        n = min(max(0, self.n_background), len(others))
        rng = np.random.default_rng(self.seed)
        selected = [self.cis_chromosome]
        if n:
            selected.extend(rng.choice(others, n, replace=False).tolist())
        if self.n_background and not n:
            warnings.warn("No chromosomes are available for background filtering", RuntimeWarning)
        return selected

    def _with_background_label(self, peaks):
        if peaks.empty:
            return peaks.assign(is_background=pd.Series(dtype=bool))
        return peaks.assign(is_background=peaks.Chromosome != self.cis_chromosome)

    def _with_signal(self, peaks):
        if "Signal" in peaks.columns or peaks.empty:
            return peaks
        if not hasattr(self, "bigwig"):
            raise ValueError("bigwig is required to calculate missing peak Signal values")
        return add_signal(peaks, self.bigwig).df

    def background_filter(self, peaks: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a quantile between zero and one")
        peaks = self._with_signal(self._with_background_label(peaks))
        background = peaks.loc[peaks.is_background, "Signal"].dropna()
        if background.empty:
            warnings.warn(
                "Background filtering skipped because no background peaks were called",
                RuntimeWarning,
            )
            return peaks.loc[~peaks.is_background].copy().reset_index(drop=True)
        cutoff = background.quantile(threshold)
        return (
            peaks.loc[(~peaks.is_background) & (peaks.Signal >= cutoff)]
            .copy()
            .reset_index(drop=True)
        )

    def z_score_filter(self, peaks: pd.DataFrame, threshold: float = 3.0):
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        peaks = self._with_signal(self._with_background_label(peaks))
        background = peaks.loc[peaks.is_background, "Signal"].dropna()
        if background.empty:
            warnings.warn(
                "Z-score filtering skipped because no background peaks were called", RuntimeWarning
            )
            return peaks.loc[~peaks.is_background].copy().reset_index(drop=True)
        std = background.std()
        if not np.isfinite(std) or std == 0:
            z_score = pd.Series(
                np.where(peaks.Signal >= background.mean(), np.inf, -np.inf), index=peaks.index
            )
        else:
            z_score = (peaks.Signal - background.mean()) / std
        peaks = peaks.assign(z_score=z_score)
        return (
            peaks.loc[(~peaks.is_background) & (peaks.z_score >= threshold)]
            .copy()
            .reset_index(drop=True)
        )

    def iqr_filter(self, peaks: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        peaks = self._with_signal(self._with_background_label(peaks))
        if peaks.empty:
            return peaks.copy()
        values = np.log1p(np.maximum(peaks.Signal.to_numpy(dtype=float), 0))
        q1, q3 = np.percentile(values, [25, 75])
        cutoff = q3 + threshold * (q3 - q1)
        return peaks.loc[(~peaks.is_background) & (values > cutoff)].copy().reset_index(drop=True)

    def call_candidate_peaks(self, n_jobs: int = 1):
        frames = self._call_chromosomes(self.select_chromosomes(), n_jobs=n_jobs)
        peaks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not peaks.empty:
            peaks = peaks.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
        if self.apply_background_filter and self.apply_signal_filter:
            # Each filter estimates its cutoff from the same candidate set;
            # the returned cis peaks must pass both independent criteria.
            background_pass = self.background_filter(peaks)
            signal_pass = self.iqr_filter(peaks)
            keys = ["Chromosome", "Start", "End"]
            passed = background_pass[keys].merge(signal_pass[keys].drop_duplicates(), on=keys)
            return peaks.merge(passed.drop_duplicates(), on=keys, how="inner").reset_index(
                drop=True
            )
        if self.apply_background_filter:
            return self.background_filter(peaks)
        if self.apply_signal_filter:
            return self.iqr_filter(peaks)
        return peaks
