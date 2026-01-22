import concurrent.futures
import itertools
import multiprocessing
import pathlib
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pybigtools
import pydantic
import pyranges as pr
import torch
import torch.multiprocessing as mp
import tqdm
from joblib import Parallel, delayed
from loguru import logger

from lanceotron_mcc.api.signal import average_array, find_enriched_regions
from lanceotron_mcc.api.utils import Chromosomes, ChromosomeStats, get_device

# Set the start method for multiprocessing to 'spawn' instead of 'fork'
# This prevents the fork-related deadlock warning
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


class EnrichedRegionsParams(pydantic.BaseModel):
    background_list: List[int] = [int(1e4), int(1e5), int(1e6)]
    window_list: List[int] = [200, 400, 600]
    threshold_list: List[float] = [2.0, 4.0, 6.0]
    threshold_cumulative: float = 9.0
    min_region_size: int = 0
    max_region_size: Optional[int] = None
    background_global_min: Optional[float] = None
    return_threshold_array: bool = False


class EnrichedRegionsResult(pydantic.BaseModel):
    enriched_regions: torch.Tensor
    threshold_array: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

    def to_dataframe(self):
        df = pd.DataFrame(self.enriched_regions.cpu(), columns=["Start", "End"])
        return df


def search_enriched_regions(
    array: torch.Tensor,
    background: int,
    window: int,
    threshold: float,
    background_global_min: Optional[float] = None,
    min_region_size: int = 0,
    max_region_size: Optional[int] = None,
) -> torch.Tensor:
    # Compute the moving average of the array
    array_windowed = average_array(array, window)

    # Compute the background array
    background_array = average_array(array, background)

    # Compute the threshold array
    threshold_array = background_array * threshold

    # Apply global minimum threshold if specified
    if background_global_min is not None:
        threshold_array = torch.maximum(
            threshold_array, torch.tensor(background_global_min)
        )

    # Identify enriched regions
    enriched_regions = find_enriched_regions(
        array_windowed, threshold_array, min_region_size, max_region_size
    )
    return enriched_regions


def search_enriched_regions_param_grid(
    array: torch.Tensor, params: EnrichedRegionsParams, label: Optional[str] = None
) -> EnrichedRegionsResult:
    """
    Find enriched regions in parallel using GPU acceleration.
    """
    # Move array to device if it's not already there
    device = array.device

    # Initialize threshold array on the same device
    threshold_array = torch.zeros(array.shape, dtype=torch.int8, device=device)

    # Generate all parameter combinations
    combinations = list(
        itertools.product(
            params.background_list, params.window_list, params.threshold_list
        )
    )

    # Process combinations in batches for better GPU utilization
    batch_size = min(16, len(combinations))  # Adjust batch size as needed

    for i in range(0, len(combinations), batch_size):
        batch = combinations[i : i + batch_size]
        batch_regions = []

        for background, window, threshold in batch:
            regions = search_enriched_regions(
                array=array,
                background=background,
                window=window,
                threshold=threshold,
                background_global_min=params.background_global_min,
                min_region_size=params.min_region_size,
                max_region_size=params.max_region_size,
            )
            batch_regions.append(regions)

        # Vectorized update of threshold_array using a difference array
        # This is much faster than sliced updates in a loop
        diff = torch.zeros(array.shape[0] + 1, dtype=torch.int32, device=device)

        for regions in batch_regions:
            if regions.shape[0] > 0:
                starts = regions[:, 0].to(torch.long)
                ends = regions[:, 1].to(torch.long)

                # Update difference array
                # Use scatter_add_ for batch updates if possible,
                # but direct indexing with multiple indices works in PyTorch
                diff.index_add_(0, starts, torch.ones_like(starts, dtype=torch.int32))
                diff.index_add_(0, ends, -torch.ones_like(ends, dtype=torch.int32))

        # Compute cumulative sum to get the threshold counts at each position
        threshold_array += torch.cumsum(diff, dim=0)[:-1].to(torch.int8)

    # Identify final enriched regions based on the cumulative threshold array
    enriched_regions_final = find_enriched_regions(
        threshold_array,
        params.threshold_cumulative,
        params.min_region_size,
        params.max_region_size,
    )

    if params.return_threshold_array:
        return EnrichedRegionsResult(
            enriched_regions=enriched_regions_final, threshold_array=threshold_array
        )
    else:
        return EnrichedRegionsResult(enriched_regions=enriched_regions_final)


class CandidatePeakCallParams(pydantic.BaseModel):
    threshold_initial: float = 9.0
    threshold_lst: List[float] = [2.0, 4.0, 6.0]
    threshold_background_lst: List[int] = [int(1e4), int(1e5), int(1e6)]
    window_sizes: List[int] = [200, 400, 600]
    min_peak_size: int = 50
    max_peak_size: int = 500
    min_read_count: int = 5
    coverage_depth_per_single_read: Union[float, Literal["estimate"]] = "estimate"
    background_global_min: Optional[Literal["mean"]] = "mean"


class ChromosomePeakCaller:
    def __init__(
        self,
        chromosome: str,
        coverage_array: torch.Tensor,
        params: CandidatePeakCallParams,
    ):
        self.chromosome = chromosome
        self.coverage_array = coverage_array
        self.params = params

    def call_candidates(self):
        # Set threshold values
        threshold_limit = (
            len(self.params.threshold_background_lst)
            * len(self.params.window_sizes)
            * len(self.params.threshold_lst)
        )
        threshold_cumulative = self.params.threshold_initial

        # Set background global minimum threshold if specified
        if self.params.background_global_min == "mean":
            background_global_min = self.coverage_array.mean()
        else:
            background_global_min = self.params.background_global_min

        enrichment_results = search_enriched_regions_param_grid(
            self.coverage_array,
            EnrichedRegionsParams(
                background_list=self.params.threshold_background_lst,
                window_list=self.params.window_sizes,
                threshold_list=self.params.threshold_lst,
                threshold_cumulative=threshold_cumulative,
                min_region_size=self.params.min_peak_size,
                max_region_size=self.params.max_peak_size,
                background_global_min=background_global_min,
                return_threshold_array=True,
            ),
            label=self.chromosome,
        )

        # Create a DataFrame of enriched regions
        df_enriched_regions = pd.DataFrame(
            enrichment_results.to_dataframe(), columns=["Start", "End"]
        )
        df_enriched_regions["Chromosome"] = self.chromosome
        df_enriched_regions["Length"] = (
            df_enriched_regions["End"] - df_enriched_regions["Start"]
        )

        # Filter out regions with length less than min_peak_size
        df_enriched_regions = df_enriched_regions.query(
            f"Length >= {self.params.min_peak_size}"
        )

        df_unsolved = df_enriched_regions.query("Length > @self.params.max_peak_size")
        df_solved = df_enriched_regions.query("Length <= @self.params.max_peak_size")

        # Iterative refinement of enriched regions based on a higher threshold and size constraints.
        # The process continues until the threshold limit is reached or all regions are solved.
        # The goal of this loop is to iteratively refine the identified unsolved enriched regions
        # by increasing the cumulative threshold and resolving regions that are too large into smaller, more precise regions.

        while not df_unsolved.empty and threshold_cumulative < threshold_limit - 1:
            # Sort unsolved regions by Start
            df_unsolved = df_unsolved.sort_values(["Chromosome", "Start"]).reset_index(
                drop=True
            )
            df_unsolved["unsolved_idx"] = df_unsolved.index

            # Increment the cumulative threshold
            threshold_cumulative += 1

            # Recompute enriched regions with the higher threshold
            enriched_regions_higher_thresh = find_enriched_regions(
                enrichment_results.threshold_array, threshold_cumulative
            )
            df_enriched_higher_thresh = pd.DataFrame(
                enriched_regions_higher_thresh, columns=["Start", "End"]
            )
            df_enriched_higher_thresh.insert(0, "Chromosome", self.chromosome)

            # Use pyranges to find overlaps between unsolved regions and enriched regions at a higher threshold
            pr_unsolved = pr.PyRanges(df_unsolved)
            pr_enriched_higher_thresh = pr.PyRanges(df_enriched_higher_thresh)
            overlap_counts_df = pr.count_overlaps(
                {
                    "unsolved": pr_unsolved,
                    "enriched_higher_thresh": pr_enriched_higher_thresh,
                }
            ).as_df()
            overlap_counts_df["size"] = overlap_counts_df.End - overlap_counts_df.Start

            # Identify regions enriched initially but lost at a higher threshold
            df_single_enriched = overlap_counts_df[
                (overlap_counts_df["unsolved"] == 1)
                & (overlap_counts_df["enriched_higher_thresh"] == 0)
            ]
            df_single_enriched = pd.merge(
                df_single_enriched,
                df_unsolved,
                on=["Chromosome", "Start", "End"],
                how="inner",
                suffixes=("_single_enriched", "_unsolved"),
            )
            df_solved = pd.concat(
                [df_solved, df_unsolved.loc[df_single_enriched["unsolved_idx"]]],
                ignore_index=True,
            )

            # Identify regions enriched both initially and at a higher threshold
            df_double_enriched = overlap_counts_df[
                (overlap_counts_df["unsolved"] == 1)
                & (overlap_counts_df["enriched_higher_thresh"] == 1)
            ]
            df_double_enriched = pd.merge_asof(
                df_double_enriched,
                df_unsolved,
                on="Start",
                by="Chromosome",
                suffixes=("_double_enriched", "_unsolved"),
            )
            df_double_enriched["End"] = df_double_enriched["End_double_enriched"]
            df_double_enriched["below_max_size"] = (
                df_double_enriched["size"] < self.params.max_peak_size
            )

            # Add double enriched regions below max size to solved_df
            df_solved = pd.concat(
                [df_solved, df_double_enriched[df_double_enriched["below_max_size"]]],
                ignore_index=True,
            )

            # Update unsolved_df with regions above max size
            df_unsolved = df_double_enriched[~df_double_enriched["below_max_size"]][
                ["Chromosome", "Start", "End"]
            ]

        # Add any remaining unsolved regions to solved_df
        df_solved = pd.concat([df_solved, df_unsolved], ignore_index=True)
        df_solved = df_solved.sort_values(["Chromosome", "Start"]).reset_index(
            drop=True
        )

        return df_solved


class CandidatePeakCaller:
    def __init__(
        self,
        bigwig: Union[pathlib.Path, str],
        params: CandidatePeakCallParams = None,
        only_standard_chromosomes: bool = True,
        device: str = None,
    ):
        if isinstance(bigwig, pathlib.Path):
            bigwig = str(bigwig)

        self.bigwig = bigwig
        self.params = params or CandidatePeakCallParams()
        self.only_standard_chromosomes = only_standard_chromosomes
        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")

    @property
    def total_coverage(self):
        bw = pybigtools.open(self.bigwig)
        return bw.info()["summary"]["sum"]

    def calculate_chromosome_stats(self, chromosome: str) -> ChromosomeStats:
        bw = pybigtools.open(self.bigwig)
        length = bw.chroms()[chromosome]
        # For per-chromosome stats, we need to extract the values
        vals = bw.values(chromosome, 0, length)
        if vals is not None:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
        else:
            mean_val = 0.0
            std_val = 0.0

        stats = ChromosomeStats(
            name=chromosome,
            length=length,
            mean=float(mean_val),
            std=float(std_val),
        )
        return stats

    @property
    def genome_stats(self) -> Dict[str, Dict[str, float]]:
        chromosomes = Chromosomes.from_bigwig(self.bigwig)
        if self.only_standard_chromosomes:
            chromosomes = chromosomes.standard_chromosomes.chromosomes

        return {chrom: self.calculate_chromosome_stats(chrom) for chrom in chromosomes}

    def extract_chromosome_signal(
        self,
        chrom_stats: ChromosomeStats,
        smoothing: int = None,
        tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract the signal from a chromosome in the bigwig file with GPU support.
        Uses pybigtools for faster extraction.
        """
        bw = pybigtools.open(self.bigwig)
        # bw.values() returns a numpy array representing the entire chromosome
        chrom_coverage_array = bw.values(chrom_stats.name, 0, chrom_stats.length)

        if chrom_coverage_array is None:
            logger.warning(f"No reads found on chromosome {chrom_stats.name}")
            chrom_coverage_array = np.zeros(chrom_stats.length, dtype=np.float32)

        # Convert to torch tensor and move to appropriate device
        coverage_tensor = torch.as_tensor(
            chrom_coverage_array, device=self.device, dtype=torch.float32
        )

        if smoothing is not None:
            coverage_tensor = average_array(coverage_tensor, window=smoothing)

        if not tensor:
            return coverage_tensor.cpu().numpy()

        return coverage_tensor

    def call_candidate_peaks(self, n_jobs: int = None):
        """
        Call peaks using parallel processing.
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()

        chromosomes = Chromosomes.from_bigwig(self.bigwig)
        if self.only_standard_chromosomes:
            chromosomes = chromosomes.standard_chromosomes.chromosomes

        logger.info(f"Parallelizing peak calling across {n_jobs} workers")

        def process_chromosome(chrom):
            logger.debug(f"Processing chromosome {chrom}")
            chromosome_stats = self.calculate_chromosome_stats(chrom)
            coverage_array = self.extract_chromosome_signal(
                chromosome_stats, tensor=True
            )
            caller = ChromosomePeakCaller(
                chromosome=chrom,
                coverage_array=coverage_array,
                params=self.params,
            )
            result = caller.call_candidates()
            logger.debug(f"Found {result.shape[0]} peaks on chromosome {chrom}")
            return result

        # Use joblib for parallel processing
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_chromosome)(chrom) for chrom in chromosomes
        )

        df_peaks = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        df_peaks = df_peaks.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
        return df_peaks


class CandidatePeakCallerMCC(CandidatePeakCaller):
    def __init__(
        self,
        bigwig: Union[pathlib.Path, str],
        params: CandidatePeakCallParams = None,
        cis_chromosome: Optional[str | Literal["estimate"]] = None,
        n_background: int = 1,
        apply_background_filter: bool = True,
        apply_signal_filter: bool = True,
        device: str = None,
    ):
        super().__init__(bigwig, params, device=device)
        if cis_chromosome is None or cis_chromosome == "estimate":
            self.cis_chromosome = self.estimate_cis_chromosome()
        else:
            self.cis_chromosome = cis_chromosome

        self.n_background = (
            n_background  # Number of background chromosomes to use for filtering
        )
        self.apply_background_filter = apply_background_filter
        self.apply_signal_filter = apply_signal_filter

    def estimate_cis_chromosome(self):
        chromosomes = Chromosomes.from_bigwig(self.bigwig)
        chromosomes = chromosomes.standard_chromosomes.chromosomes
        chromosomes_stats = {
            chrom: self.calculate_chromosome_stats(chrom).model_dump()
            for chrom in chromosomes
        }
        chromosomes_stats = pd.DataFrame(chromosomes_stats).T
        chromosomes_stats["coverage"] = (
            chromosomes_stats["length"] * chromosomes_stats["mean"]
        )
        chromosomes_stats["coverage_fraction"] = (
            chromosomes_stats["coverage"] / self.total_coverage
        )
        chromosomes_stats["coverage_fraction"] = (
            chromosomes_stats["coverage_fraction"]
            / chromosomes_stats["coverage_fraction"].sum()
        )
        chromosomes_stats = chromosomes_stats.sort_values(
            "coverage_fraction", ascending=False
        )
        return chromosomes_stats.index[0]

    def set_cis_chromosome(self, chromosome: Optional[str] = None):
        if chromosome is None:
            self.cis_chromosome = self.estimate_cis_chromosome()
        else:
            self.cis_chromosome = chromosome

    def select_chromosomes(self):
        """
        Select chromosomes for background and cis-chromosome.
        """
        chromosomes = Chromosomes.from_bigwig(self.bigwig)
        if self.only_standard_chromosomes:
            chromosomes = chromosomes.standard_chromosomes.chromosomes

        # Select the cis-chromosome
        selected_chromosomes = [self.cis_chromosome]

        # Select random background chromosomes if requested
        if self.n_background > 0:
            # Filter out the cis chromosome from potential background chromosomes
            background_choices = [c for c in chromosomes if c != self.cis_chromosome]

            if len(background_choices) > 0:
                # Determine how many chromosomes to select (don't select more than available)
                n_to_select = min(self.n_background, len(background_choices))

                # Select random background chromosomes
                background_chromosomes = [
                    str(chrom)
                    for chrom in np.random.choice(
                        np.array(background_choices), n_to_select, replace=False
                    )
                ]
                selected_chromosomes.extend(background_chromosomes)

                logger.info(
                    f"Selected {len(background_chromosomes)} background chromosomes: {background_chromosomes}"
                )
            else:
                logger.warning("No chromosomes available for background selection")
        else:
            logger.info("No background chromosomes requested (n_background=0)")

        chromosomes = Chromosomes(selected_chromosomes)
        logger.info(f"Selected cis-chromosome: {self.cis_chromosome}")
        logger.info(f"Total number of chromosomes selected: {len(chromosomes)}")
        return chromosomes

    def background_filter(
        self, peaks: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Apply background filter to the peaks DataFrame.
        Args:
            peaks: DataFrame containing the peaks
            threshold: Threshold for filtering peaks based on background signal. Default is 0.95 i.e. 95th percentile of background signal.
        """
        from lanceotron_mcc.api.signal import add_signal

        logger.info(
            f"Applying background filter - removing peaks with signal below {threshold * 100} percentile of background signal"
        )

        peaks = add_signal(peaks, self.bigwig).df
        peaks = peaks.assign(
            is_background=lambda df: np.where(
                df.Chromosome != self.cis_chromosome, True, False
            )
        )
        threshold = peaks.query("is_background")["Signal"].quantile(threshold)
        logger.info(f"Background threshold: {threshold}")
        peaks = peaks.query("Signal >= @threshold and is_background == False")
        logger.info(f"Number of peaks after filtering: {peaks.shape[0]}")
        return peaks

    def z_score_filter(self, peaks: pd.DataFrame, threshold: float = 3.0):
        """
        Apply z-score filter to the peaks DataFrame.
        Args:
            peaks: DataFrame containing the peaks
            threshold: Threshold for filtering peaks based on z-score. Default is 3.0.
        """
        from lanceotron_mcc.api.signal import add_signal

        peaks = add_signal(peaks, self.bigwig).df
        mean = peaks.query("is_background")["Signal"].mean()
        std = peaks.query("is_background")["Signal"].std()
        z_score = (peaks["Signal"] - mean) / std
        peaks = peaks.assign(z_score=z_score)

        logger.info(f"Z-score threshold: {threshold}")

        peaks = peaks.query("z_score >= @threshold and is_background == False")
        logger.info(f"Number of peaks after Z-score filtering: {peaks.shape[0]}")
        return peaks

    def iqr_filter(self, peaks: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """
        Apply IQR filter to the peaks DataFrame.
        Args:
            peaks: DataFrame containing the peaks
            threshold: Threshold for filtering peaks based on IQR. Default is 3.0.
        """
        from lanceotron_mcc.api.signal import add_signal

        logger.info(
            f"Applying IQR filter - removing peaks with signal below {threshold} IQR"
        )
        logger.info(f"Calculating IQR threshold: {threshold} * IQR")

        peaks = add_signal(peaks, self.bigwig).df
        peaks = peaks.assign(signal_log1p=lambda df: np.log1p(df.Signal))

        # Calculate quartiles and IQR
        signal = peaks["signal_log1p"].values
        q1 = np.percentile(signal, 25)
        q3 = np.percentile(signal, 75)
        iqr = q3 - q1

        # Calculate threshold
        threshold = q3 + (threshold * iqr)

        # Binarize the data
        binary_data = (signal > threshold).astype(int)

        peaks = peaks.assign(binary_signal=binary_data)
        logger.info(f"IQR threshold: {threshold}")
        peaks = peaks.query("binary_signal == 1 and is_background == False")
        logger.info(f"Number of peaks after IQR filtering: {peaks.shape[0]}")
        return peaks

    def _process_chromosome(self, chrom):
        logger.info(f"Processing chromosome {chrom}")
        chromosome_stats = self.calculate_chromosome_stats(chrom)
        coverage_array = self.extract_chromosome_signal(chromosome_stats, tensor=True)
        caller = ChromosomePeakCaller(
            chromosome=chrom,
            coverage_array=coverage_array,
            params=self.params,
        )
        return caller.call_candidates()

    def call_candidate_peaks(self, n_jobs: int = 1):
        """
        Call peaks from a bigwig file with parallel processing.

        Args:
            n_jobs: Number of parallel processes to use
        """
        # Select chromosomes for background and cis-chromosome
        chromosomes = self.select_chromosomes()

        if n_jobs > 1:
            logger.info(f"Parallelizing peak calling across {n_jobs} workers")
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(self._process_chromosome)(chrom)
                for chrom in chromosomes.chromosomes
            )
        else:
            logger.info("Sequential processing of chromosomes")
            results = []
            for chrom in chromosomes.chromosomes:
                result = self._process_chromosome(chrom)
                results.append(result)

        df_peaks = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        if not df_peaks.empty:
            df_peaks = (
                df_peaks.sort_values(["Chromosome", "Start"])
                .reset_index(drop=True)
                .assign(
                    Start=lambda df: df.Start.astype(int),
                    End=lambda df: df.End.astype(int),
                )
            )

            logger.info(f"Total number of peaks identified: {df_peaks.shape[0]}")

            # Apply filters. add_signal is called only once inside if needed.
            from lanceotron_mcc.api.signal import add_signal

            df_peaks = add_signal(df_peaks, self.bigwig).df
            df_peaks = df_peaks.assign(
                is_background=lambda df: np.where(
                    df.Chromosome != self.cis_chromosome, True, False
                )
            )

            if self.apply_background_filter:
                # Optimized background filter
                threshold_val = df_peaks.query("is_background")["Signal"].quantile(0.95)
                logger.info(f"Background threshold: {threshold_val}")
                df_peaks = df_peaks.query("Signal >= @threshold_val")
                logger.info(
                    f"Number of peaks after background filtering: {df_peaks.shape[0]}"
                )

            if self.apply_signal_filter:
                # Optimized IQR filter
                logger.info("Applying IQR filter")
                df_peaks = df_peaks.assign(signal_log1p=lambda df: np.log1p(df.Signal))
                signal_vals = df_peaks["signal_log1p"].values
                if len(signal_vals) > 0:
                    q1 = np.percentile(signal_vals, 25)
                    q3 = np.percentile(signal_vals, 75)
                    iqr = q3 - q1
                    iqr_threshold = q3 + (3.0 * iqr)
                    logger.info(f"IQR threshold: {iqr_threshold}")
                    df_peaks = df_peaks.query("signal_log1p > @iqr_threshold")
                logger.info(f"Number of peaks after IQR filtering: {df_peaks.shape[0]}")

            # Final output should only contain cis-chromosome peaks
            df_peaks = df_peaks.query("is_background == False").drop(
                columns=["is_background", "signal_log1p"], errors="ignore"
            )

        return df_peaks
