import numpy as np
import pyBigWig
import pandas as pd
import pydantic
import torch
from typing import List, Optional, Tuple, Union, Literal, Dict
import itertools
import pathlib
from loguru import logger
import pyranges as pr
import tqdm
import concurrent.futures

from .signal import average_array, find_enriched_regions
from .utils import Chromosomes, ChromosomeStats


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
        df = pd.DataFrame(self.enriched_regions, columns=["Start", "End"])
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
    Find enriched regions in an array using a parameter grid of background sizes, window sizes, and threshold values.

    Uses Ray to parallelize the search across all parameter combinations.

    Parameters
    ----------
    array : np.ndarray
        The input array to analyze.
    params : EnrichedRegionsParams
        The parameters for the search.

    Returns
    -------
    EnrichedRegionsResult
        The result containing a list of enriched regions identified in the array and optionally the cumulative threshold array.
    """

    # Initialize an array to accumulate threshold values
    threshold_array = torch.zeros(array.shape, dtype=torch.int8)
    combinations = list(
        itertools.product(
            params.background_list, params.window_list, params.threshold_list
        )
    )

    for background, window, threshold in combinations:
        regions = search_enriched_regions(
            array=array,
            background=background,
            window=window,
            threshold=threshold,
            background_global_min=params.background_global_min,
            min_region_size=params.min_region_size,
            max_region_size=params.max_region_size,
        )

        for i in range(regions.shape[0]):
            start = int(regions[i, 0].item())
            end = int(regions[i, 1].item())
            threshold_array[start:end] += 1

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
    ):
        self.bigwig = bigwig
        self.params = params or CandidatePeakCallParams()
        self.only_standard_chromosomes = only_standard_chromosomes

    @property
    def total_coverage(self):
        with pyBigWig.open(self.bigwig) as bw:
            return bw.header()["sumData"]

    def calculate_chromosome_stats(self, chromosome: str) -> ChromosomeStats:
        with pyBigWig.open(self.bigwig) as bw:
            stats = ChromosomeStats(
                name=chromosome,
                length=bw.chroms(chromosome),
                mean=bw.stats(chromosome, type="mean", exact=True)[0],
                std=bw.stats(chromosome, type="std", exact=True)[0],
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
        Extract the signal from a chromosome in the bigwig file.

        Args:
            chrom_stats (ChromosomeStats): Chromosome statistics object.
            smoothing (int): Size of the moving average window for smoothing the signal.
            tensor (bool): Whether to return a torch tensor.

        Returns:
            Union[np.ndarray, torch.Tensor]: Array of signal values for the chromosome.
        """

        chrom_coverage_array = np.zeros(chrom_stats.length, dtype=np.float32)

        with pyBigWig.open(self.bigwig) as bw:
            reads_intervals = bw.intervals(chrom_stats.name)
            if reads_intervals is not None:
                for start, end, value in reads_intervals:
                    chrom_coverage_array[start:end] = value
            else:
                logger.warning(f"No reads found on chromosome {chrom_stats.name}")

        if smoothing is not None:
            chrom_coverage_array = average_array(chrom_coverage_array, window=smoothing)

        if tensor:
            chrom_coverage_array = torch.tensor(chrom_coverage_array)

        return chrom_coverage_array

    def call_candidate_peaks(self, n_jobs: int = 1):
        chromosomes = Chromosomes.from_bigwig(self.bigwig)
        if self.only_standard_chromosomes:
            chromosomes = chromosomes.standard_chromosomes.chromosomes

        df_peaks = pd.DataFrame()
        for chrom in tqdm.tqdm(
            chromosomes,
            desc="Calling candidate peaks for chromosomes",
            total=len(chromosomes),
        ):
            chromosome_stats = self.calculate_chromosome_stats(chrom)
            coverage_array = self.extract_chromosome_signal(chromosome_stats)
            caller = ChromosomePeakCaller(
                chromosome=chrom,
                coverage_array=coverage_array,
                params=self.params,
            )
            df_peaks_chrom = caller.call_candidates()
            logger.info(f"Found {df_peaks_chrom.shape[0]} peaks on chromosome {chrom}")
            df_peaks = pd.concat([df_peaks, df_peaks_chrom], ignore_index=True)

        df_peaks = df_peaks.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
        return df_peaks


class CandidatePeakCallerMCC(CandidatePeakCaller):
    def __init__(
        self,
        bigwig: Union[pathlib.Path, str],
        params: CandidatePeakCallParams = None,
        cis_chromosome: Optional[str] = None,
    ):
        super().__init__(bigwig, params)
        self.cis_chromosome = cis_chromosome

    def estimate_cis_chromosome(self):
        chromosomes = Chromosomes.from_bigwig(self.bigwig)
        chromosomes = chromosomes.standard_chromosomes.chromosomes
        chromosomes_stats = {
            chrom: self.calculate_chromosome_stats(chrom).dict()
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

    def call_candidate_peaks(self, n_background: int = 1):
        """
        Call candidate peaks for the bigwig file.

        Args:
            n_backrgound_chromosomes (int): Number of background chromosomes to sample.

        Returns:
            pd.DataFrame: DataFrame of candidate peaks.

        """
        chromosomes = Chromosomes.from_bigwig(self.bigwig)

        if self.cis_chromosome is not None and n_background > 0:
            chromosomes_cis = chromosomes.filter_chromosomes(
                include=self.cis_chromosome
            )
            chromosomes_background = (
                chromosomes.standard_chromosomes.filter_chromosomes(
                    exclude=self.cis_chromosome
                )
            )
            chromosomes_background = chromosomes_background.sample(n_background)
            chromosomes = chromosomes_cis + chromosomes_background

        elif self.cis_chromosome is not None:
            chromosomes = chromosomes.filter_chromosomes(include=self.cis_chromosome)

        if self.only_standard_chromosomes:
            chromosomes = chromosomes.standard_chromosomes.chromosomes
        else:
            chromosomes = chromosomes.chromosomes

        df_peaks = pd.DataFrame()
        logger.info(f"Calling candidate peaks for chromosomes: {chromosomes}")

        for chrom in tqdm.tqdm(
            chromosomes,
            desc="Calling candidate peaks for chromosomes",
            total=len(chromosomes),
        ):
            chromosome_stats = self.calculate_chromosome_stats(chrom)
            coverage_array = self.extract_chromosome_signal(
                chromosome_stats, tensor=True
            )
            caller = ChromosomePeakCaller(
                chromosome=chrom,
                coverage_array=coverage_array,
                params=self.params,
            )
            df_peaks_chrom = caller.call_candidates()
            logger.info(f"Found {df_peaks_chrom.shape[0]} peaks on chromosome {chrom}")
            df_peaks = pd.concat([df_peaks, df_peaks_chrom], ignore_index=True)

        df_peaks = df_peaks.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
        return df_peaks
