import concurrent.futures
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


def search_enriched_regions_param_grid(
    array: torch.Tensor, params: EnrichedRegionsParams, label: Optional[str] = None
) -> EnrichedRegionsResult:
    """
    Find enriched regions using a vectorized parameter grid search on the GPU.
    """
    device = array.device

    # 1. Pre-calculate unique moving averages for windows and backgrounds
    unique_windows = sorted(list(set(params.window_list + params.background_list)))
    averages_cache = {w: average_array(array, w) for w in unique_windows}

    # 2. Cumulative threshold array (vectorized summation of boolean results)
    # Using float32 for summation to avoid overflow before converting to int8/int16
    threshold_sum = torch.zeros(array.shape, dtype=torch.float32, device=device)

    # Generate background global min tensor once
    bg_min_t = None
    if params.background_global_min is not None:
        bg_min_t = torch.tensor(
            params.background_global_min, device=device, dtype=array.dtype
        )

    # 3. Process the parameter grid
    for background in params.background_list:
        bg_avg = averages_cache[background]
        for window in params.window_list:
            win_avg = averages_cache[window]
            for threshold in params.threshold_list:
                # Calculate threshold array for this combination
                t_arr = bg_avg * threshold
                if bg_min_t is not None:
                    t_arr = torch.maximum(t_arr, bg_min_t)

                # Vectorized comparison: directly add boolean results to the cumulative sum
                # Multiplying by win_avg > 0 ensures we don't pick up regions with no signal
                threshold_sum += (win_avg >= t_arr) & (win_avg > 0)

    # Convert final sum to target storage type
    threshold_array = threshold_sum.to(torch.int8)

    # 4. Identify final enriched regions based on the cumulative threshold
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
    return EnrichedRegionsResult(enriched_regions=enriched_regions_final)


class PeakSet:
    """A lightweight abstraction for a set of candidate peaks."""

    def __init__(self, df: pd.DataFrame, bigwig: str, cis_chromosome: str):
        self.df = df
        self.bigwig = bigwig
        self.cis_chromosome = cis_chromosome

    @property
    def empty(self) -> bool:
        return self.df.empty

    @property
    def count(self) -> int:
        return len(self.df)

    def sort(self):
        self.df = self.df.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
        return self

    def add_signal(self):
        from lanceotron_mcc.api.signal import add_signal

        self.df = add_signal(self.df, self.bigwig).df
        self.df["is_background"] = self.df.Chromosome != self.cis_chromosome
        return self

    def filter_background(self, percentile: float = 0.95):
        if self.empty or "Signal" not in self.df.columns:
            return self

        bg_peaks = self.df[self.df["is_background"]]
        if bg_peaks.empty:
            logger.warning("No background peaks found for filtering.")
            return self

        threshold = bg_peaks["Signal"].quantile(percentile)
        logger.info(f"Applying background filter: threshold={threshold:.4f}")

        # Keep cis peaks above threshold
        self.df = self.df[
            (self.df["Signal"] >= threshold) & (~self.df["is_background"])
        ]
        return self

    def filter_iqr(self, multiplier: float = 3.0):
        if self.empty or "Signal" not in self.df.columns:
            return self

        logger.info(f"Applying IQR filter: multiplier={multiplier}")
        self.df["signal_log1p"] = np.log1p(self.df["Signal"])
        vals = self.df["signal_log1p"].values

        if len(vals) > 0:
            logger.info(
                f"Signal (log1p) range: {vals.min():.4f} to {vals.max():.4f}, mean: {vals.mean():.4f}"
            )
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            threshold = q3 + (multiplier * iqr)
            logger.info(f"IQR threshold: {threshold:.4f}")

            self.df = self.df[self.df["signal_log1p"] >= threshold]

        return self

    def filter_ml(
        self,
        model: torch.nn.Module,
        window_size: int = 200,
        batch_size: int = 64,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        """Filter peaks using a machine learning model."""
        if self.empty:
            return self

        from lanceotron_mcc.api.signal import extract_fixed_signal_windows

        logger.info(f"Applying ML filter with threshold {threshold}...")
        model.eval()
        model.to(device)

        with torch.no_grad():
            # Process in batches to avoid memory issues
            indices = self.df.index.tolist()
            keep_indices = []

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                batch_df = self.df.loc[batch_idx]

                signals = extract_fixed_signal_windows(
                    self.bigwig, batch_df, window_size=window_size, device=device
                )

                preds = model(signals).cpu().numpy().flatten()

                # Identify which rows in this batch to keep
                batch_keep = [
                    batch_idx[j] for j, p in enumerate(preds) if p >= threshold
                ]
                keep_indices.extend(batch_keep)

        logger.info(f"ML filter: kept {len(keep_indices)} / {len(indices)} peaks")
        self.df = self.df.loc[keep_indices].reset_index(drop=True)
        return self

    def finalize(self) -> pd.DataFrame:
        """Return the final filtered DataFrame on the cis chromosome."""
        if self.empty:
            return pd.DataFrame(columns=["Chromosome", "Start", "End", "Signal"])

        columns_to_keep = ["Chromosome", "Start", "End"]
        if "Signal" in self.df.columns:
            columns_to_keep.append("Signal")

        return (
            self.df.query("Chromosome == @self.cis_chromosome")[columns_to_keep]
            .sort_values(["Start"])
            .reset_index(drop=True)
        )


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

    def _get_background_min(self) -> float:
        if self.params.background_global_min == "mean":
            return float(self.coverage_array.mean())
        return self.params.background_global_min

    def _initial_search(self) -> Tuple[pd.DataFrame, torch.Tensor]:
        """Perform the initial search for enriched regions using a parameter grid."""
        background_global_min = self._get_background_min()

        res = search_enriched_regions_param_grid(
            self.coverage_array,
            EnrichedRegionsParams(
                background_list=self.params.threshold_background_lst,
                window_list=self.params.window_sizes,
                threshold_list=self.params.threshold_lst,
                threshold_cumulative=self.params.threshold_initial,
                min_region_size=self.params.min_peak_size,
                max_region_size=self.params.max_peak_size,
                background_global_min=background_global_min,
                return_threshold_array=True,
            ),
            label=self.chromosome,
        )

        df = pd.DataFrame(res.to_dataframe(), columns=["Start", "End"])
        df["Chromosome"] = self.chromosome
        df["Length"] = df["End"] - df["Start"]

        # Initial filtering by minimum size
        df = df[df["Length"] >= self.params.min_peak_size]
        return df, res.threshold_array

    def _refine_unsolved_regions(
        self,
        df_unsolved: pd.DataFrame,
        threshold_array: torch.Tensor,
        threshold_cumulative: float,
        threshold_limit: int,
    ) -> pd.DataFrame:
        """Iteratively refine regions that are larger than `max_peak_size`."""
        df_solved_list = []

        while not df_unsolved.empty and threshold_cumulative < threshold_limit - 1:
            df_unsolved = df_unsolved.sort_values(["Chromosome", "Start"]).reset_index(
                drop=True
            )
            df_unsolved["unsolved_idx"] = df_unsolved.index
            threshold_cumulative += 1

            # Find regions that meet the higher threshold
            higher_regions = find_enriched_regions(
                threshold_array, threshold_cumulative
            )
            df_higher = pd.DataFrame(higher_regions, columns=["Start", "End"])
            df_higher.insert(0, "Chromosome", self.chromosome)

            # Analyze overlaps between current unsolved and higher-threshold regions
            pr_unsolved = pr.PyRanges(df_unsolved)
            pr_higher = pr.PyRanges(df_higher)
            overlap_df = pr.count_overlaps(
                {"unsolved": pr_unsolved, "higher": pr_higher}
            ).as_df()
            overlap_df["size"] = overlap_df.End - overlap_df.Start

            # 1. Single overlap or lost (Solved): No new peaks emerged at higher threshold
            df_single = overlap_df[
                (overlap_df["unsolved"] == 1) & (overlap_df["higher"] == 0)
            ]
            if not df_single.empty:
                indices = pd.merge(
                    df_single, df_unsolved, on=["Chromosome", "Start", "End"]
                )["unsolved_idx"]
                df_solved_list.append(df_unsolved.loc[indices])

            # 2. Double overlap (Potential splitting or remaining above max size)
            df_multi = overlap_df[
                (overlap_df["unsolved"] == 1) & (overlap_df["higher"] == 1)
            ]
            if not df_multi.empty:
                df_multi = pd.merge_asof(
                    df_multi,
                    df_unsolved,
                    on="Start",
                    by="Chromosome",
                    suffixes=("_multi", "_unsolved"),
                )
                df_multi["End"] = df_multi["End_multi"]
                is_below = df_multi["size"] < self.params.max_peak_size

                # If below max size, it's solved
                df_solved_list.append(df_multi[is_below])

                # If still above max size, it remains unsolved for the next iteration
                df_unsolved = df_multi[~is_below][["Chromosome", "Start", "End"]]
            else:
                df_unsolved = pd.DataFrame(columns=["Chromosome", "Start", "End"])

        if not df_unsolved.empty:
            df_solved_list.append(df_unsolved)

        return (
            pd.concat(df_solved_list, ignore_index=True)
            if df_solved_list
            else df_unsolved
        )

    def call_candidates(self) -> pd.DataFrame:
        """Call candidate peaks using multi-threshold refinement."""
        threshold_limit = (
            len(self.params.threshold_background_lst)
            * len(self.params.window_sizes)
            * len(self.params.threshold_lst)
        )

        df_enriched, threshold_array = self._initial_search()

        # Split into solved (correct size) and unsolved (too large)
        df_unsolved = df_enriched[df_enriched["Length"] > self.params.max_peak_size]
        df_solved = df_enriched[df_enriched["Length"] <= self.params.max_peak_size]

        if not df_unsolved.empty:
            df_refined = self._refine_unsolved_regions(
                df_unsolved,
                threshold_array,
                self.params.threshold_initial,
                threshold_limit,
            )
            df_solved = pd.concat([df_solved, df_refined], ignore_index=True)

        return df_solved.sort_values(["Chromosome", "Start"]).reset_index(drop=True)


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
        iqr_multiplier: float = 3.0,
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
        self.iqr_multiplier = iqr_multiplier

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

    def _identify_all_candidates(
        self, chromosomes: List[str], n_jobs: int
    ) -> pd.DataFrame:
        if n_jobs > 1:
            logger.info(f"Parallelizing peak calling across {n_jobs} workers")
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(self._process_chromosome)(chrom) for chrom in chromosomes
            )
        else:
            logger.info("Sequential processing of chromosomes")
            results = [self._process_chromosome(chrom) for chrom in chromosomes]

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def call_candidate_peaks(
        self,
        n_jobs: int = 1,
        train_ml: bool = False,
        ml_model_path: Optional[str] = None,
        ml_window_size: int = 200,
        ml_threshold: float = 0.5,
        epochs: int = 10,
    ) -> pd.DataFrame:
        """Call candidate peaks and apply filtering pipeline."""
        target_chroms = self.select_chromosomes().chromosomes

        # 1. Identification phase
        df_raw = self._identify_all_candidates(target_chroms, n_jobs)
        if df_raw.empty:
            return df_raw

        # 2. Filtering phase using PeakSet abstraction
        peaks = PeakSet(df_raw, self.bigwig, self.cis_chromosome)
        logger.info(f"Total raw candidates identified: {peaks.count}")

        peaks.sort().add_signal()

        if self.apply_background_filter:
            peaks.filter_background()
            logger.info(f"Candidates after background filtering: {peaks.count}")

        if self.apply_signal_filter:
            peaks.filter_iqr(multiplier=self.iqr_multiplier)
            logger.info(f"Candidates after IQR filtering: {peaks.count}")

        # 3. Semi-supervised ML Filtering
        if train_ml or (ml_model_path and pathlib.Path(ml_model_path).exists()):
            from lanceotron_mcc.api.models import is_ml_available

            if not is_ml_available():
                logger.warning(
                    "Machine learning features (lightning) are not available. Skipping ML filtering."
                )
                return peaks.finalize()

        if train_ml:
            from lanceotron_mcc.api.models import SignalViT
            from lanceotron_mcc.api.signal import extract_fixed_signal_windows
            import lightning as L
            from torch.utils.data import DataLoader, TensorDataset

            logger.info(
                "Starting semi-supervised ML training with PyTorch Lightning..."
            )

            # Prepare training data
            # Positive: Current candidate peaks on cis-chromosome
            pos_df = peaks.df[peaks.df.Chromosome == self.cis_chromosome]
            if pos_df.empty:
                logger.warning("No candidate peaks for ML training. Skipping.")
            else:
                # Negative: Random regions from background chromosomes
                bg_chroms = [c for c in target_chroms if c != self.cis_chromosome]
                if not bg_chroms:
                    logger.warning(
                        "No background chromosomes for ML training. Skipping."
                    )
                else:
                    neg_regions = []
                    bw = pybigtools.open(self.bigwig)
                    chrom_sizes = bw.chroms()

                    # Sample negatives roughly equal to positives
                    for i in range(len(pos_df)):
                        chrom = np.random.choice(bg_chroms)
                        size = chrom_sizes[chrom]
                        start = np.random.randint(0, size - ml_window_size)
                        neg_regions.append(
                            {
                                "Chromosome": chrom,
                                "Start": start,
                                "End": start + ml_window_size,
                            }
                        )

                    neg_df = pd.DataFrame(neg_regions)

                    # Extract signals
                    pos_signal = extract_fixed_signal_windows(
                        self.bigwig,
                        pos_df,
                        window_size=ml_window_size,
                        device=self.device,
                    )
                    neg_signal = extract_fixed_signal_windows(
                        self.bigwig,
                        neg_df,
                        window_size=ml_window_size,
                        device=self.device,
                    )

                    # Create Dataset
                    x_train = torch.cat([pos_signal, neg_signal], dim=0)
                    y_train = torch.cat(
                        [
                            torch.ones(pos_signal.shape[0], 1),
                            torch.zeros(neg_signal.shape[0], 1),
                        ],
                        dim=0,
                    )
                    train_loader = DataLoader(
                        TensorDataset(x_train, y_train), batch_size=32, shuffle=True
                    )

                    # Initialize model and trainer
                    model = SignalViT(input_dim=ml_window_size).to(self.device)
                    trainer = L.Trainer(
                        max_epochs=epochs,
                        accelerator="auto",
                        devices=1,
                        enable_checkpointing=False,
                        logger=False,
                    )

                    trainer.fit(model, train_loader)

                    if ml_model_path:
                        torch.save(model.state_dict(), ml_model_path)
                        logger.info(f"Model saved to {ml_model_path}")

                    # Apply filter
                    peaks.filter_ml(
                        model,
                        window_size=ml_window_size,
                        threshold=ml_threshold,
                        device=self.device,
                    )

        elif ml_model_path and pathlib.Path(ml_model_path).exists():
            from lanceotron_mcc.api.models import load_model

            logger.info(f"Loading ML model from {ml_model_path} for filtering...")
            model = load_model(
                ml_model_path, device=self.device, input_dim=ml_window_size
            )
            peaks.filter_ml(
                model,
                window_size=ml_window_size,
                threshold=ml_threshold,
                device=self.device,
            )

        return peaks.finalize()
