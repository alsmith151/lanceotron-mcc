import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pyranges as pr
from cyclopts import App, Parameter
from loguru import logger
from typing_extensions import Annotated

from lanceotron_mcc.api import candidates, signal, utils
from lanceotron_mcc.api.utils import get_device

app = App()


@app.command()
def call_mcc_peaks(
    bigwig: pathlib.Path,
    outfile: pathlib.Path,
    n_background_chromosomes: int = 1,
    cis_chromosome: Union[str, Literal["estimate"]] = "estimate",
    threshold_initial: float = 1.0,
    threshold_lst: Annotated[List[float], Parameter(negative=[])] = [2.0, 4.0, 6.0],
    threshold_background_lst: Annotated[List[int], Parameter(negative=[])] = [
        int(1e4),
        int(1e5),
        int(1e6),
    ],
    window_sizes: Annotated[List[int], Parameter(negative=[])] = [200, 400, 600],
    min_peak_size: int = 50,
    max_peak_size: int = 500,
    min_read_count: int = 5,
    coverage_depth_per_single_read: Union[float, Literal["estimate"]] = "estimate",
    background_global_min: Union[float, Literal["mean"]] = "mean",
    use_gpu: bool = True,
    n_jobs: Optional[int] = 2,
    mcc_filter_background: bool = True,
    mcc_filter_enriched_regions: bool = True,
    mcc_iqr_multiplier: float = 3.0,
    train_ml: bool = False,
    ml_model_path: Optional[pathlib.Path] = None,
    ml_window_size: int = 200,
    ml_threshold: float = 0.5,
    ml_epochs: int = 10,
):
    """
    Call peaks from a bigwig file with GPU and parallel processing support.
    Designed for MCC (Malignant Cell Cycle) data but applicable to general peak calling.

    Args:
        bigwig: Path to the input bigwig file.
        outfile: Path where the output bed file will be saved.
        n_background_chromosomes: Number of non-peak chromosomes to use for building a null model (default: 1).
        cis_chromosome: The target chromosome for peak calling. Use "estimate" to auto-detect the one with most enrichment (default: "estimate").
        threshold_initial: Initial enrichment threshold for finding candidate region seeds. Lower values (e.g. 1.0) provide more candidates for the internal scoring model (default: 1.0).
        threshold_lst: List of enrichment thresholds for multi-threshold refinement (default: [2.0, 4.0, 6.0]).
        threshold_background_lst: List of background smoothing windows to use for local enrichment calculation (default: [1e4, 1e5, 1e6]).
        window_sizes: List of signal smoothing windows for sensitivity at different peak sizes (default: [200, 400, 600]).
        min_peak_size: Minimum width of peaks to keep (default: 50).
        max_peak_size: Maximum width of peaks to keep (default: 500).
        min_read_count: Minimum raw signal sum in a region to be considered a peak (default: 5).
        coverage_depth_per_single_read: Used for scaling. "estimate" will attempt to calculate read depth (default: "estimate").
        background_global_min: Minimum background value for enrichment calculation. "mean" uses the chromosome average (default: "mean").
        use_gpu: Whether to use MPS/CUDA acceleration (default: True).
        n_jobs: Number of parallel processes for chromosome processing (default: 2).
        mcc_filter_background: Enable/disable the background signal distribution filter (default: True).
        mcc_filter_enriched_regions: Enable/disable the IQR-based outlier filter (default: True).
        mcc_iqr_multiplier: Multiplier for the IQR outlier filter (Q3 + multiplier * IQR). Higher values are more conservative (default: 3.0).
    """
    import torch

    device = get_device(use_gpu=use_gpu)
    logger.info(f"Reading bigwig file: {bigwig} and using device: {device}")

    if train_ml or ml_model_path:
        from lanceotron_mcc.api.models import is_ml_available

        if not is_ml_available():
            logger.error(
                "Machine learning features (lightning) are not available. Please install 'lightning' to use these flags."
            )
            return

    params = candidates.CandidatePeakCallParams(
        threshold_initial=threshold_initial,
        threshold_lst=threshold_lst,
        threshold_background_lst=threshold_background_lst,
        window_sizes=window_sizes,
        min_peak_size=min_peak_size,
        max_peak_size=max_peak_size,
        min_read_count=min_read_count,
        coverage_depth_per_single_read=coverage_depth_per_single_read,
        background_global_min=background_global_min,
    )
    logger.info(f"Peak calling parameters: {params}")
    peak_caller = candidates.CandidatePeakCallerMCC(
        bigwig=bigwig,
        params=params,
        n_background=n_background_chromosomes,
        cis_chromosome=cis_chromosome,
        apply_background_filter=mcc_filter_background,
        apply_signal_filter=mcc_filter_enriched_regions,
        iqr_multiplier=mcc_iqr_multiplier,
        device=device,
    )

    peaks = peak_caller.call_candidate_peaks(
        n_jobs=n_jobs,
        train_ml=train_ml,
        ml_model_path=str(ml_model_path) if ml_model_path else None,
        ml_window_size=ml_window_size,
        ml_threshold=ml_threshold,
        epochs=ml_epochs,
    )

    logger.info(f"Total number of peaks identified: {peaks.shape[0]}")
    logger.info(f"Writing peaks to {outfile}")
    gr_peak_calls = pr.PyRanges(peaks)
    gr_peak_calls.to_bed(outfile, keep=False)


def main():
    app()


if __name__ == "__main__":
    main()
