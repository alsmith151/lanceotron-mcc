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
    threshold_initial: float = 9.0,
    threshold_lst: Annotated[List[float], Parameter(negative=[])] = [2.0, 4.0, 6.0],
    threshold_background_lst: Annotated[List[int], Parameter(negative=[])] = [int(1e4), int(1e5), int(1e6)],
    window_sizes: Annotated[List[int], Parameter(negative=[])] = [200, 400, 600],
    min_peak_size: int = 50,
    max_peak_size: int = 500,
    min_read_count: int = 5,
    coverage_depth_per_single_read: Union[float, Literal['estimate']] = 'estimate',
    background_global_min: Union[float, Literal["mean"]] = "mean",
    use_gpu: bool = True,
    n_jobs: Optional[int] = 2,
    mcc_filter_background: bool = True,
    mcc_filter_enriched_regions: bool = True,   
):
    """
    Call peaks from a bigwig file with GPU and parallel processing support.

    Args:
        bigwig: Path to the input bigwig file
        outfile: Path to the output bed file
        n_background_chromosomes: Number of background chromosomes to use
        cis_chromosome: Chromosome to use for peak calling (default: "estimate")
        threshold_initial: Initial threshold for peak calling
        threshold_lst: List of thresholds for peak calling
        threshold_background_lst: List of thresholds for background regions
        window_sizes: List of window sizes for peak calling
        min_peak_size: Minimum size of peaks to call
        max_peak_size: Maximum size of peaks to call
        min_read_count: Minimum read count for peak calling
        coverage_depth_per_single_read: Coverage depth per single read (default: "estimate")
        background_global_min: Minimum value for background regions (default: "mean")
        use_gpu: Whether to use GPU acceleration if available
        n_jobs: Number of parallel processes to use (default: all CPU cores)
        mcc_filter_background: Whether to filter background regions (default: True)
    """
    import torch

    device = get_device(use_gpu=use_gpu)
    logger.info(f"Reading bigwig file: {bigwig} and using device: {device}")
    
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
        device=device,

    )

    logger.info(f"Calling candidate peaks with {n_jobs} parallel workers...")
    peaks = peak_caller.call_candidate_peaks(
        n_jobs=n_jobs,
    )

    logger.info(f'Total number of peaks identified: {peaks.shape[0]}')
    logger.info(f"Writing peaks to {outfile}")
    gr_peak_calls = pr.PyRanges(peaks)
    gr_peak_calls.to_bed(outfile, keep=False)



def main():
    app()



if __name__ == "__main__":
    main()
