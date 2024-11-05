import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pyranges as pr
from cyclopts import App, Parameter
from loguru import logger
from typing_extensions import Annotated

from lanceotron_mcc.api import candidates, signal, utils

app = App()

@app.command()
def call_peaks(
    bigwig: pathlib.Path,
    outfile: pathlib.Path,
    n_background_chromosomes: int = 0,
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
):
    """
    Call peaks from a bigwig file.

    Args:
        bigwig: Path to the bigwig file.
        outfile: Path to the output file.
        n_background_chromosomes: Number of background chromosomes to use for estimating the background.
        cis_chromosome: User supplied name of the cis chromosome for the viewpoint. If 'estimate', the chromosome with the most reads is used.
        threshold_initial: Initial threshold to use for peak calling.
        threshold_lst: List of thresholds to use for peak calling.
        threshold_background_lst: List of thresholds to use for background estimation.
        window_sizes: List of window sizes to use for peak calling.
        min_peak_size: Minimum size of a peak.
        max_peak_size: Maximum size of a peak.
        min_read_count: Minimum number of reads in a peak.
        coverage_depth_per_single_read: Coverage depth per single read.
        background_global_min: Global minimum to use for background estimation.

    """

    logger.info(f"Reading bigwig file: {bigwig}")
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
        cis_chromosome=cis_chromosome,
    )

    logger.info("Calling candidate peaks....")
    peaks = peak_caller.call_candidate_peaks(
        n_background=n_background_chromosomes
    )

    logger.info(f'Total number of peaks identified: {peaks.shape[0]}')
    logger.info(f"Writing peaks to {outfile}")
    gr_peak_calls = pr.PyRanges(peaks)
    gr_peak_calls.to_bed(outfile, keep=False)



def main():
    app()



if __name__ == "__main__":
    main()
