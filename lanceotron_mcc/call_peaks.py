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
    use_gpu: bool = True,
    n_jobs: int = None
):
    """
    Call peaks from a bigwig file with GPU and parallel processing support.

    Args:
        ...
        use_gpu: Whether to use GPU acceleration if available
        n_jobs: Number of parallel processes to use (default: all CPU cores)
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
        cis_chromosome=cis_chromosome,
        device=device,
    )

    logger.info(f"Calling candidate peaks with {n_jobs} parallel workers...")
    peaks = peak_caller.call_candidate_peaks(
        n_background=n_background_chromosomes,
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
