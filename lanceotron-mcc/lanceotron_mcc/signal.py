import pathlib
import numpy as np
import numba
import pandas as pd
import pyranges as pr
from scipy.sparse import csc_array, diags, csr_array
from typing import Union, List, Tuple, Dict, Optional, Any
import pyBigWig

import torch

@torch.jit.script
def average_array(array: torch.Tensor, window: int = 0) -> torch.Tensor:
    # If no window size is provided, set the window size to the length of the array
    if window == 0:
        window = len(array)

    # If the window size is greater than the length of the array, adjust the window size
    if window > len(array):
        window = len(array)

    # If the window size is equal to the length of the array, return an array of the same length with the mean value of the input array
    if window == len(array):
        return torch.full_like(array, array.mean())

    # Calculate the cumulative sum of the array
    return_array = torch.cumsum(array, dim=0)

    # Adjust the cumulative sum to get the moving sum for the window size
    if window < len(array):
        return_array[window:] = return_array[window:] - return_array[:-window]
        # Divide by the window size to get the moving average
        return_array[window - 1 :] /= window

    # Calculate the padding needed for the left and right ends of the array
    left_pad = window // 2
    right_pad = window - left_pad - 1

    # If left padding is greater than 0, adjust the array to handle edge effects
    if left_pad > 0:
        # Create a new tensor for the result
        result = torch.empty_like(array)
        # Copy the calculated averages to the middle part of the array
        result[left_pad : len(array) - right_pad] = return_array[window - 1 :]
        # Set the values for the left and right pads to the mean of the corresponding array segments
        result[:left_pad] = array[:left_pad].mean()
        result[-right_pad:] = array[-right_pad:].mean()
    else:
        # If no left padding is needed, fill the entire return array with the mean value
        result = torch.full_like(array, array.mean())

    return result


@torch.jit.script
def _filter_enriched_regions(
    regions: torch.Tensor, 
    min: int = 0, 
    max: int = int(1e6),
) -> torch.Tensor:
    filtered_regions = torch.empty((0, 2), dtype=torch.float32)
    for i in range(regions.shape[0]):
        start = regions[i, 0]
        end = regions[i, 1]
        if end - start >= min and end - start <= max:
            filtered_regions = torch.cat((filtered_regions, regions[i].unsqueeze(0)))

    return filtered_regions


def find_enriched_regions(
    array: torch.Tensor,
    threshold: torch.Tensor,
    min_region_size: int = 0,
    max_region_size: int = float("inf"),
):
    # Ensure inputs are PyTorch tensors
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor(threshold)

    # Create a boolean tensor where values are True if they are greater than 0 and above the threshold
    truth_array = (array > 0) & (array >= threshold)

    # Check if there are any True values in the truth array
    if truth_array.any():
        # Find the indices where the value changes from True to False or vice versa
        coord_array = torch.where(truth_array[:-1] != truth_array[1:])[0]

        # If there are any changes in the truth array
        if coord_array.numel() > 0:
            # Shift indices by 1 to mark the start of new regions
            coord_array = coord_array + 1

            # Handle the case where the first value is part of an enriched region
            if coord_array.numel() % 2 == 0:
                if truth_array[0]:
                    coord_array = torch.cat([torch.tensor([0]), coord_array])
                    coord_array = torch.cat([coord_array, torch.tensor([len(array)])])
            else:
                if truth_array[0]:
                    coord_array = torch.cat([torch.tensor([0]), coord_array])
                else:
                    coord_array = torch.cat([coord_array, torch.tensor([len(array)])])

            # Reshape the coordinate array to pairs of start and end indices of enriched regions
            coord_array = coord_array.reshape(-1, 2)

            # Convert the coordinate array to a list of enriched regions
            enriched_regions = coord_array
        else:
            # Handle the case where the entire array is an enriched region
            if truth_array[0]:
                enriched_regions = torch.float32([0, len(array)])
    else:
        enriched_regions = torch.empty((0, 2), dtype=torch.float32)

    # If max_region_size is not specified, set it to the length of the array
    if max_region_size is None:
        max_region_size = len(array)

    filtered_region_list = _filter_enriched_regions(
        enriched_regions, min_region_size, max_region_size
    )
    return filtered_region_list


def add_signal(
    peaks: Union[pr.PyRanges, pd.DataFrame], bigwig: Union[pathlib.Path, str]
) -> pr.PyRanges:
    if isinstance(bigwig, pathlib.Path):
        bigwig = str(bigwig)

    if isinstance(peaks, pr.PyRanges):
        peaks = peaks.df

    signal = np.zeros(peaks.shape[0])
    with pyBigWig.open(bigwig) as bw:
        for i, peak in enumerate(peaks.itertuples()):
            signal[i] = bw.stats(peak.Chromosome, peak.Start, peak.End)[0]

    peaks["Signal"] = signal
    peaks["Signal"] = peaks["Signal"].fillna(0)
    peaks = pr.PyRanges(peaks)

    return peaks
