import pathlib
import numpy as np
import pandas as pd
import pyranges as pr
from typing import Union, List, Tuple, Dict, Optional, Any
import pyBigWig

import torch

@torch.jit.script
def average_array(array: torch.Tensor, window: int = 0) -> torch.Tensor:
    """GPU-optimized moving average function."""
    # Store original device for consistent return
    device = array.device
    
    # If no window size is provided, set the window size to the length of the array
    if window == 0:
        window = len(array)

    # If the window size is greater than the length of the array, adjust the window size
    if window > len(array):
        window = len(array)

    # If the window size is equal to the length of the array, return an array of the same length with the mean value
    if window == len(array):
        return torch.full_like(array, array.mean())

    # Calculate the cumulative sum of the array
    return_array = torch.cumsum(array, dim=0)

    # Adjust the cumulative sum to get the moving sum for the window size
    if window < len(array):
        return_array[window:] = return_array[window:] - return_array[:-window]
        # Divide by the window size to get the moving average
        return_array[window - 1:] /= window

    # Calculate the padding needed for the left and right ends of the array
    left_pad = window // 2
    right_pad = window - left_pad - 1

    # If left padding is greater than 0, adjust the array to handle edge effects
    if left_pad > 0:
        # Create a new tensor for the result
        result = torch.empty_like(array)
        # Copy the calculated averages to the middle part of the array
        result[left_pad:len(array) - right_pad] = return_array[window - 1:]
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
    """Filter regions by size.
    
    This function ensures all operations happen on the same device.
    """
    if regions.shape[0] == 0:
        return regions
        
    # Ensure we're working with the right device throughout
    device = regions.device
    filtered_regions = torch.empty((0, 2), dtype=regions.dtype, device=device)
    
    # Calculate region sizes first
    sizes = regions[:, 1] - regions[:, 0]
    
    # Create mask for regions that meet size criteria
    valid_mask = (sizes >= min) & (sizes <= max)
    
    # Use masked selection to get valid regions
    filtered_regions = regions[valid_mask]
    
    return filtered_regions

def find_enriched_regions(
    array: torch.Tensor,
    threshold: torch.Tensor,
    min_region_size: int = 0,
    max_region_size: Optional[int] = None,
):
    """Find regions in the array where values exceed the threshold.
    
    This function is optimized to handle device compatibility issues,
    particularly with Apple MPS devices.
    """
    # Check if input is MPS device - in which case we'll process on CPU to avoid known issues
    is_mps = array.device.type == 'mps'
    original_device = array.device
    
    # Use CPU for processing with MPS devices since certain operations aren't well supported
    processing_device = torch.device('cpu') if is_mps else original_device
    
    # Ensure inputs are PyTorch tensors on the processing device
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array, device=processing_device)
    else:
        array = array.to(processing_device)
        
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor(threshold, device=processing_device)
    else:
        threshold = threshold.to(processing_device)

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
                    # Create tensors on the processing device
                    zero_tensor = torch.tensor([0], device=processing_device)
                    len_array_tensor = torch.tensor([len(array)], device=processing_device)
                    coord_array = torch.cat([zero_tensor, coord_array])
                    coord_array = torch.cat([coord_array, len_array_tensor])
            else:
                if truth_array[0]:
                    zero_tensor = torch.tensor([0], device=processing_device)
                    coord_array = torch.cat([zero_tensor, coord_array])
                else:
                    len_array_tensor = torch.tensor([len(array)], device=processing_device)
                    coord_array = torch.cat([coord_array, len_array_tensor])
            
            # Reshape the coordinate array to pairs of start and end indices of enriched regions
            coord_array = coord_array.reshape(-1, 2)
            
            # Convert the coordinate array to a list of enriched regions
            enriched_regions = coord_array.to(torch.float32)
        else:
            # Handle the case where the entire array is an enriched region
            if truth_array[0]:
                enriched_regions = torch.tensor([[0, len(array)]], dtype=torch.float32, device=processing_device)
            else:
                enriched_regions = torch.empty((0, 2), dtype=torch.float32, device=processing_device)
    else:
        enriched_regions = torch.empty((0, 2), dtype=torch.float32, device=processing_device)
    
    # If max_region_size is not specified, set it to the length of the array
    if max_region_size is None:
        max_region_size = len(array)
    
    # Filter regions by size - ensuring we're still on the processing device
    filtered_regions = torch.empty((0, 2), dtype=torch.float32, device=processing_device)
    
    for i in range(enriched_regions.shape[0]):
        start = enriched_regions[i, 0]
        end = enriched_regions[i, 1]
        if end - start >= min_region_size and end - start <= max_region_size:
            row = enriched_regions[i].unsqueeze(0)
            filtered_regions = torch.cat((filtered_regions, row))
            
    # Move back to the original device
    if filtered_regions.device != original_device:
        filtered_regions = filtered_regions.to(original_device)
            
    return filtered_regions

def add_signal(
    peaks: Union[pr.PyRanges, pd.DataFrame], bigwig: Union[pathlib.Path, str]) -> pr.PyRanges:
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
