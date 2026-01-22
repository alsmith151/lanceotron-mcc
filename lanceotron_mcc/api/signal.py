import pathlib
import numpy as np
import pandas as pd
import pyranges as pr
from typing import Union, Optional
from loguru import logger
import pybigtools

import torch

# @torch.jit.script
# def average_array(array: torch.Tensor, window: int = 0) -> torch.Tensor:
#     """GPU-optimized moving average function."""
#     # Store original device for consistent return
#     device = array.device

#     # If no window size is provided, set the window size to the length of the array
#     if window == 0:
#         window = len(array)

#     # If the window size is greater than the length of the array, adjust the window size
#     if window > len(array):
#         window = len(array)

#     # If the window size is equal to the length of the array, return an array of the same length with the mean value
#     if window == len(array):
#         return torch.full_like(array, array.mean())

#     # Calculate the cumulative sum of the array
#     return_array = torch.cumsum(array, dim=0)

#     # Adjust the cumulative sum to get the moving sum for the window size
#     if window < len(array):
#         return_array[window:] = return_array[window:] - return_array[:-window]
#         # Divide by the window size to get the moving average
#         return_array[window - 1:] /= window

#     # Calculate the padding needed for the left and right ends of the array
#     left_pad = window // 2
#     right_pad = window - left_pad - 1

#     # If left padding is greater than 0, adjust the array to handle edge effects
#     if left_pad > 0:
#         # Create a new tensor for the result
#         result = torch.empty_like(array)
#         # Copy the calculated averages to the middle part of the array
#         result[left_pad:len(array) - right_pad] = return_array[window - 1:]
#         # Set the values for the left and right pads to the mean of the corresponding array segments
#         result[:left_pad] = array[:left_pad].mean()
#         result[-right_pad:] = array[-right_pad:].mean()
#     else:
#         # If no left padding is needed, fill the entire return array with the mean value
#         result = torch.full_like(array, array.mean())

#     return result


@torch.jit.script
def average_array(array: torch.Tensor, window: int = 0) -> torch.Tensor:
    """GPU-optimized moving average function with reduced memory usage."""
    # Early return conditions
    array_len = array.size(0)

    # If no window size or window size equals or exceeds array length
    if window == 0 or window >= array_len:
        # Calculate mean once and return filled tensor
        return torch.full_like(array, array.mean())

    # Calculate the padding needed for edges
    left_pad = window // 2
    right_pad = window - left_pad - 1

    # Create output tensor (only once)
    result = torch.empty_like(array)

    # Optimization: use direct indexing to handle edge cases (start)
    if left_pad > 0:
        # For left edge, calculate mean directly instead of using array[:left_pad].mean()
        edge_mean = torch.sum(array[:left_pad]) / left_pad
        result[:left_pad] = edge_mean

    # Main calculation: use cumulative sum with direct indexing
    if window < array_len:
        # Calculate cumulative sum but only for the needed portion
        cum_sum = torch.cumsum(array, dim=0)

        # Calculate moving average for middle section
        # For indices window-1 to len-1, we have a full window
        result[left_pad : array_len - right_pad] = cum_sum[window - 1 :]

        # Subtract the elements before the window starts
        if window < array_len:
            result[left_pad : array_len - right_pad] -= torch.cat(
                [
                    torch.zeros(1, device=array.device, dtype=array.dtype),
                    cum_sum[: array_len - window],
                ]
            )

        # Divide by window size (in-place operation)
        result[left_pad : array_len - right_pad] /= window

    # Optimization: use direct indexing to handle edge cases (end)
    if right_pad > 0:
        # For right edge, calculate mean directly
        edge_mean = torch.sum(array[-right_pad:]) / right_pad
        result[-right_pad:] = edge_mean

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


# def find_enriched_regions(
#     array: torch.Tensor,
#     threshold: torch.Tensor,
#     min_region_size: int = 0,
#     max_region_size: Optional[int] = None,
# ):
#     """Find regions in the array where values exceed the threshold.

#     This function is optimized to handle device compatibility issues,
#     particularly with Apple MPS devices.
#     """
#     # Check if input is MPS device - in which case we'll process on CPU to avoid known issues
#     is_mps = array.device.type == 'mps'
#     original_device = array.device

#     # Use CPU for processing with MPS devices since certain operations aren't well supported
#     processing_device = torch.device('cpu') if is_mps else original_device

#     # Ensure inputs are PyTorch tensors on the processing device
#     if not isinstance(array, torch.Tensor):
#         array = torch.tensor(array, device=processing_device)
#     else:
#         array = array.to(processing_device)

#     if not isinstance(threshold, torch.Tensor):
#         threshold = torch.tensor(threshold, device=processing_device)
#     else:
#         threshold = threshold.to(processing_device)

#     # Create a boolean tensor where values are True if they are greater than 0 and above the threshold
#     truth_array = (array > 0) & (array >= threshold)

#     # Check if there are any True values in the truth array
#     if truth_array.any():
#         # Find the indices where the value changes from True to False or vice versa
#         coord_array = torch.where(truth_array[:-1] != truth_array[1:])[0]

#         # If there are any changes in the truth array
#         if coord_array.numel() > 0:
#             # Shift indices by 1 to mark the start of new regions
#             coord_array = coord_array + 1

#             # Handle the case where the first value is part of an enriched region
#             if coord_array.numel() % 2 == 0:
#                 if truth_array[0]:
#                     # Create tensors on the processing device
#                     zero_tensor = torch.tensor([0], device=processing_device)
#                     len_array_tensor = torch.tensor([len(array)], device=processing_device)
#                     coord_array = torch.cat([zero_tensor, coord_array])
#                     coord_array = torch.cat([coord_array, len_array_tensor])
#             else:
#                 if truth_array[0]:
#                     zero_tensor = torch.tensor([0], device=processing_device)
#                     coord_array = torch.cat([zero_tensor, coord_array])
#                 else:
#                     len_array_tensor = torch.tensor([len(array)], device=processing_device)
#                     coord_array = torch.cat([coord_array, len_array_tensor])

#             # Reshape the coordinate array to pairs of start and end indices of enriched regions
#             coord_array = coord_array.reshape(-1, 2)

#             # Convert the coordinate array to a list of enriched regions
#             enriched_regions = coord_array.to(torch.float32)
#         else:
#             # Handle the case where the entire array is an enriched region
#             if truth_array[0]:
#                 enriched_regions = torch.tensor([[0, len(array)]], dtype=torch.float32, device=processing_device)
#             else:
#                 enriched_regions = torch.empty((0, 2), dtype=torch.float32, device=processing_device)
#     else:
#         enriched_regions = torch.empty((0, 2), dtype=torch.float32, device=processing_device)

#     # If max_region_size is not specified, set it to the length of the array
#     if max_region_size is None:
#         max_region_size = len(array)

#     # Filter regions by size - ensuring we're still on the processing device
#     filtered_regions = torch.empty((0, 2), dtype=torch.float32, device=processing_device)

#     for i in range(enriched_regions.shape[0]):
#         start = enriched_regions[i, 0]
#         end = enriched_regions[i, 1]
#         if end - start >= min_region_size and end - start <= max_region_size:
#             row = enriched_regions[i].unsqueeze(0)
#             filtered_regions = torch.cat((filtered_regions, row))

#     # Move back to the original device
#     if filtered_regions.device != original_device:
#         filtered_regions = filtered_regions.to(original_device)

#     return filtered_regions


def find_enriched_regions(
    array: torch.Tensor,
    threshold: Union[torch.Tensor, float],
    min_region_size: int = 0,
    max_region_size: Optional[int] = None,
):
    """Find regions in the array where values exceed the threshold.
    Optimized for high-performance GPU execution (CUDA/MPS).
    """
    device = array.device

    # Convert threshold to tensor if it's a scalar
    if not isinstance(threshold, torch.Tensor):
        threshold_t = torch.tensor(threshold, device=device, dtype=array.dtype)
    else:
        threshold_t = threshold

    # Vectorized boolean comparison
    if threshold_t.numel() == 1:
        truth_array = array >= threshold_t
    else:
        truth_array = (array >= threshold_t) & (array > 0)

    # If no regions found, return early
    if not truth_array.any():
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    # Find boundaries (0 -> 1 or 1 -> 0)
    # prepend/append False to handle edge cases automatically
    padded = torch.cat(
        [
            torch.tensor([False], device=device),
            truth_array,
            torch.tensor([False], device=device),
        ]
    )

    # diff() gives +1 for starts and -1 for ends
    diff = padded[1:].int() - padded[:-1].int()
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    # Combine into (N, 2)
    enriched_regions = torch.stack([starts, ends], dim=1).to(torch.float32)

    # Filter by size
    if min_region_size > 0 or max_region_size is not None:
        max_size = max_region_size if max_region_size is not None else len(array)
        region_sizes = enriched_regions[:, 1] - enriched_regions[:, 0]
        size_mask = (region_sizes >= min_region_size) & (region_sizes <= max_size)
        enriched_regions = enriched_regions[size_mask]

    return enriched_regions


def add_signal(
    peaks: Union[pr.PyRanges, pd.DataFrame], bigwig: Union[pathlib.Path, str]
) -> pr.PyRanges:
    if isinstance(bigwig, pathlib.Path):
        bigwig = str(bigwig)

    if isinstance(peaks, pr.PyRanges):
        peaks = peaks.df

    if peaks.empty:
        peaks["Signal"] = 0.0
        return pr.PyRanges(peaks)

    # Use pybigtools for faster signal extraction
    bw = pybigtools.open(bigwig)

    # Optimized extraction using pybigtools

    # Group by chromosome to minimize opening/seeking
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bed", mode="w") as tmpbed:
        # Write peaks to temp bed file
        # bed format: Chromosome, Start, End
        peaks_sorted = peaks.copy().sort_values(["Chromosome", "Start"])
        peaks_sorted["Start"] = peaks_sorted["Start"].astype(int)
        peaks_sorted["End"] = peaks_sorted["End"].astype(int)
        peaks_sorted[["Chromosome", "Start", "End"]].to_csv(
            tmpbed.name, sep="\t", index=False, header=False
        )

        # Call average_over_bed
        # Returns a generator of means
        try:
            means = list(bw.average_over_bed(tmpbed.name))
            # Create a lookup or just assign if we are sure order is same
            # Since we gave it a sorted file, it should return in that order
            peaks.loc[peaks_sorted.index, "Signal"] = means
        except Exception as e:
            logger.warning(
                f"Error using average_over_bed: {e}. Falling back to iterative extraction."
            )
            # Fallback
            for chrom, group in peaks.groupby("Chromosome"):
                for idx, (s, e) in zip(group.index, zip(group["Start"], group["End"])):
                    vals = bw.values(chrom, int(s), int(e))
                    peaks.loc[idx, "Signal"] = (
                        np.mean(vals) if vals is not None else 0.0
                    )

    if "Signal" not in peaks.columns:
        peaks["Signal"] = 0.0
    peaks["Signal"] = peaks["Signal"].fillna(0)

    return pr.PyRanges(peaks)


def extract_fixed_signal_windows(
    bigwig: Union[pathlib.Path, str],
    regions: pd.DataFrame,
    window_size: int = 200,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Extract fixed-width signal windows centered on the given regions.
    Returns a torch Tensor of shape (num_regions, window_size).
    """
    if regions.empty:
        return torch.empty((0, window_size), device=device)

    if isinstance(bigwig, pathlib.Path):
        bigwig = str(bigwig)

    bw = pybigtools.open(bigwig)
    chrom_sizes = bw.chroms()

    half_window = window_size // 2
    signals = []

    for _, row in regions.iterrows():
        chrom = row["Chromosome"]
        # Center of the region
        center = int((row["Start"] + row["End"]) // 2)
        start = center - half_window
        end = start + window_size

        # Clamp to chromosome boundaries
        chrom_len = chrom_sizes.get(chrom)
        if chrom_len is None:
            logger.warning(f"Chromosome {chrom} not found in bigwig")
            signals.append(np.zeros(window_size, dtype=np.float32))
            continue

        actual_start = max(0, start)
        actual_end = min(chrom_len, end)

        vals = bw.values(chrom, actual_start, actual_end)

        if vals is None:
            vals = np.zeros(window_size, dtype=np.float32)
        else:
            # Pad if we hit boundaries
            if len(vals) < window_size:
                pad_left = actual_start - start
                pad_right = end - actual_end
                vals = np.pad(
                    vals, (max(0, pad_left), max(0, pad_right)), mode="constant"
                )
                # Ensure exactly window_size
                vals = vals[:window_size]

        signals.append(vals)

    return torch.tensor(np.array(signals), device=device, dtype=torch.float32)
