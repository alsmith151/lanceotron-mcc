

# def _average_array(array: np.ndarray, window=None):
#     # If the window size is equal to the length of the array, return an array of the same length with the mean value of the input array
#     if window == len(array):
#         return np.ones_like(array) * np.mean(array)

#     # Calculate the cumulative sum of the array
#     cumsum_array = np.cumsum(array)
#     # Adjust the cumulative sum to get the moving sum for the window size
#     cumsum_array[window:] = cumsum_array[window:] - cumsum_array[:-window]
#     # Extract the moving average values
#     return_array = cumsum_array[window - 1:] / window

#     # Calculate the padding needed for the left and right ends of the array
#     left_pad = window // 2
#     right_pad = window - left_pad - 1

#     # Pad the array to match the original array size
#     # np.pad pads arrays using the 'edge' mode which replicates the edge values
#     padded_array = np.pad(return_array, (left_pad, right_pad), mode='edge')

#     return padded_array


# def average_array(array: Union[np.ndarray, scipy.sparse.csr_array], 
#                   window: Optional[int] = None,
#                   is_sparse: bool = True) -> Union[np.ndarray, scipy.sparse.csr_array]:
#     # If no window size is provided, set the window size to the length of the array
#     if window is None:
#         window = array.shape[0]

#     # If the window size is greater than the length of the array, adjust the window size
#     if window > array.shape[0]:
#         window = array.shape[0]
    
#     if isinstance(array, np.ndarray):
#         array = _average_array(array, window)
#     elif isinstance(array, scipy.sparse.csr_array):
#         array = _average_array(array.todense(), window)
#     else:
#         raise ValueError("Input array must be a numpy array or a sparse array.")
    

#     if is_sparse:
#         array = scipy.sparse.csr_array(array.reshape(-1, 1))

#     return array


# def _filter_enriched_regions_sparse(labeled_array: scipy.sparse.csr_matrix, 
#                                     min_region_size: int, 
#                                     max_region_size: int) -> scipy.sparse.csr_matrix:
#     # Extract the row and column indices of nonzero elements
#     rows, cols = labeled_array.nonzero()

#     # We need to find contiguous regions in the sparse array, similar to labeling connected components
#     enriched_regions = []
#     current_start = cols[0]
    
#     for i in range(1, len(cols)):
#         # If there's a gap (i.e., not contiguous), mark the end of the current region
#         if cols[i] != cols[i - 1] + 1:
#             current_end = cols[i - 1] + 1
#             region_size = current_end - current_start

#             # Filter by size
#             if min_region_size <= region_size <= max_region_size:
#                 enriched_regions.append([current_start, current_end])
            
#             # Start a new region
#             current_start = cols[i]

#     # Check the last region
#     current_end = cols[-1] + 1
#     region_size = current_end - current_start
#     if min_region_size <= region_size <= max_region_size:
#         enriched_regions.append([current_start, current_end])
    
#     # Convert the result into a sparse matrix
#     enriched_regions_array = np.array(enriched_regions)
#     if enriched_regions_array.size == 0:
#         return scipy.sparse.csr_matrix((0, 2))  # Return an empty sparse matrix
#     return scipy.sparse.csr_matrix(enriched_regions_array)

# def find_enriched_regions(array: Union[np.ndarray, scipy.sparse.csr_matrix], 
#                           threshold: Union[np.ndarray, scipy.sparse.csr_matrix], 
#                           min_region_size: int = 0, 
#                           max_region_size: int = np.inf) -> scipy.sparse.csr_matrix:
    
#     # Ensure inputs are sparse matrices
#     if not scipy.sparse.issparse(array):
#         array = scipy.sparse.csr_matrix(array)
#     if not scipy.sparse.issparse(threshold):
#         threshold = scipy.sparse.csr_matrix(threshold)
        
#     # Apply the threshold condition (element-wise comparison in sparse matrices)
#     truth_array = (array > 0).multiply(array >= threshold)

#     # Find nonzero elements in the sparse matrix (this effectively acts as "labeling" contiguous regions)
#     labeled_array = scipy.sparse.csr_matrix(truth_array, dtype=int)

#     return _filter_enriched_regions_sparse(labeled_array, min_region_size, max_region_size)





# def _filter_enriched_regions(labeled_array, min_region_size, max_region_size) -> np.ndarray:
#     # Find the start and end indices of each labeled region using vectorized operations
#     region_starts = np.flatnonzero(np.diff(np.r_[0, labeled_array, 0]) > 0)
#     region_ends = np.flatnonzero(np.diff(np.r_[0, labeled_array, 0]) < 0)

#     # Calculate sizes of each region
#     region_sizes = region_ends - region_starts

#     # Filter regions based on min and max size
#     valid_regions = (region_sizes >= min_region_size) & (region_sizes <= max_region_size)

#     # Extract only valid regions and return as list of start-end pairs
#     enriched_region_list = np.column_stack((region_starts[valid_regions], region_ends[valid_regions]))

#     return enriched_region_list


# def find_enriched_regions(array: Union[np.ndarray, scipy.sparse.csr_array], 
#                           threshold: Union[np.ndarray, scipy.sparse.csr_array], 
#                           min_region_size: int = 0, 
#                           max_region_size: int = np.inf):
    
#     if isinstance(array, scipy.sparse.csr_array):
#         array = array.toarray().flatten()
#     if isinstance(threshold, scipy.sparse.csr_array):
#         threshold = threshold.toarray().flatten()
        
#     # Create a boolean array where values are True if they meet the threshold condition
#     truth_array = (array > 0) & (array >= threshold)

#     # Identify connected regions using scipy.ndimage.label
#     labeled_array, _ = scipy.ndimage.label(truth_array)
#     return _filter_enriched_regions(labeled_array, min_region_size, max_region_size)    
    
