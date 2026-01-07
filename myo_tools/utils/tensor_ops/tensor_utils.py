"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: This file implements various utility functions for tensor operations, including flattening, unflattening, padding, stacking, concatenating, splitting, and truncating tensors and dictionaries of tensors.


from typing import Iterable

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, filtfilt


def flatten_tensors(tensors):
    """
    Concatenate a list of tensors into a single 1D numpy array.

    Args:
        tensors (List[np.ndarray]): A list of numpy arrays (of possibly various shapes).

    Returns:
        np.ndarray: A 1D array containing the concatenated contents of `tensors`.
    """
    if len(tensors) > 0:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    else:
        return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    """
    Unflatten a 1D array back into a list of tensors with specified shapes.

    Args:
        flattened (np.ndarray): A 1D array of flattened tensor data.
        tensor_shapes (List[tuple]): A list of shapes corresponding to the target tensors.

    Returns:
        List[np.ndarray]: A list of numpy arrays reshaped to their corresponding shapes.
    """
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [
        np.reshape(pair[0], pair[1])
        for pair in zip(np.split(flattened, indices), tensor_shapes)
    ]


def pad_tensor(x, max_len, mode="zero"):
    """
    Pad a 1D or higher-dimensional sequence of tensors to a given max length.

    If mode is "zero", it will pad with zeros;
    if mode is "last", it will pad with the last element.

    Args:
        x (np.ndarray): A sequence of shape (N, ...) where N <= max_len.
        max_len (int): The desired length after padding.
        mode (str): The padding mode, either "zero" or "last".

    Returns:
        np.ndarray: A padded array with shape (max_len, ...).
    """
    padding = np.zeros_like(x[0])
    if mode == "last":
        padding = x[-1]
    return np.concatenate(
        [x, np.tile(padding, (max_len - len(x),) + (1,) * np.ndim(x[0]))]
    )


def pad_tensor_n(xs, max_len):
    """
    Pad each array in a list of arrays to `max_len` along the first dimension.

    Args:
        xs (List[np.ndarray]): A list of arrays, each shaped (N, ...) where N <= max_len.
        max_len (int): The desired first-dimension length for all arrays.

    Returns:
        np.ndarray: A single array of shape (len(xs), max_len, ...),
                    containing each padded array.
    """
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][: len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict, max_len, mode="zero"):
    """
    Recursively pad arrays within a dictionary to `max_len`.
    Uses pad_tensor for individual arrays.

    Args:
        tensor_dict (dict): A dictionary containing arrays or nested dictionaries of arrays.
        max_len (int): The desired first-dimension length for all arrays.
        mode (str): Padding mode, either "zero" or "last".

    Returns:
        dict: A new dictionary with padded arrays (and dictionaries).
    """
    keys = list(tensor_dict.keys())
    ret = {}
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len, mode=mode)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len, mode=mode)
    return ret


def flatten_first_axis_tensor_dict(tensor_dict):
    """
    Flatten the first two axes of each array in a dictionary of arrays.

    If the array shape is (B, T, ...), it becomes (B*T, ...).
    Recursively applies to nested dictionaries.

    Args:
        tensor_dict (dict): Dictionary of arrays (or nested dictionaries of arrays).

    Returns:
        dict: Dictionary of arrays with flattened axes.
    """
    keys = list(tensor_dict.keys())
    ret = {}
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = flatten_first_axis_tensor_dict(tensor_dict[k])
        else:
            old_shape = tensor_dict[k].shape
            ret[k] = tensor_dict[k].reshape((-1,) + old_shape[2:])
    return ret


def high_res_normalize(probs):
    """
    Normalize a list of probabilities so that they sum to 1.
    If the sum is non-positive, returns all zeros.

    Args:
        probs (List[float]): A list of probability values.

    Returns:
        List[float]: A normalized list of probabilities summing to 1, or zeros if invalid.
    """
    total = sum(map(float, probs))
    if total <= 0:  # Handle case where total is zero or negative
        return [0] * len(probs)
    return [x / total for x in map(float, probs)]


def stack_tensor_list(tensor_list):
    """
    Stack a list of tensors (with the same shape) along a new dimension.

    Args:
        tensor_list (List[np.ndarray]): A list of numpy arrays all with the same shape.

    Returns:
        np.ndarray: A stacked array with one extra dimension at the front.
    """
    return np.array(tensor_list)


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.

    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    if not tensor_dict_list:  # Check for empty list
        return {}
    keys = list(tensor_dict_list[0].keys())
    ret = {}
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def concat_tensor_list_subsample(tensor_list, f):
    """
    Randomly subsample each tensor in a list by fraction f, then concatenate along axis 0.

    Args:
        tensor_list (List[np.ndarray]): A list of arrays to be subsampled and concatenated.
        f (float): A fraction in [0,1] indicating the proportion of each array to take.

    Returns:
        np.ndarray: A concatenated array containing the subsampled elements.
    """
    return np.concatenate(
        [
            t[np.random.choice(len(t), int(np.ceil(len(t) * f)), replace=False)]
            for t in tensor_list
        ],
        axis=0,
    )


def concat_tensor_dict_list_subsample(tensor_dict_list, f):
    """
    Recursively subsample each tensor (by fraction f) in a list of dictionaries,
    then concatenate along axis 0.

    Args:
        tensor_dict_list (List[dict]): List of dictionaries containing arrays or nested dictionaries.
        f (float): A fraction in [0,1] indicating the proportion of each array to take.

    Returns:
        dict: A dictionary of concatenated arrays or dictionaries of concatenated arrays.
    """
    keys = list(tensor_dict_list[0].keys())
    ret = {}
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list_subsample([x[k] for x in tensor_dict_list], f)
        else:
            v = concat_tensor_list_subsample([x[k] for x in tensor_dict_list], f)
        ret[k] = v
    return ret


def concat_tensor_list(tensor_list):
    """
    Concatenate a list of tensors along axis 0.
    Returns an empty array if the list is empty.

    Args:
        tensor_list (List[np.ndarray]): A list of arrays.

    Returns:
        np.ndarray: A concatenated array, or an empty array if tensor_list is empty.
    """
    if not tensor_list:  # Check for empty list
        return np.array([])  # Return an empty array
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    """
    Recursively concatenate arrays in a list of dictionaries along axis 0.

    Args:
        tensor_dict_list (List[dict]): List of dictionaries containing arrays or nested dictionaries.

    Returns:
        dict: A dictionary with concatenated arrays or dictionaries of concatenated arrays.
    """
    if not tensor_dict_list:  # Check for empty list
        return {}
    keys = list(tensor_dict_list[0].keys())
    ret = {}
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def split_tensor_dict_list(tensor_dict):
    """
    Split each array in a dictionary along the first dimension, returning a list of dictionaries.

    E.g., if an array in the dictionary is shape (B, ...), we split into B elements,
    each placed into its own dictionary, matching keys to slices.

    Args:
        tensor_dict (dict): Dictionary with arrays or nested dictionaries of arrays.

    Returns:
        List[dict]: A list of dictionaries, each representing one slice along the first dimension.
    """
    if not tensor_dict:  # Check for empty dictionary
        return []
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def truncate_tensor_list(tensor_list, truncated_len):
    """
    Truncate a list of tensors to length `truncated_len`.

    Args:
        tensor_list (np.ndarray): A sequence of tensors or data of shape (N, ...).
        truncated_len (int): Desired length after truncation.

    Returns:
        np.ndarray: Truncated array of shape (truncated_len, ...).
    """
    return tensor_list[:truncated_len]


def truncate_tensor_dict(tensor_dict, truncated_len):
    """
    Recursively truncate arrays in a dictionary to length `truncated_len`.

    Args:
        tensor_dict (dict): Dictionary containing arrays or nested dictionaries of arrays.
        truncated_len (int): Desired length after truncation.

    Returns:
        dict: A new dictionary with truncated arrays.
    """
    ret = {}
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            ret[k] = truncate_tensor_dict(v, truncated_len)
        else:
            ret[k] = truncate_tensor_list(v, truncated_len)
    return ret


def is_int_or_iterable_of_ints(x: int | Iterable[int]):
    """
    Checks if input is an integer or an iterable of integers.
    Supports Python int, numpy int, and jax.numpy int of various precisions.

    Args:
    - x (int | Iterable[int]): Input to check.

    Returns:
    - bool: True if x is an int or iterable of ints, False otherwise.
    """
    if not isinstance(x, Iterable):
        return isinstance(x, (int, np.integer))

    if not hasattr(x, "dtype"):
        return all(isinstance(i, (int, np.integer)) for i in x)
    else:
        # need subdtype check as jax.array indexing returns another array
        return np.issubdtype(x.dtype, np.integer)


def forward_fill_gaps(motion_data):
    """
    Fill missing values in `motion_data` by:
      - If the first value(s) is NaN, fill them by looking ahead to the first non-NaN value.
      - Then do a standard forward fill for remaining NaNs.

    motion_data shape: (num_frames, num_markers, 3) --> (F, M, 3)

    Returns:
        filled_data (np.ndarray): Filled copy of `motion_data`.
    """
    filled_data = motion_data.copy()
    num_frames, num_markers, num_dims = filled_data.shape

    for marker_idx in range(num_markers):
        for dim_idx in range(num_dims):
            # Extract the 1D signal for this marker/dimension pair
            signal_1d = filled_data[:, marker_idx, dim_idx]

            # 1. If the first value is NaN, find the first valid value and propagate it backward
            if np.isnan(signal_1d[0]):
                valid_indices = np.where(~np.isnan(signal_1d))[0]
                if len(valid_indices) > 0:
                    first_valid_idx = valid_indices[0]
                    signal_1d[:first_valid_idx] = signal_1d[first_valid_idx]

            # 2. Forward fill
            for f in range(1, num_frames):
                if np.isnan(signal_1d[f]):
                    signal_1d[f] = signal_1d[f - 1]

            # Put the filled series back
            filled_data[:, marker_idx, dim_idx] = signal_1d

    return filled_data


def linear_interpolation_gaps(motion_data):
    """
    Perform linear interpolation to fill missing (NaN) values in motion_data.

    For each marker & dimension, NaNs are linearly interpolated between valid points.
    If NaNs exist at the edges (start or end), they remain NaN.

    motion_data shape: (num_frames, num_markers, 3)

    Returns:
        filled_data (np.ndarray): The data with any internal NaNs linearly interpolated.
    """
    filled_data = motion_data.copy()
    num_frames, num_markers, num_dims = filled_data.shape

    assert num_dims == 3, "Only 3D data is supported for linear interpolation."

    frame_indices = np.arange(num_frames)

    for marker_idx in range(num_markers):
        for dim_idx in range(num_dims):
            signal_1d = filled_data[:, marker_idx, dim_idx]

            # Identify valid indices and values
            valid_mask = ~np.isnan(signal_1d)
            if np.sum(valid_mask) < 2:
                # Fewer than 2 valid points; skip or apply alternative logic
                continue

            valid_x = frame_indices[valid_mask]
            valid_y = signal_1d[valid_mask]

            # Interpolator
            f_linear = interp1d(
                valid_x,
                valid_y,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Fill missing frames
            missing_x = frame_indices[np.isnan(signal_1d)]
            interp_values = f_linear(missing_x)
            filled_data[missing_x, marker_idx, dim_idx] = interp_values

    return filled_data


def spline_interpolation_gaps(motion_data, spline_order=3):
    """
    Use a spline (e.g., cubic) interpolation to fill missing (NaN) values.

    motion_data shape: (num_frames, num_markers, 3)
    spline_order (int): typically 1 = linear, 2 = quadratic, 3 = cubic, etc.

    Any NaNs at the edges (where there's no bounding valid data) will be extrapolated.

    Returns:
        filled_data (np.ndarray): Data with NaNs replaced by spline interpolation.
    """
    filled_data = motion_data.copy()
    num_frames, num_markers, num_dims = filled_data.shape

    frame_indices = np.arange(num_frames, dtype=float)

    for marker_idx in range(num_markers):
        for dim_idx in range(num_dims):
            signal_1d = filled_data[:, marker_idx, dim_idx]

            valid_mask = ~np.isnan(signal_1d)
            if np.sum(valid_mask) < 4:
                # Too few points for a spline of order >= 3
                continue

            valid_x = frame_indices[valid_mask]
            valid_y = signal_1d[valid_mask]

            # Fit spline
            try:
                spline = UnivariateSpline(valid_x, valid_y, k=spline_order, s=0)
            except Exception:
                # If spline can't fit, skip
                continue

            # Interpolate missing
            missing_x = frame_indices[np.isnan(signal_1d)]
            if len(missing_x) > 0:
                spline_values = spline(missing_x)
                filled_data[missing_x.astype(int), marker_idx, dim_idx] = spline_values

    return filled_data


def median_filter_motion_data(motion_data, size=3):
    """
    Apply a 1D median filter along the time axis (frames) to motion_data.

    This function treats each marker and dimension as an independent 1D signal
    and applies a median filter using the given window size.

    Args:
        motion_data (np.ndarray): A 3D array of shape (num_frames, num_markers, 3).
                                  motion_data[f, m, d] is the d-th coordinate of the
                                  m-th marker at frame f.
        size (int): Window size of the median filter along the time dimension.
                    Must be a positive odd integer (e.g., 3, 5, 7...). Defaults to 3.

    Returns:
        np.ndarray: A filtered copy of motion_data with the same shape as the input,
                    after applying the median filter.
    """

    filtered_data = motion_data.copy()
    num_frames, num_markers, num_dims = filtered_data.shape

    for m in range(num_markers):
        for d in range(num_dims):
            signal_1d = filtered_data[:, m, d]
            # Apply median filter on 1D signal (time axis)
            filtered_data[:, m, d] = median_filter(signal_1d, size=size)

    return filtered_data


def gaussian_filter_motion_data(motion_data, sigma=1.0):
    """
    Apply a 1D Gaussian filter along the time axis (frames) to motion_data.

    This function treats each marker and dimension as an independent 1D signal
    and applies a Gaussian filter with the specified standard deviation (sigma).

    Args:
        motion_data (np.ndarray): A 3D array of shape (num_frames, num_markers, 3).
        sigma (float): Standard deviation for the Gaussian kernel. A larger sigma
                       smooths more heavily. Defaults to 1.0.

    Returns:
        np.ndarray: A filtered copy of motion_data with the same shape as the input,
                    after applying the Gaussian filter.
    """

    filtered_data = motion_data.copy()
    num_frames, num_markers, num_dims = filtered_data.shape

    for m in range(num_markers):
        for d in range(num_dims):
            signal_1d = filtered_data[:, m, d]
            # Apply Gaussian filter on 1D signal (time axis)
            filtered_data[:, m, d] = gaussian_filter1d(signal_1d, sigma=sigma)

    return filtered_data


def butterworth_filter_motion_data(
    motion_data, cutoff_freq, fs, order=4, filter_type="low"
):
    """
    Apply a Butterworth filter along the time axis (frames) to motion_data.

    This function treats each marker and dimension as an independent 1D signal
    and applies a digital Butterworth filter of the specified type (lowpass,
    highpass, bandpass, etc.).

    Args:
        motion_data (np.ndarray): A 3D array of shape (num_frames, num_markers, 3).
        cutoff_freq (float or tuple): Cutoff frequency (or frequencies for bandpass/bandstop)
                                      in Hz. For example:
                                      - If filter_type="low" or "high", provide a float.
                                      - If filter_type="bandpass" or "bandstop", provide a tuple.
        fs (float): Sampling rate in Hz (frames per second).
        order (int): Order of the Butterworth filter. Defaults to 4.
        filter_type (str): One of "low", "high", "bandpass", "bandstop". Defaults to "low".

    Returns:
        np.ndarray: A filtered copy of motion_data with the same shape as the input,
                    after applying the Butterworth filter.
    """

    filtered_data = motion_data.copy()
    num_frames, num_markers, num_dims = filtered_data.shape

    # Normalize cutoff frequency to the Nyquist frequency (fs/2)
    if isinstance(cutoff_freq, tuple):
        # multiple frequencies for bandpass/bandstop
        wn = [freq / (0.5 * fs) for freq in cutoff_freq]
    else:
        # single frequency for low/high
        wn = cutoff_freq / (0.5 * fs)

    b, a = butter(order, wn, btype=filter_type, analog=False)

    for m in range(num_markers):
        for d in range(num_dims):
            signal_1d = filtered_data[:, m, d]
            # filtfilt applies a forward/backward filter to reduce phase distortion
            filtered_data[:, m, d] = filtfilt(b, a, signal_1d)

    return filtered_data


def fft_filter_motion_data(motion_data, low_cut=0.0, high_cut=None, fs=120.0):
    """
    Apply a frequency-domain (FFT) filter along the time axis (frames) to motion_data.
    This function treats each marker and dimension as an independent 1D signal.

    The default behavior is a low-pass filter if only high_cut < fs/2 is specified
    (with low_cut=0).  If both low_cut > 0 and high_cut < fs/2 are given, it behaves
    like a bandpass filter.  For a simple high-pass filter, you can set low_cut > 0
    and high_cut=None or a large value.

    Steps:
      1. For each 1D signal, compute the FFT via rfft.
      2. Zero out frequencies outside the [low_cut, high_cut] range.
      3. Inverse transform (irfft) to recover the time-domain signal.

    Args:
        motion_data (np.ndarray): A 3D array of shape (num_frames, num_markers, 3).
                                  motion_data[f, m, d] is the d-th coordinate of the
                                  m-th marker at frame f.
        low_cut (float): Lowest frequency (in Hz) to preserve. Default = 0.0 (DC).
        high_cut (float or None): Highest frequency (in Hz) to preserve. If None,
                                  defaults to fs/2 (Nyquist). Defaults to None.
        fs (float): Sampling rate in Hz (frames per second). Defaults to 120.0.

    Returns:
        np.ndarray: A filtered copy of motion_data (same shape),
                    with frequencies outside [low_cut, high_cut] removed.
    """
    import numpy as np

    # If no high_cut provided, default to Nyquist frequency
    if high_cut is None or high_cut > fs / 2:
        high_cut = fs / 2

    filtered_data = motion_data.copy()
    num_frames, num_markers, num_dims = filtered_data.shape

    # Frame-by-frame spacing in seconds
    # (Not strictly needed if we use np.fft.rfftfreq directly below.)
    frame_time = 1.0 / fs

    for m in range(num_markers):
        for d in range(num_dims):
            # 1D signal over frames
            signal_1d = filtered_data[:, m, d]

            # FFT
            freq_data = np.fft.rfft(signal_1d)
            freqs = np.fft.rfftfreq(num_frames, d=frame_time)

            # Determine which frequencies to keep
            keep_mask = (freqs >= low_cut) & (freqs <= high_cut)
            freq_data[~keep_mask] = 0.0  # zero out frequencies outside the passband

            # iFFT
            # Note: we specify n=num_frames to ensure it is the original length,
            # especially if num_frames is not a power of two.
            filtered_data[:, m, d] = np.fft.irfft(freq_data, n=num_frames)

    return filtered_data


def remove_static_trackers(
    motion_data: np.ndarray,
    tracker_names: list[str],
    static_threshold: float = 1e-6,
    max_allowed_static_frames_fraction: float = 0.5,
):
    """
    Remove trackers (markers) that are static for more than a specified percentage of frames.

    A tracker is considered static in a frame if the displacement from the previous frame
    is below the static_threshold across all dimensions. If the number of static frames
    exceeds max_allowed_static_frames_fraction * num_frames, the tracker is removed.

    Args:
        motion_data (np.ndarray): Motion capture data of shape (num_frames, num_trackers, 3).
        tracker_names (list[str]): List of tracker names corresponding to the trackers in motion_data.
        static_threshold (float): Threshold below which movement is considered static (in meters).
            Default is 1e-6.
        max_allowed_static_frames_fraction (float): Maximum ratio of static frames allowed (0.0 to 1.0).
            If a tracker is static for more than this ratio of frames, it will be removed.
            Default is 0.5 (50%).

    Returns:
        tuple: A tuple containing:
            - filtered_motion_data (np.ndarray): Filtered motion data with shape
              (num_frames, num_kept_trackers, 3).
            - filtered_tracker_names (list[str]): List of tracker names that were kept.
            - removed_tracker_names (list[str]): List of tracker names that were removed.

    Example:
        >>> motion_data = np.random.rand(100, 10, 3)
        >>> tracker_names = [f"marker_{i}" for i in range(10)]
        >>> filtered_data, kept_names, removed_names = remove_static_trackers(
        ...     motion_data, tracker_names, static_threshold=1e-6, max_allowed_static_frames_fraction=0.5
        ... )
    """
    num_frames = motion_data.shape[0]
    idxs_trackers_to_keep = []
    removed_tracker_names = []

    for it in range(motion_data.shape[1]):
        # Calculate displacement between consecutive frames
        diffs = np.linalg.norm(np.diff(motion_data[:, it, :], axis=0), axis=1)
        num_static_frames = np.sum(diffs < static_threshold)

        if num_static_frames <= num_frames * max_allowed_static_frames_fraction:
            idxs_trackers_to_keep.append(it)
        else:
            removed_tracker_names.append(tracker_names[it])

    # Filter motion data and tracker names
    filtered_motion_data = motion_data[:, idxs_trackers_to_keep, :]
    filtered_tracker_names = [tracker_names[it] for it in idxs_trackers_to_keep]

    return filtered_motion_data, filtered_tracker_names, removed_tracker_names
