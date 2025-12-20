"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for trajectory filtering functions

import numpy as np

from myo_tools.utils.tensor_ops.tensor_utils import (
    butterworth_filter_motion_data,
    fft_filter_motion_data,
    gaussian_filter_motion_data,
    median_filter_motion_data,
)


def test_median_filter_motion_data():
    """
    Test median_filter_motion_data() with a small dataset:
    Check that the result is still the same shape, and
    that the median filter removes outliers in a typical way.
    """
    motion_data = np.array(
        [
            [[1.0, 5.0, 1.0], [5.0, 6.0, 8.0]],
            [[2.0, 6.0, 100.0], [6.0, 7.0, 9.0]],
            [[100.0, 7.0, 3.0], [7.0, 8.0, 10.0]],
            [[4.0, 8.0, 2.0], [8.0, 9.0, 11.0]],
            [[5.0, 9.0, 1.0], [9.0, 10.0, 12.0]],
        ]
    )

    filtered = median_filter_motion_data(motion_data, size=3)
    assert filtered.shape == motion_data.shape

    # Check that outliers are actually removed
    # The first element in marker 0, dim 0 at frame 2 has outlier 100.0 that should be replaced with something closer to neighbors
    assert filtered[2, 0, 0] != 100.0
    assert filtered[2, 0, 0] < 10.0  # Should be closer to neighboring values

    # Same for the outlier in marker 0, dim 2 at frame 1
    assert filtered[1, 0, 2] != 100.0
    assert filtered[1, 0, 2] < 50.0  # Should be closer to neighboring values


def test_gaussian_filter_motion_data():
    """
    Test gaussian_filter_motion_data() with a small dataset:
    Verify that the shape is unchanged, and the data is smoothed.
    """
    motion_data = np.random.rand(10, 2, 3) * 100.0
    filtered = gaussian_filter_motion_data(motion_data, sigma=1.0)
    assert filtered.shape == motion_data.shape
    # Confirm the values have changed in a smoothing-like manner
    # by checking they're not identical (unless by chance).
    # We'll just do a rough check that there's at least some difference
    # if the original had non-trivial differences.
    if not np.allclose(
        motion_data, motion_data[0]
    ):  # ensure not all in motion_data is the same
        assert not np.allclose(
            filtered, motion_data
        ), "Gaussian filter should alter the data."


def test_butterworth_filter_motion_data():
    """
    Test butterworth_filter_motion_data() with a small dataset:
    Validate shape and confirm the filter runs without error.
    """
    # Create a test signal with known frequency components
    fs = 100.0  # sampling frequency
    t = np.linspace(0, 1, int(fs), endpoint=False)

    # Create a signal with both low (2Hz) and high (20Hz) frequencies
    low_freq = np.sin(2 * np.pi * 2 * t)
    high_freq = 0.5 * np.sin(2 * np.pi * 20 * t)
    combined = low_freq + high_freq

    # Reshape to match expected input format (time, markers, dimensions)
    motion_data = np.zeros((len(t), 1, 1))
    motion_data[:, 0, 0] = combined

    # Apply lowpass filter at 5Hz
    cutoff_freq = 5.0
    filtered = butterworth_filter_motion_data(
        motion_data,
        cutoff_freq=cutoff_freq,
        fs=fs,
        order=4,
        filter_type="low",
    )

    assert filtered.shape == motion_data.shape

    # Verify that high frequency component is attenuated
    # Calculate approximate amplitude of filtered signal
    filtered_amplitude = np.max(filtered[:, 0, 0]) - np.min(filtered[:, 0, 0])
    original_amplitude = np.max(motion_data[:, 0, 0]) - np.min(motion_data[:, 0, 0])

    # The filtered amplitude should be lower (closer to the low_freq amplitude of 2.0)
    assert filtered_amplitude < original_amplitude
    assert (
        filtered_amplitude < 2.2
    )  # Should be close to amplitude of low_freq component


def test_fft_filter_motion_data():
    """
    Test fft_filter_motion_data() with a small synthetic dataset:
    1. Confirm the function doesn't error out.
    2. Confirm the shape is preserved.
    3. Verify that specified frequency components are preserved or removed.
    """
    # Create a small synthetic signal: a sum of low and high frequency components
    fs = 100.0
    t = np.linspace(0, 1, int(fs), endpoint=False)

    # Low frequency (1 Hz) and high frequency (20 Hz) components
    low_freq_signal = np.sin(2 * np.pi * 1.0 * t)
    high_freq_signal = np.sin(2 * np.pi * 20.0 * t)
    combined_signal = low_freq_signal + high_freq_signal

    # Shape it to (frames, markers, dimensions)
    motion_data = np.zeros((len(t), 1, 2), dtype=float)
    motion_data[:, 0, 0] = low_freq_signal  # Pure 1 Hz component
    motion_data[:, 0, 1] = combined_signal  # 1 Hz + 20 Hz components

    # Apply a low-pass filter at 5 Hz (should remove 20 Hz, keep 1 Hz)
    filtered = fft_filter_motion_data(motion_data, low_cut=0.0, high_cut=5.0, fs=fs)
    assert filtered.shape == motion_data.shape

    # Channel 0 should be largely unchanged (already low frequency)
    assert np.allclose(filtered[:, 0, 0], motion_data[:, 0, 0], atol=0.1)

    # Channel 1 should be closer to the low frequency component
    high_freq_removed = filtered[:, 0, 1]
    assert (
        np.max(np.abs(high_freq_removed - low_freq_signal)) < 0.5
    )  # Should be closer to low_freq_signal
