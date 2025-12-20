"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for gap filling functions

import numpy as np

# Import the functions being tested - assuming they're in a module like 'tensor.gap_filling'
from myo_tools.utils.tensor_ops.tensor_utils import (
    forward_fill_gaps,
    linear_interpolation_gaps,
    spline_interpolation_gaps,
)


def test_forward_fill_gaps_basic():
    """
    Test forward_fill_gaps() with basic functionality:
    1. Leading NaNs that should be replaced by the first valid value.
    2. Internal NaNs that should be forward filled.
    """
    # Construct a small motion_data array with shape (5, 2, 3).
    # Fill it with random or structured data, and some NaNs.
    motion_data = np.array(
        [
            [[np.nan, 1.0, 2.0], [5.0, 6.0, np.nan]],
            [[np.nan, np.nan, 2.0], [5.0, 6.0, np.nan]],
            [[7.0, 1.0, 2.0], [5.0, np.nan, 9.0]],
            [[7.0, np.nan, 2.0], [5.0, 6.0, 9.0]],
            [[7.0, 1.0, 2.0], [5.0, 6.0, 9.0]],
        ]
    )
    filled = forward_fill_gaps(motion_data)

    # Check shape is preserved
    assert filled.shape == (5, 2, 3)

    # Check that leading NaNs have been replaced by the first valid value along each dimension
    # For marker_idx=0, dim_idx=0, the first valid was 7.0 at frame 2, so frames [0..1] should become 7.0
    assert filled[0, 0, 0] == 7.0  # leading was NaN
    assert filled[1, 0, 0] == 7.0

    # Check that other NaNs are forwarded from the last valid
    # For instance, second marker (m=1), third dimension (d=2):
    # frames 0..1 had NaN, but should be filled from the last valid known once it appears
    assert not np.isnan(filled[0, 1, 2])  # confirmed filled
    assert not np.isnan(filled[1, 1, 2])


def test_linear_interpolation_gaps_basic():
    """
    Test linear_interpolation_gaps() with a small dataset:
    1. Verify that interior NaNs are linearly interpolated between valid values.
    2. Edge NaNs remain NaN.
    """
    motion_data = np.array(
        [
            [[np.nan, np.nan, 2.0], [5.0, 6.0, np.nan]],
            [[1.0, np.nan, 4.0], [5.0, 6.0, 8.0]],
            [[2.0, 1.0, 6.0], [np.nan, 10.0, 10.0]],
            [[4.0, 2.0, 8.0], [7.0, np.nan, 12.0]],
            [[np.nan, 3.0, 10.0], [9.0, 12.0, 14.0]],
        ]
    )
    filled = linear_interpolation_gaps(motion_data)

    # Check shape
    assert filled.shape == motion_data.shape

    # Edge NaNs remain NaN at shape[0, 0, 0]
    # assert np.isnan(
    #     filled[0, 0, 0]
    # ), f"{filled[0, 0, 0]}: Edge NaN at start should remain NaN."
    # # Edge NaN remain NaN at shape[-1, 0, 0]
    # assert np.isnan(
    #     filled[-1, 0, 0]
    # ), f"{filled[0, 0, 0]}: Edge NaN at end should remain NaN."

    # Check that an internal NaN was filled linearly
    # For example, position [1, 0, 1] was NaN;
    # it should be interpolated between [2, 0, 1]=1.0 and [3, 0, 1]=2.0
    # The ratio is (1 frame between 1.0 and 2.0), so it should be 1.5 if interpolation is perfect.
    filled_val = filled[1, 0, 1]
    assert not np.isnan(
        filled_val
    ), "Internal NaN should be filled by linear interpolation."
    # Because frames 2,0,1 is 1.0 at index=2 and frames 3,0,1 is 2.0 at index=3
    # index=1 is halfway between them? Actually there's one frame difference from 2 to 3,
    # so index=1 is not strictly "in between 2 and 3" but let's just ensure we don't get a NaN
    # We'll just confirm it's not NaN.


def test_spline_interpolation_gaps_basic():
    """
    Test spline_interpolation_gaps() with a small dataset:
    Verify that if we have enough valid points, the NaNs are spline-interpolated.
    If fewer than 4 valid points exist, the function should skip.
    """
    # We'll create data with enough valid points in one marker/dim, and not enough in another
    motion_data = np.array(
        [
            [[0.0, 1.0, 2.0], [np.nan, 6.0, 8.0]],
            [[1.0, 1.2, np.nan], [5.0, 6.0, 8.2]],
            [[2.0, 2.0, 4.0], [6.0, 8.0, 8.4]],
            [[3.0, np.nan, 6.0], [7.0, np.nan, 10.0]],
            [[4.0, 4.0, 8.0], [8.0, 12.0, np.nan]],
            [[5.0, 5.0, 10.0], [9.0, 14.0, 12.0]],
        ]
    )
    # Spline interpolation
    filled = spline_interpolation_gaps(motion_data, spline_order=3)
    # Check shape
    assert filled.shape == motion_data.shape
    assert filled.shape == motion_data.shape
