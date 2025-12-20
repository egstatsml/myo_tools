"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for the remove_static_trackers function.

import numpy as np

from myo_tools.utils.tensor_ops.tensor_utils import remove_static_trackers


def test_remove_static_trackers_all_dynamic():
    """Test that all trackers are kept when none are static."""
    # Create motion data with all trackers moving significantly
    num_frames = 100
    num_trackers = 5
    motion_data = np.random.rand(num_frames, num_trackers, 3) * 10.0
    tracker_names = [f"marker_{i}" for i in range(num_trackers)]

    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data, tracker_names
    )

    assert filtered_data.shape == motion_data.shape
    assert len(kept_names) == num_trackers
    assert len(removed_names) == 0
    assert kept_names == tracker_names


def test_remove_static_trackers_all_static():
    """Test that all trackers are removed when all are completely static."""
    # Create motion data with all trackers completely static
    num_frames = 100
    num_trackers = 5
    motion_data = np.ones((num_frames, num_trackers, 3)) * 5.0
    tracker_names = [f"marker_{i}" for i in range(num_trackers)]

    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data, tracker_names
    )

    assert filtered_data.shape == (num_frames, 0, 3)
    assert len(kept_names) == 0
    assert len(removed_names) == num_trackers
    assert set(removed_names) == set(tracker_names)


def test_remove_static_trackers_mixed():
    """Test with a mix of static and dynamic trackers."""
    num_frames = 100
    num_trackers = 4

    # Create motion data
    motion_data = np.zeros((num_frames, num_trackers, 3))

    # Tracker 0: completely dynamic
    motion_data[:, 0, :] = np.random.rand(num_frames, 3) * 10.0

    # Tracker 1: static for 60% of frames (should be removed with default 50% threshold)
    motion_data[:60, 1, :] = 5.0  # static for first 60 frames
    motion_data[60:, 1, :] = np.random.rand(40, 3) * 10.0  # dynamic for last 40

    # Tracker 2: static for only 30% of frames (should be kept)
    motion_data[:30, 2, :] = 3.0  # static for first 30 frames
    motion_data[30:, 2, :] = np.random.rand(70, 3) * 10.0  # dynamic for last 70

    # Tracker 3: completely static
    motion_data[:, 3, :] = 2.0

    tracker_names = [f"marker_{i}" for i in range(num_trackers)]

    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data, tracker_names, max_allowed_static_frames_fraction=0.5
    )

    # Should keep trackers 0 and 2, remove trackers 1 and 3
    assert filtered_data.shape == (num_frames, 2, 3)
    assert len(kept_names) == 2
    assert len(removed_names) == 2
    assert "marker_0" in kept_names
    assert "marker_2" in kept_names
    assert "marker_1" in removed_names
    assert "marker_3" in removed_names


def test_remove_static_trackers_boundary_case():
    """Test tracker that is static for exactly the threshold ratio."""
    num_frames = 100
    num_trackers = 2

    motion_data = np.zeros((num_frames, num_trackers, 3))

    # Tracker 0: static for exactly 50% of frames
    motion_data[:50, 0, :] = 1.0
    motion_data[50:, 0, :] = np.random.rand(50, 3) * 10.0

    # Tracker 1: dynamic
    motion_data[:, 1, :] = np.random.rand(num_frames, 3) * 10.0

    tracker_names = ["boundary_marker", "dynamic_marker"]

    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data, tracker_names, max_allowed_static_frames_fraction=0.5
    )

    # With <= condition, exactly 50% static should be kept
    assert "boundary_marker" in kept_names
    assert len(removed_names) == 0


def test_remove_static_trackers_custom_threshold():
    """Test with custom static threshold and max_allowed_static_frames_fraction."""
    num_frames = 100
    num_trackers = 2

    motion_data = np.zeros((num_frames, num_trackers, 3))

    # Tracker 0: small movements below default threshold but above custom threshold
    motion_data[:, 0, :] = np.arange(num_frames).reshape(-1, 1) * 1e-7

    # Tracker 1: larger movements
    motion_data[:, 1, :] = np.arange(num_frames).reshape(-1, 1) * 1e-5

    tracker_names = ["tiny_movement", "small_movement"]

    # With default threshold (1e-6), tracker 0 should be considered static
    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data,
        tracker_names,
        static_threshold=1e-6,
        max_allowed_static_frames_fraction=0.5,
    )

    assert "tiny_movement" in removed_names
    assert "small_movement" in kept_names


def test_remove_static_trackers_empty_input():
    """Test with empty motion data."""
    motion_data = np.zeros((100, 0, 3))
    tracker_names = []

    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data, tracker_names
    )

    assert filtered_data.shape == (100, 0, 3)
    assert len(kept_names) == 0
    assert len(removed_names) == 0


def test_remove_static_trackers_single_frame():
    """Test with only one frame (no diffs to compute)."""
    num_trackers = 3
    motion_data = np.random.rand(1, num_trackers, 3)
    tracker_names = [f"marker_{i}" for i in range(num_trackers)]

    # With only 1 frame, diff will be empty, so all trackers should be kept
    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data, tracker_names
    )

    assert filtered_data.shape == motion_data.shape
    assert len(kept_names) == num_trackers
    assert len(removed_names) == 0


def test_remove_static_trackers_preserves_order():
    """Test that the order of kept trackers is preserved."""
    num_frames = 100
    motion_data = np.zeros((num_frames, 5, 3))

    # Make trackers 0, 2, 4 dynamic and trackers 1, 3 static
    for i in [0, 2, 4]:
        motion_data[:, i, :] = np.random.rand(num_frames, 3) * 10.0
    for i in [1, 3]:
        motion_data[:, i, :] = 1.0

    tracker_names = ["marker_0", "marker_1", "marker_2", "marker_3", "marker_4"]

    filtered_data, kept_names, removed_names = remove_static_trackers(
        motion_data, tracker_names
    )

    assert kept_names == ["marker_0", "marker_2", "marker_4"]
    assert removed_names == ["marker_1", "marker_3"]
