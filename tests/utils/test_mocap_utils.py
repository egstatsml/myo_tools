"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for mocap_utils

from unittest.mock import patch

import numpy as np
import pytest

from myo_tools.mjs.marker.marker_api import get_marker_names
from myo_tools.utils.mocap_ops.mocap_utils import (
    load_trackers,
    load_trackers_and_markerset,
)

# Mock data for testing
# Note: mock_motion_data will be divided by mocap_scale=1000 in load_trackers_and_markerset
# So we multiply by 1000 here to get the expected final values
np.random.seed(42)  # For reproducible tests
mock_motion_data_raw = np.random.rand(100, 15, 4)
mock_motion_data = (
    mock_motion_data_raw * 1000
)  # Will be divided by 1000 in the function
mock_marker_names = [
    "marker1",
    "marker2",
    "marker3",
    "marker4",
    "marker5",
    "marker6",
    "marker7",
    "marker8",
    "marker9",
    "marker10",
    "marker_11",
    "marker-12",
    "Cluster_SubjName_marker_13",
    "Cluster_SubjName_marker14",
    "SubjName:marker15",
]
mock_framerate = 100.0


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(mock_motion_data[:, :, :3].copy(), mock_marker_names, mock_framerate),
)
def test_load_trackers_and_markerset(mock_from_c3d_to_numpy):
    # Create a mock markerset
    markerset_xml = """
    <markerset>
        <marker name="marker1"/>
        <marker name="marker3"/>
        <marker name="marker10"/>
        <marker name="marker_11"/>
        <marker name="marker-12"/>
        <marker name="marker_13"/>
        <marker name="marker14"/>
        <marker name="marker15"/>
    </markerset>
    """

    motion_data_subject_list, markerset, framerate = load_trackers_and_markerset(
        trackers_file_path="dummy_path.c3d", markerset_handle=markerset_xml
    )
    marker_names = get_marker_names(markerset)
    motion_data = motion_data_subject_list[0][0]  # first subject, first chunk

    # Check the shape of the filtered motion data
    assert motion_data.shape == (100, 8, 3)
    assert len(marker_names) == motion_data.shape[1]  # check marker_names filtering
    assert len(marker_names) == len(set(marker_names))  # ensure no duplicates
    assert framerate == mock_framerate

    # Check that mocap scaling was applied (values should be 1000x smaller than input)
    # Compare marker1 (index 0 in both)
    assert np.allclose(motion_data[:, 0, :], mock_motion_data_raw[:, 0, :3], rtol=1e-5)


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(mock_motion_data[:, :, :3].copy(), mock_marker_names, mock_framerate),
)
def test_load_trackers_and_markerset_fallback(mock_from_c3d_to_numpy):
    # Create a mock markerset
    markerset_xml = """
    <markerset>
        <marker name="marker1"/>
        <marker name="marker3"/>
        <marker name="marker10"/>
        <marker name="marker_11"/>
        <marker name="marker-12"/>
        <marker name="marker_13"/>
        <marker name="marker14"/>
        <marker name="marker15"/>
    </markerset>
    """

    motion_data_subject_list, markerset, framerate = load_trackers_and_markerset(
        trackers_file_path="dummy_path.c3d", markerset_handle=markerset_xml
    )
    marker_names = get_marker_names(markerset)
    motion_data = motion_data_subject_list[0][0]

    # Check the shape of the filtered motion data
    assert motion_data.shape == (100, 8, 3)
    assert len(marker_names) == motion_data.shape[1]  # check marker_names filtering
    assert len(marker_names) == len(set(marker_names))  # ensure no duplicates
    assert framerate == mock_framerate


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(
        np.concatenate(
            [mock_motion_data[:, :, :3].copy(), mock_motion_data[:, :1, :3].copy()],
            axis=1,
        ),  # 16 columns: original 15 + duplicate of column 0
        mock_marker_names + ["marker1"],  # duplicate marker1
        mock_framerate,
    ),
)
def test_load_trackers_and_markerset_with_duplicate_markers(mock_from_c3d_to_numpy):
    # Create a mock markerset
    markerset_xml = """
    <markerset>
        <marker name="marker1"/>
        <marker name="marker2"/>
    </markerset>
    """

    # Expect an exception due to duplicate marker names
    with pytest.raises(Exception):
        load_trackers_and_markerset(
            trackers_file_path="dummy_path.c3d", markerset_handle=markerset_xml
        )


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(mock_motion_data[:, :, :3].copy(), mock_marker_names, mock_framerate),
)
def test_load_trackers_and_markerset_warns_on_missing_markers(mock_from_c3d_to_numpy):
    # Create a mock markerset
    markerset_xml = """
    <markerset>
        <marker name="marker1"/>
        <marker name="markerNotInC3D"/>
    </markerset>
    """

    with pytest.warns(UserWarning):
        load_trackers_and_markerset(
            trackers_file_path="dummy_path.c3d", markerset_handle=markerset_xml
        )


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(mock_motion_data[:, :, :3].copy(), mock_marker_names, mock_framerate),
)
def test_load_trackers_basic(mock_from_c3d_to_numpy):
    """Test basic functionality of load_trackers."""
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.c3d", mocap_scale=1000, clip_length=-1
    )

    # Check that from_c3d_to_numpy was called
    mock_from_c3d_to_numpy.assert_called_once_with("dummy_path.c3d")

    # Check that data was scaled by mocap_scale
    assert motion_data.shape == (100, 15, 3)
    assert len(tracker_names) == 15
    assert framerate == mock_framerate
    # Verify scaling was applied (data should be 1000x smaller than mock data)
    # mock_motion_data is already 1000x the raw values, so final should match raw
    assert np.allclose(motion_data[0, 0, 0], mock_motion_data_raw[0, 0, 0], rtol=1e-5)


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(mock_motion_data[:, :, :3].copy(), mock_marker_names, mock_framerate),
)
def test_load_trackers_with_clipping(mock_from_c3d_to_numpy):
    """Test that load_trackers clips motion data when clip_length is specified."""
    clip_length = 50
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.c3d", mocap_scale=1000, clip_length=clip_length
    )

    # Check that data was clipped
    assert motion_data.shape == (clip_length, 15, 3)
    assert len(tracker_names) == 15
    assert framerate == mock_framerate


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(mock_motion_data[:, :, :3].copy(), mock_marker_names, mock_framerate),
)
def test_load_trackers_with_custom_scale(mock_from_c3d_to_numpy):
    """Test that load_trackers applies custom mocap_scale."""
    custom_scale = 500
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.c3d", mocap_scale=custom_scale, clip_length=-1
    )

    # Verify custom scaling was applied
    # mock_motion_data is 1000x the raw values, so dividing by 500 gives 2x raw
    assert np.allclose(
        motion_data[0, 0, 0], mock_motion_data_raw[0, 0, 0] * 2, rtol=1e-5
    )


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(mock_motion_data[:, :, :3].copy(), mock_marker_names, mock_framerate),
)
def test_load_trackers_no_clipping(mock_from_c3d_to_numpy):
    """Test that load_trackers doesn't clip when clip_length is -1."""
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.c3d", mocap_scale=1000, clip_length=-1
    )

    # Check that data was not clipped
    assert motion_data.shape == (100, 15, 3)


def test_load_trackers_unsupported_format():
    """Test that load_trackers raises exception for unsupported file formats."""
    with pytest.raises(Exception, match="Unsupported trackers file format"):
        load_trackers("dummy_path.b3d", mocap_scale=1000, clip_length=-1)


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_c3d_to_numpy",
    return_value=(
        np.random.rand(100, 20, 3) * 1000,  # More trackers than names
        mock_marker_names,  # Only 15 names
        mock_framerate,
    ),
)
def test_load_trackers_filters_unnamed_trackers(mock_from_c3d_to_numpy):
    """Test that load_trackers filters out trackers without names."""
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.c3d", mocap_scale=1000, clip_length=-1
    )

    # Check that only trackers with names are kept
    assert motion_data.shape == (100, 15, 3)
    assert len(tracker_names) == 15


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_dataframe_to_array",
    return_value=(
        np.arange(100, dtype=np.float64) / 100.0,  # time axis
        mock_motion_data[:, :, :3],  # motion data
        mock_marker_names,  # tracker names
    ),
)
def test_load_trackers_from_parquet(mock_from_dataframe):
    """Test that load_trackers can load from .parquet files."""
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.parquet", mocap_scale=1000, clip_length=-1
    )

    # Check that from_dataframe_to_array was called
    mock_from_dataframe.assert_called_once_with("dummy_path.parquet")

    # Check that data was loaded and scaled
    assert motion_data.shape == (100, 15, 3)
    assert len(tracker_names) == 15
    # Framerate should be calculated from time axis (100 Hz in this case)
    assert np.isclose(framerate, 100.0)


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_dataframe_to_array",
    return_value=(
        np.arange(100, dtype=np.float64) / 100.0,  # time axis
        mock_motion_data[:, :, :3],  # motion data
        mock_marker_names,  # tracker names
    ),
)
def test_load_trackers_from_csv(mock_from_dataframe):
    """Test that load_trackers can load from .csv files."""
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.csv", mocap_scale=1000, clip_length=-1
    )

    # Check that from_dataframe_to_array was called
    mock_from_dataframe.assert_called_once_with("dummy_path.csv")

    # Check that data was loaded and scaled
    assert motion_data.shape == (100, 15, 3)
    assert len(tracker_names) == 15
    # Framerate should be calculated from time axis
    assert np.isclose(framerate, 100.0)


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_dataframe_to_array",
    return_value=(
        np.arange(100, dtype=np.float64) / 120.0,  # time axis at 120 Hz
        mock_motion_data[:, :, :3],  # motion data
        mock_marker_names,  # tracker names
    ),
)
def test_load_trackers_csv_with_different_framerate(mock_from_dataframe):
    """Test that framerate is correctly calculated from time axis for CSV/Parquet."""
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.csv", mocap_scale=1000, clip_length=-1
    )

    # Framerate should be calculated from time axis (120 Hz in this case)
    assert np.isclose(framerate, 120.0)


@patch(
    "myo_tools.utils.mocap_ops.mocap_utils.from_dataframe_to_array",
    return_value=(
        np.arange(100, dtype=np.float64) / 100.0,
        mock_motion_data[:, :, :3],
        mock_marker_names,
    ),
)
def test_load_trackers_csv_with_clipping(mock_from_dataframe):
    """Test that clipping works with CSV/Parquet files."""
    clip_length = 50
    motion_data, tracker_names, framerate = load_trackers(
        "dummy_path.csv", mocap_scale=1000, clip_length=clip_length
    )

    # Check that data was clipped
    assert motion_data.shape == (clip_length, 15, 3)


def test_load_trackers_trc(trc_file_path, markerset_file_path):
    """Test load_trackers with a TRC file."""
    motion_data, tracker_names, framerate = load_trackers_and_markerset(
        trackers_file_path=trc_file_path,
        markerset_handle=markerset_file_path,
        mocap_scale=1,
    )

    # Basic assertions
    assert motion_data is not None
    assert tracker_names is not None
    assert framerate is not None

    # Check shapes
    assert motion_data[0][0].ndim == 3  # (num_frames, num_trackers, 3)
    assert len(tracker_names) == motion_data[0][0].shape[1]
