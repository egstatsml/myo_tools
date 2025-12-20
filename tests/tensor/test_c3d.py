"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for c3d file loading utilities

from unittest.mock import mock_open, patch

import ezc3d
import numpy as np
import pytest

from myo_tools.utils.mocap_ops.c3d_utils import (
    c3dpy_loader,
    ezc3d_loader,
    from_c3d_to_numpy,
)


@pytest.fixture
def example_c3d_file(tmp_path):
    """
    Creates a small C3D file with two markers over two frames, and saves it
    to a temporary path. Yields the path to the created file.
    """
    test_file = tmp_path / "test_created.c3d"

    # Create a new C3D file
    c3d = ezc3d.c3d()

    # Set up parameters
    c3d["parameters"]["POINT"]["USED"]["value"] = [2]  # 2 markers
    c3d["parameters"]["POINT"]["LABELS"]["value"] = ["Marker1", "Marker2"]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [480.0]  # Frame rate
    c3d["parameters"]["POINT"]["FRAMES"]["value"] = [2]  # 2 frames

    # Prepare point data: shape should be (3, n_markers, n_frames)
    # Each marker has (x, y, z) coordinates
    point_data = np.zeros((3, 2, 2))  # 3 coords, 2 markers, 2 frames

    # Frame 1: marker1 at (0,0,0), marker2 at (1,1,1)
    point_data[:, 0, 0] = [0, 0, 0]  # Marker1, Frame1
    point_data[:, 1, 0] = [1, 1, 1]  # Marker2, Frame1

    # Frame 2: marker1 at (2,2,2), marker2 at (3,3,3)
    point_data[:, 0, 1] = [2, 2, 2]  # Marker1, Frame2
    point_data[:, 1, 1] = [3, 3, 3]  # Marker2, Frame2

    # Set the point data
    c3d["data"]["points"] = point_data

    # Write to file
    c3d.write(str(test_file))

    yield str(test_file)


def test_create_and_from_c3d_to_numpy(example_c3d_file):
    """
    Test reading the generated C3D file using from_c3d_to_numpy,
    and ensure the motion data and metadata are as expected.
    """
    motion_data, marker_names, framerate = from_c3d_to_numpy(example_c3d_file)

    # We expect shape: 2 frames, 2 markers, dimension=3 (x,y,z only from ezc3d_loader)
    print(motion_data)
    assert motion_data.shape == (2, 2, 3), f"Unexpected shape: {motion_data.shape}"

    # Validate the exact frame data we wrote above
    # ezc3d_loader only returns x, y, z coordinates
    expected_data = np.array(
        [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]],
        dtype=np.float32,
    )

    # Check the x, y, z coordinates
    np.testing.assert_allclose(motion_data, expected_data, atol=1e-5)

    # Check the marker names
    assert marker_names == ["Marker1", "Marker2"], f"Unexpected labels: {marker_names}"

    # Check that we get the expected frame rate (480 as set in ezc3d creation)
    assert framerate == 480, f"Unexpected frame rate: {framerate}"


# Mock data for testing
mock_c3d_data = {
    "data": {"points": np.random.rand(3, 2, 100)},  # 3 coords, 2 markers, 100 frames
    "parameters": {
        "POINT": {
            "LABELS": {"value": ["marker1", "marker2"]},
            "RATE": {"value": [100.0]},
        }
    },
}

mock_motion_data = np.random.rand(100, 2, 3)  # 100 frames, 2 markers, 3 coords (x,y,z)
mock_marker_names = ["marker1", "marker2"]
mock_framerate = 100.0


@patch("ezc3d.c3d", return_value=mock_c3d_data)
def test_ezc3d_loader(mock_ezc3d):
    motion_data, marker_names, framerate, success = ezc3d_loader("dummy_path.c3d")
    assert success
    assert motion_data.shape == (100, 2, 3)  # Updated to match 2 markers
    assert marker_names == ["marker1", "marker2"]
    assert framerate == 100.0


@patch("c3d.Reader")
def test_c3dpy_loader(mock_reader):
    mock_reader.return_value.point_labels = ["marker1", "marker2"]
    mock_reader.return_value.header.frame_rate = 100.0
    mock_reader.return_value.read_frames.return_value = [
        (None, mock_motion_data[i], None) for i in range(100)
    ]

    with patch("builtins.open", mock_open(read_data="data")):
        motion_data, marker_names, framerate, success = c3dpy_loader("dummy_path.c3d")

    assert success
    assert motion_data.shape == (100, 2, 3)  # Updated to match 2 markers
    assert marker_names == ["marker1", "marker2"]
    assert framerate == 100.0


@patch(
    "myo_tools.utils.mocap_ops.c3d_utils.c3dpy_loader",
    return_value=(None, None, None, False),
)
@patch(
    "myo_tools.utils.mocap_ops.c3d_utils.ezc3d_loader",
    return_value=(mock_motion_data, mock_marker_names, mock_framerate, True),
)
def test_from_c3d_to_numpy(mock_ezc3d_loader, mock_c3dpy_loader):
    motion_data, marker_names, framerate = from_c3d_to_numpy("dummy_path.c3d")
    assert motion_data.shape == (100, 2, 3)  # Updated to match 2 markers
    assert marker_names == ["marker1", "marker2"]
    assert framerate == 100.0


@patch(
    "myo_tools.utils.mocap_ops.c3d_utils.c3dpy_loader",
    return_value=(None, None, None, False),
)
@patch(
    "myo_tools.utils.mocap_ops.c3d_utils.ezc3d_loader",
    return_value=(None, None, None, False),
)
def test_from_c3d_to_numpy_failure(mock_ezc3d_loader, mock_c3dpy_loader):
    with pytest.raises(Exception, match="Could not load c3d file"):
        from_c3d_to_numpy("dummy_path.c3d")
