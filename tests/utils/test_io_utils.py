"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from myo_tools.utils.file_ops.io_utils import (
    from_qpos_to_joint_angles,
    load_and_convert_trackers_and_markerset,
    save_qpos,
)


class TestLoadAndConvertTrackersAndMarkerset:
    """Test suite for load_and_convert_trackers_and_markerset function."""

    @pytest.fixture
    def mock_trackers_data(self):
        """Create mock tracker data."""
        # Shape: (num_frames, num_markers, 3)
        return np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]],
            ]
        )

    @pytest.fixture
    def mock_markerset(self):
        """Create mock markerset XML element."""
        root = ET.Element("markers")
        for i, name in enumerate(["marker1", "marker2", "marker3"]):
            marker = ET.SubElement(root, "marker")
            marker.set("name", name)
            marker.set("body", f"body{i}")
            marker.set("pos", f"{i} {i+1} {i+2}")
        return root

    @pytest.fixture
    def mock_framerate(self):
        """Mock framerate."""
        return 100.0

    def test_basic_conversion(self, mock_trackers_data, mock_markerset, mock_framerate):
        """Test basic conversion from trackers and markerset to output files."""
        # Mock the wrapped subject list structure
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input_markerset.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output_markerset.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                )

                # Verify load_trackers_and_markerset was called correctly
                mock_load.assert_called_once_with(
                    trackers_file_path=input_trackers,
                    markerset_handle=input_markerset,
                    mocap_scale=1000,  # default mocap_scale
                    clip_length=-1,  # default clip_length
                )

            # Check output trackers file exists and is valid
            assert os.path.exists(output_trackers)
            df = pd.read_parquet(output_trackers)
            assert len(df) == 3  # 3 frames
            assert "time" in df.columns
            assert "marker1_x" in df.columns
            assert "marker2_y" in df.columns
            assert "marker3_z" in df.columns

            # Check output markerset file exists and is valid
            assert os.path.exists(output_markerset)
            tree = ET.parse(output_markerset)
            root = tree.getroot()
            assert root.tag == "markers"
            markers = root.findall("marker")
            assert len(markers) == 3

    def test_custom_mocap_scale(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test conversion with custom mocap_scale parameter."""
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                    mocap_scale=500,
                )

                # Verify mocap_scale was passed correctly
                mock_load.assert_called_once_with(
                    trackers_file_path=input_trackers,
                    markerset_handle=input_markerset,
                    mocap_scale=500,
                    clip_length=-1,
                )

    def test_custom_clip_length(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test conversion with custom clip_length parameter."""
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                    clip_length=100,
                )

                # Verify clip_length was passed correctly
                mock_load.assert_called_once_with(
                    trackers_file_path=input_trackers,
                    markerset_handle=input_markerset,
                    mocap_scale=1000,
                    clip_length=100,
                )

    def test_both_custom_parameters(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test conversion with both custom mocap_scale and clip_length."""
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                    mocap_scale=2000,
                    clip_length=50,
                )

                # Verify both parameters were passed correctly
                mock_load.assert_called_once_with(
                    trackers_file_path=input_trackers,
                    markerset_handle=input_markerset,
                    mocap_scale=2000,
                    clip_length=50,
                )

    def test_output_trackers_content(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test that output trackers file contains correct data."""
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                )

            # Read and verify output trackers
            df = pd.read_parquet(output_trackers)

            # Check shape
            assert len(df) == 3  # 3 frames

            # Check time column
            expected_time = np.array([0.0, 0.01, 0.02])
            np.testing.assert_array_almost_equal(df["time"].to_numpy(), expected_time)

            # Check first frame values
            assert df["marker1_x"].iloc[0] == 1.0
            assert df["marker1_y"].iloc[0] == 2.0
            assert df["marker1_z"].iloc[0] == 3.0

            # Check last frame values
            assert df["marker3_x"].iloc[2] == 25.0
            assert df["marker3_y"].iloc[2] == 26.0
            assert df["marker3_z"].iloc[2] == 27.0

    def test_output_markerset_content(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test that output markerset file contains correct XML structure."""
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                )

            # Read and verify output markerset
            tree = ET.parse(output_markerset)
            root = tree.getroot()

            assert root.tag == "markers"
            markers = root.findall("marker")
            assert len(markers) == 3

            # Check marker attributes
            assert markers[0].get("name") == "marker1"
            assert markers[1].get("name") == "marker2"
            assert markers[2].get("name") == "marker3"

            assert markers[0].get("body") == "body0"
            assert markers[1].get("body") == "body1"
            assert markers[2].get("body") == "body2"

    def test_nested_output_directories(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test that function creates nested output directories for parquet."""
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "nested", "dir", "output.parquet")
            # XML output requires parent directory to exist
            xml_dir = os.path.join(tmpdir, "xml_output")
            os.makedirs(xml_dir, exist_ok=True)
            output_markerset = os.path.join(xml_dir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                )

            # Check that nested directories were created and files exist
            assert os.path.exists(output_trackers)
            assert os.path.exists(output_markerset)

    def test_single_subject_extraction(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test that function correctly extracts single subject from nested list."""
        # Create multi-level nested structure as returned by load_trackers_and_markerset
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                # Should not raise any errors
                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                )

            # Verify the output is correct
            df = pd.read_parquet(output_trackers)
            assert len(df) == 3

    def test_marker_names_integration(
        self, mock_trackers_data, mock_markerset, mock_framerate
    ):
        """Test that marker names are correctly extracted and used in output."""
        motion_data_subject_list = [[mock_trackers_data]]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    mock_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                )

            # Verify marker names are in the output
            df = pd.read_parquet(output_trackers)
            expected_columns = [
                "time",
                "marker1_x",
                "marker1_y",
                "marker1_z",
                "marker2_x",
                "marker2_y",
                "marker2_z",
                "marker3_x",
                "marker3_y",
                "marker3_z",
            ]
            assert list(df.columns) == expected_columns

    def test_large_dataset(self, mock_markerset, mock_framerate):
        """Test with a larger dataset to ensure scalability."""
        # Create larger dataset
        n_frames = 1000
        n_markers = 10
        large_trackers = np.random.rand(n_frames, n_markers, 3)
        motion_data_subject_list = [[large_trackers]]

        # Create corresponding markerset
        large_markerset = ET.Element("markers")
        for i in range(n_markers):
            marker = ET.SubElement(large_markerset, "marker")
            marker.set("name", f"marker_{i}")
            marker.set("body", f"body_{i}")
            marker.set("pos", "0 0 0")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_trackers = os.path.join(tmpdir, "input.c3d")
            input_markerset = os.path.join(tmpdir, "input.xml")
            output_trackers = os.path.join(tmpdir, "output.parquet")
            output_markerset = os.path.join(tmpdir, "output.xml")

            with patch(
                "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
            ) as mock_load:
                mock_load.return_value = (
                    motion_data_subject_list,
                    large_markerset,
                    mock_framerate,
                )

                load_and_convert_trackers_and_markerset(
                    input_trackers,
                    input_markerset,
                    output_trackers,
                    output_markerset,
                )

            # Verify output
            df = pd.read_parquet(output_trackers)
            assert len(df) == n_frames
            assert len(df.columns) == 1 + n_markers * 3  # time + markers * xyz

    def test_different_framerates(self, mock_trackers_data, mock_markerset):
        """Test with different framerates."""
        motion_data_subject_list = [[mock_trackers_data]]

        for fps in [30.0, 60.0, 120.0, 240.0]:
            with tempfile.TemporaryDirectory() as tmpdir:
                input_trackers = os.path.join(tmpdir, "input.c3d")
                input_markerset = os.path.join(tmpdir, "input.xml")
                output_trackers = os.path.join(tmpdir, "output.parquet")
                output_markerset = os.path.join(tmpdir, "output.xml")

                with patch(
                    "myo_tools.utils.file_ops.io_utils.load_trackers_and_markerset"
                ) as mock_load:
                    mock_load.return_value = (
                        motion_data_subject_list,
                        mock_markerset,
                        fps,
                    )

                    load_and_convert_trackers_and_markerset(
                        input_trackers,
                        input_markerset,
                        output_trackers,
                        output_markerset,
                    )

                # Verify time column has correct values based on framerate
                df = pd.read_parquet(output_trackers)
                expected_time = np.arange(3, dtype=np.float64) / fps
                np.testing.assert_array_almost_equal(
                    df["time"].to_numpy(), expected_time
                )


class TestSaveQpos:
    """Test suite for save_qpos function."""

    def test_basic_qpos_save(self):
        """Test basic qpos saving to parquet file."""
        qpos = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )
        colnames = ["joint1", "joint2", "joint3", "joint4"]
        framerate = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            # Check file exists
            assert os.path.exists(output_path)

            # Check content
            df = pd.read_parquet(output_path)
            assert len(df) == 3
            assert "time" in df.columns
            assert all(col in df.columns for col in colnames)

            # Check values
            assert df["joint1"].iloc[0] == 1.0
            assert df["joint2"].iloc[1] == 6.0
            assert df["joint4"].iloc[2] == 12.0

    def test_qpos_time_calculation(self):
        """Test that time column is calculated correctly."""
        n_frames = 1000
        qpos = np.random.rand(n_frames, 5)
        colnames = ["q1", "q2", "q3", "q4", "q5"]
        framerate = 120.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            df = pd.read_parquet(output_path)

            # Verify time values
            expected_time = np.arange(n_frames, dtype=np.float64) / framerate
            np.testing.assert_array_almost_equal(df["time"].to_numpy(), expected_time)

    def test_qpos_with_nested_directory(self):
        """Test that nested directories are created for qpos output."""
        qpos = np.array([[1.0, 2.0], [3.0, 4.0]])
        colnames = ["joint_a", "joint_b"]
        framerate = 60.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "dir", "qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            # Check file was created in nested directory
            assert os.path.exists(output_path)

            df = pd.read_parquet(output_path)
            assert len(df) == 2
            assert list(df.columns) == ["time", "joint_a", "joint_b"]

    def test_qpos_single_frame(self):
        """Test qpos with a single frame."""
        qpos = np.array([[1.0, 2.0, 3.0]])
        colnames = ["j1", "j2", "j3"]
        framerate = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "single_frame.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            df = pd.read_parquet(output_path)
            assert len(df) == 1
            assert df["time"].iloc[0] == 0.0
            assert df["j1"].iloc[0] == 1.0

    def test_qpos_large_dataset(self):
        """Test qpos with a large dataset."""
        n_frames = 10000
        n_joints = 50
        qpos = np.random.rand(n_frames, n_joints)
        colnames = [f"joint_{i}" for i in range(n_joints)]
        framerate = 240.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "large_qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            df = pd.read_parquet(output_path)
            assert len(df) == n_frames
            assert len(df.columns) == n_joints + 1  # joints + time

    def test_qpos_different_framerates(self):
        """Test qpos with various framerates."""
        qpos = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
        colnames = ["joint1", "joint2"]

        framerates = [30.0, 60.0, 120.0, 240.0]

        for fps in framerates:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, f"qpos_{fps}.parquet")

                save_qpos(qpos, colnames, fps, output_path)

                df = pd.read_parquet(output_path)
                expected_time = np.arange(4, dtype=np.float64) / fps
                np.testing.assert_array_almost_equal(
                    df["time"].to_numpy(), expected_time
                )

    def test_qpos_column_order(self):
        """Test that columns are in the correct order (time first, then joints)."""
        qpos = np.array([[1.0, 2.0, 3.0]])
        colnames = ["alpha", "beta", "gamma"]
        framerate = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            df = pd.read_parquet(output_path)
            assert list(df.columns) == ["time", "alpha", "beta", "gamma"]

    def test_qpos_data_preservation(self):
        """Test that qpos data is preserved accurately."""
        qpos = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [2.1, 2.2, 2.3, 2.4, 2.5],
            ]
        )
        colnames = ["q0", "q1", "q2", "q3", "q4"]
        framerate = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            df = pd.read_parquet(output_path)

            # Check all values are preserved
            for i, col in enumerate(colnames):
                np.testing.assert_array_almost_equal(df[col].to_numpy(), qpos[:, i])

    def test_qpos_empty_array(self):
        """Test qpos with empty array (edge case)."""
        qpos = np.array([]).reshape(0, 3)
        colnames = ["j1", "j2", "j3"]
        framerate = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "empty_qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            df = pd.read_parquet(output_path)
            assert len(df) == 0
            assert list(df.columns) == ["time", "j1", "j2", "j3"]

    def test_qpos_with_special_characters_in_names(self):
        """Test qpos with special characters in column names."""
        qpos = np.array([[1.0, 2.0, 3.0]])
        colnames = ["joint_1", "joint-2", "joint:3"]
        framerate = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "qpos.parquet")

            save_qpos(qpos, colnames, framerate, output_path)

            df = pd.read_parquet(output_path)
            assert "joint_1" in df.columns
            assert "joint-2" in df.columns
            assert "joint:3" in df.columns


class TestFromQposToJointAngles:
    """Test suite for from_qpos_to_joint_angles function."""

    def test_basic_conversion_hinge_joints(self):
        """Test basic conversion with hinge joints (no ball shoulders)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with hinge joints
            qpos_data = np.array(
                [
                    [0.0, 0.1, 0.2, 0.3, 0.4],
                    [0.01, 0.15, 0.25, 0.35, 0.45],
                    [0.02, 0.2, 0.3, 0.4, 0.5],
                ]
            )
            colnames = [
                "elbow_flex_r",
                "knee_flex_r",
                "ankle_flex_r",
                "wrist_flex_r",
                "hip_flex_r",
            ]

            # Create DataFrame and save as parquet
            time_axis = np.array([0.0, 0.01, 0.02])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.parquet")
            df.to_parquet(input_path)

            # Convert to joint angles
            result_df = from_qpos_to_joint_angles(input_path)

            # Check that result is a DataFrame
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 3

            # Check that all angles are in degrees
            assert "time" in result_df.columns
            for col in colnames:
                assert col in result_df.columns

            # Check degree conversion
            np.testing.assert_almost_equal(
                result_df["elbow_flex_r"].iloc[0], 0.0, decimal=5
            )
            np.testing.assert_almost_equal(
                result_df["knee_flex_r"].iloc[0], np.rad2deg(0.1), decimal=5
            )

    def test_ball_shoulder_conversion(self):
        """Test conversion with ball shoulders (quaternions to euler)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with ball shoulder joints (quaternions)
            qpos_data = np.array(
                [
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.1,
                        0.2,
                    ],  # qw, qx, qy, qz for each shoulder + other joints
                    [0.9239, 0.3827, 0.0, 0.0, 0.9239, 0.3827, 0.0, 0.0, 0.15, 0.25],
                    [0.7071, 0.7071, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0, 0.2, 0.3],
                ]
            )
            colnames = [
                "shoulder_r_qw",
                "shoulder_r_qx",
                "shoulder_r_qy",
                "shoulder_r_qz",
                "shoulder_l_qw",
                "shoulder_l_qx",
                "shoulder_l_qy",
                "shoulder_l_qz",
                "elbow_flex_r",
                "knee_flex_r",
            ]

            # Create DataFrame
            time_axis = np.array([0.0, 0.01, 0.02])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.parquet")
            df.to_parquet(input_path)

            # Convert to joint angles
            result_df = from_qpos_to_joint_angles(input_path)

            # Check that quaternions were converted to euler angles
            assert "shoulder_abdu_r" in result_df.columns
            assert "humerus_arot_r" in result_df.columns
            assert "shoulder_flex_r" in result_df.columns
            assert "shoulder_abdu_l" in result_df.columns
            assert "humerus_arot_l" in result_df.columns
            assert "shoulder_flex_l" in result_df.columns

            # Check that quaternion columns were removed
            assert "shoulder_r_qw" not in result_df.columns
            assert "shoulder_r_qx" not in result_df.columns
            assert "shoulder_l_qw" not in result_df.columns
            assert "shoulder_l_qx" not in result_df.columns

            # Check that other joints remain and are converted to degrees
            assert "elbow_flex_r" in result_df.columns
            assert "knee_flex_r" in result_df.columns
            assert result_df["elbow_flex_r"].iloc[0] == pytest.approx(
                5.729577951308232, abs=1e-5
            )
            assert result_df["knee_flex_r"].iloc[0] == pytest.approx(
                11.459155902616465, abs=1e-5
            )

    def test_ball_shoulder_filtered_columns(self):
        """Test that patella, abs, translation, and root joints are filtered out with ball shoulders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with ball shoulders and various joint types to filter
            qpos_data = np.array(
                [
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                    ],
                    [
                        0.9239,
                        0.3827,
                        0.0,
                        0.0,
                        0.9239,
                        0.3827,
                        0.0,
                        0.0,
                        0.01,
                        0.15,
                        0.25,
                        0.35,
                        0.45,
                        0.55,
                    ],
                ]
            )
            colnames = [
                "shoulder_r_qw",
                "shoulder_r_qx",
                "shoulder_r_qy",
                "shoulder_r_qz",
                "shoulder_l_qw",
                "shoulder_l_qx",
                "shoulder_l_qy",
                "shoulder_l_qz",
                "root_x",  # Should be filtered
                "abs_flex",  # Should be filtered
                "patella_r",  # Should be filtered
                "joint_tx",  # Should be filtered (translation)
                "elbow_flex_r",  # Should be kept
                "knee_flex_r",  # Should be kept
            ]

            # Create DataFrame
            time_axis = np.array([0.0, 0.01])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.parquet")
            df.to_parquet(input_path)

            # Convert to joint angles
            result_df = from_qpos_to_joint_angles(input_path)

            # Check that filtered columns are removed
            assert "root_x" not in result_df.columns
            assert "abs_flex" not in result_df.columns
            assert "patella_r" not in result_df.columns
            assert "joint_tx" not in result_df.columns

            # Check that shoulder euler angles and other joints remain
            assert "shoulder_abdu_r" in result_df.columns
            assert "elbow_flex_r" in result_df.columns
            assert "knee_flex_r" in result_df.columns

    def test_ball_shoulder_degrees_conversion(self):
        """Test that angles are converted from radians to degrees."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with ball shoulders
            qpos_data = np.array(
                [
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        np.pi / 4,
                        np.pi / 2,
                    ],  # 0°, 45°, 90° in radians
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        np.pi,
                        3 * np.pi / 2,
                        2 * np.pi,
                    ],  # 180°, 270°, 360° in radians
                ]
            )
            colnames = [
                "shoulder_r_qw",
                "shoulder_r_qx",
                "shoulder_r_qy",
                "shoulder_r_qz",
                "shoulder_l_qw",
                "shoulder_l_qx",
                "shoulder_l_qy",
                "shoulder_l_qz",
                "joint1",
                "joint2",
                "joint3",
            ]

            # Create DataFrame
            time_axis = np.array([0.0, 0.01])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.parquet")
            df.to_parquet(input_path)

            # Convert to joint angles
            result_df = from_qpos_to_joint_angles(input_path)

            # Check conversion to degrees
            np.testing.assert_almost_equal(result_df["joint1"].iloc[0], 0.0, decimal=5)
            np.testing.assert_almost_equal(result_df["joint2"].iloc[0], 45.0, decimal=5)
            np.testing.assert_almost_equal(result_df["joint3"].iloc[0], 90.0, decimal=5)
            np.testing.assert_almost_equal(
                result_df["joint1"].iloc[1], 180.0, decimal=5
            )
            np.testing.assert_almost_equal(
                result_df["joint2"].iloc[1], 270.0, decimal=5
            )
            np.testing.assert_almost_equal(
                result_df["joint3"].iloc[1], 360.0, decimal=5
            )

    def test_ball_shoulder_save_to_file(self):
        """Test that output is saved to file when output_angles_file_path is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with ball shoulders
            qpos_data = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                    [0.9239, 0.3827, 0.0, 0.0, 0.9239, 0.3827, 0.0, 0.0, 0.15, 0.25],
                ]
            )
            colnames = [
                "shoulder_r_qw",
                "shoulder_r_qx",
                "shoulder_r_qy",
                "shoulder_r_qz",
                "shoulder_l_qw",
                "shoulder_l_qx",
                "shoulder_l_qy",
                "shoulder_l_qz",
                "joint1",
                "joint2",
            ]

            # Create DataFrame
            time_axis = np.array([0.0, 0.01])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.parquet")
            output_path = os.path.join(tmpdir, "angles.parquet")
            df.to_parquet(input_path)

            # Convert and save
            result_df = from_qpos_to_joint_angles(input_path, output_path)

            # Check that file was created
            assert os.path.exists(output_path)

            # Check that saved file matches returned DataFrame
            saved_df = pd.read_parquet(output_path)
            pd.testing.assert_frame_equal(result_df, saved_df)

    def test_ball_shoulder_save_to_csv(self):
        """Test that output can be saved to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with ball shoulders
            qpos_data = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                    [0.9239, 0.3827, 0.0, 0.0, 0.9239, 0.3827, 0.0, 0.0, 0.15, 0.25],
                ]
            )
            colnames = [
                "shoulder_r_qw",
                "shoulder_r_qx",
                "shoulder_r_qy",
                "shoulder_r_qz",
                "shoulder_l_qw",
                "shoulder_l_qx",
                "shoulder_l_qy",
                "shoulder_l_qz",
                "joint1",
                "joint2",
            ]

            # Create DataFrame
            time_axis = np.array([0.0, 0.01])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.csv")
            output_path = os.path.join(tmpdir, "angles.csv")
            df.to_csv(input_path, index=False)

            # Convert and save
            result_df = from_qpos_to_joint_angles(input_path, output_path)

            # Check that CSV file was created
            assert os.path.exists(output_path)

            # Check that saved file matches returned DataFrame
            saved_df = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(result_df, saved_df)

    def test_ball_shoulder_input_as_dataframe(self):
        """Test that function accepts DataFrame as input."""
        # Create qpos DataFrame with ball shoulders
        qpos_data = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1],
                [0.9239, 0.3827, 0.0, 0.0, 0.9239, 0.3827, 0.0, 0.0, 0.15],
            ]
        )
        colnames = [
            "shoulder_r_qw",
            "shoulder_r_qx",
            "shoulder_r_qy",
            "shoulder_r_qz",
            "shoulder_l_qw",
            "shoulder_l_qx",
            "shoulder_l_qy",
            "shoulder_l_qz",
            "joint1",
        ]
        time_axis = np.array([0.0, 0.01])
        df = pd.DataFrame(qpos_data, columns=colnames)
        df.insert(0, "time", time_axis)

        # Convert using DataFrame directly
        result_df = from_qpos_to_joint_angles(df)

        # Check that result is valid
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "time" in result_df.columns
        assert "shoulder_abdu_r" in result_df.columns

    def test_ball_shoulder_framerate_calculation(self):
        """Test that framerate is correctly calculated from time axis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with ball shoulders and specific framerate
            qpos_data = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.15],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2],
                ]
            )
            colnames = [
                "shoulder_r_qw",
                "shoulder_r_qx",
                "shoulder_r_qy",
                "shoulder_r_qz",
                "shoulder_l_qw",
                "shoulder_l_qx",
                "shoulder_l_qy",
                "shoulder_l_qz",
                "joint1",
            ]

            # Create time axis for 100 Hz (0.01 second intervals)
            time_axis = np.array([0.0, 0.01, 0.02])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.parquet")
            df.to_parquet(input_path)

            # Convert
            result_df = from_qpos_to_joint_angles(input_path)

            # Check that time intervals are preserved
            result_time = result_df["time"].to_numpy()
            time_diffs = np.diff(result_time)
            expected_interval = 0.01
            np.testing.assert_array_almost_equal(
                time_diffs, [expected_interval, expected_interval], decimal=5
            )

    def test_ball_shoulder_mixed_joints(self):
        """Test with ball shoulders and a mix of kept and filtered joints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create qpos data with ball shoulders
            qpos_data = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4],
                    [
                        0.9239,
                        0.3827,
                        0.0,
                        0.0,
                        0.9239,
                        0.3827,
                        0.0,
                        0.0,
                        0.01,
                        0.15,
                        0.25,
                        0.35,
                        0.45,
                    ],
                ]
            )
            colnames = [
                "shoulder_r_qw",
                "shoulder_r_qx",
                "shoulder_r_qy",
                "shoulder_r_qz",
                "shoulder_l_qw",
                "shoulder_l_qx",
                "shoulder_l_qy",
                "shoulder_l_qz",
                "root_x",
                "elbow_flex_r",
                "abs_flex",
                "knee_flex_r",
                "patella_r",
            ]

            # Create DataFrame
            time_axis = np.array([0.0, 0.01])
            df = pd.DataFrame(qpos_data, columns=colnames)
            df.insert(0, "time", time_axis)

            input_path = os.path.join(tmpdir, "qpos.parquet")
            df.to_parquet(input_path)

            # Convert
            result_df = from_qpos_to_joint_angles(input_path)

            # Check that only non-filtered joints remain
            assert "shoulder_abdu_r" in result_df.columns
            assert "elbow_flex_r" in result_df.columns
            assert "knee_flex_r" in result_df.columns
            assert "root_x" not in result_df.columns
            assert "abs_flex" not in result_df.columns
            assert "patella_r" not in result_df.columns
