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

import numpy as np
import pandas as pd
import pytest

from myo_tools.utils.file_ops.dataframe_utils import (
    from_array_to_dataframe,
    from_dataframe_to_array,
    validate_dataframe_file,
)


class TestFromArrayToDataframe:
    """Test suite for from_array_to_dataframe function."""

    def test_3d_array_basic(self):
        """Test conversion of 3D array (N, T, 3) to DataFrame."""
        nparray = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        column_names = ["tracker1", "tracker2"]
        fps = 100.0

        df = from_array_to_dataframe(nparray, column_names, fps)

        # Check shape
        assert len(df) == 1
        assert len(df.columns) == 7  # time + 2 trackers * 3 coords

        # Check time column
        assert "time" in df.columns
        assert df.columns[0] == "time"
        assert df["time"].iloc[0] == 0.0

        # Check coordinate columns
        assert "tracker1_x" in df.columns
        assert "tracker1_y" in df.columns
        assert "tracker1_z" in df.columns
        assert "tracker2_x" in df.columns
        assert "tracker2_y" in df.columns
        assert "tracker2_z" in df.columns

        # Check values
        assert df["tracker1_x"].iloc[0] == 1.0
        assert df["tracker1_y"].iloc[0] == 2.0
        assert df["tracker1_z"].iloc[0] == 3.0
        assert df["tracker2_x"].iloc[0] == 4.0
        assert df["tracker2_y"].iloc[0] == 5.0
        assert df["tracker2_z"].iloc[0] == 6.0

    def test_3d_array_multiple_frames(self):
        """Test 3D array with multiple frames."""
        nparray = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        )
        column_names = ["marker_a", "marker_b"]
        fps = 50.0

        df = from_array_to_dataframe(nparray, column_names, fps)

        # Check shape
        assert len(df) == 3
        assert len(df.columns) == 7

        # Check time column
        np.testing.assert_array_almost_equal(df["time"].to_numpy(), [0.0, 0.02, 0.04])

        # Check first frame
        assert df["marker_a_x"].iloc[0] == 1.0
        # Check last frame
        assert df["marker_b_z"].iloc[2] == 18.0

    def test_2d_array_basic(self):
        """Test conversion of 2D array (N, J) to DataFrame."""
        nparray = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        column_names = ["joint1", "joint2", "joint3"]
        fps = 100.0

        df = from_array_to_dataframe(nparray, column_names, fps)

        # Check shape
        assert len(df) == 2
        assert len(df.columns) == 4  # time + 3 joints

        # Check columns
        assert df.columns[0] == "time"
        assert "joint1" in df.columns
        assert "joint2" in df.columns
        assert "joint3" in df.columns

        # Check values
        assert df["joint1"].iloc[0] == 1.0
        assert df["joint2"].iloc[1] == 5.0

    def test_write_to_parquet(self):
        """Test writing DataFrame to Parquet file."""
        nparray = np.array([[[1.0, 2.0, 3.0]]])
        column_names = ["tracker1"]
        fps = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.parquet")
            df = from_array_to_dataframe(nparray, column_names, fps, output_path)

            # Check file was created
            assert os.path.exists(output_path)

            # Check we can read it back
            df_loaded = pd.read_parquet(output_path)
            pd.testing.assert_frame_equal(df, df_loaded)

    def test_write_to_nested_directory(self):
        """Test writing to a nested directory structure."""
        nparray = np.array([[1.0, 2.0]])
        column_names = ["col1", "col2"]
        fps = 60.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir", "nested", "data.parquet")
            df = from_array_to_dataframe(nparray, column_names, fps, output_path)

            # Check file and directories were created
            assert os.path.exists(output_path)
            df_loaded = pd.read_parquet(output_path)
            pd.testing.assert_frame_equal(df, df_loaded)

    def test_write_to_csv(self):
        """Test writing DataFrame to CSV file."""
        nparray = np.array([[[1.0, 2.0, 3.0]]])
        column_names = ["tracker1"]
        fps = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.csv")
            df = from_array_to_dataframe(nparray, column_names, fps, output_path)

            # Check file was created
            assert os.path.exists(output_path)

            # Check we can read it back
            df_loaded = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(df, df_loaded)

    def test_write_csv_nested_directory(self):
        """Test writing CSV to a nested directory structure."""
        nparray = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        column_names = ["col1", "col2", "col3"]
        fps = 60.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "data.csv")
            df = from_array_to_dataframe(nparray, column_names, fps, output_path)

            # Check file and directories were created
            assert os.path.exists(output_path)
            df_loaded = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(df, df_loaded)

    def test_unsupported_file_extension(self):
        """Test that unsupported file extensions raise ValueError."""
        nparray = np.array([[1.0, 2.0]])
        column_names = ["col1", "col2"]
        fps = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "data.txt")

            with pytest.raises(ValueError, match="Unsupported file format"):
                from_array_to_dataframe(nparray, column_names, fps, output_path)

    def test_invalid_dimensionality(self):
        """Test that invalid array dimensions raise ValueError."""
        nparray = np.array([1.0, 2.0, 3.0])  # 1D array
        column_names = ["col1"]
        fps = 100.0

        with pytest.raises(ValueError, match="nparray must have shape"):
            from_array_to_dataframe(nparray, column_names, fps)

    def test_4d_array_raises_error(self):
        """Test that 4D arrays raise ValueError."""
        nparray = np.array([[[[1.0]]]])  # 4D array
        column_names = ["col1"]
        fps = 100.0

        with pytest.raises(ValueError, match="nparray must have shape"):
            from_array_to_dataframe(nparray, column_names, fps)

    def test_time_calculation_accuracy(self):
        """Test that time values are calculated accurately."""
        n_samples = 1000
        nparray = np.random.rand(n_samples, 5)
        column_names = ["a", "b", "c", "d", "e"]
        fps = 120.0

        df = from_array_to_dataframe(nparray, column_names, fps)

        # Check time values
        expected_time = np.arange(n_samples, dtype=np.float64) / fps
        np.testing.assert_array_almost_equal(df["time"].to_numpy(), expected_time)

    def test_column_order_xyz(self):
        """Test that columns are ordered correctly (x, y, z per tracker)."""
        nparray = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        column_names = ["A", "B", "C"]
        fps = 100.0

        df = from_array_to_dataframe(nparray, column_names, fps)

        expected_cols = [
            "time",
            "A_x",
            "A_y",
            "A_z",
            "B_x",
            "B_y",
            "B_z",
            "C_x",
            "C_y",
            "C_z",
        ]
        assert list(df.columns) == expected_cols


class TestFromDataframeToArray:
    """Test suite for from_dataframe_to_array function."""

    def test_3d_data_from_dataframe(self):
        """Test conversion from DataFrame with _x, _y, _z columns to 3D array."""
        data = {
            "time": [0.0, 0.01, 0.02],
            "tracker1_x": [1.0, 2.0, 3.0],
            "tracker1_y": [4.0, 5.0, 6.0],
            "tracker1_z": [7.0, 8.0, 9.0],
            "tracker2_x": [10.0, 11.0, 12.0],
            "tracker2_y": [13.0, 14.0, 15.0],
            "tracker2_z": [16.0, 17.0, 18.0],
        }
        df = pd.DataFrame(data)

        time, nparray, column_names = from_dataframe_to_array(df)

        # Check time
        np.testing.assert_array_equal(time, [0.0, 0.01, 0.02])

        # Check shape
        assert nparray.shape == (3, 2, 3)

        # Check column names
        assert column_names == ["tracker1", "tracker2"]

        # Check values
        assert nparray[0, 0, 0] == 1.0
        assert nparray[0, 0, 1] == 4.0
        assert nparray[0, 0, 2] == 7.0
        assert nparray[2, 1, 2] == 18.0

    def test_2d_data_from_dataframe(self):
        """Test conversion from DataFrame with scalar columns to 2D array."""
        data = {
            "time": [0.0, 0.01],
            "joint1": [1.0, 2.0],
            "joint2": [3.0, 4.0],
            "joint3": [5.0, 6.0],
        }
        df = pd.DataFrame(data)

        time, nparray, column_names = from_dataframe_to_array(df)

        # Check time
        np.testing.assert_array_equal(time, [0.0, 0.01])

        # Check shape
        assert nparray.shape == (2, 3)

        # Check column names
        assert column_names == ["joint1", "joint2", "joint3"]

        # Check values
        assert nparray[0, 0] == 1.0
        assert nparray[1, 2] == 6.0

    def test_from_parquet_file(self):
        """Test reading from a Parquet file path."""
        # Create a temporary Parquet file
        data = {
            "time": [0.0, 0.01],
            "col1_x": [1.0, 2.0],
            "col1_y": [3.0, 4.0],
            "col1_z": [5.0, 6.0],
        }
        df_orig = pd.DataFrame(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")
            df_orig.to_parquet(file_path)

            time, nparray, column_names = from_dataframe_to_array(file_path)

            # Check time
            np.testing.assert_array_equal(time, [0.0, 0.01])

            # Check shape and values
            assert nparray.shape == (2, 1, 3)
            assert nparray[0, 0, 0] == 1.0
            assert nparray[1, 0, 2] == 6.0

            # Check column names
            assert column_names == ["col1"]

    def test_from_csv_file(self):
        """Test reading from a CSV file path."""
        # Create a temporary CSV file
        data = {
            "time": [0.0, 0.01],
            "col1_x": [1.0, 2.0],
            "col1_y": [3.0, 4.0],
            "col1_z": [5.0, 6.0],
        }
        df_orig = pd.DataFrame(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.csv")
            df_orig.to_csv(file_path, index=False)

            time, nparray, column_names = from_dataframe_to_array(file_path)

            # Check time
            np.testing.assert_array_equal(time, [0.0, 0.01])

            # Check shape and values
            assert nparray.shape == (2, 1, 3)
            assert nparray[0, 0, 0] == 1.0
            assert nparray[1, 0, 2] == 6.0

            # Check column names
            assert column_names == ["col1"]

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            from_dataframe_to_array("/nonexistent/path/file.parquet")

    def test_unsupported_file_format_read(self):
        """Test that unsupported file formats raise ValueError when reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "data.txt")
            # Create a dummy file
            with open(file_path, "w") as f:
                f.write("dummy")

            with pytest.raises(ValueError, match="Unsupported file format"):
                from_dataframe_to_array(file_path)

    def test_invalid_input_type(self):
        """Test that ValueError is raised for invalid input types."""
        with pytest.raises(
            ValueError, match="must be a file path or a pandas DataFrame"
        ):
            from_dataframe_to_array(12345)

    def test_roundtrip_3d(self):
        """Test full roundtrip: array -> dataframe -> array for 3D data."""
        original_array = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        column_names = ["marker_a", "marker_b"]
        fps = 100.0

        # Convert to DataFrame
        df = from_array_to_dataframe(original_array, column_names, fps)

        # Convert back to array
        time, recovered_array, recovered_names = from_dataframe_to_array(df)

        # Check shape
        assert recovered_array.shape == original_array.shape

        # Check values (note: dtype might differ, so use allclose)
        np.testing.assert_allclose(recovered_array, original_array, rtol=1e-6)

        # Check time
        expected_time = np.arange(2, dtype=np.float64) / fps
        np.testing.assert_array_almost_equal(time, expected_time)

        # Check column names
        assert recovered_names == column_names

    def test_roundtrip_2d(self):
        """Test full roundtrip: array -> dataframe -> array for 2D data."""
        original_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        column_names = ["j1", "j2", "j3"]
        fps = 50.0

        # Convert to DataFrame
        df = from_array_to_dataframe(original_array, column_names, fps)

        # Convert back to array
        time, recovered_array, recovered_names = from_dataframe_to_array(df)

        # Check shape
        assert recovered_array.shape == original_array.shape

        # Check values
        np.testing.assert_allclose(recovered_array, original_array, rtol=1e-6)

        # Check column names
        assert recovered_names == column_names

    def test_roundtrip_with_file(self):
        """Test full roundtrip with Parquet file I/O."""
        original_array = np.random.rand(50, 10, 3)
        column_names = [f"marker_{i}" for i in range(10)]
        fps = 120.0

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "roundtrip.parquet")

            # Save to file
            from_array_to_dataframe(original_array, column_names, fps, file_path)

            # Load from file
            time, recovered_array, recovered_names = from_dataframe_to_array(file_path)

            # Check shape and values
            assert recovered_array.shape == original_array.shape
            np.testing.assert_allclose(recovered_array, original_array, rtol=1e-6)

            # Check column names
            assert recovered_names == column_names

    def test_roundtrip_with_csv_file(self):
        """Test full roundtrip with CSV file I/O."""
        original_array = np.random.rand(50, 10, 3)
        column_names = [f"marker_{i}" for i in range(10)]
        fps = 120.0

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "roundtrip.csv")

            # Save to file
            from_array_to_dataframe(original_array, column_names, fps, file_path)

            # Load from file
            time, recovered_array, recovered_names = from_dataframe_to_array(file_path)

            # Check shape and values
            assert recovered_array.shape == original_array.shape
            np.testing.assert_allclose(recovered_array, original_array, rtol=1e-6)

            # Check column names
            assert recovered_names == column_names

    def test_roundtrip_2d_with_csv(self):
        """Test roundtrip with 2D data and CSV format."""
        original_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        column_names = ["j1", "j2", "j3"]
        fps = 50.0

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "qpos.csv")

            # Convert to CSV
            from_array_to_dataframe(original_array, column_names, fps, file_path)

            # Load from CSV
            time, recovered_array, recovered_names = from_dataframe_to_array(file_path)

            # Check shape and values
            assert recovered_array.shape == original_array.shape
            np.testing.assert_allclose(recovered_array, original_array, rtol=1e-6)

            # Check column names
            assert recovered_names == column_names

    def test_mixed_column_suffixes_treated_as_scalar(self):
        """Test that mixed suffixes (some _x, some not) are treated as scalar data."""
        data = {
            "time": [0.0, 0.01],
            "col1_x": [1.0, 2.0],
            "col2": [3.0, 4.0],  # No suffix
            "col3_z": [5.0, 6.0],
        }
        df = pd.DataFrame(data)

        time, nparray, column_names = from_dataframe_to_array(df)

        # Should be treated as 2D array since not all have _x/_y/_z
        assert nparray.shape == (2, 3)

        # Check column names
        assert column_names == ["col1_x", "col2", "col3_z"]

    def test_dtype_preservation(self):
        """Test that data types are handled correctly."""
        nparray = np.array([[[1.5, 2.5, 3.5]]], dtype=np.float32)
        column_names = ["tracker"]
        fps = 100.0

        df = from_array_to_dataframe(nparray, column_names, fps)
        time, recovered, recovered_names = from_dataframe_to_array(df)

        # Check that values are preserved (dtype conversion is acceptable)
        np.testing.assert_allclose(recovered, nparray, rtol=1e-6)

        # Check column names
        assert recovered_names == column_names

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame (edge case)."""
        data = {"time": [], "col1": []}
        df = pd.DataFrame(data)

        time, nparray, column_names = from_dataframe_to_array(df)

        assert len(time) == 0
        assert nparray.shape[0] == 0

        # Check column names
        assert column_names == ["col1"]


class TestValidateDataframeFile:
    """Test suite for validate_dataframe_file function."""

    def test_valid_parquet_3d_data(self):
        """Test validation of a valid parquet file with 3D data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create valid 3D data
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "marker1_x": [1.0, 2.0, 3.0],
                    "marker1_y": [4.0, 5.0, 6.0],
                    "marker1_z": [7.0, 8.0, 9.0],
                    "marker2_x": [10.0, 11.0, 12.0],
                    "marker2_y": [13.0, 14.0, 15.0],
                    "marker2_z": [16.0, 17.0, 18.0],
                }
            )
            df.to_parquet(file_path)

            # Should not raise any exception
            assert validate_dataframe_file(file_path) is True

    def test_valid_csv_3d_data(self):
        """Test validation of a valid CSV file with 3D data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.csv")

            # Create valid 3D data
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "marker1_x": [1.0, 2.0, 3.0],
                    "marker1_y": [4.0, 5.0, 6.0],
                    "marker1_z": [7.0, 8.0, 9.0],
                }
            )
            df.to_csv(file_path, index=False)

            # Should not raise any exception
            assert validate_dataframe_file(file_path) is True

    def test_valid_scalar_data(self):
        """Test validation of valid scalar (non-3D) data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create valid scalar data
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "joint1": [1.0, 2.0, 3.0],
                    "joint2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            # Should not raise any exception
            assert validate_dataframe_file(file_path) is True

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_dataframe_file("/nonexistent/path/to/file.parquet")

    def test_unsupported_file_format(self):
        """Test that ValueError is raised for unsupported file formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")

            # Create a text file
            with open(file_path, "w") as f:
                f.write("some text")

            with pytest.raises(ValueError, match="Unsupported file format"):
                validate_dataframe_file(file_path)

    def test_corrupted_file(self):
        """Test that ValueError is raised for corrupted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create a corrupted file
            with open(file_path, "w") as f:
                f.write("corrupted data")

            with pytest.raises(ValueError, match="Failed to open file"):
                validate_dataframe_file(file_path)

    def test_too_few_columns(self):
        """Test that ValueError is raised when DataFrame has fewer than 2 columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with only 1 column (time only)
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(ValueError, match="must contain at least 2 columns"):
                validate_dataframe_file(file_path)

    def test_too_few_rows(self):
        """Test that ValueError is raised when DataFrame has fewer than 2 rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with only 1 row
            df = pd.DataFrame(
                {
                    "time": [0.0],
                    "col1": [1.0],
                    "col2": [2.0],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(ValueError, match="must contain at least 2 rows"):
                validate_dataframe_file(file_path)

    def test_non_numeric_time_column(self):
        """Test that ValueError is raised when time column is not numeric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with non-numeric time
            df = pd.DataFrame(
                {
                    "time": ["a", "b", "c"],
                    "col1": [1.0, 2.0, 3.0],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(
                ValueError, match="First column \\(time\\) must be numeric"
            ):
                validate_dataframe_file(file_path)

    def test_infinite_time_values(self):
        """Test that ValueError is raised when time column contains infinite values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with infinite time values
            df = pd.DataFrame(
                {
                    "time": [0.0, np.inf, 0.02],
                    "col1": [1.0, 2.0, 3.0],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(
                ValueError, match="Time column contains infinite values"
            ):
                validate_dataframe_file(file_path)

    def test_non_monotonic_time(self):
        """Test that ValueError is raised when time is not strictly increasing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with non-monotonic time
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.02, 0.01],  # Not strictly increasing
                    "col1": [1.0, 2.0, 3.0],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(
                ValueError, match="Time column must be strictly increasing"
            ):
                validate_dataframe_file(file_path)

    def test_non_numeric_data_column(self):
        """Test that ValueError is raised when data column is not numeric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with non-numeric data column
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "col1": ["a", "b", "c"],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(ValueError, match="Column 'col1' is not numeric"):
                validate_dataframe_file(file_path)

    def test_infinite_data_values(self):
        """Test that ValueError is raised when data column contains infinite values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with infinite data values
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "col1": [1.0, np.inf, 3.0],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(
                ValueError, match="Column 'col1' contains infinite values"
            ):
                validate_dataframe_file(file_path)

    def test_invalid_3d_column_count(self):
        """Test that ValueError is raised when 3D data column count is not multiple of 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with invalid 3D structure (4 columns instead of 3 or 6)
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "marker1_x": [1.0, 2.0, 3.0],
                    "marker1_y": [4.0, 5.0, 6.0],
                    "marker1_z": [7.0, 8.0, 9.0],
                    "marker2_x": [10.0, 11.0, 12.0],  # Missing _y and _z
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(ValueError, match="column count is not a multiple of 3"):
                validate_dataframe_file(file_path)

    def test_invalid_3d_column_order(self):
        """Test that ValueError is raised when 3D data columns are not in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with wrong order (x, z, y instead of x, y, z)
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "marker1_x": [1.0, 2.0, 3.0],
                    "marker1_z": [7.0, 8.0, 9.0],  # Wrong order
                    "marker1_y": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            with pytest.raises(ValueError, match="Invalid column ordering for 3D data"):
                validate_dataframe_file(file_path)

    def test_nan_values_allowed_in_time(self):
        """Test that NaN values in time column are allowed (but skipped for monotonicity check)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with NaN in time
            df = pd.DataFrame(
                {
                    "time": [0.0, np.nan, 0.02, 0.03],
                    "col1": [1.0, 2.0, 3.0, 4.0],
                    "col2": [5.0, 6.0, 7.0, 8.0],
                }
            )
            df.to_parquet(file_path)

            # Should not raise exception
            assert validate_dataframe_file(file_path) is True

    def test_nan_values_allowed_in_data(self):
        """Test that NaN values in data columns are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with NaN in data
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "col1": [1.0, np.nan, 3.0],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            # Should not raise exception
            assert validate_dataframe_file(file_path) is True

    def test_mixed_3d_and_scalar_columns_rejected(self):
        """Test that mixing 3D and scalar columns is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with mixed column types
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "marker1_x": [1.0, 2.0, 3.0],
                    "marker1_y": [4.0, 5.0, 6.0],
                    "marker1_z": [7.0, 8.0, 9.0],
                    "scalar_col": [10.0, 11.0, 12.0],  # Doesn't end with _x/_y/_z
                }
            )
            df.to_parquet(file_path)

            # This should be invalid because not all columns have _x/_y/_z suffixes
            # The function checks: all columns must have suffixes, or none should
            # Since scalar_col doesn't have suffix, this is valid as scalar data
            assert validate_dataframe_file(file_path) is True

    def test_large_valid_dataset(self):
        """Test validation with a large dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create large dataset
            n_samples = 10000
            n_markers = 50
            time = np.arange(n_samples) / 100.0

            data = {"time": time}
            for i in range(n_markers):
                data[f"marker{i}_x"] = np.random.rand(n_samples)
                data[f"marker{i}_y"] = np.random.rand(n_samples)
                data[f"marker{i}_z"] = np.random.rand(n_samples)

            df = pd.DataFrame(data)
            df.to_parquet(file_path)

            # Should not raise exception
            assert validate_dataframe_file(file_path) is True

    def test_minimum_valid_dataset(self):
        """Test validation with minimum valid dataset (2 rows, 2 columns)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create minimum valid dataset (time + 1 data column)
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01],
                    "col1": [1.0, 2.0],
                }
            )
            df.to_parquet(file_path)

            # Should not raise exception
            assert validate_dataframe_file(file_path) is True

    def test_time_with_duplicates_not_allowed(self):
        """Test that duplicate time values are not allowed (must be strictly increasing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with duplicate time values
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.01, 0.02],  # Duplicate at 0.01
                    "col1": [1.0, 2.0, 3.0, 4.0],
                    "col2": [5.0, 6.0, 7.0, 8.0],
                }
            )
            df.to_parquet(file_path)

            # Should raise exception (time must be strictly increasing)
            with pytest.raises(
                ValueError, match="Time column must be strictly increasing"
            ):
                validate_dataframe_file(file_path)

    def test_negative_time_values_allowed(self):
        """Test that negative time values are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with negative time values
            df = pd.DataFrame(
                {
                    "time": [-1.0, -0.5, 0.0, 0.5],
                    "col1": [1.0, 2.0, 3.0, 4.0],
                    "col2": [5.0, 6.0, 7.0, 8.0],
                }
            )
            df.to_parquet(file_path)

            # Should not raise exception
            assert validate_dataframe_file(file_path) is True

    def test_different_3d_base_names(self):
        """Test validation with different base names in 3D data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            # Create DataFrame with various base names
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "head_x": [1.0, 2.0, 3.0],
                    "head_y": [4.0, 5.0, 6.0],
                    "head_z": [7.0, 8.0, 9.0],
                    "left_hand_x": [10.0, 11.0, 12.0],
                    "left_hand_y": [13.0, 14.0, 15.0],
                    "left_hand_z": [16.0, 17.0, 18.0],
                }
            )
            df.to_parquet(file_path)

            # Should not raise exception
            assert validate_dataframe_file(file_path) is True

    def test_return_df_false(self):
        """Test that validate_dataframe_file returns True when return_df=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "col1": [1.0, 2.0, 3.0],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            df.to_parquet(file_path)

            result = validate_dataframe_file(file_path, return_df=False)
            assert result is True

    def test_return_df_true(self):
        """Test that validate_dataframe_file returns DataFrame when return_df=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            original_df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "col1": [1.0, 2.0, 3.0],
                    "col2": [4.0, 5.0, 6.0],
                }
            )
            original_df.to_parquet(file_path)

            result = validate_dataframe_file(file_path, return_df=True)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert list(result.columns) == ["time", "col1", "col2"]
            pd.testing.assert_frame_equal(result, original_df)

    def test_return_df_default(self):
        """Test that validate_dataframe_file defaults to return_df=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")

            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "col1": [1.0, 2.0, 3.0],
                }
            )
            df.to_parquet(file_path)

            # Call without return_df parameter
            result = validate_dataframe_file(file_path)
            assert result is True
