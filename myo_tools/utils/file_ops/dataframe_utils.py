"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import os

import numpy as np
import pandas as pd


def validate_dataframe_file(dataframe_file_path: str, return_df: bool = False):
    """
    Validate an input dataframe file intended for mocap or kinematic data.

    The function performs lightweight structural and semantic checks to ensure
    compatibility with ``from_dataframe_to_array``.

    Guarantees:
    - file exists and can be opened
    - dataframe has at least 2 columns (time + data)
    - dataframe has at least 2 rows
    - first column represents a numeric, strictly increasing time axis
    - remaining columns contain numeric values
    - NaN values are allowed
    - infinite values are not allowed
    - if 3D data is detected, columns follow the _x, _y, _z convention
      in the correct order
    """
    if not os.path.exists(dataframe_file_path):
        raise FileNotFoundError(f"File not found: {dataframe_file_path}")

    ext = os.path.splitext(dataframe_file_path)[1].lower()

    try:
        if ext == ".parquet":
            df = pd.read_parquet(dataframe_file_path)
        elif ext == ".csv":
            df = pd.read_csv(dataframe_file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        raise ValueError(f"Failed to open file: {e}")

    # -------- Basic shape checks
    if df.shape[1] < 2:
        raise ValueError("DataFrame must contain at least 2 columns (time + data).")

    if df.shape[0] < 2:
        raise ValueError("DataFrame must contain at least 2 rows.")

    # -------- Time column checks
    time_values = df.iloc[:, 0]

    if not pd.api.types.is_numeric_dtype(time_values):
        raise ValueError("First column (time) must be numeric.")

    if np.any(np.isinf(time_values.to_numpy())):
        raise ValueError("Time column contains infinite values.")

    time_clean = time_values.dropna().to_numpy()
    if len(time_clean) > 1 and not np.all(np.diff(time_clean) > 0):
        raise ValueError(
            "Time column must be strictly increasing (no repeated or decreasing values)."
        )

    # -------- Data columns checks
    data_cols = df.columns[1:]

    for col in data_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric.")

        if np.any(np.isinf(df[col].to_numpy())):
            raise ValueError(f"Column '{col}' contains infinite values.")

    # -------- Optional 3D structure checks
    has_xyz_suffixes = all(
        col.lower().endswith(("_x", "_y", "_z", ".x", ".y", ".z")) for col in data_cols
    )

    if has_xyz_suffixes:
        if len(data_cols) % 3 != 0:
            raise ValueError(
                "3D data detected (x,y,z suffixes) but column count is not a multiple of 3."
            )

        for i in range(0, len(data_cols), 3):
            base = data_cols[i][:-2]
            suffix = data_cols[i][-2:]  # either '_x' or '.x'
            expected = [
                f"{base}{suffix[0]}x",
                f"{base}{suffix[0]}y",
                f"{base}{suffix[0]}z",
            ]
            if list(data_cols[i : i + 3]) != expected:
                raise ValueError(
                    f"Invalid column ordering for 3D data: expected {expected}"
                )

    return df if return_df else True


def from_array_to_dataframe(
    nparray: np.ndarray,
    column_names: list[str],
    fps: float,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from a NumPy array and optionally write it to disk
    as a Parquet or CSV file.

    The resulting DataFrame always contains a first column named ``time``,
    computed from the provided sampling frequency as ``time[i] = i / fps``.
    Remaining columns encode either scalar signals (e.g. joints) or 3D signals
    (i.e. trackers).

    Supported input shapes:
    - (N_samples, N_trackers, 3):
        Each tracker is expanded into three columns with suffixes
        ``_x``, ``_y``, ``_z``. Columns are ordered as
        ``<name>_x, <name>_y, <name>_z`` for each tracker, and trackers are
        ordered according to ``column_names``.
    - (N_samples, N_joints):
        Each joint is mapped to a single column, ordered according to
        ``column_names``.

    Supported output formats (inferred from ``output_path`` extension):
    - ``.parquet``
    - ``.csv``

    Args:
        nparray (np.ndarray):
            Input NumPy array of shape (N, T, 3) or (N, J), where N is the
            number of samples.
        column_names (list[str]):
            Names of trackers or joints. Length must match T or J.
        fps (float):
            Sampling frequency in Hz, used to compute the time axis.
        output_path (str | None):
            Optional file path. If provided, the DataFrame is written to disk.
            Parent directories are created automatically if they do not exist.

    Returns:
        pd.DataFrame:
            DataFrame with shape (N, 1 + T*3) or (N, 1 + J). The first column
            is ``time``.

    Raises:
        ValueError:
            If the input array dimensionality is unsupported or the file
            extension is not recognized.
    """
    n_samples = nparray.shape[0]
    time_axis = np.arange(n_samples, dtype=np.float64) / fps

    if nparray.ndim == 3:
        n_items = nparray.shape[1]
        flat = nparray.reshape(n_samples, n_items * 3)
        cols = [f"{name}_{c}" for name in column_names for c in ("x", "y", "z")]
        df = pd.DataFrame(flat, columns=cols)
    elif nparray.ndim == 2:
        df = pd.DataFrame(nparray, columns=column_names)
    else:
        raise ValueError("nparray must have shape (N,T,3) or (N,J)")

    df.insert(0, "time", time_axis)

    if output_path is not None:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".parquet":
            df.to_parquet(output_path)
        elif ext == ".csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    return df


def from_dataframe_to_array(
    dataframe_handle: str | pd.DataFrame,
):
    """
    Convert a pandas DataFrame, or a DataFrame stored on disk, back into a
    NumPy array representation.

    The function assumes that:
    - the first column represents a time axis,
    - remaining columns represent either scalar signals or 3D signals.

    3D signals are detected by column name suffixes ``*x``, ``*y``, ``*z``.
    Where * can be either ``_`` or ``.``.
    When present, columns are assumed to be ordered as
    ``<name>*x, <name>*y, <name>*z`` for each item, and are reshaped back to
    an array of shape (N, T, 3). Otherwise, scalar data are returned as
    shape (N, J).

    Supported input formats (when a path is provided):
    - ``.parquet``
    - ``.csv``

    Args:
        dataframe_handle (str | pd.DataFrame):
            Either a file path pointing to a ``.parquet`` or ``.csv`` file,
            or an in-memory pandas DataFrame.

    Returns:
        - time_axis (np.ndarray):
            Array of shape (N,) containing the time axis.
        - nparray (np.ndarray):
            Reconstructed NumPy array of shape (N, T, 3) for 3D data
            or (N, J) for scalar data.
        - column_names (list[str]):
            Tracker or joint names, in the same order as in the reconstructed array.

    Raises:
        FileNotFoundError:
            If a file path is provided but does not exist.
        ValueError:
            If the input type or file extension is unsupported.
    """
    if isinstance(dataframe_handle, str):
        if not os.path.exists(dataframe_handle):
            raise FileNotFoundError(f"File not found: {dataframe_handle}")

        ext = os.path.splitext(dataframe_handle)[1].lower()
        if ext == ".parquet":
            df = pd.read_parquet(dataframe_handle)
        elif ext == ".csv":
            df = pd.read_csv(dataframe_handle)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    elif isinstance(dataframe_handle, pd.DataFrame):
        df = dataframe_handle
    else:
        raise ValueError("dataframe_handle must be a file path or a pandas DataFrame")

    time_axis = df.iloc[:, 0].to_numpy()
    data_cols = df.columns[1:]

    if all(
        col.lower().endswith(("_x", "_y", "_z", ".x", ".y", ".z")) for col in data_cols
    ):
        n_samples = len(df)
        n_items = len(data_cols) // 3
        arr = df[data_cols].to_numpy(dtype=np.float32)
        nparray = arr.reshape(n_samples, n_items, 3)

        # extract base names in order
        column_names = [data_cols[i][:-2] for i in range(0, len(data_cols), 3)]
    else:
        nparray = df[data_cols].to_numpy()
        column_names = list(data_cols)

    return time_axis, nparray, column_names
