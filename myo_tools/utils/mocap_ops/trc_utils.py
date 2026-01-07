"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Utilities for loading TRC motion capture files into numpy arrays.

import numpy as np

from myo_tools.utils.log_ops import logger

logger = logger.getLogger("myo_tools.utils.trc_utils")


def trc_loader(trc_file_path: str, rotate_yup_to_zup: bool = True):
    """
    Attempt to load a TRC file.

    Args:
        trc_file_path (str): Path to the .trc file
        rotate_yup_to_zup (bool): Whether to rotate the axes from osim to MuJoCo format.
            Default is True.

    Returns:
        motion_data (np.ndarray or None): Shape is (num_frames, num_markers, 3)
        marker_names (list or None): Marker names
        framerate (float or None): Frame rate extracted from TRC file
        success (bool): True if successful, else False
    """
    try:
        with open(trc_file_path, "r") as f:
            lines = f.readlines()

        # Parse header line 1 (contains file metadata column names)
        header_line_idx = 0
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            if "DataRate" in line or "CameraRate" in line:
                header_line_idx = i
                break

        # Parse header keys and values
        header_keys = lines[header_line_idx].strip().split("\t")
        header_values = lines[header_line_idx + 1].strip().split("\t")

        # Create a mapping of header keys to values
        header_dict = {}
        for i, key in enumerate(header_keys):
            if i < len(header_values):
                header_dict[key] = header_values[i]

        # Extract metadata using the mapping
        framerate = float(
            header_dict.get("DataRate", header_dict.get("CameraRate", 0.0))
        )

        # Parse marker names line (next line after header values)
        marker_line = lines[header_line_idx + 2].strip().split("\t")

        # Remove empty strings and Frame#/Time columns from marker names
        marker_names = []
        for name in marker_line[2:]:  # Skip first two columns (Frame# and Time)
            if name and name not in ["", "Frame#", "Time"]:
                marker_names.append(name)

        # Data starts after the coordinate labels line (header_line_idx + 4)
        data_start_idx = header_line_idx + 4

        # Parse data lines (starting from line 5, index 4)
        motion_data = []
        for line in lines[data_start_idx:]:
            if line.strip():  # Skip empty lines
                values = line.strip().split("\t")
                # Skip first two columns (Frame# and Time)
                coords = [float(v) for v in values[2:]]

                # Reshape into (num_markers, 3)
                frame_data = []
                for i in range(0, len(coords), 3):
                    if i + 2 < len(coords):
                        frame_data.append([coords[i], coords[i + 1], coords[i + 2]])

                motion_data.append(frame_data)

        motion_data = np.array(motion_data)  # Shape: (num_frames, num_markers, 3)

        if rotate_yup_to_zup:
            # Rotate axes from OpenSim (Y-up) to MuJoCo (Z-up)
            # Rotate from Y-up to Z-up: R_x(-90°)
            # This rotates around X-axis by -90 degrees
            # [x, y, z] -> [x, -z, y]
            rotation_matrix = np.array(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
            )
            motion_data = motion_data @ rotation_matrix.T

        return motion_data, marker_names, framerate, True

    except Exception:
        logger.error("could not load trc file")
        return None, None, None, False


def from_trc_to_numpy(
    trc_file_path: str,
    rotate_yup_to_zup: bool = True,
):
    """
    Load a TRC file.
    Then fill missing values in the resulting motion data with a forward fill.
    Finally, apply the mocap scale and clip the length if needed.
    Args:
        trc_file_path (str): Path to the .trc file
        rotate_yup_to_zup (bool): Whether to rotate the axes from osim to MuJoCo format.
    Returns:
        motion_data (np.ndarray): Shape is (num_frames, num_trackers, 3)
        tracker_names (list): Tracker names
        framerate (float): Frame rate extracted from TRC file
    """
    motion_data, tracker_names, framerate, success = trc_loader(
        trc_file_path, rotate_yup_to_zup=rotate_yup_to_zup
    )

    if not success:
        raise Exception("Could not load trc file")

    return motion_data, tracker_names, framerate
