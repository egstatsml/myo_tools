"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Utilities for loading C3D motion capture files into numpy arrays.

import c3d  # https://c3d.readthedocs.io/en/stable/
import ezc3d  # https://pyomeca.github.io/Documentation/ezc3d/
import numpy as np

from myo_tools.utils.log_ops import logger

logger = logger.getLogger("myo_tools.utils.c3d_utils")


def ezc3d_loader(c3d_file_path: str):
    """
    Attempt to load a C3D file using the ezc3d library.

    Args:
        c3d_file_path (str): Path to the .c3d file

    Returns:
        motion_data (np.ndarray or None): Shape is (num_frames, num_markers, 3)
        marker_names (list or None): Marker names
        framerate (float or None): Frame rate extracted from C3D file
        success (bool): True if successful, else False
    """
    try:
        c3d_data = ezc3d.c3d(c3d_file_path)
        points = c3d_data["data"]["points"][:3, :, :]
        motion_data = np.transpose(points, (2, 1, 0))

        marker_names = c3d_data["parameters"]["POINT"]["LABELS"]["value"]
        framerate = c3d_data["parameters"]["POINT"]["RATE"]["value"][0]

        return motion_data, marker_names, framerate, True
    except Exception:
        logger.error("could not load c3d file with ezc3d")
        return None, None, None, False


def c3dpy_loader(c3d_file_path):
    """
    Attempt to load a C3D file using python-c3d (c3d.py).

    Args:
        c3d_file_path (str): Path to the .c3d file

    Returns:
        motion_data (np.ndarray or None): Shape is (num_frames, num_markers, 3)
        marker_names (list or None): Marker names
        framerate (float or None): Frame rate extracted from C3D file
        success (bool): True if successful, else False
    """
    try:
        with open(c3d_file_path, "rb") as handle:
            reader = c3d.Reader(handle)
            all_frames = []

            for _, points, _ in reader.read_frames():
                # points is an array of shape (num_markers, 4)
                # (x, y, z, residual error for each marker).
                all_frames.append(points)

            motion_data = np.array(all_frames)  # (num_frames, num_markers, 4)

            motion_data = motion_data[:, :, :3]  # keep only x, y, z
            marker_names = reader.point_labels
            framerate = reader.header.frame_rate

        return motion_data, marker_names, framerate, True
    except Exception:
        logger.error("could not load c3d file with c3dpy")
        return None, None, None, False


def from_c3d_to_numpy(
    c3d_file_path: str,
):
    """
    Load a C3D file, using ezc3d first. If that fails, fallback to python-c3d.

    Args:
        c3d_file_path (str): Path to the .c3d file

    Returns:
        motion_data (np.ndarray): Shape is (num_frames, num_trackers, 3)
        tracker_names (list): Tracker names
        framerate (float): Frame rate extracted from C3D file
    """
    motion_data, tracker_names, framerate, success = ezc3d_loader(c3d_file_path)
    if not success:
        motion_data, tracker_names, framerate, success = c3dpy_loader(c3d_file_path)

    if not success:
        raise Exception("Could not load c3d file")

    return motion_data, tracker_names, framerate
