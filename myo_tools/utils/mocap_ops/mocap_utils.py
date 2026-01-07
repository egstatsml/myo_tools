"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Utilities for loading and processing motion capture (mocap) data from various file formats,
# including .c3d, .trc, .csv, and .parquet. The functions handle loading trackers, cleaning motion data by filling gaps
# and removing static trackers, and ensuring consistency between the motion data and a given markerset.

import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from myo_tools.utils.file_ops.dataframe_utils import from_dataframe_to_array
from myo_tools.utils.file_ops.xml_utils import load_markerset
from myo_tools.utils.log_ops import logger
from myo_tools.utils.mocap_ops.c3d_utils import from_c3d_to_numpy
from myo_tools.utils.mocap_ops.trc_utils import from_trc_to_numpy

logger = logger.getLogger("myo_tools.utils.mocap_utils")


def load_trackers(
    trackers_file_path, mocap_scale, clip_length, rotate_yup_to_zup=False
):
    """
    Load trackers data from a file, apply scaling and clipping.
    Args:
        trackers_file_path (str): Path to the trackers file (e.g., .c3d)
        mocap_scale (int): Scale factor for mocap data
        clip_length (int): Length of the clip to use
        rotate_yup_to_zup (bool): Whether to rotate the axes from osim to MuJoCo format. Default is False.
    Returns:
        motion_data (np.ndarray): Loaded and processed motion data
        tracker_names (list): List of tracker names
        framerate (float): Frame rate of the motion data
    """
    # Load trackers file
    if trackers_file_path.lower().endswith(".c3d"):
        motion_data, tracker_names, framerate = from_c3d_to_numpy(trackers_file_path)

    elif trackers_file_path.lower().endswith(".trc"):
        motion_data, tracker_names, framerate = from_trc_to_numpy(trackers_file_path)

    elif trackers_file_path.lower().endswith(
        ".csv"
    ) or trackers_file_path.lower().endswith(".parquet"):
        time_axis, motion_data, tracker_names = from_dataframe_to_array(
            trackers_file_path
        )
        framerate = np.mean(1.0 / np.diff(time_axis))
    else:
        raise Exception(f"Unsupported trackers file format: {trackers_file_path}")

    # Consider only motion data with known corresponding tracker name
    if motion_data.shape[1] > len(tracker_names):
        motion_data = motion_data[:, : len(tracker_names), :]

    # Apply mocap scale
    motion_data /= mocap_scale

    # Optionally rotate axes from Y-up to Z-up
    if rotate_yup_to_zup:
        logger.info("Rotating mocap data from Y-up to Z-up")
        # Rotate axes from OpenSim (Y-up) to MuJoCo (Z-up)
        # Rotate from Y-up to Z-up: R_x(-90°)
        # This rotates around X-axis by -90 degrees
        # [x, y, z] -> [x, -z, y]
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        motion_data = motion_data @ rotation_matrix.T

    # Clip length if needed
    if clip_length > 0:
        motion_data = motion_data[:clip_length]

    return motion_data, tracker_names, framerate


def load_trackers_and_markerset(
    trackers_file_path: str,
    markerset_handle: str | ET.Element,
    mocap_scale: int = 1000,
    rotate_yup_to_zup: bool = False,
    clip_length: int = -1,
    chunk_size: int = -1,
    allow_multisubject: bool = False,
):
    """
    Load a trackers file and the markerset. Then filter the motion data and the markerset to be consistent
    between each other. If allow_multisubject is True, handle multi-subject trackers files. If chunk_size
    is provided, split the motion data into chunks of the given size.

    Args:
        trackers_file_path (str): Path to the trackers file (.c3d, .trc, .csv, .parquet)
        markerset_handle (str | ET.Element): The markerset definition, either as a file path or an XML element.
        mocap_scale (int, optional): Scale factor for mocap data. Defaults to 1000 (i.e. from mm to m).
        rotate_yup_to_zup (bool, optional): Whether to rotate the axes from osim to MuJoCo format. Defaults to False.
        clip_length (int): Length of the clip to use. Defaults to -1 (use full length).
        chunk_size (int): Size of chunks to split the motion data into. Defaults to -1 (use full length).
        allow_multisubject (bool, optional): If True, allows loading trackers files with multiple subjects.
            Defaults to False.

    Returns:
        (motion_data_list, tracker_names, framerate):
            motion_data_list (list): A list of list of motion_data (one list per subject containing one list per chunk):
                motion_data (np.ndarray): shape (num_frames, num_trackers, 3)
            tracker_names (list): List of tracker names
            framerate (float): Frame rate
    """
    # Load trackers data
    motion_data, tracker_names, framerate = load_trackers(
        trackers_file_path, mocap_scale, clip_length, rotate_yup_to_zup
    )

    # Load markerset
    markerset = load_markerset(markerset_handle)
    markerset_names = [m.get("name") for m in markerset]

    # Remove trackers that are NaN for more than 80% of the frames
    idxs_valid_trackers = []
    for i in range(motion_data.shape[1]):
        if np.sum(np.isnan(motion_data[:, i, 0])) / motion_data.shape[0] <= 0.8:
            idxs_valid_trackers.append(i)
    motion_data = motion_data[:, idxs_valid_trackers, :]
    tracker_names = [tracker_names[i] for i in idxs_valid_trackers]

    # Handle multi-subject trackers files
    subjects_list = []
    if allow_multisubject:
        for t in tracker_names:
            if ":" in t:
                subject = t.split(":")[0]
                subjects_list.append(subject)
        subjects_list = list(set(subjects_list))
    if len(subjects_list) == 0:
        subjects_list = ["single_subject"]

    motion_data_subject_list = []
    all_tracker_names_cleaned = []
    for subject in subjects_list:
        # we have a markerset and we need to ensure the trackers data gets filtered
        duplicates = []  # duplicated markers in the trackers file (not allowed)
        absences = []  # markerset markers not found in the trackers file (warned)
        mapping = []  # mapping from trackers markers to markerset markers
        tracker_names_cleaned = []
        for m in markerset_names:
            m_found = False
            for it, t in enumerate(tracker_names):
                t_splitted = re.findall(r"[A-Za-z0-9]+(?:[_-][0-9]+)?", t)
                if subject != "single_subject" and subject not in t_splitted:
                    continue
                if m in t_splitted or m == t:
                    if m_found:
                        duplicates.append(m)
                    mapping.append(it)
                    tracker_names_cleaned.append(m)
                    m_found = True
            if not m_found:
                absences.append(m)

        if len(duplicates):
            raise Exception(
                f"Markers {', '.join(list(set(duplicates)))} found multiple times in the trackers file"
            )
        if len(absences):
            warnings.warn(
                f"Markers {', '.join(absences)} not found in the trackers file"
            )

        # Filter motion data with the obtained mapping
        motion_data = motion_data[:, mapping, :3]

        # Split motion data into chunks if needed
        chunk_size = chunk_size if chunk_size > 0 else motion_data.shape[0]
        motion_data_chunk_list = [
            motion_data[i : i + chunk_size]
            for i in range(0, motion_data.shape[0], chunk_size)
        ]
        motion_data_subject_list.append(motion_data_chunk_list)
        all_tracker_names_cleaned += tracker_names_cleaned

    # Remove from the markerset the markers that are not in the trackers file
    tracker_names_cleaned = list(set(tracker_names_cleaned))

    idxs_markers_to_delete = []
    for i, marker in enumerate(markerset):
        if marker.attrib.get("name") not in tracker_names_cleaned:
            idxs_markers_to_delete.append(i)
    for idx in sorted(idxs_markers_to_delete, reverse=True):
        del markerset[idx]

    return motion_data_subject_list, markerset, framerate
