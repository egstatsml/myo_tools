"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import numpy as np

from myo_tools.mjs.marker.marker_api import get_marker_names
from myo_tools.utils.file_ops.dataframe_utils import (
    from_array_to_dataframe,
    from_dataframe_to_array,
)
from myo_tools.utils.file_ops.xml_utils import save_xml_to_file
from myo_tools.utils.mocap_ops.mocap_utils import load_trackers_and_markerset
from myo_tools.utils.tensor_ops.quat_utils import quat2euler


def load_and_convert_trackers_and_markerset(
    input_trackers_file_path: str,
    input_markerset_file_path: str,
    output_trackers_file_path: str,
    output_markerset_file_path: str,
    mocap_scale: int = 1000,
    clip_length: int = -1,
):
    """
    Load trackers and markerset from input files, process them, and save to output files.

    Args:
        input_trackers_file_path (str): Path to the input trackers file (e.g., .c3d).
        input_markerset_file_path (str): Path to the input markerset .xml file.
        output_trackers_file_path (str): Path to save the processed trackers .parquet file.
        output_markerset_file_path (str): Path to save the processed markerset .xml file.
        mocap_scale (int): Scale factor for mocap data. Defaults to 1000 (i.e. from mm to m).
        clip_length (int): Length of the clip to use. Defaults to -1 (use full length).
    """

    # Load paired trackers and markerset
    motion_data_subject_list, markerset, framerate = load_trackers_and_markerset(
        trackers_file_path=input_trackers_file_path,
        markerset_handle=input_markerset_file_path,
        mocap_scale=mocap_scale,
        clip_length=clip_length,
    )

    # For the SDK we currently only support single-subject trackers data, and we
    # are not splitting the data in multiple chunks (see previous function call).
    trackers = motion_data_subject_list[0][0]  # (num_frames, num_markers, 3)

    # Save processed trackers to parquet
    marker_names = get_marker_names(markerset)
    from_array_to_dataframe(
        trackers,
        marker_names,
        framerate,
        output_trackers_file_path,
    )

    # Save processed markerset to XML
    save_xml_to_file(markerset, output_markerset_file_path)


def save_qpos(
    all_qpos,
    all_qpos_colnames,
    framerate,
    output_qpos_file_path,
):
    """
    Save joint positions (qpos) to a parquet or csv file.

    Args:
        all_qpos (np.ndarray): Array of joint positions with shape (num_frames, num_joints).
        all_qpos_colnames (list): List of joint position column names.
        framerate (float): Frame rate of the motion data.
        output_qpos_file_path (str): Path to save the qpos file.
    """
    from_array_to_dataframe(
        all_qpos,
        all_qpos_colnames,
        framerate,
        output_qpos_file_path,
    )


def from_qpos_to_joint_angles(
    qpos_dataframe_handle,
    output_angles_file_path=None,
):
    """
    Convert a qpos dataframe to joint angles, and optionally save them to a parquet or csv file.

    Args:
        qpos_dataframe_handle (str | pd.DataFrame): Path to the qpos file or a pandas DataFrame.
        output_angles_file_path (str): Path to save the joint angles file. If None, the file is not saved.
    """
    time_axis, qpos, joint_qpos_names = from_dataframe_to_array(qpos_dataframe_handle)
    framerate = np.mean(1.0 / np.diff(time_axis))
    use_ball_shoulders = False
    idxs_columns_to_delete = []
    for idx, name in enumerate(joint_qpos_names):
        if "shoulder_" in name:
            use_ball_shoulders = True
        if (
            "patella_" in name
            or "abs_" in name
            or "_tx" in name
            or "_ty" in name
            or "_tz" in name
            or "root" in name
        ):
            idxs_columns_to_delete.append(idx)
    joint_angles = np.delete(qpos, idxs_columns_to_delete, axis=1)
    joint_angle_names = [
        name
        for idx, name in enumerate(joint_qpos_names)
        if idx not in idxs_columns_to_delete
    ]
    if use_ball_shoulders:
        sr_id = joint_angle_names.index("shoulder_r_qw")
        sl_id = joint_angle_names.index("shoulder_l_qw")
        sr_ids = [sr_id, sr_id + 1, sr_id + 2, sr_id + 3]
        sl_ids = [sl_id, sl_id + 1, sl_id + 2, sl_id + 3]
        euler_shoulder_r = np.zeros((joint_angles.shape[0], 3))
        euler_shoulder_l = np.zeros((joint_angles.shape[0], 3))
        for i in range(joint_angles.shape[0]):
            quat_r = joint_angles[i, sr_ids]
            quat_l = joint_angles[i, sl_ids]
            euler_shoulder_r[i, :] = quat2euler(quat_r)
            euler_shoulder_l[i, :] = quat2euler(quat_l)
        # delete quaternion columns and add euler angles columns
        joint_angles = np.delete(joint_angles, sr_ids, axis=1)
        joint_angles = np.insert(joint_angles, sr_id, euler_shoulder_r.T, axis=1)
        sl_ids_shift = [
            idx - 1 for idx in sl_ids
        ]  # 4 cols deleted, 3 added --> shift = -1
        joint_angles = np.delete(joint_angles, sl_ids_shift, axis=1)
        joint_angles = np.insert(joint_angles, sl_id - 1, euler_shoulder_l.T, axis=1)
        # replace quaternion names with euler angle names
        joint_angle_names[sr_id : sr_id + 4] = [
            "shoulder_abdu_r",
            "humerus_arot_r",
            "shoulder_flex_r",
        ]
        joint_angle_names[sl_id - 1 : sl_id + 3] = [
            "shoulder_abdu_l",
            "humerus_arot_l",
            "shoulder_flex_l",
        ]
    # convert all the angles from radians to degrees
    joint_angles = np.rad2deg(joint_angles)

    return from_array_to_dataframe(
        joint_angles,
        joint_angle_names,
        framerate,
        output_angles_file_path,
    )
