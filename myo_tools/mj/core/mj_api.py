"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: This file provides APIs for creation and operations related to a mujoco model object

from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from myo_tools.utils.file_ops.import_utils import get_asset_dict
from myo_tools.utils.log_ops import logger
from myo_tools.utils.tensor_ops.tensor_utils import is_int_or_iterable_of_ints

logger = logger.getLogger("mj_api_logger")


def get_model(
    model_handle: str | bytes | mujoco.MjModel, asset_handle: str | dict | None = None
):
    """
    Constructs a Mujoco model from the provided simulation and asset handle.

    Args:
    - model_handle (str, bytes, Path or mujoco.MjModel):
        The handle to the simulation file or Mujoco model.
    - asset_handle (str or dict, optional):
        The handle to the assets directory, or a dictionary
        of pre-loaded assets, of the form `{filename: bytestring}`.
        If present, assets in this dictionary will be used before
        attempting to load them from the filesystem.

    Returns:
    - mj_model: mujoco.MjModel
        The constructed Mujoco model.
    - asset_dict: dict
        A dictionary of pre-loaded assets, of the form `{filename: bytestring}`.


    Raises:
    - TypeError: If the model_handle format is not recognized.
    """
    logger.info("Loading model from model_handle")
    # Parse asset handle
    if isinstance(asset_handle, dict):
        asset_dict = asset_handle
    elif isinstance(asset_handle, str):
        asset_dict = get_asset_dict(asset_handle)
    elif (
        asset_handle is None
        and isinstance(model_handle, str)
        and model_handle.endswith(".xml")
    ):
        asset_handle = str(Path(model_handle).parent)
        asset_dict = get_asset_dict(asset_handle)
    else:
        asset_dict = None

    # Construct model from provided sim handle
    if isinstance(model_handle, Path):
        model_handle = str(model_handle)

    if isinstance(model_handle, str):
        if model_handle.endswith(".mjb"):
            mj_model = mujoco.MjModel.from_binary_path(model_handle)
        elif model_handle.endswith(".xml"):
            mj_model = mujoco.MjModel.from_xml_path(model_handle, asset_dict)
        elif model_handle.endswith("</mujoco>") or model_handle.endswith("</mujoco>\n"):
            if asset_dict is None:
                import ipdb

                ipdb.set_trace()  # so this is possible  # noqa: I001, E702
            mj_model = mujoco.MjModel.from_xml_string(model_handle, asset_dict)
        else:
            raise TypeError("Sim format not recognized. Check input")
    elif isinstance(model_handle, bytes):
        mj_model = mujoco.MjModel.from_xml_string(model_handle, asset_dict)
    elif isinstance(model_handle, mujoco.MjModel):
        mj_model = model_handle
    elif isinstance(model_handle, mujoco.MjSpec):
        mj_model = model_handle.compile()
    else:
        raise TypeError("Sim format not recognized. Check input")

    return mj_model, asset_dict


def get_data(
    model_handle: str | bytes | mujoco.MjModel, asset_handle: str | dict | None = None
):
    """
    Creates a Mujoco data object from the given model and asset handles.

    Args:
    - model_handle (str | bytes | mujoco.MjModel): The handle to the
      simulation file or Mujoco model.
    - asset_handle (str | dict, optional): The handle to the assets directory, or a
      dictionary of pre-loaded assets, of the form `{filename: bytestring}`.

    Returns:
    - mujoco.MjData: The Mujoco data object constructed from the model.

    Raises:
    - TypeError: If the model_handle format is not recognized.
    """
    mj_model = get_model(model_handle, asset_handle)

    return mujoco.MjData(mj_model)


def get_model_data(
    model_handle: str | bytes | mujoco.MjModel, asset_handle: str | dict | None = None
):
    """
    Constructs Mujoco model and data from the provided model handle and asset handle.

    Args:
    - model_handle (str, bytes, or mujoco.MjModel):
        The handle to the simulation file or Mujoco model.
    - asset_handle (str or dict, optional):
        The handle to the assets directory, or a dictionary
        of pre-loaded assets, of the form `{filename: bytestring}`.
        If present, assets in this dictionary will be used before
        attempting to load them from the filesystem.

    Returns:
    - mj_model: mujoco.MjModel
        The constructed Mujoco model.
    - mj_data: mujoco.MjData
        The constructed Mujoco data.
    - asset_dict: dict
        A dictionary of pre-loaded assets, of the form `{filename: bytestring}`.

    Raises:
    - TypeError: If the model_handle format is not recognized.
    """
    mj_model, asset_dict = get_model(model_handle, asset_handle)
    mj_data = mujoco.MjData(mj_model)

    return mj_model, mj_data, asset_dict


def get_jnt_qpos(mj_data: mujoco.MjData, jnt_name: str | Iterable[str]):
    """
    Get the qpos values for the specified joint names.

    Args:
    - mj_data (MjData): The MJCData object containing the joint data.
    - jnt_name (str | Iterable[str]): The name of the joint(s) for which to retrieve the qpos.

    Returns:
    - np.ndarray: A 1D numpy array containing the joint positions (qpos) for the specified joint names.
    """
    assert isinstance(mj_data, mujoco.MjData), "mj_data must be a mujoco.MjData object"
    assert isinstance(jnt_name, str) or isinstance(
        jnt_name, Iterable
    ), "jnt_name must be a string or an iterable of strings"
    if isinstance(jnt_name, str):
        return np.array([mj_data.joint(jnt_name).qpos.copy()])
    elif isinstance(jnt_name, Iterable):
        assert isinstance(jnt_name[0], str), "jnt_name must be an iterable of strings"
        return np.concatenate(
            [mj_data.joint(jnt_name).qpos.copy() for jnt_name in jnt_name]
        )
    else:
        raise ValueError("jnt_name must be a string or an iterable of strings")


def get_jnt_qpos0(mj_model: mujoco.MjModel, jnt_name: str | Iterable[str]):
    """
    Get the initial joint positions (qpos0) for the specified joint names.

    Args:
    - mj_model (MjModel): The MuJoCo model object.
    - jnt_name (str | Iterable[str]): The name of the joint(s) for which to retrieve the qpos0.

    Returns:
    - np.ndarray: The concatenated array of initial joint positions.

    """
    assert isinstance(
        mj_model, mujoco.MjModel
    ), f"mj_model must be a {mujoco.MjModel} object"
    assert isinstance(jnt_name, str) or isinstance(
        jnt_name, Iterable
    ), "jnt_name must be a string or an iterable of strings"
    if isinstance(jnt_name, str):
        return np.array([mj_model.joint(jnt_name).qpos0.copy()])
    elif isinstance(jnt_name, Iterable):
        assert isinstance(jnt_name[0], str), "jnt_name must be an iterable of strings"
        return np.concatenate(
            [mj_model.joint(jnt_name).qpos0.copy() for jnt_name in jnt_name]
        )
    else:
        raise ValueError("jnt_name must be a string or an iterable of strings")


def get_site_pos(mj_model: mujoco.MjModel, site_name: str | Iterable[str]):
    """
    Get the positions of the specified sites in the Mujoco model.

    Args:
    - mj_model (MjModel): The Mujoco model object.
    - site_name (str | Iterable[str]): The name of the site(s) whose positions need to be retrieved.

    Returns:
    - numpy.ndarray[len(site_name) * 3]: An array containing the positions of the specified sites.
    """
    assert isinstance(
        mj_model, mujoco.MjModel
    ), f"mj_model must be a {mujoco.MjModel} object"
    assert isinstance(site_name, str) or isinstance(
        site_name, Iterable
    ), "site_name must be a string or an iterable of strings"
    if isinstance(site_name, str):
        return np.array([mj_model.site(site_name).pos.copy()])
    elif isinstance(site_name, Iterable):
        assert isinstance(site_name[0], str), "site_name must be an iterable of strings"
        return np.concatenate(
            [mj_model.site(s_name).pos.copy() for s_name in site_name]
        )
    else:
        raise ValueError("site_name must be a string or an iterable of strings")


def get_site_xpos(mj_data: mujoco.MjData, site_name: str | Iterable[str]):
    """
    Get the global cartesian positions of the specified sites.

    Args:
    - mj_data (MjData): The Mujoco data object.
    - site_name (str | Iterable[str]): The name of the site(s) whose positions need to be retrieved.

    Returns:
    - numpy.ndarray[len(site_name) * 3]: An array containing the positions of the specified sites.
    """
    assert isinstance(mj_data, mujoco.MjData), "mj_data must be a mujoco.MjData object"
    assert isinstance(site_name, str) or isinstance(
        site_name, Iterable
    ), "site_name must be a string or an iterable of strings"
    if isinstance(site_name, str):
        return np.array([mj_data.site(site_name).xpos.copy()])
    elif isinstance(site_name, Iterable):
        assert isinstance(site_name[0], str), "site_name must be an iterable of strings"
        return np.concatenate(
            [mj_data.site(s_name).xpos.copy() for s_name in site_name]
        )
    else:
        raise ValueError("site_name must be a string or an iterable of strings")


def get_site_dist(mj_data: mujoco.MjData, site1_name: str, site2_name: str):
    """
    Calculate the Euclidean distance between two markers in the Mujoco simulation.

    Args:
    - mj_data (MjData): The Mujoco data object containing simulation data.
    - site1_name (str): The name of the first marker.
    - site2_name (str): The name of the second marker.

    Returns:
    - float: The Euclidean distance between the two markers.
    """
    return np.linalg.norm(
        mj_data.site(site1_name).xpos - mj_data.site(site2_name).xpos,
    )


def get_jnt_dist(mj_data: mujoco.MjData, jnt1_name: str, jnt2_name: str):
    """
    Calculate the distance between two joints in global coordinates.

    Args:
    - mj_data: The MujocoData object containing the joint data.
    - joint_name1: The name of the first joint.
    - joint_name2: The name of the second joint.

    Returns:
    - The Euclidean distance between the two joints.
    """
    return np.linalg.norm(
        mj_data.joint(jnt1_name).xanchor - mj_data.joint(jnt2_name).xanchor,
    )


def set_site_pos(
    mj_model: mujoco.MjModel,
    site_names: Iterable[str],
    site_pos: Iterable[Iterable[float]],
):
    """
    Set the local position of multiple sites in a Mujoco model.

    Args:
    - mj_model (MjModel): The Mujoco model object.
    - site_names (list): A list of site names to set the position for.
    - site_pos (list): A list of site positions corresponding to the site names.

    Returns:
    - None
    """
    for idx, site_name in enumerate(site_names):
        mj_model.site(site_name).pos[:] = site_pos[idx]


def get_qpos_indices(mj_model: mujoco.MjModel, joint_name: str | Iterable[str]):
    """
    Gets the position index for a specified joint in the model's qpos array.

    Args:
    - mj_model (mujoco.MjModel): The Mujoco model object.
    - joint_name (str | Iterable[str]): Name of the joint(s) to get index for.

    Returns:
    - list[int] | list[list[int]]: List of position indices for the specified joint(s) or list of lists of position indices for multiple joints.

    Raises:
    - AssertionError: If mj_model is not a MjModel object or joint_name is not a string or iterable of strings.
    """
    assert isinstance(
        mj_model, mujoco.MjModel
    ), f"mj_model must be a {mujoco.MjModel} object"
    assert isinstance(joint_name, str) or isinstance(
        joint_name, Iterable
    ), "joint_name must be a string or an iterable of strings"
    if isinstance(joint_name, str):
        res = [int(mj_model.joint(joint_name).qposadr[0])]
        if mj_model.joint(joint_name).type == mujoco.mjtJoint.mjJNT_FREE:
            for count in range(1, 7):
                res.append(int(res[0]) + count)
        elif mj_model.joint(joint_name).type == mujoco.mjtJoint.mjJNT_BALL:
            for count in range(1, 4):
                res.append(int(res[0]) + count)
        return res
    elif isinstance(joint_name, Iterable):
        assert isinstance(
            joint_name[0], str
        ), "joint_name must be an iterable of strings"
        return [get_qpos_indices(mj_model, jnt_name) for jnt_name in joint_name]
    else:
        raise ValueError("joint_name must be a string or an iterable of strings")


def get_qvel_indices(mj_model: mujoco.MjModel, joint_name: str | Iterable[str]):
    """
    Gets the velocity index for a specified joint in the model's qvel array.

    Args:
    - mj_model (mujoco.MjModel): The Mujoco model object.
    - joint_name (str): Name of the joint to get index for.

    Returns:
    - list[int] | list[list[int]]: List of velocity index for the specified joint, or list of lists of velocity indices for multiple joints.

    Raises:
    - AssertionError: If mj_model is not a MjModel object or joint_name is not a string.
    """
    assert isinstance(
        mj_model, mujoco.MjModel
    ), f"mj_model must be a {mujoco.MjModel} object"
    assert isinstance(joint_name, str) or isinstance(
        joint_name, Iterable
    ), "joint_name must be a string or an iterable of strings"
    if isinstance(joint_name, str):
        res = [int(mj_model.joint(joint_name).dofadr[0])]
        if mj_model.joint(joint_name).type == mujoco.mjtJoint.mjJNT_FREE:
            for count in range(1, 6):
                res.append(int(res[0]) + count)
        elif mj_model.joint(joint_name).type == mujoco.mjtJoint.mjJNT_BALL:
            for count in range(1, 3):
                res.append(int(res[0]) + count)
        return res
    elif isinstance(joint_name, Iterable):
        assert isinstance(
            joint_name[0], str
        ), "joint_name must be an iterable of strings"
        return [get_qvel_indices(mj_model, jnt_name) for jnt_name in joint_name]
    else:
        raise ValueError("joint_name must be a string or an iterable of strings")


def get_body_name(model: mujoco.MjModel, body_index: int | Iterable[int]):
    """
    Gets body name from model using name_bodyadr ensuring compatibility with
    CPU MuJoCo and MJX.

    Args:
    - model (mujoco.MjModel | mjx._src.types.Model): The Mujoco model object.
    - body_index (int | Iterable[int]): Index of the body to get name for.

    Returns:
    - str | list[str]: Name of the body at the specified index if input is int,
        or list of body names if input is an iterable of index.
    """
    assert is_int_or_iterable_of_ints(
        body_index
    ), "body_index must be an int or an iterable of ints"

    if isinstance(body_index, int):
        start = model.name_bodyadr[body_index]
        # Find the null terminator in the names bytes
        end = model.names.index(b"\x00", start)
        return model.names[start:end].decode("utf-8")

    elif isinstance(body_index, Iterable):
        return [get_body_name(model, int(idx)) for idx in body_index]

    else:
        raise ValueError("body_index must be an int or an iterable of ints")


def get_mocap_names(
    model_handle,
    asset_handle=None,
):
    """
    Get a list of all the mocap bodies in the model

    Args:
        model_handle (str): The path to the model file, or an XML string containing the model content.
        asset_handle (str): The path to the asset directory containing ./meshes ./textures ./skins etc. If none, provide an empty dictionary

    Returns:
        List: A list containing the names of all the mocap bodies in the provided MjSpec
    """

    # Get the model
    mj_model, asset_handle = get_model(
        model_handle=model_handle, asset_handle=asset_handle
    )

    # Get all the mocap body names from the model
    mocap_bodies = [
        mj_model.body(bidx).name
        for bidx in range(mj_model.nbody)
        if mj_model.body_mocapid[bidx] != -1
    ]

    return mocap_bodies


def name2enum_geomtype(name):
    """
    Converts a geometry type name (string) to its corresponding MuJoCo enum value.
    Supported names:
        - "mjGEOM_PLANE" or "plane"
        - "mjGEOM_HFIELD" or "hfield"
        - "mjGEOM_SPHERE" or "sphere"
        - "mjGEOM_CAPSULE" or "capsule"
        - "mjGEOM_ELLIPSOID" or "ellipsoid"
        - "mjGEOM_CYLINDER" or "cylinder"
        - "mjGEOM_BOX" or "box"
        - "mjGEOM_MESH" or "mesh"
        - "mjGEOM_SDF" or "sdf"
    Args:
        name (str): The geometry type name.
    Returns:
        mujoco.mjtGeom: The corresponding MuJoCo geometry enum value.
    Raises:
        Exception: If the provided name does not match any known geometry type.
    """

    if name == "mjGEOM_PLANE" or name.lower() == "plane":
        return mujoco.mjtGeom.mjGEOM_PLANE

    elif name == "mjGEOM_HFIELD" or name.lower() == "hfield":
        return mujoco.mjtGeom.mjGEOM_HFIELD

    elif name == "mjGEOM_SPHERE" or name.lower() == "sphere":
        return mujoco.mjtGeom.mjGEOM_SPHERE

    elif name == "mjGEOM_CAPSULE" or name.lower() == "capsule":
        return mujoco.mjtGeom.mjGEOM_CAPSULE

    elif name == "mjGEOM_ELLIPSOID" or name.lower() == "ellipsoid":
        return mujoco.mjtGeom.mjGEOM_ELLIPSOID

    elif name == "mjGEOM_CYLINDER" or name.lower() == "cylinder":
        return mujoco.mjtGeom.mjGEOM_CYLINDER

    elif name == "mjGEOM_BOX" or name.lower() == "box":
        return mujoco.mjtGeom.mjGEOM_BOX

    elif name == "mjGEOM_MESH" or name.lower() == "mesh":
        return mujoco.mjtGeom.mjGEOM_MESH

    elif name == "mjGEOM_SDF" or name.lower() == "sdf":
        return mujoco.mjtGeom.mjGEOM_SDF

    else:
        raise Exception("Unknown geom type", name)


def name2enum_jointtype(name):
    """
    Converts a joint type name (string) to its corresponding MuJoCo joint enum value.

        - "mjJNT_FREE" or "free"
        - "mjJNT_BALL" or "ball"
        - "mjJNT_SLIDE" or "slide"
        - "mjJNT_HINGE" or "hinge"

        name (str): The joint type name.
        mujoco.mjtJoint: The corresponding MuJoCo joint enum value.
        Exception: If the provided name does not match any known joint type.
    """
    if name.lower() == "free" or name == "mjJNT_FREE":
        return mujoco.mjtJoint.mjJNT_FREE

    elif name.lower() == "ball" or name == "mjJNT_BALL":
        return mujoco.mjtJoint.mjJNT_BALL

    elif name.lower() == "slide" or name == "mjJNT_SLIDE":
        return mujoco.mjtJoint.mjJNT_SLIDE

    elif name.lower() == "hinge" or name == "mjJNT_HINGE":
        return mujoco.mjtJoint.mjJNT_HINGE
    else:
        raise Exception("Unknown joint type", name)
