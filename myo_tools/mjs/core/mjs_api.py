"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: This file provides APIs for creation and operations related to a mujoco spec object

from pathlib import Path
from typing import Iterable, List

import mujoco
import numpy as np

from myo_tools.utils.file_ops.import_utils import get_asset_dict


def get_model_spec(
    mjs_handle: str | mujoco.MjSpec | Path, asset_handle: str | dict | None = None
):
    """
    Constructs a  MjSpec object from the provided simulation and asset handle.

    Args:
    - mjs_handle (str, Path or mujoco.MjSpec):
        The handle to the simulation file or the editable MjSpec.
    - asset_handle (str or dict, optional):
        The handle to the assets directory, or a dictionary
        of pre-loaded assets, of the form `{filename: bytestring}`.
        If present, assets in this dictionary will be used before
        attempting to load them from the filesystem.

    Returns:
    - mj_spec: mujoco.MjSpec
        The constructed Mujoco spec.
    - asset_dict: dict
        A dictionary of pre-loaded assets, of the form `{filename: bytestring}`.


    Raises:
    - TypeError: If the mjs_handle format is not recognized.
    """
    # Parse asset handle
    if isinstance(asset_handle, str):
        asset_dict = get_asset_dict(asset_handle)
    elif (
        asset_handle is None
        and isinstance(mjs_handle, str)
        and mjs_handle.endswith(".xml")
    ):
        asset_handle = str(Path(mjs_handle).parent)
        asset_dict = get_asset_dict(asset_handle)
    elif isinstance(asset_handle, dict):
        asset_dict = asset_handle
    else:
        asset_dict = None

    # Construct model from provided spec handle
    if isinstance(mjs_handle, Path):
        mjs_handle = str(mjs_handle)

    if isinstance(mjs_handle, str):
        if mjs_handle.endswith(".xml"):
            mj_spec = mujoco.MjSpec.from_file(mjs_handle, assets=asset_dict)
        elif mjs_handle.endswith("</mujoco>") or mjs_handle.endswith("</mujoco>\n"):
            mj_spec = mujoco.MjSpec.from_string(mjs_handle, assets=asset_dict)
        else:
            raise TypeError("Spec format not recognized. Check input")
    elif isinstance(mjs_handle, mujoco.MjSpec):
        mj_spec = mjs_handle
    else:
        raise TypeError("Spec format not recognized. Check input")

    return mj_spec, asset_dict


def get_model_spec_with_skn(
    mjs_handle: str,
    skin_name: str,
    skin_skn: str,
    skin_texture: str | None = None,
    reflectance: float = 0,
    emission: float = 1,
):
    """
    Generates a compiled MuJoCo model with optional skin and texture.

    Args:
    - mjs_handle (str, Path or mujoco.MjSpec): Path to the model XML file.
    - skin_name (str): Name for the skin and associated materials.
    - skin_skn (str): Path to the skin file.
    - skin_texture (str, optional): Path to the texture file.
    - reflectance (float, optional): reflectance parameter used for rendering
    - emission (float, optional): emission parameter used for rendering

    Returns:
    - mujoco.MjSpec: MuJoCo spec with the specified skin and texture.
    """
    # get model
    model_spec, _ = get_model_spec(mjs_handle)

    if skin_texture is not None:
        # create texture
        model_spec.add_texture(
            file=skin_texture,
            name=f"{skin_name}_texture",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
        )

        # Create a material and assign texture to the material
        screen_material = model_spec.add_material(
            name=f"{skin_name}_material",
            texrepeat=[1, 1],
            reflectance=reflectance,
            emission=emission,
        )
        screen_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = (
            f"{skin_name}_texture"
        )

        # create skin
        model_spec.add_skin(file=skin_skn, material=f"{skin_name}_material")
    else:
        # create skin
        model_spec.add_skin(file=skin_skn)

    return model_spec


def _removeJoints(spec: mujoco.MjSpec, jointIdxList: List[int]) -> None:
    """
    Remove joints from a MuJoCo model specification.

    Args:
        model: Either a path to XML file or a MuJoCo model specification handle
        jointIdxList: List of joint indices to remove

    Raises:
        ValueError: If model is neither a string path nor a MuJoCo MjSpec

    Note:
        Removes joints from last to first to prevent index shifting from affecting removal
    """
    # remove in reverse order to prevent the removal to have an effect on the index of the joint to remove
    jointNameList = [spec.joints[jnt_idx].name for jnt_idx in jointIdxList]
    for joint_idx in np.sort(jointIdxList)[::-1]:
        spec.joints[joint_idx].delete()
    for eq in spec.equalities:
        if eq.name1 in jointNameList or eq.name2 in jointNameList:
            eq.delete()


def _removeBodies(spec: mujoco.MjSpec, bodyIdxList: List[int]) -> None:
    """
    Remove bodies from a MuJoCo model specification.

    Args:
        model: Either a path to XML file or a MuJoCo model specification handle
        bodyIdxList: List of body indices for bodies to be removed

    Raises:
        ValueError: If model is neither a string path nor a MuJoCo MjSpec

    Note:
        Removes bodies from last to first to prevent index shifting from affecting removal
    """
    # Never remove the world body
    if 0 in bodyIdxList:
        bodyIdxList = [idx for idx in bodyIdxList if idx != 0]

    # Remove in reverse order to prevent the removal to have an effect on the index of the body to remove
    for body_idx in np.sort(bodyIdxList)[::-1]:
        spec.detach_body(spec.bodies[body_idx])


def filter_model_joints(
    mjs_handle: str | mujoco.MjSpec | Path,
    joint_selection: List[str],
    keep=False,
):
    """
    Compile a reduced MuJoCo model by keeping or removing only the specified joints.

    Args:
        - mjs_handle (str, Path or mujoco.MjSpec):
          The handle to the simulation file or the editable MjSpec.
        - joint_selection (list):
          List of joint names. Keep the joints if the keep variable is true otherwise remove them (default).
        - keep (bool):
          Flag to indicate if the joint selected should be removed or the only one mantained (remove all the others) in the model

    Returns:
        tuple: (spec, model_dict) where:
            - spec (mujoco.MjSpec): Compiled reduced model specification
            - model_dict (dict): dict mapping body names to lists of (joint_name, joint_type) pairs for removed joints
    """
    # Get the spec
    spec, _ = get_model_spec(mjs_handle)

    # Get a list of all the joints
    jnt_list = [j.name for j in spec.joints]

    # Create a dictionary of body names and their corresponding joints
    model_dict = {}
    for model_body in spec.bodies:
        temp_list = []

        jnt = model_body.first_joint()
        while jnt:
            temp_list.append((jnt.name, jnt.type))
            jnt = model_body.next_joint(jnt)

        if any(jnt_name in joint_selection for (jnt_name, _) in temp_list):
            model_dict[model_body.name] = temp_list

    # List all the indices of the model joints to be removed
    jnt_to_delete = [
        j_idx
        for j_idx, j_name in enumerate(jnt_list)
        if (j_name not in joint_selection if keep else (j_name in joint_selection))
    ]

    _removeJoints(spec, jnt_to_delete)
    return spec, model_dict


def filter_model_bodies(
    mjs_handle: str | mujoco.MjSpec | Path,
    body_selection: List[str],
):
    """
    Compile a reduced MuJoCo model by keeping or removing only specified bodies.

    Args:
        - mjs_handle (str, Path or mujoco.MjSpec):
          The handle to the simulation file or the editable MjSpec.

        - body_selection (list):
          List of body names to remove from the model

    Returns:
        tuple: (spec, model_dict) where:
            - spec (mujoco.MjSpec): Compiled reduced model specification
            - model_dict (dict): dict mapping body names to lists of (joint_name, joint_type) pairs for removed bodies
    """
    # Get the spec
    spec, _ = get_model_spec(mjs_handle)

    # Get a list of all the model bodies
    body_list = [b.name for b in spec.bodies]

    # List all the indices of the bodies to be removed
    bodies_to_delete = [
        b_idx for b_idx, b_name in enumerate(body_list) if (b_name in body_selection)
    ]

    # Create a dictionary of body names and their corresponding joints
    model_dict = {}
    for model_body in spec.bodies:
        temp_list = []

        jnt = model_body.first_joint()
        while jnt:
            temp_list.append((jnt.name, jnt.type))
            jnt = model_body.next_joint(jnt)

        if any(
            spec.bodies[b_idx].find_child(model_body.name) for b_idx in bodies_to_delete
        ):
            model_dict[model_body.name] = temp_list

    _removeBodies(spec, bodies_to_delete)
    return spec, model_dict


def get_mocap_names(mjs_handle, asset_handle=None):
    """
    Get a list of all the mocap bodies in the model

    Args:
        mjs_handle (str): The path to the model file, an XML string or an MjSpec containing the model content.
        asset_handle (str): The path to the asset directory containing ./meshes ./textures ./skins etc. If none, provide an empty dictionary

    Returns:
        List: A list containing the names of all the mocap bodies in the provided MjSpec
    """

    # Get the model spec
    spec, asset_handle = get_model_spec(
        mjs_handle=mjs_handle, asset_handle=asset_handle
    )

    # Get all body names where the body has attribute mocap = true
    mocap_bodies = [body.name for body in spec.bodies if body.mocap]

    return mocap_bodies


def remove_mocap_bodies(mjs_handle, asset_handle=None):
    """
    Remove all mocap bodies from the model

    Args:
        mjs_handle (str): The path to the model file, an XML string or an MjSpec containing the model content.
        asset_handle (str): The path to the asset directory containing ./meshes ./textures ./skins etc. If none, provide an empty dictionary

    Returns:
        tuple: A tuple containing the updated Mujoco model and the handle of preloaded assets.
    """

    # Get the mocap body names
    mocap_body_names = get_mocap_names(mjs_handle, asset_handle)

    # Get the model spec without the mocap bodies
    spec, _ = filter_model_bodies(mjs_handle, mocap_body_names)

    return spec, asset_handle


def set_site_pos(
    mjs_handle: str | mujoco.MjSpec | Path,
    site_names: Iterable[str],
    site_pos: Iterable[Iterable[float]],
):
    """
    Set the local position of multiple sites in a Mujoco model.

    Args:
    - mjs_handle (str | mujoco.MjSpec | Path): The handle to the Mujoco model.
    - site_names (list): A list of site names to set the position for.
    - site_pos (list): A list of site positions corresponding to the site names.

    Returns:
    - mujoco.MjSpec: MuJoCo spec with the specified site positions.
    """
    # Get the spec
    spec, _ = get_model_spec(mjs_handle)

    spec_sites_names = [spec.sites[i].name for i in range(len(spec.sites))]
    for idx, site_name in enumerate(site_names):
        spec.sites[spec_sites_names.index(site_name)].pos[:] = site_pos[idx]

    return spec


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


def get_joint_names(mjs_handle, asset_handle=None):
    """
    Get a list of all the joint names in the model

    Args:
        mjs_handle (str): The path to the model file, an XML string or an MjSpec containing the model content.
        asset_handle (str): The path to the asset directory containing ./meshes ./textures ./skins etc. If none, provide an empty dictionary

    Returns:
        List: A list containing the names of all the joints in the provided MjSpec
    """

    # Get the model spec
    spec, asset_handle = get_model_spec(
        mjs_handle=mjs_handle, asset_handle=asset_handle
    )

    # Get all body names where the body has attribute mocap = true
    joint_names = [joint.name for joint in spec.joints]

    return joint_names


def remove_contact_pairs(
    mjs_handle: str | mujoco.MjSpec | Path,
    pair_names: List[str] = None,
):
    """
    Remove specified contact pairs from a MuJoCo spec. If no pairs are specified, all contact pairs are removed.

    Args:
        mjs_handle (str | mujoco.MjSpec | Path): The handle to the simulation file or the editable MjSpec.
        pair_names (List[str]): List of contact pair names to be removed.
    Returns:
        mujoco.MjSpec: The updated MuJoCo model specification with specified contact pairs removed
    """
    # Get the spec
    model_spec, _ = get_model_spec(mjs_handle)
    removed_pair_names = []

    if not pair_names:
        pair_names = []

    # If no pair names are provided, remove all contact pairs
    for pair in model_spec.pairs:
        if pair.name in pair_names or not pair_names:
            removed_pair_names.append(pair.name)
            pair.delete()

    return model_spec, removed_pair_names


def remove_nonmesh_geoms(
    mjs_handle: str | mujoco.MjSpec | Path,
    geom_names: List[str] = None,
):
    """
    Remove specified non-mesh geoms from a MuJoCo spec. If no geoms are specified, all non-mesh geoms are removed.

    Args:
        mjs_handle (str | mujoco.MjSpec | Path): The handle to the simulation file or the editable MjSpec.
        geom_names (List[str]): List of non-mesh geom names to be removed.
    Returns:
        mujoco.MjSpec: The updated MuJoCo model specification with specified non-mesh geoms removed
        List[str]: List of names of the removed geoms
    """
    # Get the spec
    model_spec, _ = get_model_spec(mjs_handle)
    removed_geom_names = []

    if not geom_names:
        geom_names = []

    # If no geom names are provided, remove all non-mesh geoms
    for geom in model_spec.geoms:
        if (
            geom.name in geom_names or (not geom_names)
        ) and geom.type != name2enum_geomtype("mesh"):
            removed_geom_names.append(geom.name)
            geom.delete()

    return model_spec, removed_geom_names


def remove_equality_constraints(
    mjs_handle: str | mujoco.MjSpec | Path,
    eq_names: List[str] = None,
):
    """
    Remove specified equality constraints from a MuJoCo spec. If no constraints are specified, all equality constraints are removed.

    Args:
        mjs_handle (str | mujoco.MjSpec | Path): The handle to the simulation file or the editable MjSpec.
        eq_names (List[str]): List of equality constraint names to be removed.
    Returns:
        mujoco.MjSpec: The updated MuJoCo model specification with specified equality constraints removed
        List(str): List of names of the removed equality constraints
    """

    # Get the spec
    model_spec, _ = get_model_spec(mjs_handle)
    removed_eq_names = []

    if not eq_names:
        eq_names = []

    # If no equality constraint names are provided, remove all equality constraints
    for eq in model_spec.equalities:
        if eq.name in eq_names or not eq_names:
            removed_eq_names.append(eq.name)
            eq.delete()

    return model_spec, removed_eq_names


def remove_sites(
    mjs_handle: str | mujoco.MjSpec | Path,
    site_names: List[str] = None,
):
    """
    Remove specified sites from a MuJoCo spec. If no sites are specified, all sites are removed.

    Args:
        mjs_handle (str | mujoco.MjSpec | Path): The handle to the simulation file or the editable MjSpec.
        site_names (List[str]): List of site names to be removed.
    Returns:
        mujoco.MjSpec: The updated MuJoCo model specification with specified sites removed
        List(str): List of names of the removed sites
    """

    # Get the spec
    model_spec, _ = get_model_spec(mjs_handle)
    removed_site_names = []

    if not site_names:
        site_names = []

    # If no site names are provided, remove all sites
    for site in model_spec.sites:
        if site.name in site_names or not site_names:
            removed_site_names.append(site.name)
            site.delete()

    return model_spec, removed_site_names


def remove_sensors(
    mjs_handle: str | mujoco.MjSpec | Path,
    sensor_names: List[str] = None,
):
    """
    Remove specified sensors from a MuJoCo spec. If no sensors are specified, all sensors are removed.

    Args:
        mjs_handle (str | mujoco.MjSpec | Path): The handle to the simulation file or the editable MjSpec.
        sensor_names (List[str]): List of sensor names to be removed.
    Returns:
        mujoco.MjSpec: The updated MuJoCo model specification with specified sensors removed
        List(str): List of names of the removed sensors
    """

    # Get the spec
    model_spec, _ = get_model_spec(mjs_handle)
    removed_sensor_names = []

    if not sensor_names:
        sensor_names = []

    # If no sensor names are provided, remove all sensors
    for sensor in model_spec.sensors:
        if sensor.name in sensor_names or not sensor_names:
            removed_sensor_names.append(sensor.name)
            sensor.delete()

    return model_spec, removed_sensor_names
