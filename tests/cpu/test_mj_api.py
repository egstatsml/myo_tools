"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for mj_api

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myo_tools.mj.core import mj_api


def test_mj_api_Path(model_path, assets_folder):
    mj_model, mj_data, asset_dict = mj_api.get_model_data(
        Path(model_path), assets_folder
    )
    for i in range(2):
        mujoco.mj_step(mj_model, mj_data)


def test_mj_api_getters(model_path, assets_folder):
    mj_model, mj_data, asset_dict = mj_api.get_model_data(model_path, assets_folder)
    iq_j1 = mj_api.get_qpos_indices(mj_model, "joint1")
    assert iq_j1 == [1]

    iv_j1 = mj_api.get_qvel_indices(mj_model, "joint1")
    assert iv_j1 == [1]

    name_b1 = mj_api.get_body_name(mj_model, 1)
    assert name_b1 == "box"

    name_b2 = mj_api.get_body_name(mj_model, 2)
    assert name_b2 == "body0"


# Fixture for a simple MuJoCo model
@pytest.fixture
def simple_mjmodel():
    xml_content = """
    <mujoco>
        <default>
            <geom contype="0" conaffinity="0" type="capsule" size=".01"/>
            <joint type="hinge" axis="1 0 0" damping=".002"/>
            <default class="myo_marker">
                <site size="0.02" group="1" rgba="0.8 0.2 0.6 1"/>
            </default>
        </default>
        <worldbody>
            <body name="body0">
                <joint type="free" name="free_joint"/>
                <geom fromto="0.1 0 0 -.1 0 0"/>
                <site name="site1" pos="0 0 0"/>
            </body>
            <body name="body1" pos="0 0 0">
                <geom fromto="0 0 0 0 0 .2" />
                <joint name="ball_joint" type="ball"/>
                <site name="site2" pos="0 0 0"/>
                <body name="body2" pos="0 0 .2">
                    <joint name="hinge_joint" type="hinge"/>
                    <geom fromto="0 0 0 0 .2 .2"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    model_path = Path("temp_model.xml")
    model_path.write_text(xml_content)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    model_path.unlink()  # Clean up temporary file
    return model, data


def test_get_qpos_indices(simple_mjmodel):
    model, _ = simple_mjmodel

    # Test free joint
    free_indices = mj_api.get_qpos_indices(model, "free_joint")
    assert len(free_indices) == 7  # Free joint has 7 DoF

    # Test hinge joint
    hinge_indices = mj_api.get_qpos_indices(model, "hinge_joint")
    assert len(hinge_indices) == 1  # Hinge joint has 1 DoF

    # Test ball joint
    ball_indices = mj_api.get_qpos_indices(model, "ball_joint")
    assert len(ball_indices) == 4  # Ball joint has 4 DoF

    # Test list of joints
    idx_list = mj_api.get_qpos_indices(model, ["free_joint", "hinge_joint"])
    assert isinstance(idx_list, list)

    # Test tuple of joints
    idx_tuple = mj_api.get_qpos_indices(model, ("free_joint", "hinge_joint"))
    assert isinstance(idx_tuple, list)

    # Test invalid input
    with pytest.raises((AssertionError, ValueError)):
        mj_api.get_qpos_indices(model, 123)


def test_get_qvel_indices(simple_mjmodel):
    model, _ = simple_mjmodel

    # Test free joint
    free_indices = mj_api.get_qvel_indices(model, "free_joint")
    assert len(free_indices) == 6  # Free joint has 6 velocity components

    # Test hinge joint
    hinge_indices = mj_api.get_qvel_indices(model, "hinge_joint")
    assert len(hinge_indices) == 1  # Hinge joint has 1 velocity component

    # Test ball joint
    ball_indices = mj_api.get_qvel_indices(model, "ball_joint")
    assert len(ball_indices) == 3  # Ball joint has 3 velocity components

    # Test list of joints
    idx_list = mj_api.get_qvel_indices(model, ["free_joint", "hinge_joint"])
    assert isinstance(idx_list, list)

    # Test tuple of joints
    idx_tuple = mj_api.get_qvel_indices(model, ("free_joint", "hinge_joint"))
    assert isinstance(idx_tuple, list)

    # Test invalid input
    with pytest.raises((AssertionError, ValueError)):
        mj_api.get_qvel_indices(model, 123)


def test_get_body_name(model_XML):
    model, _ = mj_api.get_model(model_XML)
    # int
    body_name = mj_api.get_body_name(model, 0)
    assert body_name == "world"
    # list of int
    body_name = mj_api.get_body_name(model, [0, 1])
    assert body_name[0] == "world"
    # numpy array of int
    body_name = mj_api.get_body_name(model, np.array([0, 1, 2]))
    assert body_name[0] == "world"


@pytest.fixture
def mocap_mjmodel():
    xml_content = """
    <mujoco>
        <default>
            <geom contype="0" conaffinity="0" type="capsule" size=".01"/>
            <joint type="hinge" axis="1 0 0" damping=".002"/>
            <default class="myo_marker">
                <site size="0.02" group="1" rgba="0.8 0.2 0.6 1"/>
            </default>
        </default>
        <worldbody>
            <body name="mocap_body1" pos="0 1 0" mocap="true"/>
            <body name="mocap_body2" pos="1 0 0" mocap="true"/>
            <body name="not_mocap" pos="0 0 1" mocap="false"/>
        </worldbody>
    </mujoco>
    """
    model_path = Path("temp_model.xml")
    model_path.write_text(xml_content)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model_path.unlink()  # Clean up temporary file
    return model


def test_get_mocap_names(mocap_mjmodel):
    mj_model = mocap_mjmodel
    mocap_body_names = mj_api.get_mocap_names(mj_model, None)

    assert "mocap_body1" in mocap_body_names and "mocap_body2" in mocap_body_names


def test_get_jnt_qpos0(model_XML):
    model, _ = mj_api.get_model(model_XML)

    # Test single joint (str)
    qpos0_str = mj_api.get_jnt_qpos0(model, "joint1")
    assert isinstance(qpos0_str, np.ndarray)

    # Test list of joints
    qpos0_list = mj_api.get_jnt_qpos0(model, ["joint1", "joint2"])
    assert isinstance(qpos0_list, np.ndarray)

    # Test tuple of joints
    qpos0_tuple = mj_api.get_jnt_qpos0(model, ("joint1", "joint2"))
    assert isinstance(qpos0_tuple, np.ndarray)

    # Test invalid input
    with pytest.raises(AssertionError):
        mj_api.get_jnt_qpos0(model, 123)


def test_get_site_xpos(model_XML):
    model, data, _ = mj_api.get_model_data(model_XML)

    # Test single site (str)
    xpos_str = mj_api.get_site_xpos(data, "site0")
    assert isinstance(xpos_str, np.ndarray)
    assert xpos_str.shape[-1] == 3

    # Test list of sites
    xpos_list = mj_api.get_site_xpos(data, ["site0", "site1"])
    assert isinstance(xpos_list, np.ndarray)
    assert xpos_list.shape[-1] == 6

    # Test tuple of sites
    xpos_tuple = mj_api.get_site_xpos(data, ("site0", "site1"))
    assert isinstance(xpos_tuple, np.ndarray)
    assert xpos_tuple.shape[-1] == 6

    # Test invalid inputs
    with pytest.raises((AssertionError, ValueError)):
        mj_api.get_site_xpos(data, 123)


def test_set_site_pos(model_XML):
    model, _ = mj_api.get_model(model_XML)

    # Test single site
    single_site = ["site0"]
    single_pos = [[1.0, 2.0, 3.0]]
    mj_api.set_site_pos(model, single_site, single_pos)
    np.testing.assert_array_almost_equal(model.site("site0").pos, single_pos[0])

    # Test multiple sites with list
    list_sites = ["site0", "site1"]
    list_pos = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    mj_api.set_site_pos(model, list_sites, list_pos)
    for site_name, expected_pos in zip(list_sites, list_pos):
        np.testing.assert_array_almost_equal(model.site(site_name).pos, expected_pos)

    # Test multiple sites with tuple
    tuple_sites = ("site0", "site1")
    tuple_pos = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    mj_api.set_site_pos(model, tuple_sites, tuple_pos)
    for site_name, expected_pos in zip(tuple_sites, tuple_pos):
        np.testing.assert_array_almost_equal(model.site(site_name).pos, expected_pos)


def test_get_jnt_dist(model_XML):
    model, data, _ = mj_api.get_model_data(model_XML)

    # Test with str inputs
    dist_str = mj_api.get_jnt_dist(data, "joint1", "joint2")
    assert isinstance(dist_str, float)
    assert dist_str >= 0

    # Verify symmetry
    dist_reverse = mj_api.get_jnt_dist(data, "joint2", "joint1")
    assert np.isclose(dist_str, dist_reverse)


# Test joint enums
def test_joint_enum():
    # Test each joint type
    for joint_name, expected_type, type_enum in zip(
        ["free", "ball", "hinge", "slide"],
        ["mjJNT_FREE", "mjJNT_BALL", "mjJNT_HINGE", "mjJNT_SLIDE"],
        [
            mujoco.mjtJoint.mjJNT_FREE,
            mujoco.mjtJoint.mjJNT_BALL,
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ],
    ):
        # Test that the output is the same using joint name and type name
        assert mj_api.name2enum_jointtype(joint_name) == mj_api.name2enum_jointtype(
            expected_type
        )

        # Test that the output matches the expected enum value
        assert mj_api.name2enum_jointtype(joint_name) == type_enum


# Test geom enums
def test_geom_enum():
    # Test each geom type
    for geom_name, expected_type, type_enum in zip(
        [
            "plane",
            "hfield",
            "sphere",
            "capsule",
            "ellipsoid",
            "cylinder",
            "box",
            "mesh",
            "sdf",
        ],
        [
            "mjGEOM_PLANE",
            "mjGEOM_HFIELD",
            "mjGEOM_SPHERE",
            "mjGEOM_CAPSULE",
            "mjGEOM_ELLIPSOID",
            "mjGEOM_CYLINDER",
            "mjGEOM_BOX",
            "mjGEOM_MESH",
            "mjGEOM_SDF",
        ],
        [
            mujoco.mjtGeom.mjGEOM_PLANE,
            mujoco.mjtGeom.mjGEOM_HFIELD,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_MESH,
            mujoco.mjtGeom.mjGEOM_SDF,
        ],
    ):
        # Test that the output is the same using geom name and type name
        assert mj_api.name2enum_geomtype(geom_name) == mj_api.name2enum_geomtype(
            expected_type
        )

        # Test that the output matches the expected enum value
        assert mj_api.name2enum_geomtype(geom_name) == type_enum
