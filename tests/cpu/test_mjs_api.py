"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

from pathlib import Path

import mujoco
import pytest

from myo_tools.mjs.core.mjs_api import (
    filter_model_bodies,
    filter_model_joints,
    get_joint_names,
    get_mocap_names,
    get_model_spec,
    name2enum_geomtype,
    name2enum_jointtype,
    remove_contact_pairs,
    remove_equality_constraints,
    remove_mocap_bodies,
    remove_nonmesh_geoms,
    remove_sensors,
    remove_sites,
)

model_handle = """
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
            <geom fromto="0 0 0 0 0.2 0" />
            <joint name="ball_joint" type="ball"/>
            <site name="site2" pos="0 0 0"/>
            <body name="body2" pos="0 0 .2">
                <joint name="hinge_joint" type="hinge"/>
                <geom fromto="0 0 0 0 .2 .2"/>
                <geom name="coll_geom_1" fromto="0 0 0 .1 0 0" contype="1" conaffinity="1"/>
            </body>
            <body name="body3" pos="0 0.2 0">
                <joint name="hinge_joint_2" type="hinge"/>
                <geom fromto="0 0 0 0 .2 .2"/>
                <geom name="coll_geom_2" fromto="0 0 0 .1 0 0" contype="1" conaffinity="1"/>
            </body>
        </body>
    </worldbody>
    <equality>
        <joint name="hinge_constraint" joint1="hinge_joint" joint2="hinge_joint_2" polycoef="0 1 0 0 0" />
    </equality>
    <sensor>
        <!-- Defines a touch sensor that senses contact forces at the specified site -->
        <touch site="site1"/>
    </sensor>
</mujoco>"""
asset_handle = None


def test_get_model_spec_from_string():
    mj_spec, _ = get_model_spec(model_handle, asset_handle)
    assert isinstance(mj_spec, mujoco.MjSpec)


def test_get_model_spec_from_file(model_path):
    # Mock return values
    mj_spec, _ = get_model_spec(model_path)

    assert isinstance(mj_spec, mujoco.MjSpec)


def test_get_model_spec_invalid_model_handle():
    model_handle = "/path/to/model.txt"  # Invalid extension

    with pytest.raises(TypeError, match="Spec format not recognized. Check input"):
        get_model_spec(model_handle)


def test__removeJoints_with_name():
    """Test removing joints using name list."""
    sample_mjspec, _ = get_model_spec(model_handle)
    initial_joint_count = len(sample_mjspec.joints)
    sample_mjspec, _ = filter_model_joints(sample_mjspec, ["ball_joint"], False)

    assert len(sample_mjspec.joints) == initial_joint_count - 1
    assert "joint1" not in [j.name for j in sample_mjspec.joints]


def test__removeJoints_with_equality_removal():
    """Test that equalities referencing removed joints are also removed."""
    sample_mjspec, _ = get_model_spec(model_handle)
    initial_equality_count = len(sample_mjspec.equalities)
    sample_mjspec, _ = filter_model_joints(sample_mjspec, ["hinge_joint"], False)

    assert len(sample_mjspec.equalities) == initial_equality_count - 1


def test__removeJoints_invalid_input():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(TypeError):
        filter_model_joints(123, [])


def test__removeBodies_with_name():
    """Test removing bodies using names."""
    sample_mjspec, _ = get_model_spec(model_handle)
    initial_body_count = len(sample_mjspec.bodies)
    sample_mjspec, _ = filter_model_bodies(sample_mjspec, ["body2"])

    assert len(sample_mjspec.bodies) == initial_body_count - 1
    assert "body2" not in [b.name for b in sample_mjspec.bodies]


def test__removeBodies_with_joint_removal():
    """Test that when bodies are removed the joints in the body are also removed."""
    sample_mjspec, _ = get_model_spec(model_handle)

    initial_joint_count = len(sample_mjspec.joints)
    sample_mjspec, _ = filter_model_bodies(sample_mjspec, ["body3"])

    assert len(sample_mjspec.joints) == initial_joint_count - 1
    assert "hinge_joint_2" not in [j.name for j in sample_mjspec.joints]


def test__removeBodies_invalid_input():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(TypeError):
        filter_model_bodies(123, [])  # Invalid model type


@pytest.fixture
def mocap_mjspec():
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
    spec = mujoco.MjSpec.from_file(str(model_path))
    model_path.unlink()  # Clean up temporary file
    return spec


def test_get_mocap_names(mocap_mjspec):
    spec = mocap_mjspec
    mocap_body_names = get_mocap_names(spec, None)

    assert (
        "mocap_body1" in mocap_body_names
        and "mocap_body2" in mocap_body_names
        and "not_mocap" not in mocap_body_names
    )


def test_remove_mocap_bodies(mocap_mjspec):
    spec = mocap_mjspec
    filtered_spec, _ = remove_mocap_bodies(spec, None)

    assert filtered_spec.body("mocap_body1") is None
    assert filtered_spec.body("mocap_body2") is None
    assert filtered_spec.body("not_mocap") is not None


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
        assert name2enum_jointtype(joint_name) == name2enum_jointtype(expected_type)

        # Test that the output matches the expected enum value
        assert name2enum_jointtype(joint_name) == type_enum


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
        assert name2enum_geomtype(geom_name) == name2enum_geomtype(expected_type)

        # Test that the output matches the expected enum value
        assert name2enum_geomtype(geom_name) == type_enum


def test_get_joint_names_from_string():
    """Test that get_joint_names returns all joint names from a string model."""
    joint_names = get_joint_names(model_handle, asset_handle)

    # Check that all expected joints are present
    assert "free_joint" in joint_names
    assert "ball_joint" in joint_names
    assert "hinge_joint" in joint_names
    assert "hinge_joint_2" in joint_names

    # Check that we have the correct number of joints
    assert len(joint_names) == 4


def test_get_joint_names_from_file(model_path):
    """Test that get_joint_names works with a file path."""
    joint_names = get_joint_names(model_path)

    # Should return a list of strings
    assert isinstance(joint_names, list)
    assert all(isinstance(name, str) for name in joint_names)
    assert len(joint_names) > 0


def test_get_joint_names_from_spec():
    """Test that get_joint_names works with an MjSpec object."""
    spec, _ = get_model_spec(model_handle, asset_handle)
    joint_names = get_joint_names(spec)

    # Check that all expected joints are present
    assert "free_joint" in joint_names
    assert "ball_joint" in joint_names
    assert "hinge_joint" in joint_names
    assert "hinge_joint_2" in joint_names


def test_get_joint_names_order():
    """Test that get_joint_names returns joints in the correct order."""
    joint_names = get_joint_names(model_handle, asset_handle)

    # The order should match the order in the spec
    spec, _ = get_model_spec(model_handle, asset_handle)
    expected_names = [joint.name for joint in spec.joints]

    assert joint_names == expected_names


def test_get_joint_names_empty_model():
    """Test that get_joint_names handles models with no joints."""
    empty_model = """
    <mujoco>
        <worldbody>
            <body name="body1">
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody>
    </mujoco>
"""

    joint_names = get_joint_names(empty_model)
    assert joint_names == []


def test_remove_nonmesh_geoms():
    mj_spec, _ = get_model_spec(model_handle)
    initial_geom_count = len(mj_spec.geoms)

    assert initial_geom_count > 0

    # Remove all non-mesh geoms
    mj_spec, _ = remove_nonmesh_geoms(mj_spec)

    assert len(mj_spec.geoms) < initial_geom_count


def test_remove_nonmesh_geoms_doesnt_remove_mesh():
    mj_spec, _ = get_model_spec(model_handle)

    # Add a mesh geom for testing
    body = mj_spec.body("body3")
    body.add_geom(
        name="mesh_geom",
        type=name2enum_geomtype("mesh"),
        meshname="cube_mesh",
        contype=0,
        conaffinity=0,
    )
    mj_spec.add_mesh(
        name="cube_mesh",
        file="tests/assets/meshes/cube.stl",
        scale=[0.1, 0.1, 0.1],
    )
    mj_spec.compile()

    initial_geom_count = len(mj_spec.geoms)

    assert initial_geom_count > 0

    # Remove all non-mesh geoms
    mj_spec, _ = remove_nonmesh_geoms(mj_spec, geom_names=["mesh_geom"])

    assert len(mj_spec.geoms) == initial_geom_count


def test_remove_equality_constraints():
    mj_spec, _ = get_model_spec(model_handle)
    initial_equality_count = len(mj_spec.equalities)

    assert initial_equality_count

    # Remove all equalities
    mj_spec, _ = remove_equality_constraints(mj_spec)

    assert len(mj_spec.equalities) == 0
    assert len(mj_spec.equalities) == initial_equality_count - initial_equality_count


def test_remove_sites():
    mj_spec, _ = get_model_spec(model_handle)
    initial_site_count = len(mj_spec.sites)

    assert initial_site_count > 0

    # Remove all sites
    mj_spec, _ = remove_sites(mj_spec)

    assert len(mj_spec.sites) == 0
    assert len(mj_spec.sites) == initial_site_count - initial_site_count


def test_remove_contact_pairs():
    mj_spec, _ = get_model_spec(model_handle)
    initial_geom_count = len(mj_spec.geoms)

    # Add a contact pair for testing
    if initial_geom_count >= 2:
        mj_spec.add_pair(
            name="test_pair",
            geomname1=mj_spec.geoms[0].name,
            geomname2=mj_spec.geoms[1].name,
            condim=3,
        )

    initial_contact_pair_count = len(mj_spec.pairs)

    assert initial_contact_pair_count > 0

    # Remove all contact pairs
    mj_spec, _ = remove_contact_pairs(mj_spec)

    assert len(mj_spec.pairs) == 0
    assert len(mj_spec.pairs) == initial_contact_pair_count - initial_contact_pair_count


def test_remove_sensors():
    mj_spec, _ = get_model_spec(model_handle)
    initial_sensor_count = len(mj_spec.sensors)

    assert initial_sensor_count > 0

    # Remove all sensors
    mj_spec, _ = remove_sensors(mj_spec)

    assert len(mj_spec.sensors) == 0
    assert len(mj_spec.sensors) == initial_sensor_count - initial_sensor_count
