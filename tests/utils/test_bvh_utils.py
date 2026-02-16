"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import os

import mujoco
import numpy as np
import pytest

from myo_tools.utils.file_ops.bvh_utils import (
    _axis_angle_to_mat,
    _mat_to_euler_zxy,
    _mat_to_euler_zxy_batch,
    _mjc_to_bvh_pos,
    _mjc_to_bvh_rot,
    export_mujoco_to_bvh,
)


@pytest.fixture
def assets_folder():
    return os.path.join(os.path.dirname(__file__), "..", "assets")


@pytest.fixture
def model_path(assets_folder):
    return os.path.join(assets_folder, "test_model.xml")


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def test_mjc_to_bvh_pos_forward():
    """MuJoCo X (forward) maps to BVH X (forward)."""
    p = np.array([1.0, 0.0, 0.0])
    out = _mjc_to_bvh_pos(p, scale=1.0)
    np.testing.assert_array_almost_equal(out, [1.0, 0.0, 0.0])


def test_mjc_to_bvh_pos_up():
    """MuJoCo Z (up) maps to BVH Y (up)."""
    p = np.array([0.0, 0.0, 1.0])
    out = _mjc_to_bvh_pos(p, scale=1.0)
    np.testing.assert_array_almost_equal(out, [0.0, 1.0, 0.0])


def test_mjc_to_bvh_pos_scale():
    """Scale multiplies position."""
    p = np.array([1.0, 2.0, 3.0])
    out = _mjc_to_bvh_pos(p, scale=100.0)
    np.testing.assert_array_almost_equal(out, _mjc_to_bvh_pos(p, scale=1.0) * 100.0)


def test_mjc_to_bvh_rot_identity():
    """Identity in MuJoCo stays identity in BVH frame."""
    R = np.eye(3)
    out = _mjc_to_bvh_rot(R)
    np.testing.assert_array_almost_equal(out, np.eye(3))


# ---------------------------------------------------------------------------
# ZXY Euler (single and batch)
# ---------------------------------------------------------------------------


def test_mat_to_euler_zxy_identity():
    """Identity matrix -> [0, 0, 0] degrees (Z, X, Y)."""
    R = np.eye(3)
    euler = _mat_to_euler_zxy(R)
    np.testing.assert_array_almost_equal(euler, [0.0, 0.0, 0.0])


def test_mat_to_euler_zxy_rx_90():
    """Rotation 90° about X -> Euler x ≈ 90, z,y ≈ 0."""
    R = _axis_angle_to_mat(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    euler = _mat_to_euler_zxy(R)
    np.testing.assert_array_almost_equal(euler[0], 0.0, decimal=5)
    np.testing.assert_array_almost_equal(euler[1], 90.0, decimal=5)
    np.testing.assert_array_almost_equal(euler[2], 0.0, decimal=5)


def test_mat_to_euler_zxy_batch_identity():
    """Batch: N identity matrices -> N rows of zeros."""
    R = np.tile(np.eye(3), (5, 1, 1))
    euler = _mat_to_euler_zxy_batch(R)
    assert euler.shape == (5, 3)
    np.testing.assert_array_almost_equal(euler, np.zeros((5, 3)))


def test_mat_to_euler_zxy_batch_matches_single():
    """Batch Euler matches element-wise single conversion."""
    np.random.seed(42)
    for _ in range(3):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(-np.pi, np.pi)
        R = _axis_angle_to_mat(axis, angle)
        single = _mat_to_euler_zxy(R)
        batch = _mat_to_euler_zxy_batch(R[np.newaxis, ...])
        np.testing.assert_array_almost_equal(batch[0], single)


# ---------------------------------------------------------------------------
# Export (rest pose, trajectory, scale)
# ---------------------------------------------------------------------------


def test_export_mujoco_to_bvh_rest_pose(model_path, tmp_path):
    """Export with no trajectory produces valid BVH with one frame."""
    out = tmp_path / "rest.bvh"
    export_mujoco_to_bvh(model_path, str(out), trajectories=None)
    assert out.is_file()
    text = out.read_text()
    assert "HIERARCHY" in text
    assert "MOTION" in text
    assert "Frames:\t1" in text
    assert "Frame Time:" in text
    assert "ROOT" in text or "Joint" in text


def test_export_mujoco_to_bvh_with_trajectory_dicts(model_path, tmp_path):
    """Export with list of dicts (qpos) produces BVH with correct frame count."""
    model = mujoco.MjModel.from_xml_path(model_path)
    qpos0 = model.qpos0.copy() if model.nq else np.array([])
    if model.nq == 0:
        pytest.skip("test_model has nq=0")
    trajectories = [{"qpos": qpos0}, {"qpos": qpos0}, {"qpos": qpos0}]
    out = tmp_path / "three_frames.bvh"
    export_mujoco_to_bvh(model_path, str(out), trajectories=trajectories)
    assert out.is_file()
    text = out.read_text()
    assert "Frames:\t3" in text


def test_export_mujoco_to_bvh_with_trajectory_tuples(model_path, tmp_path):
    """Export with list of (time, qpos, qvel) produces valid BVH."""
    model = mujoco.MjModel.from_xml_path(model_path)
    if model.nq == 0:
        pytest.skip("test_model has nq=0")
    qpos0 = model.qpos0.copy()
    trajectories = [
        (0.0, qpos0, np.zeros(model.nv)),
        (0.033, qpos0, np.zeros(model.nv)),
    ]
    out = tmp_path / "tuples.bvh"
    export_mujoco_to_bvh(model_path, str(out), trajectories=trajectories)
    assert out.is_file()
    text = out.read_text()
    assert "Frames:\t2" in text


def test_export_mujoco_to_bvh_scale_affects_positions(model_path, tmp_path):
    """Export with scale=100 produces position values 100x larger than scale=1."""
    out1 = tmp_path / "scale1.bvh"
    out100 = tmp_path / "scale100.bvh"
    export_mujoco_to_bvh(model_path, str(out1), trajectories=None, scale=1.0)
    export_mujoco_to_bvh(model_path, str(out100), trajectories=None, scale=100.0)
    t1 = out1.read_text()
    t100 = out100.read_text()

    # Find first OFFSET line (root or first joint)
    def first_offset(s):
        for line in s.splitlines():
            if "OFFSET" in line:
                parts = line.split()
                return [float(parts[1]), float(parts[2]), float(parts[3])]
        return None

    o1 = first_offset(t1)
    o100 = first_offset(t100)
    assert o1 is not None and o100 is not None
    np.testing.assert_array_almost_equal(np.array(o100), np.array(o1) * 100.0)


def test_export_mujoco_to_bvh_fps_in_header(model_path, tmp_path):
    """Frame Time in file equals 1/fps."""
    out = tmp_path / "fps30.bvh"
    export_mujoco_to_bvh(model_path, str(out), trajectories=None, fps=30.0)
    text = out.read_text()
    assert "Frame Time:\t0.033333" in text or "Frame Time:\t0.033334" in text


def test_export_mujoco_to_bvh_invalid_model_path(tmp_path):
    """Non-existent model path raises (or we require file to exist)."""
    out = tmp_path / "out.bvh"
    with pytest.raises(Exception):
        export_mujoco_to_bvh("/nonexistent/model.xml", str(out))
