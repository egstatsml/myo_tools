"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import json
import os
import struct
import tempfile

import mujoco
import pytest

from myo_tools.utils.file_ops.gltf_utils import (
    MujocoGLTFExporter,
    choose_dice_uv_mapping,
    export_mujoco_to_glb,
    extract_texture_image,
    find_textured_geometries,
    get_uv_mapping_for_dice_file_texture,
    make_plane_uv,
)


@pytest.fixture
def assets_folder():
    return os.path.join(os.path.dirname(__file__), "..", "assets")


@pytest.fixture
def model_path(assets_folder):
    return os.path.join(assets_folder, "test_model.xml")


@pytest.fixture
def textured_model_path(assets_folder):
    return os.path.join(assets_folder, "gltf_textured_model.xml")


def test_find_textured_geometries_no_texture(model_path):
    """Model without textures should return empty dict."""
    model = mujoco.MjModel.from_xml_path(model_path)
    textured = find_textured_geometries(model)
    assert isinstance(textured, dict)
    assert len(textured) == 0


def test_find_textured_geometries_with_texture(textured_model_path):
    """Model with textured geom should return geom_idx -> (matid, texid)."""
    model = mujoco.MjModel.from_xml_path(textured_model_path)
    textured = find_textured_geometries(model)
    assert isinstance(textured, dict)
    assert len(textured) >= 1
    for geom_idx, (matid, texid) in textured.items():
        assert isinstance(geom_idx, int)
        assert matid >= 0
        assert texid >= 0


def test_extract_texture_image_invalid(model_path):
    """Invalid texid returns None."""
    model = mujoco.MjModel.from_xml_path(model_path)
    assert extract_texture_image(model, -1) is None
    assert extract_texture_image(model, None) is None


def test_extract_texture_image_valid(textured_model_path):
    """Valid texture from model returns (H, W, C) uint8 array."""
    model = mujoco.MjModel.from_xml_path(textured_model_path)
    textured = find_textured_geometries(model)
    if not textured:
        pytest.skip("no textured geoms in model")
    (matid, texid) = next(iter(textured.values()))
    img = extract_texture_image(model, texid)
    assert img is not None
    assert img.ndim in (2, 3)
    assert img.dtype == "uint8"
    if img.ndim == 3:
        assert img.shape[2] in (1, 3, 4)


def test_make_plane_uv():
    """Plane UVs: 4 corners repeated for n vertices."""
    uv = make_plane_uv(4)
    assert uv.shape == (4, 2)
    assert uv.dtype == "float32"


def test_get_uv_mapping_for_dice_file_texture():
    """Dice file texture returns 4x3 T-shaped face mapping."""
    mapping = get_uv_mapping_for_dice_file_texture()
    assert "top" in mapping
    assert "front" in mapping
    for _face, rect in mapping.items():
        assert len(rect) == 4
        u_min, v_min, u_max, v_max = rect
        assert 0 <= u_min <= u_max <= 1
        assert 0 <= v_min <= v_max <= 1


def test_choose_dice_uv_mapping():
    """choose_dice_uv_mapping returns face -> (u_min, v_min, u_max, v_max)."""
    # 4:3 aspect -> file texture mapping
    m = choose_dice_uv_mapping((3, 4))
    assert "front" in m
    # 6:1 vertical strip
    m2 = choose_dice_uv_mapping((10, 60))
    assert "front" in m2


def test_export_mujoco_to_glb_basic(model_path):
    """Export without textures produces valid GLB with meshes."""
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tf:
        out = tf.name
    try:
        export_mujoco_to_glb(model_path, out, trajectories=None)
        assert os.path.isfile(out)
        with open(out, "rb") as f:
            data = f.read()
        # GLB: magic, version, length, then chunks
        assert len(data) >= 12
        magic, version, total = struct.unpack("<III", data[:12])
        assert magic == 0x46546C67
        assert version == 2
        assert total == len(data)
        # JSON chunk
        jlen, jtype = struct.unpack("<II", data[12:20])
        assert jtype == 0x4E4F534A
        gltf = json.loads(data[20 : 20 + jlen].decode("utf-8"))
        assert "asset" in gltf
        assert gltf["asset"].get("version") == "2.0"
        assert "meshes" in gltf
        assert "nodes" in gltf
    finally:
        if os.path.isfile(out):
            os.unlink(out)


def test_export_mujoco_to_glb_includes_textures_and_sampler(textured_model_path):
    """Export with textured model includes images, textures, samplers in GLB."""
    try:
        from PIL import Image  # noqa: F401  # used for skip check
    except ImportError:
        pytest.skip("PIL required for texture export")
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tf:
        out = tf.name
    try:
        export_mujoco_to_glb(textured_model_path, out, trajectories=None)
        assert os.path.isfile(out)
        with open(out, "rb") as f:
            data = f.read()
        jlen, jtype = struct.unpack("<II", data[12:20])
        assert jtype == 0x4E4F534A
        gltf = json.loads(data[20 : 20 + jlen].decode("utf-8"))
        # Must include images and textures when model has textures
        assert "images" in gltf, "GLB should include images when model has textures"
        assert "textures" in gltf, "GLB should include textures when model has textures"
        assert len(gltf["images"]) >= 1
        assert len(gltf["textures"]) >= 1
        # Samplers added for proper texture display
        assert "samplers" in gltf
        assert len(gltf["samplers"]) >= 1
        # Each texture should reference image and sampler
        for tex in gltf["textures"]:
            assert "source" in tex
            assert "sampler" in tex
    finally:
        if os.path.isfile(out):
            os.unlink(out)


def test_exporter_textured_geom_info(textured_model_path):
    """Exporter finds textured geometries on init."""
    model = mujoco.MjModel.from_xml_path(textured_model_path)
    exporter = MujocoGLTFExporter(model)
    assert hasattr(exporter, "textured_geom_info")
    assert isinstance(exporter.textured_geom_info, dict)
