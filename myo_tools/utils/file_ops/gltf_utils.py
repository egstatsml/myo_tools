"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import io
import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np

from myo_tools.utils.tensor_ops.quat_utils import mat2quat, quat2mat, rotVecMat

# MuJoCo geometry type for readability
mjtGeom = mujoco.mjtGeom

# MuJoCo uses Z-up; glTF uses Y-up. Single root node applies this rotation so the
# world (e.g. ground plane) is oriented correctly; body transforms stay in MuJoCo coords.
# R_MJC_TO_GLTF: gltf_xyz = R @ mjc_xyz  =>  (x, z, -y)
R_MJC_TO_GLTF = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# -----------------------------------------------------------------------------
# Texture discovery and extraction (aligned with reference)
# -----------------------------------------------------------------------------


def find_textured_geometries(model: mujoco.MjModel) -> Dict[int, Tuple[int, int]]:
    """Find geoms that have a texture and return their material and texture ids.

    Args:
        model: MuJoCo model.

    Returns:
        Dict mapping geom index to (matid, texid).
    """
    textured = {}
    for i in range(model.ngeom):
        matid = int(model.geom_matid[i]) if i < len(model.geom_matid) else -1
        if matid < 0:
            continue
        rgb = int(model.mat_texid[matid, mujoco.mjtTextureRole.mjTEXROLE_RGB])
        rgba = int(model.mat_texid[matid, mujoco.mjtTextureRole.mjTEXROLE_RGBA])
        texid = rgb if rgb >= 0 else rgba
        if texid >= 0:
            textured[i] = (matid, texid)
    return textured


def extract_texture_image(
    model: mujoco.MjModel, texid: Optional[int]
) -> Optional[np.ndarray]:
    """Extract texture image from the model.

    Args:
        model: MuJoCo model.
        texid: Texture id, or None.

    Returns:
        Texture as (H, W, C) uint8 array, or None if texid is invalid.
    """
    if texid is None or int(texid) < 0:
        return None
    texid = int(texid)
    adr = model.tex_adr[texid]
    w = model.tex_width[texid]
    h = model.tex_height[texid]
    c = model.tex_nchannel[texid]
    data = model.tex_data[adr : adr + w * h * c]
    img = data.reshape(h, w, c).astype(np.uint8)
    return img


def make_plane_uv(n: int) -> np.ndarray:
    """Build UVs for a quad plane (4 corners repeated to match vertex count).

    Args:
        n: Number of vertices (e.g. 4 for a single quad).

    Returns:
        (n, 2) float32 array of UV coordinates.
    """
    base = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return np.tile(base, (n // 4 + 1, 1))[:n]


# Dice-style box: 24 vertices (4 per face), face order for UV mapping
_DICE_FACE_DEFS_NAMED = {
    "front": [4, 5, 6, 7],  # Z+
    "back": [1, 0, 3, 2],  # Z-
    "right": [5, 1, 2, 6],  # X+
    "left": [0, 4, 7, 3],  # X-
    "top": [7, 6, 2, 3],  # Y+
    "bottom": [4, 0, 1, 5],  # Y-
}


def get_uv_mapping_for_dice_file_texture() -> (
    Dict[str, Tuple[float, float, float, float]]
):
    """UV mapping for 4x3 T-shaped grid (file texture).

    Returns:
        Dict mapping face name to (u_min, v_min, u_max, v_max).
    """
    rows, cols = 3, 4
    cw, ch = 1.0 / cols, 1.0 / rows
    positions = {
        "top": (2, 0),
        "left": (0, 1),
        "front": (1, 1),
        "right": (2, 1),
        "back": (3, 1),
        "bottom": (2, 2),
    }
    uv_mapping = {}
    for face, (col, row) in positions.items():
        uv_mapping[face] = (col * cw, row * ch, (col + 1) * cw, (row + 1) * ch)
    return uv_mapping


def get_uv_mapping_for_dice_model_texture_vertical_strip() -> (
    Dict[str, Tuple[float, float, float, float]]
):
    """UV mapping for 6x1 vertical strip (MuJoCo model texture): L, R, F, B, U, D.

    Returns:
        Dict mapping face name to (u_min, v_min, u_max, v_max).
    """
    face_order = ["left", "right", "front", "back", "top", "bottom"]
    uv_mapping = {}
    for i, face in enumerate(face_order):
        uv_mapping[face] = (0.0, i / 6.0, 1.0, (i + 1) / 6.0)
    return uv_mapping


def choose_dice_uv_mapping(
    texture_shape: Union[Tuple[int, ...], List[int]],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Choose 4x3 or 6x1 dice UV mapping from texture shape.

    Args:
        texture_shape: (height, width) or (h, w, c) from texture array shape.

    Returns:
        Dict mapping face name to (u_min, v_min, u_max, v_max).
    """
    if len(texture_shape) >= 2:
        h, w = texture_shape[0], texture_shape[1]
    else:
        return get_uv_mapping_for_dice_file_texture()
    if h == 0:
        return get_uv_mapping_for_dice_file_texture()
    aspect = w / h
    if abs(aspect - (4.0 / 3.0)) < 1e-3:
        return get_uv_mapping_for_dice_file_texture()
    if abs(aspect - (1.0 / 6.0)) < 1e-3:
        return get_uv_mapping_for_dice_model_texture_vertical_strip()
    if abs(aspect - 6.0) < 1e-3:
        face_order = ["left", "right", "front", "back", "top", "bottom"]
        return {
            face: (i / 6.0, 0.0, (i + 1) / 6.0, 1.0)
            for i, face in enumerate(face_order)
        }
    return get_uv_mapping_for_dice_file_texture()


class MujocoGLTFExporter:
    """
    MuJoCo Geometry Type Constants (from mjtGeom enum):
    - mjGEOM_PLANE     = 0
    - mjGEOM_HFIELD    = 1
    - mjGEOM_SPHERE    = 2
    - mjGEOM_CAPSULE   = 3
    - mjGEOM_ELLIPSOID = 4
    - mjGEOM_CYLINDER  = 5
    - mjGEOM_BOX       = 6
    - mjGEOM_MESH      = 7
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize exporter with a MuJoCo model.

        Args:
            model: MuJoCo MjModel to export.
        """
        self.model = model

        # glTF containers
        self.buffers = []
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.animations = []
        self.scenes = []
        self.materials = []
        self.images = []
        self.textures = []
        self.samplers = []

        # Binary data accumulator
        self.binary_data = bytearray()

        # Track which bodies have meshes (can have multiple per body)
        self.body_to_meshes = {}  # body_id -> list of mesh indices

        # Texture discovery: geom_idx -> (matid, texid)
        self.textured_geom_info = find_textured_geometries(model)
        # Cache: (texid,) -> material index
        self._texid_to_material = {}
        # Default texture sampler (ensured when first texture is added)
        self._default_sampler_index = None

    def _transform_vertices_by_geom(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        geom_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform vertices and normals from geom local space to body space.

        MuJoCo geom_pos, geom_quat are in body frame (geom relative to body).
        Uses quat_math for consistent rotation.

        Args:
            vertices: (N, 3) vertices in geom local space.
            normals: (N, 3) normals in geom local space.
            geom_id: Geom index.

        Returns:
            Tuple of (vertices, normals) in body space, float32.
        """
        geom_pos = self.model.geom_pos[geom_id].copy()
        geom_quat = self.model.geom_quat[geom_id].copy()  # MuJoCo: (w,x,y,z)
        rot_mat = quat2mat(geom_quat)  # 3x3
        vertices_out = np.zeros_like(vertices)
        for i in range(len(vertices)):
            vertices_out[i] = rotVecMat(vertices[i], rot_mat) + geom_pos
        normals_out = np.zeros_like(normals)
        for i in range(len(normals)):
            normals_out[i] = rotVecMat(normals[i], rot_mat)
            n = np.linalg.norm(normals_out[i])
            if n > 1e-8:
                normals_out[i] /= n
        return vertices_out.astype(np.float32), normals_out.astype(np.float32)

    def add_buffer_view(
        self, byte_length: int, byte_offset: int, target: Optional[int] = None
    ) -> int:
        """Add a glTF buffer view.

        Args:
            byte_length: Length in bytes.
            byte_offset: Offset into the binary buffer.
            target: Optional glTF target (e.g. 34962 ARRAY_BUFFER, 34963 ELEMENT_ARRAY_BUFFER).

        Returns:
            Index of the new buffer view.
        """
        view = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length,
        }
        if target:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(
        self,
        buffer_view: int,
        component_type: int,
        count: int,
        accessor_type: str,
        min_vals: Optional[List[float]] = None,
        max_vals: Optional[List[float]] = None,
    ) -> int:
        """Add a glTF accessor.

        Args:
            buffer_view: Buffer view index.
            component_type: glTF component type (e.g. 5126 FLOAT).
            count: Number of elements.
            accessor_type: glTF type string (e.g. "VEC3", "SCALAR").
            min_vals: Optional min bounds.
            max_vals: Optional max bounds.

        Returns:
            Index of the new accessor.
        """
        accessor = {
            "bufferView": buffer_view,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if min_vals is not None:
            accessor["min"] = min_vals
        if max_vals is not None:
            accessor["max"] = max_vals
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def append_binary_data(self, data: Union[bytes, bytearray]) -> int:
        """Append binary data to the GLB buffer (4-byte aligned).

        Args:
            data: Raw bytes to append.

        Returns:
            Byte offset of the appended data.
        """
        offset = len(self.binary_data)
        self.binary_data.extend(data)
        # Align to 4-byte boundary
        while len(self.binary_data) % 4 != 0:
            self.binary_data.append(0)
        return offset

    def _ensure_default_sampler(self) -> int:
        """Ensure a default texture sampler exists.

        Returns:
            Index of the default sampler.
        """
        if self._default_sampler_index is not None:
            return self._default_sampler_index
        # glTF 2.0: magFilter 9729=LINEAR, minFilter 9987=LINEAR_MIPMAP_LINEAR, wrap 10497=REPEAT
        self.samplers.append(
            {
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497,
            }
        )
        self._default_sampler_index = len(self.samplers) - 1
        return self._default_sampler_index

    def _add_texture_material(
        self, texid: int, rgba: Optional[np.ndarray] = None
    ) -> int:
        """Add texture from model to glTF as image + texture + material.

        Caches by texid so the same texture is not added twice.

        Args:
            texid: Model texture id.
            rgba: Optional (4,) base color factor for the material.

        Returns:
            glTF material index, or -1 if texture could not be added.
        """
        if texid in self._texid_to_material:
            return self._texid_to_material[texid]
        img = extract_texture_image(self.model, texid)
        if img is None:
            return -1
        try:
            from PIL import Image as PILImage
        except Exception as e:
            print(
                f"PIL failed: {e} -- skipping texture. \n to install: pip install Pillow"
            )
            return -1
        # img is (H, W, C) uint8; PIL expects (H, W) or (H, W, C)
        if img.ndim == 2:
            pil_img = PILImage.fromarray(img, mode="L")
        elif img.ndim == 3 and img.shape[2] == 1:
            pil_img = PILImage.fromarray(img.squeeze(-1), mode="L")
        elif img.ndim == 3 and img.shape[2] == 3:
            pil_img = PILImage.fromarray(img, mode="RGB")
        else:
            pil_img = PILImage.fromarray(img, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        byte_offset = self.append_binary_data(png_bytes)
        # bufferView for image (no target)
        bv = self.add_buffer_view(len(png_bytes), byte_offset)
        # glTF image
        img_idx = len(self.images)
        self.images.append({"bufferView": bv, "mimeType": "image/png"})
        # Default sampler so viewers that expect it get valid texture sampling
        sampler_idx = self._ensure_default_sampler()
        # glTF texture (source = image index, sampler = sampler index)
        tex_idx = len(self.textures)
        self.textures.append({"source": img_idx, "sampler": sampler_idx})
        # glTF material (metallic-roughness, baseColorTexture)
        mat_idx = len(self.materials)
        pbr = {
            "baseColorTexture": {"index": tex_idx},
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        }
        if rgba is not None and len(rgba) >= 4:
            pbr["baseColorFactor"] = [
                float(rgba[0]),
                float(rgba[1]),
                float(rgba[2]),
                float(rgba[3]),
            ]
        self.materials.append({"pbrMetallicRoughness": pbr})
        self._texid_to_material[texid] = mat_idx
        return mat_idx

    def export_geometry(self) -> Dict[str, int]:
        """Export mesh and primitive geometry from the MuJoCo model.

        Returns:
            Dict mapping geometry type name to count (e.g. "BOX": 3).
        """
        geom_counts = {}

        for geom_id in range(self.model.ngeom):
            geom_type = self.model.geom_type[geom_id]
            body_id = self.model.geom_bodyid[geom_id]

            # Track geometry types
            type_name = self._get_geom_type_name(geom_type)
            geom_counts[type_name] = geom_counts.get(type_name, 0) + 1

            # Handle mesh geometries (type 7 = mjGEOM_MESH)
            if geom_type == 0:  # mjGEOM_PLANE
                mesh_idx = self._export_plane_geom(geom_id)
                if mesh_idx is not None:
                    if body_id not in self.body_to_meshes:
                        self.body_to_meshes[body_id] = []
                    self.body_to_meshes[body_id].append(mesh_idx)
                continue
            elif geom_type == 7:  # mjGEOM_MESH - CORRECT!
                mesh_id = self.model.geom_dataid[geom_id]
                if mesh_id >= 0:
                    mesh_idx = self._export_mesh(mesh_id, geom_id)
                    # Add to body's mesh list
                    if body_id not in self.body_to_meshes:
                        self.body_to_meshes[body_id] = []
                    self.body_to_meshes[body_id].append(mesh_idx)
            else:
                # Handle primitive geometries (box, sphere, cylinder, etc.)
                mesh_idx = self._export_primitive_geom(geom_id)
                if mesh_idx is not None:
                    # Add to body's mesh list
                    if body_id not in self.body_to_meshes:
                        self.body_to_meshes[body_id] = []
                    self.body_to_meshes[body_id].append(mesh_idx)

        return geom_counts

    def _get_geom_type_name(self, geom_type: int) -> str:
        """Get human-readable geometry type name.

        Args:
            geom_type: MuJoCo mjtGeom value.

        Returns:
            String like "BOX", "MESH", etc.
        """
        types = {
            0: "PLANE",
            1: "HFIELD",
            2: "SPHERE",
            3: "CAPSULE",
            4: "ELLIPSOID",
            5: "CYLINDER",
            6: "BOX",
            7: "MESH",
        }
        return types.get(geom_type, f"UNKNOWN({geom_type})")

    def _export_plane_geom(self, geom_id: int) -> Optional[int]:
        """Export plane as a finite quad with optional texture.

        Geom transform (pos/quat) is applied. Returns glTF mesh index or None.

        Args:
            geom_id: Geom index.

        Returns:
            Mesh index, or None.
        """
        matid, texid = self.textured_geom_info.get(geom_id, (-1, -1))
        size = self.model.geom_size[geom_id]
        s = max(float(size[0]), float(size[1]), 1.0) * 2
        # Quad in local XY: 4 vertices, 2 triangles
        vertices = np.array(
            [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        uv = make_plane_uv(4)
        normals = self._compute_normals(vertices, faces)
        vertices, normals = self._transform_vertices_by_geom(vertices, normals, geom_id)
        material_index = -1
        if texid >= 0:
            rgba = None
            if 0 <= matid < len(self.model.mat_rgba):
                rgba = self.model.mat_rgba[matid]
            material_index = self._add_texture_material(texid, rgba=rgba)

        return self._create_mesh_from_vertices(
            vertices, faces, normals=normals, uv=uv, material_index=material_index
        )

    def _export_primitive_geom(self, geom_id: int) -> Optional[int]:
        """Export primitive geometry with geom transform applied.

        Supports box, sphere, cylinder, capsule, ellipsoid. Height fields are skipped.

        Args:
            geom_id: Geom index.

        Returns:
            Mesh index, or None if unsupported.
        """
        geom_type = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id]
        matid, texid = self.textured_geom_info.get(geom_id, (-1, -1))
        # Box with texture: 24-vertex box with per-face UVs (dice mapping)
        if geom_type == 6:  # mjGEOM_BOX
            if texid >= 0:
                img = extract_texture_image(self.model, texid)
                if img is not None:
                    uv_mapping = choose_dice_uv_mapping(img.shape)
                    vertices, faces, uvs = self._create_box_mesh_with_uvs(
                        size, uv_mapping
                    )
                    normals = self._compute_normals(vertices, faces)
                    vertices, normals = self._transform_vertices_by_geom(
                        vertices, normals, geom_id
                    )
                    rgba = None
                    if 0 <= matid < len(self.model.mat_rgba):
                        rgba = self.model.mat_rgba[matid]
                    material_index = self._add_texture_material(texid, rgba=rgba)
                    print(
                        f"geom_id: {geom_id}, texid: {texid}, matid: {matid} material_index: {material_index}"
                    )
                    return self._create_mesh_from_vertices(
                        vertices,
                        faces,
                        normals=normals,
                        uv=uvs,
                        material_index=material_index,
                    )
            vertices, faces = self._create_box_mesh(size)
        elif geom_type == 2:  # mjGEOM_SPHERE
            vertices, faces = self._create_sphere_mesh(size[0])
        elif geom_type == 3:  # mjGEOM_CAPSULE
            vertices, faces = self._create_capsule_mesh(size[0], size[1])
        elif geom_type == 4:  # mjGEOM_ELLIPSOID
            vertices, faces = self._create_sphere_mesh(size[0])  # Simplified
        elif geom_type == 5:  # mjGEOM_CYLINDER
            vertices, faces = self._create_cylinder_mesh(size[0], size[1])
        elif geom_type == 1:  # mjGEOM_HFIELD (height field)
            return None  # Not supported yet
        else:
            return None

        # Transform from geom local space to body space using geom_pos, geom_quat
        normals = self._compute_normals(vertices, faces)
        vertices, normals = self._transform_vertices_by_geom(vertices, normals, geom_id)

        return self._create_mesh_from_vertices(vertices, faces, normals=normals)

    def _create_box_mesh(self, size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create box mesh vertices and faces (8 shared vertices).

        Args:
            size: (3,) half-extents [sx, sy, sz].

        Returns:
            (vertices, faces) in geom local space.
        """
        sx, sy, sz = size[0], size[1], size[2]

        vertices = np.array(
            [
                [-sx, -sy, -sz],
                [sx, -sy, -sz],
                [sx, sy, -sz],
                [-sx, sy, -sz],  # Bottom
                [-sx, -sy, sz],
                [sx, -sy, sz],
                [sx, sy, sz],
                [-sx, sy, sz],  # Top
            ],
            dtype=np.float32,
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom
                [4, 6, 5],
                [4, 7, 6],  # Top
                [0, 4, 5],
                [0, 5, 1],  # Front
                [2, 6, 7],
                [2, 7, 3],  # Back
                [1, 5, 6],
                [1, 6, 2],  # Right
                [0, 3, 7],
                [0, 7, 4],  # Left
            ],
            dtype=np.uint32,
        )

        return vertices, faces

    def _create_box_mesh_with_uvs(
        self,
        size: np.ndarray,
        uv_mapping: Dict[str, Tuple[float, float, float, float]],
        flip_uv_horizontal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create box with 24 vertices (4 per face) and per-face UVs for textures.

        Args:
            size: (3,) half-extents [sx, sy, sz].
            uv_mapping: Dict face_name -> (u_min, v_min, u_max, v_max).
            flip_uv_horizontal: Whether to flip U on each face.

        Returns:
            (vertices, faces, uvs) in geom local space.
        """
        sx, sy, sz = size[0], size[1], size[2]
        corners = np.array(
            [
                [-sx, -sy, -sz],
                [sx, -sy, -sz],
                [sx, sy, -sz],
                [-sx, sy, -sz],
                [-sx, -sy, sz],
                [sx, -sy, sz],
                [sx, sy, sz],
                [-sx, sy, sz],
            ],
            dtype=np.float32,
        )
        vertices = []
        faces = []
        uvs = []
        vo = 0
        for face_name, vert_indices in _DICE_FACE_DEFS_NAMED.items():
            face_verts = corners[vert_indices]
            vertices.append(face_verts)
            faces.append([vo + 0, vo + 1, vo + 2])
            faces.append([vo + 0, vo + 2, vo + 3])
            u_min, v_min, u_max, v_max = uv_mapping.get(face_name, (0.0, 0.0, 1.0, 1.0))
            if flip_uv_horizontal:
                face_uvs = [
                    (u_max, v_max),
                    (u_min, v_max),
                    (u_min, v_min),
                    (u_max, v_min),
                ]
            else:
                face_uvs = [
                    (u_min, v_max),
                    (u_max, v_max),
                    (u_max, v_min),
                    (u_min, v_min),
                ]
            uvs.extend(face_uvs)
            vo += 4
        vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.array(faces, dtype=np.uint32)
        uvs = np.array(uvs, dtype=np.float32)
        return vertices, faces, uvs

    def _create_sphere_mesh(
        self, radius: float, segments: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sphere mesh vertices and faces.

        Args:
            radius: Sphere radius.
            segments: Latitude/longitude segments.

        Returns:
            (vertices, faces) in geom local space.
        """
        vertices = []

        for i in range(segments + 1):
            theta = i * np.pi / segments
            for j in range(segments):
                phi = j * 2 * np.pi / segments
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                vertices.append([x, y, z])

        vertices = np.array(vertices, dtype=np.float32)

        faces = []
        for i in range(segments):
            for j in range(segments):
                a = i * segments + j
                b = a + segments
                c = a + 1
                d = b + 1

                if i < segments - 1:
                    faces.append([a, b, c])
                    faces.append([c, b, d])

        faces = np.array(faces, dtype=np.uint32)
        return vertices, faces

    def _create_cylinder_mesh(
        self, radius: float, half_height: float, segments: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create cylinder mesh vertices and faces.

        Args:
            radius: Cylinder radius.
            half_height: Half-length along Z.
            segments: Angular segments.

        Returns:
            (vertices, faces) in geom local space.
        """
        vertices = []

        # Bottom circle
        for i in range(segments):
            angle = i * 2 * np.pi / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, -half_height])

        # Top circle
        for i in range(segments):
            angle = i * 2 * np.pi / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, half_height])

        vertices = np.array(vertices, dtype=np.float32)

        faces = []
        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([i, i + segments, next_i])
            faces.append([next_i, i + segments, next_i + segments])

        faces = np.array(faces, dtype=np.uint32)
        return vertices, faces

    def _create_capsule_mesh(
        self, radius: float, half_height: float, segments: int = 12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create capsule mesh vertices and faces (cylinder with caps).

        Args:
            radius: Capsule radius.
            half_height: Half-length of cylinder.
            segments: Angular segments.

        Returns:
            (vertices, faces) in geom local space.
        """
        # Simplified capsule as cylinder with hemisphere caps
        vertices, faces = self._create_cylinder_mesh(radius, half_height, segments)
        return vertices, faces

    def _create_mesh_from_vertices(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        uv: Optional[np.ndarray] = None,
        material_index: Optional[int] = None,
    ) -> int:
        """Create glTF mesh from vertices and faces.

        Args:
            vertices: (N, 3) positions.
            faces: (M, 3) triangle indices.
            normals: Optional (N, 3) normals; computed if None.
            uv: Optional (N, 2) TEXCOORD_0.
            material_index: Optional glTF material index.

        Returns:
            Index of the created mesh.
        """
        # Compute normals if not provided
        if normals is None:
            normals = self._compute_normals(vertices, faces)

        # Convert to correct format
        vertices = vertices.astype(np.float32)
        normals = normals.astype(np.float32)
        faces = faces.astype(np.uint32).flatten()

        # Add to binary buffer
        vert_offset = self.append_binary_data(vertices.tobytes())
        norm_offset = self.append_binary_data(normals.tobytes())
        indices_offset = self.append_binary_data(faces.tobytes())

        # Create buffer views
        vert_view = self.add_buffer_view(vertices.nbytes, vert_offset, 34962)
        norm_view = self.add_buffer_view(normals.nbytes, norm_offset, 34962)
        indices_view = self.add_buffer_view(faces.nbytes, indices_offset, 34963)

        # Create accessors
        vert_min = vertices.min(axis=0).tolist()
        vert_max = vertices.max(axis=0).tolist()
        pos_accessor = self.add_accessor(
            vert_view, 5126, len(vertices), "VEC3", vert_min, vert_max
        )
        norm_accessor = self.add_accessor(norm_view, 5126, len(normals), "VEC3")
        indices_accessor = self.add_accessor(indices_view, 5125, len(faces), "SCALAR")

        prim = {
            "attributes": {
                "POSITION": pos_accessor,
                "NORMAL": norm_accessor,
            },
            "indices": indices_accessor,
            "mode": 4,  # TRIANGLES
        }
        if uv is not None:
            uv = np.asarray(uv, dtype=np.float32)
            uv_offset = self.append_binary_data(uv.tobytes())
            uv_view = self.add_buffer_view(uv.nbytes, uv_offset, 34962)
            uv_accessor = self.add_accessor(uv_view, 5126, len(uv), "VEC2")
            prim["attributes"]["TEXCOORD_0"] = uv_accessor
        if material_index is not None and material_index >= 0:
            prim["material"] = material_index

        mesh = {"primitives": [prim]}
        mesh_idx = len(self.meshes)
        self.meshes.append(mesh)

        return mesh_idx

    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute vertex normals from triangle faces.

        Args:
            vertices: (N, 3) positions.
            faces: (M, 3) triangle indices.

        Returns:
            (N, 3) unit normals, float32.
        """
        normals = np.zeros_like(vertices, dtype=np.float32)

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            normals[face[0]] += normal
            normals[face[1]] += normal
            normals[face[2]] += normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals /= norms

        return normals

    def _export_mesh(self, mesh_id: int, geom_id: int) -> int:
        """Export a MuJoCo mesh with geom transform applied to body space.

        Uses mesh_texcoord for UVs when the geom has a texture.

        Args:
            mesh_id: Model mesh id (mesh_vertadr, etc.).
            geom_id: Geom index (for transform and texture).

        Returns:
            Index of the created glTF mesh.
        """
        # Get mesh data
        vert_start = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        face_start = self.model.mesh_faceadr[mesh_id]
        face_num = self.model.mesh_facenum[mesh_id]

        # Get vertices and faces in mesh local space
        vertices = self.model.mesh_vert[vert_start : vert_start + vert_num].copy()
        faces = self.model.mesh_face[face_start : face_start + face_num].copy()

        # Mesh UVs: use when available (e.g. mesh_texcoordadr in MuJoCo 3.x)
        uv = None
        material_index = -1
        matid, texid = self.textured_geom_info.get(geom_id, (-1, -1))
        if texid >= 0:
            texcoordadr = getattr(self.model, "mesh_texcoordadr", None)
            if texcoordadr is not None:
                tc_start = texcoordadr[mesh_id]
                if tc_start >= 0 and hasattr(self.model, "mesh_texcoord"):
                    tc_num = vert_num  # one UV per vertex
                    uv = self.model.mesh_texcoord[tc_start : tc_start + tc_num].copy()
            if uv is not None and len(uv) == len(vertices):
                rgba = None
                if 0 <= matid < len(self.model.mat_rgba):
                    rgba = self.model.mat_rgba[matid]
                material_index = self._add_texture_material(texid, rgba=rgba)

        # Compute normals in local space, then transform vertices and normals to body space
        normals = self._compute_normals(vertices, faces)
        vertices, normals = self._transform_vertices_by_geom(vertices, normals, geom_id)

        return self._create_mesh_from_vertices(
            vertices, faces, normals=normals, uv=uv, material_index=material_index
        )

    def export_skeleton(self) -> None:
        """Export skeleton hierarchy from MuJoCo bodies and build glTF nodes/scene.

        Adds a Z-up to Y-up conversion root so the scene is correct in glTF viewers.
        """
        # Validate hierarchy first - check for cycles
        if not self._validate_hierarchy():
            raise ValueError("Invalid body hierarchy detected - contains cycles")

        # Create nodes for each body
        for body_id in range(self.model.nbody):
            body_name = self.model.body(body_id).name
            if not body_name:
                body_name = f"body_{body_id}"

            node = {"name": body_name}

            # Get body transform (relative to parent body; MuJoCo body_pos, body_quat are wxyz)
            pos = self.model.body_pos[body_id].copy()
            quat = self.model.body_quat[body_id].copy()  # (w,x,y,z)

            # Only add transform if not identity. glTF rotation is xyzw.
            if not np.allclose(pos, 0):
                node["translation"] = pos.tolist()
            if not np.allclose(quat, [1, 0, 0, 0]):
                node["rotation"] = [quat[1], quat[2], quat[3], quat[0]]  # xyzw for glTF

            # Add mesh if this body has one
            if body_id in self.body_to_meshes:
                mesh_indices = self.body_to_meshes[body_id]

                if len(mesh_indices) == 1:
                    # Single mesh - attach directly to node
                    node["mesh"] = mesh_indices[0]
                else:
                    # Multiple meshes - create child nodes for additional meshes
                    # First mesh goes on the body node
                    node["mesh"] = mesh_indices[0]

                    # Store info about additional meshes to create child nodes later
                    if not hasattr(self, "_multi_mesh_bodies"):
                        self._multi_mesh_bodies = []
                    self._multi_mesh_bodies.append(
                        (body_id, body_name, mesh_indices[1:])
                    )

            # Add children - but only valid children
            children = []
            for i in range(self.model.nbody):
                parent_id = self.model.body_parentid[i]
                # Add i as child if:
                # 1. Its parent is this body (parent_id == body_id)
                # 2. It's not the same as this body (i != body_id) - prevents self-reference
                if parent_id == body_id and i != body_id:
                    children.append(i)

            # Safety check: ensure no self-references
            if body_id in children:
                children.remove(body_id)
                print(f"Warning: Removed self-reference from body {body_id}")

            if children:
                node["children"] = children

            self.nodes.append(node)

        # Create child nodes for bodies with multiple meshes
        if hasattr(self, "_multi_mesh_bodies"):
            for body_id, body_name, extra_meshes in self._multi_mesh_bodies:
                # Get the body's node and add children for extra meshes
                body_node = self.nodes[body_id]
                extra_children = []

                for i, mesh_idx in enumerate(extra_meshes, 1):
                    child_idx = len(self.nodes)
                    extra_children.append(child_idx)

                    # Create child node for this mesh
                    self.nodes.append(
                        {"name": f"{body_name}_geom_{i}", "mesh": mesh_idx}
                    )

                # Add extra children to body's children list
                if "children" in body_node:
                    body_node["children"].extend(extra_children)
                else:
                    body_node["children"] = extra_children

            delattr(self, "_multi_mesh_bodies")

        # Single world root: apply Z-up -> Y-up rotation so ground is horizontal in glTF viewers.
        # All body nodes (0..nbody-1 and any extra mesh children) stay in MuJoCo coords.
        # No translation here — model keeps its vertical position from the MuJoCo file.
        q_conv = mat2quat(R_MJC_TO_GLTF)  # (w,x,y,z)
        conversion_root = {
            "name": "Scene",
            "rotation": [
                float(q_conv[1]),
                float(q_conv[2]),
                float(q_conv[3]),
                float(q_conv[0]),
            ],  # xyzw for glTF
            "children": [0],
        }
        conversion_root_idx = len(self.nodes)
        self.nodes.append(conversion_root)

        self.scenes.append(
            {
                "name": "Scene",
                "nodes": [conversion_root_idx],
            }
        )

    def _validate_hierarchy(self) -> bool:
        """Validate that the body hierarchy has no cycles.

        Returns:
            True if all bodies can reach world without cycles, False otherwise.
        """
        # Check each body can reach world without cycles
        for body_id in range(self.model.nbody):
            if not self._can_reach_root(body_id, max_depth=self.model.nbody):
                print(
                    f"Warning: Body {body_id} ({self.model.body(body_id).name}) cannot reach root or forms a cycle"
                )
                return False
        return True

    def _can_reach_root(self, body_id: int, max_depth: int) -> bool:
        """Check if a body can reach the world body without cycles.

        Args:
            body_id: Body index.
            max_depth: Max parent steps to avoid infinite loops.

        Returns:
            True if a path to world (body 0) exists with no cycle.
        """
        visited = set()
        current = body_id
        depth = 0

        while current != 0 and depth < max_depth:
            if current in visited:
                # Cycle detected
                return False
            visited.add(current)
            current = self.model.body_parentid[current]
            depth += 1

        return current == 0

    def export_animation(
        self,
        trajectories: List[Union[Tuple[float, np.ndarray, Any], Dict[str, Any]]],
        fps: float = 30.0,
    ) -> None:
        """Export animation from trajectories into glTF animation channels.

        Args:
            trajectories: List of (time, qpos, qvel) tuples or dicts with "qpos" key.
            fps: Frames per second for the animation (default 30.0). Used when
                trajectories are dicts to build the time axis.

        Returns:
            None. Animation data is appended to self.animations.
        """
        if not trajectories:
            return

        # Parse trajectories
        if isinstance(trajectories[0], dict):
            qpos_data = [t["qpos"] for t in trajectories]
            times = np.arange(len(qpos_data)) / fps
        else:
            times = np.array([t[0] for t in trajectories])
            qpos_data = [t[1] for t in trajectories]
        times = times.astype(np.float32)

        # Per-body rotation and translation from qpos (shared logic in qpos_to_pose).
        # Precompute once per frame (not per body×frame) so cost is O(frames) not O(bodies×frames).
        from myo_tools.utils.file_ops.qpos_to_pose import body_poses_from_qpos

        body_ids = [
            bid
            for bid in range(1, self.model.nbody)
            if self.model.body_jntadr[bid] >= 0
        ]
        per_frame_quats = []
        per_frame_trans = []
        for qpos in qpos_data:
            qs, ts = body_poses_from_qpos(self.model, qpos)
            per_frame_quats.append(qs)
            per_frame_trans.append(ts)

        node_animations = {}
        for body_id in body_ids:
            rotation_list = []
            translation_list = []
            for f in range(len(qpos_data)):
                qs, ts = per_frame_quats[f], per_frame_trans[f]
                if body_id not in qs:
                    continue
                q = qs[body_id]
                t = ts[body_id]
                rotation_list.append(
                    [float(q[1]), float(q[2]), float(q[3]), float(q[0])]
                )  # xyzw for glTF
                translation_list.append(t.astype(np.float32).tolist())
            if rotation_list:
                node_animations[body_id] = {
                    "rotation": np.array(rotation_list, dtype=np.float32),
                    "translation": np.array(translation_list, dtype=np.float32),
                }

        # Create animation channels and samplers
        if not node_animations:
            return

        channels = []
        samplers = []

        for node_id, anim_data in node_animations.items():
            for path, values in anim_data.items():
                # Add times accessor
                times_offset = self.append_binary_data(times.tobytes())
                times_view = self.add_buffer_view(times.nbytes, times_offset)
                times_accessor = self.add_accessor(
                    times_view,
                    5126,
                    len(times),
                    "SCALAR",
                    [float(times.min())],
                    [float(times.max())],
                )

                # Add values accessor
                values_offset = self.append_binary_data(values.tobytes())
                values_view = self.add_buffer_view(values.nbytes, values_offset)

                value_type = "VEC3" if path == "translation" else "VEC4"
                values_accessor = self.add_accessor(
                    values_view, 5126, len(values), value_type
                )

                # Create sampler
                sampler_idx = len(samplers)
                samplers.append(
                    {
                        "input": times_accessor,
                        "output": values_accessor,
                        "interpolation": "LINEAR",
                    }
                )

                # Create channel
                channels.append(
                    {
                        "sampler": sampler_idx,
                        "target": {
                            "node": node_id,
                            "path": path,
                        },
                    }
                )

        if channels:
            self.animations.append(
                {
                    "name": "Animation",
                    "channels": channels,
                    "samplers": samplers,
                }
            )

    def export_glb(
        self,
        output_path: str,
        trajectories: Optional[
            List[Union[Tuple[float, np.ndarray, Any], Dict[str, Any]]]
        ] = None,
        fps: float = 30.0,
        verbose: bool = False,
    ) -> None:
        """Export model and optional animation as a GLB file.

        Args:
            output_path: Path to output GLB file.
            trajectories: Optional list of (time, qpos, qvel) or dicts with "qpos".
            fps: Frames per second for animation (default 30.0).
            verbose: If True, print model info, geometry counts, and glTF structure.

        Returns:
            None. Writes the GLB to output_path.
        """
        if verbose:
            print("\n=== MuJoCo Model Info ===")
            print(f"Bodies: {self.model.nbody}")
            print(f"Geometries: {self.model.ngeom}")
            print(f"Joints: {self.model.njnt}")
            print("\nBody hierarchy:")
            for i in range(min(20, self.model.nbody)):  # Show first 20
                parent = self.model.body_parentid[i]
                name = self.model.body(i).name or f"body_{i}"
                parent_name = (
                    self.model.body(parent).name or f"body_{parent}"
                    if parent >= 0
                    else "none"
                )
                print(f"  {i}: {name} -> parent: {parent} ({parent_name})")
            if self.model.nbody > 20:
                print(f"  ... and {self.model.nbody - 20} more bodies")

        # Export components
        geom_counts = self.export_geometry()

        if verbose:
            print("\n=== Geometry Types ===")
            for geom_type, count in sorted(geom_counts.items()):
                print(f"  {geom_type}: {count}")

            # Show bodies with multiple geometries
            multi_geom_bodies = {
                k: v for k, v in self.body_to_meshes.items() if len(v) > 1
            }
            if multi_geom_bodies:
                print("\n=== Bodies with Multiple Geometries ===")
                print(f"  {len(multi_geom_bodies)} bodies have multiple geometries")
                for body_id in sorted(multi_geom_bodies.keys())[:10]:  # Show first 10
                    body_name = self.model.body(body_id).name or f"body_{body_id}"
                    geom_count = len(self.body_to_meshes[body_id])
                    print(f"  Body {body_id} ({body_name}): {geom_count} geometries")
                if len(multi_geom_bodies) > 10:
                    print(f"  ... and {len(multi_geom_bodies) - 10} more")

        self.export_skeleton()

        if trajectories:
            self.export_animation(trajectories, fps)

        # Create glTF structure
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "MuJoCo glTF Exporter v2.0",
            },
            "scene": 0,
            "scenes": self.scenes,
            "nodes": self.nodes,
            "buffers": [
                {
                    "byteLength": len(self.binary_data),
                }
            ],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
        }

        if self.meshes:
            gltf["meshes"] = self.meshes

        if self.materials:
            gltf["materials"] = self.materials
        if self.textures:
            gltf["textures"] = self.textures
        if self.images:
            gltf["images"] = self.images
        if self.samplers:
            gltf["samplers"] = self.samplers

        if self.animations:
            gltf["animations"] = self.animations

        if verbose:
            print("\n=== glTF Structure ===")
            print(f"Scenes: {len(self.scenes)}")
            print(f"Nodes: {len(self.nodes)}")
            print(f"Meshes: {len(self.meshes)}")
            print(f"Animations: {len(self.animations)}")
            print(f"\nScene root nodes: {self.scenes[0]['nodes']}")
            print("\nNode hierarchy (first 20):")
            for i in range(min(20, len(self.nodes))):
                node = self.nodes[i]
                children = node.get("children", [])
                mesh = node.get("mesh", "none")
                print(f"  {i}: {node['name']} -> children: {children}, mesh: {mesh}")
            if len(self.nodes) > 20:
                print(f"  ... and {len(self.nodes) - 20} more nodes")

        # Convert to JSON
        json_data = json.dumps(gltf, separators=(",", ":")).encode("utf-8")

        # Pad JSON to 4-byte boundary
        json_padding = (4 - len(json_data) % 4) % 4
        json_data += b" " * json_padding

        # Pad binary to 4-byte boundary
        bin_padding = (4 - len(self.binary_data) % 4) % 4
        self.binary_data.extend([0] * bin_padding)

        # Write GLB file
        with open(output_path, "wb") as f:
            # GLB header
            f.write(struct.pack("<I", 0x46546C67))  # Magic: 'glTF'
            f.write(struct.pack("<I", 2))  # Version
            total_length = 12 + 8 + len(json_data) + 8 + len(self.binary_data)
            f.write(struct.pack("<I", total_length))

            # JSON chunk
            f.write(struct.pack("<I", len(json_data)))
            f.write(struct.pack("<I", 0x4E4F534A))  # 'JSON'
            f.write(json_data)

            # Binary chunk
            f.write(struct.pack("<I", len(self.binary_data)))
            f.write(struct.pack("<I", 0x004E4942))  # 'BIN\0'
            f.write(self.binary_data)

        print(f"✓ Exported GLB to: {output_path}")
        print(f"  - Nodes: {len(self.nodes)}")
        print(f"  - Meshes: {len(self.meshes)}")
        print(f"  - Animations: {len(self.animations)}")
        print(f"  - File size: {total_length / 1024:.1f} KB")


def load_animation_from_parquet(
    parquet_path: str, model: mujoco.MjModel
) -> Optional[List[Dict[str, np.ndarray]]]:
    """
    Load animation trajectories from a parquet file.

    Uses the same convention as the tutorial: first column is time, remaining
    columns are qpos in model order (same layout as mj_data.qpos[:]).
    So: qpos_df.iloc[i, 1:].values == qpos for frame i.

    The parquet file is expected to have:
    - First column: time axis
    - Remaining columns: qpos[0], qpos[1], ..., qpos[nq-1] in model order
      (same order as when assigning mj_data.qpos[:] = row[1:]).

    Args:
        parquet_path: Path to the parquet file.
        model: MuJoCo MjModel (used to validate qpos size).

    Returns:
        List of dicts with "qpos" key (each value is a 1D numpy array), or None
        if the parquet is empty or qpos column count does not match model.nq.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    # Same as tutorial: skip first column (time), rest is qpos in model order
    qpos_array = np.array(df.iloc[:, 1:], dtype=np.float64)
    if qpos_array.size == 0:
        return None
    nq = model.nq
    if qpos_array.shape[1] != nq:
        raise ValueError(
            f"Parquet qpos columns ({qpos_array.shape[1]}) do not match model.nq ({nq})"
        )
    trajectories = [{"qpos": qpos_array[f].copy()} for f in range(qpos_array.shape[0])]
    return trajectories


def export_mujoco_to_glb(
    model_path: str,
    output_path: str,
    trajectories: Optional[
        List[Union[Tuple[float, np.ndarray, Any], Dict[str, Any]]]
    ] = None,
    fps: float = 30.0,
) -> None:
    """Convenience function to load a MuJoCo model from XML and export it to GLB.

    Args:
        model_path: Path to MuJoCo XML model file.
        output_path: Path to output GLB file.
        trajectories: Optional list of (time, qpos, qvel) tuples or dicts with "qpos".
        fps: Frames per second for animation (default 30.0).

    Returns:
        None. Writes the GLB to output_path.
    """
    import mujoco as mj

    # Resolve to absolute path so MuJoCo resolves texture/mesh paths relative to
    # the XML directory regardless of cwd (matches behavior when same XML path
    # is used from different projects).
    model_path = os.path.abspath(model_path)
    model = mj.MjModel.from_xml_path(model_path)
    exporter = MujocoGLTFExporter(model)
    exporter.export_glb(output_path, trajectories, fps)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export MuJoCo model to GLB")
    parser.add_argument("model", help="Path to MuJoCo XML model file")
    parser.add_argument(
        "-o", "--output", help="Output GLB file path", default="output.glb"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Animation FPS")

    parser.add_argument(
        "--animation-file",
        metavar="PARQUET",
        help="Load animation from a parquet file (first column: time, rest: qpos in model order)",
    )

    args = parser.parse_args()

    trajectories = None
    if args.animation_file:
        import mujoco as mj

        model = mj.MjModel.from_xml_path(args.model)
        trajectories = load_animation_from_parquet(args.animation_file, model)
        if trajectories:
            print(f"Animation from file: {len(trajectories)} frames")
        else:
            print("No animation loaded from parquet")

    export_mujoco_to_glb(
        args.model, args.output, trajectories=trajectories, fps=args.fps
    )
