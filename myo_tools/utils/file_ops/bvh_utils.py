"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import argparse
import os

import mujoco
import numpy as np

from myo_tools.utils.file_ops.gltf_utils import (
    R_MJC_TO_GLTF,
    load_animation_from_parquet,
)
from myo_tools.utils.tensor_ops.quat_utils import quat2mat

# ---------------------------------------------------------------------------
# Coordinate-system helpers (match GLB for same heading and scale)
# ---------------------------------------------------------------------------
# BVH/GLB X = MJC X (forward), Y = MJC Z (up), Z = -MJC Y (right).
# Scale: default 1.0 = metres (match GLB); use scale=100 for centimetres.


def _mjc_to_bvh_pos(p: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Convert MuJoCo position (m, Z-up) to BVH frame (same as GLB)."""
    return (R_MJC_TO_GLTF @ p) * scale


def _mjc_to_bvh_rot(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 MuJoCo rotation to BVH frame (same as GLB)."""
    return R_MJC_TO_GLTF @ R @ R_MJC_TO_GLTF.T


# ---------------------------------------------------------------------------
# Tiny rotation-math primitives (no external deps)
# ---------------------------------------------------------------------------


def _axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula: axis (unit) + angle (rad) → 3x3 rotation."""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _mat_to_euler_zxy(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → Euler angles (Z, X, Y) in *degrees*.

    Decomposition: R = Rz(z) · Rx(x) · Ry(y)  (BVH ZXY intrinsic order).
    Standard extraction (see e.g. cookierobotics.com/081):
      x = asin(R[2,1]), y = atan2(-R[2,0], R[2,2]), z = atan2(-R[0,1], R[1,1]).
    Gimbal lock (|R[2,1]| ≈ 1) handled by setting z = 0 and solving y.
    """
    sin_x = np.clip(R[2, 1], -1.0, 1.0)
    x = np.arcsin(sin_x)
    cos_x = np.cos(x)

    if np.abs(cos_x) > 1e-6:
        y = np.arctan2(-R[2, 0], R[2, 2])
        z = np.arctan2(-R[0, 1], R[1, 1])
    else:
        # Gimbal lock: cos(x) ≈ 0; set z = 0, solve y from R[2,0], R[2,2]
        z = 0.0
        y = np.arctan2(-R[2, 0], R[2, 2])

    return np.degrees(np.array([z, x, y]))


def _mjc_to_bvh_rot_batch(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) MuJoCo rotations → (N, 3, 3) BVH frame. R_bvh[i] = R_MJC_TO_GLTF @ R[i] @ R_MJC_TO_GLTF.T"""
    return np.einsum("ij,njk,kl->nil", R_MJC_TO_GLTF, R, R_MJC_TO_GLTF.T)


def _mat_to_euler_zxy_batch(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices → (N, 3) Euler (Z, X, Y) in *degrees*.
    Vectorized; gimbal lock uses z=0 fallback per frame."""
    sin_x = np.clip(R[:, 2, 1], -1.0, 1.0)
    x = np.arcsin(sin_x)
    cos_x = np.cos(x)
    y = np.arctan2(-R[:, 2, 0], R[:, 2, 2])
    z = np.where(
        np.abs(cos_x) > 1e-6,
        np.arctan2(-R[:, 0, 1], R[:, 1, 1]),
        0.0,
    )
    y = np.where(np.abs(cos_x) > 1e-6, y, np.arctan2(-R[:, 2, 0], R[:, 2, 2]))
    return np.degrees(np.column_stack([z, x, y]))


# ---------------------------------------------------------------------------
# Skeleton extraction  (mirrors export_skeleton + export_animation in gltf_utils)
# ---------------------------------------------------------------------------


class _Joint:
    """Lightweight stand-in for a BVH joint node during tree construction."""

    __slots__ = (
        "name",
        "offset",
        "children",
        "is_root",
        "end_site",
        "translations",
        "rotations",
    )

    def __init__(self, name: str, offset: np.ndarray, is_root: bool = False):
        self.name = name
        self.offset = offset  # 3-vector, BVH coords (cm, Y-up)
        self.children: list[_Joint] = []
        self.is_root = is_root
        self.end_site: np.ndarray | None = None  # 3-vector or None
        # Per-frame data (filled later)
        self.translations: np.ndarray | None = None  # (F, 3) – root only
        self.rotations: np.ndarray | None = None  # (F, 3) – ZXY degrees


def _build_skeleton(model: mujoco.MjModel, scale: float = 1.0) -> _Joint:
    """Walk body_parentid and return the BVH joint tree rooted at body 0 (world).

    body 0 in MuJoCo is always the *world* body.  Its direct children become
    the candidates for the BVH root.  If world has exactly one child we use
    that child as root (typical humanoid).  Otherwise we synthesise a virtual
    "root" at the origin that parents all of world's children.
    """
    # --- 1. Compute rest-pose global transforms for every body ----------
    #   MuJoCo's body_pos / body_quat are *relative to parent*, so we
    #   accumulate down the tree exactly as mj_forward would.
    global_pos = np.zeros((model.nbody, 3), dtype=np.float64)  # in MuJoCo coords
    global_rot = np.zeros((model.nbody, 3, 3), dtype=np.float64)
    global_rot[0] = np.eye(3)  # world

    # Traverse in body-id order; MuJoCo guarantees parent_id < child_id.
    for bid in range(1, model.nbody):
        pid = model.body_parentid[bid]
        local_pos = model.body_pos[bid]
        local_rot = quat2mat(model.body_quat[bid])  # wxyz
        global_rot[bid] = global_rot[pid] @ local_rot
        global_pos[bid] = global_pos[pid] + global_rot[pid] @ local_pos

    # --- 2. Find world's children and pick / synthesise root ------------
    world_children = [
        bid for bid in range(1, model.nbody) if model.body_parentid[bid] == 0
    ]

    if len(world_children) == 1:
        root_bid = world_children[0]
    else:
        root_bid = 0  # will create a synthetic root at origin

    # --- 3. Recursively build _Joint tree --------------------------------
    # Pre-compute children map
    children_of = [[] for _ in range(model.nbody)]
    for bid in range(1, model.nbody):
        children_of[model.body_parentid[bid]].append(bid)

    def _make_joint(
        bid: int, parent_global_pos, parent_global_rot, is_root: bool
    ) -> _Joint:
        name = model.body(bid).name or f"body_{bid}"

        # Offset: difference in global pos, expressed in *parent's local* frame,
        # then swapped to BVH coords.  For the root joint the offset is simply
        # its global position in BVH coords.
        if is_root:
            offset = _mjc_to_bvh_pos(global_pos[bid], scale)
        else:
            delta_world = global_pos[bid] - parent_global_pos
            delta_local = parent_global_rot.T @ delta_world  # into parent local
            offset = _mjc_to_bvh_pos(delta_local, scale)

        joint = _Joint(name, offset, is_root=is_root)

        # Recurse
        for cid in children_of[bid]:
            child = _make_joint(cid, global_pos[bid], global_rot[bid], is_root=False)
            joint.children.append(child)

        # End site for leaves: use the longest geom extent attached to this body,
        # or a short default segment along the local Y axis (BVH convention).
        if not joint.children:
            joint.end_site = _compute_end_site(model, bid, scale)

        return joint

    if root_bid == 0:
        # Synthetic root at world origin
        root = _Joint("root", np.zeros(3), is_root=True)
        for cid in world_children:
            child = _make_joint(cid, global_pos[0], global_rot[0], is_root=False)
            root.children.append(child)
        if not root.children:
            root.end_site = np.array([0.0, 10.0, 0.0])  # fallback
    else:
        root = _make_joint(root_bid, global_pos[0], global_rot[0], is_root=True)

    return root


def _compute_end_site(
    model: mujoco.MjModel, bid: int, scale: float = 1.0
) -> np.ndarray:
    """Heuristic end-site: longest capsule/cylinder half-height or box half-size
    attached to this body, oriented along BVH Y.  Length in same units as scale."""
    best = 0.0
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        gtype = model.geom_type[gid]
        sz = model.geom_size[gid]
        if gtype in (3, 5):  # capsule or cylinder: size[1] = half-height
            best = max(best, sz[1])
        elif gtype == 6:  # box: longest half-axis
            best = max(best, max(sz))
        elif gtype == 2:  # sphere
            best = max(best, sz[0])

    length = best * scale if best > 0 else (0.05 if scale == 1.0 else 5.0)
    return np.array([0.0, length, 0.0])  # BVH Y-up


# ---------------------------------------------------------------------------
# Animation: qpos → per-joint local rotations (and root translation)
# ---------------------------------------------------------------------------


def _populate_frames(
    root: _Joint,
    model: mujoco.MjModel,
    trajectories: list[dict],
    scale: float = 1.0,
) -> None:
    """Fill translation / rotation arrays on every _Joint for each frame.

    Uses parent-relative poses from body_poses_from_qpos() (same as GLB).
    Each joint's rotation is the body's *parent-relative* rotation in MuJoCo,
    converted to BVH frame and expressed as ZXY Euler. No global propagation:
    we never convert global quaternions to Euler.
    """
    from myo_tools.utils.file_ops.qpos_to_pose import body_poses_from_qpos

    n_frames = len(trajectories)
    # Precompute parent-relative poses once per frame (same as GLB pipeline)
    body_quats_per_frame = []
    body_trans_per_frame = []
    for traj in trajectories:
        qs, ts = body_poses_from_qpos(model, traj["qpos"])
        body_quats_per_frame.append(qs)
        body_trans_per_frame.append(ts)

    name_to_bid = {}
    for bid in range(model.nbody):
        name = model.body(bid).name or f"body_{bid}"
        name_to_bid[name] = bid

    def _fill(joint: _Joint) -> None:
        """Recursively fill joint.rotations (and .translations for root)."""
        bid = name_to_bid.get(joint.name)

        if bid is None:
            # Synthetic root (world has multiple children)
            joint.rotations = np.zeros((n_frames, 3), dtype=np.float64)
            joint.translations = np.zeros((n_frames, 3), dtype=np.float64)
            for child in joint.children:
                _fill(child)
            return

        # Batch: (n_frames, 4) quats → (n_frames, 3, 3) → BVH → (n_frames, 3) Euler
        quats = np.array(
            [
                body_quats_per_frame[f].get(bid, model.body_quat[bid].copy())
                for f in range(n_frames)
            ],
            dtype=np.float64,
        )
        R_mjc = quat2mat(quats)
        R_bvh = _mjc_to_bvh_rot_batch(R_mjc)
        joint.rotations = _mat_to_euler_zxy_batch(R_bvh)

        if joint.is_root:
            # Root position must match GLB: body_pos[0] + R_0 @ body_trans[root], then to BVH
            R_parent = quat2mat(model.body_quat[0])
            p_parent = model.body_pos[0]
            t = np.array(
                [
                    body_trans_per_frame[f].get(bid, model.body_pos[bid].copy())
                    for f in range(n_frames)
                ],
                dtype=np.float64,
            )
            root_global_mjc = p_parent + (R_parent @ t.T).T
            joint.translations = (root_global_mjc @ R_MJC_TO_GLTF.T) * scale

        for child in joint.children:
            _fill(child)

    _fill(root)


# ---------------------------------------------------------------------------
# BVH file writer  (pure string formatting, no external lib)
# ---------------------------------------------------------------------------


def _write_bvh(root: _Joint, filepath: str, fps: float) -> None:
    """Serialise the joint tree to a BVH file."""
    frametime = 1.0 / fps

    # --- Count total frames (all joints share the same count) -----------
    def _get_frames(j: _Joint) -> int:
        if j.rotations is not None:
            return j.rotations.shape[0]
        for c in j.children:
            n = _get_frames(c)
            if n > 0:
                return n
        return 1  # rest-pose fallback

    n_frames = _get_frames(root)

    # If no animation data was populated, create single-frame rest pose
    def _ensure_rest(j: _Joint) -> None:
        if j.rotations is None:
            j.rotations = np.zeros((n_frames, 3), dtype=np.float64)
        if j.is_root and j.translations is None:
            j.translations = np.zeros((n_frames, 3), dtype=np.float64)
        for c in j.children:
            _ensure_rest(c)

    _ensure_rest(root)

    lines: list[str] = []

    # --- HIERARCHY section ------------------------------------------------
    lines.append("HIERARCHY")

    def _write_joint(j: _Joint, indent: int) -> None:
        prefix = "\t" * indent
        tag = "ROOT" if j.is_root else "Joint"
        lines.append(f"{prefix}{tag} {j.name}")
        lines.append(f"{prefix}{{")

        # OFFSET
        o = j.offset
        lines.append(f"{prefix}\tOFFSET\t{o[0]:.6f}\t{o[1]:.6f}\t{o[2]:.6f}")

        # CHANNELS
        if j.is_root:
            lines.append(
                f"{prefix}\tCHANNELS 6 Xposition Yposition Zposition "
                f"Zrotation Xrotation Yrotation"
            )
        else:
            lines.append(f"{prefix}\tCHANNELS 3 Zrotation Xrotation Yrotation")

        # Children or End Site
        if j.children:
            for child in j.children:
                _write_joint(child, indent + 1)
        elif j.end_site is not None:
            es = j.end_site
            lines.append(f"{prefix}\tEnd Site")
            lines.append(f"{prefix}\t{{")
            lines.append(f"{prefix}\t\tOFFSET\t{es[0]:.6f}\t{es[1]:.6f}\t{es[2]:.6f}")
            lines.append(f"{prefix}\t}}")

        lines.append(f"{prefix}}}")

    _write_joint(root, 0)

    # --- MOTION section ---------------------------------------------------
    lines.append("")
    lines.append("MOTION")
    lines.append(f"Frames:\t{n_frames}")
    lines.append(f"Frame Time:\t{frametime:.6f}")

    # Collect joints in BVH channel order (depth-first, same as HIERARCHY)
    ordered: list[_Joint] = []

    def _collect(j: _Joint) -> None:
        ordered.append(j)
        for c in j.children:
            _collect(c)

    _collect(root)

    # One line per frame
    for f in range(n_frames):
        vals: list[str] = []
        for j in ordered:
            if j.is_root:
                t = j.translations[f]
                vals.append(f"{t[0]:.6f}")
                vals.append(f"{t[1]:.6f}")
                vals.append(f"{t[2]:.6f}")
            r = j.rotations[f]  # Z, X, Y degrees
            vals.append(f"{r[0]:.6f}")
            vals.append(f"{r[1]:.6f}")
            vals.append(f"{r[2]:.6f}")
        lines.append("\t".join(vals))

    with open(filepath, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_mujoco_to_bvh(
    model_path: str,
    output_path: str,
    trajectories: list | None = None,
    fps: float = 30.0,
    scale: float = 1.0,
) -> None:
    """Export a MuJoCo model (+ optional trajectory) to BVH.

    Uses the same coordinate frame as GLB (X forward, Y up, Z right) and
    default scale 1.0 (metres) so heading and scale match glTF export.

    Args:
        model_path:    Path to the .xml / .mjcf / .urdf model file.
        output_path:   Destination .bvh path.
        trajectories:  List of {"qpos": ndarray} dicts  *or*
                       list of (time, qpos, qvel) tuples.
                       ``None`` → single rest-pose frame.
        fps:           Frames per second written into the BVH header.
        scale:         Position scale (default 1.0 = metres, match GLB; use 100 for cm).
    """
    model_path = os.path.abspath(model_path)
    model = mujoco.MjModel.from_xml_path(model_path)

    print(f"Model: {model.nbody} bodies, {model.njnt} joints, nq={model.nq}")

    # --- Normalise trajectory format (same as gltf_utils) -----------------
    if trajectories is not None:
        if isinstance(trajectories[0], tuple):
            trajectories = [{"qpos": np.asarray(t[1])} for t in trajectories]
        else:
            trajectories = [{"qpos": np.asarray(t["qpos"])} for t in trajectories]

    # --- Build skeleton tree from rest pose --------------------------------
    root = _build_skeleton(model, scale)
    print(f"BVH root joint: {root.name}")

    # --- Populate per-frame data -------------------------------------------
    if trajectories and len(trajectories) > 0:
        print(f"Populating {len(trajectories)} frames …")
        _populate_frames(root, model, trajectories, scale)
    else:
        print("No trajectory — exporting single rest-pose frame.")

        # Single frame of zeros (identity local rotations, zero translation)
        def _rest(j: _Joint) -> None:
            j.rotations = np.zeros((1, 3), dtype=np.float64)
            if j.is_root:
                j.translations = np.zeros((1, 3), dtype=np.float64)
            for c in j.children:
                _rest(c)

        _rest(root)

    # --- Write BVH file ----------------------------------------------------
    _write_bvh(root, output_path, fps)
    print(f"✓ Wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a MuJoCo model (+ trajectory) to BVH.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="MuJoCo XML model file")
    parser.add_argument(
        "--animation-file",
        metavar="PARQUET",
        help="Load animation from a parquet file (first column: time, rest: qpos in model order)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.bvh",
        help="Output BVH file (default: output.bvh)",
    )

    parser.add_argument(
        "--fps", type=float, default=30.0, help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Position scale: 1.0 = metres (match GLB), 100 = centimetres (default: 1.0)",
    )

    args = parser.parse_args()

    # --- Load trajectory if supplied ----------------------------------------
    trajectories = None
    if args.animation_file:
        import mujoco as mj

        model = mj.MjModel.from_xml_path(args.model)
        trajectories = load_animation_from_parquet(args.animation_file, model)
        if trajectories:
            print(f"Animation from file: {len(trajectories)} frames")
        else:
            print("No animation loaded from parquet")
    if args.traj:
        data = np.load(args.traj, allow_pickle=True)
        if data.dtype == object:
            # Object array of dicts
            trajectories = list(data)
        else:
            # Plain (N, nq) array — wrap each row
            trajectories = [{"qpos": row} for row in data]
        print(f"Loaded {len(trajectories)} frames from {args.traj}")

    export_mujoco_to_bvh(args.model, args.output, trajectories, args.fps, args.scale)


if __name__ == "__main__":
    main()
