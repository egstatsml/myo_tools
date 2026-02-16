"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

import numpy as np

# MuJoCo joint type constants
_mjJNT_FREE = 0
_mjJNT_BALL = 1
_mjJNT_SLIDE = 2
_mjJNT_HINGE = 3


def body_poses_from_qpos(model, qpos):
    """
    Compute per-body combined rotation (quat wxyz) and translation (3) from qpos.

    Uses qpos - qpos0 for HINGE and SLIDE; combines multiple joints per body
    with mju_mulQuat. Matches get_limb_rotations_translations semantics.

    Args:
        model: mujoco.MjModel
        qpos: 1D array of length model.nq

    Returns:
        body_quats: dict body_id -> np.ndarray (4,) wxyz
        body_trans: dict body_id -> np.ndarray (3,)
        Only bodies with at least one joint (body_jntadr >= 0) are included.
    """
    import mujoco

    qpos0 = model.qpos0
    body_quats = {}
    body_trans = {}

    for body_id in range(1, model.nbody):
        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr < 0:
            continue

        jnt_num = model.body_jntnum[body_id]
        combined_q = model.body_quat[body_id].copy()
        combined_t = model.body_pos[body_id].copy()

        for j in range(jnt_num):
            jnt_id = jnt_adr + j
            qpos_adr = int(model.jnt_qposadr[jnt_id])
            jnt_type = int(model.jnt_type[jnt_id])
            axis = model.jnt_axis[jnt_id].copy()

            if jnt_type == _mjJNT_FREE:
                # FREE qpos is full pose relative to parent; use it directly (no body_pos).
                combined_t = qpos[qpos_adr : qpos_adr + 3].copy()
                q = np.zeros(4, dtype=np.float64)
                q[:] = qpos[qpos_adr + 3 : qpos_adr + 7]
                mujoco.mju_mulQuat(combined_q, combined_q, q)
            elif jnt_type == _mjJNT_BALL:
                q = np.zeros(4, dtype=np.float64)
                q[:] = qpos[qpos_adr : qpos_adr + 4]
                mujoco.mju_mulQuat(combined_q, combined_q, q)
            elif jnt_type == _mjJNT_SLIDE:
                j_value = qpos[qpos_adr] - qpos0[qpos_adr]
                combined_t += axis * j_value
            elif jnt_type == _mjJNT_HINGE:
                j_value = qpos[qpos_adr] - qpos0[qpos_adr]
                q = np.zeros(4, dtype=np.float64)
                mujoco.mju_axisAngle2Quat(q, axis, float(j_value))
                mujoco.mju_mulQuat(combined_q, combined_q, q)

        body_quats[body_id] = combined_q.copy()
        body_trans[body_id] = combined_t.copy()

    return body_quats, body_trans


def root_body_ground_height_mjc(model):
    """
    Default height (MuJoCo z) of the first body that has joints
    (typically the root).

    Used to place the model on the ground: subtract this from glTF Y
    (or BVH Y) so the root sits at y=0.

    Returns:
        float: z coordinate in MuJoCo frame (metres), or 0.0 if no such body.
    """
    _, body_trans = body_poses_from_qpos(model, model.qpos0)
    if not body_trans:
        return 0.0
    # First body with joints is typically body_id=1 (root)
    first_bid = min(body_trans.keys())
    return float(body_trans[first_bid][2])
