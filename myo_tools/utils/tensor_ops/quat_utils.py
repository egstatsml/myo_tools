"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: This file implements a quaternion algebra library

import warnings

import numpy as np

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def quat2rotvec(quaternion_wxyz_array):
    """
    Converts a 4-element NumPy array quaternion (w, x, y, z) to its
    3D rotation vector (logarithmic space) representation.
    This implementation manually computes the rotation vector.

    Args:
        quaternion_wxyz_array (np.ndarray): A 4-element NumPy array
                                            representing the quaternion (w, x, y, z).
                                            Assumed to be a unit quaternion.

    Returns:
        np.ndarray: A 3-element NumPy array representing the rotation vector (rx, ry, rz).
                    The magnitude is the angle in radians, and the direction is the axis.
    """
    w, x, y, z = quaternion_wxyz_array

    # Ensure the quaternion is normalized (important for acos)
    norm = np.linalg.norm(quaternion_wxyz_array)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])  # Identity quaternion maps to zero vector

    # Normalize if not already unit length (though input is assumed unit)
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    # Calculate the angle of rotation
    # Clamp w to [-1, 1] to avoid numerical issues with acos
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))

    # If angle is very small, the rotation vector is approximately (x, y, z) * 2
    # This avoids division by zero or very small numbers
    if np.isclose(angle, 0.0):
        return np.array(
            [0.0, 0.0, 0.0]
        )  # Very small angle means near identity rotation

    # Calculate the scale factor for the vector part
    sin_half_angle = np.sin(angle / 2)

    # Avoid division by zero if sin_half_angle is very small (angle near 0 or pi*2n)
    if np.isclose(sin_half_angle, 0.0):
        return np.array(
            [0.0, 0.0, 0.0]
        )  # Should be handled by angle check above, but for safety

    scale = angle / (2 * sin_half_angle)

    rotation_vector = np.array([x * scale, y * scale, z * scale])
    return rotation_vector


def rotvec2quat(rotation_vector):
    """
    Converts a 3D rotation vector (logarithmic space) back to a
    4-element NumPy array quaternion (w, x, y, z).
    This implementation manually computes the exponential map.

    Args:
        rotation_vector (np.ndarray): A 3-element NumPy array representing the
                                      rotation vector (rx, ry, rz).

    Returns:
        np.ndarray: A 4-element NumPy array representing the quaternion (w, x, y, z).
    """
    angle = np.linalg.norm(rotation_vector)

    # If angle is very small, return identity quaternion
    if np.isclose(angle, 0.0):
        return np.array([1.0, 0.0, 0.0, 0.0])

    # Calculate half angle and its sine/cosine
    half_angle = angle / 2
    sin_half_angle = np.sin(half_angle)
    cos_half_angle = np.cos(half_angle)

    # Calculate the scalar part (w)
    w = cos_half_angle

    # Calculate the vector part (x, y, z)
    # Avoid division by zero if angle is very small
    if np.isclose(angle, 0.0):
        x, y, z = 0.0, 0.0, 0.0
    else:
        scale = sin_half_angle / angle
        x = rotation_vector[0] * scale
        y = rotation_vector[1] * scale
        z = rotation_vector[2] * scale

    quaternion_wxyz_array = np.array([w, x, y, z])

    # Ensure it's a unit quaternion (should be by formula, but for numerical stability)
    return quaternion_wxyz_array / np.linalg.norm(quaternion_wxyz_array)


def mulQuat(qa, qb):
    """
    Multiplies two quaternions.

    Args:
        qa (np.array): The first quaternion [w, x, y, z].
        qb (np.array): The second quaternion [w, x, y, z].

    Returns:
        np.array: The resulting quaternion [w, x, y, z] after multiplication.
    """
    res = np.zeros(4)
    # Quaternion multiplication formula:
    # (w1 + x1i + y1j + z1k) * (w2 + x2i + y2j + z2k)
    # w = w1w2 - x1x2 - y1y2 - z1z2
    # x = w1x2 + x1w2 + y1z2 - z1y2
    # y = w1y2 - x1z2 + y1w2 + z1x2
    # z = w1z2 + x1y2 - y1x2 + z1w2
    res[0] = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3]
    res[1] = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2]
    res[2] = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1]
    res[3] = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0]
    return res


def negQuat(quat):
    """
    Calculates the conjugate (negation) of a quaternion.

    The conjugate of a quaternion [w, x, y, z] is [w, -x, -y, -z].
    This corresponds to the inverse rotation if the quaternion is a unit quaternion.

    Args:
        quat (np.array): The input quaternion [w, x, y, z].

    Returns:
        np.array: The conjugate of the quaternion [w, -x, -y, -z].
    """
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def quat2Vel(quat, dt=1):
    """
    Converts a quaternion representing a rotation difference to angular speed and rotation axis.

    This function assumes the quaternion represents a rotation that occurred over a time `dt`.

    Args:
        quat (np.array): The quaternion [w, x, y, z] representing the rotation.
        dt (float, optional): The time interval over which the rotation occurred. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - speed (float): The angular speed in radians per unit time.
            - axis (np.array): The normalized rotation axis [x, y, z].
    """
    # Extract the vector part (x, y, z) of the quaternion
    axis = quat[1:].copy()
    # Calculate sin(angle/2) from the magnitude of the vector part
    sin_a_2 = np.sqrt(np.sum(axis**2))
    # Normalize the axis to get the direction of rotation. Add a small epsilon
    # to prevent division by zero if sin_a_2 is zero (i.e., no rotation).
    axis = axis / (sin_a_2 + 1e-8)
    # Calculate the angle of rotation using atan2(sin(angle/2), cos(angle/2))
    # Then divide by dt to get angular speed.
    speed = 2 * np.arctan2(sin_a_2, quat[0]) / dt
    return speed, axis


def diffQuat(quat1, quat2):
    """
    Calculates the quaternion difference between two quaternions (quat2 relative to quat1).

    The difference quaternion `diff` represents the rotation from `quat1` to `quat2`.
    It is calculated as `quat2 * quat1_conjugate`.

    Args:
        quat1 (np.array): The first quaternion [w, x, y, z].
        quat2 (np.array): The second quaternion [w, x, y, z].

    Returns:
        np.array: The difference quaternion [w, x, y, z].
    """
    # Get the conjugate of the first quaternion
    neg = negQuat(quat1)
    # Multiply the second quaternion by the conjugate of the first
    diff = mulQuat(quat2, neg)
    return diff


def quatDiff2Vel(quat1, quat2, dt):
    """
    Converts the difference between two quaternions to angular speed and rotation axis.

    This function first calculates the quaternion difference and then converts it
    to angular speed and axis, assuming the difference occurred over a time `dt`.

    Args:
        quat1 (np.array): The initial quaternion [w, x, y, z].
        quat2 (np.array): The final quaternion [w, x, y, z].
        dt (float): The time interval over which the rotation difference occurred.

    Returns:
        tuple: A tuple containing:
            - speed (float): The angular speed in radians per unit time.
            - axis (np.array): The normalized rotation axis [x, y, z].
    """
    # Calculate the quaternion representing the difference in rotation
    diff = diffQuat(quat1, quat2)
    # Convert the difference quaternion to angular speed and axis
    return quat2Vel(diff, dt)


def axis_angle2quat(axis, angle):
    """
    Converts an axis-angle representation of a rotation to a quaternion.

    Args:
        axis (np.array): A 3-element array representing the rotation axis [x, y, z].
                         This axis should be a unit vector.
        angle (float): The angle of rotation in radians around the axis.

    Returns:
        np.array: The corresponding quaternion [w, x, y, z].
    """
    # Quaternion components: w = cos(angle/2), (x, y, z) = axis * sin(angle/2)
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([c, s * axis[0], s * axis[1], s * axis[2]])


def euler2mat(euler):
    """
    Convert Euler Angles to Rotation Matrix.

    Assumes a ZYX (yaw-pitch-roll) convention for Euler angles.

    Args:
        euler (np.array): A 3-element array representing the Euler angles [roll, pitch, yaw]
                          in radians.

    Returns:
        np.array: A 3x3 rotation matrix.
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    # Angles are typically ordered Z-Y-X (yaw-pitch-roll).
    # The negative signs are applied because the matrix formulation often
    # assumes positive rotations around each axis, and here the input Euler
    # angles might be defined in a way that requires negation for this specific
    # matrix construction.
    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]  # Yaw, Pitch, Roll
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    # Populate the rotation matrix based on the ZYX Euler angle convention
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """
    Convert Euler Angles to Quaternions.

    Assumes a ZYX (yaw-pitch-roll) convention for Euler angles.

    Args:
        euler (np.array): A 3-element array representing the Euler angles [roll, pitch, yaw]
                          in radians.

    Returns:
        np.array: The corresponding quaternion [w, x, y, z].
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    # Half angles for quaternion calculation
    # The order and signs here align with a common ZYX (yaw-pitch-roll) convention
    # where the angles are divided by 2 for the quaternion components.
    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    # Populate quaternion components (w, x, y, z)
    quat[..., 0] = cj * cc + sj * ss  # w
    quat[..., 3] = cj * sc - sj * cs  # z
    quat[..., 2] = -(cj * ss + sj * cc)  # y
    quat[..., 1] = cj * cs - sj * sc  # x
    return quat


def mat2euler(mat):
    """
    Convert Rotation Matrix to Euler Angles.

    This function converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    It handles potential gimbal lock scenarios where pitch is near +/- 90 degrees.
    The output Euler angles are in radians.

    Args:
        mat (np.array): A 3x3 rotation matrix.

    Returns:
        np.array: A 3-element array representing the Euler angles [roll, pitch, yaw]
                  in radians.
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    # Calculate 'cy' which helps determine if gimbal lock is close
    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    # Condition to check for gimbal lock (pitch close to +/- 90 degrees)
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)

    # Calculate Yaw (Z-axis rotation)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),  # Normal case
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),  # Gimbal lock case
    )
    # Calculate Pitch (Y-axis rotation)
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    # Calculate Roll (X-axis rotation)
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def mat2quat(mat):
    """
    Convert Rotation Matrix to Quaternion.

    This function converts a 3x3 rotation matrix to a quaternion.
    It uses an approach that handles various rotation matrix configurations
    and prefers quaternions with a positive 'w' component.

    Args:
        mat (np.array): A 3x3 rotation matrix.

    Returns:
        np.array: The corresponding quaternion [w, x, y, z].
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

    # Create a symmetric matrix K, from which the quaternion can be derived
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0

    q = np.empty(K.shape[:-2] + (4,))
    # Iterate over the batch dimensions to calculate the quaternion for each matrix
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Compute eigenvalues and eigenvectors of K.
        # The eigenvector corresponding to the largest eigenvalue is the quaternion.
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select the eigenvector corresponding to the largest eigenvalue.
        # Reorder components to (w, x, y, z) as per common quaternion convention.
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Ensure the 'w' component (scalar part) is non-negative.
        # A quaternion and its negative represent the same rotation.
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def m4_from_qt(q, t):
    """
    DEPRECATED: Please use quatvec2H instead.
    Builds a 4x4 homogeneous transformation matrix from a quaternion and translation vector.

    Args:
        q (Iterable): A size=4 iterable representing a quaternion [w, x, y, z].
        t (Iterable): A size=3 iterable representing a translation vector [tx, ty, tz].

    Returns:
        np.array: A 4x4 homogeneous transformation matrix.
    """
    warnings.warn(
        "m4_from_qt will be deprecated. Please use quatvec2H instead",
        DeprecationWarning,
    )
    return quatvec2H(q, t)


def quatvec2H(q, t):
    """
    Builds a 4x4 (H)omogeneous transformation matrix from a quaternion and translation vector.

    Args:
        q (Iterable): A size=4 iterable representing a quaternion [w, x, y, z].
        t (Iterable): A size=3 iterable representing a translation vector [tx, ty, tz].

    Returns:
        np.array: A 4x4 homogeneous transformation matrix.
    """
    # Convert the quaternion to a 3x3 rotation matrix
    rw_3 = quat2mat(q)
    # Initialize a 4x4 identity matrix
    rw = np.eye(4)
    # Place the 3x3 rotation matrix in the top-left corner
    rw[:3, :3] = rw_3
    # Place the translation vector in the rightmost column (top 3 rows)
    rw[:3, 3] = t.T  # Transpose t to ensure correct assignment if t is row vector
    return rw


def qt_from_m4(M):
    """
    DEPRECATED: Please use H2quatvec instead.
    Extracts a quaternion and translation vector from a 4x4 homogeneous transformation matrix.

    Args:
        M (Iterable): An Iterable with shape=(4,4) representing a 3D transformation matrix.

    Returns:
        tuple: A tuple containing:
            - q (np.array): The extracted quaternion [w, x, y, z].
            - t (np.array): The extracted translation vector [tx, ty, tz].
    """
    warnings.warn(
        "qt_from_m4 will be deprecated. Please use H2quatvec instead",
        DeprecationWarning,
    )
    return H2quatvec(M)


def H2quatvec(H):
    """
    Extracts a quaternion and translation vector from a 4x4 (H)omogeneous transformation matrix.

    Args:
        H (Iterable): An Iterable with shape=(4,4) representing a 3D transformation matrix.

    Returns:
        tuple: A tuple containing:
            - q (np.array): The extracted quaternion [w, x, y, z].
            - t (np.array): The extracted translation vector [tx, ty, tz].
    """
    # Extract the 3x3 rotation sub-matrix
    R = H[:3, :3]
    # Extract the translation vector
    T = H[:3, 3].T
    # Convert the 3x3 rotation matrix to a quaternion
    return mat2quat(R), T


def quat2euler(quat):
    """
    Convert Quaternion to Euler Angles.

    This function converts a quaternion to a 3x3 rotation matrix first,
    and then converts the matrix to Euler angles (roll, pitch, yaw).

    Args:
        quat (np.array): A 4-element array representing the quaternion [w, x, y, z].

    Returns:
        np.array: A 3-element array representing the Euler angles [roll, pitch, yaw]
                  in radians.
    """
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    """
    Convert Quaternion to Rotation Matrix 3x3.

    Args:
        quat (np.array): A 4-element array representing the quaternion [w, x, y, z].

    Returns:
        np.array: A 3x3 rotation matrix.
    """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # Calculate the squared norm of the quaternion
    Nq = np.sum(quat * quat, axis=-1)
    # Calculate the scaling factor 's'. This normalizes the quaternion
    # before constructing the matrix if it's not a unit quaternion.
    s = 2.0 / Nq

    # Pre-calculate terms to optimize matrix computation
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    # Populate the rotation matrix components
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)

    # If the quaternion norm is close to zero, return an identity matrix
    # (or handle as an error/special case, depending on desired behavior).
    # This prevents division by zero or large errors if Nq is very small.
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def rotVecMatT(vec, mat):
    """
    Multiplies a 3D vector by the transpose of a 3x3 rotation matrix.

    This is equivalent to rotating the vector by the inverse of the rotation
    represented by the matrix.

    Args:
        vec (np.array): A 3-element array representing the vector [x, y, z].
        mat (np.array): A 3x3 rotation matrix.

    Returns:
        np.array: The rotated vector.
    """
    # Equivalent to np.dot(mat.T, vec)
    return np.array(
        [
            mat[0, 0] * vec[0] + mat[1, 0] * vec[1] + mat[2, 0] * vec[2],
            mat[0, 1] * vec[0] + mat[1, 1] * vec[1] + mat[2, 1] * vec[2],
            mat[0, 2] * vec[0] + mat[1, 2] * vec[1] + mat[2, 2] * vec[2],
        ]
    )


def rotVecMat(vec, mat):
    """
    Multiplies a 3D vector by a 3x3 rotation matrix.

    This rotates the vector by the rotation represented by the matrix.

    Args:
        vec (np.array): A 3-element array representing the vector [x, y, z].
        mat (np.array): A 3x3 rotation matrix.

    Returns:
        np.array: The rotated vector.
    """
    # Equivalent to np.dot(mat, vec)
    return np.array(
        [
            mat[0, 0] * vec[0] + mat[0, 1] * vec[1] + mat[0, 2] * vec[2],
            mat[1, 0] * vec[0] + mat[1, 1] * vec[1] + mat[1, 2] * vec[2],
            mat[2, 0] * vec[0] + mat[2, 1] * vec[1] + mat[2, 2] * vec[2],
        ]
    )


def rotVecQuat(vec, quat):
    """
    Rotates a 3D vector by a quaternion.

    This function converts the quaternion to a 3x3 rotation matrix and then
    applies the matrix rotation to the vector.

    Args:
        vec (np.array): A 3-element array representing the vector [x, y, z].
        quat (np.array): A 4-element array representing the quaternion [w, x, y, z].

    Returns:
        np.array: The rotated vector.
    """
    # Convert quaternion to rotation matrix
    mat = quat2mat(quat)
    # Apply matrix rotation to the vector
    return rotVecMat(vec, mat)


def is_unit_quaternion(q):
    """
    Checks if a quaternion is a unit quaternion (i.e., its magnitude is approximately 1).

    Args:
        q (np.array): A 4-element array representing the quaternion [w, x, y, z].

    Returns:
        bool: True if the quaternion is a unit quaternion, False otherwise.
    """
    # Check if the quaternion is approximately equal to the identity quaternion [1, 0, 0, 0]
    # This also implicitly checks if its magnitude is 1, assuming it's normalized.
    # A more robust check might be np.isclose(np.linalg.norm(q), 1.0).
    return np.allclose(q, np.array([1.0, 0.0, 0.0, 0.0]))


def rotVecQuatBatch(q, v):
    """
    Rotates a batch of 3D vectors by a batch of quaternions.

    This function efficiently rotates multiple vectors by multiple quaternions
    using vectorized NumPy operations.

    Args:
        q (np.array): A batch of quaternions, shape [B, 4], where B is the batch size.
                      Each quaternion is [w, x, y, z].
        v (np.array): A batch of 3D vectors, shape [B, 3], where B is the batch size.
                      Each vector is [x, y, z].

    Returns:
        np.array: A batch of rotated vectors, shape [B, 3].
    """
    # Extract scalar (w) and vector (x, y, z) parts of the quaternions
    q_w = q[..., 0][..., None]  # Shape [B, 1]
    q_vec = q[..., 1:]  # Shape [B, 3]

    # Formula for rotating a vector v by a quaternion q = [w, vec]:
    # v_rotated = (w^2 - |vec|^2) * v + 2 * (vec . v) * vec + 2 * w * (vec x v)
    # This implementation uses a simplified version derived from the quaternion product (0, v) = q * (0, v) * q_conjugate
    # It decomposes into three terms:
    # a = v * (2 * w^2 - 1)
    # b = 2 * w * (vec x v)
    # c = 2 * (vec . v) * vec
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v, axis=-1) * q_w * 2.0
    c = q_vec * (q_vec[..., None, :] @ v[..., None]).squeeze(-1) * 2.0
    return a + b + c
