"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for myo_tools utils.

import numpy as np
import pytest

from myo_tools.utils.tensor_ops.dict_utils import (
    dict_numpify,
    flatten_dict,
    print_dtype,
    unflatten_dict,
)
from myo_tools.utils.tensor_ops.quat_utils import (
    axis_angle2quat,
    diffQuat,
    euler2mat,
    euler2quat,
    is_unit_quaternion,
    mat2euler,
    mat2quat,
    mulQuat,
    negQuat,
    quat2euler,
    quat2mat,
    quat2Vel,
    quatDiff2Vel,
    rotVecMat,
    rotVecMatT,
)
from myo_tools.utils.tensor_ops.tensor_utils import (
    concat_tensor_dict_list,
    concat_tensor_list,
    flatten_tensors,
    high_res_normalize,
    pad_tensor,
    pad_tensor_dict,
    split_tensor_dict_list,
    stack_tensor_dict_list,
    stack_tensor_list,
    truncate_tensor_dict,
    truncate_tensor_list,
    unflatten_tensors,
)

##########################################
# tests/test_quat_utils.py


def test_mulQuat():
    # Test identity quaternion multiplication
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([1.0, 0.0, 0.0, 0.0])
    result = mulQuat(q1, q2)
    assert np.allclose(result, q1)

    # Test basic rotation quaternions
    q_90x = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90° around x
    q_90y = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # 90° around y
    result = mulQuat(q_90x, q_90y)
    expected = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(result, expected, atol=1e-7)


def test_negQuat():
    quat = np.array([1, 2, 3, 4])
    expected = np.array([1, -2, -3, -4])
    np.testing.assert_array_almost_equal(negQuat(quat), expected)


def test_quat2Vel():
    quat = np.array([1, 0, 1, 0])
    dt = 1
    expected_speed = 1.57079633  # Expected speed value
    expected_axis = np.array([0, 1, 0])  # Expected axis value
    speed, axis = quat2Vel(quat, dt)
    assert np.isclose(speed, expected_speed)
    np.testing.assert_array_almost_equal(axis, expected_axis)


def test_diffQuat():
    quat1 = np.array([1, 0, 0, 0])
    quat2 = np.array([0, 1, 0, 0])
    expected = np.array([0, 1, 0, 0])
    assert np.allclose(diffQuat(quat1, quat2), expected)


def test_quatDiff2Vel():
    quat1 = np.array([1, 0, 0, 0])
    quat2 = np.array([0, 1, 0, 0])
    dt = 1
    expected_speed = np.pi
    expected_axis = np.array([1, 0, 0])
    speed, axis = quatDiff2Vel(quat1, quat2, dt)
    assert np.allclose(speed, expected_speed, atol=1e-7)
    assert np.allclose(axis, expected_axis)


def test_is_unit_quaternion():
    # Test identity quaternion
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert is_unit_quaternion(q_identity)

    # Test non-identity quaternion
    q_non_identity = np.array([0.5, 0.5, 0.5, 0.5])
    assert not is_unit_quaternion(q_non_identity)


def test_axis_angle2quat():
    axis = np.array([1, 0, 0])
    angle = np.pi / 2
    expected = np.array([np.cos(angle / 2), np.sin(angle / 2) * axis[0], 0, 0])
    result = axis_angle2quat(axis, angle)
    np.testing.assert_array_almost_equal(result, expected)


def test_euler2mat():
    euler = np.array([0, 0, np.pi / 2])
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = euler2mat(euler)
    np.testing.assert_array_almost_equal(result, expected)


def test_euler2quat():
    euler = np.array([0, 0, np.pi / 2])
    expected = [0.70710678, 0.0, -0.0, 0.70710678]
    np.allclose(euler2quat(euler), expected)


def test_mat2euler():
    mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = np.array([0, 0, np.pi / 2])
    result = mat2euler(mat)
    np.allclose(result, expected)


def test_mat2quat():
    mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = [0.70710678, 0.0, 0.0, 0.70710678]
    result = mat2quat(mat)
    np.allclose(result, expected)


def test_rotVecMatT():
    vec = np.array([1, 0, 0])
    mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = np.array([0, -1, 0])
    result = rotVecMatT(vec, mat)
    np.testing.assert_array_almost_equal(result, expected)


def test_rotVecMat():
    vec = np.array([1, 0, 0])
    mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = np.array([0, 1, 0])
    result = rotVecMat(vec, mat)
    np.allclose(result, expected)


def test_quat2euler():
    quat = np.array([0, 1, 0, 0])
    expected = np.array([-np.pi, 0, 0])
    result = quat2euler(quat)
    np.allclose(result, expected)


def test_quat2mat():
    quat = np.array([0, 1, 0, 0])
    expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    result = quat2mat(quat)
    np.allclose(result, expected)


###################################
def test_dict_numpify():
    data = {
        "int": 1,
        "float": 1.0,
        "bool": True,
        "str": "test",
        "nested": {"int": 2, "float": 2.0},
        "list": [1, 2, 3],
    }
    expected = {
        "int": np.array([1], dtype=np.int8),
        "float": np.array([1.0], dtype=np.float16),
        "bool": np.array([True], dtype=np.bool_),
        "str": np.array(["test"]),
        "nested": {
            "int": np.array([2], dtype=np.int8),
            "float": np.array([2.0], dtype=np.float16),
        },
        "list": np.array([1, 2, 3], dtype=np.int8),
    }
    result = dict_numpify(data, i_res=np.int8, f_res=np.float16)

    # Use np.array_equal for comparison
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])


def test_print_dtype(capfd):
    data = {
        "int": 1,
        "float": 1.0,
        "bool": True,
        "nested": {"int": 2},
    }
    print_dtype(data)
    captured = capfd.readouterr()
    assert "int :" in captured.out
    assert "float :" in captured.out
    assert "bool :" in captured.out
    assert "nested/int :" in captured.out


def test_flatten_dict():
    data = {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": 4,
    }
    expected = {
        "a": 1,
        "b/c": 2,
        "b/d": 3,
        "e": 4,
    }
    result = flatten_dict(data)
    assert result == expected


def test_unflatten_dict():
    data = {
        "a": 1,
        "b/c": 2,
        "b/d": 3,
        "e": 4,
    }
    expected = {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": 4,
    }
    result = unflatten_dict(data)
    assert result == expected


########################################


def test_flatten_tensors():
    assert np.array_equal(
        flatten_tensors([np.array([[1, 2], [3, 4]])]), np.array([1, 2, 3, 4])
    )


def test_unflatten_tensors():
    flattened = np.array([1, 2, 3, 4])
    shapes = [(2, 2)]
    assert np.array_equal(
        unflatten_tensors(flattened, shapes), [np.array([[1, 2], [3, 4]])]
    )


def test_pad_tensor():
    x = np.array([[1, 2], [3, 4]])
    padded = pad_tensor(x, 5, mode="zero")
    assert padded.shape == (5, 2)


def test_high_res_normalize():
    probs = [0.1, 0.2, 0.7]
    normalized = high_res_normalize(probs)
    assert np.isclose(sum(normalized), 1.0)


def test_flatten_tensors_multiple():
    # Test flattening multiple tensors
    tensors = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]
    expected = np.array([1, 2, 3, 4, 5, 6])
    result = flatten_tensors(tensors)
    np.testing.assert_array_equal(result, expected)


def test_unflatten_tensors_multiple():
    # Test unflattening multiple tensors
    flattened = np.array([1, 2, 3, 4, 5, 6])
    shapes = [(2, 2), (1, 2)]
    expected = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]
    result = unflatten_tensors(flattened, shapes)
    for r, e in zip(result, expected):
        np.testing.assert_array_equal(r, e)


def test_pad_tensor_with_nonzero_mode():
    # Test padding with non-zero mode
    x = np.array([[1, 2], [3, 4]])
    padded = pad_tensor(x, 5, mode="edge")
    expected = np.array([[1, 2], [3, 4], [0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(padded, expected)


def test_high_res_normalize_empty():
    # Test high_res_normalize with empty input
    probs = []
    normalized = high_res_normalize(probs)
    assert len(normalized) == 0


def test_flatten_tensors_empty():
    # Test flattening an empty list of tensors
    result = flatten_tensors([])
    expected = np.asarray([])
    np.testing.assert_array_equal(result, expected)


def test_unflatten_tensors_invalid_shape():
    # Test unflattening with mismatched shapes
    flattened = np.array([1, 2, 3, 4])
    tensor_shapes = [(2, 2), (1, 1)]  # This will cause a mismatch
    with pytest.raises(ValueError):
        unflatten_tensors(flattened, tensor_shapes)


def test_pad_tensor_dict():
    # Test padding a dictionary of tensors
    tensor_dict = {
        "a": np.array([[1, 2], [3, 4]]),
        "b": np.array([[5, 6]]),
    }
    padded_dict = pad_tensor_dict(tensor_dict, max_len=3, mode="zero")
    expected_dict = {
        "a": np.array([[1, 2], [3, 4], [0, 0]]),
        "b": np.array([[5, 6], [0, 0], [0, 0]]),
    }
    for key in expected_dict:
        np.testing.assert_array_equal(padded_dict[key], expected_dict[key])


def test_high_res_normalize_non_empty():
    # Test high_res_normalize with non-empty input
    probs = [0.1, 0.2, 0.7]
    normalized = high_res_normalize(probs)
    assert np.isclose(sum(normalized), 1.0)


def test_pad_tensor_with_large_max_len():
    # Test padding with a larger max_len than the tensor length
    x = np.array([[1, 2], [3, 4]])
    padded = pad_tensor(x, 10, mode="zero")
    expected = np.concatenate([x, np.zeros((8, 2))])
    np.testing.assert_array_equal(padded, expected)


def test_pad_tensor_with_last_mode():
    # Test padding with mode "last"
    x = np.array([[1, 2], [3, 4]])
    padded = pad_tensor(x, 5, mode="last")
    expected = np.array([[1, 2], [3, 4], [3, 4], [3, 4], [3, 4]])
    np.testing.assert_array_equal(padded, expected)


def test_high_res_normalize_with_negative_values():
    # Test high_res_normalize with negative values
    probs = [-0.1, -0.2, -0.7]
    normalized = high_res_normalize(probs)
    assert np.all(np.array(normalized) == 0)  # Ensure all values are zero


def test_stack_tensor_list_empty():
    # Test stacking an empty list of tensors
    result = stack_tensor_list([])
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)


def test_stack_tensor_dict_list_empty():
    # Test stacking an empty list of tensor dictionaries
    result = stack_tensor_dict_list([])
    expected = {}
    np.testing.assert_equal(result, expected)


def test_concat_tensor_list_empty():
    # Test concatenating an empty list of tensors
    result = concat_tensor_list([])
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)


def test_concat_tensor_dict_list_empty():
    # Test concatenating an empty list of tensor dictionaries
    result = concat_tensor_dict_list([])
    expected = {}
    np.testing.assert_equal(result, expected)


def test_split_tensor_dict_list_empty():
    # Test splitting an empty tensor dictionary
    result = split_tensor_dict_list({})
    expected = []
    np.testing.assert_equal(result, expected)


def test_truncate_tensor_list_longer_than_truncated_len():
    # Test truncating a tensor list longer than the truncated length
    tensor_list = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    result = truncate_tensor_list(tensor_list, 2)
    expected = [np.array([1, 2]), np.array([3, 4])]
    for r, e in zip(result, expected):
        np.testing.assert_array_equal(r, e)


def test_truncate_tensor_dict_longer_than_truncated_len():
    # Test truncating a tensor dictionary longer than the truncated length
    tensor_dict = {
        "a": [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],
        "b": [np.array([7, 8]), np.array([9, 10])],
    }
    result = truncate_tensor_dict(tensor_dict, 1)
    expected = {
        "a": [np.array([1, 2])],
        "b": [np.array([7, 8])],
    }
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
