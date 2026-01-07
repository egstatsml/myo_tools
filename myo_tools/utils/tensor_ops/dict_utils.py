"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: This file implements a series of utilities to manipulate dictionaries, including functions to convert dictionary values to numpy arrays, print the data types of dictionary values, flatten nested dictionaries, and unflatten flattened dictionaries.


import unittest

import numpy as np


def dict_numpify(data: dict, u_res=np.uint8, i_res=np.int8, f_res=np.float16) -> dict:
    """
    Convert all data to numpy using specified resolution
    data:   Input dict
    i_res:  int resolution: Skip if none
    f_res:  float resolution: Skip if none
    """
    for key, val in data.items():
        # non iteratables
        if np.isscalar(val):
            if isinstance(val, (bool, np.bool_)):
                val = np.array([val], dtype=np.bool_)
            elif isinstance(val, (np.unsignedinteger,)):
                val = np.array([val], dtype=u_res)
            elif isinstance(val, (int, np.signedinteger)):
                val = np.array([val], dtype=i_res)
            elif isinstance(val, (float, np.floating)):
                val = np.array([val], dtype=f_res)
            elif isinstance(val, str):
                val = [val]

        # numpy
        elif isinstance(val, np.ndarray):
            if np.issubdtype(val.dtype, np.unsignedinteger) and u_res:
                val = val.astype(u_res, copy=False)
            elif np.issubdtype(val.dtype, np.signedinteger) and i_res:
                val = val.astype(i_res, copy=False)
            elif np.issubdtype(val.dtype, np.floating) and f_res:
                val = val.astype(f_res, copy=False)
            elif val.dtype == np.dtype("O"):
                val = val.astype(np.float16, copy=False)  # switch none with nan

        # dict
        elif isinstance(val, dict):
            val = dict_numpify(val, i_res, f_res)

        # lists/ tuples
        elif "__len__" in dir(val) and len(val) > 0:
            if isinstance(val[0], bool):
                val = np.array(val, dtype=np.bool_)
            elif isinstance(val[0], int):
                val = np.array(val, dtype=i_res)
            elif isinstance(val[0], float):
                val = np.array(val, dtype=f_res)
            elif not isinstance(val[0], str):
                val = np.array(val)  # let numpy handle it for nested stuctures
                # raise TypeError("Data type {} not supported for {}".format(type(val[0]), key))

        data[key] = val
    return data


def print_dtype(data: dict, name: str = "", delimiter: str = "/") -> None:
    """
    Print dtype of the provided dict
    """
    for key, val in data.items():
        flat_key = key if name == "" else name + delimiter + key

        if isinstance(val, dict):
            print_dtype(data=val, name=flat_key)
        elif "__len__" in dir(val):
            print(flat_key, ":", type(val), "::", type(val[0]))
        else:
            print(flat_key, ":", type(val))


def flatten_dict(data: dict, name: str = "", delimiter: str = "/") -> dict:
    """
    Flatten a dict with keys seperated by the delimiter
    """
    flat_dict = {}

    if not isinstance(data, dict):
        return data

    for key, val in data.items():
        flat_key = key if name == "" else name + delimiter + key
        if isinstance(val, dict):
            flat_dict.update(flatten_dict(data=val, name=flat_key))
        else:
            flat_dict[flat_key] = val
    return flat_dict


def unflatten_dict(d, sep="/"):
    """
    Unflatten a dictionary into a nested one.

    Args:
        d (dict): The dictionary to unflatten. Keys should be strings with parts separated by `sep`.
        sep (str, optional): The separator used in the keys of the flattened dictionary. Defaults to "/".

    Returns:
        dict: A nested dictionary reconstructed from the flattened dictionary.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ref = result
        for part in parts[:-1]:
            d_ref = d_ref.setdefault(part, {})
        d_ref[parts[-1]] = value
    return result


def find_key_recursively(data: dict, key: str):
    """
    Recursively search a dictionary for a key.
    Returns the first found key if multiple keys with the given name are found.

    Args:
        data: Dictionary or any nested structure to search
        key: The key to search for
    Returns:
        The value of the key if found, None otherwise
    """
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            result = find_key_recursively(value, key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursively(item, key)
            if result is not None:
                return result
    return None


def demo_dict_util():
    data = {
        "none": None,
        "bool": True,
        "int": 1,
        "float": 1.0,
        "bool_list": [False, True],
        "int_list": [1, 2, 3],
        "float_list": [1.0, 2.0, 3.0],
        "bool_tuple": (False, True),
        "int_tuple": (1, 2, 3),
        "float_tuple": (1.0, 2.0, 3.0),
        "bool_np": np.array([0, 1], dtype=np.bool_),
        "u08_np": np.array([0, 1, 3], dtype=np.uint8),
        "u16_np": np.array([0, 1, 3], dtype=np.uint16),
        "u32_np": np.array([0, 1, 3], dtype=np.uint32),
        "u64_np": np.array([0, 1, 3], dtype=np.uint64),
        "i08_np": np.array([0, 1, 3], dtype=np.int8),
        "i16_np": np.array([0, 1, 3], dtype=np.int16),
        "i32_np": np.array([0, 1, 3], dtype=np.int32),
        "i64_np": np.array([0, 1, 3], dtype=np.int64),
        "f16_np": np.array([0, 1, 3], dtype=np.float16),
        "f32_np": np.array([0, 1, 3], dtype=np.float32),
        "f64_np": np.array([0, 1, 3], dtype=np.float64),
        "f128_np": np.array([0, 1, 3], dtype=np.float128),
    }
    # data['dict'] = data.copy()

    print("Original data")
    print_dtype(data)

    print("\nFlattened data")
    print_dtype(flatten_dict(data))

    print("\nNumpy-fied data")
    data = dict_numpify(data)
    print_dtype(data)


class TestMain(unittest.TestCase):
    def test_main(self):
        # Call your function and test its output/assertions
        self.assertEqual(demo_dict_util(), None)


if __name__ == "__main__":
    unittest.main()
