"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: tests for marker api


import os

import mujoco
import pytest

from myo_tools.mjs.marker import marker_api
from myo_tools.utils.file_ops.xml_utils import load_xml


@pytest.mark.parametrize(
    "marker_set_fname, numMarkers",
    [
        ["test_marker.xml", 2],
    ],
)
def test_marker_set_from_path(
    model_path, assets_folder, marker_sets_path, marker_set_fname: str, numMarkers: int
):
    marker_set_path = os.path.join(marker_sets_path + marker_set_fname)

    assert os.path.isfile(
        marker_set_path
    ), f"marker_set file {marker_set_fname} is not found"

    apply_and_run_marker_set(model_path, assets_folder, marker_set_path)


@pytest.mark.parametrize(
    "marker_set_fname, numMarkers",
    [
        ["test_marker.xml", 2],
    ],
)
def test_marker_set_from_xml_string(
    model_path, assets_folder, marker_sets_path, marker_set_fname: str, numMarkers: int
):
    marker_set_path = os.path.join(marker_sets_path + marker_set_fname)
    marker_set = load_xml(marker_set_path)

    assert marker_set is not None

    assert (
        len(marker_set) == numMarkers
    ), f"marker_set {marker_set_fname} has {len(marker_set)} instead of {numMarkers} markers"

    apply_and_run_marker_set(model_path, assets_folder, marker_set)


def apply_and_run_marker_set(model_path, assets_folder, marker_set_handle):
    mj_spec, asset_handle, marker_set_names = marker_api.apply_marker_set(
        model_path, assets_folder, marker_set_handle
    )

    assert isinstance(mj_spec, mujoco.MjSpec)

    mj_model = mj_spec.compile()

    mj_data = mujoco.MjData(mj_model)
    for i in range(2):
        mujoco.mj_step(mj_model, mj_data)

    ## test input from MjSpec
    spec = mujoco.MjSpec().from_file(model_path)

    mjspec, asset_handle, marker_set_names = marker_api.apply_marker_set(
        spec, {}, marker_set_handle
    )

    assert mjspec is not None
    assert isinstance(mjspec, mujoco.MjSpec)

    mj_model = mjspec.compile()

    mj_data = mujoco.MjData(mj_model)
    for i in range(2):
        mujoco.mj_step(mj_model, mj_data)
