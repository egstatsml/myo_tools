"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: General configurations for tests.

import os

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    tmp_dir = tmp_path / "myo_tools_test"
    tmp_dir.mkdir()
    return tmp_dir


@pytest.fixture
def assets_folder():
    return os.path.join(os.path.dirname(__file__), "assets")


@pytest.fixture
def model_path(assets_folder):
    return os.path.join(assets_folder, "test_model.xml")


@pytest.fixture
def model_XML(assets_folder):
    return os.path.join(assets_folder, "test_model.xml")


@pytest.fixture
def skin_skn(assets_folder):
    return os.path.join(assets_folder, "meshes/test_skn.skn")


@pytest.fixture
def skin_texture(assets_folder):
    return os.path.join(assets_folder, "textures/10x10.png")


@pytest.fixture
def marker_sets_path(assets_folder):
    return os.path.join(assets_folder, "marker_sets/")


@pytest.fixture
def trc_file_path(assets_folder):
    return os.path.join(assets_folder, "mocap_files/test_file.trc")


@pytest.fixture
def markerset_file_path(assets_folder):
    return os.path.join(assets_folder, "marker_sets/test_marker.xml")
