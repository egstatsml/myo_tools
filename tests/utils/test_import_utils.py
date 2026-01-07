"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for import_utils

from unittest.mock import MagicMock, patch

import pytest

from myo_tools.utils.file_ops.import_utils import (
    get_package_directory,
    get_repo_info,
    list_files,
)


def test_list_files():
    with patch("os.path.exists", return_value=True):
        with patch("os.listdir", return_value=["file1.txt", "file2.txt", "dir1"]):
            with patch("os.path.isfile", side_effect=lambda x: not x.endswith("dir1")):
                files = list_files("dummy_directory")
                assert files == ["file1.txt", "file2.txt"]

    with patch("os.path.exists", return_value=False):
        with pytest.warns(
            ResourceWarning, match="Directory: dummy_directory not found"
        ):
            files = list_files("dummy_directory")
            assert files == []


def test_get_package_directory():
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_spec = MagicMock()
        mock_spec.origin = "/path/to/package/__init__.py"
        mock_find_spec.return_value = mock_spec

        package_directory = get_package_directory("dummy_package")
        assert package_directory == "/path/to/package"

    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(
            ImportError, match="Cannot find the package 'dummy_package'"
        ):
            get_package_directory("dummy_package")

    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_spec = MagicMock()
        mock_spec.origin = None
        mock_find_spec.return_value = mock_spec

        with pytest.raises(
            ImportError, match="Cannot find the path for package 'dummy_package'"
        ):
            get_package_directory("dummy_package")


@patch("git.Repo")
def test_get_repo_info(mock_repo):
    mock_repo.return_value.head.object.hexsha = "abcdef1234567890"
    mock_repo.return_value.remotes[0].config_reader.get.return_value = (
        "https://github.com/user/repo.git"
    )

    repo_name, sha = get_repo_info("dummy_directory", "dummy_package")
    assert repo_name == "repo"
    assert sha == "abcdef1234567890"

    mock_repo.side_effect = Exception
    with patch(
        "myo_tools.utils.file_ops.import_utils.get_repo_and_sha",
        return_value=("fallback_repo", "fallback_sha"),
    ):
        repo_name, sha = get_repo_info("dummy_directory", "dummy_package")
        assert repo_name == "fallback_repo"
        assert sha == "fallback_sha"
