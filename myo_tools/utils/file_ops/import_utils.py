"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Utilities on packages/import/ directories

import importlib.util
import os
import subprocess
import warnings
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

from myo_tools.utils.log_ops import logger

logger = logger.getLogger("myo_tools.utils.file_ops.import_utils")


import git


def list_files(directory):
    """
    Returns a list of files in the specified directory.

    Args:
        directory (str): The directory to search for files.

    Returns:
        list: A list of filenames in the directory.

    """
    files = []

    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                files.append(filename)
    else:
        warnings.warn(f"Directory: {directory} not found", ResourceWarning)
    return files


def get_package_directory(package_name):
    """
    Get the directory of a Python package.

    Args:
        package_name (str): The name of the package.

    Returns:
        str: The directory path of the package.

    Raises:
        ImportError: If the package cannot be found or its path cannot be determined.
    """
    # Find the specification of the module
    spec = importlib.util.find_spec(package_name)

    if spec is None:
        raise ImportError(f"Cannot find the package '{package_name}'")

    # Get the origin of the module
    module_path = spec.origin

    if module_path is None:
        raise ImportError(f"Cannot find the path for package '{package_name}'")

    # Get the directory of the module
    package_directory = os.path.dirname(module_path)

    return package_directory


def get_repo_and_sha(package_name):
    """
    Get the repository name and SHA hash from the package name.
    """
    try:
        # Locate the package directory
        package_dir = os.path.dirname(__import__(package_name).__file__)

        # Check if it's a Git repository
        repo_url = subprocess.check_output(
            ["git", "-C", package_dir, "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        sha = subprocess.check_output(
            ["git", "-C", package_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        return repo_url, sha
    except subprocess.CalledProcessError:
        logger.error(f"Not a Git repository: {package_dir}.")
    except Exception as e:
        logger.error(f"Failed to retrieve Git information: {e}")

    # Fallback to metadata for repository URL
    try:
        meta = metadata(package_name)
        if "Home-page" in meta:
            return meta["Home-page"], "unknown"
        if "Project-URL" in meta:
            return meta["Project-URL"], "unknown"
    except PackageNotFoundError:
        logger.error(f"Package '{package_name}' not found in installed metadata.")
    except Exception as e:
        logger.error(f"Failed to retrieve package metadata: {e}")

    return "unknown", "unknown"


def get_repo_info(current_dir, package_name):
    """
    Get the repository name and SHA hash from either the current directory or package name.

    This function first attempts to get repository information by treating the current_dir
    as a git repository. If that fails (e.g. if the code was pip installed), it falls back
    to getting the information from the package metadata.

    The repository name is extracted from the remote URL, while the SHA represents the
    current commit hash of the repository's HEAD.

    Args:
        current_dir (str): The current directory.
        package_name (str): The package name.

    Returns:
        repo_name (str): The repo name.
        sha (str): The sha.
    """
    try:
        repo = git.Repo(current_dir, search_parent_directories=True)
        sha = repo.head.object.hexsha
        remote_url = repo.remotes[0].config_reader.get("url")
        repo_name = os.path.splitext(os.path.basename(remote_url))[0]
    except Exception:
        # probably not in a git repo, the code was pip installed
        repo_name, sha = get_repo_and_sha(package_name)
    return repo_name, sha


def get_asset_dict(asset_dir: str, key_prefix: str | None = None) -> dict[str, bytes]:
    """
    Constructs a dictionary of binary assets from the specified directory structure.

    Args:
    - asset_dir (str): The root directory containing asset subdirectories.
    - key_prefix (str, optional): Prefix to prepend to asset dictionary keys.

    Returns:
    - dict: Dictionary mapping asset paths to binary content, of the form
        `{filename: bytestring}`.

    Raises:
    - ResourceWarning: If any of the asset subdirectories are not found.
    """

    asset_dir = str(Path(asset_dir).resolve()) + "/"
    ASSETS = {}
    logger.info(f"Loading assets from {asset_dir}")
    for file_name in list_files(asset_dir + "meshes/"):
        with open(asset_dir + "meshes/" + file_name, "rb") as f:
            key = (
                str(Path.joinpath(Path(key_prefix), "meshes", file_name))
                if key_prefix
                else file_name
            )
            ASSETS[key] = f.read()

    for file_name in list_files(asset_dir + "../scene/"):
        with open(asset_dir + "../scene/" + file_name, "rb") as f:
            key = (
                str(Path.joinpath(Path(key_prefix), "../scene", file_name))
                if key_prefix
                else file_name
            )
            ASSETS[key] = f.read()

    for file_name in list_files(asset_dir + "textures/"):
        with open(asset_dir + "textures/" + file_name, "rb") as f:
            key = (
                str(Path.joinpath(Path(key_prefix), "textures", file_name))
                if key_prefix
                else file_name
            )
            ASSETS[key] = f.read()

    for file_name in list_files(asset_dir + "skins/"):
        logger.info(f"Loading skin file: {file_name}")
        with open(asset_dir + "skins/" + file_name, "rb") as f:
            key = (
                str(Path.joinpath(Path(key_prefix), "skins", file_name))
                if key_prefix
                else file_name
            )
            ASSETS[key] = f.read()

    for file_name in list_files(asset_dir + "skins/muscles/"):
        logger.info(f"Loading skin file: {file_name}")
        with open(asset_dir + "skins/muscles/" + file_name, "rb") as f:
            key = (
                str(Path.joinpath(Path(key_prefix), "skins/muscles", file_name))
                if key_prefix
                else file_name
            )
            ASSETS[key] = f.read()

    return ASSETS
