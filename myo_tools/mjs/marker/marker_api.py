"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: This file implements an API to apply a marker set to a Mujoco model.

from myo_tools.mjs.core.mjs_api import get_model_spec
from myo_tools.utils.file_ops.xml_utils import load_markerset
from myo_tools.utils.log_ops import logger

logger = logger.getLogger("myo_tools.mjs.marker.marker_api")


def apply_marker_set(
    mjs_handle,
    asset_handle,
    marker_set_handle,
    mocap_bodies=True,
    mocap_rgba=[0, 1, 0, 1],
    add_connectors=True,
    connector_rgba=[1, 0, 0, 1],
    marker_size=[0.005, 0.005, 0.005],  # Size of the marker [rx, ry, rz]
):
    """
    Apply a marker set to a Mujoco model.

    Args:
        mjs_handle (str): The path to the model file, an XML string or an MjSpec containing the model content.
        asset_handle (str): The path to the asset directory containing ./meshes ./textures ./skins etc. If none, provide an empty dictionary
        marker_set_handle (str): The path to the marker set file or an ET object containing the marker set content.
        mocap_bodies (bool, optional): Flag to add mocap bodies to the model. Defaults to True.
        mocap_rgba (list, optional): RGBA color for mocap bodies. Defaults to green: [0, 1, 0, 1].
        add_connectors (bool, optional): Flag to add connectors between markers and mocap bodies. Defaults to True.
        connector_rgba (list, optional): RGBA color for connectors. Defaults to red: [1, 0, 0, 1].
        marker_size (float, optional): Size of the marker. Defaults to None.


    Returns:
        tuple: A tuple containing the updated Mujoco model, handle of preloaded assets, and a list of marker names.
    """

    spec, asset_handle = get_model_spec(
        mjs_handle=mjs_handle, asset_handle=asset_handle
    )

    # Now add markers to the model
    logger.info("Adding markers from marker_set to MyoSkeleton")
    marker_set = load_markerset(marker_set_handle)
    marker_class = spec.find_default("myo_marker")
    assert marker_class, "marker_class not found"
    marker_set_name = marker_set.get("name")

    # Lets start updating the model
    spec.modelname += marker_set_name
    marker_names = []
    for marker in marker_set:
        # Get marker details
        marker_name = marker.attrib.get("name")
        marker_body_name = marker.attrib.get("body")

        # Add mocap body to model
        if mocap_bodies:
            tracker_name = "mocap_" + marker_name
            mocap_body = spec.worldbody.add_body()
            mocap_body.name = tracker_name
            mocap_body.mocap = True
            site = mocap_body.add_site(marker_class)
            site.name = tracker_name
            site.rgba = mocap_rgba
            site.size = marker_size

        # Add markers to model
        body = spec.body(marker_body_name)
        if body is None:
            logger.warning(
                f"Warning {marker_body_name} not found, {marker_name} will not be added"
            )
            continue

        marker_pos = [float(num_str) for num_str in marker.attrib.get("pos").split()]
        site = body.add_site(marker_class)
        site.name = marker_name
        site.pos = marker_pos
        site.size = marker_size

        if mocap_bodies and add_connectors:
            connector = spec.add_tendon(name=f"t_{marker_name}", rgba=connector_rgba)
            connector.wrap_site(marker_name)
            connector.wrap_site(tracker_name)

        marker_names.append(marker_name)
        logger.info(marker_name)

    return spec, asset_handle, marker_names


def get_marker_names(
    marker_set_handle,
):
    """
    Get the names of the markers in the marker set

    Args:
        marker_set_handle (str | ET): The path to the marker set file or an ET object containing the marker set content.

    Returns:
        List: A list containing the marker names
    """

    # Get the marker set
    marker_set = load_markerset(marker_set_handle)

    # Get the marker names
    marker_names = []
    for marker in marker_set:
        marker_name = marker.attrib.get("name")
        marker_names.append(marker_name)

    return marker_names
