"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

DESC = """
This tutorials on MyoMarker outlines: How to load marker set onto MyoSkeleton model
Example:
    mjpython myo_tools/examples/examine_marker.py
        --sim <path_to_model>/model.xml
        --marker <path_to_marker_set>/marker_set.xml
        --assset <path_to_model_assets> (optional)
"""
import click
import mujoco

from myo_tools.examples.examine_mj import examine_sim
from myo_tools.mj.marker.marker_api import apply_marker_set


@click.command(help=DESC)
@click.option("-s", "--sim", type=str, help="simulation model", required=True)
@click.option("-a", "--asset", type=str, help="asset path", default=None)
@click.option("-m", "--marker", type=str, help="marker path", required=True)
def main(sim, asset, marker):

    output_model, asset_handle, marker_set_names = apply_marker_set(
        model_handle=sim, asset_handle=asset, marker_set_handle=marker
    )

    # lets examine the returned model
    examine_sim(model_handle=output_model, asset_handle=None, motion=None)

    # lets look at marker locations
    mjdata = mujoco.MjData(output_model)
    mujoco.mj_fwdPosition(output_model, mjdata)
    print("\nMarker locations:")
    for m_name in marker_set_names:
        print(f"{m_name}: \t {mjdata.site(m_name).xpos}")


if __name__ == "__main__":
    main()
