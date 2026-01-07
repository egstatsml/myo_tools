"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

DESC = """A simple utility to quickly examine MuJoCo models"""

import time

import click
import mujoco
from mujoco import viewer

from myo_tools.mj.core.mj_api import get_model_data

is_paused = True


def key_callback(keycode):
    global is_paused
    if chr(keycode) == " ":
        is_paused = not is_paused


def examine_sim(model_handle, asset_handle):

    # resolve model, data, and viewer
    mj_model, mj_data, asset = get_model_data(
        model_handle=model_handle, asset_handle=asset_handle
    )
    window = viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )

    # run viewer to examine the model
    global is_paused
    while window.is_running():

        if is_paused:
            step_duration = 0.2
        else:
            step_begin = time.time()
            mujoco.mj_step(mj_model, mj_data)

            # adjust remaining duration
            step_duration -= time.time() - step_begin

        # adjust total step duration
        if step_duration > 0.002:
            time.sleep(step_duration)
        window.sync()

    # Close and cleanup
    window.close()


@click.command(help=DESC)
@click.option(
    "--sim",
    "-s",
    type=str,
    help="simultion model to examine (XML/MJB path or compiled model)",
    required=True,
)
@click.option(
    "--asset", "-a", type=str, help="rootdir for meshes, textures, etc", default=None
)
def main(sim, asset):
    print(DESC)
    examine_sim(model_handle=sim, asset_handle=asset)


if __name__ == "__main__":
    main()
