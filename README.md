# myo_tools

`myo_tools` is a collection of functions and utilities designed to support common workflows involving human embodiment and embodied motion. While originally developed to support MyoLab.ai’s work on human embodied intelligence, the APIs are general-purpose and can be extended to other embodied systems and applications.

An overview of the available functionality is provided below. Refer to the individual module folders for detailed feature descriptions and functionality details.

## Modules

### → MuJoCo ([mj](myo_tools/mj))
[MuJoCo](https://mujoco.readthedocs.io/en/stable/APIreference/index.html) is the core physics engine used to simulate motion at MyoLab. This module provides core routines for working directly with MuJoCo’s **compiled** data structures: `MjModel` and `MjData`.

These APIs are intended for robust instantiation, inspection, and manipulation of compiled MuJoCo models and simulation state.


### → MuJoCo Spec ([mjs](myo_tools/mjs))
As of MuJoCo 3.2.0, models can be created and modified using the
[MjSpec](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html)
structure and its associated API. This data structure has a one-to-one correspondence with MJCF; MuJoCo’s own MJCF and URDF XML parsers use this API internally.

This module provides foundational APIs that operate on **uncompiled** models and produce MuJoCo `mjSpec` structures. These tools are useful for editing and programmatically generating models while preserving editability prior to compilation.


### → Utilities ([utils](myo_tools/utils))
The utilities module contains functionality that is independent of MuJoCo, including:
- File and path handling
- Logging helpers
- Mathematical utilities (e.g., tensor and quaternion operations)

This module provide helpful utilities to accelerate your workflows.


## Installation

Install the latest release from [PyPI](https://pypi.org/project/myo_tools/):

```bash
pip install myo_tools
```
Or, install in editable mode using
```bash
git clone https://github.com/myolab/myo_tools.git
cd myo_tools
pip install -e '.[test]'
```
