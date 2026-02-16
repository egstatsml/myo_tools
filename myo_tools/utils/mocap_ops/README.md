# Motion Capture Operations (`mocap_ops`)

This module provides utilities for loading, processing, and cleaning motion capture data from various file formats.

## Overview

The `mocap_ops` module handles motion capture data in multiple formats commonly used in biomechanics and animation research, including C3D and TRC files. It provides robust loading mechanisms with fallback options, data cleaning utilities, and integration with markerset definitions.

## Contents

- [Files](#files)
  - [`c3d_utils.py`](#c3d_utilspy) - C3D file loading utilities
  - [`trc_utils.py`](#trc_utilspy) - TRC file loading utilities
  - [`mocap_utils.py`](#mocap_utilspy) - High-level motion capture processing
- [Data Format](#data-format)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)
- [Dependencies](#dependencies)

## Files

<details id="c3d_utilspy">
<summary><b><code>c3d_utils.py</code></b> - Utilities for loading C3D motion capture files</summary>

**Key Functions:**
- `ezc3d_loader(c3d_file_path)`: Load C3D files using the ezc3d library
- `c3dpy_loader(c3d_file_path)`: Load C3D files using the python-c3d library (fallback)
- `from_c3d_to_numpy(c3d_file_path)`: Main loader that attempts ezc3d first, then falls back to c3dpy

**Returns:**
- Motion data as numpy arrays with shape `(num_frames, num_markers, 3)`
- Marker names list
- Frame rate (Hz)

**Dependencies:**
- `c3d` (python-c3d)
- `ezc3d`

</details>

<details id="trc_utilspy">
<summary><b><code>trc_utils.py</code></b> - Utilities for loading TRC motion capture files, commonly used with OpenSim</summary>

**Key Functions:**
- `trc_loader(trc_file_path)`: Load TRC files
- `from_trc_to_numpy(trc_file_path)`: Main TRC loading function

**Features:**
- Parses TRC header metadata (DataRate, CameraRate)
- Handles variable header formats
- Returns data in original coordinate system (coordinate transformations handled by `mocap_utils.py`)

**Returns:**
- Motion data as numpy arrays with shape `(num_frames, num_markers, 3)`
- Marker names list
- Frame rate (Hz)

**Note:** For coordinate system conversions (Y-up to Z-up, etc.), use the higher-level functions in `mocap_utils.py` which provide the `rotation` parameter.

</details>

<details id="mocap_utilspy">
<summary><b><code>mocap_utils.py</code></b> - High-level utilities for loading and processing motion capture data with markerset integration</summary>

**Key Functions:**

#### `rotate_mocap_yup_to_zup(motion_data)`
Rotate motion capture data from Y-up coordinate system (OpenSim) to Z-up coordinate system (MuJoCo).

**Parameters:**
- `motion_data`: Motion capture data array with shape `(num_frames, num_markers, 3)`

**Returns:**
- Transformed motion data array in Z-up coordinate system

#### `rotate_mocap_ydown_to_zup(motion_data)`
Rotate motion capture data from Y-down coordinate system (OpenCV) to Z-up coordinate system (MuJoCo).

**Parameters:**
- `motion_data`: Motion capture data array with shape `(num_frames, num_markers, 3)`

**Returns:**
- Transformed motion data array in Z-up coordinate system

#### `load_trackers(trackers_file_path, mocap_scale, clip_length, rotation)`
Load tracker data from various file formats (.c3d, .trc, .csv, .parquet).

**Parameters:**
- `trackers_file_path`: Path to the trackers file
- `mocap_scale`: Scale factor for mocap data (e.g., 1000 for mm to m conversion)
- `clip_length`: Number of frames to clip (use -1 for full length)
- `rotation`: Rotation type to apply. Options are:
  - `None`: No rotation applied (default)
  - `"yup_to_zup"`: Rotate from OpenSim (Y-up) to MuJoCo (Z-up)
  - `"ydown_to_zup"`: Rotate from OpenCV (Y-down) to MuJoCo (Z-up)

#### `load_trackers_and_markerset(trackers_file_path, markerset_handle, ...)`
Comprehensive loading function that integrates tracker data with markerset definitions.

**Parameters:**
- `trackers_file_path`: Path to trackers file
- `markerset_handle`: Markerset definition (file path or XML element tree)
- `mocap_scale`: Scale factor (default: 1000)
- `rotation`: Rotation type to apply (default: None). Options: `None`, `"yup_to_zup"`, `"ydown_to_zup"`
- `clip_length`: Clip length (default: -1, full length)
- `chunk_size`: Size of chunks to split data into (default: -1, no chunking)

**Returns:**
- `motion_data_list`: List of lists containing motion data per subject and chunk
- `markerset`: Filtered markerset with only markers present in tracker data
- `framerate`: Frame rate (Hz) calculated from the input file

**Features:**
- Validates consistency between tracker data and markerset definitions
- Handles multi-subject tracker files with naming convention `subject:marker_name`
- Automatic filtering and mapping of markers
- Duplicate marker detection
- Missing marker warnings
- Optional data chunking for processing large files

</details>

## Data Format

All functions return motion data in a consistent format:
- **Shape**:
  - For `load_trackers` and `from_*_to_numpy`: `(num_frames, num_markers, 3)` where the last dimension represents [x, y, z] coordinates
  - For `load_trackers_and_markerset`, returns a list of lists: one list per subject, each containing chunks with shape `(num_frames, num_markers, 3)`
- **Units**: Typically meters after applying `mocap_scale`
- **Coordinate System**: Z-up (MuJoCo convention) after conversion

## Usage Examples

### Load C3D File
```python
from myo_tools.utils.mocap_ops.c3d_utils import from_c3d_to_numpy

motion_data, marker_names, framerate = from_c3d_to_numpy("path/to/file.c3d")
print(f"Shape: {motion_data.shape}, FPS: {framerate}")
```

### Load TRC File
```python
from myo_tools.utils.mocap_ops.trc_utils import from_trc_to_numpy

# Load TRC file (returns data in original coordinate system)
motion_data, marker_names, framerate = from_trc_to_numpy("path/to/file.trc")

# For coordinate system conversion, use load_trackers from mocap_utils:
from myo_tools.utils.mocap_ops.mocap_utils import load_trackers

motion_data, marker_names, framerate = load_trackers(
    "path/to/file.trc",
    mocap_scale=1000,
    clip_length=-1,
    rotation="yup_to_zup"  # Convert from OpenSim to MuJoCo coordinates
)
```

### Load with Markerset Integration
```python
from myo_tools.utils.mocap_ops.mocap_utils import load_trackers_and_markerset

motion_data_list, markerset, framerate = load_trackers_and_markerset(
    trackers_file_path="path/to/mocap.c3d",
    markerset_handle="path/to/markerset.xml",
    mocap_scale=1000,       # Convert from mm to m
    rotation="yup_to_zup",  # Convert from OpenSim to MuJoCo coordinates
    clip_length=-1,         # Use full length
    chunk_size=-1,          # No chunking
)
```

## Error Handling

The module implements robust error handling with fallback mechanisms:
- C3D loading attempts `ezc3d` first, then falls back to `c3d-py`
- Detailed error logging via the `myo_tools.utils.log_ops.logger`
- Warnings for missing markers and data quality issues
- Exceptions for critical failures (duplicate markers, unsupported formats)

## Dependencies

- `numpy`: Array operations and data structures
- `c3d`: Python-c3d library for C3D file reading
- `ezc3d`: Alternative C3D library (preferred)
- `myo_tools.utils.file_ops.dataframe_utils`: DataFrame to array conversion
- `myo_tools.utils.file_ops.xml_utils`: XML markerset loading
- `myo_tools.utils.tensor_ops.tensor_utils`: Gap filling and static tracker removal
- `myo_tools.utils.log_ops`: Logging utilities
