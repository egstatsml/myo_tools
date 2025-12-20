# marker_set
_MyoSkeleton_ presents a comprehensive parameterization of human anatomy that can enable diverse usecases across multiple domains. Such as -
  1. Motion analysis in Biomechanics
  2. Motion retargeting for behaviors in graphics and animation
  3. Embodiment matching/transfer in Robotics
  4. Human environment/robot interactions in the field of human-robot interaction

While these diverse applications can be powered by the universal parameterization of the human body presented by _MyoSkeleton_, each of them also requires attention to a set of key landmarks (markers) in conjunction, usually called `marker_set`.

## API ([marker_api](marker_api.py))
The marker_set API provides an easy-to-use programmatic interface for adding usecase-specific `marker_set` (or sites for MuJoCo users) over the _MyoSkeleton_. A marker_set can be applied to a model using -

``` python
model_w_marker_set, marker_set_names = apply_marker_set(
    model_path = <pathto/myoskeleton.xml>,
    asset_dir = <pathto/assets.xml>
    marker_set_handle = <pathto/marker_setdefinition.xml> | <ET object>
)
```
The function takes as inputs
- an input path to a MuJoCo model on which we want to add the marker_set,
- an asset directory (this is only relevant if your model uses meshes or textures) to make sure when loading the model assets are resolved properly, and
- a path to a marker_set definition file (outlined next).


## marker_set definition format ([examples](https://github.com/myolab/myo_model/tree/main/myo_model/markerset))
A `marker_set` constitutes a set of key landmarks over the human body. While _MyoSkeleton_ is universal, these landmarks are usecase specific and are best outlined by usecase practitioners/experts; for example -
  1. to mark the location of motion capture markers over MyoSkeleton
  2. to mark, easily access, and compute properties of physiological landmarks over the human body
  3. to mark, and easily access landmarks for rewards design in behavior synthesis algorithms

The `marker_set` is designed as an XML using the following schema
``` xml
<markers name="str">
  <marker name="str" body="str" pos="float float float" movable="float"/>
  <marker name="str" body="str" pos="float float float" movable="float"/>
  ...
</markers>
```
We begin by defining the `name` of the marker_set. For each marker in the marker_set, specify
- `name` of the marker (for ease of reference),
- `body` name of the marker's parent body (must exist) in the model,
- `pos` relative position of the marker in the parent body, and
- `movable` a measure on the given marker. The value should be normalized between 0 and 1. This measure can be useful for various usecase such as
    - to represent levels of mobility/flexibility of the marker in motion capture setup (default)
    - to represent weights over the markers in optimization routines such as inverse kinematics, trajectory optimization, etc
    - to capture sensor readings from a measuring instrument such as ground reaction forces from foot sensors, etc.

## Practitioner's guide
  - A [minimal example](../../examples/examine_marker.py)  of how to use the API
  - A few pre-packaged marker_set definitions for reference:
    - [CMU marker_set](https://github.com/myolab/myo_model/blob/main/myo_model/markerset/cmu_markerset.xml) used in [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/)
    - [Movi Metrabs marker_set](https://github.com/myolab/myo_model/blob/main/myo_model/markerset/movi_metrabs_markerset.xml) used in [MoVi: A Large Multipurpose Motion and Video Dataset](https://www.biomotionlab.ca/movi/)
