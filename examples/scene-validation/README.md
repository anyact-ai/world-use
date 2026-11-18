# Scene reference checks

This example compares a declared MuJoCo scene with reference observations. It fits parameters on some
observations, checks the result on others held back until the fit is written, and records failed checks as well
as passing ones. It runs on CPU MuJoCo, without a robot daemon or graphics dependencies.

It has three parts:

- A synthetic fixture tests the tooling. The refined candidate passes its camera, geometry and contact checks, but
  the report stays `unvalidated`: synthetic observations cannot show that a simulation matches reality.
- One push from MIT's planar pushing dataset, inspected during development, serves as a diagnostic. Both
  candidates miss the 3 mm motion tolerance (28.16 mm and 29.78 mm).
- Five other recorded MIT pushes were chosen before their measurements were read. After fitting friction on two,
  both reserved repeats pass the 3 mm tolerance and the reserved new contact fails (4.25 mm before fitting,
  4.31 mm after).

## Synthetic fixture

```sh
uv run python examples/scene-validation/validate.py runs/scene-validation-synthetic
```

The reference is ideal Coulomb sliding with known camera projections; MuJoCo predicts the motion of a freely
sliding block. The initial candidate has the wrong dimensions, camera pose and friction, and fails the reserved
checks. Refinement reads only the fit file: it takes the dimensions from it, fits the camera with the existing
camera fitter and searches a small friction grid. The reserved file, with a different initial velocity and
different landmarks, is compared only after the refined parameters are written. The image panels are labelled
landmark projections, not photographic comparisons. The thresholds in `validate.py` are task tolerances fixed in
advance, not estimated measurement uncertainty.

## Measured pushing

[MIT's planar pushing dataset](https://web.mit.edu/mcube/push-dataset/index.html) records object and pusher poses
in metres, forces, and the objects' published dimensions. [`reference.json`](reference.json) records the
geometry, the probe, the physical assumptions and pinned links to the sources, and `measured.py` builds its own
MuJoCo scene from it. Keep downloaded measurements under the ignored `runs/` folder, and check the authors' reuse
terms before redistributing them.

The [paper](https://arxiv.org/abs/1604.04038) gives a 9.5 mm diameter, 156 mm long probe and a probe/object
friction of about 0.25; the pinned collection scripts put its bottom about 7 mm above the support. The simulated
probe slides on two horizontal joints and is welded to a target that follows the measured path, so it stays
vertical and pushes at the recorded velocity; how closely it follows the path is checked separately. Explicit
contact pairs keep the floor-friction fit from changing the probe's friction. Probe mass, compliance and the
support's pressure distribution are assumed.

Motion error is the largest displacement of the object's polygon vertices, placed from its recorded XY and yaw. It
covers planar translation and rotation, not full 3D orientation. The pusher and object streams keep their own
timestamps, shifted to start at the beginning of their common interval, and yaw is unwrapped before
interpolation. The object poses initialise, fit and evaluate the scene; no robot policy is involved. The
recordings end with the push, so finite state, penetration and support height are checked but settling is not.
There are no synchronised camera frames or independent geometry measurements.

### Development diagnostic

One pinned plywood/hex push, including the later part originally held back, was inspected during development. The
whole push is therefore development data, and rerunning it cannot give a held-out result.

```sh
uv run python examples/scene-validation/measured.py \
  runs/scene-reference-mit-hex/measured.json runs/scene-validation-measured
```

The script checks the input's hash and fits floor friction on the first half of the push. With the corrected probe
model, the fit chooses 0.1 from the fixed grid, but the full-trajectory vertex error rises from 28.16 mm at
friction 0.4 to 29.78 mm; both fail the 3 mm tolerance. The probe follows its path within 0.46 mm and penetration
stays below 0.06 mm. The results are in `runs/scene-validation-measured-03`, and those of the original probe model
in `runs/scene-validation-measured-02`. The few support contacts MuJoCo makes resist rotation differently from the
assumed uniform pressure. That is one observed modelling limitation, not a complete explanation of the error.

### Independent recorded trials

[`rect1-reference.json`](rect1-reference.json) pins five plywood square-object trials from the authors by archive
member and SHA-256. They were chosen by filename before their measurements were downloaded: two fitting cases, two
repeats reserved at repetition 99, and one reserved new contact position and angle. Each push moves about 1 cm at
20 mm/s on the same setup. Such short pushes say nothing about transfer to another setup or to the faster hex push.

Place the five JSON files under the local names declared in that manifest, then run:

```sh
uv run python examples/scene-validation/trials.py \
  runs/scene-reference-mit-rect1 runs/scene-validation-rect1
```

The script picks the friction from the same six-value grid that minimises the worse of the two fitting errors. It
writes `frozen-candidate.json` (source hashes, parameters, tolerances and fitting scores) before reading any
reserved measurement, and stops if a source file changes during evaluation. Both candidates face the same cases
and the same 3 mm motion and 1 mm penetration tolerances. Each completed case is saved as it finishes, failures
included.

The first independent evaluation (`runs/scene-validation-rect1-01`) selected floor friction 0.6. Maximum planar
vertex errors were:

| Trial | Initial, friction 0.4 | Fitted, friction 0.6 |
| --- | ---: | ---: |
| Fit, central contact | 0.75 mm | 0.75 mm |
| Fit, offset contact | 2.38 mm | 2.36 mm |
| Reserved central repeat | 1.49 mm | 1.49 mm |
| Reserved offset repeat | 2.61 mm | 2.33 mm |
| Reserved new contact | 4.25 mm — fail | 4.31 mm — fail |

Both reserved repeats meet the motion tolerance; the new contact fails, and fitting does not help it. Probe
tracking stays below 0.08 mm and penetration below 0.02 mm. These motion checks leave appearance, independent
geometry, forces, pressure distribution and settling unvalidated, and the chosen friction is a fit, not a
measurement of the material. Further work can use these results for diagnosis, but a claim of improvement needs
new reserved trials.

## Use from Python

[`world_use.scene_validation`](../../src/world_use/scene_validation.py) provides `scene_identity`, `Observation`,
`Validation` and `fit_camera`, a wrapper around the existing camera fitter. A comparison directory contains:

- `specification.json`: source hashes, scene parameters, fixed tolerances, observation provenance, coordinate and
  time alignment, and split.
- `scene-validation.json`: separate fit and held-out results for `appearance_camera`, `geometry` and
  `physical_response`, with missing checks, limitations, and the path and hash of every artifact.
- `artifacts/`: the original reference bytes, metric comparison inputs, raw stability samples, and the original
  and comparison images when supplied.

The top-level `status` is `passed`, `failed` or `unvalidated`, and a pass covers only the declared observations
and tolerances. A report missing a required check, or with a reference that is not entirely measured, cannot pass:
at best it is `unvalidated`, although its individual numerical checks can pass. Materials, lighting, force
prediction and transfer are not checked.

`scene_identity` hashes the source and asset files and the parameters you declare, so declare all geometry,
calibration, dynamics, initial conditions, engine version and solver settings; it cannot see anything else. A
changed scene needs a new report: compare the report's `scene_revision` with the current one before relying on it.

An observation name cannot be repeated or used in both splits. One file can serve both splits only with disjoint
time intervals, and motion checks must stay within the observation's interval. `fit_camera` refuses reserved
observations. These checks catch mistakes in trusted experiment code; they cannot stop a caller that mislabels its
data.
