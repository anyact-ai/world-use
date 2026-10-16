# Scene reference checks

This example compares an explicitly declared scene with reference observations,
keeps fitting and reserved observations separate, and records failed as well as
passing checks. It runs CPU MuJoCo without a robot daemon or graphics dependency.

```sh
uv run python examples/scene-validation/validate.py runs/scene-validation-synthetic
```

The synthetic fixture uses ideal Coulomb sliding and known camera projections as
its reference. MuJoCo predicts the motion of a freely sliding block. An initial
candidate has the wrong dimensions, camera pose and friction. Refinement reads
only the development file: measured dimensions, the existing camera fitter, and
a small friction search. A different initial velocity and different landmarks
are reserved until the candidate is fixed. The example's image panels are
labelled landmark projections, not photographic appearance comparisons.

The initial candidate fails the reserved checks. The refined candidate passes
the numerical camera, geometry and contact checks, but its overall report remains
`unvalidated`: synthetic observations cannot establish real-to-sim accuracy.
The thresholds in `validate.py` are explicit task tolerances, not estimated
measurement uncertainty. The example does not fit them after seeing the errors.

## Measured pushing

[MIT's planar pushing dataset](https://web.mit.edu/mcube/push-dataset/index.html)
provides metric object and pusher poses, forces and published object dimensions.
[`reference.json`](reference.json) records the geometry, probe, physical assumptions
and pinned primary sources. The scene is constructed independently in ordinary
Python. Raw measurements remain local under ignored `runs/`; consult the authors'
reuse terms before redistributing them.

The [paper](https://arxiv.org/abs/1604.04038) specifies a 9.5 mm diameter, 156 mm
long probe and approximately 0.25 probe/object friction. The pinned collection
scripts place its bottom about 7 mm above the support. The simulated probe has
two horizontal translation joints and follows the measured path through a weld
constraint. This supplies contact velocity while keeping the probe vertical.
Actual probe tracking error is checked separately. Explicit contact pairs prevent
floor-friction fitting from changing probe friction. Probe mass, compliance and
support pressure distribution remain assumptions.

Motion error is the maximum displacement of labelled polygon vertices reconstructed
from object XY and yaw. It measures planar translation and rotation, not full 3D
orientation. Input streams retain their asynchronous timestamps, rebased to their
common interval; yaw is unwrapped before interpolation. Object poses are used for
initialization, fitting and evaluation, and never exposed to a robot policy.
The recordings end with the push. Finite state, penetration and support height are
checked, while settling remains unvalidated because no resting interval is declared.
There are no synchronized camera frames or independent geometry measurements.

### Development diagnostic

One pinned plywood/hex trajectory was inspected during development, including its
formerly reserved temporal tail. The whole trajectory is now development evidence;
rerunning it cannot supply a new held-out result.

```sh
uv run python examples/scene-validation/measured.py \
  runs/scene-reference-mit-hex/measured.json runs/scene-validation-measured
```

The script verifies the reference hash and fits floor friction on the first half.
With the corrected probe model, fitting selects 0.1 from the fixed grid, but the
full-trajectory vertex error increases from 28.16 mm at friction 0.4 to 29.78 mm.
Both fail the existing 3 mm tolerance. Probe tracking is within 0.46 mm and maximum
penetration below 0.06 mm. The evidence is retained in
`runs/scene-validation-measured-03`; the original model's results remain separately
in `runs/scene-validation-measured-02`. Sparse support contacts resist rotation
differently from an assumed uniform pressure distribution. That is an observed
modeling limitation, not a demonstrated complete explanation of the motion error.

### Independent recorded trials

[`rect1-reference.json`](rect1-reference.json) pins five author-provided plywood
square-object trials by archive member and SHA-256. They were selected by filename
before their measurements were downloaded: two fitting cases, two repeated cases
reserved at repetition 99, and one reserved new contact position and angle. Each
push is approximately 1 cm at 20 mm/s on the same setup. These short pushes do not
establish transfer to another setup or to the faster hex push.

Place the five JSON files under the local names declared in that manifest, then run:

```sh
uv run python examples/scene-validation/trials.py \
  runs/scene-reference-mit-rect1 runs/scene-validation-rect1
```

The script fits the same six-value friction grid against the worst error of the two
fitting trials. It writes `frozen-candidate.json` with source hashes, parameters,
tolerances and fitting scores before reading reserved measurements, and checks that
sources stay unchanged throughout evaluation. The initial and fitted candidates
use identical cases and the unchanged 3 mm motion / 1 mm penetration tolerances.
Completed cases are preserved separately, including failures.

The first independent evaluation (`runs/scene-validation-rect1-01`) selected floor
friction 0.6. Maximum planar vertex errors were:

| Trial | Initial, friction 0.4 | Fitted, friction 0.6 |
| --- | ---: | ---: |
| Fit, central contact | 0.75 mm | 0.75 mm |
| Fit, offset contact | 2.38 mm | 2.36 mm |
| Reserved central repeat | 1.49 mm | 1.49 mm |
| Reserved offset repeat | 2.61 mm | 2.33 mm |
| Reserved new contact | **4.25 mm — fail** | **4.31 mm — fail** |

Both reserved repeats satisfy the motion tolerance. The different contact fails,
and fitting does not improve it. Probe tracking stays below 0.08 mm and penetration
below 0.02 mm across these trials. The passing motion checks do not validate the
whole scene: appearance, independent geometry, forces, pressure distribution and
settling remain unvalidated. This friction selection is an empirical fit, not an
independent measurement of the material coefficient. Further development can use
these outcomes diagnostically, but fresh reserved trials are needed for a new
claim of improvement.

## Artifacts and use from Python

[`world_use.scene_validation`](../../src/world_use/scene_validation.py) provides
`scene_identity`, `Observation`, `Validation`, and a wrapper around the existing
camera fitter. A fresh comparison directory contains:

- `specification.json`: source hashes, scene parameters, fixed tolerances,
  observation provenance, coordinate/time alignment and split.
- `scene-validation.json`: separate fit and held-out results for
  `appearance_camera`, `geometry` and `physical_response`, including missing
  checks and explicit limitations.
- `artifacts/`: original reference bytes, metric comparison inputs, raw
  stability samples, and original/comparison images when supplied.

The top-level `status` is `passed`, `failed` or `unvalidated`. A pass covers only
the declared observations and tolerances. Missing required checks, assumed
evidence, or wholly synthetic references cannot yield a complete measured pass.
Individual synthetic numerical checks can pass. Materials, lighting, force
prediction and transfer are explicitly outside this first slice.

Scene identity hashes declared source/asset bytes plus parameters; include all
geometry, calibration, dynamics, initial conditions, engine version and solver
settings. Any relevant change requires another report. The helper cannot detect
undeclared dependencies. Evaluation should compare the report's
`scene_revision` with the current declared revision.

Fit and reserved observation IDs cannot be reused. Identical source bytes in
both splits require explicit nonoverlapping time intervals, and motion checks
must stay within those intervals. Camera fitting refuses reserved observations.
These are safeguards for trusted experiment code, not a sandbox against a caller
that mislabels evidence.
