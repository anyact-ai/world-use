# Measuring from pictures

An agent can turn pixels in a camera picture into numbers a plan can use, and make a plan stop once those
numbers are too old to trust. It learns three things: a camera frame, a measurement, and `requires`.
Checking the result is the same step again: measure again and compare. The
[perception example](../examples/perception) does all of it on a simulated arm.

Depth comes only from simulated (MuJoCo) cameras for now. Other cameras give pictures without depth, and
measuring them reports `missing_depth`.

## The flow

```text
camera_frame(camera="top", depth=true)               -> picture + frame id
measure_pixels(frame=ID, point=[374, 112], target="block")
                                                     -> measurement id, surface_center, from_tool + picture
run(plan=[...], requires=[{"evidence": MEASUREMENT_ID, "max_age_s": 30}])
measure_pixels(...) again after the step, and compare
```

MCP `run`, `job` and `answer` also accept `camera="top", depth=true`: once the job reaches a checkpoint or
outcome, the same reply includes a fresh picture and `frame.id` for measuring. A job still running at the wait
timeout returns without a picture; wait with `job`, never resubmit it. A camera failure preserves the job and
outcome and reports `camera_error`.

To try it, start the block workcell (`wu up --workcell block`): its block's top face is near pixel (374, 112)
of the `top` camera, and a measurement there gives a `surface_center` of about [0.34, 0.03, 0.25], the middle
of that face in the work frame.

`camera_frame` returns the picture at native resolution with nothing drawn on it, and a frame id. Pixel
coordinates refer to that picture: `look` scales and annotates its pictures, so do not take coordinates
from it. `inspect_image(frame, crop=[left, top, right, bottom])` shows part of a frame again, enlarged,
with the matrix that maps its pixels back to the frame's. The daemon keeps the last 8 frames (64 MiB).

`measure_pixels` measures the visible surface under a point or a box of a frame. The daemon measures its
own copy of the frame with the camera calibration it had when the picture was taken, saves the result
with the run, and returns it. A box should hold one surface: when its depths split into two surfaces more
than 5 cm apart, the measurement comes back invalid, with the reason `mixed_depth_surfaces`. A point is the
simplest choice.

The same calls in Python:

```python
from world_use import Client

robot = Client()
frame = robot.frame("top", depth=True)               # frame.image, frame.depth (metres), frame.id
block = robot.measure(frame, point=[374, 112], target="block")
plan = [{"do": "line", "up": 0.05}]
robot.run(plan, requires=[{"evidence": block["id"], "max_age_s": 30}], wait=60)
```

`Client.measure` also takes a boolean `mask` the size of the picture, for example from a tracker.
`world_use.perception.measure(frame, point=...)` computes the same points locally, without registering them.

### Features on a measured plane

At a hole or silhouette, the selected pixel can contain background depth rather than the edge's depth.
If the feature lies on a visible flat surface, select a box wholly inside that surface and pass it as `plane`:

```python
edge = robot.measure(frame, point=[420, 210],
                     plane={"box": [360, 180, 390, 205], "max_error_m": .001})
```

The same option works with MCP `measure_pixels`. It fits a plane to the box's depth samples, then intersects
the point's camera ray with that plane. It supports tilted planes and cameras; no surface height is supplied.
The caller establishes that the feature belongs to the selected plane. This is an inferred point on that
plane, not a depth observation at the point; selecting the wrong flat surface can still give a valid result.

The box needs depth samples spread across an area. The fit is refused if any sample's perpendicular error
exceeds `max_error_m`, or the ray is nearly parallel to the plane or intersects behind the camera. A clean,
broad patch gives a better estimate than a tiny or occluded patch. Plane fit error is not an accuracy bound
for projected points, especially far outside the supporting patch. No outlier rejection or feature matching
is performed. A poor patch should be reselected, not accommodated by relaxing its tolerance.

Replies have `method: "plane_projection"` and a `plane` diagnostic with the supporting box, fitted center,
normal and maximum fit error (geometry in the base frame). Sample/depth diagnostics describe the supporting
patch. The overlay marks that patch in blue; the saved `.npz` also keeps its `support_points`. `surface_center`,
`from_tool` and `in_tool` describe the inferred point. Its freshness and calibration are those of the original
frame, so projection neither refreshes evidence nor changes the existing `requires` contract.

## What a measurement contains

| field | meaning |
|---|---|
| `id` | what `requires` refers to |
| `valid`, `reason` | `reason` says why a measurement failed: `missing_depth`, `missing_calibration`, `invalid_depth`, `insufficient_support`, `mixed_depth_surfaces`, or for plane projection: `degenerate_plane_support`, `nonplanar_support`, `grazing_plane_ray`, `plane_behind_camera` |
| `surface_center` | the median of the measured surface points, in work-frame metres: the frame plans use |
| `visible_bounds` | the 2nd and 98th percentiles of those points along each work axis |
| `from_tool` | `surface_center` minus the tool point when the picture was taken, in the work frame |
| `in_tool` | the measured surface point in the captured tool's axes, in metres; null without a valid surface and captured tool pose |
| `samples`, `valid_fraction`, `depth_spread_m` | how many pixels were sampled, how many had depth, and the depth range they spanned |
| `age_s`, `capture_t` | the picture's age when measured, and its time on the run's clock |
| `image` | the picture with the measured pixels (green, red when invalid) and the surface centre drawn on it |

The numbers describe what the camera sees, not a whole object: the top face of a block, not its centre.
Shape is the agent's knowledge. The example knows its block is a 4 x 4 x 10 cm upright box, so it places
the block's centre 5 cm below the measured top. A measurement never changes the world model; tell the
kernel about an object with `add_box` and name the measurement in its source.

For a grip check, measure the same visible feature before and after a short lift or rotation. Its `in_tool`
coordinates should stay within the procedure's measurement tolerance while `surface_center` moves. Unlike
`from_tool`, `in_tool` accounts for wrist rotation and is independent of the work frame. This is a coordinate
conversion, not tracking: a different selected surface, occlusion or a shifting box median makes the comparison
inconclusive. One point cannot detect rotation about that point; inspect separated features when rotation matters.

After release and clear withdrawal, measure the expected position and depth in a fresh frame. If revisiting
an apparently incomplete placement, remeasure from the approach view before closing the gripper: the object
may have settled in the meantime. Preserve an outcome that now meets the criteria; base recovery on a current
measured discrepancy. Freshness guards bound evidence age, but cannot verify correspondence or seating.

## Aligning measured landmarks

For planar placement, Python integrations can use `world_use.geometry.align_planar(source, target, tool,
max_error_m=...)`. Supply 3-8 corresponding XYZ landmarks in each list and the captured 4x4 tool pose,
all in the same coordinate frame. The caller chooses which visible features correspond; no object
model or feature matching is supplied. Measure the source after verifying the grasp, while the part
is held in the same way that the proposed motion assumes.

The helper fits translation and yaw without scale or reflection, then applies the same correction to
the captured tool pose. This accounts for an off-centre grasp. `tool_pose` preserves the tool's height;
approach, insertion depth and collision checking remain the procedure's decisions. Inspect above contact,
then remeasure if the part slips or the grasp changes. The inputs must span an area, not one edge.

The result includes each horizontal fit error and disagreement between relative landmark heights.
If either exceeds the caller's `max_error_m`, `valid` is false and `tool_pose` is absent (`None`). Check
correspondence, depth and tilt instead of loosening the tolerance to force a fit. A small residual can
still come from consistently wrong correspondences; it is not confidence in an object pose or proof
of a secure grasp. Keep the original measurement IDs in the plan's `requires`: fitting does not
create fresh evidence, change the world model or move the robot.

## Freshness

A run that `requires` a measurement is refused when it is submitted if the daemon does not know that
measurement (it is from before a restart, or older than the last 256 measurements) or if the measurement found
no surface (`valid` is false). Once submitted, the run is refused, or stops before its next step, holding where
the arm is, when the measurement:

- is older than its `max_age_s`, counted from when the picture was taken;
- came from a camera that has been calibrated since, even to the same numbers;
- was withdrawn, because a tracker lost the target it measured.

This check runs before rehearsal, when the run is accepted, and before every step starts moving, also with
`rehearse=false`: inside sequences, after a path computed in the background, and between a grip's opening and
its closing. A step that is already moving finishes. The check compares numbers copied when the run was
submitted, so it reads no files or pictures in the control loop, and the run keeps its copy even after the daemon
forgets the measurement. Plans with custom (plugin) steps cannot require measurements, because their own moves
would go unchecked, unless an embedded adapter explicitly audits and admits those exact types through
[`Kernel.guarded_steps`](adapters.md#guarded-custom-steps). The agent picks `max_age_s`: long enough for its
decisions, short enough for how fast the scene can change.

## Tracking

`wu mcp --vision` loads [EdgeTAM](../examples/tracking) in its own process before the arm is powered, and
adds two tools. `select_target(frame, target, point|box)` selects an object in a frame and measures it;
`observe_targets([target, ...])` measures it again in a new frame without selecting it again. Both return
the same measurements as `measure_pixels`, with `tracking` set to `tracked` or `lost`.

An agent often thinks for longer than the tracker accepts a picture's age (15 s). A selection made in such
a picture is followed into a new picture before it is measured. A target the tracker loses is dropped, and
the measurements made of it are withdrawn: a run that requires one stops before its next step. Select it
again to continue. The tracker follows up to four targets and runs one request at a time; selecting a name
again withdraws its previous measurements. A stopped provider withdraws all its targets. Each target keeps 256
measurement IDs; before forgetting an older one, it withdraws it too, including from already accepted jobs.
Nothing tracks in the background. In Python, `Tracking(client, TrackerProcess())` from
`world_use.tracking` and `world_use.vision_worker` offers the same `select` and `observe`.

## Records

Each measurement adds one `measurement` event and two files to the run: `perception/<id>.png`, the picture
with what was measured drawn on it, and `perception/<id>.npz` with the measured `points` (base frame,
metres) and their `pixels`. They are written before the reply, so a full disk fails the measurement rather
than losing it later. [Rerun](records.md#view-in-rerun) shows the picture at its capture time and the points
from when they were measured.

## Limits

Depth is simulated: physical RGB-D needs a chosen sensor, its calibration and measured timing first. One
point or box measures one visible surface; there is no object pose, segmentation or grasp planning. The
calibration's accuracy bounds every number, and `requires` checks age and calibration, not whether the
scene changed in a way no camera saw. Cameras that move with the arm are not supported.
