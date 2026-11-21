# Measuring from pictures

world-use turns pixels an agent selects into positions a plan can use. The agent captures a frame, measures a
point or box in it, and plans with the result. `requires` keeps the next step from starting on a measurement
that is too old, withdrawn, or from a camera calibrated since. After moving, the agent measures again to check
the result. The [perception example](../examples/perception) shows the whole loop.

The [shape-sorting experiment](shape-sorting.md) uses plane projection and landmark alignment; the
[lunchbox task](../examples/robocasa-lunchbox) adds EdgeTAM tracking and grasp checks.

Depth comes only from simulated (MuJoCo) cameras. Other cameras give pictures without depth, and measuring them
reports `missing_depth`.

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
timeout returns without a picture; wait with `job`, never resubmit it. If the camera fails, the reply still
carries the job and its outcome, with `camera_error`.

To try it, start the block workcell (`wu up --workcell block`): its block's top face is near pixel (374, 112)
of the `top` camera, and a measurement there gives a `surface_center` of about [0.34, 0.03, 0.25], the middle
of that face in the work frame.

`camera_frame` returns the picture at native resolution with nothing drawn on it, and a frame id. Pixel
coordinates refer to that picture: `look` scales and annotates its pictures, so do not take coordinates
from it. `inspect_image(frame, crop=[left, top, right, bottom])` shows part of a frame again, enlarged,
with the matrix that maps its pixels back to the frame's. The daemon keeps the last 8 frames (64 MiB at most).

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

At a hole or a silhouette edge, the selected pixel may see the background behind the edge. If the feature lies
on a flat surface you can see, select a box wholly inside that surface and pass it as `plane`:

```python
edge = robot.measure(frame, point=[420, 210],
                     plane={"box": [360, 180, 390, 205], "max_error_m": .001})
```

MCP `measure_pixels` takes the same option. It fits a plane to the box's depth samples and intersects the
point's camera ray with that plane, so tilted surfaces and cameras work and no height is needed. The result is a
point inferred on the plane, not a depth reading at the pixel: you decide that the feature lies on that plane,
and a feature on a different surface still gives a valid, wrong result.

The box needs depth samples spread over an area. The measurement is invalid when the samples do not span an
area (`degenerate_plane_support`), when any sample lies farther than `max_error_m` from the fitted plane
(`nonplanar_support`), when the ray nearly grazes the plane (`grazing_plane_ray`) or when the plane lies behind
the camera (`plane_behind_camera`). There is no outlier rejection: when a patch fails, select a cleaner, broader
one rather than raising the tolerance. The fit error is not an accuracy bound for the projected point,
especially far from the patch.

Replies have `method: "plane_projection"` and a `plane` entry with the box, the fitted `center_m` and `normal`
(base frame) and the largest fit error, `fit_error_m`. The sample counts describe the patch, which the overlay
draws in blue; the saved `.npz` keeps its `support_points`. `surface_center`, `from_tool` and `in_tool` describe
the projected point. Its age and calibration are those of the frame, as for any other measurement.

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

To check a grip, measure the same visible feature before and after a short lift or turn. If the object moved
with the hand, its `in_tool` coordinates stay the same, within your measurement tolerance, while
`surface_center` changes. `in_tool` follows the wrist's rotation and does not depend on the work frame, unlike
`from_tool`. The comparison holds only for the same surface: occlusion, a different selection or a box whose
median shifts make it inconclusive. One point cannot show rotation about itself; measure separated features
when rotation matters.

To check a placement, withdraw clear of the object and measure its position and depth in a fresh frame. Before
regrasping a placement that looked incomplete, measure again from the approach view: the object may have
settled into place since. Keep a result that now meets the criteria, and correct only a discrepancy you have
just measured.

## Aligning measured landmarks

For planar placement in Python, `world_use.geometry.align_planar(source, target, tool, max_error_m=...)` finds
the tool pose that moves a held part's landmarks onto their targets. `source` and `target` are lists of at least
three corresponding XYZ points, and `tool` is the 4x4 tool pose captured with the source, all in one coordinate
frame. You choose which visible features correspond; there is no object model or feature matching. Measure the
source after checking the grasp, while the part is held as the coming motion assumes. The landmarks must span
an area, not lie along one edge.

It fits a translation and a yaw, without scale or reflection, and applies them to the captured tool pose, which
accounts for an off-centre grasp. The returned `tool_pose` keeps the tool's height: approach, insertion depth
and collision checks remain the procedure's decisions. Inspect from above before contact, and measure again if
the part slips or the grasp changes.

The result reports each landmark's horizontal error (`errors_m`) and how far the landmarks' relative heights
disagree (`height_error_m`). If either exceeds `max_error_m`, `valid` is false and `tool_pose` is `None`; check
correspondence, depth and tilt rather than loosening the tolerance. A small error can still come from
consistently wrong correspondences, so it is no proof of the part's pose or a secure grasp. Fitting moves
nothing and creates no new measurement: keep the source measurement IDs in the plan's `requires`.

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
its closing. A step that is already moving finishes. The run copies what the check needs when it is submitted,
so the control loop reads no files or pictures and the check still works after the daemon forgets the
measurement. Plans with custom (plugin) steps cannot require measurements, because their own moves would go
unchecked, unless an embedded adapter admits those exact types through
[`Kernel.guarded_steps`](adapters.md#guarded-custom-steps). The agent picks `max_age_s`: long enough for its
decisions, short enough for how fast the scene can change.

## Tracking

`wu mcp --vision` loads [EdgeTAM](../examples/tracking) in its own process before the arm is powered, and
adds two tools. `select_target(frame, target, point|box)` selects an object in a frame, names it `target` and
measures it; `observe_targets([target, ...])` measures named targets again in one new frame per camera, without
selecting them again. Both return the same measurements as `measure_pixels`, with `tracking` set to `tracked` or
`lost`. In Python, `Tracking(client, TrackerProcess())` from `world_use.tracking` and `world_use.vision_worker`
offers the same `select` and `observe`.

The tracker accepts frames up to 15 s old, and an agent often thinks for longer. A selection made in an older
frame is followed into a new frame and measured there. The tracker follows up to four targets, handles one
request at a time, and runs only when one of these calls is made: nothing tracks in the background.

When the tracker loses a target, the target is retired and every measurement made of it is withdrawn, so a run
that requires one stops before its next step. Observing that name again returns the same loss until you select
it again; Python callers find lost targets in `Tracking.lost`, beside the active `Tracking.targets`. Selecting a
name again also withdraws its earlier measurements, and if the tracker process stops, all its targets are lost.
Each target remembers its latest 256 measurement IDs and withdraws older ones as they drop out, which also stops
accepted jobs that require them, so a later loss never leaves a job relying on a measurement nobody can withdraw.

The tracker recognises a frame by its `Frame.id`: a camera source must keep a capture's `id` and `timestamp`
when it reads that capture again, and give every new capture a new `id`, even when the pixels are unchanged.

### Tracking without depth

For an RGB camera without depth, select with `space="image"`. Both tools then return the mask's geometry in
pixels instead of a measurement: `bbox` (native pixels, right and bottom exclusive), `center` and `area_px`,
with the `camera`, `frame` id, `captured_at` (on the host's monotonic clock) and `age_s`. From the second
observation on, `motion` compares the mask with the previous frame's: `center_delta_px`, `area_ratio` and
`mask_iou` (intersection over union), with `from_frame`. These are changes in the image, not object motion in
metres, and a moving camera or occlusion changes them too (`camera_motion_compensated: false`). A low
`mask_iou` is a reason to look at the picture again, not proof that the tracker switched objects.

`dominant_region` describes the largest connected part of the mask: its `bbox`, `center` and `area_px`, the area
outside it (`other_area_px`) and the number of `components`. A few stray pixels can stretch the whole mask's
`bbox` across unrelated objects; the region's box leaves them out. The raw mask is unchanged, and the largest
region is not necessarily the intended object. `dominant_region` also gives the narrowest box around the region:
`width_px`, `length_px` and `axis_deg`, the direction of its length from image x toward image y.

Image results have no measurement `id` and cannot satisfy `run(requires=...)`. A tracked mask does not show
that it is the right object, or that the object is held or supported: check the picture. After a loss, select
the target again yourself, and do not take a new mask as proof that it is the same object. The default
`space="world"` returns metric measurements and needs RGB-D.

## Positions from a camera on the hand

A camera fixed to the hand measures no depth, but the tool pose gives its pose in every frame, so looks from two
places fix a point. Describe the camera once as a `View` in the tool frame, its mount; `mount.on(tool_pose)` is
its base-frame view for a measured tool pose. For an image published mirrored, give a negative `fx` (or, for
flipped rows, a negative `fy`) and mirror the principal point to match (`width - 1 - cx` or
`height - 1 - cy`), so pixels and points convert both ways without turning the image.

`world_use.parallax` places a tracked region from such looks:

- `Look(view, hand, center, bbox)` is one sighting: the camera's base-frame `View` and the tool point (base
  frame, metres) when the frame was taken, and the region's centre and box in native pixels, `None` when it is
  not visible. The box ends one past its last pixel, as the tracker reports it.
- `triangulate(looks)` returns the base-frame point nearest the looks' rays, or `None` unless two of the camera
  positions are at least 3 cm apart and the point lies at least 5 cm in front of every camera.
- `locate(move, look, arrive_m=...)` moves the hand level, looking after each move, until it is within
  `arrive_m` of being over the point the looks fix. `look()` returns a `Look`; `move(xy)` takes the hand to
  base-frame x, y through your own executor and returns its `Outcome`. Like the
  [grasp and place procedures](adapters.md#grasp-and-place-procedures), it sends no command itself, so it works
  with the kernel client or another controller.

`locate` returns a `done` `Outcome` with `x`, `y`, `z` in the base frame, the number of `looks` and `moves`,
`pixel`, where the point is in the latest image while the region is visible, and `extent_m`, the region's box
size at that depth, when the last look shows the whole region. `region` notes a region lost after the looks or
never seen whole. A move that fails ends the search with that move's outcome. A region the camera does not
show before it is placed is a `surprise`, and one still not placed after five moves is `stopped`.

Once a look has shown the whole region, only looks that show all of it count: the centre of a region the image
border cuts off is not the region's centre. The result is the position of the region the tracker follows,
tracker errors included. It has no measurement `id`, and it goes stale when the object moves, not with age.
Pass a position found earlier as `known`: the first move heads straight over it, and one look that shows the
region within `arrive_m` of it confirms it; a region that has moved is placed afresh. Until the looks fix a
point, each move heads for where the latest ray meets `assumed_z`, a base-frame height, when you give one;
otherwise the hand sidesteps for a second look.

`footprint(view, region, depth_m)` converts a `dominant_region` seen face-on, at a known depth along the view's
optical axis, into metres: `across_m` and `along_m`, and `across`, the base-frame unit direction across the
region's narrow side, which is the direction a gripper's fingers close along to take it by that side. A region
that covers only part of an object gives that part's narrow side.

In the simulator, `body.capture(mount, mounted=True)` renders such a camera from its mount in the tool frame and
returns the image, depth, tool pose and timestamp of one physics state; `mount.on(tool)` is the view to look
with. A workcell camera can be one: `mount = "tool"` with `eye`, `look_at` and `up` in the tool frame. Each of
its frames carries the tool pose and base-frame view of the state it shows, and `calibrate` refuses it: its
mount is its calibration. The [hand tools](hand-tools.md) look through it to locate and grasp. [`tests/test_parallax.py`](../tests/test_parallax.py) locates a block this way to within a centimetre and
picks it up there, with the kernel told nothing about the block. The [LIBERO runner](../examples/libero) uses the
same functions for a simulated Panda's wrist camera.

## Records

Each measurement adds a `measurement` event and two files to the run's
[flight record](records.md#what-a-record-holds): `perception/<id>.png`, the picture with what was measured drawn
on it, and `perception/<id>.npz` with the measured `points` (base frame, metres) and their `pixels`. They are
written before the reply, so a full disk fails the measurement rather than losing it later.
[Rerun](records.md#view-in-rerun) shows the picture at its capture time and the points from when they were
measured.

## Limits

- Physical RGB-D needs a sensor integration with measured calibration and timing; depth comes only from
  simulated cameras.
- A point, box or mask measures a visible surface. EdgeTAM can supply the mask; object pose estimation, feature
  correspondence and grasp selection are up to the task.
- Calibration accuracy limits every measurement. `requires` checks age, calibration and withdrawal; it cannot
  notice a scene change that nobody observed.
- Only a simulated workcell camera can be mounted on the tool: a physical frame's time cannot yet be paired with
  the tool pose. The kitchen examples' custom
  [wrist camera](../examples/robocasa-lunchbox/sensor_views.py) captures RGB-D, camera pose and tool pose from
  the same simulation state; its calibration identity covers the mount and intrinsics, and each frame carries
  the current camera transform. There is no calibration procedure for a physical wrist camera, and a position
  from parallax is only as good as the mount it is given.
