"""Where a region is, from looks at it by one calibrated camera that moves with the hand.

Such a camera measures no depth, but its pose is known for every frame. One look at a region fixes a
ray; looks from places far enough apart fix a point. Nothing here recognises objects or commands a
robot: `locate` moves the hand only through the callable it is given.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .behaviors import Outcome
from .cameras import View

# Rays from nearly one place meet anywhere along themselves: two of the camera positions must be this far apart.
BASELINE_M = 0.03
# Looks at a region whose outline changed between them can meet behind a camera. Nearer than this a camera
# on a hand sees only the hand.
NEAR_M = 0.05
MAX_MOVES = 5
# A region the image cuts off is assumed this far below the camera for the first move toward it.
CUT_OFF_DROP_M = 0.25


@dataclass(frozen=True)
class Look:
    """One sighting of a tracked region. center and bbox are native pixels; None when it is not visible.
    Without its bbox a region cannot be known to be whole."""
    view: View                  # the camera when the frame was taken, in the base frame
    hand: np.ndarray            # the tool point then, base frame, metres
    center: Sequence[float] | None = None
    bbox: Sequence[float] | None = None     # x0, y0, x1, y1; x1 and y1 are one past the last pixel

    @property
    def whole(self) -> bool:
        """The image border cuts none of the region off. A cut-off region's centre is not the region's centre."""
        if self.center is None or self.bbox is None:
            return False
        x0, y0, x1, y1 = self.bbox
        return min(x0, y0) > 0 and x1 < self.view.width and y1 < self.view.height


def ray(view: View, pixel) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame origin and unit direction of the ray through a native pixel."""
    origin = view.T[:3, 3]
    direction = view.unproject(pixel, 1.0) - origin
    return origin, direction / np.linalg.norm(direction)


def triangulate(looks: Sequence[Look]) -> np.ndarray | None:
    """The base-frame point nearest the looks' rays, or None while they cannot be trusted to fix one:
    the camera positions are less than BASELINE_M apart, the rays are parallel, or the point is not
    NEAR_M in front of every camera."""
    rays = [ray(look.view, look.center) for look in looks]
    origins = [origin for origin, _ in rays]
    if max((float(np.linalg.norm(a - b)) for a in origins for b in origins), default=0.0) < BASELINE_M:
        return None
    normal = [np.eye(3) - np.outer(d, d) for _, d in rays]
    total = sum(normal)
    if np.linalg.eigvalsh(total)[0] < 1e-9:
        return None
    point = np.linalg.solve(total, sum(n @ o for n, o in zip(normal, origins, strict=True)))
    if min(float(look.view.project(point)[1][0]) for look in looks) < NEAR_M:
        return None
    return point


def footprint(view: View, region: dict, depth_m: float) -> dict | None:
    """The size of a region seen face-on at a known depth: across its narrow side and along its long one.

    region is a tracker's dominant_region; depth_m is the surface's distance along this view's optical
    axis. `across` is the base-frame unit direction across the narrow side, the one a gripper's fingers
    close along to take the region by its narrow side. None for an empty region.
    """
    if not region.get("center"):
        return None
    centre, turn = np.asarray(region["center"], float), np.radians(region["axis_deg"])
    long = np.array([np.cos(turn), np.sin(turn)])

    def span(direction, pixels):
        ends = view.unproject([centre - direction * pixels / 2, centre + direction * pixels / 2], [depth_m] * 2)
        return ends[1] - ends[0]

    across, along = span(np.array([-long[1], long[0]]), region["width_px"]), span(long, region["length_px"])
    width = float(np.linalg.norm(across))
    return dict(across_m=width, along_m=float(np.linalg.norm(along)), across=(across / width).tolist())


def image_directions(view: View, pixel, depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The level base-frame unit directions that right and down in this view's image point along, at a pixel
    seen at depth_m: an offset toward something seen there is its distance times these."""
    pixel = np.asarray(pixel, float)
    here, right, down = view.unproject([pixel, pixel + [1, 0], pixel + [0, 1]], [depth_m] * 3)[:, :2]
    return (right - here) / np.linalg.norm(right - here), (down - here) / np.linalg.norm(down - here)


def locate(move: Callable[[np.ndarray], Outcome], look: Callable[[], Look], *, arrive_m: float,
           assumed_z: float | None = None, known=None) -> Outcome:
    """Move the hand level until it is over a region its camera tracks, and report where the region is.

    look() returns the region as the camera sees it now. move(xy) takes the hand level to base-frame
    x, y through the caller's own executor, which counts and checks it, and returns its Outcome; a move
    that fails ends the search with that Outcome. The search ends once the hand is within arrive_m of
    the point the looks fix, so the last look is taken from over the region, or after MAX_MOVES moves
    with the position the looks fix from wherever the arm got to. A done Outcome carries x, y, z in
    the base frame, the number of looks and moves, pixel (where that point is in the latest image,
    while the region is visible), extent_m (the region's box at that depth along the image's x and
    y, when the last look shows the whole region) and region, a note when the region was lost or
    never seen whole.

    One look gives only a direction. The first move heads for where it meets assumed_z when the
    caller knows a height below the camera to assume, or CUT_OFF_DROP_M below the camera for a region
    the image cuts off; a whole region with no height to assume needs only a second look from a
    sidestep away. Only looks at the whole region place it once one has been had.

    known is a position found for this region earlier. The first move then heads straight over it,
    and one look from there that shows the region within arrive_m of it confirms it without a
    second. A region that has moved is not confirmed and is placed afresh.
    """
    sidestep = 1.5 * BASELINE_M     # leaves the looks a baseline apart even when the arm arrives short
    known = None if known is None else np.asarray(known, float)
    looks, moves, fix, whole = [], 0, None, False
    while True:
        seen = look()
        visible = seen.center is not None
        if seen.whole and not whole:
            looks, fix, whole = [], None, True
        if seen.whole or (visible and not whole):
            looks.append(seen)
            fix = triangulate(looks)
            if fix is None and known is not None and np.linalg.norm(known[:2] - seen.hand[:2]) <= arrive_m:
                (pixel,), (depth,) = seen.view.project(known)
                if depth >= NEAR_M and np.linalg.norm(pixel - seen.center) * depth / abs(seen.view.fx) <= arrive_m:
                    fix = known
        if fix is not None:
            # A region the tracker drops after its position is fixed keeps that position.
            if not visible or moves >= MAX_MOVES or np.linalg.norm(fix[:2] - seen.hand[:2]) <= arrive_m:
                x, y, z = map(float, fix)
                found: dict = dict(x=x, y=y, z=z, looks=len(looks), moves=moves)
                box = seen.bbox if seen.whole else None
                if not visible:
                    found["region"] = "lost after these looks"
                else:
                    (pixel,), (depth,) = seen.view.project(fix)
                    found["pixel"] = [round(float(pixel[0])), round(float(pixel[1]))]
                    if box is not None:
                        found["extent_m"] = [float((box[2] - box[0]) * depth / abs(seen.view.fx)),
                                             float((box[3] - box[1]) * depth / abs(seen.view.fy))]
                    elif not whole:
                        found["region"] = "cut off by the image border in every look"
                return Outcome("done", "locate", "the region's position, from looks at it from different places",
                               found)
            goal = fix[:2]
        elif not visible:
            return Outcome("surprise", "locate", "the selected region is not visible to the hand's camera",
                           dict(moves=moves),
                           hint="hover with the object in that camera's view and select it there; "
                                "later stages did not run")
        elif moves >= MAX_MOVES:
            return Outcome("stopped", "locate", "the region could not be placed from these looks",
                           dict(moves=moves))
        elif known is not None and not moves and np.linalg.norm(known[:2] - seen.hand[:2]) > arrive_m:
            goal = known[:2]
        else:
            origin, direction = ray(looks[-1].view, looks[-1].center)
            drop = origin[2] - assumed_z if assumed_z is not None else None if whole else CUT_OFF_DROP_M
            toward, step = direction[:2], sidestep
            if drop is not None and drop > 0 and direction[2] < -1e-6:
                toward = (origin - direction * (drop / direction[2]))[:2] - seen.hand[:2]
                step = max(float(np.linalg.norm(toward)), sidestep)
            distance = float(np.linalg.norm(toward))
            goal = seen.hand[:2] + (toward / distance if distance > 1e-6 else np.array([1.0, 0.0])) * step
        moved = move(goal)
        moves += 1
        if not moved.ok:
            return moved
