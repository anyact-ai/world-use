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


@dataclass(frozen=True)
class Look:
    """One sighting of a tracked region. center and bbox are native pixels; None when it is not visible."""
    view: View                  # the camera when the frame was taken, in the base frame
    hand: np.ndarray            # the tool point then, base frame, metres
    center: Sequence[float] | None = None
    bbox: Sequence[float] | None = None     # x0, y0, x1, y1

    @property
    def whole(self) -> bool:
        """The image border cuts none of the region off. A cut-off region's centre is not the region's centre."""
        if self.center is None or self.bbox is None:
            return False
        x0, y0, x1, y1 = self.bbox
        return min(x0, y0) > 0 and x1 < self.view.width - 1 and y1 < self.view.height - 1


@dataclass(frozen=True)
class Fix:
    """The point nearest to the rays of several looks."""
    point: np.ndarray           # base frame, metres
    looks: int
    baseline_m: float           # the greatest distance between two of the camera positions
    miss_m: float               # root-mean-square distance from the point to the rays
    depth_m: float              # the point's least distance along a look's optical axis


def ray(view: View, pixel) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame origin and unit direction of the ray through a native pixel."""
    origin = view.T[:3, 3]
    direction = view.unproject(pixel, 1.0) - origin
    return origin, direction / np.linalg.norm(direction)


def triangulate(looks: Sequence[Look], *, baseline_m: float = 0.03, near_m: float = 0.05) -> Fix | None:
    """Where the looks' rays meet, or None while that cannot be trusted.

    Rays from nearly one place meet anywhere along themselves, so two of the camera positions must
    be at least baseline_m apart. Looks at a region whose outline changed between them can meet
    behind a camera: the point must lie at least near_m in front of every one.
    """
    rays = [ray(look.view, look.center) for look in looks]
    origins = [origin for origin, _ in rays]
    baseline = max((float(np.linalg.norm(a - b)) for a in origins for b in origins), default=0.0)
    if baseline < baseline_m:
        return None
    normal = [np.eye(3) - np.outer(d, d) for _, d in rays]
    point = np.linalg.solve(sum(normal), sum(n @ o for n, o in zip(normal, origins, strict=True)))
    depth = min(float(look.view.project(point)[1][0]) for look in looks)
    if depth < near_m:
        return None
    miss = np.sqrt(np.mean([np.sum((n @ (point - o)) ** 2) for n, o in zip(normal, origins, strict=True)]))
    return Fix(point, len(looks), baseline, float(miss), depth)


def locate(move: Callable[[np.ndarray], Outcome], look: Callable[[], Look], *, camera: str, arrive_m: float,
           assumed_z: float | None = None, max_moves: int = 5, baseline_m: float = 0.03, near_m: float = 0.05,
           cut_off_drop_m: float = 0.25) -> Outcome:
    """Move the hand level until it is over a region its camera tracks, and report where the region is.

    look() returns the region as the camera sees it now. move(xy) takes the hand level to base-frame
    x, y through the caller's own executor, which counts and checks it, and returns its Outcome.
    The search ends once the hand is within arrive_m of the point the looks fix, so the last look is
    taken from over the region. A done Outcome carries x, y, z in the base frame, the number of looks
    and moves, and either pixel (where that point is in the image now) with extent_m (the region's box
    at that depth, along the image's x and y) or region, saying why those are missing.

    One look gives only a direction. The first move heads for where it meets assumed_z when the
    caller knows a height to assume, or cut_off_drop_m below the camera for a region the image cuts
    off; a whole region with no height to assume needs only a second look from a sidestep away.
    Only looks at the whole region place it once one has been had. A move that ends short of its
    goal still brings a new look; a second one ends the search with that move's Outcome.
    """
    sidestep = 1.5 * baseline_m     # leaves the looks a baseline apart even when the arm arrives short
    looks, moves, short, fix, whole = [], 0, 0, None, False
    while True:
        seen = look()
        visible = seen.center is not None
        if seen.whole and not whole:
            looks, fix, whole = [], None, True
        if seen.whole or (visible and not whole):
            looks.append(seen)
            fix = triangulate(looks, baseline_m=baseline_m, near_m=near_m)
        if fix is not None:
            # A region the tracker drops after its position is fixed keeps that position.
            if not visible or moves >= max_moves or np.linalg.norm(fix.point[:2] - seen.hand[:2]) <= arrive_m:
                x, y, z = map(float, fix.point)
                found: dict = dict(x=x, y=y, z=z, looks=fix.looks, moves=moves)
                box = seen.bbox if seen.whole else None
                if not visible:
                    found["region"] = "lost after these looks"
                else:
                    (pixel,), (depth,) = seen.view.project(fix.point)
                    found["pixel"] = [round(float(pixel[0])), round(float(pixel[1]))]
                    if box is not None:
                        found["extent_m"] = [float((box[2] - box[0]) * depth / abs(seen.view.fx)),
                                             float((box[3] - box[1]) * depth / abs(seen.view.fy))]
                    elif not whole:
                        found["region"] = "cut off by the image border in every look"
                return Outcome("done", "locate", "the region's position, from looks at it from different places",
                               found)
            goal = fix.point[:2]
        elif not visible:
            return Outcome("surprise", "locate", f"the selected region is not visible in the {camera} image",
                           dict(moves=moves),
                           hint=f"hover with the object in the {camera} view and select it there; "
                                "later stages did not run")
        elif moves >= max_moves:
            return Outcome("stopped", "locate", f"the region could not be placed from the {camera} views",
                           dict(moves=moves))
        else:
            origin, direction = ray(looks[-1].view, looks[-1].center)
            drop = origin[2] - assumed_z if assumed_z is not None else None if whole else cut_off_drop_m
            toward, step = direction[:2], sidestep
            if drop is not None:
                toward = (origin - direction * (drop / direction[2]))[:2] - seen.hand[:2]
                step = max(float(np.linalg.norm(toward)), sidestep)
            distance = float(np.linalg.norm(toward))
            goal = seen.hand[:2] + (toward / distance if distance > 1e-6 else np.array([1.0, 0.0])) * step
        moved = move(goal)
        moves += 1
        if not moved.ok:
            short += 1
            if short > 1 or moved.status != "surprise":
                return moved
