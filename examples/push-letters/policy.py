"""Camera-only pose estimates and a bounded quasi-static pushing policy.

The program knows the manufactured glyphs and goal poses. It never reads live
object poses, velocities or contacts. Geometry and colour segmentation are
specific to this deliberately simple workcell.
"""
from __future__ import annotations

import numpy as np
from scene import RADIUS, bars, boundary, rotation, template

from world_use.perception import measure


def angle_difference(a, b):
    return float((a-b+np.pi) % (2*np.pi)-np.pi)


def color_mask(image, colors, index):
    rgb = np.asarray(image, dtype=float) / 255
    chroma = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), .01)
    references = np.array(colors)
    references /= references.sum(axis=1, keepdims=True)
    distance = np.sum((chroma[:, :, None, :] - references[None, None, :, :])**2, axis=3)
    saturation = (rgb.max(axis=2)-rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), .01)
    return ((distance.argmin(axis=2) == index) & (distance[:, :, index] < .018)
            & (saturation > .38) & (rgb.max(axis=2) > .07))


def edge_distance(points, strokes):
    values = []
    for x, y, hx, hy, yaw in strokes:
        q = np.abs((points-[x, y]) @ rotation(yaw)) - [hx, hy]
        values.append(np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0))
    # Signed distance to the union's boundary is sufficient for exterior samples;
    # internal stroke seams never attract the silhouette fit.
    return np.min(values, axis=0)


def estimate(frame, letter, colors, index):
    mask = color_mask(frame.image, colors, index)
    measurement = measure(frame, mask=mask, target=letter)
    if not measurement.valid:
        raise ValueError(f"{letter} is not measurable: {measurement.reason}")
    points = measurement.points[:, :2]
    reference = template(letter)
    centre, reference_centre = points.mean(axis=0), reference.mean(axis=0)
    covariance = np.cov(points.T)
    ref_covariance = np.cov(reference.T)
    _, axes = np.linalg.eigh(covariance)
    _, ref_axes = np.linalg.eigh(ref_covariance)
    initial = np.arctan2(axes[1, -1], axes[0, -1]) - np.arctan2(ref_axes[1, -1], ref_axes[0, -1])
    strokes = bars(letter)
    candidates = []
    for yaw in (initial, initial+np.pi):
        origin = centre - reference_centre @ rotation(yaw).T
        local = (points-origin) @ rotation(yaw)
        candidates.append((np.mean(np.maximum(edge_distance(local, strokes), 0)**2), yaw, origin))
    _, yaw, origin = min(candidates, key=lambda entry: entry[0])
    # Refine the pose against the visible RGB silhouette at the measured top plane.
    eroded = mask.copy()
    eroded[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    yy, xx = np.where(mask & ~eroded)
    pixels = np.column_stack([xx, yy]) + .5
    if len(pixels) < 20:
        raise ValueError(f"{letter}: insufficient visible boundary")
    # The overhead camera is perpendicular to the task plane. Optical depth of
    # the eroded surface avoids boundary pixels that see the bench or side wall.
    depth = np.median(frame.depth[measurement.pixels[:, 1], measurement.pixels[:, 0]])
    edge = frame.view.unproject(pixels, np.full(len(pixels), depth))[:, :2]

    def score(pose):
        residual = edge_distance((edge-pose[:2]) @ rotation(pose[2]), strokes)
        return np.mean(np.minimum(residual**2, .004**2))

    pose = np.array([*origin, yaw])
    for scale in (1., .5, .2, .08):
        for _ in range(12):
            choices = [pose]
            for axis, delta in enumerate((.001*scale, .001*scale, np.radians(2)*scale)):
                for sign in (-1, 1):
                    candidate = pose.copy()
                    candidate[axis] += sign*delta
                    choices.append(candidate)
            best = min(choices, key=score)
            if best is pose:
                break
            pose = best
    fit = float(np.sqrt(score(pose)))
    if fit > .0015:
        raise ValueError(f"{letter}: silhouette fit {fit*1000:.2f} mm exceeds 1.5 mm")
    return dict(xy=pose[:2], yaw=float(pose[2]), top=float(np.median(measurement.points[:, 2])),
                fit_m=fit, measurement=measurement)


def next_push(letter, pose, target_xy, target_yaw):
    """Choose an accessible point push using an ellipsoidal planar friction model."""
    xy, yaw = pose["xy"], pose["yaw"]
    error = np.asarray(target_xy)-xy
    angle = angle_difference(target_yaw, yaw)
    if letter == "O":
        angle = (angle+np.pi/2) % np.pi - np.pi/2
    if np.linalg.norm(error) <= .004 and abs(angle) <= np.radians(4):
        return None
    reference = template(letter)
    local_centre = reference.mean(axis=0)
    centre = xy + local_centre @ rotation(yaw).T
    target_centre = np.asarray(target_xy) + local_centre @ rotation(target_yaw).T
    error = target_centre-centre
    radius2 = np.mean(np.sum((reference-local_centre)**2, axis=1))
    points, normals = boundary(letter)
    points, normals = points @ rotation(yaw).T + xy, normals @ rotation(yaw).T
    best = None
    for point, normal in zip(points, normals, strict=True):
        for shear in (-.25, 0., .25):
            tangent = np.array([-normal[1], normal[0]])
            direction = -normal + shear*tangent
            direction /= np.linalg.norm(direction)
            start = point + normal*(RADIUS+.006)
            # Reject approaches where the pin's radius intersects another stroke.
            approach = start[None, :] + np.linspace(0, .035, 12)[:, None]*normal
            if np.any(edge_distance((approach-xy) @ rotation(yaw), bars(letter)) < RADIUS+.002):
                continue
            lever = point-centre
            torque = lever[0]*direction[1]-lever[1]*direction[0]
            response = np.array([*direction, torque/radius2])
            # Map pin displacement to body twist at the contact.
            contact_speed = 1 + torque**2/radius2
            response /= contact_speed
            weights = np.array([1, 1, .045])
            desired = np.array([*error, angle])
            weighted = response*weights
            distance = np.clip(np.dot(desired*weights, weighted)/np.dot(weighted, weighted), .002, .045)
            predicted = desired-response*distance
            score = float(np.sum((predicted*weights)**2))
            # Prefer aligned normal pushes where two candidates have similar progress.
            score += abs(shear)*1e-6
            if best is None or score < best[0]:
                best = (score, dict(start=start, end=start+direction*(.006+distance),
                                   contact=point, direction=direction, stroke_m=float(distance),
                                   predicted_twist=(response*distance).tolist()))
    if best is None:
        raise ValueError(f"{letter}: no unobstructed pushing contact")
    return best[1]
