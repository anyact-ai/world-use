"""Small geometry and outcome calculations shared by robot procedures and tool adapters.

Inputs are measured support and explicit task assumptions. None of these functions
commands a robot, looks up simulator truth, or certifies object identity.
"""
from __future__ import annotations

import math
from copy import deepcopy
from uuid import uuid4

import numpy as np

from .validation import number, vector

EFFECTS = {
    "lift": {
        "required": ["before", "min_up_m", "max_error_m", "max_age_s"],
        "description": "Object displacement follows synchronized measured tool displacement.",
    },
    "placement": {
        "required": ["before", "target_m", "position_tolerance_m", "support_z_m", "height_tolerance_m",
                     "min_clearance_m", "min_aperture_mm", "max_age_s"],
        "description": "Known upright object is at the destination/support, with open, withdrawn gripper.",
    },
}


def positive(value, name):
    number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def upright_box(points, size):
    """Known dimensions, upright and frame-aligned, with a substantially visible top face."""
    vector(size, "size_m", 3)
    size = np.asarray(size, float)
    if (size <= 0).any():
        raise ValueError("size_m must be positive")
    if len(points) < 16:
        return None, dict(reason="insufficient_support")
    top = np.quantile(points[:, 2], .95)
    surface = points[np.abs(points[:, 2] - top) < .003]
    if len(surface) < 16:
        return None, dict(reason="insufficient_top_support", top_samples=len(surface))
    lo, hi = np.quantile(surface[:, :2], [.02, .98], axis=0)
    diagnostics = dict(top_samples=len(surface), visible_extent_m=(hi - lo).tolist())
    if np.any(hi - lo < size[:2] * .65) or np.any(hi - lo > size[:2] * 1.15):
        return None, dict(diagnostics, reason="extent_disagrees_with_shape")
    return np.array([*(lo + hi) / 2, top - size[2] / 2]), diagnostics


def fit(measurement, receipt, world, *, kind="known_box", frame="work", size_m=None, max_residual_m=.003):
    """Fit bounded visible support; results are always expressed in base metres."""
    if kind not in {"known_box", "plane", "axis"}:
        raise ValueError("kind must be known_box, plane or axis")
    if kind == "known_box":
        if size_m is None:
            raise ValueError("known_box requires size_m")
        vector(size_m, "size_m", 3)
        if any(v <= 0 for v in size_m):
            raise ValueError("size_m must be positive")
    elif size_m is not None:
        raise ValueError("size_m is only an assumption for known_box")
    positive(max_residual_m, "max_residual_m")
    transform = world.frame(frame).T.copy()
    out: dict = dict(id=uuid4().hex, schema_version=1, kind=kind, valid=False, reason=measurement.reason,
               coordinates="base", units="m", evidence=[receipt["id"]], target=receipt["target"],
               timestamp=receipt["timestamp"], session=receipt["session"], camera=receipt["camera"],
               calibration=receipt["calibration"], tool=receipt.get("tool"),
               aperture_mm=receipt.get("aperture_mm"), provenance="derived", components={},
               frame_assumption=dict(name=frame, transform=transform.tolist()), assumptions={}, diagnostics={})
    if kind == "known_box":
        assert size_m is not None
        out["assumptions"] = dict(size_m=list(size_m), upright=True, aligned_to=frame,
                                  substantially_visible_top=True)
    if not measurement.valid:
        return out
    points = (measurement.points - transform[:3, 3]) @ transform[:3, :3]
    if len(points) < 16:
        return dict(out, reason="insufficient_support")
    directions = {}
    if kind == "known_box":
        center, diagnostics = upright_box(points, size_m)
        out["diagnostics"] = diagnostics
        if center is None:
            return dict(out, reason=diagnostics["reason"])
    else:
        center = np.mean(points, axis=0)
        _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
        out["diagnostics"] = dict(singular_values_m=singular.tolist())
        if kind == "plane":
            residual = float(np.sqrt(np.mean(((points - center) @ axes[-1]) ** 2)))
            out["diagnostics"]["residual_m"] = residual
            if singular[1] < .001 or residual > max_residual_m:
                return dict(out, reason="degenerate_fit" if singular[1] < .001 else "residual_exceeded")
            directions["normal"] = axes[-1]
        else:
            if singular[0] < .001 or singular[0] < 2 * singular[1]:
                return dict(out, reason="ambiguous_axis")
            directions["axis"] = axes[0]
    out["components"]["center"] = dict(kind="position", value=world.to_base(frame, center).tolist())
    for name, direction in directions.items():
        out["components"][name] = dict(kind="direction", value=(transform[:3, :3] @ direction).tolist(),
                                       unsigned=True)
    return dict(out, valid=True, reason=None)


def lift_effect(before, after, tool_delta, *, min_up_m=.035, max_error_m=.015):
    if before is None or after is None or tool_delta is None:
        return dict(status="unknown", reason="missing_geometry_or_synchronized_tool", measured={})
    moved = np.asarray(after) - before
    error = float(np.linalg.norm(moved - tool_delta))
    passed = moved[2] > min_up_m and error < max_error_m
    return dict(status="pass" if passed else "fail", reason=None,
                measured=dict(object_delta_m=moved.tolist(), tool_delta_m=np.asarray(tool_delta).tolist(),
                              displacement_error_m=error))


def placement_effect(center, tool, aperture_mm, *, target_m, position_tolerance_m,
                     support_center_z_m, height_tolerance_m, min_clearance_m, min_aperture_mm):
    if center is None or tool is None or aperture_mm is None:
        return dict(status="unknown", reason="missing_geometry_or_synchronized_feedback", measured={})
    distance = float(np.linalg.norm(np.asarray(center) - target_m))
    height_error = float(abs(center[2] - support_center_z_m))
    clearance = float(tool[2] - center[2])
    passed = (distance < position_tolerance_m and height_error < height_tolerance_m
              and clearance > min_clearance_m and aperture_mm >= min_aperture_mm)
    return dict(status="pass" if passed else "fail", reason=None,
                measured=dict(destination_error_m=distance, height_error_m=height_error,
                              clearance_m=clearance, aperture_mm=float(aperture_mm)))


def compile_effects(specs, world, geometry):
    """Freeze criteria and baseline geometry before submission; reject unknown/ignored parameters."""
    if not isinstance(specs, list) or len(specs) > 8:
        raise ValueError("effects must contain at most eight specifications")
    result = []
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("kind") not in EFFECTS:
            raise ValueError(f"effect kind must be one of {list(EFFECTS)}")
        kind = spec["kind"]
        required = set(EFFECTS[kind]["required"])
        if spec.keys() - required - {"kind", "frame"} or required - spec.keys():
            raise ValueError(f"{kind} requires {sorted(required)}; only frame is optional")
        for name in required - {"before", "target_m", "support_z_m"}:
            positive(spec[name], name)
        frame = spec.get("frame", "work")
        compiled = dict(spec=deepcopy(spec), frame=frame, transform=world.frame(frame).T.tolist())
        before = geometry(spec["before"])
        if not before["valid"] or before["kind"] != "known_box" or not before.get("target"):
            raise ValueError("effect baseline needs a valid known_box estimate with a target reference")
        compiled["before"] = deepcopy(before)
        if kind == "placement":
            vector(spec["target_m"], "target_m", 3)
            number(spec["support_z_m"], "support_z_m")
        result.append(compiled)
    return result


def verify(compiled, after, job, *, now):
    """Verify using captured feedback. Unknown cannot be repaired with commanded state or World.held."""
    spec = compiled["spec"]
    out = dict(id=uuid4().hex, schema_version=1, job=job["id"], criteria=deepcopy(spec),
               evidence=[] if after is None else after["evidence"], status="unknown", measured={})
    if after is None or not after["valid"]:
        return dict(out, reason="geometry_unavailable" if after is None else after["reason"])
    if after["kind"] != "known_box" or after.get("tool") is None:
        return dict(out, reason="known_shape_and_synchronized_tool_required")
    window = job.get("capture_window", {})
    if window.get("end") is None or after["timestamp"] < window["end"]:
        return dict(out, reason="observation_must_follow_job_completion")
    age = now - after["timestamp"]
    if not math.isfinite(age) or not 0 <= age <= spec["max_age_s"]:
        return dict(out, reason="stale_observation")
    T = np.asarray(compiled["transform"])
    center = (np.asarray(after["components"]["center"]["value"]) - T[:3, 3]) @ T[:3, :3]
    tool = (np.asarray(after["tool"])[:3, 3] - T[:3, 3]) @ T[:3, :3]
    before = compiled["before"]
    out["evidence"] = before["evidence"] + after["evidence"]
    if any(before.get(key) != after.get(key) for key in
           ("target", "session", "camera", "calibration", "assumptions", "frame_assumption")):
        return dict(out, reason="incompatible_observations")
    if window.get("start") is None or before["timestamp"] > window["start"]:
        return dict(out, reason="baseline_must_precede_job")
    if spec["kind"] == "lift":
        if before["tool"] is None:
            return dict(out, reason="missing_synchronized_tool")
        origin = (np.asarray(before["components"]["center"]["value"]) - T[:3, 3]) @ T[:3, :3]
        tool_before = (np.asarray(before["tool"])[:3, 3] - T[:3, 3]) @ T[:3, :3]
        if window.get("tool_start_base_m") is None or window.get("tool_end_base_m") is None:
            return dict(out, reason="missing_job_feedback")
        start = (np.asarray(window["tool_start_base_m"]) - T[:3, 3]) @ T[:3, :3]
        end = (np.asarray(window["tool_end_base_m"]) - T[:3, 3]) @ T[:3, :3]
        if max(np.linalg.norm(start - tool_before), np.linalg.norm(end - tool)) >= spec["max_error_m"]:
            return dict(out, reason="observation_does_not_match_job_boundary")
        result = lift_effect(origin, center, end - start,
                             min_up_m=spec["min_up_m"], max_error_m=spec["max_error_m"])
    else:
        # An upright support-height predicate requires the shape's vertical to match the criterion frame.
        if after["frame_assumption"]["transform"] != compiled["transform"]:
            return dict(out, reason="incompatible_shape_frame")
        result = placement_effect(center, tool, after["aperture_mm"], target_m=spec["target_m"],
                                  position_tolerance_m=spec["position_tolerance_m"],
                                  support_center_z_m=spec["support_z_m"] + after["assumptions"]["size_m"][2] / 2,
                                  height_tolerance_m=spec["height_tolerance_m"],
                                  min_clearance_m=spec["min_clearance_m"], min_aperture_mm=spec["min_aperture_mm"])
    return dict(out, **result)
