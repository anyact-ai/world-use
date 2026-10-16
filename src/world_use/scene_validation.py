"""Explicit reference checks for a declared scene; independent of task success and robot control.

This is an evidence recorder, not a reconstruction algorithm. Callers supply correspondences,
metric measurements and aligned motion. Synthetic evidence exercises the checks but cannot
establish real-to-sim accuracy. Declare every geometry, camera, dynamics and solver input in
``scene_identity``; the helper cannot discover inputs a caller leaves out.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from .calibrate import solve
from .cameras import View
from .tasks import digest, json_bytes, write_new

GROUPS = {
    "appearance_camera": ("landmarks",),
    "geometry": ("dimensions",),
    "physical_response": ("motion", "stability"),
}
LIMITS = ("landmark_px", "geometry_m", "motion_m", "penetration_m", "support_m", "drift_m")


def scene_identity(sources: Mapping[str, Path], parameters: dict) -> dict:
    """Hash declared source/asset bytes and all scene parameters, including calibration and solver settings."""
    if not sources:
        raise ValueError("declare the scene source files")
    value = dict(sources={name: digest(Path(path).read_bytes()) for name, path in sorted(sources.items())},
                 parameters=parameters)
    # Canonicalize nested parameter dictionaries as well as the top-level source names.
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return dict(revision=digest(encoded), **json.loads(encoded))


@dataclass(frozen=True)
class Observation:
    """A reserved observation and its source. Alignment is declared before comparison.

    ``alignment`` must identify the metric coordinate frame, image crop/resize (or none),
    and simulation-time = reference-time + time_offset_s. Splits are observation IDs,
    not random samples taken from an otherwise shared fitting trajectory.
    """
    name: str
    path: Path
    origin: str
    kind: Literal["measured", "synthetic", "assumed"]
    split: Literal["fit", "held_out"]
    alignment: dict
    checks: tuple[str, ...] = ("landmarks", "dimensions", "motion", "stability")
    interval_s: tuple[float, float] | None = None

    def __post_init__(self):
        if not self.name or not self.origin or self.kind not in ("measured", "synthetic", "assumed"):
            raise ValueError("observation needs a name, source and evidence kind")
        if self.split not in ("fit", "held_out"):
            raise ValueError("observation split must be fit or held_out")
        if not {"frame", "image_transform", "time_offset_s"} <= self.alignment.keys():
            raise ValueError("declare the frame, image_transform and time_offset_s")
        if not np.isfinite(self.alignment["time_offset_s"]):
            raise ValueError("time_offset_s must be finite")
        if not self.checks or set(self.checks) - {name for names in GROUPS.values() for name in names}:
            raise ValueError("declare the required checks for this observation")
        if self.interval_s is not None and (
            len(self.interval_s) != 2 or not np.isfinite(self.interval_s).all()
            or self.interval_s[0] >= self.interval_s[1]
        ):
            raise ValueError("observation interval must have finite increasing endpoints")


def fit_camera(observation: Observation, points_m, pixels, size, *, fov_deg=60.0, focal=None) -> View:
    """Reuse the camera fitter on development observations only; no reserved-point selection or tuning."""
    if observation.split != "fit":
        raise ValueError("camera fitting requires a fit observation; held-out observations are checks only")
    return solve(points_m, pixels, size, fov_deg=fov_deg, focal=focal).view


def _array(values, *, columns=None):
    array = np.asarray(values, float)
    if not array.size or (columns is not None and (array.ndim != 2 or array.shape[1] != columns)):
        raise ValueError("measurements must be nonempty arrays with the declared dimensions")
    return array


def _status(checks):
    if any(check["status"] == "failed" for check in checks):
        return "failed"
    return "passed" if checks and all(check["status"] == "passed" for check in checks) else "unvalidated"


class Validation:
    """One immutable scene revision, fixed tolerances and observation split, saved to a fresh directory.

    Each group retains its own fit and held-out result. Passing means only the declared
    numerical checks passed; materials, lighting, forces and transfer remain omitted.
    """

    def __init__(self, output: Path, *, scene: dict, observations: list[Observation], tolerances: dict,
                 limitations: tuple[str, ...] = ()):
        if set(tolerances) != set(LIMITS) or any(
            not np.isfinite(value) or value < 0 for value in tolerances.values()
        ):
            raise ValueError(f"declare finite nonnegative tolerances for {', '.join(LIMITS)}")
        if len({item.name for item in observations}) != len(observations):
            raise ValueError("an observation ID cannot appear in both splits or be repeated")
        if not scene.get("revision") or not scene.get("sources"):
            raise ValueError("use scene_identity to declare the scene revision")
        self.output = Path(output)
        self.scene = json.loads(json_bytes(scene))
        self.tolerances = dict(tolerances)
        self.limitations = list(limitations)
        self.observations = {item.name: replace(item, alignment=json.loads(json_bytes(item.alignment)))
                             for item in observations}
        payloads = [(item, item.path.read_bytes()) for item in observations]
        for index, (item, data) in enumerate(payloads):
            for other, other_data in payloads[:index]:
                if item.split == other.split or digest(data) != digest(other_data):
                    continue
                a, b = item.interval_s, other.interval_s
                if a is None or b is None or not (a[1] < b[0] or b[1] < a[0]):
                    raise ValueError("fit and held-out evidence overlaps; declare disjoint observation intervals")
        self.output.mkdir(parents=True, exist_ok=False)
        self.evidence = {}
        for item, data in payloads:
            self.evidence[item.name] = dict(**self.artifact(data, item.path.suffix.lstrip(".") or "bin"),
                                           origin=item.origin, kind=item.kind, split=item.split,
                                           alignment=self.observations[item.name].alignment, checks=list(item.checks),
                                           interval_s=None if item.interval_s is None else list(item.interval_s))
        self.checks: list[dict] = []
        self.finished = False
        write_new(self.output / "specification.json", json_bytes(dict(
            scene=scene, tolerances=tolerances, observations=self.evidence)))

    def artifact(self, data: bytes, suffix: str) -> dict:
        sha = digest(data)
        relative = f"artifacts/{sha}.{suffix}"
        path = self.output / relative
        if not path.exists():
            write_new(path, data)
        return dict(path=relative, sha256=sha)

    def _check(self, observation: str, name: str, metrics: dict, details: dict, *,
               valid=True, complete=True, reason=None):
        if self.finished:
            raise ValueError("validation is already finished")
        item = self.observations[observation]
        if name not in item.checks:
            raise ValueError("check was not declared for this observation")
        if any(check["observation"] == observation and check["name"] == name for check in self.checks):
            raise ValueError("a check cannot be replaced after seeing its result")
        finite = all(np.isfinite(value) for value in metrics.values())
        passed = valid and finite and all(value <= self.tolerances[key] for key, value in metrics.items())
        status = "passed" if passed else "failed"
        if (item.kind == "assumed" or not complete) and passed:
            status = "unvalidated"
        artifact = self.artifact(json_bytes(details), "json")
        result = dict(name=name, observation=observation, split=item.split, status=status,
                      metrics={key: float(value) if np.isfinite(value) else None for key, value in metrics.items()},
                      tolerances={key: self.tolerances[key] for key in metrics}, details=artifact,
                      reason=reason or (None if finite else "nonfinite measurement or simulation state"))
        self.checks.append(result)
        return result

    def landmarks(self, observation: str, view: View, points_m, pixels):
        points, expected = _array(points_m, columns=3), _array(pixels, columns=2)
        if len(points) != len(expected) or not np.isfinite(points).all() or not np.isfinite(expected).all():
            raise ValueError("landmark correspondences must be finite and have matching lengths")
        projected, depth = view.project(points)
        valid = bool(np.isfinite(projected).all() and np.all(depth > 0))
        error = float(np.max(np.linalg.norm(projected - expected, axis=1))) if valid else float("inf")
        return self._check(observation, "landmarks", {"landmark_px": error}, dict(
            view=view.to_dict(), points_m=points.tolist(), reference_pixels=expected.tolist(),
            projected_pixels=projected.tolist() if valid else None), valid=valid,
            reason=None if valid else "landmarks behind camera or nonfinite projection")

    def dimensions(self, observation: str, reference_m, predicted_m, *, labels: list[str]):
        expected, actual = _array(reference_m), _array(predicted_m)
        if expected.shape != actual.shape or expected.ndim != 1 or len(labels) != len(expected):
            raise ValueError("dimension labels and measurement arrays must match")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("metric dimensions must be finite")
        return self._check(observation, "dimensions", {"geometry_m": float(np.max(np.abs(actual - expected)))},
                           dict(labels=labels, reference_m=expected.tolist(), predicted_m=actual.tolist()))

    def motion(self, observation: str, reference_time_s, reference_xyz_m, simulation_time_s, simulation_xyz_m):
        """Compare centre positions or corresponding rigid-body landmarks (time, landmark, xyz).

        Landmark trajectories measure rotation as physical displacement at the object's extent.
        Reference times use the declared interval and offset; interpolation never extrapolates.
        """
        ref_t, sim_t = _array(reference_time_s), _array(simulation_time_s)
        expected, actual = _array(reference_xyz_m), _array(simulation_xyz_m)
        if (expected.ndim not in (2, 3) or actual.ndim != expected.ndim or expected.shape[1:] != actual.shape[1:]
                or expected.shape[-1] != 3
                or ref_t.ndim != 1 or sim_t.ndim != 1 or len(ref_t) != len(expected) or len(sim_t) != len(actual)
                or len(sim_t) < 2 or not all(np.isfinite(a).all() for a in (ref_t, sim_t, expected, actual))
                or np.any(np.diff(ref_t) <= 0) or np.any(np.diff(sim_t) <= 0)):
            raise ValueError("motion needs finite matching samples and strictly increasing timestamps")
        aligned_t = ref_t + self.observations[observation].alignment["time_offset_s"]
        interval = self.observations[observation].interval_s
        if interval is not None and (ref_t[0] < interval[0] or ref_t[-1] > interval[1]):
            raise ValueError("reference motion lies outside the observation's reserved interval")
        covered = bool(aligned_t[0] >= sim_t[0] and aligned_t[-1] <= sim_t[-1])
        flat = actual.reshape(len(sim_t), -1)
        interpolated = np.column_stack([np.interp(aligned_t, sim_t, flat[:, i]) for i in range(flat.shape[1])])
        error = float(np.max(np.linalg.norm(interpolated.reshape(expected.shape) - expected, axis=-1)))
        return self._check(observation, "motion", {"motion_m": error}, dict(
            reference_time_s=ref_t.tolist(), simulation_time_s=sim_t.tolist(),
            reference_xyz_m=expected.tolist(), simulation_xyz_m=actual.tolist(),
            aligned_simulation_time_s=aligned_t.tolist()), valid=covered,
            reason=None if covered else "simulation does not cover the declared reference time interval")

    def stability(self, observation: str, *, state, penetration_m, support_error_m, rest_xyz_m=None):
        """Check physical validity; omit rest positions when no measured resting interval exists.

        Missing rest evidence leaves drift unvalidated, without masking failures in
        the available state, penetration or support measurements.
        """
        states = _array(state)
        penetration, support = _array(penetration_m), _array(support_error_m)
        rest = None if rest_xyz_m is None else _array(rest_xyz_m, columns=3)
        if states.ndim < 1 or (rest is not None and len(rest) < 2):
            raise ValueError("stability needs state samples and, when supplied, at least two resting positions")
        samples = dict(state=states, penetration_m=penetration, support_error_m=support)
        metrics = dict(penetration_m=float(np.max(np.maximum(0, penetration))),
                       support_m=float(np.max(np.abs(support))))
        if rest is not None:
            samples["rest_xyz_m"] = rest
            metrics["drift_m"] = float(np.max(np.linalg.norm(rest - rest[0], axis=1)))
        finite = all(np.isfinite(a).all() for a in samples.values())
        missing = ["measured rest interval"] if rest is None else []
        reasons = ([] if finite else ["nonfinite simulation state"]) + (
            ["no measured rest interval; drift is unvalidated"] if missing else [])
        # Invalid simulator values still produce a durable failed check rather than invalid JSON.
        stream = BytesIO()
        np.savez_compressed(stream, **samples)
        return self._check(observation, "stability", metrics, dict(
            finite_state=finite, state_samples=int(states.shape[0]), rest_samples=0 if rest is None else len(rest),
            missing=missing, samples=self.artifact(stream.getvalue(), "npz")),
            valid=finite, complete=rest is not None, reason="; ".join(reasons) or None)

    def images(self, observation: str, reference: Image.Image, render: Image.Image) -> dict:
        """Save originals and a labelled, same-pixel-grid comparison; no automatic image fitting."""
        if self.finished or "images" in self.evidence[observation]:
            raise ValueError("image comparison is already recorded or validation is finished")
        if reference.size != render.size:
            raise ValueError("align image dimensions explicitly before comparison")
        item = self.observations[observation]
        refs = {}
        for name, image in (("reference", reference), ("render", render)):
            stream = BytesIO()
            image.save(stream, format="PNG")
            refs[name] = self.artifact(stream.getvalue(), "png")
        panel = Image.new("RGB", (reference.width * 2, reference.height + 32), "#eeeeee")
        panel.paste(reference, (0, 32))
        panel.paste(render, (reference.width, 32))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f"{item.kind.upper()} REFERENCE | {item.name} | {item.split}", fill="black")
        draw.text((reference.width + 8, 8), f"SCENE {self.scene['revision'][:12]}", fill="black")
        stream = BytesIO()
        panel.save(stream, format="PNG")
        refs["comparison"] = self.artifact(stream.getvalue(), "png")
        self.evidence[observation]["images"] = refs
        return refs

    def finish(self) -> dict:
        if self.finished:
            raise ValueError("validation is already finished")
        groups = {}
        for group, required in GROUPS.items():
            groups[group] = {}
            for split in ("fit", "held_out"):
                checks = [check for check in self.checks if check["name"] in required and check["split"] == split]
                missing = [name for name in required if not any(check["name"] == name for check in checks)]
                missing += [f"{item.name}:{name}" for item in self.observations.values() if item.split == split
                            for name in item.checks if name in required and not any(
                                check["name"] == name and check["observation"] == item.name for check in checks)]
                status = _status(checks)
                if missing and status != "failed":
                    status = "unvalidated"
                groups[group][split] = dict(status=status, missing=missing, checks=checks)
        kinds = {item.kind for item in self.observations.values()}
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        status = _status([value for group in groups.values() for value in group.values()])
        if kind != "measured" and status == "passed":
            status = "unvalidated"
        report = dict(format_version=1, scene_revision=self.scene["revision"], scene=self.scene,
                      reference=dict(kind=kind, observations=self.evidence), status=status, groups=groups,
                      tolerances=self.tolerances, scope="declared observations and tolerances only",
                      unvalidated=self.limitations +
                      ["materials and lighting", "forces", "transfer to another setup"] + (
                          ["real-to-sim accuracy: reference is not entirely measured"] if kind != "measured" else []))
        write_new(self.output / "scene-validation.json", json_bytes(report))
        self.finished = True
        return report
