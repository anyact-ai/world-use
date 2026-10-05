"""Session-local geometry and prepared plans, composed from existing measurements and PlanSpec.

This is data preparation on request threads. The kernel receives ordinary numeric
plans and compact prerequisites; it never reads these stores or runs perception.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4

import numpy as np

from . import procedures
from .behaviors import build
from .errors import Refused
from .validation import vector
from .world import World


class Phases:
    MAX_GEOMETRY = 256
    MAX_PLANS = 64

    def __init__(self, kernel, evidence):
        self.k, self.evidence = kernel, evidence
        self.geometries: OrderedDict[str, dict] = OrderedDict()
        self.plans: OrderedDict[str, dict] = OrderedDict()
        self.jobs: OrderedDict[int, dict] = OrderedDict()
        self.lock = threading.RLock()

    def world(self):
        with self.k.lock:
            return World.from_dict(self.k.world.to_dict())

    def fit(self, evidence, **options):
        measurement = self.evidence.measurement(evidence)
        # The support is the daemon's registered measurement, not coordinates supplied by a caller.
        result = procedures.fit(measurement, measurement.to_dict(), self.world(), **options)
        with self.lock:
            self.geometries[result["id"]] = result
            while len(self.geometries) > self.MAX_GEOMETRY:
                self.geometries.popitem(last=False)
        result = self.geometry(result["id"])
        self.k.emit("geometry", f"{result['kind']}: {result['reason'] or 'estimated'}", geometry=result)
        return result

    def geometry(self, identity):
        if not isinstance(identity, str):
            raise ValueError("geometry must be an ID")
        with self.lock:
            result = deepcopy(self.geometries.get(identity))
        if result is None:
            raise Refused("geometry expired or belongs to another session", "data_unavailable", "fit again")
        if result["valid"]:
            try:
                requirements = self.evidence.resolve([dict(evidence=e, max_age_s=1e12) for e in result["evidence"]])
                self.k.check_requirements(requirements)
                assumed = result["frame_assumption"]
                if not np.array_equal(self.world().frame(assumed["name"]).T, assumed["transform"]):
                    raise Refused("geometry's reference frame changed", "frame_changed")
            except (Refused, KeyError) as e:
                result.update(valid=False, reason=getattr(e, "rule", "frame_changed"), components={})
        return result

    def prepare(self, spec, *, requires=None, max_age_s=None, effects=None):
        world = self.world()
        requirements = {}  # Tightest age wins when multiple references share a source.
        for item in ([] if requires is None else requires):
            self.evidence.resolve([item])
            requirements[item["evidence"]] = min(item["max_age_s"], requirements.get(item["evidence"], float("inf")))
        derivations = []

        def resolve(node, depth=0):
            if depth > 32:
                raise ValueError("plan nesting exceeds 32 levels")
            if isinstance(node, list):
                return [resolve(step, depth + 1) for step in node]
            if not isinstance(node, dict):
                raise ValueError("plan steps must be objects")
            node = deepcopy(node)
            if node.get("do") == "seq":
                node["steps"] = resolve(node.get("steps"), depth + 1)
            if node.get("do") == "move_to":
                step_frame = node.get("frame", "work")
                T = world.frame(step_frame).T
                for field in ("to", "point", "jaws"):
                    ref = node.get(field)
                    if not isinstance(ref, dict):
                        continue
                    if ref.keys() - {"geometry", "component", "offset_m", "offset_frame", "sign"}:
                        raise ValueError("unknown geometry reference parameter")
                    age = procedures.positive(max_age_s, "max_age_s for referenced geometry")
                    geometry = self.geometry(ref.get("geometry"))
                    if not geometry["valid"]:
                        raise Refused(f"geometry is invalid: {geometry['reason']}", "invalid_evidence", "observe again")
                    component = geometry["components"].get(ref.get("component"))
                    expected = "position" if field == "to" else "direction"
                    if component is None or component["kind"] != expected:
                        raise ValueError(f"{field} needs a {expected} component")
                    value = np.asarray(component["value"], float)
                    if field == "to":
                        if "sign" in ref:
                            raise ValueError("sign is only for an unsigned direction")
                        if "offset_m" in ref:
                            vector(ref["offset_m"], "offset_m", 3)
                            if "offset_frame" not in ref:
                                raise ValueError("offset_m needs an explicit offset_frame")
                            value = value + world.frame(ref["offset_frame"]).T[:3, :3] @ ref["offset_m"]
                        elif "offset_frame" in ref:
                            raise ValueError("offset_frame needs offset_m")
                        node[field] = world.from_base(step_frame, value).tolist()
                    else:
                        if "offset_m" in ref or "offset_frame" in ref:
                            raise ValueError("directions cannot have position offsets")
                        if component.get("unsigned") and (isinstance(ref.get("sign"), bool)
                                                          or ref.get("sign") not in (-1, 1)):
                            raise ValueError("unsigned directions need an explicit sign of -1 or 1")
                        node[field] = (T[:3, :3].T @ value * ref.get("sign", 1)).tolist()
                    for identity in geometry["evidence"]:
                        requirements[identity] = min(age, requirements.get(identity, float("inf")))
                    derivations.append(dict(field=field, reference=ref, resolved=node[field],
                                            assumptions=geometry["assumptions"], evidence=geometry["evidence"]))
            return node

        resolved = resolve(spec)
        build(resolved)
        compiled = procedures.compile_effects([] if effects is None else effects, world, self.geometry)
        for effect in compiled:
            for identity in effect["before"]["evidence"]:
                requirements[identity] = min(effect["spec"]["max_age_s"],
                                             requirements.get(identity, float("inf")))
        prerequisites = [dict(evidence=e, max_age_s=age) for e, age in requirements.items()]
        self.k.check_requirements(self.evidence.resolve(prerequisites))
        return dict(plan=resolved, requires=prerequisites, derivations=derivations, effects=compiled,
                    frames={name: f.T.tolist() for name, f in world.frames.items()})

    def save(self, prepared, report):
        identity = uuid4().hex
        result = dict(deepcopy(prepared), id=identity, session=self.evidence.session, report=report,
                      request_id=f"{self.evidence.session}:{identity}")
        with self.lock:
            self.plans[identity] = result
            while len(self.plans) > self.MAX_PLANS:
                self.plans.popitem(last=False)
        self.k.emit("prepared", "phase checked; no motion submitted", phase=result)
        return deepcopy(result)

    def load(self, identity):
        with self.lock:
            result = deepcopy(self.plans.get(identity))
        if result is None:
            raise Refused("prepared plan expired or belongs to another session", "plan_invalidated", "check again")
        current = self.world()
        if any(name not in current.frames or not np.array_equal(current.frame(name).T, T)
               for name, T in result["frames"].items()):
            raise Refused("a prepared plan's reference frame changed", "plan_invalidated", "check again")
        self.k.check_requirements(self.evidence.resolve(result["requires"]))
        return result

    def submitted(self, job, prepared):
        with self.lock:
            self.jobs[job] = deepcopy(prepared)
            while len(self.jobs) > self.MAX_GEOMETRY:
                self.jobs.popitem(last=False)
        self.k.emit("phase_submitted", f"job {job}: checked phase", job=job, phase=prepared["id"],
                    derivations=prepared["derivations"], effects=prepared["effects"])

    def verify(self, job_id, effect, after):
        if isinstance(effect, bool) or not isinstance(effect, int) or effect < 0:
            raise ValueError("effect must be a nonnegative index")
        with self.lock:
            prepared = self.jobs.get(job_id)
        if prepared is None or effect >= len(prepared["effects"]):
            raise ValueError("job has no such predeclared effect")
        with self.k.lock:
            job = self.k.jobs[job_id].to_dict()
        try:
            observed = self.geometry(after)
        except Refused:
            observed = None
        result = procedures.verify(prepared["effects"][effect], observed, job, now=self.k.evidence_now())
        result.update(effect=effect, phase=prepared["id"])
        self.k.emit("verification", f"job {job_id}: {result['status']}", job=job_id, verification=result)
        return result
