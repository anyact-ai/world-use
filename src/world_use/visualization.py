"""Read-only Rerun visualization of flight records, including a running recorder's committed chunks."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

from .kinematics import Chain
from .recorder import read_chunks
from .records import overview, robot_of
from .robot_assets import resolve
from .world import World

TIMELINE = "elapsed"
COLORS = {"surface": [155, 173, 184], "object": [238, 130, 50], "keep_out": [228, 75, 75],
          "fragile": [187, 110, 215], "slow": [233, 198, 74]}


def _sdk():
    try:
        import rerun as rr
    except ImportError as e:
        raise ValueError("visualization needs the rerun extra: uv tool install --force "
                         "'world-use[rerun] @ git+https://github.com/anyact-ai/world-use'") from e
    return rr


class RecordReader:
    """Read each committed chunk once; leave a partially written event for the next poll."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.parts: set[Path] = set()
        self.offset = 0
        self.losses = {}
        self.complete = False

    def poll(self) -> tuple[dict, list[dict]]:
        # Read completion before listing chunks: a close committed during this poll belongs to the next one.
        marker = self.folder / "complete.json"
        complete = json.loads(marker.read_text()) if marker.exists() else None
        paths = set((self.folder / "tape").glob("[0-9]*.npz"))
        samples = read_chunks(sorted(paths - self.parts))
        self.parts.update(paths)
        # Journal writes events before publishing the corresponding telemetry chunk.
        events = []
        path = self.folder / "events.jsonl"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(self.offset)
                for line in stream:
                    if not line.endswith(b"\n"):
                        break
                    events.append(json.loads(line))
                    self.offset += len(line)
        path = self.folder / "recording.json"
        if path.exists():
            losses = json.loads(path.read_text())
            if losses != self.losses:
                self.losses = losses
                t = max(float(samples["t"][-1]) if len(samples.get("t", [])) else 0.0,
                        events[-1]["t"] if events else 0.0)
                events.append(dict(t=t, kind="recording", level="alarm",
                                   message=f"Incomplete record: {losses}; estimates may span missing observations"))
        self.complete = (complete is not None and len(self.parts) == complete["parts"]
                         and self.offset == complete["events_bytes"])
        return samples, events


def blueprint(base_frame: str, follow: bool, eye: tuple, cameras=()):
    import rerun as rr
    import rerun.blueprint as rrb

    images = [rrb.Spatial2DView(name=f"Camera · {name}",
                                origin=f"/observations/{rr.escape_entity_path_part(name)}") for name in cameras]
    if not images:
        images = [rrb.TextDocumentView(name="Camera observations", origin="/camera-help")]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Robot · observed surfaces · estimated world",
                                  contents=["/robot/**", "/scene/**"],
                                  spatial_information=rrb.SpatialInformation(base_frame),
                                  eye_controls=rrb.EyeControls3D(position=eye[0], look_target=eye[1],
                                                               eye_up=[0, 0, 1]),
                                  background=[24, 29, 36], line_grid=True),
                rrb.Tabs(*images, rrb.TextDocumentView(name="About this run", origin="/about")),
                column_shares=[3, 2]),
            rrb.Tabs(
                rrb.TimeSeriesView(name="Joints · measured / commanded (deg)", origin="/signals/joints"),
                rrb.TimeSeriesView(name="Torque (Nm)", origin="/signals/torque"),
                rrb.TimeSeriesView(name="Temperature (°C)", origin="/signals/temperature"),
                rrb.TimeSeriesView(name="Gripper (native units)", origin="/signals/gripper"),
                rrb.TimeSeriesView(name="Power and active job", origin="/signals/state")),
            rrb.TextLogView(name="Execution events", origin="/events"), row_shares=[6, 3, 2]),
        rrb.TimePanel(timeline=TIMELINE, play_state="Following" if follow else "Paused"),
        rrb.SelectionPanel(state="Collapsed"), collapse_panels=True)


class RecordingView:
    """One Rerun stream; no adapter, MuJoCo rollout, or command client is created."""

    def __init__(self, folder: Path, recording, *, follow=False, update_layout=True):
        from rerun.urdf import UrdfTree

        self.rr, self.rec, self.folder = _sdk(), recording, folder.resolve()
        self.follow, self.camera_names = follow, set()
        self.update_layout = update_layout
        self.meta = json.loads((folder / "session.json").read_text())
        self.manifest = robot_of(folder)
        self.chain = Chain(self.manifest.urdf, self.manifest.tool_link)
        self.world = World.from_dict(self.meta["initial"]["world"])
        root = ET.parse(self.manifest.urdf).getroot()
        for link in root.findall("link"):
            for collision in link.findall("collision"):
                link.remove(collision)   # Display the manufacturer's visual geometry, not collision hulls.
        for mesh in root.findall(".//mesh"):
            asset = resolve(self.manifest.urdf, mesh.attrib["filename"])
            if not asset.is_file():
                raise ValueError(f"viewer mesh does not exist: {asset}")
            mesh.set("filename", str(asset.resolve()))
        with tempfile.TemporaryDirectory(prefix="world-use-view-") as temp:
            urdf = Path(temp) / "robot.urdf"
            ET.ElementTree(root).write(urdf)
            self.tree = UrdfTree.from_file_path(urdf, entity_path_prefix="robot", frame_prefix="robot/")
            self.tree.log_urdf_to_recording(self.rec)
        self.base_frame = f"robot/{self.tree.root_link().name}"
        self.eye = tuple(v.tolist() for v in overview(self.chain))
        self.joints = {j.name: j for j in self.tree.joints() if j.joint_type != "fixed"}
        arm = {j.name for j in self.manifest.joints}
        others = [j for name, j in self.joints.items() if name not in arm]
        g = self.manifest.gripper
        # The recorded aperture moves two prismatic fingers; any other gripper joint stays where the URDF puts it.
        animated = (g is not None and g.m_per_unit is not None and len(others) == 2
                    and all(j.joint_type == "prismatic" for j in others))
        self.fingers = others if animated else []
        self.last_t = -np.inf
        self.last_scene = -np.inf
        self.pending: deque[dict] = deque()
        self.trail: deque[tuple[float, np.ndarray]] = deque(maxlen=200)
        self.box_names: set[str] = set()
        rr = self.rr
        self.rec.log("scene", rr.CoordinateFrame(self.base_frame), rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.rec.log("camera-help", rr.TextDocument(
            "No saved camera observations yet.\n\nUse `wu look CAMERA` during a session, or run "
            "`wu demo` with video enabled.\n\nCamera frames appear on the same elapsed timeline as the measurements.",
            media_type="text/markdown"), static=True)
        self.rec.log("about", rr.TextDocument(
            f"# {self.meta['body']}\n\n{self.meta['mode']} · world-use {self.meta['package_version']}\n\n"
            "The 3D view combines measured joints with the **estimated world**. Objects in that view are beliefs, "
            "not reconstructed physical motion. Camera observations show the captured scene, "
            "with any saved overlays.\n\n"
            "Joint plots retain every committed sample. The world and tool trail update at up to 20 Hz. "
            "Live following reads background recordings, usually about one second behind control.\n\n"
            "This viewer is read-only. Closing it does not stop a job or change motor power.",
            media_type="text/markdown"), static=True)
        self.rec.set_time(TIMELINE, duration=0.0)
        if not animated and others:
            for joint in others:
                self.rec.log(self._path("transforms", joint.name), joint.compute_transform(0.0, clamp=False))
            self.rec.log("events", rr.TextLog(
                f"Gripper joints {', '.join(j.name for j in others)} are drawn static: the viewer moves two "
                "prismatic fingers with a calibrated aperture.", level="WARN"))
        initial = self.meta["initial"]
        for spec, value in zip(self.manifest.joints, initial["q"], strict=True):
            self.rec.log(self._path("transforms", spec.name),
                         self.joints[spec.name].compute_transform(value, clamp=False))
        if g is not None and initial["gripper"] is not None:
            for joint in self.fingers:
                self.rec.log(self._path("transforms", joint.name),
                             joint.compute_transform((g.aperture(initial["gripper"]) or 0) / 2, clamp=False))
        self._world(self.chain.fk(initial["q"]))

    def _path(self, prefix, name):
        return f"{prefix}/{self.rr.escape_entity_path_part(name)}"

    def _event(self, event):
        rr, rec = self.rr, self.rec
        rec.set_time(TIMELINE, duration=event["t"])
        rec.log("events", rr.TextLog(f"{event['kind']}: {event['message']}",
                                    level={"info": "INFO", "warn": "WARN", "alarm": "ERROR"}[event["level"]]))
        data = event.get("data", {})
        if "world" in data:
            self.world = World.from_dict(data["world"])
            # Log even changes after the last telemetry sample (e.g. an operator annotation).
            self._world(None)
        if event["kind"] == "look" and data.get("path"):
            path = (self.folder / data["path"]).resolve()
            if not path.is_relative_to(self.folder):
                raise ValueError("recorded observation path escapes its run folder")
            if path.is_file():
                rec.log(self._path("observations", data["camera"]), rr.EncodedImage(path=path))
            else:
                rec.log("events", rr.TextLog(f"Missing saved observation: {data['path']}", level="WARN"))
        if event["kind"] == "measurement":
            self._measurement(data["measurement"], event["t"])

    def _measurement(self, m, t):
        """The measured picture at its capture time; its surface points from when they were measured."""
        rr, rec = self.rr, self.rec
        image = (self.folder / m["image"]).resolve()
        if not image.is_relative_to(self.folder):
            raise ValueError("recorded measurement path escapes its run folder")
        points = image.with_suffix(".npz")
        if not image.is_file() or not points.is_file():
            rec.log("events", rr.TextLog(f"Missing measurement files: {m['image']}", level="WARN"))
            return
        rec.set_time(TIMELINE, duration=m["capture_t"])
        rec.log(self._path("observations", m["camera"]), rr.EncodedImage(path=image))
        rec.set_time(TIMELINE, duration=t)
        path = self._path("scene/observed", m["target"] or m["id"])
        rec.log(path, rr.Clear(recursive=True))
        if m["valid"]:
            with np.load(points) as arrays:
                rec.log(path, rr.CoordinateFrame(self.base_frame),
                        rr.Points3D(arrays["points"], colors=[30, 220, 100], radii=.002,
                                    labels=[f"measured: {m['target'] or 'surface'} at {m['capture_t']:.2f}s"]))

    def _world(self, tool):
        rr, rec = self.rr, self.rec
        if tool is not None:
            self.world.carry(tool)
        names = set(self.world.boxes)
        for name in self.box_names - names:
            rec.log(self._path("scene/world", name), rr.Clear(recursive=True))
        self.box_names = names
        for name, box in self.world.boxes.items():
            path = self._path("scene/world", name)
            frame_id = f"world/{name}"
            rec.log(path, rr.CoordinateFrame(frame_id),
                    rr.Transform3D(translation=box.pose[:3, 3], mat3x3=box.pose[:3, :3],
                                   parent_frame=self.base_frame, child_frame=frame_id),
                    rr.Boxes3D(sizes=box.size, colors=COLORS[box.kind], labels=[f"{name} ({box.kind})"],
                               fill_mode="Solid" if box.kind in ("surface", "object") else "MajorWireframe"))
        for name, frame in self.world.frames.items():
            frame_id = f"world-frame/{name}"
            rec.log(self._path("scene/frames", name),
                    rr.CoordinateFrame(frame_id),
                    rr.Transform3D(translation=frame.T[:3, 3], mat3x3=frame.T[:3, :3],
                                   parent_frame=self.base_frame, child_frame=frame_id),
                    rr.Arrows3D(vectors=np.eye(3) * .06, colors=[[230, 70, 70], [70, 200, 100], [80, 140, 240]]))

    def _scalars(self, path, times, values):
        values = np.asarray(values)
        valid = np.isfinite(values)
        if valid.any():
            self.rec.send_columns(path, indexes=[self.rr.TimeColumn(TIMELINE, duration=times[valid])],
                                  columns=self.rr.Scalars.columns(scalars=values[valid]))

    def append(self, samples: dict, events: list[dict]):
        rr, rec = self.rr, self.rec
        cameras = {e["data"]["camera"] for e in events if e["kind"] == "look" and e.get("data", {}).get("camera")}
        cameras.update(e["data"]["measurement"]["camera"] for e in events if e["kind"] == "measurement")
        if cameras - self.camera_names:
            self.camera_names.update(cameras)
            if self.update_layout:
                rec.send_blueprint(blueprint(self.base_frame, self.follow, self.eye, sorted(self.camera_names)))
        self.pending.extend(events)
        times = np.asarray(samples.get("t", []))
        mask = times > self.last_t
        times = times[mask]
        if not len(times):
            # Events can be committed before their telemetry. Keep them queued until that sample arrives.
            return
        a = {key: value[mask] for key, value in samples.items() if not key.startswith("power_")}
        indexes = [rr.TimeColumn(TIMELINE, duration=times)]
        for i, spec in enumerate(self.manifest.joints):
            joint = self.joints[spec.name]
            rec.send_columns(self._path("transforms", spec.name), indexes=indexes,
                             columns=joint.compute_transform_columns(a["q"][:, i], clamp=False))
            for source, label in (("q", "measured"), ("q_cmd", "commanded")):
                self._scalars(self._path("signals/joints", spec.name) + f"/{label}", times,
                              np.degrees(a[source][:, i]))
            self._scalars(self._path("signals/torque", spec.name), times, a["tau"][:, i])
            self._scalars(self._path("signals/temperature", spec.name), times, a["temp"][:, i])
        g = self.manifest.gripper
        if g is not None:
            valid = np.isfinite(a["grip"])
            values = (a["grip"][valid] - g.closed) * (g.m_per_unit or 0) / 2
            for joint in self.fingers:
                rec.send_columns(self._path("transforms", joint.name),
                                 indexes=[rr.TimeColumn(TIMELINE, duration=times[valid])],
                                 columns=joint.compute_transform_columns(values, clamp=False))
        for key in ("grip", "grip_cmd", "grip_tau"):
            self._scalars(f"signals/gripper/{key}", times, a[key])
        for key in ("enabled", "job"):
            self._scalars(f"signals/state/{key}", times, a[key])
        for i, t in enumerate(times):
            while self.pending and self.pending[0]["t"] <= t:
                self._event(self.pending.popleft())
            if t - self.last_scene < .05 - 1e-9 and i != len(times) - 1:
                continue
            rec.set_time(TIMELINE, duration=t)
            tool = self.chain.fk(a["q"][i])
            self._world(tool)
            self.trail.append((float(t), tool[:3, 3]))
            while self.trail and self.trail[0][0] < t - 10:
                self.trail.popleft()
            rec.log("scene/tool", rr.CoordinateFrame("measured-tool"),
                    rr.Transform3D(translation=tool[:3, 3], mat3x3=tool[:3, :3],
                                   parent_frame=self.base_frame, child_frame="measured-tool"),
                    rr.Arrows3D(vectors=np.eye(3) * .04, colors=[[230, 70, 70], [70, 200, 100], [80, 140, 240]]))
            rec.log("scene/trail", rr.CoordinateFrame(self.base_frame),
                    rr.LineStrips3D([np.array([p for _, p in self.trail])],
                                                  colors=[70, 180, 240], radii=.0015))
            self.last_scene = t
        self.last_t = float(times[-1])

    def finish_events(self):
        while self.pending:
            self._event(self.pending.popleft())


def view(folder: Path | str, *, output: Path | None = None, follow: bool = False) -> Path | None:
    rr = _sdk()
    folder = Path(folder).expanduser().resolve()
    if not (folder / "session.json").is_file():
        raise ValueError(f"view needs a local run folder containing session.json: {folder}")
    rec = rr.RecordingStream("world-use")
    viewer = None
    try:
        # Set the sink before loading meshes, so large static assets don't accumulate in the SDK.
        if output is not None:
            output = output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            rec.save(output)
        else:
            # A tool install keeps rerun-sdk's viewer app beside this Python, off the PATH; elsewhere, use the PATH.
            app = shutil.which("rerun", path=str(Path(sys.executable).parent))
            rr.spawn(recording=rec, memory_limit="1GiB", hide_welcome_screen=True, executable_path=app)
        viewer = RecordingView(folder, rec, follow=follow, update_layout=output is None)
        if output is None:
            rec.send_blueprint(blueprint(viewer.base_frame, follow, viewer.eye))
        reader = RecordReader(folder)
        while True:
            samples, events = reader.poll()
            viewer.append(samples, events)
            if not follow or reader.complete:
                viewer.finish_events()
                break
            time.sleep(.25)
    except KeyboardInterrupt:
        if viewer is not None:
            viewer.finish_events()
    finally:
        if output is not None and viewer is not None:
            # Save one complete layout after all camera names are known, including on Ctrl+C.
            rec.send_blueprint(blueprint(viewer.base_frame, False, viewer.eye, sorted(viewer.camera_names)))
        rec.flush()
        rec.disconnect()
    return output
