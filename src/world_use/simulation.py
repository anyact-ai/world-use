"""Local MuJoCo sessions using the normal daemon, client and task records.

Scene construction and scoring belong to the caller. Physics state is an evaluator/replay
artifact; it is never added to the client observations or the kernel's estimated world.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageDraw, ImageFont

from .bodies.mujoco_render import CameraRenderer
from .bodies.mujoco_scene import mj
from .cameras import View
from .client import Client
from .daemon import Daemon
from .kernel import Kernel
from .recorder import save_arrays, save_summary
from .tasks import Record, RecordedClient, RecordingError, prepare
from .world import World


class MujocoRecording:
    """Sample completed control steps; persist bounded batches off the control thread."""

    def __init__(self, body: Any, output: Path):
        self.body, self.output = body, Path(output)
        self.folder = self.output / "physics"
        self.folder.mkdir()
        self.start = time.monotonic()
        self.error: str | None = None
        self.closed = False
        self.parts = 0
        self.last_time = -float("inf")
        self.state_spec = mj.mjtState.mjSTATE_INTEGRATION
        self.pending: queue.Queue = queue.Queue(maxsize=256)
        mj.mj_saveModel(body.model, str(self.output / "scene.mjb"), None)
        save_summary(self.folder / "format.json", dict(
            version=1, mujoco=mj.__version__, state_spec=int(self.state_spec),
            time="elapsed monotonic seconds", simulation_time="MuJoCo seconds",
            sample_period_s=.02, model="../scene.mjb",
            base_to_world=np.asarray(getattr(body, "base", np.eye(4))).tolist()))
        self.thread = threading.Thread(target=self._write, name="scene-recorder", daemon=True)
        self.thread.start()
        self.capture()

    def capture(self):
        if self.closed or self.error:
            return
        with self.body.lock:
            model, data = self.body.model, self.body.data
            if data.time < self.last_time:
                self.error = "scene time moved backwards; start a new session after a reset"
                return
            if data.time - self.last_time < .02 - 1e-9:
                return
            state = np.empty(mj.mj_stateSize(model, self.state_spec))
            mj.mj_getState(model, data, state, self.state_spec)
            sample = (time.monotonic() - self.start, float(data.time),
                      data.qpos.copy(), data.qvel.copy(), state)
            self.last_time = float(data.time)
        try:
            self.pending.put_nowait(sample)
        except queue.Full:
            self.error = "scene recording queue overflow; physics evidence is incomplete"

    def _write(self):
        batch = []
        while True:
            try:
                item = self.pending.get(timeout=.5)
            except queue.Empty:
                item = ()
            if item:
                batch.append(item)
            if batch and (len(batch) >= 50 or not item):
                try:
                    keys = ("time", "simulation_time", "qpos", "qvel", "state")
                    save_arrays(self.folder / f"{self.parts:06d}.npz",
                                dict(zip(keys, map(np.asarray, zip(*batch, strict=True)), strict=True)))
                    self.parts += 1
                except Exception as exc:
                    self.error = f"scene recording failed: {exc}"
                batch.clear()
            if item is None:
                return

    def close(self):
        if self.closed:
            return
        self.capture()
        self.closed = True
        self.pending.put(None)
        self.thread.join()
        save_arrays(self.output / "final-state.npz", dict(qpos=self.body.data.qpos, qvel=self.body.data.qvel))
        chunks = []
        for path in sorted(self.folder.glob("*.npz")):
            with np.load(path) as arrays:
                chunks.append({key: arrays[key] for key in arrays.files})
        if chunks:
            save_arrays(self.output / "replay.npz",
                        {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]})
        save_summary(self.folder / "complete.json", dict(parts=self.parts, error=self.error))


class _RecordedBody:
    """Keep scene sampling on the body's owner thread, after each completed read."""

    simulated = True

    def __init__(self, body, recording=None):
        self.body, self.recording = body, recording
        self.manifest = body.manifest

    def __getattr__(self, name):
        return getattr(self.body, name)

    def read(self):
        state = self.body.read()
        if self.recording is not None:
            self.recording.capture()
        return state

    def connect(self):
        return self.body.connect()

    def enable(self):
        return self.body.enable()

    def disable(self):
        return self.body.disable()

    def command(self, q, dq, gripper, gripper_v=0):
        return self.body.command(q, dq, gripper, gripper_v)

    def close(self):
        try:
            if self.recording is not None:
                try:
                    self.recording.close()
                except Exception as exc:
                    self.recording.error = f"scene recording could not finish: {exc}"
                    raise
        finally:
            self.body.close()


class SimulationSession:
    """Own one external MuJoCo episode, initially idle and unpowered.

    The caller constructs a simulated body with model/data and a reentrant lock, and
    calibrated cameras over that body. Output may contain setup assets, but no earlier
    session. Sources are declared explicitly, as in the recorded task launcher.
    """

    def __init__(self, body: Any, output: Path, *, cameras: dict, sources: dict[str, Path],
                 entrypoint: str, parameters: dict | None = None, context: dict | None = None,
                 world: World | None = None, clock=None):
        if not getattr(body, "simulated", False):
            raise ValueError("SimulationSession accepts simulated bodies only")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if any((self.output / name).exists() for name in ("flight", "task", "physics", "scene.mjb")):
            raise FileExistsError("use a fresh output directory for each simulation session")
        self.body = body
        self.closed = False
        self.daemon: Daemon | None = None
        self.record: Record | None = None
        recorded_body = _RecordedBody(body)
        try:
            self.kernel = Kernel(recorded_body,
                                 World.from_dict(world.to_dict()) if world is not None else World(),
                                 clock, run_dir=self.output / "flight")
        except BaseException:
            body.close()
            raise
        try:
            self.kernel.connect()
            self.kernel.tick()  # Materialize lazy simulators and obtain current, unpowered feedback.
            self.physics = MujocoRecording(body, self.output)
            recorded_body.recording = self.physics
            self.daemon = Daemon(self.kernel, port=0, cams=cameras,
                                 session=dict(mode="simulation", adapter=body.manifest.name,
                                              scene="external MuJoCo"))
            self.daemon.start()
            url = f"http://127.0.0.1:{self.daemon.http.server_address[1]}"
            folder = prepare(self.output / "task", sources, entrypoint=entrypoint,
                             parameters=parameters or {}, url=url)
            self.record = Record(folder)
            self.client = RecordedClient(url, self.record)
            self.client.begin()
            self.client.record(context=dict(execution="live caller with declared source snapshot", **(context or {})))
            save_summary(self.output / "flight/external-scene.json", dict(path="..", format="mujoco-v1"))
            save_summary(self.output / "cameras.json",
                         {name: cam.view.to_dict() for name, cam in cameras.items() if cam.view is not None})
        except BaseException:
            if self.daemon is not None:
                self.daemon.shutdown()
                self.daemon.http.server_close()
            else:
                self.kernel.close()
            raise

    def close(self, result: dict | None = None, *, exception: str | None = None) -> dict:
        """Finish after power is resolved. A refusal leaves the session available for recovery."""
        if self.closed:
            raise RuntimeError("session already closed")
        assert self.daemon is not None and self.record is not None
        status = self.client.status()
        if any(status.get(key) for key in ("enabled", "power_uncertain", "job", "queued")):
            raise RuntimeError("stop active work and release power before closing the simulation session")
        # The ordinary shutdown path owns release, stopping the loop and closing the body.
        reply = self.client.shutdown()
        self.daemon.http.server_close()
        self.closed = True
        final_error = reply.get("summary", {}).get("recording_error")
        if final_error:
            status["recording"]["error"] = final_error
        status["physics_recording"] = dict(path=str(self.physics.folder), error=self.physics.error)
        with suppress(RecordingError):
            self.record.append("end", status=status)
        saved = self.record.finish(result, status=status, exception=exception)
        return saved

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.physics.start

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if not self.closed:
            # Recovery remains usable even if recording failed. No implicit retreat motion.
            try:
                recovery = Client(self.client.url)
                recovery.stop("simulation session ended")
                recovery.release()
                self.close(exception=None if value is None else repr(value))
            except Exception as cleanup:
                if value is None:
                    raise
                # The task's own failure stays the reported exception; the session remains open for recovery.
                value.add_note(f"simulation session cleanup failed: {cleanup!r}")


def replay_scene(folder: Path, output: Path, *, fps: int, speed: float) -> Path:
    """Render saved full-scene states through a fixed initial camera; never step physics."""
    metadata = json.loads((folder / "physics/format.json").read_text())
    if metadata["version"] != 1:
        raise ValueError("unsupported scene recording version")
    cameras = json.loads((folder / "cameras.json").read_text())
    if not cameras:
        raise ValueError("this scene record has no replay camera")
    name, camera = next(iter(cameras.items()))
    view = View.from_dict(camera)
    scale = min(1., 800 / view.width, 600 / view.height)
    view = view.scaled(round(view.width * scale), round(view.height * scale))
    view = replace(view, T=np.asarray(metadata["base_to_world"]) @ view.T)
    with np.load(folder / "replay.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    elapsed = arrays["time"]
    if not len(elapsed):
        raise ValueError("this scene record has no physics samples")
    moments = np.arange(elapsed[0], elapsed[-1] + .5 * speed / fps, speed / fps)
    if len(moments) > 7200:
        raise ValueError("replay exceeds 7200 frames; increase --speed")
    model = mj.MjModel.from_binary_path(str(folder / "scene.mjb"))
    data = mj.MjData(model)
    renderer = CameraRenderer()
    frames = []
    try:
        for moment in moments:
            index = min(int(np.searchsorted(elapsed, moment)), len(elapsed) - 1)
            mj.mj_setState(model, data, arrays["state"][index], metadata["state_spec"])
            mj.mj_forward(model, data)
            frame = renderer.render(model, data, view)
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, frame.width, 50), fill=(246, 246, 246))
            label = (f"RECORDED SIMULATION  |  {speed:g}x  |  fixed camera: {name}\n"
                     f"wall {elapsed[index]:.1f}s  |  simulation {arrays['simulation_time'][index]:.1f}s")
            draw.text((10, 6), label, fill=(35, 40, 45), font=ImageFont.load_default(size=14))
            frames.append(frame)
    finally:
        renderer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
    return output.resolve()
