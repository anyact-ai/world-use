"""Interactive recording of an agent choosing world-use MCP calls, one call at a time.

Start with python -i. `act` logs actual requests/replies and saves every returned
image. There is no task policy or prewritten motion sequence in this recorder.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from adapter import PandaBody

from world_use import Kernel, RealClock, World
from world_use.cameras import SimCamera, View
from world_use.client import Client
from world_use.daemon import Daemon
from world_use.mcp_server import build
from world_use.vision_worker import TrackerProcess


class Episode:
    def __init__(self, output, scene, *, park_left_m=.20, views=None, task=None, setup=None,
                 camera_factory=None, sensors=None):
        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        self.log = []
        self.samples = []
        self.stop = threading.Event()
        self.tracker = TrackerProcess(device="cpu", startup_timeout_s=240)
        self.body = PandaBody(self.out, scene, park_left_m=park_left_m)
        self.k = Kernel(self.body, World(), RealClock(100), run_dir=self.out / "flight")
        self.k.connect()
        self.k.set_home_route([], note="Initial parked Panda pose")
        mujoco.mj_saveModel(self.body.model, str(self.out / "scene.mjb"), None)
        camera = SimCamera("counter", self.body.lens(), self.body)
        side = SimCamera("side", View.look_at([.05, 1.05, .75], [.43, .10, .28],
                                             fov_deg=52, size=(1024, 768)), self.body)
        cameras = {"counter": camera, "side": side} if views is None else {
            name: SimCamera(name, view, self.body) for name, view in views.items()}
        if camera_factory is not None:
            cameras = camera_factory(self.body)
        (self.out / 'cameras.json').write_text(json.dumps({
            name: cam.view.to_dict() for name, cam in cameras.items()}, indent=2))
        self.daemon = Daemon(self.k, port=0, cams=cameras,
                             session={"adapter": "RoboCasa Panda (simulation)"}, config={})
        self.daemon.start()
        self.client = Client(f"http://127.0.0.1:{self.daemon.http.server_address[1]}")
        self.server = build(self.client.url, tracker=self.tracker)
        self.client.record(context={
            "agent": "Codex, current development conversation",
            "role": "Interactive model-chosen tool calls from observations; not a prewritten controller",
            "prior_context": "Agent configured and inspected the upstream RoboCasa scene",
            "task": task or "Put the mug in the sink, verify with RGB-D after release, then power off",
            "sensors": sensors or "Fixed simulated RGB-D cameras; EdgeTAM CPU",
            "setup": setup or "0.75-scale cup; mobile base parked 0.20 m left before the session",
        })
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        print("READY: act(tool, intent='short user-visible action description', **arguments)", flush=True)

    def stamp(self):
        return time.monotonic() - self.start

    def sample(self):
        while not self.stop.wait(1 / 30):
            with self.body.lock:
                if self.body.data is not None:
                    self.samples.append((self.stamp(), self.body.data.time, self.body.data.qpos.copy()))

    def act(self, tool, *, intent, **arguments):
        index = len(self.log)
        entry = dict(index=index, tool=tool, intent=intent, arguments=arguments, start=self.stamp())
        self.log.append(entry)
        try:
            reply = asyncio.run(self.server.call_tool(tool, arguments))
            entry["result"] = reply.structured_content
            entry["text"] = [c.text for c in reply.content if c.type == "text"]
            entry["images"] = []
            for i, content in enumerate(reply.content):
                if content.type == "image":
                    path = self.out / f"call-{index:02d}-{tool}-{i}.png"
                    path.write_bytes(base64.b64decode(content.data))
                    entry["images"].append(str(path.resolve()))
            print(json.dumps(dict(result=entry["result"], images=entry["images"]), indent=2), flush=True)
            if entry["result"] is None:
                print("\n".join(entry["text"]), flush=True)
            return entry["result"]
        except Exception as error:
            entry["error"] = str(error)
            print(json.dumps(dict(error=str(error))), flush=True)
            return None
        finally:
            entry["end"] = self.stamp()
            (self.out / "calls.json").write_text(json.dumps(self.log, indent=2) + "\n")

    def finish(self, result="Not yet verified"):
        state = self.client.status()
        if state["enabled"] or state["power_uncertain"]:
            raise RuntimeError("Go home and release before finishing")
        self.stop.set()
        self.thread.join()
        np.savez_compressed(self.out / "replay.npz", time=[s[0] for s in self.samples],
                            simulation_time=[s[1] for s in self.samples], qpos=[s[2] for s in self.samples])
        (self.out / "final.json").write_text(json.dumps(
            dict(torque_off=True, result=result), indent=2))
        self.act("shutdown", intent="Save the record and close the simulated robot")
        self.tracker.close()
        print("SAVED", self.out.resolve(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/robocasa-agent-06')
    parser.add_argument('--scene', type=Path, default=Path('runs/robocasa-small-cup/scene'))
    args = parser.parse_args()
    episode = Episode(args.output, args.scene)
    act = episode.act
    finish = episode.finish
