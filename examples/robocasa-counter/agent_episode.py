"""Interactive recording of an agent choosing world-use MCP calls, one call at a time.

Start with python -i. `act` logs actual requests/replies and saves every returned
image. There is no task policy or prewritten motion sequence in this recorder.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

from adapter import PandaBody

from world_use.cameras import SimCamera, View
from world_use.mcp_server import build
from world_use.simulation import SimulationSession
from world_use.vision_worker import TrackerProcess


class Episode:
    def __init__(self, output, scene, *, park_left_m=.20, views=None, task=None, setup=None,
                 camera_factory=None, sensors=None, sources=None, entrypoint="counter/agent_episode.py"):
        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=False)
        self.log = []
        self.body = PandaBody(self.out, scene, park_left_m=park_left_m)
        camera = SimCamera("counter", self.body.lens(), self.body)
        side = SimCamera("side", View.look_at([.05, 1.05, .75], [.43, .10, .28],
                                             fov_deg=52, size=(1024, 768)), self.body)
        cameras = {"counter": camera, "side": side} if views is None else {
            name: SimCamera(name, view, self.body) for name, view in views.items()}
        if camera_factory is not None:
            cameras = camera_factory(self.body)
        declared = {"counter/"+p.name: p for p in Path(__file__).parent.glob("*.py")}
        declared.update(sources or {})
        self.session = SimulationSession(self.body, self.out, cameras=cameras,
            sources=declared, entrypoint=entrypoint, parameters=dict(scene=str(scene)))
        self.k, self.daemon, self.client = self.session.kernel, self.session.daemon, self.session.client
        try:
            self.k.set_home_route([], note="Initial parked Panda pose")
            self.tracker = TrackerProcess(device="cpu", startup_timeout_s=240)
            self.server = build(tracker=self.tracker, client=self.client)
            self.client.record(context={
                "agent": "Codex, current development conversation",
                "role": "Interactive model-chosen tool calls from observations; not a prewritten controller",
                "prior_context": "Agent configured and inspected the upstream RoboCasa scene",
                "task": task or "Put the mug in the sink, verify with RGB-D after release, then power off",
                "sensors": sensors or "Fixed simulated RGB-D cameras; EdgeTAM CPU",
                "setup": setup or "0.75-scale cup; mobile base parked 0.20 m left before the session",
            })
        except BaseException as error:
            try:
                self.session.close(exception=repr(error))
            finally:
                if hasattr(self, "tracker"):
                    self.tracker.close()
            raise
        print("READY: act(tool, intent='short user-visible action description', **arguments)", flush=True)

    def stamp(self):
        return self.session.elapsed_s

    def act(self, tool, *, intent, **arguments):
        index = len(self.log)
        entry = dict(index=index, tool=tool, intent=intent, arguments=arguments, start=self.stamp())
        self.log.append(entry)
        try:
            with self.session.record.phase(intent, context=dict(tool=tool, arguments=arguments)) as phase:
                entry["phase"] = phase
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
        try:
            saved = self.session.close(dict(verdict="inconclusive", task_result=result))
            error = saved["recording_error"] or saved["flight_recording_error"] or self.session.physics.error
            (self.out / "final.json").write_text(json.dumps(
                dict(torque_off=True, result=result, recording_error=error), indent=2))
            if error:
                raise RuntimeError(f"Episode recording incomplete: {error}")
        finally:
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
