"""Run a camera-feedback pushing experiment and retain the physics evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from adapter import PandaPusher
from audit import audit
from policy import angle_difference, estimate, next_push

from world_use.cameras import SimCamera, View
from world_use.simulation import SimulationSession


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, data):
    path.write_text(json.dumps(data, default=jsonable, indent=2) + "\n")


def run(scene, output, *, max_pushes=20, feedback="camera", preview=False, model_file="scene.xml"):
    scene, output = Path(scene), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = json.loads((scene / "task.json").read_text())
    save(output / "task.json", task)
    body = PandaPusher(scene, output, model_file=model_file)
    overhead = View.look_at([.49, 0, 1.05], [.49, 0, 0], fov_deg=46, size=(1000, 1000), up=(1, 0, 0))
    overview = View.look_at([1.12, -.85, .9], [.30, 0, .12], size=(1200, 800))
    camera = SimCamera("overhead", overhead, body)
    save(output / "views.json", dict(overhead=overhead.to_dict(), overview=overview.to_dict()))
    session = SimulationSession(body, output, cameras={"overhead": camera},
        sources={p.name: p for p in Path(__file__).parent.glob("*.py")}, entrypoint="episode.py",
        parameters=dict(scene=str(scene.resolve()), model_file=model_file, max_pushes=max_pushes, feedback=feedback),
        context=dict(task="Push each letter onto its matching target; then withdraw and release power",
                     policy="Codex-authored analytic feedback program; no model API at runtime",
                     inputs=feedback, builder_context=True, setup=str(scene.resolve()),
                     adaptation="RSS-inspired rigid MuJoCo task; not an exact reproduction"))
    with session:
        client = session.client
        actions, observations = [], []
        start = time.monotonic()
        wide = len(task["letters"]) > 3
        park = [.26, -.32, .16] if wide else [.30, -.27, .16]
        transit = [{"do": "move_to", "to": [.43, 0, .14]}] if wide else []

        def execute(plan, label):
            action = dict(label=label, plan=plan, start_s=body.data.time)
            actions.append(action)
            outcome = client.run(plan, wait=120, check=False)
            while outcome["status"] in ("queued", "running"):
                outcome = client.job(outcome["id"], wait=120)
            action.update(end_s=body.data.time, outcome=outcome)
            save(output / "actions.json", actions)
            if outcome["status"] != "done":
                raise RuntimeError(f"{label}: {outcome}")

        def observe(i):
            item = task["letters"][i]
            if feedback == "state":
                obj = body.data.body(item["name"])
                return dict(xy=obj.xpos[:2].copy(), yaw=float(np.arctan2(obj.xmat[3], obj.xmat[0])))
            frame = client.frame("overhead", depth=True)
            index = len(observations)
            frame.image.save(output / f"observation-{index:03d}.png")
            result = estimate(frame, item["letter"], [x["color"] for x in task["letters"]], i)
            measurement = result.pop("measurement")
            np.savez_compressed(output / f"observation-{index:03d}.npz", depth=frame.depth,
                                points=measurement.points, pixels=measurement.pixels)
            observations.append(dict(index=index, item=item["name"], time_s=body.data.time, **result))
            save(output / "observations.json", observations)
            client.record(context=json.loads(json.dumps(dict(letter_pose=observations[-1]), default=jsonable)))
            return result

        error = None
        try:
            body.render(overview).save(output / "initial.png")
            body.render(overhead).save(output / "initial-overhead.png")
            if preview:
                return dict(preview=str(output))
            client.enable()
            execute([{"do": "move_to", "to": park}], "Clear the camera view")
            for i, item in enumerate(task["letters"]):
                for attempt in range(max_pushes):
                    pose = observe(i)
                    push = session.record.call(next_push, item["letter"], pose, item["target_xy"], item["target_yaw"])
                    xy_error = np.linalg.norm(pose["xy"]-item["target_xy"])
                    yaw_error = np.degrees(angle_difference(item["target_yaw"], pose["yaw"]))
                    print(f"{item['name']} {attempt}: {xy_error*1000:.1f} mm, {yaw_error:.1f} deg", flush=True)
                    if push is None:
                        break
                    p, q = push["start"], push["end"]
                    execute(transit + [{"do": "move_to", "to": [*p, .10]},
                             {"do": "move_to", "to": [*p, .003], "speed": .3},
                             {"do": "move_to", "to": [*q, .003], "speed": .15},
                             {"do": "line", "up": .10}] + transit + [{"do": "move_to", "to": park},
                             {"do": "hold", "seconds": .3}], f"{item['name']} push {attempt+1}")
                    actions[-1]["decision"] = push
                else:
                    raise RuntimeError(f"{item['name']} exhausted {max_pushes} pushes")
            execute([{"do": "hold", "seconds": 2}], "Verify settled letters")
        except BaseException as exc:
            error = repr(exc)
            print("STOP:", error, flush=True)
        finally:
            state = client.status()
            if state.get("job") or state.get("queued"):
                client.stop("pushing episode ended")
            if state["enabled"] and not state["faulted"] and session.kernel.tool[2, 3] < .09:
                execute([{"do": "line", "up": .10}], "Recovery lift")
            client.release()
            time.sleep(2.1)  # The disabled robot is braked; let released letters settle.
            state = client.status()
            result = dict(error=error, feedback=feedback, wall_s=time.monotonic()-start,
                          simulation_s=body.data.time, torque_off=not state["enabled"] and not state["power_uncertain"],
                          max_penetration_m=body.max_penetration, pusher_contacts=body.contact_counts,
                          worst_contact=body.worst_contact,
                          unexpected_contacts=body.unexpected_contacts,
                          solver_warnings={str(i): int(w.number) for i, w in enumerate(body.data.warning) if w.number},
                          letters=[])
            for item in task["letters"]:
                obj = body.data.body(item["name"])
                angle = float(np.arctan2(obj.xmat[3], obj.xmat[0]))
                yaw_error = angle_difference(angle, item["target_yaw"])
                if item["letter"] == "O":
                    yaw_error = (yaw_error+np.pi/2) % np.pi - np.pi/2
                result["letters"].append(dict(name=item["name"], xy=obj.xpos[:2].tolist(),
                                              position_error_m=float(np.linalg.norm(obj.xpos[:2]-item["target_xy"])),
                                              yaw_error_deg=float(np.degrees(yaw_error)), height_m=float(obj.xpos[2])))
            result["success"] = bool(error is None and all(
                x["position_error_m"] < .005 and abs(x["yaw_error_deg"]) < 5 for x in result["letters"]))
            save(output / "actions.json", actions)
            save(output / "result.json", result)
            body.render(overview).save(output / "final.png")
            body.render(overhead).save(output / "final-overhead.png")
            saved = session.close(dict(verdict="inconclusive", phase="awaiting independent physical audit"),
                                  exception=error)
            result["recording_error"] = (saved["recording_error"] or saved["flight_recording_error"]
                                         or session.physics.error)
            if result["recording_error"]:
                result["error"] = result["error"] or result["recording_error"]
            save(output / "result.json", result)
        result["audit"] = audit(output)
        result["success"] = result["audit"]["passed"]
        save(output / "result.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pushes", type=int, default=20)
    parser.add_argument("--feedback", choices=["camera", "state"], default="camera")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    result = run(args.scene, args.output, max_pushes=args.max_pushes, feedback=args.feedback, preview=args.preview)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if args.preview or result["success"] else 1)
