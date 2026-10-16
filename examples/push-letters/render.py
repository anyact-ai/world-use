"""Render the recorded physics, with an overhead inset and explicit playback speed."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from world_use.bodies.mujoco_render import CameraRenderer
from world_use.cameras import View


def render(folder, speed=2., fps=30):
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("playback speed must be positive and finite")
    folder = Path(folder)
    replay = np.load(folder / "replay.npz")
    times = replay["simulation_time"] if "simulation_time" in replay else replay["time"]
    model = mj.MjModel.from_binary_path(str(folder / "scene.mjb"))
    data = mj.MjData(model)
    views = json.loads((folder / "views.json").read_text())
    top = View.from_dict(views["overhead"]).scaled(620, 620)
    main = View.look_at([1.40, -.95, 1.05], [.25, 0, .22], fov_deg=48, size=(1280, 880))
    renderer, top_renderer = CameraRenderer(), CameraRenderer()
    actions = json.loads((folder / "actions.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    font = ImageFont.load_default(size=25)
    small = ImageFont.load_default(size=19)
    width, height = 1920, 1080
    output = folder / "demo.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for time in np.arange(times[0], times[-1], speed/fps):
            index = int(np.argmin(np.abs(times-time)))
            data.qpos[:] = replay["qpos"][index]
            data.qvel[:] = replay["qvel"][index]
            mj.mj_forward(model, data)
            canvas = Image.new("RGB", (width, height), "#f4f2ec")
            canvas.paste(renderer.render(model, data, main), (0, 125))
            canvas.paste(top_renderer.render(model, data, top), (1290, 220))
            draw = ImageDraw.Draw(canvas)
            draw.text((36, 32), "WORLD USE  /  PUSH LETTERS", font=font, fill="#23282d")
            draw.text((36, 76), f"MuJoCo simulation · camera-feedback program · {speed:g}x continuous replay",
                      font=small, fill="#565b60")
            active = next((a for a in reversed(actions) if a.get("start_s", float("inf")) <= time), {})
            draw.text((1320, 160), "OVERHEAD REPLAY", font=small, fill="#565b60")
            label = active.get("label", "Initial scene")
            if label.startswith("letter_"):
                label = label.split("_", 2)[2].replace(" push ", " · push ")
            draw.text((1320, 890), label, font=font, fill="#23282d")
            draw.text((1320, 935), f"Simulation time  {time:.1f} s", font=small, fill="#565b60")
            draw.text((36, 1030), "Codex-authored policy · RGB-D + known glyph geometry · no runtime model calls",
                      font=small, fill="#565b60")
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        save = dict(video=output.name, speed=speed, fps=fps, continuous=True,
                    source="recorded qpos/qvel; original simulation model and materials",
                    audit_passed=result["success"])
        (folder / "render.json").write_text(json.dumps(save, indent=2)+"\n")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        renderer.close()
        top_renderer.close()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--speed", type=float, default=2.)
    args = parser.parse_args()
    print(render(args.folder, args.speed))
