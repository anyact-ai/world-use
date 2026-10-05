"""Run with a clean interpreter containing the wheel, outside the source checkout. Simulation only."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from PIL import Image

from world_use import Client, policy_text
from world_use.config import WORKCELLS
from world_use.mcp_server import build
from world_use.records import inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-camera", action="store_true",
                        help="test camera transport with a file on hosts without OpenGL (macOS CI)")
    args = parser.parse_args()
    assert "power_uncertain" in policy_text()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        workcell = "block"
        if args.file_camera:
            frame = root / "camera.png"
            Image.new("RGB", (800, 600), "gray").save(frame)
            cell = root / "workcell.toml"
            cell.write_text((WORKCELLS / "block.toml").read_text() +
                            '\n[[camera]]\nname = "side"\nmax_age_s = 300\n' +
                            f"path = {json.dumps(str(frame))}\n")
            workcell = str(cell)
        wu = [sys.executable, "-m", "world_use"]
        subprocess.run(wu + ["demo", "--out", str(root / "demo"), "--no-video"],
                       check=True, cwd=root, stdout=subprocess.DEVNULL)
        assert inspect(root / "demo")["closed"]
        moved = root / "moved-record"
        (root / "demo").rename(moved)
        subprocess.run(wu + ["view", str(moved), "--out", str(root / "demo.rrd")],
                       check=True, cwd=root, stdout=subprocess.DEVNULL)
        from rerun.chunk import RrdReader
        recording = RrdReader(root / "demo.rrd")
        assert recording.recordings() and recording.blueprints()
        assert any("visual_geometries" in c.entity_path for c in recording.stream())
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        # wu up and wu down as a person runs them: the address and record folder come from the environment.
        env = dict(os.environ, WORLD_USE_URL=url, WORLD_USE_RUNS=str(root / "runs"))
        subprocess.run(wu + ["up", "--workcell", workcell], check=True, cwd=root, env=env, stdout=subprocess.DEVNULL)
        c = Client(url)
        viewer = None
        try:
            assert c.status()["session"]["mode"] == "simulation"
            viewer_cwd = root / "viewer"
            viewer_cwd.mkdir()
            viewer = subprocess.Popen(wu + ["view", "--out", str(root / "live.rrd")], cwd=viewer_cwd, env=env,
                                      stdout=subprocess.DEVNULL)
            assert Path(c.look("side")["path"]).is_file()
            assert not c.status()["enabled"]
            c.enable()
            assert c.run({"do": "line", "up": .06}, wait=10)["status"] == "done"
            assert c.home_route([])["ok"]
            assert c.home(wait=10)["status"] == "done"
            c.release()
            assert not c.status()["enabled"]
            server = build(c.url)
            assert "job" in {t.name for t in asyncio.run(server.list_tools())}
            assert "world-use://policy" in {str(r.uri) for r in asyncio.run(server.list_resources())}
            subprocess.run(wu + ["down"], check=True, cwd=root, env=env, stdout=subprocess.DEVNULL)
            deadline = time.monotonic() + 10
            while c.alive():
                assert time.monotonic() < deadline, "the daemon still answers after wu down"
                time.sleep(.1)
            viewer.wait(timeout=20)
            assert viewer.returncode == 0
            live = RrdReader(root / "live.rrd")
            assert any(c.entity_path == "/observations/side" for c in live.stream())
        finally:
            if viewer is not None and viewer.poll() is None:
                viewer.terminate()
                viewer.wait(timeout=5)
            if c.alive():               # a check failed: bring the simulated arm home so the daemon can stop
                for step in (c.stop, lambda: c.home_route([]), lambda: c.home(wait=30), c.shutdown):
                    with suppress(Exception):
                        step()
    camera = "file camera" if args.file_camera else "MuJoCo camera"
    print(f"Installed wheel: demo, portable Rerun export, policy, {camera}, MCP, wu up, motion, home, wu down passed")


if __name__ == "__main__":
    main()
