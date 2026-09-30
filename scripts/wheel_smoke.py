"""Run with a clean interpreter containing the wheel, outside the source checkout. Simulation only."""
from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from world_use import policy_text
from world_use.client import Client
from world_use.mcp_server import build
from world_use.records import inspect


def main():
    assert "power_uncertain" in policy_text()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        subprocess.run([sys.executable, "-m", "world_use", "demo", "--out", str(root / "demo"), "--no-video"],
                       check=True, cwd=root, stdout=subprocess.DEVNULL)
        assert inspect(root / "demo")["closed"]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with open(root / "daemon.log", "w+") as log:
            p = subprocess.Popen([sys.executable, "-m", "world_use.daemon", "--workcell", "block", "--port",
                                  str(port), "--runs", str(root / "runs")], cwd=root,
                                 stdout=log, stderr=subprocess.STDOUT)
            c = Client(f"http://127.0.0.1:{port}")
            try:
                deadline = time.monotonic() + 20
                while not c.alive():
                    if p.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    time.sleep(.05)
                assert c.status()["session"]["mode"] == "simulation"
                assert Path(c.look("side")["path"]).is_file()
                c.enable()
                assert c.run({"do": "line", "up": .06}, wait=10)["status"] == "done"
                c.home_route([])
                assert c.home(wait=10)["status"] == "done"
                c.release()
                assert not c.status()["enabled"]
                server = build(c.url)
                assert "job" in {t.name for t in asyncio.run(server.list_tools())}
                assert "world-use://policy" in {str(r.uri) for r in asyncio.run(server.list_resources())}
                c.shutdown()
                p.wait(timeout=10)
                assert p.returncode == 0
            finally:
                if p.poll() is None:
                    p.kill()                # this script only ever creates a simulated daemon
                    p.wait(timeout=5)
    print("Installed wheel: demo, policy, camera, MCP, checked motion, home and shutdown passed")


if __name__ == "__main__":
    main()
