"""The MCP server: the CLI's verbs as tools, answering in the same short text (and pictures)."""
import asyncio

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError

from world_use.mcp_server import build


@pytest.mark.usefixtures("file_camera")
def test_mcp_capture_measurement_and_evidence_refusal(daemon):
    import json

    _, c = daemon
    server = build(c.url)

    async def session():
        captured = await server.call_tool("camera_frame", {"camera": "side"})
        metadata = json.loads(captured.content[0].text)
        assert any(item.type == "image" for item in captured.content)
        measured = await server.call_tool("measure_pixels", {"frame": metadata["frame"], "point": [200, 200]})
        receipt = json.loads(measured.content[0].text)
        assert not receipt["valid"] and receipt["reason"] == "missing_depth"
        with pytest.raises(ToolError, match="no valid geometry"):
            await server.call_tool("run", {"plan": {"do": "gripper", "aperture_mm": 65}, "rehearse": False,
                                          "requires": [{"evidence": receipt["id"], "max_age_s": 10}]})
    asyncio.run(session())


@pytest.mark.usefixtures("file_camera")
def test_mcp_tools_drive_the_daemon(daemon):
    _, c = daemon
    server = build(c.url)

    async def session():
        names = {t.name for t in await server.list_tools()}
        assert {"card", "status", "run", "check", "answer", "look", "world", "add_box", "help", "stop",
                "policy", "job", "reset", "shutdown"} <= names
        r = await server.call_tool("policy", {})
        assert "power_uncertain" in r.content[0].text
        assert "world-use://policy" in {str(r.uri) for r in await server.list_resources()}
        r = await server.call_tool("status", {})
        assert "idle" in r.content[0].text
        assert "power_uncertain" in r.content[0].text
        r = await server.call_tool("run", {"plan": [{"do": "line", "up": 0.03, "duration": 1.0}], "wait_s": 0})
        assert "job 1" in r.content[0].text
        r = await server.call_tool("job", {"job": 1, "wait_s": 10})
        assert '"status": "done"' in r.content[0].text
        r = await server.call_tool("job", {"job": 1, "wait_s": 0})
        assert '"outcome"' in r.content[0].text
        r = await server.call_tool("reset", {})
        assert not r.is_error
        r = await server.call_tool("run", {"plan": {"do": "hold", "seconds": 0.1}, "wait_s": 10})
        assert "job 2 done" in r.content[0].text
        r = await server.call_tool("look", {"camera": "side"})
        assert {x.type for x in r.content} == {"text", "image"}
        # The SDK maps ToolError to is_error on the wire.
        with pytest.raises(ToolError, match="refused: unknown behavior"):
            await server.call_tool("run", {"plan": {"do": "teleport"}})
        with pytest.raises(ToolError, match="error:"):
            await server.call_tool("job", {"job": 99999})
        r = await server.call_tool("add_box", {"name": "tray", "kind": "surface", "center": [0.32, 0, 0.14],
                                               "size": [0.3, 0.4, 0.02]})
        assert r.content[0].text.startswith("surface 'tray'")
        r = await server.call_tool("add_box", {"name": "careful", "kind": "slow", "center": [0.3, 0, 0.3],
                                               "size": [0.1, 0.1, 0.1], "speed": 0.02})
        assert "speed=0.02" in r.content[0].text

    asyncio.run(session())
