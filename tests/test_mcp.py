"""The MCP server: the CLI's verbs as tools, answering in the same short text (and pictures)."""
import asyncio
import sys

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError

from world_use.mcp_server import build


def test_mcp_rejects_original_arguments_before_the_sdk_can_drop_or_coerce_them(daemon):
    d, c = daemon
    server = build(c.url)

    async def session():
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert tools["run"].input_schema["additionalProperties"] is False
        requirements = tools["run"].input_schema["properties"]["requires"]["anyOf"][0]
        assert requirements["maxItems"] == 16
        evidence = tools["run"].input_schema["$defs"]["Requirement"]
        assert evidence["required"] == ["evidence", "max_age_s"]
        assert evidence["additionalProperties"] is False
        plane = tools["measure_pixels"].input_schema["$defs"]["Plane"]
        assert plane["required"] == ["box"] and plane["additionalProperties"] is False
        for extra in ({"require": []}, {"rehearse": "false"}, {"rehearse": None}, {"wait_s": True},
                      {"wait_s": float("nan")}, {"plan": '[{"do":"hold","seconds":0.01}]'},
                      {"requires": [{"evidence": "missing", "max_age_s": 1, "typo": True}]}):
            with pytest.raises(ToolError, match="invalid arguments"):
                await server.call_tool("run", {"plan": {"do": "hold", "seconds": 0.01}} | extra)
        with pytest.raises(ToolError, match="invalid arguments"):
            await server.call_tool("measure_pixels", {"frame": "missing", "point": [0, 0],
                "plane": {"box": [0, 0, 10, 10], "max_error_m": True}})
        assert not d.k.jobs
        c.box("glass", "fragile", [0.8, 0, 0.5], [0.1] * 3, dtau=0.1)
        original = d.k.world.boxes["glass"]
        for extra in ({"datu": 0.03}, {"dtau": True}):
            with pytest.raises(ToolError, match="invalid arguments"):
                await server.call_tool("add_box", dict(name="glass", kind="fragile", center=[0.8, 0, 0.5],
                                                      size=[0.2] * 3) | extra)
            assert d.k.world.boxes["glass"] is original
    asyncio.run(session())


def test_mcp_wire_requests_use_the_same_strict_arguments(daemon):
    from mcp import ClientSession, StdioServerParameters, stdio_client

    d, c = daemon
    params = StdioServerParameters(command=sys.executable, args=["-c",
        "import sys; from world_use.mcp_server import build; build(sys.argv[1]).run('stdio')", c.url])

    async def session():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools["run"].input_schema["additionalProperties"] is False
            for extra, field in (({"require": []}, "require"), ({"rehearse": "false"}, "rehearse"),
                                 ({"wait_s": True}, "wait_s")):
                result = await client.call_tool("run", {"plan": {"do": "hold", "seconds": 0.01}} | extra)
                assert result.is_error and field in result.content[0].text
            assert not d.k.jobs
            result = await client.call_tool("run", {"plan": {"do": "hold", "seconds": 0.01},
                                                    "rehearse": False, "wait_s": 5})
            assert not result.is_error and result.structured_content["status"] == "done"
            assert len(d.k.jobs) == 1
    asyncio.run(session())


@pytest.mark.usefixtures("file_camera")
def test_mcp_frames_measurements_and_crops_come_with_their_pictures(daemon):
    import json

    _, c = daemon
    server = build(c.url)

    async def session():
        captured = await server.call_tool("camera_frame", {"camera": "side"})
        frame = json.loads(captured.content[0].text)["frame"]
        assert any(item.type == "image" for item in captured.content)
        measured = await server.call_tool("measure_pixels", {"frame": frame, "point": [200, 200],
            "plane": {"box": [100, 100, 150, 150], "max_error_m": .001}})
        measurement = json.loads(measured.content[0].text)
        assert not measurement["valid"] and measurement["reason"] == "missing_depth"
        assert any(item.type == "image" for item in measured.content)
        with pytest.raises(ToolError, match="found no surface"):
            await server.call_tool("run", {"plan": {"do": "gripper", "aperture_mm": 65}, "rehearse": False,
                                          "requires": [{"evidence": measurement["id"], "max_age_s": 10}]})
        crop = (await server.call_tool("inspect_image", {"frame": frame, "crop": [10, 20, 50, 80],
                                                          "max_side": 600})).structured_content
        assert crop["size"] == [400, 600]
        assert crop["native_from_image"] @ np.array([200, 300, 1]) == pytest.approx([30, 50, 1])
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
        assert "idle" in r.content[0].text and "\n" not in r.content[0].text
        assert r.structured_content["power_uncertain"] is False
        r = await server.call_tool("run", {"plan": [{"do": "line", "up": 0.03, "duration": 1.0}], "wait_s": 0,
                                          "camera": "side"})
        assert "frame" not in r.structured_content and not any(item.type == "image" for item in r.content)
        assert "job 1 is still" in r.content[0].text and "job(job=1)" in r.content[0].text
        assert "wu " not in r.content[0].text                  # an MCP agent has tools, not a shell
        r = await server.call_tool("job", {"job": 1, "wait_s": 10})
        assert "job 1 done" in r.content[0].text
        r = await server.call_tool("job", {"job": 1, "wait_s": 0})
        assert r.structured_content["outcome"]["status"] == "done"
        with pytest.raises(ToolError, match="no step 'nope'"):
            await server.call_tool("help", {"step": "nope"})
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


@pytest.mark.usefixtures("file_camera")
def test_mcp_phase_camera_preserves_outcomes_and_never_resubmits_on_camera_failure(daemon):
    d, c = daemon
    server = build(c.url)

    async def session():
        r = await server.call_tool("run", {"plan": [{"do": "checkpoint", "ask": "Continue?"},
            {"do": "hold", "seconds": .01}], "rehearse": False, "camera": "side"})
        job = r.structured_content["id"]
        assert r.structured_content["status"] == "waiting" and any(i.type == "image" for i in r.content)
        frame = r.structured_content["frame"]["id"]
        assert c.frame(id=frame).id == frame
        r = await server.call_tool("job", {"job": job, "camera": "side"})
        assert any(i.type == "image" for i in r.content)
        assert c.frame(id=r.structured_content["frame"]["id"]).camera == "side"
        r = await server.call_tool("answer", {"job": job, "answer": "yes", "camera": "side"})
        assert r.structured_content["status"] == "done" and any(i.type == "image" for i in r.content)
        r = await server.call_tool("run", {"plan": {"do": "hold", "seconds": .01},
                                           "rehearse": False, "camera": "missing"})
        assert r.structured_content["status"] == "done" and r.structured_content["camera_error"]
        assert len(d.k.jobs) == 2
    asyncio.run(session())
