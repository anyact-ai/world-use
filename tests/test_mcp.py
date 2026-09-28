"""The MCP server: the CLI's verbs as tools, answering in the same short text (and pictures)."""
import asyncio

import pytest

pytest.importorskip("mcp")

from world_use.mcp_server import build


def test_mcp_tools_drive_the_daemon(daemon):
    _, c = daemon
    server = build(c.url)

    async def session():
        names = {t.name for t in await server.list_tools()}
        assert {"card", "status", "run", "check", "answer", "look", "world", "add_box", "help", "stop"} <= names
        r = await server.call_tool("status", {})
        assert "idle" in r.content[0].text
        r = await server.call_tool("run", {"plan": [{"do": "line", "up": 0.03, "duration": 1.0}], "wait_s": 10})
        assert "job 1 done" in r.content[0].text
        r = await server.call_tool("look", {"camera": "side"})
        assert {x.type for x in r.content} == {"text", "image"}
        r = await server.call_tool("run", {"plan": {"do": "teleport"}})
        assert "refused: unknown behavior" in r.content[0].text
        r = await server.call_tool("add_box", {"name": "tray", "kind": "surface", "center": [0.32, 0, 0.14],
                                               "size": [0.3, 0.4, 0.02]})
        assert r.content[0].text.startswith("surface 'tray'")

    asyncio.run(session())
