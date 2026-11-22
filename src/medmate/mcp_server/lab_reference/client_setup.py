"""
client_setup.py

MCP client wiring for the Lab Reference Agent. Spawns lab_reference_server
as a subprocess over stdio and exposes its tools as LangChain tools that
can be bound directly to your Lab Reference sub-agent node.

Usage inside your LangGraph app (async context):

    from client_setup import get_lab_reference_tools

    async def build_lab_reference_agent():
        tools = await get_lab_reference_tools()
        # bind `tools` to your LLM / ToolNode as usual
        return tools
"""

import os
import sys
from mcp import StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")
# print(SERVER_SCRIPT)

server_params = StdioServerParameters(
    # Use the current Python interpreter for cross-platform compatibility
    command=sys.executable,
    args=[SERVER_SCRIPT],
)


async def get_lab_reference_tools():
    """
    Connects to the lab reference MCP server over stdio and returns a list
    of LangChain-compatible tools (get_reference_range, get_test_info,
    list_available_tests) ready to bind to the Lab Reference sub-agent.

    IMPORTANT: this helper opens the stdio session, loads tools, and closes
    the session before returning. The returned LangChain tool objects will
    then fail if invoked later, because their underlying session is closed
    (ClosedResourceError) — this mirrors real MCP session lifecycle rules.
    Use this helper only to inspect available tools/schemas at startup.
    For actually invoking tools in a running LangGraph app, use
    LabReferenceMCPClient below, which keeps the session open for the
    lifetime of the app.
    """
    async with stdio_client(server_params) as (read, write):
        from mcp import ClientSession

        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools


# ---------------------------------------------------------------------------
# Persistent-session pattern (recommended for a long-running LangGraph app):
# open the session once at startup and keep it alive, rather than
# reconnecting on every tool call.
# ---------------------------------------------------------------------------

class LabReferenceMCPClient:
    """
    Holds a long-lived MCP session so the Lab Reference sub-agent doesn't
    pay subprocess-spawn cost on every request. Call `.start()` once at
    app startup and `.stop()` on shutdown.
    """

    def __init__(self):
        self._stdio_ctx = None
        self._session_ctx = None
        self.session = None
        self.tools = None

    async def start(self):
        self._stdio_ctx = stdio_client(server_params)
        read, write = await self._stdio_ctx.__aenter__()

        from mcp import ClientSession

        self._session_ctx = ClientSession(read, write)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()

        self.tools = await load_mcp_tools(self.session)
        return self.tools

    async def stop(self):
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)