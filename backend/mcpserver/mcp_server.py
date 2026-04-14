"""CodeSpectra MCP server — Claude Code integration."""
from mcp.server.fastmcp import FastMCP
from .tools import setup, retrieve, agentic


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("CodeSpectra")
    setup.register(mcp)
    retrieve.register(mcp)
    agentic.register(mcp)
    return mcp
