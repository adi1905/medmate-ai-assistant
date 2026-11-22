"""
server.py

MCP server for the Lab Reference Agent. Exposes read-only, rule-based
lookups over a static, curated reference dataset (data/tests.json,
data/reference_ranges.json). No LLM calls happen inside this server —
it is a pure data-lookup service, run as a separate process and reached
via MCP stdio transport.

Run directly for local testing:
    python server.py

Wire into LangGraph via langchain-mcp-adapters using a stdio client
pointed at this script.
"""

from mcp.server.fastmcp import FastMCP

import tools

mcp = FastMCP("lab-reference-server")


@mcp.tool()
def get_reference_range(test_name: str, demographic: str = "") -> dict:
    """
    Get the reference (normal) range for a lab test, optionally sliced by
    demographic.

    Args:
        test_name: Canonical test name or known alias (e.g. "Creatinine", "A1C").
        demographic: One of "adult", "adult_male", "adult_female", "pediatric".
            Leave empty if unknown or not applicable.

    Returns a dict with found/low/high/unit/source, or found=False with a
    message if the test or demographic isn't in the reference catalog.
    Never fabricates a range — only returns curated, sourced data.
    """
    dem = demographic.strip() if demographic else None
    return tools.get_reference_range(test_name, dem)


@mcp.tool()
def get_test_info(test_name: str) -> dict:
    """
    Get plain-language catalog info for a lab test: what it measures,
    its unit, category, and known aliases.

    Args:
        test_name: Canonical test name or known alias.

    Returns a dict with found/description/unit/category/aliases, or
    found=False with a message if the test isn't in the catalog.
    This is informational only — not diagnostic or treatment advice.
    """
    return tools.get_test_info(test_name)


@mcp.tool()
def list_available_tests() -> dict:
    """
    List all lab tests currently covered by the reference catalog.
    Useful for the calling agent to check coverage before querying,
    or to surface options to the user.
    """
    return tools.list_available_tests()


if __name__ == "__main__":
    # stdio transport — spawned as a subprocess by the MCP client
    mcp.run(transport="stdio")