"""
FastMCP Server Entry Point (mcp_server/server.py).
Fulfills Phase 3 of HDB-Compass Architecture (Page 14).
Mounts Member 2's Spatial & OneMap tools alongside other domain toolsets.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP
from agents.schemas import QueryResaleBlocksOutput
from mcp_server.market_sql_tools import query_resale_blocks_by_criteria
from mcp_server.spatial_onemap_tools import (
    mcp as spatial_mcp,
    verify_primary_school_distance,
    filter_blocks_by_amenity_proximity,
    compute_dual_transit_commute,
    sync_interactive_map_ui,
)

# Unified FastMCP Server for HDB-Compass
mcp = FastMCP("HDB-Compass-MCP-Server")

# Re-register Member 2 Tools on the unified server
mcp.add_tool(verify_primary_school_distance)
mcp.add_tool(filter_blocks_by_amenity_proximity)
mcp.add_tool(compute_dual_transit_commute)
mcp.add_tool(sync_interactive_map_ui)
mcp.tool(output_schema=QueryResaleBlocksOutput.model_json_schema())(query_resale_blocks_by_criteria)

if __name__ == "__main__":
    print("[HDB-Compass] Starting FastMCP Server on stdio...", file=sys.stderr)
    mcp.run()
