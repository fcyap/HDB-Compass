"""
FastMCP Server Entry Point (mcp_server/server.py).
Fulfills Phase 3 of HDB-Compass Architecture (Page 14).
Mounts Member 2's Spatial & OneMap tools alongside other domain toolsets.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP

from agents.schemas import (
    TOOL_INPUT_MODEL,
    CalculateBsdAndUpfrontOutlayOutput,
    CalculateCpfHousingGrantsOutput,
    CalculateMaximumAffordabilityOutput,
    ComputeTownPriceTimeseriesOutput,
    QueryComparableResaleOutput,
    QueryResaleBlocksOutput,
    CompareBtoLaunchVsResaleOutput,
)
from mcp_server.financial_calc_tools import (
    calculate_bsd_and_upfront_outlay,
    calculate_cpf_housing_grants,
    calculate_maximum_affordability,
)
from mcp_server.market_sql_tools import (
    compute_town_price_timeseries,
    query_comparable_resale_transactions,
    query_resale_blocks_by_criteria,
)
from mcp_server.spatial_onemap_tools import (
    compute_dual_transit_commute,
    filter_blocks_by_amenity_proximity,
    sync_interactive_map_ui,
    verify_primary_school_distance,
)
from mcp_server.policy_rag_tools import retrieve_hdb_policy_knowledge

# Unified FastMCP Server for HDB-Compass
mcp = FastMCP("HDB-Compass-MCP-Server")

# Re-register Member 2 Tools on the unified server
mcp.add_tool(verify_primary_school_distance)
mcp.add_tool(filter_blocks_by_amenity_proximity)
mcp.add_tool(compute_dual_transit_commute)
mcp.add_tool(sync_interactive_map_ui)
mcp.tool(output_schema=QueryResaleBlocksOutput.model_json_schema())(
    query_resale_blocks_by_criteria
)
mcp.tool(output_schema=ComputeTownPriceTimeseriesOutput.model_json_schema())(
    compute_town_price_timeseries
)
comparable_tool = mcp.tool(output_schema=QueryComparableResaleOutput.model_json_schema())(
    query_comparable_resale_transactions
)
comparable_tool.parameters = TOOL_INPUT_MODEL["query_comparable_resale_transactions"].model_json_schema()

for function, output_model in (
    (calculate_cpf_housing_grants, CalculateCpfHousingGrantsOutput),
    (calculate_maximum_affordability, CalculateMaximumAffordabilityOutput),
    (calculate_bsd_and_upfront_outlay, CalculateBsdAndUpfrontOutlayOutput),
):
    registered = mcp.tool(output_schema=output_model.model_json_schema())(function)
    registered.parameters = TOOL_INPUT_MODEL[function.__name__].model_json_schema()

# Re-register Member 3 Policy RAG Tool on the unified server
mcp.add_tool(retrieve_hdb_policy_knowledge)
# compare_bto_launch_vs_resale is a core tool, now on real launch data (mcp_server/bto_tools.py)
from mcp_server.bto_tools import compare_bto_launch_vs_resale  # noqa: E402

mcp.tool(output_schema=CompareBtoLaunchVsResaleOutput.model_json_schema())(compare_bto_launch_vs_resale)

# Add-on packs (agents/addons: research tools, school lookup): each tool with its declared output schema, and its
# Pydantic input model advertised so MCP clients see the same contract as the agents.
from agents import addons  # noqa: E402

for tool in addons.all_addons():
    module_name, attr = tool.impl
    function = getattr(importlib.import_module(module_name), attr)
    registered = mcp.tool(output_schema=addons.output_model(tool.name).model_json_schema())(function)
    registered.parameters = TOOL_INPUT_MODEL[tool.name].model_json_schema()

if __name__ == "__main__":
    print("[HDB-Compass] Starting FastMCP Server on stdio...", file=sys.stderr)
    mcp.run()
