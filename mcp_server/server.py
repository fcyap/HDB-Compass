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

from agents.schemas import (
    TOOL_INPUT_MODEL,
    CalculateBsdAndUpfrontOutlayOutput,
    CalculateCpfHousingGrantsOutput,
    CalculateMaximumAffordabilityOutput,
    ComputeTownPriceTimeseriesOutput,
    QueryComparableResaleOutput,
    QueryResaleBlocksOutput,
    FindSchoolsNearAddressOutput,
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
from mcp_server.school_lookup_tools import find_primary_schools_near_address

# Unified FastMCP Server for HDB-Compass
mcp = FastMCP("HDB-Compass-MCP-Server")

# Re-register Member 2 Tools on the unified server
mcp.add_tool(verify_primary_school_distance)
mcp.add_tool(filter_blocks_by_amenity_proximity)
mcp.add_tool(compute_dual_transit_commute)
mcp.add_tool(sync_interactive_map_ui)
school_lookup = mcp.tool(output_schema=FindSchoolsNearAddressOutput.model_json_schema())(
    find_primary_schools_near_address
)
school_lookup.parameters = TOOL_INPUT_MODEL["find_primary_schools_near_address"].model_json_schema()
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

# Research tools (Member 1; location tools shared with Member 2). The Pydantic
# input models are advertised so MCP clients see the same contract as the agents.
from agents import research_schemas as R  # noqa: E402
from agents.schemas import CompareBtoLaunchVsResaleOutput  # noqa: E402
from mcp_server.area_resale_tools import (  # noqa: E402
    find_resale_blocks_near,
    get_area_price_trend,
    get_area_resale_statistics,
)
from mcp_server.bto_tools import compare_bto_launch_vs_resale  # noqa: E402
from mcp_server.location_tools import get_nearby_amenities, resolve_location  # noqa: E402
from mcp_server.market_research_tools import (  # noqa: E402
    assess_resale_price,
    compare_resale_options,
    get_block_transaction_history,
    get_market_snapshot,
    rank_towns_by_value,
)

for function, output_model in (
    (compare_bto_launch_vs_resale, CompareBtoLaunchVsResaleOutput),
    (get_block_transaction_history, R.GetBlockTransactionHistoryOutput),
    (get_market_snapshot, R.GetMarketSnapshotOutput),
    (rank_towns_by_value, R.RankTownsByValueOutput),
    (compare_resale_options, R.CompareResaleOptionsOutput),
    (assess_resale_price, R.AssessResalePriceOutput),
    (find_resale_blocks_near, R.FindResaleBlocksNearOutput),
    (get_area_resale_statistics, R.GetAreaResaleStatisticsOutput),
    (get_area_price_trend, R.GetAreaPriceTrendOutput),
    (resolve_location, R.ResolveLocationOutput),
    (get_nearby_amenities, R.GetNearbyAmenitiesOutput),
):
    registered = mcp.tool(output_schema=output_model.model_json_schema())(function)
    registered.parameters = TOOL_INPUT_MODEL[function.__name__].model_json_schema()

if __name__ == "__main__":
    print("[HDB-Compass] Starting FastMCP Server on stdio...", file=sys.stderr)
    mcp.run()
