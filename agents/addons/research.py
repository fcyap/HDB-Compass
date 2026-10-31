"""
Research tools pack (Member 1; the two location tools shared with Member 2).

Questions the original block filter could not answer: where a place is, what is near it, what a block has sold for,
how the market stands now, which town is cheapest, how named options compare, whether an asking price is fair, and
prices around a place. Intents are read by agents/research_intents.py; this file only declares the tools.
"""
from __future__ import annotations

from typing import Any, Dict

from agents.addons import ToolAddon, ToolPack

R = "agents.research_schemas"


def _place(c: Dict[str, Any]) -> bool:
    return bool(c.get("addresses") or c.get("map_pin"))


def _area(c: Dict[str, Any]) -> bool:
    return bool(c.get("wants_area_statistics")) and _place(c)


def _plan(agent: str, c: Dict[str, Any], query: str):
    from agents import research_intents as ri   # noqa: PLC0415 — research_intents imports agents.schemas
    return ri.research_plan(agent, c, query)


TOOLS = (
    ToolAddon("get_market_snapshot", ("mcp_server.market_research_tools", "get_market_snapshot"), "market_sql",
              f"{R}:GetMarketSnapshotInput", f"{R}:GetMarketSnapshotOutput",
              lambda c: bool(c.get("wants_market_snapshot") or (c.get("wants_price_trend") and not c.get("towns"))),
              "how the resale market or prices stand now", "The current market snapshot",
              "How is the HDB resale market doing now?"),
    ToolAddon("rank_towns_by_value", ("mcp_server.market_research_tools", "rank_towns_by_value"), "market_sql",
              f"{R}:RankTownsByValueInput", f"{R}:RankTownsByValueOutput",
              lambda c: bool(c.get("wants_town_ranking")),
              "which town is cheapest or best value for a flat type", "The town ranking",
              "Which towns are cheapest for a 4-room resale flat?"),
    ToolAddon("get_block_transaction_history", ("mcp_server.market_research_tools", "get_block_transaction_history"),
              "market_sql", f"{R}:GetBlockTransactionHistoryInput", f"{R}:GetBlockTransactionHistoryOutput",
              lambda c: bool(c.get("wants_block_history")) and _place(c),
              "a named block's past sales (also takes area or block_keys)", "Past sales of the block",
              "What did flats at Blk 1 Bishan St 1 sell for recently?"),
    ToolAddon("assess_resale_price", ("mcp_server.market_research_tools", "assess_resale_price"), "market_sql",
              f"{R}:AssessResalePriceInput", f"{R}:AssessResalePriceOutput",
              lambda c: bool(c.get("wants_price_assessment")) and _place(c),
              "whether an asking price for a named block is in line with nearby sales", "The asking-price check",
              "Is $700,000 a fair price for a 4-room at Blk 1 Bishan St 1?"),
    ToolAddon("compare_resale_options", ("mcp_server.market_research_tools", "compare_resale_options"), "market_sql",
              f"{R}:CompareResaleOptionsInput", f"{R}:CompareResaleOptionsOutput",
              lambda c: bool(c.get("wants_option_comparison")),
              "two or more named blocks side by side", "The comparison of the named blocks",
              "Compare Blk 1 Bishan St 1 vs Blk 2 Bishan St 1 for a 4-room"),
    ToolAddon("find_resale_blocks_near", ("mcp_server.area_resale_tools", "find_resale_blocks_near"), "market_sql",
              f"{R}:FindResaleBlocksNearInput", f"{R}:FindResaleBlocksNearOutput", _area,
              "which blocks are around a place and each block's median (prices AROUND a place, not one block)",
              "Blocks around the place", "What are 4-room resale prices within 200m of Blk 1 Bishan St 1?"),
    ToolAddon("get_area_resale_statistics", ("mcp_server.area_resale_tools", "get_area_resale_statistics"),
              "market_sql", f"{R}:GetAreaResaleStatisticsInput", f"{R}:GetAreaResaleStatisticsOutput", _area,
              "the pooled price distribution around a place; pass the same area={address, radius_meters}",
              "Prices around the place", "What are 4-room resale prices within 200m of Blk 1 Bishan St 1?"),
    ToolAddon("get_area_price_trend", ("mcp_server.area_resale_tools", "get_area_price_trend"), "market_sql",
              f"{R}:GetAreaPriceTrendInput", f"{R}:GetAreaPriceTrendOutput", _area,
              "how prices around a place changed over time", "The price trend around the place",
              "How have 4-room prices within 300m of Blk 1 Bishan St 1 changed over the last 3 years?"),
    ToolAddon("resolve_location", ("mcp_server.location_tools", "resolve_location"), "spatial",
              f"{R}:ResolveLocationInput", f"{R}:ResolveLocationOutput",
              lambda c: _place(c) and any(c.get(k) for k in ("wants_nearby_amenities", "wants_area_statistics",
                                                              "wants_block_history", "wants_price_assessment",
                                                              "wants_option_comparison")),
              "turn a user's place (address, postal code, MRT station) into coordinates", "Where the place is",
              "What amenities are near Blk 1 Bishan St 1?"),
    ToolAddon("get_nearby_amenities", ("mcp_server.location_tools", "get_nearby_amenities"), "spatial",
              f"{R}:GetNearbyAmenitiesInput", f"{R}:GetNearbyAmenitiesOutput",
              lambda c: bool(c.get("wants_nearby_amenities")) and _place(c),
              "what is near a place (MRT, hawker centres, supermarkets, schools)", "Amenities near the place",
              "What amenities are near Blk 1 Bishan St 1?"),
)

PACK = ToolPack(
    name="research",
    tools=TOOLS,
    plan=_plan,
    prompt_preamble={
        "market_sql": "pass the user's addresses verbatim (these tools resolve them) and never estimate coordinates",
        "spatial": "pass addresses exactly as given and never estimate coordinates",
    },
)
