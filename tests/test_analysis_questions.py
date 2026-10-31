"""Broad market questions (grouped statistics, drill-down, analysis mode): how each wording is read, routed and
answered. Sections: rule shapes, then (added with later steps) rendering, analysis mode, the drill-down walk."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"

import pytest  # noqa: E402

from agents import research_intents as ri  # noqa: E402
from agents import supervisor as sv  # noqa: E402


def read(q):
    c = sv.rule_based_extract(q).model_dump()
    return c, ri.stats_call_args(c, []) if c["wants_group_stats"] else None


# question -> the query_resale_statistics arguments the rule layer must produce
RULE_SHAPES = [
    ("What are the historical statistics of HDBs sold near MRT stations?",
     {"group_by": ["distance_band"], "near": "mrt", "lookback_years": 10, "include_trend": True}),
    ("Show me 4-room resale prices by town near MRT stations over the last 5 years",
     {"group_by": ["town"], "near": "mrt", "max_distance_m": 500, "flat_types": ["4 ROOM"], "lookback_years": 5,
      "include_trend": True, "sort_by": "median_price", "order": "desc"}),
    ("Which MRT stations have the most expensive resale flats nearby?",
     {"group_by": ["station"], "near": "mrt", "max_distance_m": 500, "sort_by": "median_price", "order": "desc"}),
    ("Which MRT stations have the cheapest resale flats nearby?",
     {"group_by": ["station"], "near": "mrt", "max_distance_m": 500, "sort_by": "median_price", "order": "asc"}),
    ("HDB listings near MRT", {"group_by": ["distance_band"], "near": "mrt"}),
    ("How do resale prices compare for flats near MRT vs far from MRT?", {"group_by": ["distance_band"], "near": "mrt"}),
    ("Sales statistics across Singapore", {"group_by": ["town"], "sort_by": "median_price", "order": "desc"}),
    ("Island-wide resale statistics", {"group_by": ["town"], "sort_by": "median_price", "order": "desc"}),
    ("Resale prices near MRT stations in Bishan by station",
     {"group_by": ["station"], "near": "mrt", "towns": ["BISHAN"], "max_distance_m": 500, "sort_by": "median_price",
      "order": "desc"}),
    ("What do flats near hawker centres sell for?", {"group_by": ["distance_band"], "near": "hawker"}),
    ("Resale prices near LRT stations since 2019",
     {"group_by": ["distance_band"], "near": "lrt", "start_month": "2019-01", "include_trend": True}),
    ("Resale prices within 300m of an MRT", {"group_by": ["distance_band"], "near": "mrt", "bands": [300]}),
    ("Resale prices within 1km of MRT stations by town",
     {"group_by": ["town"], "near": "mrt", "max_distance_m": 1000, "sort_by": "median_price", "order": "desc"}),
    ("Resale prices near an MRT, bands of 300/800/1500", {"group_by": ["distance_band"], "near": "mrt", "bands": [300, 800, 1500]}),
    ("Resale prices 0-300m, 300-800m, beyond 800m from an MRT", {"group_by": ["distance_band"], "near": "mrt", "bands": [300, 800]}),
    ("Which towns have the most expensive flats near supermarkets?",
     {"group_by": ["town"], "near": "supermarket", "max_distance_m": 500, "sort_by": "median_price", "order": "desc"}),
]


@pytest.mark.parametrize("query, expected", RULE_SHAPES, ids=[q[:50] for q, _ in RULE_SHAPES])
def test_rule_shape_is_read_and_planned(query, expected):
    c, args = read(query)
    assert args == expected, c
    assert sv.decide_routing(c) == ["market_sql"]
    assert ri.answerable_without_model(c, query)
    assert not c["addresses"] and not c["amenity_radii_meters"] and not c["wants_price_trend"]
    plan = sv.deterministic_plan("market_sql", c, query)
    assert [name for name, _ in plan] == ["query_resale_statistics"] and plan[0][1] == expected


def test_a_stated_distance_outside_the_range_is_clamped_and_said():
    c, args = read("Resale prices within 5km of an MRT")
    assert args["bands"] == [3000] and any("3,000 m was used" in a for a in c["assumptions"])
    c, args = read("Resale prices within 20m of an MRT by town")
    assert args["max_distance_m"] == 50 and any("50 m was used" in a for a in c["assumptions"])


def test_the_listings_wording_is_remembered():
    assert read("HDB listings near MRT")[0]["stats_listings"] is True
    assert read("Resale prices near MRT")[0]["stats_listings"] is False


@pytest.mark.parametrize("query, tools", [
    ("What are resale prices within 500m of Bishan MRT?", {"spatial"}),                       # a named station is a place
    ("What did flats at Blk 1 Bishan St 1 sell for recently?", {"market_sql"}),              # one block
    ("Find 4-room flats in Tampines under $600k within 400m of an MRT", {"market_sql", "spatial"}),   # a block search
    ("Flats near an MRT", {"spatial"}),                                                      # a plain proximity search
    ("Which towns are cheapest for a 4-room resale flat?", {"market_sql"}),                  # the existing town ranking
    ("What is the minimum occupation period for HDB?", {"policy_rag"}),
])
def test_other_questions_are_not_read_as_grouped_statistics(query, tools):
    c = sv.rule_based_extract(query).model_dump()
    assert not c["wants_group_stats"], query
    assert set(sv.decide_routing(c)) == tools, query
