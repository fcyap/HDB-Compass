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


# ---------------------------------------------------------------------------
# Whole-graph answers (Variant C, the app's fast path, the real local database)
# ---------------------------------------------------------------------------
import json  # noqa: E402

from langchain_core.messages import AIMessage  # noqa: E402

from agents import addons, scope  # noqa: E402
from agents.arch_3_specialist_5agent import graph, run  # noqa: E402


def ask(query, previous=None, fast=True):
    return graph.invoke(sv.initial_state(query, "C", None, previous, fast_path=fast))


def tools_run(out):
    return [r["tool"] for r in out["tool_call_log"] if r["tool"] != "sync_interactive_map_ui"]


def chips(out):
    return out.get("suggestions") or []


@pytest.fixture(scope="module")
def answers():
    """Each rule-shape question answered once on the fast path."""
    return {q: ask(q) for q, _ in RULE_SHAPES}


@pytest.mark.parametrize("query, expected", RULE_SHAPES, ids=[q[:50] for q, _ in RULE_SHAPES])
def test_rule_shape_answer_is_a_table_of_real_figures_with_chips(answers, query, expected):
    out = answers[query]
    assert out["active_subagents"] == ["market_sql"]
    assert tools_run(out) == ["query_resale_statistics"]
    call = next(r for r in out["tool_call_log"] if r["tool"] == "query_resale_statistics")
    for k, v in expected.items():
        assert call["arguments"][k] == v, (k, call["arguments"])
    text = out["draft_response"]
    assert "|---" in text and "Median price" in text and "$" in text
    assert "Question:" not in text and "could not" not in text.lower()
    assert out["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE      # code only, no model
    assert out["clarification"] is None                                             # chips are not a pending question
    assert len(chips(out)) >= 2 and all(c["label"] and c["text"] for c in chips(out))
    assert out["errors"] == []


def test_listings_question_says_it_is_past_sales(answers):
    text = answers["HDB listings near MRT"]["draft_response"]
    assert "Live listings are not available" in text and "past resale sales" in text


def test_every_grouped_answer_states_the_distance_it_used(answers):
    assert "Within 500 m straight-line of an MRT station (LRT stations excluded)" in \
        answers["Which MRT stations have the most expensive resale flats nearby?"]["draft_response"]
    assert "Distance bands (0-200 m, 200-500 m, 500-1,000 m, >1,000 m)" in answers["HDB listings near MRT"]["draft_response"]
    assert "Within 1,000 m" in answers["Resale prices within 1km of MRT stations by town"]["draft_response"]
    assert "0-300 m, >300 m" in answers["Resale prices within 300m of an MRT"]["draft_response"]


def test_stations_map_markers_carry_their_drill_question(answers):
    payload = answers["Which MRT stations have the most expensive resale flats nearby?"]["map_ui_payload"]
    feats = payload["markers"]["features"]
    stations = [f["properties"] for f in feats if f["properties"].get("kind") == "station"]
    assert stations and all(p["drill_question"].startswith("Resale prices near ") for p in stations)
    assert all(f["geometry"]["coordinates"][0] > 103 for f in feats)


# --- negatives: these must not become grouped statistics --------------------------------------------------------
def test_a_named_station_goes_to_the_area_tools():
    out = ask("Resale prices near Bishan MRT, within 500m")
    assert tools_run(out) == ["find_resale_blocks_near", "get_area_resale_statistics"]


def test_a_specific_block_goes_to_block_history():
    out = ask("What did flats at Blk 105 Bishan St 12 sell for recently?")
    assert tools_run(out) == ["get_block_transaction_history"]


def test_a_policy_question_goes_to_policy_rag():
    out = ask("What is the minimum occupation period for HDB flats?")
    assert out["active_subagents"] == ["policy_rag"]


def test_a_greeting_gets_the_fixed_reply():
    assert scope.classify("hi") == scope.GREETING
    assert sv.rule_based_extract("hi").wants_group_stats is False


def test_a_block_search_with_limits_is_not_grouped():
    out = ask("Find 4-room flats in Tampines under $600k within 400m of an MRT")
    assert "query_resale_statistics" not in tools_run(out)


# ---------------------------------------------------------------------------
# Analysis mode: wordings no rule covers, with a model choosing the calls
# ---------------------------------------------------------------------------
ANALYSIS_NAMES = {"describe_resale_data", "query_resale_statistics", "find_resale_blocks_near", "get_area_resale_statistics",
                  "get_area_price_trend", "get_block_transaction_history", "get_market_snapshot", "rank_towns_by_value",
                  "compute_town_price_timeseries"}


class ToolModel:
    """A model that makes one scripted tool call, then stops; it records the tools it was offered."""
    offered: list = []

    def __init__(self, call):
        self.call = call
        self.turn = 0

    def with_structured_output(self, *a, **k):
        raise RuntimeError("no structured output")        # extraction falls back to code

    def bind_tools(self, tools, tool_choice=None):
        ToolModel.offered = [t.name for t in tools]
        return self

    def invoke(self, _messages):
        self.turn += 1
        if self.turn == 1:
            return AIMessage(content="", tool_calls=[{"name": "query_resale_statistics", "id": "c1", "args": self.call}])
        return AIMessage(content="The figures are below.")


LONG_TAIL = [
    ("Which streets in Bishan have the priciest flats?", {"group_by": ["street"], "towns": ["BISHAN"], "sort_by": "median_price", "top_n": 5}),
    # storey/lease/year wordings are rule shapes now (test_floor_and_lease_questions_group_by_that_dimension)
    ("Are bigger flats worth the premium per square metre?", {"group_by": ["flat_type"], "sort_by": "median_psm"}),
    ("Does remaining lease affect resale prices?", {"group_by": ["lease_band"]}),
    ("Compare executive and 5-room flat prices", {"group_by": ["flat_type"], "flat_types": ["EXECUTIVE", "5 ROOM"]}),
    ("Do maisonettes hold their value?", {"group_by": ["flat_type"]}),
    ("Are corner units more valuable?", {"group_by": ["flat_type"], "sort_by": "median_price"}),
    ("Are prices on the rise?", {"group_by": ["year"], "lookback_years": 5}),
    ("How much more do bigger flats cost per square metre?", {"group_by": ["flat_type"], "sort_by": "median_psm"}),
    ("Show the price distribution of 3-room flats", {"group_by": ["town"], "flat_types": ["3 ROOM"], "top_n": 5}),
]


@pytest.mark.parametrize("query, call", LONG_TAIL, ids=[q[:45] for q, _ in LONG_TAIL])
def test_long_tail_question_is_answered_by_the_models_call(monkeypatch, query, call):
    monkeypatch.setattr(sv, "get_llm", lambda: ToolModel(call))
    c = sv.rule_based_extract(query).model_dump()
    assert not c["wants_group_stats"] and not sv.decide_routing({**c, "towns": []})      # no rule recognised it
    out = ask(query, fast=False)
    assert out["user_constraints"]["analysis_mode"] is True
    assert out["active_subagents"] == ["market_sql"]
    assert set(ToolModel.offered) == ANALYSIS_NAMES                    # the analysis set, not the block search
    assert tools_run(out) == ["query_resale_statistics"]
    assert "|---" in out["draft_response"] and "Question:" not in out["draft_response"]
    assert "Happy to help" not in out["draft_response"]


def test_analysis_mode_without_a_valid_call_falls_back_to_an_overview(monkeypatch):
    class Silent(ToolModel):
        def invoke(self, _m):
            return AIMessage(content="I am not sure.")
    monkeypatch.setattr(sv, "get_llm", lambda: Silent({}))
    out = ask("Is it pricier around a supermarket?", fast=False)
    assert tools_run(out) == ["query_resale_statistics"] and "|---" in out["draft_response"]
    assert any("deterministic plan used" in e for e in out["errors"])


def test_analysis_mode_with_nothing_to_go_on_gives_the_menu():
    out = ask("How much do flats cost these days", fast=False)
    assert out["clarification"]["kind"] == "menu" and "Happy to help" in out["draft_response"]
    assert tools_run(out) == []


def test_data_words_are_needed_for_analysis_mode_otherwise_the_menu():
    out = ask("What flat should I buy?", fast=False)
    assert out["user_constraints"]["analysis_mode"] is False and out["clarification"]["kind"] == "menu"


def test_analysis_mode_is_off_when_add_ons_are_off(monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "off")
    out = ask("Do maisonettes hold their value?", fast=False)
    assert out["user_constraints"]["analysis_mode"] is False
    assert out["clarification"]["kind"] == "menu" and tools_run(out) == []


def test_the_analysis_prompt_has_the_data_guide_and_worked_examples():
    guide = addons.analysis_section("market_sql")
    assert "query_resale_statistics(group_by=" in guide and guide.count("->") >= 4 and "describe_resale_data" in guide
    assert addons.analysis_section("spatial") == ""


def test_six_tool_iterations_only_in_analysis_mode():
    assert ri.ANALYSIS_MAX_ITERATIONS == 6


# ---------------------------------------------------------------------------
# Drill-down walk: island -> distance bands -> town -> station -> area -> block, every chip a fresh fast-path question
# ---------------------------------------------------------------------------
def test_chip_walk_from_bands_to_a_blocks_sales():
    state = ask("What are the historical statistics of HDBs sold near MRT stations?")
    assert tools_run(state) == ["query_resale_statistics"]
    by_label = {c["label"]: c["text"] for c in chips(state)}
    assert {"By town", "By station"} <= set(by_label)

    state = ask(by_label["By town"], state)                                      # L1: towns
    assert state["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE
    assert tools_run(state) == ["query_resale_statistics"]
    assert state["user_constraints"]["stats_group_by"] == ["town"] and "Town" in state["draft_response"]
    town_chip = chips(state)[0]
    assert town_chip["text"].startswith("Resale prices by station near MRT stations in ")

    state = ask(town_chip["text"], state)                                        # L2: stations of that town
    assert state["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE
    assert state["user_constraints"]["stats_group_by"] == ["station"]
    assert state["user_constraints"]["towns"] == [town_chip["label"].upper()]
    assert "Nearest station" in state["draft_response"]
    station_chip = next(c for c in chips(state) if c["text"].startswith("Resale prices near "))

    state = ask(station_chip["text"], state)                                     # L3: the area around the station
    assert state["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE
    assert tools_run(state) == ["find_resale_blocks_near", "get_area_resale_statistics"]
    block_chip = next(c for c in chips(state) if c["text"].startswith("What did"))
    assert any(c["label"].startswith("All stations in") for c in chips(state))

    state = ask(block_chip["text"], state)                                       # L4: that block's sales
    assert tools_run(state) == ["get_block_transaction_history"]
    assert "Sales history" in state["draft_response"]


def test_a_chip_also_works_through_run_with_the_previous_state():
    first = ask("Which MRT stations have the most expensive resale flats nearby?")
    follow = run(chips(first)[0]["text"], previous=first)       # the experiment entry point: no fast path, no model
    assert [r["tool"] for r in follow["tool_call_log"] if r["tool"] != "sync_interactive_map_ui"][:1] == ["find_resale_blocks_near"]


# ---------------------------------------------------------------------------
# Category A seed: the tool agrees with ground truth worked out by independent SQL
# ---------------------------------------------------------------------------
CATEGORY_A = json.loads((__import__("pathlib").Path(__file__).resolve().parent.parent / "evaluation" / "category_a_queries.json")
                        .read_text(encoding="utf-8"))


def _snapshot_id():
    from data.market_database import open_market_database
    with open_market_database() as con:
        return con.execute("SELECT snapshot_id FROM source_snapshot").fetchone()[0]


@pytest.mark.parametrize("item", CATEGORY_A, ids=[i["id"] for i in CATEGORY_A])
def test_category_a_question_is_read_and_matches_independent_sql(item):
    query = item["query"]
    call = item["expected_mcp_calls"][0]
    assert call["tool"] == "query_resale_statistics"
    args = ri.stats_call_args(sv.rule_based_extract(query).model_dump(), [])
    assert args == call["required_params"]
    facts = item["ground_truth_facts"]
    if facts["snapshot_id"] != _snapshot_id():
        pytest.skip("ground truth was built on another database snapshot; rebuild with evaluation.build_category_a")
    from mcp_server.aggregate_resale_tools import query_resale_statistics
    got = query_resale_statistics(**args)
    expected = [(g["label"], g["count"], round(g["median_price"])) for g in facts["groups"]]
    assert [(g["label"], g["count"], g["median_price"]) for g in got["groups"]][:len(expected)] == expected
    assert got["period"]["start_month"] == facts["period"].split("..")[0]


# --- found in review (Groq rate-limited, so the rule layer answered): parts of a question were silently dropped -----
@pytest.mark.parametrize("query, group, near", [
    ("Do flats on higher floors near hawker centres sell for more? Show me since 2020", ["storey_band"], "hawker"),
    ("Resale prices by storey in Tampines", ["storey_band"], None),
    ("What is the price premium for higher floors?", ["storey_band"], None),
    ("Average resale price by year", ["year"], None),
    ("Do flats with more lease left sell for more near MRT?", ["lease_band"], "mrt"),
])
def test_floor_and_lease_questions_group_by_that_dimension(query, group, near):
    c = sv.rule_based_extract(query).model_dump()
    assert c["wants_group_stats"] and c["stats_group_by"] == group and c.get("stats_near") == near


@pytest.mark.parametrize("query, towns", [
    ("What's the price premium for 5-room flats close to MRT in the east?", ["BEDOK", "PASIR RIS", "TAMPINES"]),
    ("Resale statistics near MRT stations in the north-east region", ["ANG MO KIO", "HOUGANG", "PUNGGOL", "SENGKANG", "SERANGOON"]),
])
def test_a_region_becomes_its_towns_and_is_disclosed(query, towns):
    c = sv.rule_based_extract(query).model_dump()
    assert c["towns"] == towns and any("URA planning regions" in a for a in c["assumptions"])


def test_jurong_east_is_a_town_not_the_east_region():
    c = sv.rule_based_extract("Show 4-room prices near MRT in Jurong East").model_dump()
    assert c["towns"] == ["JURONG EAST"]


def test_region_towns_cover_all_hdb_towns_once():
    from agents.research_intents import REGION_TOWNS
    towns = [t for ts in REGION_TOWNS.values() for t in ts]
    assert len(towns) == len(set(towns)) == 26


def test_flat_type_question_ranked_per_square_metre_without_a_model():
    # found with Groq rate-limited: analysis mode's no-model fallback had no amenity or town, so it showed the menu
    c = sv.rule_based_extract("Which flat types sell for the most per square metre?").model_dump()
    assert c["wants_group_stats"] and c["stats_group_by"] == ["flat_type"]
    assert c["stats_sort"] == "median_psm" and c["stats_order"] == "desc"


def test_per_sqm_ranking_headline_quotes_per_sqm_figures():
    # found in review: the headline named the per-sqm ranking but quoted the median price
    from agents.research_intents import _stats_headline
    r = {"group_by": ["flat_type"], "sort_by": "median_psm", "order": "desc",
         "groups": [{"key": "2 ROOM", "label": "2 ROOM", "median_price": 372000, "median_psm": 8085, "count": 752}],
         "baseline": {"label": "all flat sales island-wide", "median_price": 630000, "median_psm": 6483}}
    line = _stats_headline(r)
    assert "$8,085 per sqm" in line and "$6,483 per sqm" in line and "$372,000" not in line
