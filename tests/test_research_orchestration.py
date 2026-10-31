"""Supervisor integration of the research intents and the Phase 0 correctness fixes.

Covers regex extraction of research intents, routing, deterministic plans, typed LLM
extraction, unknown-price handling, school-name matching, tool-message compaction,
rejection of model-invented coordinates and full Variant C runs on synthetic data.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from agents import research_intents as ri
from agents import supervisor as sv
from agents.schemas import LLMExtractedConstraints
from tests.test_research_tools import research_env, snapshot  # noqa: F401  (shared fixtures)


def extract(query):
    return sv.rule_based_extract(query).model_dump()


# ---------------------------------------------------------------------------
# Extraction and routing
# ---------------------------------------------------------------------------
def test_price_assessment_extracts_address_and_asking_price():
    c = extract("Is $720k a fair price for a 4-room at Blk 312 Shunfu Rd?")
    assert c["addresses"] == ["Blk 312 Shunfu Rd"] and c["asking_price_sgd"] == 720000
    assert c["wants_price_assessment"] and not c["max_budget_sgd"]
    assert sv.decide_routing(c) == ["market_sql"]
    assert [name for name, _ in sv.deterministic_plan("market_sql", c, "Is $720k fair")] == ["assess_resale_price"]


def test_block_history_comparison_and_nearby_intents():
    history = extract("What have 4-room flats at Blk 266 Toh Guan Rd sold for?")
    assert history["addresses"] == ["Blk 266 Toh Guan Rd"] and history["wants_block_history"]
    compare = extract("Compare Blk 288B Jurong East St 21 vs Blk 266 Toh Guan Rd for a 4-room")
    assert compare["wants_option_comparison"] and len(compare["addresses"]) == 2
    plan = sv.deterministic_plan("market_sql", compare, "Compare ...")
    assert [name for name, _ in plan] == ["compare_resale_options"]  # no generic Jurong East block search
    nearby = extract("What's near Blk 312 Shunfu Rd?")
    assert nearby["wants_nearby_amenities"] and sv.decide_routing(nearby) == ["spatial"]
    assert sv.deterministic_plan("spatial", nearby, "What's near")[0][0] == "get_nearby_amenities"


def test_snapshot_ranking_and_trend_years():
    assert sv.decide_routing(extract("How is the resale market now?")) == ["market_sql"]
    cheapest = extract("Which town is the cheapest for 5-room flats?")
    (name, args), = sv.deterministic_plan("market_sql", cheapest, "Which town is the cheapest for 5-room flats?")
    assert name == "rank_towns_by_value" and args["rank_by"] == "median_price" and args["flat_type"] == "5 ROOM"
    trend = extract("Compare Bishan and Toa Payoh 4-room prices over the last 3 years")
    plan = dict(sv.deterministic_plan("market_sql", trend, "Compare Bishan and Toa Payoh 4-room prices over the last 3 years"))
    assert plan == {"compute_town_price_timeseries": {"towns": ["BISHAN", "TOA PAYOH"], "flat_type": "4 ROOM", "lookback_years": 3}}


def test_vague_proximity_gets_a_disclosed_default():
    c = extract("Find flats near Tampines MRT under 550k")
    assert c["amenity_radii_meters"] == {"mrt": ri.VAGUE_RADIUS_METERS}
    assert any("500 m" in note for note in c["assumptions"])
    assert "spatial" in sv.decide_routing(c)


def test_generic_school_phrase_asks_instead_of_matching():
    c = extract("Which town gives the best value 4-room flat near good schools, under 600k?")
    assert c["school_name"] is None and "school" not in c["amenity_radii_meters"]
    assert c["clarifications"] == [ri.SCHOOL_CLARIFICATION]
    assert c["wants_town_ranking"] and "spatial" not in sv.decide_routing(c)


def test_classic_block_search_is_unchanged():
    c = extract("Find 4-room resale flats in Jurong East under $650,000 with at least 60 years of lease left, within 400m of an MRT.")
    assert c["amenity_radii_meters"] == {"mrt": 400} and not c["assumptions"]
    assert sv.decide_routing(c) == ["market_sql", "spatial"]
    assert [n for n, _ in sv.deterministic_plan("market_sql", c, "Find ...")] == ["query_resale_blocks_by_criteria"]


def test_llm_extraction_schema_is_typed_and_converts():
    schema = json.dumps(LLMExtractedConstraints.model_json_schema())
    radii = LLMExtractedConstraints.model_json_schema()["properties"]["amenity_radii_meters"]
    assert "additionalProperties" not in json.dumps(radii) and "AmenityRadii" in schema
    c = LLMExtractedConstraints(amenity_radii_meters={"mrt": 300, "hawker": None}).to_constraints()
    assert c.amenity_radii_meters == {"mrt": 300}


class FakeStructured:
    def __init__(self, result):
        self.result = result

    def invoke(self, _messages):
        return self.result


class FakeLLM:
    """Structured output succeeds; tool loop replays scripted AI messages."""

    def __init__(self, extraction=None, script=()):
        self.extraction, self.script = extraction, list(script)

    def with_structured_output(self, _schema, method=None, include_raw=False):
        return FakeStructured({"parsed": self.extraction, "raw": None} if include_raw else self.extraction)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self.script.pop(0) if self.script else AIMessage(content="done")


def test_extraction_records_model_source_and_drops_invented_facts(monkeypatch):
    model = LLMExtractedConstraints(
        addresses=["Blk 312 Shunfu Rd", "200m of a hawker centre"], wants_price_assessment=True,
        comparable_resale={"latitude": 1.3465, "longitude": 103.8505, "flat_type": "4 ROOM"})
    monkeypatch.setattr(sv, "get_llm", lambda: FakeLLM(extraction=model))
    out = sv.extract_constraints_node(sv.initial_state("Is $720k fair for a 4-room at Blk 312 Shunfu Rd?", "C"))
    c = out["user_constraints"]
    assert c["extraction_source"] == "llm:json_schema"
    assert c["addresses"] == ["Blk 312 Shunfu Rd"] and c["comparable_resale"] is None
    assert any("coordinates were not in the message" in e for e in out["errors"])


# ---------------------------------------------------------------------------
# Correctness fixes
# ---------------------------------------------------------------------------
def test_unpriced_blocks_are_not_within_budget():
    spatial_only = [{"block": "1", "street": "A ST", "distances": {}}]
    assert sv.compute_intersection(None, spatial_only, 600000) == []
    assert sv.compute_intersection(None, spatial_only, None) == spatial_only


def test_school_matching_requires_whole_words():
    from mcp_server.spatial_onemap_tools import _match_school
    schools = [{"name": "CHIJ OUR LADY OF GOOD COUNSEL", "alias": ""}, {"name": "AI TONG SCHOOL", "alias": "Ai Tong"}]
    assert _match_school("good school", schools) is None
    assert _match_school("Ai Tong", schools)["name"] == "AI TONG SCHOOL"
    assert _match_school("Our Lady of Good Counsel", schools)["name"] == "CHIJ OUR LADY OF GOOD COUNSEL"


def test_compact_tool_result_stays_valid_json():
    big = {"blocks": [{"block": str(i), "street": "X ST " * 20} for i in range(500)], "sql": "SELECT" * 1000}
    text = ri.compact_tool_result(big, 4000)
    parsed = json.loads(text)
    assert len(text) <= 4000 and parsed["blocks"][-1]["_omitted_items"] > 0 and "sql" not in parsed


def test_unverified_coordinates_are_rejected():
    args = {"latitude": 1.346528, "longitude": 103.850481, "flat_type": "4 ROOM"}
    assert ri.unverified_coordinates("query_comparable_resale_transactions", args, [])
    assert ri.unverified_coordinates("query_comparable_resale_transactions", args, [(1.34653, 103.85048)]) is None
    assert ri.unverified_coordinates("query_resale_blocks_by_criteria", args, []) is None


def test_react_loop_rejects_invented_coordinates_and_backfills_needed_tool(research_env, monkeypatch):
    invented = AIMessage(content="", tool_calls=[{"name": "query_comparable_resale_transactions", "id": "c1",
                                                  "args": {"latitude": 1.3465, "longitude": 103.8505, "flat_type": "4 ROOM"}}])
    valid = AIMessage(content="", tool_calls=[{"name": "get_market_snapshot", "id": "c2", "args": {}}])
    monkeypatch.setattr(sv, "get_llm", lambda: FakeLLM(script=[invented, valid]))
    c = extract("Is $700k a fair price for a 4-room at Blk 1 Bishan St 1?")
    outputs, log, errors, _ = sv.run_react("market_sql", sv.tools_for_agent("market_sql"), "x", "q", c)
    assert "query_comparable_resale_transactions" not in outputs
    assert any("rejected unverified coordinates" in e for e in errors)
    assert outputs["get_market_snapshot"]["status"] == "ok"
    assert outputs["assess_resale_price"]["status"] == "ok"  # coverage check fetched what the question needs
    assert [r["agent"] for r in log if r["tool"] == "assess_resale_price"] == ["supervisor"]


# ---------------------------------------------------------------------------
# Full Variant C runs (deterministic mode, synthetic data)
# ---------------------------------------------------------------------------
@pytest.fixture
def deterministic(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")


def test_variant_c_answers_price_check_with_map(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    out = C.run("Is $700k a fair price for a 4-room at Blk 1 Bishan St 1?")
    assert [r["tool"] for r in out["tool_call_log"]] == ["assess_resale_price"]
    assert "Price check for Blk 1 BISHAN ST 1" in out["draft_response"]
    assert "No block satisfies" not in out["draft_response"]
    names = [f["properties"]["name"] for f in out["map_ui_payload"]["markers"]["features"]]
    assert names == ["Blk 1 BISHAN ST 1"]


def test_variant_c_nearby_and_comparison(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    nearby = C.run("What's near Blk 1 Bishan St 1?")
    assert "Near Blk 1 BISHAN ST 1" in nearby["draft_response"] and "Bishan Mrt Station" in nearby["draft_response"]
    assert nearby["map_ui_payload"]["buffer_polygon"]
    compare = C.run("Compare Blk 1 Bishan St 1 vs Blk 5 Tampines St 1 for a 4-room")
    assert "| Blk 1 BISHAN ST 1 |" in compare["draft_response"] and "| Blk 5 TAMPINES ST 1 |" in compare["draft_response"]


def test_variant_c_asks_which_place_when_ambiguous(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    out = C.run("What have flats at Blk 2 Bishan sold for?")
    assert "Did you mean" in out["draft_response"] and "BISHAN ST 2" in out["draft_response"]


def test_variant_c_prices_school_results_before_applying_budget(research_env, deterministic, monkeypatch):
    from agents import arch_3_specialist_5agent as C
    school_blocks = ["Blk 1 BISHAN ST 1 (450m)", "Blk 2 BISHAN ST 2 (600m)"]
    real = sv.ToolRunner.__call__

    def fake(self, **kwargs):  # the school tool reads the full spatial index; stub it, keep logging real
        if self.name == "verify_primary_school_distance":
            return school_blocks
        return real(self, **kwargs)

    monkeypatch.setattr(sv.ToolRunner, "__call__", fake)
    out = C.run("4-room flats within 1km of Ai Tong School under $800k")
    keys = [(b["block"], b["street"]) for b in out["candidate_blocks_intersection"]]
    assert keys == [("1", "BISHAN ST 1")]  # Blk 2 Bishan St 2 has no sales, so it is not shown as within budget
    assert any(r["agent"] == "supervisor" and r["tool"] == "query_resale_blocks_by_criteria" for r in out["tool_call_log"])


# ---------------------------------------------------------------------------
# Map pin: "here" / "this block" and place questions with no address
# ---------------------------------------------------------------------------
def test_pin_on_a_block_becomes_its_address(research_env):
    c = extract("Is $700k a fair price for a 4-room here?")
    ri.apply_map_pin(c, "Is $700k a fair price for a 4-room here?", (1.35001, 103.85001))
    assert c["addresses"] == ["Blk 1 BISHAN ST 1"] and c["wants_price_assessment"] and c["asking_price_sgd"] == 700000
    assert any("map pin" in note for note in c["assumptions"])


def test_pin_off_any_block_is_used_as_a_point_for_nearby(research_env):
    c = extract("What's around here?")
    ri.apply_map_pin(c, "What's around here?", (1.3600, 103.8600))
    assert c["map_pin"] == {"latitude": 1.36, "longitude": 103.86} and c["wants_nearby_amenities"]
    assert sv.decide_routing(c) == ["spatial"]
    (name, args), = sv.deterministic_plan("spatial", c, "What's around here?")
    assert name == "get_nearby_amenities" and args["latitude"] == 1.36


def test_named_address_wins_over_an_unreferenced_pin(research_env):
    q = "What have flats at Blk 5 Tampines St 1 sold for?"
    c = extract(q)
    ri.apply_map_pin(c, q, (1.3500, 103.8500))
    assert c["addresses"] == ["Blk 5 Tampines St 1"] and "map_pin" not in c


def test_variant_c_answers_about_the_pinned_block(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    out = C.run("What's near this block?", map_state={"center": [1.35001, 103.85001], "radius_meters": 1000})
    assert [r["tool"] for r in out["tool_call_log"]] == ["get_nearby_amenities"]
    assert "Near Blk 1 BISHAN ST 1" in out["draft_response"]


# ---------------------------------------------------------------------------
# Area questions: find blocks near a place, then set statistics
# ---------------------------------------------------------------------------
def test_area_question_plans_find_then_statistics():
    q = "What are resale prices around Blk 1 Bishan St 1?"
    c = extract(q)
    assert c["wants_area_statistics"] and sv.decide_routing(c) == ["market_sql"]
    plan = sv.deterministic_plan("market_sql", c, q)
    assert [n for n, _ in plan] == ["find_resale_blocks_near", "get_area_resale_statistics"]
    assert plan[1][1]["area"] == {"address": "Blk 1 Bishan St 1", "radius_meters": ri.VAGUE_RADIUS_METERS}
    assert any("'Around' was taken as within 500 m" in a for a in c["assumptions"])


def test_area_radius_and_trend_years():
    q = "How have 4-room flats within 300m of Blk 1 Bishan St 1 sold over the last 3 years?"
    c = extract(q)
    assert c["area_radius_meters"] == 300 and not c["amenity_radii_meters"]
    names = dict(sv.deterministic_plan("market_sql", c, q))
    assert names["get_area_price_trend"] == {"area": {"address": "Blk 1 Bishan St 1", "radius_meters": 300},
                                             "flat_type": "4 ROOM", "lookback_years": 3}
    assert "get_market_snapshot" not in names  # the area trend replaces the island-wide fallback


def test_area_sales_history_runs_over_the_area():
    q = "What have flats around Blk 1 Bishan St 1 sold for?"
    names = dict(sv.deterministic_plan("market_sql", extract(q), q))
    assert names["get_block_transaction_history"] == {"area": {"address": "Blk 1 Bishan St 1", "radius_meters": 500},
                                                      "flat_type": None}


def test_amenity_radius_is_not_an_area_radius():
    c = extract("Find 4-room resale flats in Jurong East under $650,000 within 400m of an MRT.")
    assert not c["wants_area_statistics"] and c.get("area_radius_meters") is None


def test_area_question_about_the_pin(research_env):
    q = "How much do flats near here cost?"
    c = extract(q)
    ri.apply_map_pin(c, q, (1.35001, 103.85001))
    assert c["wants_area_statistics"] and c["addresses"] == ["Blk 1 BISHAN ST 1"]
    assert sv.deterministic_plan("market_sql", c, q)[0] == (
        "find_resale_blocks_near", {"address": "Blk 1 BISHAN ST 1", "radius_meters": 500, "flat_type": None})


def test_invented_coordinates_inside_area_are_rejected():
    args = {"area": {"latitude": 1.3465, "longitude": 103.8505, "radius_meters": 500}, "flat_type": "4 ROOM"}
    assert ri.unverified_coordinates("get_area_resale_statistics", args, [])
    assert ri.unverified_coordinates("get_area_resale_statistics", args, [(1.3465, 103.8505)]) is None


def test_variant_c_answers_area_prices_with_map(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    out = C.run("What are 4-room resale prices within 200m of Blk 1 Bishan St 1?")
    assert [r["tool"] for r in out["tool_call_log"]] == ["find_resale_blocks_near", "get_area_resale_statistics"]
    text = out["draft_response"]
    assert "Resale prices, 2 blocks within 200 m of Blk 1 BISHAN ST 1" in text and "median $670,000" in text
    assert "| Blk 2 BISHAN ST 1 | 111 m | 3 |" in text
    names = {f["properties"]["name"] for f in out["map_ui_payload"]["markers"]["features"]}
    assert names == {"Blk 1 BISHAN ST 1", "Blk 2 BISHAN ST 1"} and out["map_ui_payload"]["buffer_polygon"]


@pytest.mark.parametrize("variant", ["B", "C"])
def test_place_without_a_question_gets_asked_not_policy(research_env, deterministic, variant):
    import importlib
    mod = importlib.import_module({"B": "agents.arch_2_coarse", "C": "agents.arch_3_specialist_5agent"}[variant])
    out = mod.run("Tell me about Blk 1 Bishan St 1")
    assert out["tool_call_log"] == [] and "What would you like to know about Blk 1 Bishan St 1" in out["draft_response"]


def test_model_invented_town_is_dropped(monkeypatch):
    model = LLMExtractedConstraints(towns=["BUKIT MERAH", "JURONG EAST"], flat_types=["4 ROOM"],
                                    addresses=["Blk 277 Toh Guan Rd"])
    monkeypatch.setattr(sv, "get_llm", lambda: FakeLLM(extraction=model))
    out = sv.extract_constraints_node(sv.initial_state("4-room prices around Blk 277 Toh Guan Rd, Jurong East?", "C"))
    assert out["user_constraints"]["towns"] == ["JURONG EAST"]
    assert any("not named in the message" in e for e in out["errors"])
    alias = LLMExtractedConstraints(towns=["BUKIT MERAH"])
    monkeypatch.setattr(sv, "get_llm", lambda: FakeLLM(extraction=alias))
    assert sv.extract_constraints_node(sv.initial_state("5-room in Tiong Bahru", "C"))["user_constraints"]["towns"] == ["BUKIT MERAH"]


# ---------------------------------------------------------------------------
# App fast path: recognised questions need no model round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, fast", [
    ("What's near Blk 1 Bishan St 1?", True),
    ("What are resale prices around Blk 1 Bishan St 1?", True),
    ("Is $700k a fair price for a 4-room at Blk 1 Bishan St 1?", True),
    ("Which town is the cheapest for 5-room flats?", True),
    ("Compare Bishan and Toa Payoh 4-room prices over the last 3 years", True),
    ("Find 4-room resale flats in Jurong East under $650,000 within 400m of an MRT.", False),  # classic search
    ("We earn $9,000/month. Can we afford Blk 1 Bishan St 1?", False),                           # financial facts
    ("Tell me about Blk 1 Bishan St 1", False),                                                 # no intent: model may help
])
def test_fast_path_only_for_fully_recognised_questions(query, fast):
    assert ri.answerable_without_model(extract(query), query) is fast


class NoModel:
    def __getattr__(self, name):
        raise AssertionError("the fast path must not call the model")


def test_fast_path_runs_without_any_model_call(research_env, monkeypatch):
    from agents.arch_3_specialist_5agent import graph
    monkeypatch.setattr(sv, "get_llm", lambda: NoModel())
    state = sv.initial_state("What's near this block?", "C", fast_path=True,
                             map_state={"center": [1.35001, 103.85001], "radius_meters": 1500})
    out = graph.invoke(state)
    assert out["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE
    assert [(r["tool"], r["arguments"]["radius_meters"]) for r in out["tool_call_log"]] == [("get_nearby_amenities", 1500)]
    assert "Near Blk 1 BISHAN ST 1 (within 1500 m)" in out["draft_response"]


def test_experiments_keep_the_model_path(research_env, monkeypatch):
    model = LLMExtractedConstraints(addresses=["Blk 1 Bishan St 1"], wants_nearby_amenities=True)
    monkeypatch.setattr(sv, "get_llm", lambda: FakeLLM(extraction=model))
    out = sv.extract_constraints_node(sv.initial_state("What's near Blk 1 Bishan St 1?", "C"))
    assert out["user_constraints"]["extraction_source"] == "llm:json_schema"


class CountingLLM(FakeLLM):
    calls = 0

    def invoke(self, messages):
        CountingLLM.calls += 1
        return super().invoke(messages)


def test_agent_loop_stops_once_planned_tools_ran(research_env, monkeypatch):
    call = AIMessage(content="", tool_calls=[{"name": "get_nearby_amenities", "id": "c1",
                                              "args": {"address": "Blk 1 Bishan St 1", "radius_meters": 1000}}])
    c = extract("What's near Blk 1 Bishan St 1?")
    for stop, expected in ((True, 1), (False, 2)):
        CountingLLM.calls = 0
        monkeypatch.setattr(sv, "get_llm", lambda: CountingLLM(script=[call]))
        sv.run_react("spatial", sv.tools_for_agent("spatial"), "x", "What's near Blk 1 Bishan St 1?", c,
                     stop_when_covered=stop)
        assert CountingLLM.calls == expected  # without the stop, the model is asked again just to summarise
