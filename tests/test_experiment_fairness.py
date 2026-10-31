"""
Experiment 1 fairness and answer correctness (tests/test_experiment_fairness.py). Owned by Member 4.

  * Same tools for A, B and C: the runner never runs with HDB_ADDONS=app (add-ons for C only); the add-on packs are
    off for the core sets and on for every variant for a dataset that expects an add-on tool (Category A). Under
    'all', A's and B's prompts carry the same add-on lines C's specialists get, and B gets C's analysis-mode guide.
  * An empty commute result empties the candidate set; the amenity filter's [] is not replaced by commute results.
  * A commute limit no check applied (one workplace) is said to be unchecked, never "every limit you set".
  * A, B and C accept the same coordinates in a model's tool call: the map pin and coordinates the user typed.
  * C's school guard accepts real school names with "St." or a hyphen.
  * B answers a menu pick with the option's question instead of showing the menu again.

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_experiment_fairness.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

os.environ["LLM_PROFILE"] = "none"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import addons  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents import arch_1_monolith as A  # noqa: E402
from agents import arch_2_coarse as B  # noqa: E402
from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents.school_queries import SCHOOL_NAME_QUESTION  # noqa: E402
from agents.schemas import TOOL_OWNER  # noqa: E402
from evaluation import run_category_e_prelim as rp  # noqa: E402

COMPARABLES = "query_comparable_resale_transactions"
EVAL = PROJECT_ROOT / "evaluation"


def load(name: str) -> list:
    return json.loads((EVAL / name).read_text())


# --- 1. the runner: never 'app', off for the core sets, all where an add-on tool is expected ----------------------
@pytest.mark.parametrize("dataset,mode", [
    ("", "off"), ("varied", "off"), ("extra", "off"), ("category_c", "off"), ("category_b", "off"),
    ("evaluation/category_b_queries.json", "off"), ("category_a", "all"),
])
def test_runner_offers_add_ons_to_every_variant_or_none(dataset, mode):
    path = rp.dataset_path(dataset) if not dataset.startswith("evaluation/") else PROJECT_ROOT / dataset
    assert rp.addon_mode(json.loads(path.read_text())) == mode


def test_category_a_and_b_shortcuts_and_help():
    assert rp.dataset_path("category_a") == EVAL / "category_a_queries.json" == rp.CATEGORY_A
    assert rp.dataset_path("category_b") == EVAL / "category_b_queries.json" == rp.CATEGORY_B
    # Category A's only expected tool is an add-on: with the packs off no variant could make the call
    assert {c["tool"] for i in load("category_a_queries.json") for c in i["expected_mcp_calls"]} == {"query_resale_statistics"}
    assert addons.is_addon("query_resale_statistics")
    text = " ".join(rp.build_parser().format_help().split())
    assert "category_a" in text and "category_b" in text and "app (Variant C only) is refused" in text


def test_runner_refuses_app_and_keeps_an_explicit_off_or_all():
    items = load("category_a_queries.json")
    with pytest.raises(SystemExit) as stop:
        rp.addon_mode(items, "app")
    assert "Variant C only" in str(stop.value.code)
    with pytest.raises(SystemExit):
        rp.addon_mode(items, "some")
    assert rp.addon_mode(items, "off") == "off" and rp.addon_mode(load("category_e_queries.json"), " ALL ") == "all"


def run_main(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(rp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rp, "active_profile", lambda: {"name": "none", "model": None, "base_url": None, "api_key": None})
    monkeypatch.setattr(sys, "argv", ["run_category_e_prelim.py", *args])
    rp.main()
    return [json.loads(line) for f in sorted(tmp_path.glob("prelim_*.jsonl")) for line in f.read_text().splitlines()]


def test_runner_stops_under_app_before_running_anything(monkeypatch, tmp_path):
    monkeypatch.setenv("HDB_ADDONS", "app")
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, tmp_path, "--variants", "C", "--ids", "E1-01", "--tag", "fair")
    assert "Variant C only" in str(stop.value.code) and not list(tmp_path.glob("prelim_*"))
    assert os.environ["HDB_ADDONS"] == "app"


@pytest.mark.parametrize("dataset,qid,mode", [("category_a", "A1-01", "all"), ("", "E1-01", "off")])
def test_runner_records_the_mode_it_chose_and_restores_the_environment(monkeypatch, tmp_path, dataset, qid, mode):
    monkeypatch.delenv("HDB_ADDONS", raising=False)
    rows = run_main(monkeypatch, tmp_path, "--variants", "A,B,C", "--ids", qid, "--tag", "fair",
                    *(["--dataset", dataset] if dataset else []))
    assert [r["status"] for r in rows] == ["ok"] * 3
    assert {r["settings"]["addons"] for r in rows} == {mode}
    assert "HDB_ADDONS" not in os.environ


# --- 1. under 'all', A, B and C get the same tools and the same lines about them -----------------------------------
class ReadsNothing:
    """A constraint reading with nothing in it (code's own reading fills the constraints)."""

    def invoke(self, _msgs):
        from agents.schemas import ExtractedConstraints
        return {"parsed": ExtractedConstraints()}


class Recorder:
    """Records each tool-loop prompt's system text and the tools it was offered, and calls no tool."""

    def __init__(self):
        self.calls = []        # (system text, offered tool names)
        self.offered = []

    def with_structured_output(self, schema, method=None, include_raw=False):
        return ReadsNothing()

    def bind_tools(self, tools, tool_choice=None):
        self.offered = [t.name for t in tools]
        return self

    def invoke(self, msgs):
        self.calls.append((str(msgs[0].content), list(self.offered)))
        return AIMessage(content="done")


ADDON_TOOLS = {t.name for t in addons.all_addons()}


def test_tool_set_sizes_per_variant_and_mode(monkeypatch):
    sizes = {}
    for mode in ("off", "all"):
        monkeypatch.setenv("HDB_ADDONS", mode)
        sizes[mode] = (len(addons.visible_tools(list(TOOL_OWNER), "A")), len(addons.visible_tools(B.DATA_TOOLS, "B")),
                       len(addons.visible_tools(list(TOOL_OWNER), "C")))
    core = len([t for t in TOOL_OWNER if t not in ADDON_TOOLS])
    assert sizes["off"] == (core, core - 1, core)                        # B's policy tool is its other agent's
    assert sizes["all"] == (len(TOOL_OWNER), len(TOOL_OWNER) - 1, len(TOOL_OWNER))
    assert sizes["all"][0] - sizes["off"][0] == len(ADDON_TOOLS)


@pytest.mark.usefixtures("mock_domain_tools")
@pytest.mark.parametrize("mode", ["off", "app", "all"])
def test_a_and_b_get_the_add_on_lines_c_gets_for_the_same_tools(monkeypatch, mode):
    monkeypatch.setenv("HDB_ADDONS", mode)
    hints = {agent: addons.prompt_section(agent) for agent in sv.SPECIALISTS if addons.prompt_section(agent)}
    assert hints   # the packs give the market and spatial agents lines of their own
    query = "Find a 4-room flat in Tampines under $680k within 400m of an MRT."
    model = Recorder()
    with patch.object(sv, "get_llm", return_value=model):
        A.run(query)
        a_calls = list(model.calls)
        model.calls.clear()
        state = sv.initial_state(query, "B")
        B.data_agent_node({**state, **sv.extract_constraints_node(state)})
        b_calls = list(model.calls)
    (a_system, a_tools), = a_calls
    (b_system, b_tools), = b_calls
    on = mode == "all"
    assert (ADDON_TOOLS <= set(a_tools)) is on and (ADDON_TOOLS - {"find_primary_schools_near_address"} <= set(b_tools)) is on
    for agent, hint in hints.items():
        assert (hint in a_system) is on and (hint in b_system) is on
        assert hint in sv.AGENT_SYSTEM[agent]          # what C's specialist is given whenever it is offered them
    # the core prompts themselves are unchanged
    assert a_system.startswith(A.MONOLITH_SYSTEM) and b_system.startswith(B.DATA_SYSTEM)
    if not on:
        assert a_system == A.MONOLITH_SYSTEM and b_system == B.DATA_SYSTEM


@pytest.mark.usefixtures("mock_domain_tools")
@pytest.mark.parametrize("mode", ["off", "all"])
def test_b_gets_cs_analysis_guide_in_analysis_mode(monkeypatch, mode):
    monkeypatch.setenv("HDB_ADDONS", mode)
    guide = addons.analysis_section("market_sql")
    model = Recorder()
    state = {"messages": [{"role": "user", "content": "Do 5-room flats near hawker centres cost more?"}], "variant": "B",
             "user_constraints": {"analysis_mode": True}}
    with patch.object(sv, "get_llm", return_value=model):
        B.data_agent_node(state)
        sv.make_specialist_node("market_sql")({**state, "variant": "C"})
    (b_system, _), (c_system, _) = model.calls
    assert (guide in b_system) is (mode == "all") and (guide in c_system) is (mode == "all")
    assert sv.analysis_prompt({"analysis_mode": False}, "B") == "" and sv.analysis_prompt({}, "C") == ""


# --- 2. an empty commute result empties the candidates --------------------------------------------------------------
@pytest.fixture(scope="module")
def punggol_mrt():
    from mcp_server.spatial_onemap_tools import filter_blocks_by_amenity_proximity
    fn = getattr(filter_blocks_by_amenity_proximity, "fn", filter_blocks_by_amenity_proximity)
    return fn(["PUNGGOL"], {"mrt": 300})


def test_empty_commute_result_leaves_no_candidate(punggol_mrt):
    # review reproduction: 479 Punggol blocks within 300 m of an MRT, none within 30 min of Jurong East and Tuas
    assert punggol_mrt
    out = sv.slot_outputs({"filter_blocks_by_amenity_proximity": punggol_mrt, "compute_dual_transit_commute": []}, ["PUNGGOL"])
    assert out["spatial_block_candidates"] == []


def test_empty_amenity_filter_is_not_replaced_by_island_wide_commute_results():
    from mcp_server.spatial_onemap_tools import compute_dual_transit_commute
    fn = getattr(compute_dual_transit_commute, "fn", compute_dual_transit_commute)
    island = fn([], "Raffles Place", "one-north", 30)   # what the commute returns when it is given no candidates
    assert island
    out = sv.slot_outputs({"filter_blocks_by_amenity_proximity": [], "compute_dual_transit_commute": island}, ["PUNGGOL"])
    assert out["spatial_block_candidates"] == []
    # the same holds in either order, and for the school check
    out = sv.slot_outputs({"compute_dual_transit_commute": [], "filter_blocks_by_amenity_proximity": [{"block": "1", "street": "A ST"}]}, [])
    assert out["spatial_block_candidates"] == []
    out = sv.slot_outputs({"filter_blocks_by_amenity_proximity": [], "verify_primary_school_distance": ["Blk 1 A ST (300m)"]}, [])
    assert out["spatial_block_candidates"] == []


def test_commute_keeps_a_block_by_block_and_street():
    spatial = [{"block": "1", "street": "A ST"}, {"block": "2", "street": "A ST"}]
    commute = [{"block": "1", "street": "B ST"}, {"block": "2", "street": "A ST"}]
    out = sv.slot_outputs({"filter_blocks_by_amenity_proximity": spatial, "compute_dual_transit_commute": commute}, [])
    assert [b["block"] for b in out["spatial_block_candidates"]] == ["2"]   # Blk 1 met the limit only on another street
    # with no other filter the commute results are the candidates
    assert sv.slot_outputs({"compute_dual_transit_commute": commute}, [])["spatial_block_candidates"] == commute


def test_no_block_within_the_commute_relaxes_and_counts_with_the_commute_still_applied():
    q = ("I work at Raffles Place and work at one-north. Find 4-room resale flats in Punggol under $950,000 within 300m "
         "of an MRT station, with commute under 20 minutes to each workplace.")
    state = C.run(q)
    assert "compute_dual_transit_commute" in [r["tool"] for r in state["tool_call_log"] if r["agent"] == "spatial"]
    assert state["spatial_block_candidates"] == [] and state["candidate_blocks_intersection"] == []
    rel = state["relaxation"]
    assert rel["suggestions"] and all(s["unlocked_blocks"] == 0 for s in rel["suggestions"])
    assert state["draft_response"].startswith("No block meets every limit you set at once.")


# --- 3. a commute limit with one workplace is said to be unchecked --------------------------------------------------
ONE_WORKPLACE = "Find 4-room flats in Bedok under $550k with a commute under 30 min to Raffles Place"


def test_single_workplace_commute_limit_is_not_claimed_as_met():
    state = C.run(ONE_WORKPLACE)
    c = state["user_constraints"]
    assert c["work_destinations"] == ["Raffles Place"] and c["max_commute_minutes"] == 30
    assert "compute_dual_transit_commute" not in [r["tool"] for r in state["tool_call_log"]]
    text = state["draft_response"]
    assert state["candidate_blocks_intersection"]
    assert "every limit you set" not in text and "match all stated constraints" not in text
    assert text.startswith("All the blocks below meet the limits that could be checked.")
    assert sv.UNCHECKED_COMMUTE in text


def test_a_commute_limit_alone_is_not_a_limit_the_blocks_met():
    blocks = [{"block": "1", "street": "A ST", "median_resale_price": 400000.0}, {"block": "2", "street": "A ST",
                                                                                  "median_resale_price": 410000.0}]
    state = {"user_constraints": {"towns": ["BEDOK"], "flat_types": ["4 ROOM"], "max_commute_minutes": 30,
                                  "work_destinations": ["Raffles Place"]},
             "candidate_blocks_intersection": blocks, "tool_call_log": [],
             "sql_market_results": {"query_resale_blocks_by_criteria": {"blocks": blocks}}}
    facts = sv.template_response(state)
    opening, _ = sv.code_opening(state, facts)
    assert opening.startswith("Blocks with recent 4-room sales in Bedok, cheapest first")
    assert sv.UNCHECKED_COMMUTE in facts
    evidence, _ = sv.synthesis_evidence({**state, "variant": "B"})
    assert evidence["could_not_check"] == [sv.UNCHECKED_COMMUTE[2:]]


def test_a_commute_checked_for_two_workplaces_is_a_met_limit():
    blocks = [{"block": "1", "street": "A ST", "median_resale_price": 400000.0}]
    state = {"user_constraints": {"towns": ["BEDOK"], "max_commute_minutes": 30, "max_budget_sgd": 500000.0,
                                  "work_destinations": ["Raffles Place", "one-north"]},
             "candidate_blocks_intersection": blocks, "spatial_block_candidates": blocks,
             "tool_call_log": [{"agent": "spatial", "tool": "compute_dual_transit_commute", "arguments": {}, "ok": True}]}
    assert not sv.commute_unchecked(state)
    facts = sv.template_response(state)
    assert sv.UNCHECKED_COMMUTE not in facts and "match all stated constraints" in facts
    assert sv.code_opening(state, facts)[0].startswith("Only Blk 1 meets every limit you set")
    # a commute limit on a question with no block search says nothing about blocks
    assert not sv.commute_unchecked({"user_constraints": {"max_commute_minutes": 30}, "tool_call_log": []})


# --- 4. the same coordinates are accepted in A, B and C -----------------------------------------------------------
class CallsComparables:
    """Calls the comparable-sales tool once at the given point, then stops."""

    def __init__(self, lat, lng):
        self.args = {"latitude": lat, "longitude": lng, "flat_type": "4 ROOM"}
        self.called = False

    def bind_tools(self, tools, tool_choice=None):
        return self

    def invoke(self, msgs):
        if self.called:
            return AIMessage(content="done")
        self.called = True
        return AIMessage(content="", tool_calls=[{"name": COMPARABLES, "args": self.args, "id": "c1"}])


TYPED = "Show comparable 4-room resale sales near latitude 1.3521, longitude 103.9440"


def run_variant_node(variant, query, lat, lng, pin=None):
    """One model call of A's monolith, B's data agent or C's market agent with comparable sales asked for."""
    state = {"messages": [{"role": "user", "content": query}], "variant": variant, "map_ui_payload": {"center": pin} if pin else {},
             "user_constraints": {"requested_tools": [COMPARABLES]}}
    node = {"A": A.monolith_node, "B": B.data_agent_node, "C": sv.make_specialist_node("market_sql")}[variant]
    with patch.object(sv, "get_llm", return_value=CallsComparables(lat, lng)):
        return node(state)


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_typed_coordinates_and_the_map_pin_are_accepted_in_every_variant(variant):
    for query, point, pin in [(TYPED, (1.3521, 103.9440), None),
                              ("Show comparable 4-room resale sales around my pin", (1.3005, 103.8455), [1.3005, 103.8455])]:
        out = run_variant_node(variant, query, *point, pin=pin)
        assert not [e for e in out["errors"] if "rejected unverified coordinates" in e], (variant, query)
        assert [r for r in out["tool_call_log"] if r["tool"] == COMPARABLES and r["source"] == "model"]


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_made_up_coordinates_are_still_refused_in_every_variant(variant):
    out = run_variant_node(variant, TYPED, 1.30, 103.80)
    assert [e for e in out["errors"] if "rejected unverified coordinates" in e]


def test_typed_points_reads_singapore_coordinate_pairs():
    assert sv.typed_points(TYPED) == [(1.3521, 103.944)]
    assert sv.typed_points("pin (1.30, 103.85); budget $1.25m; 4.5 room") == [(1.3, 103.85)]
    assert sv.typed_points("103.85 then 1.30") == []    # a longitude before any latitude is no pair
    assert sv.trusted_points({"map_ui_payload": {"center": [1.3, 103.8]}, "messages": []}) == [(1.3, 103.8)]


# --- 5. C's school guard accepts real school names ------------------------------------------------------------------
SCHOOL_Q = "Find 4-room resale flats in Bishan under $800,000 within 1km of {}"


@pytest.mark.parametrize("name", ["St. Hilda's Primary School", "CHIJ St. Nicholas Girls' School",
                                  "Anglo-Chinese School (Junior)", "Anglo-Chinese School (Primary)",
                                  "Rosyth School", "Maris Stella High School"])
def test_school_guard_accepts_real_school_names(name):
    out = sv.input_guardrail_node({"messages": [{"role": "user", "content": SCHOOL_Q.format(name)}]})
    assert out.get("draft_response") != SCHOOL_NAME_QUESTION


@pytest.mark.parametrize("words", ["a primary school", "a good school", "the school", "MOE Primary School", "Primary School"])
def test_school_guard_still_asks_when_no_school_is_named(words):
    out = sv.input_guardrail_node({"messages": [{"role": "user", "content": SCHOOL_Q.format(words)}]})
    assert out.get("draft_response") == SCHOOL_NAME_QUESTION


# --- 6. B answers a menu pick ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("mod", [B, C])
def test_a_menu_pick_gets_its_follow_up_question(monkeypatch, mod):
    monkeypatch.setenv("HDB_ADDONS", "off")
    first = mod.run("I need some help with HDB housing")
    assert (first.get("clarification") or {}).get("kind") == "menu"
    second = mod.run("1", previous=first)
    assert second["menu_followup"]
    assert second["draft_response"] == second["menu_followup"]["question"] != first["draft_response"]
    assert second["draft_response"].startswith("Tell me the town, flat type and budget")
    assert second["clarification"] == second["menu_followup"] and not second["tool_call_log"]
