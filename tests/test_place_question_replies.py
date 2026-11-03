"""
Replies to the open question about a place (tests/test_place_question_replies.py). Owned by Member 4.

In the app (2 Nov), "Tampines blk 884" got "What would you like to know about Tampines blk 884? For example: its past
sales, ...", and the reply "resale price" got the same question again, every time. The reply is now read as the
option it picks ("resale price" and "past sales" are the block's past sales), "Tampines blk 884" is the block
"Blk 884 Tampines", and a reply no option matches gets the block's past sales instead of the question again.

    LLM_PROFILE=none HDB_ADDONS=app .venv/bin/python -m pytest tests/test_place_question_replies.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ["LLM_PROFILE"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents.schemas import LLMExtractedConstraints  # noqa: E402
from tests.test_research_orchestration import FakeLLM  # noqa: E402


@pytest.fixture(autouse=True)
def app_mode(monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "app")


def app(query, previous=None, **reading):
    """One app turn (Variant C, fast path) with a model whose reading is given."""
    with patch.object(sv, "get_llm", return_value=FakeLLM(extraction=LLMExtractedConstraints(**reading))):
        return C.graph.invoke(sv.initial_state(query=query, variant="C", map_state={}, previous=previous, fast_path=True))


@pytest.fixture
def asked():
    first = app("Tampines blk 884", addresses=["Tampines blk 884"])
    assert (first.get("clarification") or {}).get("kind") == "place"
    return first


@pytest.mark.parametrize("reply,intent,tool", [
    ("resale price", "wants_block_history", "get_block_transaction_history"),
    ("past sales", "wants_block_history", "get_block_transaction_history"),
    ("is $600k fair?", "wants_price_assessment", "assess_resale_price"),
    ("price trend", "wants_price_trend", None),
    ("amenities", "wants_nearby_amenities", "get_nearby_amenities"),
    ("resale prices around it", "wants_area_statistics", "get_area_resale_statistics"),
])
def test_a_reply_picks_the_option_and_is_answered(asked, reply, intent, tool):
    out = app(reply, asked)
    assert not out.get("clarification"), "the same question must not be asked again"
    assert out["user_constraints"][intent]
    if tool:
        assert tool in [t["tool"] for t in out["tool_call_log"]]


@pytest.mark.parametrize("reply", ["ok", "hmm not sure", "yes"])
def test_a_reply_that_picks_nothing_gets_past_sales_not_the_question_again(asked, reply):
    out = app(reply, asked)
    assert not out.get("clarification")
    assert out["user_constraints"]["wants_block_history"]
    assert "get_block_transaction_history" in [t["tool"] for t in out["tool_call_log"]]


def test_town_first_block_is_read_as_the_block():
    c = {"addresses": ["Tampines blk 884"], "clarifications": ["What would you like to know about Tampines blk 884? ..."]}
    sv.answer_place_question(c, "past sales")
    assert c["addresses"] == ["Blk 884 Tampines"] and c["wants_block_history"] and not c["clarifications"]


def test_a_place_on_its_own_still_gets_the_question_first():
    # only a reply to the open question is read as an option; a first message naming only a place is asked about
    out = app("Tampines blk 884", addresses=["Tampines blk 884"])
    assert (out.get("clarification") or {}).get("kind") == "place"
    assert not any(v for k, v in out["user_constraints"].items() if k.startswith("wants_"))


# --- a lookup that calls one candidate "ambiguous" is asked again for that candidate (tool_registry.ToolRunner) ------
def test_a_single_candidate_lookup_runs_again_for_that_place():
    from agents.tool_registry import ToolRunner
    runner = ToolRunner("get_block_transaction_history")
    runner.source = "model"
    out = runner(address="Blk 884 Tampines")   # one Blk 884 in Tampines: TAMPINES ST 83
    assert out["status"] == "ok" and out["transaction_count"] > 0
    assert [(r.source, r.arguments["address"]) for r in runner.log] == [
        ("model", "Blk 884 Tampines"), ("code", "Blk 884 TAMPINES ST 83")]
    assert runner.source == "model"


def test_several_candidates_are_left_for_the_user_to_choose():
    from agents.tool_registry import _single_candidate
    two = {"status": "ambiguous", "candidates": [{"name": "Blk 1 A ST"}, {"name": "Blk 1 B ST"}]}
    assert _single_candidate(two, {"address": "Blk 1"}) is None
    assert _single_candidate({"status": "ok"}, {"address": "Blk 1"}) is None
    assert _single_candidate({"status": "ambiguous", "candidates": [{"name": "Blk 1 A ST"}]}, {}) is None


def test_the_reply_resale_price_now_shows_the_blocks_past_sales(asked):
    out = app("resale price", asked)
    calls = [t for t in out["tool_call_log"] if t["tool"] == "get_block_transaction_history"]
    assert calls and calls[-1]["arguments"]["address"] == "Blk 884 TAMPINES ST 83"
    assert "ask the user which one they mean" not in out["draft_response"]


# --- the second app conversation (2 Nov): "Resale price for Tampines Blk 884?", then "4 room" -----------------------
@pytest.mark.parametrize("text,expected", [
    ("Resale price for Tampines Blk 884?", "Resale price for Blk 884 Tampines?"),
    ("ang mo kio blk 225a", "Blk 225a ANG MO KIO"),
    ("Tampines Blk 884 Tampines St 83", "Tampines Blk 884 Tampines St 83"),   # a street follows: already an address
    ("4-room in Tampines under $600k", "4-room in Tampines under $600k"),
])
def test_a_town_then_a_block_number_is_read_as_the_block(text, expected):
    assert sv.town_first_block(text).lower() == expected.lower()


def test_a_question_that_names_a_block_and_asks_its_price_is_answered_not_asked_back():
    out = C.graph.invoke(sv.initial_state(query="Resale price for Tampines Blk 884?", variant="C", map_state={},
                                          previous=None, fast_path=True))
    assert not out.get("clarification")
    assert out["user_constraints"]["wants_block_history"]
    assert "Blk 884 TAMPINES ST 83" in out["draft_response"]


def test_a_model_block_search_with_no_budget_or_lease_runs_with_no_limit():
    from langchain_core.messages import AIMessage
    first = C.graph.invoke(sv.initial_state(query="Resale price for Tampines Blk 884?", variant="C", map_state={},
                                            previous=None, fast_path=True))
    null_call = AIMessage(content="", tool_calls=[{"name": "query_resale_blocks_by_criteria", "id": "c1", "args": {
        "towns": ["TAMPINES"], "flat_types": ["4 ROOM"], "max_price": None, "min_lease_years": None}}])
    llm = FakeLLM(extraction=LLMExtractedConstraints(towns=["TAMPINES"], flat_types=["4 ROOM"]), script=[null_call])
    with patch.object(sv, "get_llm", return_value=llm):
        out = C.graph.invoke(sv.initial_state(query="4 room", variant="C", map_state={}, previous=first, fast_path=True))
    call = next(t for t in out["tool_call_log"] if t["tool"] == "query_resale_blocks_by_criteria")
    assert call["ok"] and call["arguments"]["max_price"] == sv.NO_PRICE_LIMIT and call["arguments"]["min_lease_years"] == 0
    assert "could not find data" not in out["draft_response"]


def test_when_every_model_call_fails_code_runs_its_own_plan():
    from langchain_core.messages import AIMessage
    bad = AIMessage(content="", tool_calls=[{"name": "query_resale_blocks_by_criteria", "id": "c1",
                                             "args": {"towns": "not a list", "flat_types": 7, "block_keys": "x"}}])
    c = sv.rule_based_extract("4-room in Tampines under $600k").model_dump()
    with patch.object(sv, "get_llm", return_value=FakeLLM(script=[bad])):
        outputs, log, errors, _ = sv.run_react("market_sql", ["query_resale_blocks_by_criteria"], "sys",
                                               "4-room in Tampines under $600k", c)
    assert [r for r in log if r["source"] == "model" and not r["ok"]]
    assert [r for r in log if r["source"] == "code" and r["ok"]]
    assert any("deterministic plan used" in e for e in errors)


def test_a_model_sentence_that_repeats_a_facts_line_is_dropped():
    text, dropped = sv.keep_supported_sentences("I could not find data for that request.",
                                                "I could not find data for that request.")
    assert text == "" and dropped == 0   # nothing is lost: not reported as a dropped sentence


# --- the live model's reading of "Resale price for Tampines Blk 884?" (2 Nov) ---------------------------------------
def test_the_live_models_reading_still_shows_the_blocks_past_sales():
    reading = dict(addresses=["Tampines Blk 884"], towns=["TAMPINES"], flat_types=["BLK 884"])
    out = app("Resale price for Tampines Blk 884?", **reading)
    c = out["user_constraints"]
    assert c["addresses"] == ["Blk 884 Tampines"] and c["flat_types"] == []
    assert any("not HDB flat types were dropped" in e for e in out["errors"])
    assert "Blk 884 TAMPINES ST 83" in out["draft_response"]
    assert sv.NO_DATA not in out["draft_response"]


def test_real_flat_types_from_the_model_are_kept():
    c = {"flat_types": ["4 room", "EXECUTIVE", "BLK 884"]}
    errors = []
    sv._verify_extraction(c, "4-room or executive in Tampines", errors, from_model=True)
    assert c["flat_types"] == ["4 ROOM", "EXECUTIVE"] and errors


@pytest.mark.parametrize("model_text", ["I could not find data for that request.",
                                        "I couldn't find any data for that request."])
def test_nothing_found_is_said_once(model_text):
    from tests.test_code_opening import Writer
    state = {"messages": [{"role": "user", "content": "Resale price for Tampines Blk 884?"}], "variant": "C",
             "user_constraints": {"towns": ["TAMPINES"]}, "tool_call_log": [], "active_subagents": ["market_sql"]}
    with patch.object(sv, "get_llm", return_value=Writer(model_text)):
        out = sv.synthesize_node(state)
    assert out["draft_response"] == sv.NO_DATA
