"""Conversation memory meets the research tools: which place a follow-up is about, what carries over between
turns, and when the app's fast path may skip the model.

Rules (agents/research_intents.resolve_place): "here"/"this block" means the map pin; "there"/"that block" may mean
the pin or a place named earlier, so the app asks when both are possible; a place question naming no place uses the
pin, else the earlier place, and asks when the two differ. A reply such as "the pin" answers that question.
"""
from __future__ import annotations

import pytest

from agents import research_intents as ri
from agents import supervisor as sv
from tests.test_research_tools import research_env, snapshot  # noqa: F401  (shared fixtures)

PIN = (1.35001, 103.85001)          # Blk 1 BISHAN ST 1 in the synthetic index
EARLIER = ["Blk 2 Bishan St 1"]


@pytest.fixture
def deterministic(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")


def resolve(query, pin=None, earlier=(), pending=None):
    c = sv.rule_based_extract(query).model_dump()
    ri.resolve_place(c, query, pin, list(earlier), pending)
    return c


# --- the rules, one by one -----------------------------------------------------------------------------------------
def test_here_means_the_pin_even_with_an_earlier_place(research_env):
    c = resolve("What's near here?", PIN, EARLIER)
    assert c["addresses"] == ["Blk 1 BISHAN ST 1"] and not c.get("place_options")


def test_there_with_a_pin_and_an_earlier_place_asks(research_env):
    c = resolve("What's near there?", PIN, EARLIER)
    assert [o["source"] for o in c["place_options"]] == ["pin", "conversation"]
    assert "Blk 1 BISHAN ST 1" in c["clarifications"][0] and "Blk 2 Bishan St 1" in c["clarifications"][0]
    assert sv.decide_routing(c) == []        # nothing runs until the user says which place


def test_there_without_a_pin_uses_the_earlier_place(research_env):
    c = resolve("What's near there?", None, EARLIER)
    assert c["addresses"] == EARLIER and c["wants_nearby_amenities"]
    assert any("mentioned earlier" in a for a in c["assumptions"])


def test_is_there_is_not_a_pointing_word(research_env):
    c = resolve("Is there an MRT near Blk 5 Tampines St 1?", PIN, EARLIER)
    assert c["addresses"] == ["Blk 5 Tampines St 1"] and not c.get("place_options")


def test_a_named_place_wins(research_env):
    c = resolve("What have flats at Blk 5 Tampines St 1 sold for?", PIN, EARLIER)
    assert c["addresses"] == ["Blk 5 Tampines St 1"] and not c.get("place_options")


def test_place_question_with_pin_and_a_different_earlier_place_asks(research_env):
    c = resolve("What amenities are nearby?", PIN, EARLIER)
    assert c.get("place_options")


def test_place_question_where_pin_and_earlier_place_agree_does_not_ask(research_env):
    c = resolve("What amenities are nearby?", PIN, ["Blk 1 Bishan St 1"])
    assert not c.get("place_options") and c["addresses"] == ["Blk 1 BISHAN ST 1"]


def test_place_question_with_only_an_earlier_place_uses_it(research_env):
    c = resolve("What amenities are nearby?", None, EARLIER)
    assert c["addresses"] == EARLIER and any("mentioned earlier" in a for a in c["assumptions"])


@pytest.mark.parametrize("reply, expected", [
    ("the pin", "Blk 1 BISHAN ST 1"),
    ("the one on the map", "Blk 1 BISHAN ST 1"),
    ("the earlier one", "Blk 2 Bishan St 1"),
    ("Blk 2 Bishan St 1", "Blk 2 Bishan St 1"),
    ("hmm", "Blk 1 BISHAN ST 1"),        # unclear after one ask: the pin, said so
])
def test_reply_to_which_place(research_env, reply, expected):
    asked = resolve("What's near there?", PIN, EARLIER)
    pending = sv.place_clarification({"user_constraints": asked})
    assert pending["kind"] == "location"
    c = resolve(reply, None, EARLIER, pending)
    assert c["addresses"][0] == expected


# --- whole conversations through Variant C --------------------------------------------------------------------------
def test_three_turn_conversation_asks_then_answers_about_the_pin(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    first = C.run("What have flats at Blk 2 Bishan St 1 sold for?")
    assert [r["tool"] for r in first["tool_call_log"]] == ["get_block_transaction_history"]
    pin = {"center": list(PIN), "radius_meters": 1000}
    second = C.run("What's near there?", map_state=pin, previous=first)
    assert second["tool_call_log"] == [] and second["clarification"]["kind"] == "location"
    assert "Do you mean the place you pinned" in second["draft_response"]
    third = C.run("the pin", map_state=pin, previous=second)
    assert [r["tool"] for r in third["tool_call_log"]] == ["get_nearby_amenities"]
    assert "Near Blk 1 BISHAN ST 1" in third["draft_response"]


def test_a_new_question_does_not_inherit_the_last_ones_asks(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    first = C.run("What have flats at Blk 2 Bishan St 1 sold for?")
    second = C.run("How is the HDB resale market doing now?", previous=first)
    tools = [r["tool"] for r in second["tool_call_log"]]
    assert "get_market_snapshot" in tools and "get_block_transaction_history" not in tools


def test_fast_path_reads_one_message_and_never_answers_a_follow_up(research_env, monkeypatch):
    calls = []
    monkeypatch.setattr(sv, "get_llm", lambda *a, **k: calls.append(1))   # any model use would be recorded
    st = sv.initial_state("What amenities are near Blk 1 Bishan St 1?", "C", fast_path=True)
    out = sv.extract_constraints_node(st)
    assert out["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE and calls == []
    previous = {"messages": st["messages"], "user_constraints": out["user_constraints"],
                "clarification": {"kind": "financial", "needed": ["cash_savings"], "question": "?"}}
    reply = sv.initial_state("What amenities are near Blk 1 Bishan St 1?", "C", previous=previous, fast_path=True)
    assert sv.extract_constraints_node(reply)["user_constraints"]["extraction_source"] != sv.FAST_PATH_SOURCE


def test_place_question_is_recorded_for_the_next_turn(research_env, deterministic):
    from agents import arch_3_specialist_5agent as C
    out = C.run("Tell me about Blk 1 Bishan St 1")
    assert out["clarification"]["kind"] == "place"
    follow = C.run("its past sales", previous=out)
    assert [r["tool"] for r in follow["tool_call_log"]] == ["get_block_transaction_history"]


def test_fast_path_still_asks_between_pin_and_earlier_place(research_env, monkeypatch):
    # found by the app smoke test: the fast path read only this message, so "there" silently meant the pin
    monkeypatch.setattr(sv, "get_llm", lambda *a, **k: None)
    first = {"messages": [{"role": "user", "content": "What have flats at Blk 2 Bishan St 1 sold for?"}],
             "user_constraints": sv.rule_based_extract("What have flats at Blk 2 Bishan St 1 sold for?").model_dump()}
    st = sv.initial_state("What's near there?", "C", map_state={"center": list(PIN)}, previous=first, fast_path=True)
    c = sv.extract_constraints_node(st)["user_constraints"]
    assert c.get("place_options") and sv.decide_routing(c) == []


@pytest.mark.parametrize("phrase", ["the pin", "there", "the earlier one", "this block", "my pinned location"])
def test_a_pointing_phrase_copied_into_addresses_is_not_a_place(research_env, phrase):
    # found with a hosted model: it extracted "the pin" as an address, so the reply searched for a place called that
    asked = resolve("What's near there?", PIN, EARLIER)
    pending = sv.place_clarification({"user_constraints": asked})
    c = {**sv.rule_based_extract("the pin").model_dump(), "addresses": [phrase]}
    ri.resolve_place(c, "the pin", None, EARLIER, pending)
    assert c["addresses"] == ["Blk 1 BISHAN ST 1"]


# --- generic amenity words are not places ------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    "What are the historical statistics of HDBs sold near MRT stations?", "Flats near an MRT", "near the LRT",
    "homes near MRTs", "near hawker centres", "near supermarkets", "near a primary school"])
def test_generic_amenity_words_are_not_addresses(query):
    from agents.research_intents import extract_addresses
    assert extract_addresses(query) == []


@pytest.mark.parametrize("query, place", [
    ("Prices near Bishan MRT", "Bishan MRT"), ("flats near Toa Payoh hawker centre", "Toa Payoh hawker centre"),
    ("near Ai Tong School", "Ai Tong School")])
def test_named_amenities_are_still_places(query, place):
    from agents.research_intents import extract_addresses
    assert extract_addresses(query) == [place]
