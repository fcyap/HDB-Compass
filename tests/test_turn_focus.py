"""HDB_ANSWER_STYLE focused (and conversational): each turn answers what this message asks, and the household facts are
asked a few at a time. With the switch unset (template) every answer is as measured in Experiment 1. No model needed."""
from __future__ import annotations

import os

import pytest

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402

SEARCH = "help me find 4 room resale flat in woodlands"
GRANT = "Help me check the amount of grant i will get"


def conversation(*turns):
    states, prev = [], None
    for text in turns:
        prev = C.graph.invoke(sv.initial_state(query=text, variant="C", map_state={}, previous=prev, fast_path=True))
        states.append(prev)
    return states


def tools(state):
    return [getattr(t, "tool", None) or (t.get("tool") if isinstance(t, dict) else None) for t in state.get("tool_call_log") or []]


@pytest.fixture
def focused(monkeypatch):
    monkeypatch.setenv("HDB_ANSWER_STYLE", "focused")


def test_the_switch_refuses_unknown_values(monkeypatch):
    monkeypatch.setenv("HDB_ANSWER_STYLE", "chatty")
    with pytest.raises(ValueError, match="HDB_ANSWER_STYLE"):
        sv.answer_style()
    monkeypatch.delenv("HDB_ANSWER_STYLE")
    assert sv.answer_style() == "template"


def test_a_grant_question_after_a_search_does_not_search_again(focused):
    _, t2 = conversation(SEARCH, GRANT)
    assert t2["user_constraints"]["turn_focus"] == "other"
    assert t2["active_subagents"] == ["financial"]
    assert "query_resale_blocks_by_criteria" not in tools(t2)
    assert "cheapest of" not in t2["draft_response"] and "candidate block" not in t2["draft_response"]


def test_template_still_lists_the_search_as_measured(monkeypatch):
    monkeypatch.delenv("HDB_ANSWER_STYLE", raising=False)
    _, t2 = conversation(SEARCH, GRANT)
    assert t2["user_constraints"].get("turn_focus") is None
    assert "market_sql" in t2["active_subagents"] and "cheapest of" in t2["draft_response"]
    clar = t2["clarification"]
    assert "asked" not in clar and len(clar["needed"]) > 5      # every fact asked at once, as before


@pytest.mark.parametrize("follow_up", ["within 300m of an MRT", "what about Bedok", "Find 3-room flats in Tampines under 500k"])
def test_a_change_to_the_search_still_searches(focused, follow_up):
    _, t2 = conversation(SEARCH, follow_up)
    assert t2["user_constraints"]["turn_focus"] == "search"
    assert "market_sql" in t2["active_subagents"]


def test_a_comparison_of_named_blocks_runs_the_comparison_only(focused):
    _, t2 = conversation(SEARCH, "Compare Blk 458 Tampines St 42 and Blk 216 Ang Mo Kio Ave 1")
    # a self-contained comparison is read on its own by the fast path (no history, so no focus is needed)
    assert t2["user_constraints"].get("turn_focus") in (None, "other")
    assert "query_resale_blocks_by_criteria" not in tools(t2)
    assert "compare_resale_options" in tools(t2)


def test_the_household_facts_are_asked_a_few_at_a_time(focused):
    _, t2, t3 = conversation(SEARCH, GRANT, "We are both Singapore citizens, first-timers, earning $9,000 a month.")
    clar = t2["clarification"]
    assert clar["asked"] == ["citizenship_pair", "first_timer_status", "monthly_income"] and clar["form"]
    assert len(clar["needed"]) > len(clar["asked"])                   # the full list is still read
    assert clar["question"].startswith("To work out your grant, could you tell me")
    nxt = t3["clarification"]["asked"]
    assert nxt and not set(nxt) & {"citizenship_pair", "first_timer_status", "monthly_income"}   # the next facts


def test_a_budget_question_also_asks_cpf_and_cash(focused):
    (t,) = conversation("What is our maximum budget for a resale flat?")
    asked = t["clarification"]["asked"]
    assert len(asked) <= 5 and "cpf_oa_balance" in asked and "cash_savings" in asked


def test_the_page_form_shows_the_asked_facts_only():
    from backend.main import chip_clarification
    assert chip_clarification({"kind": "financial", "needed": ["a", "b", "c", "d"], "asked": ["a", "b", "c"],
                               "form": True, "question": "q"})["needed"] == ["a", "b", "c"]
    assert chip_clarification({"kind": "financial", "needed": ["a", "b"], "asked": ["a"], "form": False,
                               "question": "q"})["needed"] == []
    assert chip_clarification({"kind": "financial", "needed": ["a", "b"], "question": "q"})["needed"] == ["a", "b"]
