"""
Feature 1, "Can we afford THIS flat?": detection, the flat's price and lease for the calculators, the verdict in the
answer, the household-form sentences (frontend/src/components/HouseholdForm.tsx composeHouseholdMessage) and the
fact-parser readings they rely on. No model needed.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("LANGSMITH_TRACING", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import agents.research_intents as ri  # noqa: E402
import agents.supervisor as sv  # noqa: E402

BLOCK = "Blk 231 Tampines St 21"


def _intents(q):
    return ri._place_intents(q, ri.extract_addresses(q))


# --- detection ---------------------------------------------------------------------------------------------------
def test_named_block_afford_question_is_flat_specific():
    i = _intents("Can we afford a 4-room at Blk 231 Tampines St 21?")
    assert i["wants_flat_affordability"] and i["wants_price_assessment"]


def test_card_button_wording_is_flat_specific():
    i = _intents("Can we afford a 4 ROOM at Blk 231 TAMPINES ST 21?")
    assert i["wants_flat_affordability"]


def test_town_afford_question_is_not_flat_specific():
    assert not _intents("Can I afford a 5-room in Tampines?").get("wants_flat_affordability")


def test_income_cpf_cash_are_not_the_flat_price():
    i = _intents("We earn $9,000 a month with CPF OA of $80,000 and $40,000 cash. Can we afford Blk 231 Tampines St 21?")
    assert i["wants_flat_affordability"] and not i.get("asking_price_sgd")


def test_asking_price_is_the_flat_price():
    i = _intents("Can we afford Blk 231 Tampines St 21 at asking $650k? We earn $9,000 a month and have $40,000 cash.")
    assert i["asking_price_sgd"] == 650000


# --- price source order -------------------------------------------------------------------------------------------
def _fake_assess(same_count, same_median=500000.0):
    def f(address, flat_type, asking_price=None, **kw):
        return {"status": "ok", "location": {"name": address}, "metadata": {"window_start_month": "2025-10", "window_end_month": "2026-09"},
                "same_block": {"transaction_count": same_count, "median": same_median},
                "nearby": {"transaction_count": 40, "median": 620000.0}, "remaining_lease_years": 60.0}
    return f


@pytest.fixture
def patch_assess(monkeypatch):
    import mcp_server.market_research_tools as m

    def apply(same_count):
        monkeypatch.setattr(m, "assess_resale_price", _fake_assess(same_count))
    return apply


def test_price_source_asking_then_same_block_then_nearby(patch_assess):
    c = {"addresses": [BLOCK], "flat_types": ["4 ROOM"], "asking_price_sgd": 650000}
    patch_assess(5)
    f = ri.flat_purchase_facts(c)
    assert f["purchase_price"] == 650000 and "asking price" in f["price_source"]
    f = ri.flat_purchase_facts({"addresses": [BLOCK], "flat_types": ["4 ROOM"]})
    assert f["purchase_price"] == 500000 and "sales at" in f["price_source"] and f["remaining_lease_years"] == 60.0
    patch_assess(1)
    f = ri.flat_purchase_facts({"addresses": [BLOCK], "flat_types": ["4 ROOM"]})
    assert f["purchase_price"] == 620000 and "nearby" in f["price_source"]


# --- parser readings the form relies on ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,pair", [
    ("We are both Singapore citizens and first-time buyers.", "SC_SC"),
    ("We are both Singapore citizens.", "SC_SC"),
    ("We are a Singapore citizen couple.", "SC_SC"),
    ("One of us is a Singapore citizen and the other is a PR.", "SC_PR"),
    ("I am a single Singapore citizen.", "SC_Single"),
    ("One of us is a Singapore citizen.", None),
    ("We are both PRs.", None),
    ("My wife is a foreigner and I am a Singapore citizen.", None),
])
def test_citizenship_pair(text, pair):
    assert sv.citizenship_stated(text) == pair


@pytest.mark.parametrize("text,value", [
    ("We are buying within 4 km of our parents.", "near"),
    ("Our parents will live with us.", "with"),
    ("We are buying with our parents.", "with"),
    ("We are not buying near our parents.", "none"),
    ("We are not living near our parents.", "none"),
    ("We do not qualify for the proximity grant.", "none"),
    ("My parents live with my brother.", None),
])
def test_proximity(text, value):
    got = sv.stated_financial_inputs(text).get("grants", {}).get("proximity")
    assert got == value
    if value:
        assert sv.fact_is_grounded("proximity", value, text)


def test_proximity_eligibility_confirmed():
    t = "Our proximity grant eligibility is confirmed."
    assert sv.stated_financial_inputs(t)["grants"]["proximity_eligibility_confirmed"] is True
    assert sv.fact_is_grounded("proximity_eligibility_confirmed", True, t)
    assert "proximity_eligibility_confirmed" not in sv.stated_financial_inputs(
        "Our proximity grant eligibility is not confirmed.").get("grants", {})


# The exact sentences composeHouseholdMessage sends (couple, then single)
COUPLE = ("We are both Singapore citizens. We are first-time buyers. We are aged 42 and 40. We have no children. "
          "Our household income is $9,000 a month. Our CPF OA balance is $80,000. We have $40,000 in cash savings. "
          "We have no monthly debt. We have both worked continuously for the past 12 months. "
          "We are buying within 4 km of our parents. Our grant eligibility is confirmed. "
          "Our proximity grant eligibility is confirmed. Our HDB loan eligibility is confirmed.")
SINGLE = ("I am a single Singapore citizen. I am not a first-time buyer. I am 35 years old. "
          "The youngest person who will live in the flat is 3 years old. My household income is $5,000 a month. "
          "My CPF OA balance is $50,000. I have $20,000 in cash savings. My monthly debt repayments are $500. "
          "I have worked continuously for the past 12 months. I am buying within 4 km of my parents.")


def _flat(reading):
    return {k: v for sec in reading.values() for k, v in sec.items()}


def test_form_sentence_couple_is_fully_read():
    f = _flat(sv.stated_financial_inputs(COUPLE))
    assert f["citizenship_pair"] == "SC_SC" and f["first_timer_status"] is True
    assert (f["youngest_buyer_age"], f["oldest_buyer_age"], f["youngest_core_member_age"]) == (40, 42, 40)
    assert (f["monthly_income"], f["cpf_oa_balance"], f["cash_savings"], f["monthly_debt_obligations"]) == (9000, 80000, 40000, 0)
    assert f["continuous_employment_12_months"] is True and f["proximity"] == "near"
    assert f["eligibility_confirmed"] and f["proximity_eligibility_confirmed"] and f["loan_eligibility_confirmed"]
    for k in ("citizenship_pair", "first_timer_status", "proximity", "eligibility_confirmed", "proximity_eligibility_confirmed",
              "loan_eligibility_confirmed", "continuous_employment_12_months", "monthly_income", "cpf_oa_balance",
              "cash_savings", "monthly_debt_obligations", "youngest_core_member_age"):
        assert sv.fact_is_grounded(k, f[k], COUPLE), k


def test_form_sentence_single_with_child_and_debt():
    f = _flat(sv.stated_financial_inputs(SINGLE))
    assert f["citizenship_pair"] == "SC_Single" and f["first_timer_status"] is False
    assert f["youngest_buyer_age"] == 35 and f["youngest_core_member_age"] == 3
    assert f["cpf_oa_balance"] == 50000 and f["cash_savings"] == 20000 and f["monthly_debt_obligations"] == 500
    assert f["proximity"] == "near" and f["continuous_employment_12_months"] is True
    assert sv.fact_is_grounded("youngest_core_member_age", 3, SINGLE)


# --- end to end ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def graph():
    from agents.arch_3_specialist_5agent import build_graph
    return build_graph()


def _run(graph, turns):
    prev, outs = None, []
    for q in turns:
        prev = graph.invoke(sv.initial_state(query=q, variant="C", previous=prev, fast_path=True))
        outs.append(prev)
    return outs


def test_two_turn_flow_gives_verdict(graph):
    first, second = _run(graph, ["Can we afford a 4 ROOM at Blk 231 TAMPINES ST 21?", COUPLE])
    tools = [l.get("tool") for l in first.get("tool_call_log") or []]
    assert "assess_resale_price" in tools and "query_resale_blocks_by_criteria" not in tools
    clar = first["clarification"]
    assert clar["kind"] == "financial" and "cash_savings" in clar["needed"]
    assert "Buyer's stamp duty" in first["draft_response"]
    assert second.get("clarification") is None
    text = second["draft_response"]
    assert "**Can you afford Blk 231 TAMPINES ST 21?**" in text
    fp = second["user_constraints"]["flat_purchase"]
    ceiling = second["financial_breakdown"]["calculate_maximum_affordability"]["max_purchase_price"]
    gap = ceiling - fp["purchase_price"]
    word = "within" if gap >= 0 else "above"
    assert f"{word} your ceiling by ${abs(gap):,.0f}" in text
    assert f"${fp['purchase_price']:,.0f}" in text and "excludes stamp duty" in text


def test_household_facts_carry_over_from_earlier_turn(graph):
    outs = _run(graph, [COUPLE.split(" Our grant")[0] + " Can we afford Blk 231 Tampines St 21?"])
    needed = (outs[0].get("clarification") or {}).get("needed") or []
    assert "cash_savings" not in needed and "monthly_income" not in needed
