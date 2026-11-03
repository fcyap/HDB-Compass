"""
The items the 3 Nov QA sweep left open, now handled by the Supervisor (tests/test_qa_open_items.py). Owned by Member 4.

  * "between $600k and $650k" applies both ends (the lower one by code, the price search has only an upper limit).
  * Follow-ups about the blocks just listed: "which of those is cheapest?", "stamp duty on the first one", and
    "within 1km" after a map pin or with several distance limits (asked which).
  * A monthly instalment is worked out by code from the loan, rate and tenure (asked for when one is missing).
  * Two PRs get no CPF housing grants, and a second-timer's budget is still worked out, both with $0 grants.
  * "Can we afford a $700k flat?" says whether the price is within the ceiling.
  * The offline policy headline quotes a line that shares the question's words, or none.

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_qa_open_items.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402


def app(query, previous=None, map_state=None):
    return C.graph.invoke(sv.initial_state(query=query, variant="C", map_state=map_state or {}, previous=previous,
                                           fast_path=True))


def chat(*turns, map_state=None):
    st = None
    for q in turns:
        st = app(q, st, map_state)
    return st


HOUSEHOLD = ("We are both Singapore citizens, first-time buyers aged 32 and 30, no children. Household income $9,000 a "
             "month, CPF OA $90,000, cash $40,000, no debts. We have both worked continuously for the past 12 months. "
             "Our grant eligibility is confirmed. Our HDB loan eligibility is confirmed. We are not buying near our parents. ")


def test_a_price_range_applies_both_ends():
    assert sv.price_range("4 room in Bedok between $600k and $650k") == (600_000, 650_000)
    assert sv.price_range("4 room bedok 500k-600k") == (500_000, 600_000)
    assert sv.price_range("3-4 room in bedok") is None and sv.price_range("60-70 years lease") is None
    st = app("4 room in Bedok between $600k and $650k")
    prices = [b["median_resale_price"] for b in st["candidate_blocks_intersection"]]
    assert prices and all(600_000 <= p <= 650_000 for p in prices)
    assert any("between $600,000 and $650,000" in a for a in st["user_constraints"]["assumptions"])


def test_which_of_those_is_cheapest_after_another_question():
    st = chat("Find 4-room flats in Tampines under $600k within 500m of an MRT",
              "Can I rent out my whole Prime flat after the MOP?", "which of those blocks is cheapest?")
    assert st["user_constraints"]["addresses"] and st["user_constraints"].get("wants_block_history")
    assert any("is the cheapest" in a for a in st["user_constraints"]["assumptions"])


def test_stamp_duty_on_a_listed_block_uses_its_median():
    st = chat("Find 4-room resale flats in Ang Mo Kio under $750,000 with at least 65 years of lease left, "
              "within 400m of an MRT station.", "What is the stamp duty on the first one?")
    up = st["financial_breakdown"]["calculate_bsd_and_upfront_outlay"]
    assert up["buyer_stamp_duty"] > 0 and "calculate_cpf_housing_grants" not in st["financial_breakdown"]
    assert not st.get("clarification")


def test_stamp_duty_on_that_with_many_blocks_asks_which():
    st = chat("Find 4-room flats in Tampines under $600k within 500m of an MRT", "What is the stamp duty on that?")
    assert "Which block do you mean?" in (st.get("clarification") or {}).get("question", "")


def test_a_bare_radius_with_two_limits_asks_which():
    st = chat("Find 4-room flats in Bedok under $600k within 400m of an MRT station and 500m of a hawker centre",
              "within 1km")
    assert "Which limit should be 1,000 m" in (st.get("clarification") or {}).get("question", "")
    assert st["user_constraints"]["amenity_radii_meters"] == {"mrt": 400, "hawker": 500}


def test_a_bare_radius_after_a_pin_widens_the_pin():
    pin = {"center": [1.3693, 103.8484], "radius_meters": 500}
    st = chat("What's near here?", "within 1km", map_state=pin)
    assert st["user_constraints"].get("pin_radius_meters") == 1000 and "within 1000 m" in st["draft_response"]


def test_an_instalment_is_worked_out_by_code():
    assert sv.instalment_question("monthly instalment for a $500k loan at 2.6% over 25 years")["monthly"] == 2268
    st = app("What's the monthly instalment for a $500k loan at 2.6% over 25 years?")
    assert "**Monthly instalment: $2,268**" in st["draft_response"]
    st = chat("How much is the monthly instalment for my HDB loan?", "$500k at 2.6% over 25 years")
    assert "$2,268" in st["draft_response"]


def test_a_pr_couple_gets_a_budget_with_no_grants():
    st = app("We are both PRs, first-time buyers, aged 30 and 31. We have no children. Our household income is $9,000 "
             "a month, CPF OA $80,000, cash $40,000, no debts. Our HDB loan eligibility is confirmed. What's our max budget?")
    assert st["financial_breakdown"]["calculate_maximum_affordability"]["status"] == "ok"
    assert not st.get("clarification")
    assert "no CPF housing grants" in " ".join(st["user_constraints"]["assumptions"])


def test_a_second_timer_budget_is_worked_out_with_no_grants():
    st = app("We are both Singapore citizens aged 35 and 33. We are not first-time buyers. Our household income is "
             "$10,000 a month. We have $100,000 in CPF OA and $50,000 cash. No debts. We have both worked continuously "
             "for the past 12 months. Our grant eligibility is confirmed. Our HDB loan eligibility is confirmed. We are "
             "not buying near our parents. No children. What is our maximum budget?")
    assert st["financial_breakdown"]["calculate_maximum_affordability"]["status"] == "ok"
    assert "assumes no housing grants" in st["draft_response"]


def test_can_we_afford_a_price_says_within_or_above():
    st = app(HOUSEHOLD + "Can we afford a $700k 4-room flat?")
    assert "A $700,000 flat is **within** this ceiling" in st["draft_response"]
    st = app(HOUSEHOLD + "Can we afford a $900k 4-room flat?")
    assert "A $900,000 flat is **above** this ceiling" in st["draft_response"]


def test_the_policy_headline_quotes_only_a_line_on_the_question():
    st = app("Can I rent out my whole Prime flat after the 10-year MOP?")
    assert "rent out the whole flat" in st["draft_response"].split("\n")[0]
    st = app("how long must I stay in my BTO before I can sell it?")
    assert "10 continuous years" not in st["draft_response"].split("\n")[0]
