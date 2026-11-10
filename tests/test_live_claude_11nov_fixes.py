"""
Fixes for the money flow found in the live Claude replay of 11 Nov (125 conversations, LIVE_CLAUDE_11NOV.md). Owned by
Member 4.

  * Grant questions never reached a figure: code asked "whether your grant eligibility is confirmed" (only an HFE
    letter can confirm it), the youngest occupant's age for a couple, 12 months' work and the Proximity Housing Grant.
    Code now assumes the usual case once citizenship, first-timer status, income and ages are known, and says so.
  * "Within 4 km of our parents" with PHG eligibility not confirmed gave "Conflicting proximity inputs" and no grant.
  * A grants-only question was asked for CPF, cash and debts; Financials listed raw field names ("cpf oa balance").
  * HDB loan eligibility was asked for; it is assumed (and said) once the money facts are in.
  * "How much cash do we need upfront?" with the loan given asked for income, ages and an "ltv ratio".
  * A single aged 34 was told they may get grants; "34 years old" was not read as the buyer's age.

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_live_claude_11nov_fixes.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402

GRANTS = "calculate_cpf_housing_grants"
AFFORD = "calculate_maximum_affordability"
UPFRONT = "calculate_bsd_and_upfront_outlay"


def app(query, previous=None):
    return C.graph.invoke(sv.initial_state(query=query, variant="C", previous=previous, fast_path=True))


def conversation(*turns):
    states, prev = [], None
    for text in turns:
        prev = app(text, prev)
        states.append(prev)
    return states


def question(state):
    return (state.get("clarification") or {}).get("question") or ""


def assumptions(state):
    return (state.get("user_constraints") or {}).get("assumptions") or []


def test_l_f45_grant_figure_once_the_household_is_described():
    t1, t2 = conversation("How much grant can we get?",
                          "We are both Singapore citizens. We are first-time buyers. We are aged 32 and 30. "
                          "Our household income is $9,000 a month.")
    assert "grant eligibility is confirmed" not in question(t1) and "employed continuously" not in question(t1)
    g = t2["financial_breakdown"][GRANTS]
    assert g["status"] == "ok" and g["total_grants"] == 85000
    assert not t2.get("clarification")       # a grants-only question: CPF, cash and debts are not asked for
    for note in (sv.GRANT_CONDITIONS_NOTE, sv.GRANT_EMPLOYMENT_NOTE, sv.NO_PHG_NOTE, sv.BUYERS_ONLY_NOTE):
        assert note in assumptions(t2)
    assert "Affordability: more information needed" not in t2["draft_response"]


def test_a_child_is_still_asked_about():
    s = app("We are both Singapore citizens, first-time buyers aged 32 and 30 with a son. Our household income is "
            "$9,000 a month. What grants can we get?")
    assert s["financial_breakdown"][GRANTS]["status"] == "needs_information"
    assert "youngest person who will live in the flat" in question(s)


def test_l_m_19_unconfirmed_phg_near_parents_still_gives_the_other_grants():
    s = app("We are both Singapore citizens. We are first-time buyers. We are aged 30 and 29. We have no children. Our "
            "household income is $7,000 a month. We have both worked continuously for the past 12 months. We are "
            "buying within 4 km of our parents. Our grant eligibility is confirmed. Our proximity grant eligibility is "
            "not confirmed. We want a 4 ROOM flat. Assume 70 year lease. What grants can we get?")
    g = s["financial_breakdown"][GRANTS]
    assert g["status"] == "ok" and g["proximity_housing_grant"] == 0 and g["total_grants"] == 110000


def test_l_x_05_confirmed_phg_with_every_fact_gives_the_figure():
    s = app("We are both Singapore citizens. We are first-time buyers. We are aged 30 and 29. We have no children. Our "
            "household income is $7,000 a month. We have both worked continuously for the past 12 months. We are "
            "buying within 4 km of our parents. Our proximity grant eligibility is confirmed. We want a 4 ROOM flat. "
            "Assume 70 year lease. What grants can we get?")
    assert s["financial_breakdown"][GRANTS]["total_grants"] == 130000
    assert not s.get("clarification")


def test_l_ft_sc_budget_without_asking_hdb_loan_eligibility():
    t1, t2 = conversation("We are first-timer Singapore citizens. How much can we afford for a 4-room resale?",
                          "Me and my hubby are 33 and 35. Household income $9,000 a month, CPF OA $80,000, cash "
                          "$30,000, no debts.")
    assert "HDB loan eligibility" not in question(t1)
    assert t2["financial_breakdown"][AFFORD]["status"] == "ok" and not t2.get("clarification")
    assert sv.HDB_LOAN_NOTE in assumptions(t2)


def test_l_m_37_upfront_cash_with_the_loan_given():
    s = app("We are buying a 4-room for $600,000. It is valued at $600,000. We take an HDB loan of $450,000. Our CPF OA "
            "balance is $80,000. We have $40,000 in cash savings. We have no housing grants. Legal fees of $3,000. No "
            "ABSD. How much cash do we need upfront?")
    fin = s["financial_breakdown"]
    assert fin[UPFRONT]["status"] == "ok" and AFFORD not in fin
    assert "ltv" not in s["draft_response"].lower() and not s.get("clarification")


def test_l_x_03_a_single_aged_34_is_told_the_age_rule():
    s = app("I am a single Singapore citizen, 34 years old, first-time buyer, earning $4,000 a month. I have no "
            "children. What grants can I get?")
    assert s["user_constraints"]["financial_inputs"]["grants"]["youngest_buyer_age"] == 34
    assert sv.SINGLE_UNDER_35_NOTE in assumptions(s) and not s.get("clarification")
    assert sv.GRANT_CONDITIONS_NOTE not in assumptions(s)


def test_years_old_is_read_only_for_the_buyer():
    assert sv.buyer_ages("I am a single Singapore citizen, 34 years old, first-time buyer.") == [34]
    assert sv.buyer_ages("I'm a 34-year-old single") == [34]
    assert sv.buyer_ages("The flat is 40 years old.") == []
    assert sv.buyer_ages("My son is 12 years old and I am single") == []


def test_financials_show_plain_words_not_field_names():
    s = app("How much can we afford? We are both Singapore citizens, first-timers, income $9,000 a month.")
    lines = [ln for ln in s["draft_response"].splitlines() if "more information needed" in ln]
    assert lines and not any(f in " ".join(lines) for f in ("cpf oa balance", "citizenship pair", "total grants",
                                                             "eligibility confirmed"))


# --- openings that overclaimed, and a price question answered with a block list (live on Claude, 11 Nov) -------------
def test_l_f_61_an_unnamed_school_is_not_claimed_as_checked():
    s = app("I need a 4 room in Punggol, budget around 600k, near a primary school")
    assert s["draft_response"].startswith("All the blocks below meet the limits that could be checked")
    assert "every limit you set" not in s["draft_response"].split("\n")[0]


def test_l_f_24_an_unknown_station_is_said_and_asked_about():
    assert sv.unknown_station_named("4-room flats near Fake Street MRT under 600k") == "Fake Street"
    assert sv.unknown_station_named("4-room flats near Novena MRT") is None
    assert sv.unknown_station_named("4-room flats near an MRT station") is None
    s = app("4-room flats near Fake Street MRT under 600k")
    assert "every limit you set" not in s["draft_response"].split("\n")[0]
    assert "Fake Street" in question(s) or any("Fake Street" in q for q in s["user_constraints"].get("clarifications") or [])


def test_l_m_14_a_price_question_is_judged_against_the_town_median():
    assert sv.town_price_question("is 600k too much for 4 room in AMK") == 600000
    assert sv.town_price_question("4-room in Bedok under 600k") is None
    s = app("is 600k too much for 4 room in AMK")
    assert s["draft_response"].startswith("No: that price is below the latest quarter's median for 4-room flats in Ang Mo Kio")
    assert "candidate block" not in s["draft_response"] and "median of $" in s["draft_response"]
    s = app("Is $900k too expensive for a 4-room in Tampines?")
    assert s["draft_response"].startswith("Yes: that price is above")


def test_l_f_33_we_work_at_one_place_times_that_commute():
    s = app("We work at Changi Airport. 4 room in Tampines under 650k, commute max 30 min")
    assert s["user_constraints"]["work_destinations"] == ["Changi Airport", "Changi Airport"]
    assert "second workplace" not in s["draft_response"] and "commute to Changi Airport:" in s["draft_response"]
    assert "workplace 2" not in s["draft_response"]


def test_l_m_28_a_zero_budget_is_asked_about():
    s = app("4 room in Tampines under $0")
    assert "budget of $0" in question(s)
    assert "no blocks with recent sales" not in s["draft_response"]
    assert not sv._ZERO_BUDGET.search("under $0.5m") and not sv._ZERO_BUDGET.search("under $500k")


def test_an_auditor_refusal_removes_only_the_ungrounded_sentences():
    from agents import reply_composer as rc
    reply = ("For a standard BTO flat, the minimum occupation period is 5 years before you can sell it on the open "
             "market. The clock starts from the legal completion date of your flat purchase. If your BTO is classified "
             "as Plus or Prime, the rules differ.")
    kept = rc.drop_claims(reply, ["The clock starts from the legal completion date of your flat purchase",
                                  "If your BTO is classified as Plus or Prime, the rules differ"])
    assert kept == ("For a standard BTO flat, the minimum occupation period is 5 years before you can sell it on the "
                    "open market.")
    assert rc.drop_claims(reply, []) == ""
