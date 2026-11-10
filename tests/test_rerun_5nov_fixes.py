"""Fixes for Member 4's turns that failed the independent rerun of 9653519 (5 Nov): a single buyer's age is the
youngest core member's (HOUSE-05), a price asked against the ceiling is answered, a lease stated later replaces the one
code assumed and a buyer aged 55+ is told the loan model stops there (HOUSE-09), a follow-up changing an instalment is
code's alone (DEMO-11 t2), "Use MRT stations only instead." and a search's price limit are read by code whatever the
model returns (SPATIAL-07), and a launch the message does not name or a block comparison outside a block search is
dropped (STATS-06). The model-path checks run through _finalize_constraints with a model-style reading. No model needed."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402

SINGLE_38 = ("Single citizen aged 38, first-timer, monthly income $4,500, usable CPF $75k, cash $40k, debt-free. Grants "
             "and HDB loan eligibility confirmed. Buying 3-room resale with 70 years left, no proximity grant. What can I "
             "afford?")
SENIORS = ("We are a citizen couple aged 61 and 63 with income $6k, usable CPF $150k and cash $100k. What is our max loan "
           "for a resale?")


def conversation(*turns):
    states, prev = [], None
    for text in turns:
        prev = C.graph.invoke(sv.initial_state(query=text, variant="C", map_state={}, previous=prev, fast_path=True))
        states.append(prev)
    return states


def model_reading(query, **fields):
    """_finalize_constraints on code's reading with the model's fields laid over it, as the live model path does."""
    reading = {**sv.rule_based_extract(query).model_dump(), **fields}
    errors = []
    return sv._finalize_constraints(reading, query, {"messages": [{"role": "user", "content": query}]}, errors,
                                    from_model=True), errors


# ---------------------------------------------------------------------------
# HOUSE-05: a single buyer is the youngest core member
# ---------------------------------------------------------------------------
def test_a_single_buyers_age_is_the_core_member_age():
    t1, t2, t3 = conversation(SINGLE_38, "I have worked continuously for 12 months.",
                              "If I look at a $450k flat, is that within the ceiling?")
    assert not t1.get("clarification")   # 12 months' work is assumed and said so (live on Claude, 11 Nov)
    assert t2["user_constraints"]["financial_inputs"]["grants"]["youngest_core_member_age"] == 38
    assert "Max purchase price" in t2["draft_response"]
    assert t3["draft_response"].startswith("No, that flat is above your estimated ceiling.")
    assert "A $450,000 flat is **above** this ceiling" in t3["draft_response"]
    assert "Singles $40,000" in t3["draft_response"] and "Family $" not in t3["draft_response"]


def test_a_single_buyer_living_with_a_parent_is_still_asked():
    assert sv.single_buying_alone("I am a single citizen aged 36.")
    assert not sv.single_buying_alone("I am a single citizen aged 36 buying with my mother.")
    assert not sv.single_buying_alone("We are a married couple aged 30 and 32.")


def test_a_price_against_the_budget_is_read():
    assert sv._PRICE_AGAINST_CEILING.search("Would a $500k flat fit my budget?")
    assert not sv._PRICE_AGAINST_CEILING.search("Find 4-room flats in Bedok under $500k.")


# ---------------------------------------------------------------------------
# HOUSE-09: a stated lease replaces the assumed one; buyers aged 55+ are told the loan model stops there
# ---------------------------------------------------------------------------
def test_buyers_aged_55_or_over_are_told_and_not_questioned():
    t1, t2 = conversation(SENIORS, "Use a flat with 55 years left.")
    assert t1["clarification"] is None
    assert t1["draft_response"].startswith("I can't estimate a loan or budget ceiling for your ages")
    assert sv.SENIOR_BUYER_NOTE in t1["draft_response"]
    lease = t2["user_constraints"]["financial_inputs"]["affordability"]["remaining_lease_years"]
    assert lease == 55
    assert "Affordability: not worked out — this estimate covers buyers below 55 only." in t2["draft_response"]


def test_a_later_lease_replaces_the_assumed_one():
    _, t2 = conversation("We are both Singapore citizens, first-timers, aged 32 and 30, earning $9,000 a month. "
                         "How much can we afford?", "Use a flat with 70 years left.")
    fin = t2["user_constraints"]["financial_inputs"]
    assert fin["grants"]["remaining_lease_years"] == fin["affordability"]["remaining_lease_years"] == 70


# ---------------------------------------------------------------------------
# DEMO-11 t2: a follow-up changing an instalment is code's figure alone
# ---------------------------------------------------------------------------
def test_an_instalment_follow_up_gets_no_model_opening():
    t1, t2 = conversation("Monthly repayment on a $500,000 loan at 2.6% for 25 years?",
                          "What if I repay over 20 years instead?")
    assert not t1["user_constraints"]["instalment_from_reply"]   # a question on its own keeps the model opening
    assert t2["user_constraints"]["instalment_from_reply"]
    assert t2["draft_response"].startswith("**Monthly instalment: $2,674**")


# ---------------------------------------------------------------------------
# SPATIAL-07: code reads "MRT only" and a price limit, whatever the model returns
# ---------------------------------------------------------------------------
def test_mrt_only_switches_the_rail_kind_even_when_the_model_carries_lrt():
    carried = {"towns": ["PUNGGOL"], "flat_types": ["4 ROOM"], "amenity_radii_meters": {"lrt": 250}}
    for fresh in ({"lrt": 250}, {"mrt": 250}, {}):
        out = sv.refine_search(dict(carried), {"amenity_radii_meters": fresh}, "Use MRT stations only instead.")
        assert out["amenity_radii_meters"] == {"mrt_only": 250}, fresh


def test_a_price_limit_is_not_an_affordability_question():
    c, errors = model_reading("Find blocks in Punggol within 250m of LRT, 4-room below 750k.",
                              wants_area_affordability=True)
    assert not c["wants_area_affordability"]
    assert any("affordability reading dropped" in e for e in errors)
    c, _ = model_reading("Can we afford a 4-room flat in Punggol?", wants_area_affordability=True)
    assert c["wants_area_affordability"]


# ---------------------------------------------------------------------------
# STATS-06: no launch the message does not name; no unchecked block comparisons
# ---------------------------------------------------------------------------
def test_a_launch_the_message_does_not_name_is_dropped():
    c, errors = model_reading("Show the price distribution for 4-room within 500m of Blk 510 Bedok North St 3.",
                              bto_project_name="BEDOK NORTH 2025")
    assert c["bto_project_name"] is None and any("BTO project dropped" in e for e in errors)
    c, _ = model_reading("Compare BTO project Imaginary Garden QZX with nearby resale.",
                         bto_project_name="Imaginary Garden QZX")
    assert c["bto_project_name"] == "Imaginary Garden QZX"


def test_block_comparisons_outside_a_block_search_are_dropped():
    state = {"messages": [{"role": "user", "content": "Use a 1km area instead."}], "user_constraints": {}}
    kept, dropped = sv.check_model_opening(
        "Blk 507 has a lower price and a shorter distance than Blk 515. Prices around the block are below.",
        state, "**Resale prices** (facts)\n- Blk 507 BEDOK NTH ST 3: median $560,000, 194 m\n"
               "- Blk 515 BEDOK NTH ST 3: median $590,000, 118 m")
    assert dropped["comparisons"] == 1
    assert kept == ["Prices around the block are below."]
    _, dropped = sv.check_model_opening(   # live, t1
        "Blk 507 and Blk 508 have lower prices compared to the area median, with Blk 515 offering similar prices.",
        state, "**Resale prices** (facts)\n- Blk 507: $560,000\n- Blk 508: $570,000\n- Blk 515: $590,000")
    assert dropped["comparisons"] == 1


def test_worked_continuously_12_months_in_a_list_of_facts_is_read():
    facts = sv.stated_financial_inputs("Grant and HDB loan eligibility confirmed, worked continuously 12 months, no PHG.")
    assert facts["grants"]["continuous_employment_12_months"] is True
    assert not sv.stated_financial_inputs("I worked in 2019 for 12 months.").get("grants", {}).get(
        "continuous_employment_12_months")
