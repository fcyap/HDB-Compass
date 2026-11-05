"""Fixes for Member 4's turns that failed the independent rerun of 640ada2 (5 Nov). Code now checks four things the
model's reading had added:
- a citizenship or an affordability question the message never states (DEMO-01 t1: the demo's first search ended
  with a household questionnaire);
- places nobody in the conversation named (MONEY-08: a stamp-duty follow-up checked prices at a made-up 123 Bishan
  Road);
- the statistics grouping a follow-up carries (STATS-03 t2: "Limit it to Tampines." lost the storey grouping);
- the opening of a PR couple's grant question, now code's no-grant rule (POLICY-03 t2: the live model opened with the
  EIP rule).
A single buyer taken as the only core member is told so (HOUSE-05). The model-path checks run through
_finalize_constraints with a model-style reading. No model needed."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402


def conversation(*turns):
    states, prev = [], None
    for text in turns:
        prev = C.graph.invoke(sv.initial_state(query=text, variant="C", map_state={}, previous=prev, fast_path=True))
        states.append(prev)
    return states


def model_reading(query, previous=None, **fields):
    """_finalize_constraints on code's reading of the message with the model's fields laid over it, after the earlier
    turn `previous` (a final state), as the live model path does."""
    state = sv.initial_state(query=query, variant="C", map_state={}, previous=previous, fast_path=False)
    reading = {**sv.rule_based_extract(query).model_dump(), **fields}
    errors = []
    return sv._finalize_constraints(reading, query, state, errors, from_model=True), errors


# ---------------------------------------------------------------------------
# DEMO-01 t1: a search with a price limit is not a grant question
# ---------------------------------------------------------------------------
DEMO_01 = "4-room resale in Tampines below $650k, please."


def test_a_guessed_citizenship_does_not_start_the_grant_questions():
    c, errors = model_reading(DEMO_01, citizenship_scheme="SC_SC_FirstTimer")
    assert c["citizenship_scheme"] is None
    assert sv.decide_routing(c) == ["market_sql"]
    assert any("citizenship dropped" in e for e in errors)


def test_a_guessed_flat_affordability_question_is_dropped():
    c, _ = model_reading(DEMO_01, wants_flat_affordability=True)
    assert not c["wants_flat_affordability"] and sv.decide_routing(c) == ["market_sql"]


def test_a_stated_citizenship_or_affordability_question_stays():
    c, _ = model_reading("We are first-timer citizens. 4-room resale in Tampines below $650k?",
                         citizenship_scheme="SC_SC_FirstTimer")
    assert c["citizenship_scheme"] == "SC_SC_FirstTimer"
    c, _ = model_reading("Can we afford Blk 884 Tampines St 83?", wants_flat_affordability=True)
    assert c["wants_flat_affordability"]


# ---------------------------------------------------------------------------
# MONEY-08: a place nobody named is not kept, nor carried to the next turn
# ---------------------------------------------------------------------------
BSD_ONLY = "BSD only for a residential purchase of $360,000, valuation equals price."


def test_made_up_addresses_are_dropped_and_the_follow_up_is_the_duty_again():
    c, errors = model_reading(BSD_ONLY, addresses=["123 BISHAN ROAD", "456 BISHAN ROAD", "789 BISHAN ROAD"])
    assert c["addresses"] == []
    assert any("places not in the conversation were dropped" in e for e in errors)
    t1, t2 = conversation(BSD_ONLY, "Change price and valuation to $500,000.")
    assert "Buyer's stamp duty: **$9,600**" in t2["draft_response"]
    assert "mentioned earlier" not in t2["draft_response"]


def test_places_the_user_or_the_last_answer_named_stay():
    assert sv.place_in_text("Blk 884 Tampines St 83", "Resale price for Tampines blk884?")
    assert sv.place_in_text("Blk 123A Bishan St 12", "How about 123 A Bishan?")
    assert sv.place_in_text("Ang Mo Kio Hub", "Flats near AMK hub")
    assert sv.place_in_text("Novena MRT", "Prices near Novena MRT")
    assert not sv.place_in_text("123 BISHAN ROAD", BSD_ONLY)
    (t1,) = conversation("4-room resale in Tampines below $650k, please.")
    cheapest = t1["candidate_blocks_intersection"][0]
    name = f"Blk {cheapest['block']} {cheapest['street']}"
    c, errors = model_reading("Tell me more about the cheapest one.", previous=t1, addresses=[name])
    assert c["addresses"] == [name]   # a block from the last answer is in the conversation
    assert not any("places not in the conversation" in e for e in errors)


# ---------------------------------------------------------------------------
# STATS-03 t2: the model's statistics call keeps the grouping asked for
# ---------------------------------------------------------------------------
def test_the_model_call_keeps_the_storey_grouping():
    errors = []
    args = sv.statistics_args({"group_by": ["town"], "towns": ["TAMPINES"], "start_month": "2020-01"},
                              {"wants_group_stats": True, "stats_group_by": ["storey_band"], "stats_since_year": 2020,
                               "stats_group_held": True},
                              errors, "market_sql")
    assert args["group_by"] == ["storey_band"] and args["start_month"] == "2020-01"
    assert errors == ["market_sql: statistics grouped by storey_band, as asked"]
    (t1,) = conversation("How have 4-room prices changed by storey since 2020?")
    c, _ = model_reading("Limit it to Tampines.", previous=t1, wants_group_stats=True, stats_group_by=["town"])
    assert c["stats_group_by"] == ["storey_band"] and c["towns"] == ["TAMPINES"] and c["stats_group_held"]


def test_a_call_already_grouped_as_asked_is_unchanged():
    errors = []
    args = {"group_by": ["storey_band"], "towns": ["TAMPINES"]}
    assert sv.statistics_args(dict(args), {"wants_group_stats": True, "stats_group_by": ["storey_band"]},
                              errors, "market_sql") == args
    assert errors == []


# ---------------------------------------------------------------------------
# POLICY-03 t2: a PR couple's grant question opens with code's rule
# ---------------------------------------------------------------------------
def test_a_pr_couple_grant_follow_up_opens_with_the_no_grant_rule():
    _, t2 = conversation("Can a citizen and PR couple get resale housing grants?", "What if both buyers are PRs?")
    assert t2["draft_response"].startswith(sv.PR_COUPLE_OPENING)
    assert "Repeat Buyers Eligible" not in t2["draft_response"].split("\n")[0]
    assert sv.forced_opening(t2) == (sv.PR_COUPLE_OPENING, False)   # the model's sentences do not follow it


def test_a_pr_couple_budget_keeps_its_own_opening():
    (t1,) = conversation("We are both PRs, earning $8,000 a month with $100k CPF. How much can we afford?")
    assert not t1["draft_response"].startswith(sv.PR_COUPLE_OPENING)


# ---------------------------------------------------------------------------
# HOUSE-05: a single buyer taken as the only core member is told so
# ---------------------------------------------------------------------------
SINGLE_38 = ("Single citizen aged 38, first-timer, monthly income $4,500, usable CPF $75k, cash $40k, debt-free. Grants "
             "and HDB loan eligibility confirmed. Buying 3-room resale with 70 years left, no proximity grant. What can I "
             "afford?")


def test_a_single_buyer_is_told_they_are_taken_as_buying_alone():
    t1, t2 = conversation(SINGLE_38, "I have worked continuously for 12 months.")
    assert sv.SINGLE_ALONE_NOTE in t2["draft_response"]
    assert t2["draft_response"].count(sv.SINGLE_ALONE_NOTE) == 1
    assert "Max purchase price" in t2["draft_response"]


def test_a_couple_gets_no_single_buyer_note():
    (t1,) = conversation("We are a married couple, both Singapore citizens, first-timers, aged 32 and 30, no children, "
                         "earning $9,000 a month. How much can we afford?")
    assert sv.SINGLE_ALONE_NOTE not in t1["draft_response"]


# ---------------------------------------------------------------------------
# The independent review's cases (5 Nov): what the checks above must leave alone
# ---------------------------------------------------------------------------
def test_ages_in_a_household_reply_are_still_read():
    # the buyer-status words got a name of their own; buyer_ages keeps its pronoun words
    assert sv.buyer_ages("Me and my hubby are 33 and 35.") == [33, 35]


def test_plural_scs_is_a_stated_citizenship():
    c, _ = model_reading("We're both 30 and SCs, earning $9,000 a month. Which grants can we get?",
                         citizenship_scheme="SC_SC_FirstTimer")
    assert c["citizenship_scheme"] == "SC_SC_FirstTimer"


def test_codes_own_affordability_reading_stands():
    q = "Do we have enough CPF for Blk 884 Tampines St 83?"
    assert sv.rule_based_extract(q).wants_flat_affordability
    c, _ = model_reading(q, wants_flat_affordability=True)
    assert c["wants_flat_affordability"]


def test_a_place_given_by_its_initials_or_read_by_code_stays():
    assert sv.place_in_text("National University of Singapore", "4-room flats within 1km of NUS")
    c, _ = model_reading("Resale price for Blk 884 Tampines St 83?", addresses=["Block 884 Tampines Street 83 Singapore"])
    assert c["addresses"]


def test_a_follow_up_naming_its_own_grouping_is_not_held():
    (t1,) = conversation("How have 4-room prices changed by storey since 2020?")
    for text in ("Group it by town instead.", "Now break it down by town.", "What about by year?"):
        c, _ = model_reading(text, previous=t1, wants_group_stats=True, stats_group_by=["town"])
        assert not c["stats_group_held"], text
        assert sv.statistics_args({"group_by": ["town"]}, c, [], "market_sql")["group_by"] == ["town"], text


def test_a_first_question_keeps_the_models_grouping():
    c, _ = model_reading("Resale statistics near MRT stations in the north-east region",
                         wants_group_stats=True, stats_group_by=["town", "station"])
    assert not c["stats_group_held"]
    assert sv.statistics_args({"group_by": ["station"]}, c, [], "market_sql")["group_by"] == ["station"]


def test_a_later_rule_question_of_a_pr_couple_gets_its_own_answer():
    _, t2 = conversation("We are both PRs. Can we get any grants?", "What is the SPR quota rule for buying a resale flat?")
    assert sv.forced_opening(t2)[0] != sv.PR_COUPLE_OPENING
    *_, t3 = conversation("Can a citizen and PR couple get resale housing grants?", "What if both buyers are PRs?",
                          "What is the minimum occupation period for a resale flat?")
    assert sv.forced_opening(t3)[0] != sv.PR_COUPLE_OPENING


def test_a_pr_couple_question_on_grants_and_another_rule_keeps_the_models_sentences():
    (t1,) = conversation("We are both PRs and want to know about grants. Can we buy a BTO flat?")
    assert sv.forced_opening(t1) == (sv.PR_COUPLE_OPENING, True)


# ---------------------------------------------------------------------------
# Live check of these fixes (5 Nov): two older gaps it showed
# ---------------------------------------------------------------------------
PART_TABLE = ("**Resale statistics for 4-room flats in Singapore, 2020-01 to 2026-09**\n"
              "| Town | Sales | Median price |\n|---|---|---|\n| Central Area | 537 | $885,000 |\n"
              "| Tampines | 5,403 | $578,000 |\n"
              "- ...and 16 more group(s) not shown (ask for a different sort or a narrower filter).")


def test_a_ranking_over_a_table_cut_short_is_dropped():
    # "Group it by town instead.": Tampines was the bottom of 10 shown towns, not the lowest of 26
    state = {"messages": [{"role": "user", "content": "Group it by town instead."}], "user_constraints": {}}
    kept, dropped = sv.check_model_opening(
        "Central Area has the highest median price for 4-room flats, while Tampines offers the lowest. "
        "Of the towns shown, Tampines is the cheapest.", state, PART_TABLE)
    assert dropped["part_table"] == 1
    assert kept == ["Of the towns shown, Tampines is the cheapest."]   # a ranking of the shown rows only stays
    _, dropped = sv.check_model_opening("Central Area has the highest median price.", state,
                                        PART_TABLE.replace("more group(s) not shown", "rows"))
    assert dropped["part_table"] == 0   # the whole set shown: the usual checks only


def test_first_timer_citizens_in_that_order_are_a_citizen_couple():
    assert sv.citizenship_stated("We are first-timer Singapore citizens.") == "SC_SC"
    assert sv.citizenship_stated("We are first-timer citizens but my wife is a PR.") == "SC_PR"
    assert sv.citizenship_stated("We are first-timer Singapore citizens buying with my friend.") is None
    _, t2 = conversation("We are first-timer Singapore citizens. How much can we afford for a 4-room resale?",
                         "Me and my hubby are 33 and 35. Household income $9,000 a month, CPF OA $80,000, cash $30,000, "
                         "no debts.")
    grants = t2["user_constraints"]["financial_inputs"]["grants"]
    assert grants["citizenship_pair"] == "SC_SC" and grants["youngest_buyer_age"] == 33
    assert "citizenship_pair" not in (t2["clarification"] or {}).get("needed", [])


def test_the_cut_off_rule_leaves_shown_row_rankings_idioms_and_the_headline_alone():
    # the review's cases (5 Nov): limited to the table, ranking nothing, or the headline's own claim
    (t,) = conversation("4-room median price by town in the last year")
    facts = t["draft_response"]
    assert sv._GROUPS_NOT_SHOWN in facts
    top = sv._TABLE_HEADLINE.search(facts)
    assert top and top.group(2) == "highest"
    for sentence in ("Punggol is the cheapest town listed.", "In the table below, Punggol has the lowest median price.",
                     "For the most part, prices in these mature towns rose over the last year.",
                     "The most recent year shows prices rising in mature towns.",
                     "On top of the price, remember the remaining lease.",
                     f"{top.group(1)} has the highest median price for 4-room flats."):
        assert not sv.ranks_part_table(sentence, facts), sentence
    for sentence in ("Punggol offers the lowest prices.", "Bishan prices rose the most over the year.",
                     f"{top.group(1)} has the highest median price, while Punggol is the cheapest."):
        assert sv.ranks_part_table(sentence, facts), sentence


def test_a_listed_block_sentence_is_comparisons_holds_even_beside_a_cut_off_table():
    (blocks,) = conversation("4-room resale in Tampines below $650k, please.")
    facts = blocks["draft_response"] + "\n" + PART_TABLE
    kept, dropped = sv.check_model_opening("Blk 440 is the cheapest, while Blk 477 has the most lease left.", blocks, facts)
    assert dropped["part_table"] == 0
