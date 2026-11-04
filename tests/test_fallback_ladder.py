"""The checks route_node makes before it shows the menu (Member 4, 4 Nov). The menu is the answer only when nothing in
the message, the open question or the last answer gives a lead. Each test is a turn the 4 Nov replay sent to the menu
(data/local/qa/out/now_*.jsonl): a reply that answers none of the open question stays on it (F06, F46); a question
ranking the market snapshot just shown is answered from it (STATS-07 t2); a bare loan amount asks for the rest of the
instalment (CLARIFY-07 t2); a rule question in words no policy topic names goes to the policy search (P-34). Vague
messages still get the menu. No model needed."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402

MENU_END = "Reply with a number, or ask your own question."
SNAPSHOT_QOQ = "How is the resale market doing compared with last quarter?"


def conversation(*turns):
    """Each turn's final state, every turn carrying the last (as the app does)."""
    states, prev = [], None
    for text in turns:
        prev = C.graph.invoke(sv.initial_state(query=text, variant="C", map_state={}, previous=prev, fast_path=True))
        states.append(prev)
    return states


def is_menu(state):
    return MENU_END in state["draft_response"]


# ---------------------------------------------------------------------------
# 1. A reply that answers none of the open question stays on it
# ---------------------------------------------------------------------------
def test_skip_on_the_household_form_asks_only_the_key_facts():
    _, t2 = conversation("How much grant can we get and what's our budget?", "skip")   # F06 t2
    assert not is_menu(t2)
    assert t2["draft_response"].startswith("No problem, the full list can wait.")
    clar = t2["clarification"]
    assert clar["kind"] == "financial" and clar["reasked"]
    assert clar["needed"] == ["citizenship_pair", "first_timer_status", "monthly_income"]   # the app's form fields
    assert "each buyer's citizenship, whether you are first-timers and your HFE-assessed monthly household income" \
        in clar["question"]
    assert "the age of each buyer" not in clar["question"]   # the full list waits


def test_a_second_non_answer_leaves_the_form_and_offers_the_menu():
    *_, t3 = conversation("How much grant can we get and what's our budget?", "skip", "I don't know")   # F06 t3
    assert t3["draft_response"].startswith("Okay, let's leave the grant estimate for now.")
    assert is_menu(t3) and t3["clarification"]["kind"] == "menu"
    assert "To start your grant" not in t3["draft_response"]   # nothing asked twice


def test_yes_says_it_answers_no_question_and_the_facts_then_complete_it():
    _, t2, t3 = conversation("How much grant can we get?", "yes",   # F46 t2
                             "We are both Singapore citizens, first-timers, earning $9,000 a month.")
    assert t2["draft_response"].startswith("I couldn't tell which of my questions that answers.")
    assert t2["clarification"]["kind"] == "financial"
    grants = t3["user_constraints"]["financial_inputs"]["grants"]
    assert grants["citizenship_pair"] == "SC_SC" and grants["monthly_income"] == 9000.0
    assert "I couldn't tell" not in t3["draft_response"]   # the lead belongs to its own turn


def test_a_new_question_after_the_form_is_not_held_to_it():
    _, t2 = conversation("How much grant can we get?", "4-room flats in Tampines under $600k")
    assert "fallback_lead" not in t2["user_constraints"]
    assert t2["candidate_blocks_intersection"]


# ---------------------------------------------------------------------------
# 2. A question ranking the market snapshot just shown
# ---------------------------------------------------------------------------
def test_which_flat_types_changed_the_most_ranks_the_snapshot():
    t1, t2 = conversation(SNAPSHOT_QOQ, "Which flat types changed the most?")   # STATS-07 t2
    assert not is_menu(t2)
    rows = t1["sql_market_results"]["get_market_snapshot"]["rows"]
    biggest = max(rows, key=lambda r: abs(r["qoq_change_pct"]))
    name = biggest["flat_type"].replace(" ROOM", "-room").replace("EXECUTIVE", "executive")
    assert t2["draft_response"].startswith(f"Compared with the previous quarter, {name} prices ")
    assert "**Resale market" in t2["draft_response"]   # the figures follow


def test_a_short_follow_up_keeps_the_ranking_and_its_period():
    *_, t3 = conversation(SNAPSHOT_QOQ, "Which flat types changed the most?", "Which fell the least?")
    assert t3["user_constraints"]["snapshot_rank"] == {"by": "flat_type", "period": "qoq_change_pct",
                                                      "direction": "down", "least": True}
    assert t3["draft_response"].startswith("Compared with the previous quarter,")


def test_towns_are_ranked_when_the_snapshot_named_them():
    t1, t2 = conversation("Market snapshot for Tampines and Bedok 4-room", "Which town rose the most over the year?")
    rows = [r for r in t1["sql_market_results"]["get_market_snapshot"]["rows"] if r["town"] != "ALL TOWNS"]
    top = max(rows, key=lambda r: r["yoy_change_pct"])["town"].title()
    assert t2["draft_response"].startswith(f"Compared with a year earlier, {top} prices rose the most")
    assert not t2["user_constraints"].get("wants_town_ranking")


def test_no_ranking_without_a_snapshot_before_it():
    assert sv.snapshot_rank("Which flat types changed the most?", None) is None
    assert sv.snapshot_rank("Which flat types changed the most?", {"towns": ["BEDOK"]}) is None


# ---------------------------------------------------------------------------
# 3. A bare loan amount asks for the rest of the instalment, keeping what is given
# ---------------------------------------------------------------------------
def test_the_loan_is_400k_asks_for_rate_and_tenure_then_works_it_out():
    t1, t2 = conversation("The loan is $400k.", "2.6% over 25 years")   # CLARIFY-07 t2
    assert not is_menu(t1)
    assert "on a $400,000 loan, tell me the interest rate and the loan tenure" in t1["draft_response"]
    assert t2["user_constraints"]["instalment"]["loan"] == 400_000
    assert "Monthly instalment: $1,815" in t2["draft_response"]


def test_partial_replies_are_kept_until_the_instalment_is_complete():
    t1, t2, t3 = conversation("$350k loan", "3%", "over 20 years")
    assert "on a $350,000 loan at 3%, tell me the loan tenure" in t2["draft_response"]
    assert t3["user_constraints"]["instalment"]["years"] == 20
    assert t3["user_constraints"]["instalment"]["rate"] == 3.0


def test_an_instalment_question_keeps_the_loan_it_gives():
    t1, _, t3 = conversation("Monthly payment on a $480,000 loan?", "2.6%", "25 years")
    assert "on a $480,000 loan, tell me the interest rate" in t1["draft_response"]
    assert t3["user_constraints"]["instalment"]["loan"] == 480_000


def test_a_new_loan_after_an_instalment_reworks_it():
    *_, t3 = conversation("The loan is $400k.", "2.6% over 25 years", "The loan is $500k.")
    inst = t3["user_constraints"]["instalment"]
    assert (inst["loan"], inst["rate"], inst["years"]) == (500_000, 2.6, 25)


def test_a_loan_inside_a_sentence_is_not_a_bare_loan():
    assert sv._BARE_LOAN.match("The loan is $400k.")
    assert sv._BARE_LOAN.match("$400k loan")
    assert not sv._BARE_LOAN.match("Can we take a $400k loan for a 4-room in Tampines?")


# ---------------------------------------------------------------------------
# 4. A rule question in other words goes to the policy search
# ---------------------------------------------------------------------------
def test_declining_a_flat_when_invited_gets_the_penalty_rule():
    (t1,) = conversation("What happens if I decline to choose a flat when invited in a BTO exercise?")   # P-34
    assert not is_menu(t1)
    assert t1["user_constraints"]["policy_fallback"]
    assert "declines to select a flat when invited" in t1["draft_response"]


def test_a_rule_question_with_no_matching_passage_says_so():
    (t1,) = conversation("What happens if my spouse is not on the HDB lease?")
    assert t1["policy_no_match"]
    assert sv.NO_RULE_FOUND in t1["draft_response"] and "HFE letter" not in t1["draft_response"]


def test_a_loosely_related_passage_is_not_shown():
    (t1,) = conversation("Can my parents stay in my HDB flat if they are not on the application?")   # live, 5 Nov
    assert t1["policy_no_match"] and "Multi-Generation" not in t1["draft_response"]


def test_an_instalment_finished_from_a_reply_is_code_alone():
    *_, t3 = conversation("$350k loan", "3%", "over 20 years")
    assert t3["user_constraints"]["instalment_from_reply"]
    (single,) = conversation("Monthly repayment on a $500,000 loan at 2.6% for 25 years?")
    assert not single["user_constraints"]["instalment_from_reply"]   # an eval-style question keeps the model opening


def test_advice_and_vague_messages_still_get_the_menu():
    for text in ("Should I buy BTO or resale?", "what flat should I buy?", "Find a nice flat."):
        assert not sv.rule_question_in_other_words(text), text
        (state,) = conversation(text)
        assert is_menu(state), text


def test_the_passage_check_needs_the_questions_own_words():
    passage = {"section": "Strict Non-Selection Rules & Penalties",
               "text": "Any first-timer applicant who declines to select a flat when invited loses priority in "
                       "subsequent BTO balloting exercises."}
    assert sv.passage_shares_words(passage, "What happens if I decline to choose a flat when invited in a BTO exercise?")
    assert not sv.passage_shares_words(passage, "How many times can I reject a BTO flat before I am penalised?")
