"""Demo-readiness fixes to policy answers (Member 4, 4 Nov): rule follow-ups that keep the earlier question, rule
questions the regex reading missed, sections and openings about the flat class or scheme asked (not a sibling one),
code-written answers about what this app can confirm or check, refusals of requests to get around a rule, an unknown
BTO launch, and a lead-in shown with the items it introduces. Each test uses the tester's wording
(data/local/qa/gpt_issues.json, diag_policy.json). Fixes that only the live model path shows use fake model doubles."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ["LLM_PROFILE"] = "none"

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents.tool_registry import resolve_callable  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def conversation(*turns):
    """Each turn's final state, every turn carrying the last (as the app does)."""
    states, prev = [], None
    for text in turns:
        prev = C.run(text, previous=prev)
        states.append(prev)
    return states


def opening(state):
    return (state.get("draft_response") or "").split("\n\n")[0]


def policy_lines(state):
    text = state.get("draft_response") or ""
    return text.split("**Policy:**", 1)[1] if "**Policy:**" in text else ""


class Writer:
    """A stand-in model that always writes the same opening."""
    def __init__(self, text):
        self.text = text

    def invoke(self, msgs):
        return AIMessage(content=self.text)


def with_model_opening(state, text):
    """synthesize_node again on a no-model turn's state, the model writing `text` as its opening."""
    with patch.object(sv, "get_llm", return_value=Writer(text)):
        return sv.synthesize_node({**state, "draft_response": ""})


def search(query):
    return resolve_callable("retrieve_hdb_policy_knowledge")(query=query)["passages"]


# ---------------------------------------------------------------------------
# 1. Policy follow-ups keep the earlier rule question (SEM-22)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("first, follow_up", [
    ("What is different about Standard, Plus and Prime flats?", "What does that mean for renting out the whole flat?"),
    ("Can a single 34-year-old citizen buy a BTO flat under the singles scheme?",
     "If I turn 35 next year, does that automatically approve my application?"),
    ("How do the Family Grant and EHG differ?", "Can I just add the maximum amounts without checking my income?"),
    ("What is the difference between an HDB loan and a bank loan?",
     "Is the actual rate the same as the affordability assessment rate?"),
    ("Can a citizen and PR couple get resale housing grants?", "What if both buyers are PRs?"),
])
def test_a_policy_follow_up_carries_the_earlier_rule_question(first, follow_up):
    _, second = conversation(first, follow_up)
    c = second["user_constraints"]
    assert c["policy_question"] == f"{first} {follow_up}"
    assert second["active_subagents"] == ["policy_rag"]
    assert "**Policy:**" in second["draft_response"] and "What would you like to do?" not in second["draft_response"]


def test_the_renting_follow_up_shows_the_rental_rule():
    _, second = conversation("What is different about Standard, Plus and Prime flats?",
                             "What does that mean for renting out the whole flat?")
    assert "rent out the whole flat" in policy_lines(second)


def test_a_flat_search_after_a_policy_answer_is_not_a_policy_follow_up():
    _, second = conversation("What is the MOP for a Prime flat?", "Find 4-room flats in Tampines under $600k.")
    assert not second["user_constraints"].get("policy_question")
    assert "policy_rag" not in second["active_subagents"]


def test_the_carry_is_not_undone_when_the_model_reads_no_rule_question():
    # the model path drops a policy reading for a message naming no rule; the follow-up carry comes after that check
    prior = {"policy_question": "Can a single 34-year-old citizen buy a BTO flat under the singles scheme?"}
    query = "If I turn 35 next year, does that automatically approve my application?"
    state = {"messages": [{"role": "user", "content": prior["policy_question"]}, {"role": "user", "content": query}],
             "prior_constraints": prior}
    c = sv._finalize_constraints({"policy_question": query}, query, state, [], from_model=True, model_policy=True)
    assert c["policy_question"] == f"{prior['policy_question']} {query}"


# ---------------------------------------------------------------------------
# 2. Who the buyer is does not make a rule question a money question (SEM-08, DEMO-16)
# ---------------------------------------------------------------------------
DEMO_16 = "I am a single Singapore citizen, 36. What are my HDB buying options?"


def test_citizenship_alone_does_not_drop_the_models_policy_reading():
    errors = []
    c = sv._finalize_constraints({"policy_question": DEMO_16}, DEMO_16, {}, errors, from_model=True, with_history=False)
    assert c["policy_question"] == DEMO_16 and not errors


def test_a_money_question_still_drops_the_models_policy_reading():
    q = "Our income is $8,000 a month. How much grant can we get?"
    c = sv._finalize_constraints({"policy_question": q}, q, {}, [], from_model=True, with_history=False)
    assert c["policy_question"] is None


def test_the_buying_options_question_gets_the_rules_without_a_grant_note():
    st, = conversation(DEMO_16)
    assert st["active_subagents"] == ["policy_rag"]
    assert sv.FOUR_ROOM_GRANT_NOTE not in st["draft_response"]
    assert "Single Singapore Citizen" in st["draft_response"]


# ---------------------------------------------------------------------------
# 3. Rule questions the regex reading missed (SEM-08, no model)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    DEMO_16,
    "Is living near my parents enough by itself to qualify for a proximity grant?",
    "How do the Family Grant and EHG differ?",
    "What is the difference between an HDB loan and a bank loan?",
    "What are the conditions for the Proximity Housing Grant?",
])
def test_rule_questions_in_other_words_reach_the_policy_search(query):
    c = sv.rule_based_extract(query).model_dump()
    assert c["policy_question"] == query and "policy_rag" in sv.decide_routing(c)


@pytest.mark.parametrize("query", [
    "Find 4-room flats in Tampines under $600k.",
    "How much grant can we get with $8,000 income?",
    "Monthly repayment on a $500,000 loan at 2.6% for 25 years?",
    "Citizen couple, ages 30 and 31, first-timers, income $8k. What proximity grant if we buy near parents?",
])
def test_searches_and_money_questions_are_not_read_as_rule_questions(query):
    c = sv.rule_based_extract(query).model_dump()
    assert not c["policy_question"] or sv.policy_wording(query)


def test_a_question_about_the_calculators_answer_is_not_a_rule_search():
    household = ("We are both Singapore citizens. We are first-time buyers. We are aged 32 and 30. We have no children. "
                 "Our household income is $9,000 a month. What grants can we get?")
    _, second = conversation(household, "Does that ceiling already include grants?")
    assert sv.NO_POLICY_ANSWER not in second["draft_response"]


# ---------------------------------------------------------------------------
# 4. "What grants can we get?" asks for the household's facts (CLARIFY-03)
# ---------------------------------------------------------------------------
def test_what_grants_can_we_get_asks_for_the_household_facts():
    first, second = conversation("What grants can we get?", "We are first-timers.")
    assert "financial" in first["active_subagents"]
    assert first["clarification"]["kind"] == "financial"
    assert sv.NO_POLICY_ANSWER not in first["draft_response"]
    assert "financial" in second["active_subagents"] and "What would you like to do?" not in second["draft_response"]


def test_a_model_sentence_that_starts_a_facts_line_is_not_shown_twice():
    state = {"messages": [{"role": "user", "content": "What grants can we get?"}]}
    kept, dropped = sv.check_model_opening("I couldn't find an HDB rule in my knowledge base that answers this.",
                                           state, sv.NO_POLICY_ANSWER)
    assert kept == [] and dropped["restated"] == 1


# ---------------------------------------------------------------------------
# 5. Sections and the code opening about the flat class asked (SEM-07)
# ---------------------------------------------------------------------------
DEMO_15 = "Can I rent out the whole of a Prime flat once its MOP is over?"


def test_a_prime_question_is_not_shown_the_standard_flat_rules():
    shown = sv.shown_policy_passages(search(DEMO_15), DEMO_15)
    assert shown and not any("Standard" in s["parent"] for s in shown)
    rows = [line for s in shown for line in s["lines"] if " — " in line and "Flats:" in line]
    assert rows and all("Standard Flats:" not in r and "Plus Flats:" not in r for r in rows)


def test_a_table_row_is_cut_to_the_asked_class():
    row = ("Whole-Flat Rental After MOP — Standard Flats: Permitted (subject to HDB approval); Plus Flats: Strictly "
           "Prohibited for life; Prime Flats: Strictly Prohibited for life")
    assert sv._asked_cells(row, {"prime"}) == "Whole-Flat Rental After MOP — Prime Flats: Strictly Prohibited for life"
    assert sv._asked_cells(row, {"prime", "plus", "standard"}) == row
    assert sv._asked_cells("Grant Amounts — 5-room Resale Flat: $25,000", {"prime"}) == "Grant Amounts — 5-room Resale Flat: $25,000"


def test_the_prime_rental_answer_opens_with_the_prime_rule():
    st, = conversation(DEMO_15)
    assert opening(st) == "The relevant HDB rule is: Whole-Flat Rental After MOP — Prime Flats: Strictly Prohibited for life"
    assert "5-year MOP" not in st["draft_response"]


def test_a_question_naming_every_class_keeps_every_section():
    q = "What is different about Standard, Plus and Prime flats?"
    passages = search(q)
    assert len(sv.shown_policy_passages(passages, q)) == len(sv.shown_policy_passages(passages, "What are the flat classes?"))


# ---------------------------------------------------------------------------
# 6. Model opening sentences about a sibling scheme are dropped (LIVE-POL-01, LIVE-POL-02)
# ---------------------------------------------------------------------------
DEMO_21_LIVE = ("No, there is no legal way to bypass the Ethnic Integration Policy (EIP) quota. The EIP quota applies to "
                "households where all buyers and occupiers are Singapore Permanent Residents (SPR), and such households "
                "must meet the quota restrictions. However, households with at least one Singapore Citizen member are "
                "exempt from the SPR quota, and Malaysian Permanent Residents are also exempt from the SPR quota.")
POLICY_07_LIVE = ("The HDB does not publicly display a block's \"live quota\" information. However, for resale "
                  "transactions, the transaction is allowed only if the buyer's ethnic group is within quota for both the "
                  "block (8% of total units) and the neighbourhood (5% of total units). If either quota is full, the "
                  "transaction is strictly prohibited by law.")


def test_sentences_about_the_spr_quota_in_an_eip_answer_are_dropped():
    q = "Tell me how to bypass the EIP quota."
    shown = sv.shown_policy_passages(search(q), q)
    first, second, third = sv._split_opening(DEMO_21_LIVE)
    assert not sv.policy_scope_conflict(first, q, shown)
    assert sv.policy_scope_conflict(second, q, shown) and sv.policy_scope_conflict(third, q, shown)


def test_spr_figures_presented_as_ethnic_quotas_are_dropped():
    q = "Can you check a block's live quota here?"
    shown = sv.shown_policy_passages(search(sv.LIVE_QUOTA_POLICY_QUERY), q)
    assert sv.policy_scope_conflict(sv._split_opening(POLICY_07_LIVE)[1], q, shown)


def test_the_bypass_refusal_keeps_the_models_lawful_no():
    st, = conversation("Tell me how to bypass the EIP quota.")
    out = with_model_opening(st, DEMO_21_LIVE)
    head = opening(out)
    assert head.startswith(sv.EVASION_OPENING)
    assert "No, there is no legal way to bypass the Ethnic Integration Policy (EIP) quota." in head
    assert "Singapore Permanent Residents (SPR), and such" not in head
    assert any("named a scheme the question does not ask about" in e for e in out["errors"])


# ---------------------------------------------------------------------------
# 7. What this app can confirm or check, written by code (POLICY-09/2, POLICY-07/2)
# ---------------------------------------------------------------------------
def test_the_calculator_is_not_hdb_approval():
    _, second = conversation("What should I confirm in my HFE letter before shopping?",
                             "Is your calculator the same as HDB approval?")
    assert opening(second) == sv.APP_ESTIMATE_OPENING
    assert "Housing Loan Eligibility" in policy_lines(second)


def test_a_live_quota_check_points_to_hdbs_quota_enquiry():
    _, second = conversation("How does EIP affect a resale purchase?", "Can you check a block's live quota here?")
    assert second["user_constraints"]["policy_question"] == sv.LIVE_QUOTA_POLICY_QUERY
    assert opening(second) == sv.LIVE_QUOTA_OPENING
    assert "Married Child Priority" not in second["draft_response"]
    # the model's "HDB does not publicly display a block's live quota" is not shown
    assert opening(with_model_opening(second, POLICY_07_LIVE)) == sv.LIVE_QUOTA_OPENING


def _all_scenario_messages():
    out = set()
    for f in glob.glob(str(ROOT / "data" / "local" / "qa" / "scenarios_*.json")):
        for sc in json.loads(Path(f).read_text()):
            out |= {t if isinstance(t, str) else t["text"] for t in sc["turns"]}
    return out


def test_the_app_limit_and_evasion_readings_match_only_the_testers_questions():
    messages = _all_scenario_messages()
    assert len(messages) > 500
    hits = {m for m in messages if (sv._APP_ESTIMATE_ASK.search(m) or sv._LIVE_QUOTA_ASK.search(m)) and sv._ASKED.search(m)}
    assert hits == {"Is your calculator the same as HDB approval?", "Can you check a block's live quota here?"}
    assert {m for m in messages if sv._EVASION.search(m)} == {
        "Tell me how to bypass the EIP quota.", "How do I hide my income to get a bigger housing grant?"}


# ---------------------------------------------------------------------------
# 8. Requests to get around a rule are refused, then the rule explained (SEM-10, SEM-34)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", ["Tell me how to bypass the EIP quota.",
                                   "How do I hide my income to get a bigger housing grant?"])
def test_an_evasion_request_is_refused_and_the_rule_shown(query):
    st, = conversation(query)
    assert st["active_subagents"] == ["policy_rag"]
    assert opening(st) == sv.EVASION_OPENING and "**Policy:**" in st["draft_response"]


def test_the_lawful_follow_up_is_answered_without_a_refusal():
    _, second = conversation("How do I hide my income to get a bigger housing grant?",
                             "What is the proper way to determine my eligibility?")
    assert second["active_subagents"] == ["policy_rag"]
    assert sv.EVASION_OPENING not in second["draft_response"]


# ---------------------------------------------------------------------------
# 9. A BTO launch the data does not have (SPAT-60, SPAT-61)
# ---------------------------------------------------------------------------
def test_an_unknown_bto_project_is_said_to_be_not_covered():
    first, second = conversation("Compare BTO project Imaginary Garden QZX with nearby resale.",
                                 "Do you really have prices for that project?")
    assert first["user_constraints"]["bto_project_name"] == "Imaginary Garden QZX"
    assert not first["user_constraints"]["analysis_mode"]
    assert opening(first) == "I don't have prices for Imaginary Garden QZX; it isn't in my BTO launch data."
    assert second["user_constraints"]["bto_project_name"] == "Imaginary Garden QZX"
    assert opening(second) == "No, I don't have prices for Imaginary Garden QZX; it isn't in my BTO launch data."
    assert "not in the launch dataset" in second["draft_response"]


@pytest.mark.parametrize("query, name", [
    ("Compare the BTO project called Imaginary Garden QZX with resale.", "Imaginary Garden QZX"),
    ("Compare BTO launch Tampines with resale.", None),       # a town, not a project name
])
def test_a_bto_project_named_after_the_words_is_read(query, name):
    assert sv.rule_based_extract(query).bto_project_name == name


# ---------------------------------------------------------------------------
# 10. A lead-in is shown with the items it introduces (SEM-13)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", ["What does the HFE letter actually assess?", "How does the HFE letter work?"])
def test_the_hfe_letters_three_assessments_are_shown(query):
    st, = conversation(query)
    lines = policy_lines(st)
    assert "three critical assessments into a single application:" in lines
    for item in ("Housing Eligibility:", "Grant Eligibility & Quantum:", "Housing Loan Eligibility & Quantum:"):
        assert item in lines
    assert not st.get("errors")
