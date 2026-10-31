"""
Code's own opening for variant C (Member 4). In the 31 Oct live runs Qwen3 put figures in every C opening and the
figure check dropped them, so many answers began straight at "**N candidate block(s)**" with no direct answer and no
trade-off. When nothing of the model's opening survives, or there is no model, code writes it from the same state as
the facts; it names only listed blocks and states no other figure. No model needed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

from agents import supervisor as sv  # noqa: E402
from agents import arch_3_specialist_5agent as C  # noqa: E402
from evaluation import architecture_metrics as M  # noqa: E402

ITEMS = {i["id"]: i for f in ("category_e_queries.json", "category_e_extra.json")
         for i in json.loads((ROOT / "evaluation" / f).read_text())}
_STATES: dict = {}


def state_for(qid: str) -> dict:
    """The no-model C answer to a Category E question (first turn), run once per test session."""
    if qid not in _STATES:
        _STATES[qid] = C.graph.invoke(sv.initial_state(ITEMS[qid]["query"], "C"))
    return _STATES[qid]


def block(n, price, lease, mrt=None, street="TAMPINES ST 21", **dist):
    distances = {**({"mrt_dist_m": mrt} if mrt is not None else {}), **{f"{k}_dist_m": v for k, v in dist.items()}}
    return {"block": n, "street": street, "town": "TAMPINES", "median_resale_price": price,
            "median_remaining_lease_years": lease, "distances": distances}


class Writer:
    """A stand-in model that always writes the same opening and keeps the prompt it was given."""
    def __init__(self, text):
        self.text, self.prompt = text, ""

    def invoke(self, msgs):
        self.prompt = msgs[-1].content
        return AIMessage(content=self.text)


# --- the trade-off sentence -------------------------------------------------------------------------------------------

def test_two_blocks_are_compared_and_three_are_ranked():
    two = [block("627", 655000, 65.5, 379.2), block("102", 710000, 65.7, 333.9)]
    assert sv.trade_off_sentence(two, {"amenity_radii_meters": {"mrt": 400}}) == \
        "Blk 627 is cheaper, while Blk 102 is nearer an MRT station."          # both leases show as 66 yrs: no role
    three = [block("228", 532500, 60.2, 394), block("288B", 598888, 72.7, 107.3), block("266", 602000, 71.1, 244.1)]
    assert sv.trade_off_sentence(three, {}) == \
        "Blk 228 is the cheapest, while Blk 288B has the most lease left and is nearest an MRT station."


def test_a_block_holding_every_role_is_named_once():
    blocks = [block("1", 500000, 80, 100), block("2", 600000, 70, 300), block("3", 700000, 60, 500)]
    assert sv.trade_off_sentence(blocks, {}) == \
        "Blk 1 is the cheapest and nearest an MRT station, and has the most lease left."


def test_a_tie_at_the_precision_the_facts_show_names_no_one():
    # the facts write $600,000 and 73 yrs for both: neither is "the cheapest" or has "the most lease left"
    tied = [block("761", 600000.4, 73.3, 188), block("763", 599999.8, 73.2, 297), block("762", 641500, 72.6, 241)]
    assert sv.trade_off_sentence(tied, {}) == "Blk 761 is nearest an MRT station."
    assert sv.trade_off_sentence([block("1", 5, 5, 5), block("2", 5, 5, 5)], {}) == ""
    assert sv.trade_off_sentence([block("1", 500000, 70, 100)], {}) == ""               # one block: nothing to trade


def test_a_missing_value_names_no_one_for_that_role():
    blocks = [block("1", None, 80, 100), block("2", 600000, 70, 300)]
    assert sv.trade_off_sentence(blocks, {}) == "Blk 1 has more lease left and is nearer an MRT station."


def test_the_amenity_the_user_limited_when_there_is_no_mrt_distance():
    blocks = [block("774", 538888, 69, school=237), block("639", 580000, 68, school=566)]
    assert sv.trade_off_sentence(blocks, {"amenity_radii_meters": {"school": 1000}}) == \
        "Blk 774 is cheaper and nearer the school, and has more lease left."


def test_shared_block_numbers_carry_the_street_and_a_longer_list_says_so():
    blocks = [block("101", 500000, 80, 300, street="TAMPINES ST 11"), block("101", 600000, 70, 100, street="BEDOK NTH AVE 1")]
    assert sv.trade_off_sentence(blocks, {}) == ("Blk 101 TAMPINES ST 11 is cheaper and has more lease left, "
                                                 "while Blk 101 BEDOK NTH AVE 1 is nearer an MRT station.")
    assert sv.trade_off_sentence(blocks[:1] + [block("9", 1, 1, 1)], {}, more_than_listed=True).startswith(
        "Of the listed blocks, ")


# --- the whole opening, by kind of answer -------------------------------------------------------------------------------

def test_match_no_match_and_tied_relaxations():
    s = state_for("E1-03")
    facts = sv.template_response(s)
    assert sv.code_opening(s, facts) == ("Both blocks below meet every limit you set, and your grant and budget estimate "
                                         "follows the list. Blk 627 is cheaper, while Blk 102 is nearer an MRT station.", [])
    s = state_for("E2-01")
    assert sv.code_opening(s, sv.template_response(s))[0] == (
        "No block meets every limit you set at once. Relaxing the minimum remaining lease unlocks the most blocks; "
        "each option is below.")
    s = state_for("E2-07")                    # MRT distance and lease each unlock the most: both are named
    assert "Relaxing either the distance to an MRT station or the minimum remaining lease" in \
        sv.code_opening(s, sv.template_response(s))[0]
    rel = {"suggestions": [{"constraint": "max_budget_sgd", "original": 6e5, "relaxed": 6.6e5, "unlocked_blocks": 0}],
           "best_single_change": None}
    s = {**s, "relaxation": rel}
    assert sv.code_opening(s, sv.template_response(s))[0] == (
        "No block meets every limit you set at once. Relaxing any one limit on its own unlocks no block either, as "
        "shown below.")


def test_one_block_a_missing_fact_and_a_search_that_could_not_run():
    s = {**state_for("E1-03")}
    s["candidate_blocks_intersection"] = s["candidate_blocks_intersection"][:1]
    assert sv.code_opening(s, sv.template_response(s))[0].startswith("Only Blk 627 meets every limit you set, and")
    s = state_for("E1-08")                    # the ages are missing: the estimate is asked for, not promised
    assert sv.code_opening(s, sv.template_response(s))[0].startswith(
        "All the blocks below meet every limit you set; your grant and budget estimate needs a few more facts, asked "
        "for at the end.")
    s = {**state_for("E1-03"), "tool_call_log": [{"tool": "filter_blocks_by_amenity_proximity", "error": "unavailable: x"}]}
    assert sv.code_opening(s, sv.template_response(s))[0].startswith("Both blocks below meet the limits that could be checked")


def test_financial_clarification_and_policy_leads():
    grants = {"status": "ok", "enhanced_housing_grant": 0, "family_grant": 80000, "proximity_housing_grant": 0,
              "total_grants": 80000}
    s = {"messages": [{"role": "user", "content": "What grants can we get?"}],
         "financial_breakdown": {"calculate_cpf_housing_grants": grants}}
    assert sv.code_opening(s, sv.template_response(s))[0] == \
        "Here are your housing grants, worked out from the facts you gave."
    budget = {"status": "ok", "max_purchase_price": 784366, "max_loan_amount": 569366, "monthly_instalment": 2583,
              "binding_constraint": "MSR"}
    s["financial_breakdown"] = {"calculate_cpf_housing_grants": grants, "calculate_maximum_affordability": budget}
    assert sv.code_opening(s, sv.template_response(s)) == \
        ("Here are your housing grants and maximum purchase price, worked out from the facts you gave.", [])
    needs = {"status": "needs_information", "missing_inputs": ["cash_savings"]}
    s["financial_breakdown"] = {"calculate_cpf_housing_grants": grants, "calculate_maximum_affordability": needs}
    assert sv.code_opening(s, sv.template_response(s))[0] == (
        "Here are your housing grants, worked out from the facts you gave. The rest of your grant and budget estimate "
        "needs a few more facts, asked for at the end.")
    s["financial_breakdown"] = {"calculate_maximum_affordability": needs}
    assert sv.code_opening(s, sv.template_response(s))[0] == \
        "Your grant and budget estimate needs a few more facts, asked for at the end."
    s["financial_breakdown"] = {"calculate_maximum_affordability": {"error": "boom"}}
    assert sv.code_opening(s, sv.template_response(s))[0] == \
        "Your estimate could not be worked out in full; what is missing is said below."
    passage = [{"text": "[Plus Flats > MOP]\nPlus flats have a 10-year MOP.", "source_url": "https://www.hdb.gov.sg/x"}]
    p = {"messages": [{"role": "user", "content": "What is the MOP for Plus flats?"}], "rag_policy_passages": passage}
    assert sv.code_opening(p, sv.template_response(p))[0] == "The HDB policy passages that bear on your question are below."
    empty = {**p, "rag_policy_passages": [{"text": "[Plus Flats > MOP]\n---", "source_url": ""}]}
    assert sv.code_opening(empty, sv.template_response(empty))[0] == \
        "The policy search found no HDB passage that answers this directly."
    with patch.object(sv, "unbuilt_tools_used", return_value=["retrieve_hdb_policy_knowledge"]):
        assert sv.code_opening(p, sv.template_response(p))[0] == ""      # mocked search: the facts already say so


def test_a_code_sentence_that_fails_the_figure_check_is_left_out_and_reported():
    # two Blk 101s: the trade-off names the streets, and "DR 6" reads as a figure, so that sentence is not used
    s = {"candidate_blocks_intersection": [block("101", 500000, 80, 300, street="PASIR RIS DR 6"),
                                           block("101", 600000, 70, 100, street="PASIR RIS ST 71")],
         "sql_market_results": {"query_resale_blocks_by_criteria": {"blocks": []}}}
    opening, errors = sv.code_opening(s, sv.template_response(s))
    assert opening == "Both blocks below meet every limit you set."
    assert len(errors) == 1 and errors[0].startswith("synthesize: code opening sentence left out")


def test_every_code_opening_on_the_evaluation_questions_passes_the_figure_check_and_changes_no_score():
    for qid, item in ITEMS.items():
        s = state_for(qid)
        facts = sv.template_response(s)
        opening, errors = sv.code_opening(s, facts)
        assert opening and errors == [], qid
        assert sv.keep_supported_sentences(opening, facts) == (opening, 0), qid    # block names only, no figure
        assert s["draft_response"].startswith(opening + "\n\n" + facts), qid    # the no-model C answer leads with it
        # the evaluation reads the same blocks, relaxation and amounts with or without the opening
        without = s["draft_response"][len(opening) + 2:]
        assert M.score_item(item, s) == M.score_item(item, {**s, "draft_response": without}), qid
        listed = {b["block"].upper() for b in s.get("candidate_blocks_intersection", [])[:sv.LISTED_BLOCKS]}
        assert set(M.recommended_block_numbers(opening)) <= listed, qid
        if item.get("sub_use_case") == "E2":
            assert M.recommended_block_numbers(opening) == [], qid               # no block presented as a match


# --- synthesis: the model's opening first, code's when none of it survives ------------------------------------------

def test_an_all_figures_model_opening_is_replaced_by_code_opening():
    s = state_for("E1-03")
    w = Writer("Blk 627 is at $655,000 with 66 years left. Blk 102 is 334m from the MRT. Both are under $750k.")
    with patch.object(sv, "get_llm", return_value=w):
        out = sv.synthesize_node({**s, "draft_response": ""})
    facts = sv.template_response(s)
    assert out["draft_response"] == sv.code_opening(s, facts)[0] + "\n\n" + facts
    assert out["errors"] == ["synthesize: dropped 3 opening sentence(s) that stated figures; code opening used"]
    # the prompt says plainly that numbers are deleted, shows a figure-free example and code's own reading
    assert "is deleted before the user sees it" in w.prompt and sv.OPENING_EXAMPLE in w.prompt
    assert "Blk 627 is cheaper, while Blk 102 is nearer an MRT station." in w.prompt


def test_a_surviving_model_sentence_is_kept_and_a_bare_lead_in_is_not():
    s = state_for("E1-03")
    facts = sv.template_response(s)
    w = Writer("Both Ang Mo Kio blocks fit; Blk 102 is the better pick for the train. Blk 627 costs $655,000.")
    with patch.object(sv, "get_llm", return_value=w):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert out["draft_response"] == "Both Ang Mo Kio blocks fit; Blk 102 is the better pick for the train.\n\n" + facts
    assert out["errors"] == ["synthesize: dropped 1 opening sentence(s) that stated figures"]
    for bare in ("Blk 627 costs $655,000. The options are:", "Blk 627 is $655k. **Trade-offs:**"):
        with patch.object(sv, "get_llm", return_value=Writer(bare)):
            out = sv.synthesize_node({**s, "draft_response": ""})
        assert out["draft_response"].startswith(sv.code_opening(s, facts)[0] + "\n\n"), bare
    assert not sv.says_something("The options are: **Trade-offs:**") and sv.says_something("Both fit. Options:")


def test_a_failed_model_and_no_model_both_get_code_opening_but_b_keeps_the_plain_template():
    s = state_for("E2-01")
    facts = sv.template_response(s)

    class Broken:
        def invoke(self, _msgs):
            raise TimeoutError

    with patch.object(sv, "get_llm", return_value=Broken()):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert out["draft_response"] == sv.code_opening(s, facts)[0] + "\n\n" + facts
    assert out["errors"] == ["synthesize: LLM failed, template used: TimeoutError"]
    assert sv.synthesize_node({**s, "draft_response": ""})["draft_response"].startswith("No block meets every limit")
    b = sv.synthesize_node({**s, "variant": "B", "draft_response": ""})
    assert b["draft_response"] == facts                 # B has no code-written facts, so no code opening either
