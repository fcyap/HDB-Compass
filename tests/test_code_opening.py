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
        # code's own trade-off passes the comparison check the model's sentences face
        assert sv.comparisons_hold(opening, s.get("candidate_blocks_intersection", [])[:sv.LISTED_BLOCKS]), qid
        if sv.clarification_needed(s):                 # an estimate that still needs facts is never said to follow
            assert "budget estimate follow" not in opening and "needs a few more facts" in opening, qid


# --- what the lead says follows the list (review of the code opening, 31 Oct) ------------------------------------------

GRANTS = {"status": "ok", "enhanced_housing_grant": 0, "family_grant": 80000, "proximity_housing_grant": 0,
          "total_grants": 80000}
BUDGET = {"status": "ok", "max_purchase_price": 784366, "max_loan_amount": 569366, "monthly_instalment": 2583,
          "binding_constraint": "MSR"}
NEEDS = {"status": "needs_information", "missing_inputs": ["cash_savings"]}
MOP = [{"text": "[Plus Flats > MOP]\nPlus flats have a 10-year MOP.", "source_url": "https://www.hdb.gov.sg/x"}]


def search_with(**extra):
    """A two-block search answer with the given financial results or policy passages."""
    return {"messages": [{"role": "user", "content": "What is the MOP for Plus flats?"}],
            "candidate_blocks_intersection": [block("627", 655000, 65.5, 379.2), block("102", 710000, 65.7, 333.9)],
            "sql_market_results": {"query_resale_blocks_by_criteria": {"blocks": []}}, **extra}


def answer_of(s):
    return sv.code_opening_parts(s, sv.template_response(s))[0][0]


def test_the_lead_says_only_what_was_worked_out_follows_the_list():
    # E1-07 gives no cash savings: the grants are worked out, the budget is asked for at the end
    s = state_for("E1-07")
    assert sv.code_opening(s, sv.template_response(s))[0].startswith(
        "Both blocks below meet every limit you set, and your housing grants follow the list; the rest of your grant "
        "and budget estimate needs a few more facts, asked for at the end.")
    meets = "Both blocks below meet every limit you set"
    assert answer_of(search_with(financial_breakdown={"calculate_cpf_housing_grants": GRANTS,
                                                      "calculate_maximum_affordability": NEEDS})) == (
        meets + ", and your housing grants follow the list; the rest of your grant and budget estimate needs a few "
        "more facts, asked for at the end.")
    assert answer_of(search_with(financial_breakdown={"calculate_maximum_affordability": BUDGET})) == \
        meets + ", and your maximum purchase price follows the list."
    assert answer_of(search_with(financial_breakdown={"calculate_cpf_housing_grants": GRANTS,
                                                      "calculate_maximum_affordability": BUDGET})) == \
        meets + ", and your grant and budget estimate follows the list."
    assert answer_of(search_with(financial_breakdown={"calculate_cpf_housing_grants": GRANTS,
                                                      "calculate_maximum_affordability": {"error": "boom"}})) == \
        meets + ", and your housing grants follow the list; part of your estimate could not be worked out, as said below."
    assert answer_of(search_with(financial_breakdown={"calculate_maximum_affordability": {"error": "boom"}})) == \
        meets + "; your estimate could not be worked out, as said below."


def test_the_verb_agrees_with_what_follows_the_list():
    # "the relevant HDB rules follow", not "follows" (one item, but a plural noun)
    assert answer_of(search_with(rag_policy_passages=MOP)) == \
        "Both blocks below meet every limit you set, and the relevant HDB rules follow the list."
    assert answer_of(search_with(rag_policy_passages=MOP, financial_breakdown={"calculate_maximum_affordability": BUDGET})) \
        == "Both blocks below meet every limit you set, and your maximum purchase price and the relevant HDB rules " \
           "follow the list."


def test_trend_and_comparables_leads_promise_a_summary_only_when_there_is_one():
    q = {"messages": [{"role": "user", "content": "How have Bishan resale prices moved?"}]}
    trend = {"summary": [{"town": "BISHAN", "latest_quarter": "2026Q3", "latest_median_price": 700000,
                          "median_price_per_sqm": 7000, "yoy_appreciation_pct": 2.0}], "metadata": {}}
    none = {"summary": [], "metadata": {"towns_without_data": ["BISHAN"]}}
    assert answer_of({**q, "sql_market_results": {"compute_town_price_timeseries": trend}}) == \
        "The latest town price trends for your search are below."
    s = {**q, "sql_market_results": {"compute_town_price_timeseries": none}}
    assert answer_of(s) == "No recent town price trend could be found for your search; the reason is below."
    assert sv.synthesize_node({**s, "draft_response": ""})["draft_response"] == (
        answer_of(s) + "\n\n- BISHAN: no transactions for the requested flat type in the trend window.")
    found = {"status": "ok", "criteria": {"flat_type": "4 ROOM", "radius_meters": 500},
             "metadata": {"window_start_month": "2025-10", "window_end_month": "2026-09"},
             "summary": {"transaction_count": 12, "block_count": 4,
                         "price": {"median": 650000, "lower_quartile": 620000, "upper_quartile": 690000}}}
    assert answer_of({**q, "sql_market_results": {"query_comparable_resale_transactions": found}}) == \
        "Recent comparable resale sales for your search are summarised below."
    for empty in ({"status": "ok", "summary": None, "metadata": {}}, {"error": "database unavailable"},
                  {"status": "no_coverage", "summary": None, "metadata": {}}):
        s = {**q, "sql_market_results": {"query_comparable_resale_transactions": empty}}
        assert answer_of(s) == "No comparable resale sales could be summarised for your search; the reason is below.", empty


# --- the model's comparisons are checked against the listed blocks -----------------------------------------------------

def test_comparisons_are_checked_against_the_listed_blocks():
    pair = [block("201E", 600000, 70, 355), block("899", 610000, 71, 346)]        # live31 E1-05: 355 m against 346 m
    assert not sv.comparisons_hold("The slight price difference is offset by a longer lease for Blk 899, while Blk 201E "
                                   "is marginally closer to the MRT.", pair)
    assert sv.comparisons_hold("Blk 899 is a little nearer the MRT station, while Blk 201E is cheaper.", pair)
    assert sv.comparisons_hold("Blk 201E is cheaper than Blk 899.", pair)
    assert not sv.comparisons_hold("Blk 899 is cheaper than Blk 201E.", pair)
    e107 = [block("62", 600000, 72, 384), block("11", 640000, 70, 354)]           # live31_varied E1-07~casual
    assert not sv.comparisons_hold("Blk 62 is cheaper and closer to MRT, but has a slightly shorter lease.", e107)
    assert sv.comparisons_hold("The higher price of Blk 11 offers slightly less lease time but is closer to the MRT "
                               "station.", e107)
    three = [block("761", 600000, 73, 188), block("763", 590000, 73, 297), block("762", 641500, 72, 241)]
    assert sv.comparisons_hold("The price difference is minimal, but **Blk 761** is closest to the MRT station, while "
                               "**Blk 763** is the most affordable.", three)
    # what cannot be checked is not kept: no block named, "cheaper" among three with no "than", a block list, a
    # negation, a distance to no named place
    for sentence in ("Offers the lowest price but also the shortest lease.", "Blk 761 is more expensive.",
                     "Blk 761 and 762 are cheaper.", "Blk 763 is not cheaper.", "Blk 763 is closer."):
        assert not sv.comparisons_hold(sentence, three), sentence
    assert not sv.comparisons_hold("This change allows options slightly further from the MRT.", [])   # no-match answer
    assert sv.comparisons_hold("Both blocks are in Ang Mo Kio.", pair)                               # compares nothing
    # a place the user limited that has no words of its own is read by its name
    cc = [block("1", 500000, 70, community_club=100), block("2", 600000, 70, community_club=300)]
    assert sv.trade_off_sentence(cc, {"amenity_radii_meters": {"community_club": 500}}) == \
        "Blk 1 is cheaper and nearer the community club."
    assert sv.comparisons_hold("Blk 1 is cheaper and nearer the community club.", cc)


# --- synthesis: code's answer first in a block search; the model's checked sentences follow ----------------------------

def test_a_false_model_comparison_is_dropped_and_code_trade_off_is_used():
    s = state_for("E1-05")
    facts = sv.template_response(s)
    w = Writer("Blk 201E is marginally closer to the MRT. This purchase-only ceiling excludes taxes, fees, and cash over "
               "valuation.")
    with patch.object(sv, "get_llm", return_value=w):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert out["draft_response"] == sv.code_opening(s, facts)[0] + "\n\n" + facts
    assert "Blk 899 has more lease left and is nearer an MRT station" in out["draft_response"]
    assert out["errors"] == [
        "synthesize: dropped 1 opening sentence(s) that compared blocks in a way the facts do not bear out; code opening used",
        "synthesize: dropped 1 opening sentence(s) that leaned on the sentence before"]


def test_a_true_model_comparison_follows_code_answer_in_place_of_code_trade_off():
    s = state_for("E1-04")
    facts = sv.template_response(s)
    answer, trade, _ = sv.code_opening_parts(s, facts)
    text = "Blk 86 offers a longer lease but is farther from the MRT. Blk 14 is closer to the MRT but has a shorter lease."
    w = Writer(text)
    with patch.object(sv, "get_llm", return_value=w):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert trade and out["draft_response"] == " ".join(answer) + " " + text + "\n\n" + facts
    assert out["errors"] == []
    # the prompt gives code's first sentence, says not to restate it and that comparisons are checked
    assert " ".join(answer) in w.prompt and "Do not say again whether the blocks meet the limits" in w.prompt
    assert "checked against the facts" in w.prompt and trade in w.prompt


def test_notes_a_restated_answer_and_a_sentence_leaning_on_a_dropped_one_give_way_to_code_opening():
    s = state_for("E1-01")
    facts = sv.template_response(s)
    for text in ("Ensure you account for additional costs like taxes and fees.",                       # live32 E1-01
                 "Note: This is a conditional estimate. Final eligibility and amounts require an HDB Flat Eligibility "
                 "(HFE) letter."):                                                                     # live31 E1-02
        with patch.object(sv, "get_llm", return_value=Writer(text)):
            out = sv.synthesize_node({**s, "draft_response": ""})
        assert out["draft_response"].startswith(sv.code_opening(s, facts)[0] + "\n\n" + facts), text
        assert out["errors"][0].startswith("synthesize: dropped") and "note the facts carry" in out["errors"][0], text
    s = state_for("E2-03")
    facts = sv.template_response(s)
    with patch.object(sv, "get_llm", return_value=Writer("No block satisfies every constraint at once.")):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert out["draft_response"] == sv.code_opening(s, facts)[0] + "\n\n" + facts and out["errors"] == []
    w = Writer("Relaxing the MRT distance to 550m unlocks 3 blocks. This increases the budget requirement but keeps the "
               "lease unchanged. No other single limit unlocks as many.")
    with patch.object(sv, "get_llm", return_value=w):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert out["draft_response"] == sv.code_opening(s, facts)[0] + " No other single limit unlocks as many.\n\n" + facts
    assert out["errors"] == ["synthesize: dropped 1 opening sentence(s) that stated figures",
                             "synthesize: dropped 1 opening sentence(s) that leaned on the sentence before"]
    # live31_varied E2-03~shorthand: kept after a restated answer, "this" would point at code's sentence instead
    with patch.object(sv, "get_llm", return_value=Writer("No block satisfies every constraint at once. This increases the "
                                                         "budget requirement but keeps the lease unchanged.")):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert out["draft_response"] == sv.code_opening(s, facts)[0] + "\n\n" + facts
    assert out["errors"] == ["synthesize: dropped 1 opening sentence(s) that leaned on the sentence before; code opening used"]


def test_outside_a_block_search_the_model_sentences_still_lead_and_policy_notes_stay():
    grants_only = {"messages": [{"role": "user", "content": "What grants can we get?"}],
                   "financial_breakdown": {"calculate_cpf_housing_grants": GRANTS}}
    facts = sv.template_response(grants_only)
    with patch.object(sv, "get_llm", return_value=Writer("You qualify for the Family Grant.")):
        out = sv.synthesize_node({**grants_only, "draft_response": ""})
    assert out["draft_response"].startswith("You qualify for the Family Grant.\n\n" + facts)
    with patch.object(sv, "get_llm", return_value=Writer("These figures are indicative; the HFE letter confirms them.")):
        out = sv.synthesize_node({**grants_only, "draft_response": ""})
    assert out["draft_response"].startswith("Here are your housing grants, worked out from the facts you gave.\n\n")
    p = {"messages": [{"role": "user", "content": "What is the MOP for Plus flats?"}], "rag_policy_passages": MOP}
    with patch.object(sv, "get_llm", return_value=Writer("You can check this with your HFE letter.")):
        out = sv.synthesize_node({**p, "draft_response": ""})
    assert out["draft_response"].startswith("You can check this with your HFE letter.\n\n")    # policy: the HFE may be asked


def test_a_failure_after_the_model_replied_lets_none_of_its_text_through():
    s = state_for("E1-03")
    facts = sv.template_response(s)
    with patch.object(sv, "get_llm", return_value=Writer("Blk 627 is at $1.")), \
            patch.object(sv, "check_model_opening", side_effect=ValueError):
        out = sv.synthesize_node({**s, "draft_response": ""})
    assert out["draft_response"] == sv.code_opening(s, facts)[0] + "\n\n" + facts and "$1." not in out["draft_response"]
    assert out["errors"] == ["synthesize: LLM failed, template used: ValueError"]


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
    # code's answer and trade-off first (the model compared nothing), then the model's sentence; "Both ... fit" only
    # restated code's answer
    assert out["draft_response"] == (sv.code_opening(s, facts)[0] + " Blk 102 is the better pick for the train.\n\n"
                                     + facts)
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
