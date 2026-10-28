"""
Calculator facts come from the user's words, not the model's reading of them (Member 4). The 27 Oct live run showed
the model turning a married citizen couple into SC_Single, inventing ages and an interest rate, passing zero grants
and filling an empty financial_inputs that routed every search to the finance agent. No model needed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import supervisor as sv  # noqa: E402
from evaluation.ground_category_e import SPECS, financial_facts, question_text  # noqa: E402

ITEMS = {i["id"]: i for i in json.loads((ROOT / "evaluation" / "category_e_queries.json").read_text())}


def test_every_category_e_question_parses_to_its_ground_truth_facts():
    for spec in SPECS:
        want, got = financial_facts(spec), sv.stated_financial_inputs(question_text(spec))
        if want is None:
            assert got == {}, question_text(spec)        # search-only questions state no calculator facts
            continue
        for sec, facts in want.items():
            assert got[sec] == facts, (sec, question_text(spec))


def test_nothing_is_inferred():
    got = sv.stated_financial_inputs("We are a couple earning $9,000 a month with $80,000 in CPF OA.")
    assert "citizenship_pair" not in got["grants"]       # "couple" alone states no citizenship
    assert "youngest_buyer_age" not in got["grants"] and "proximity" not in got["grants"]
    aged = sv.stated_financial_inputs("We are a Singapore citizen couple aged 30 and 28 earning $9,000 a month.")
    assert "youngest_core_member_age" not in aged["grants"]  # children not ruled out
    assert aged["grants"]["citizenship_pair"] == "SC_SC"


def test_follow_up_answers_complete_the_facts():
    for qid, field, value in (("E1-07", "cash_savings", 25000.0), ("E1-08", "youngest_buyer_age", 30)):
        before = sv.stated_financial_inputs(ITEMS[qid]["query"])["affordability"]
        after = sv.stated_financial_inputs(ITEMS[qid]["query"] + " " + ITEMS[qid]["follow_up"]["answer"])["affordability"]
        assert field not in before and after[field] == value


def test_model_values_for_facts_are_dropped():
    model = {"grants": {"citizenship_pair": "SC_Single", "proximity": None, "flat_category": "resale"},
             "affordability": {"interest_rate": 2.6, "tenure_years": 25, "total_grants": 0.0, "youngest_buyer_age": 35},
             "upfront": {"purchase_price": 650000.0, "legal_and_valuation_fees": 10000.0}}
    stated = sv.stated_financial_inputs(ITEMS["E1-01"]["query"])
    out = sv.trusted_financial_inputs(model, stated)
    assert out["grants"]["citizenship_pair"] == "SC_SC" and out["grants"]["proximity"] == "none"
    assert not {"interest_rate", "tenure_years", "total_grants"} & set(out["affordability"])
    assert out["affordability"]["youngest_buyer_age"] == 35   # stated ("aged 36 and 35"), so kept from the text
    assert "upfront" not in out                              # the user named no purchase to cost
    assert sv.trusted_financial_inputs({"grants": None, "affordability": None, "upfront": None}, {}) is None


def test_empty_financial_inputs_do_not_route_to_finance():
    c = {"towns": ["TAMPINES"], "max_budget_sgd": 700000.0, "financial_inputs": {"grants": None, "affordability": None}}
    assert "financial" not in sv.decide_routing(c)
    c["financial_inputs"] = {"grants": {"monthly_income": 9000.0}}
    assert "financial" in sv.decide_routing(c)


def test_guard_overrides_model_arguments_and_passes_grants_forward():
    c = {"financial_inputs": sv.stated_financial_inputs(ITEMS["E1-01"]["query"])}
    calls = []

    def grants(**kw):
        calls.append(kw)
        return {"status": "ok", "total_grants": 90000.0}

    outputs = {}
    args = sv.guard_financial_args("calculate_maximum_affordability",
                                   {"monthly_income": 8500, "interest_rate": 2.6, "total_grants": 0.0, "cash_savings": 1.0},
                                   c, outputs, {"calculate_cpf_housing_grants": grants})
    assert len(calls) == 1 and calls[0]["citizenship_pair"] == "SC_SC"   # grants run first, from stated facts
    assert args["total_grants"] == 90000.0 and "interest_rate" not in args and args["cash_savings"] == 30000.0
    assert sv.guard_financial_args("query_resale_blocks_by_criteria", {"towns": ["X"]}, c, {}, {}) == {"towns": ["X"]}


def test_follow_up_answer_addresses_the_original_request():
    state = {"messages": [{"role": "user", "content": "Find flats and tell us our grants."},
                          {"role": "assistant", "content": "Please tell me your ages."},
                          {"role": "user", "content": "We are aged 32 and 30."}],
             "pending_clarification": {"needed": ["youngest_buyer_age"]}}
    req = sv.user_request(state)
    assert req.startswith("Find flats and tell us our grants.") and "We are aged 32 and 30." in req
    assert sv.user_request({"messages": [{"role": "user", "content": "Find flats."}]}) == "Find flats."


def test_unrequested_specialist_results_are_not_evidence():
    c = {"towns": ["BUKIT MERAH"], "flat_types": ["4 ROOM"], "amenity_radii_meters": {"mrt": 400, "hawker": 500}}
    outputs = {"filter_blocks_by_amenity_proximity": [{"block": "14"}], "verify_primary_school_distance": [],
               "compute_dual_transit_commute": []}
    kept = sv.requested_outputs("spatial", outputs, c, "")
    assert list(kept) == ["filter_blocks_by_amenity_proximity"]          # school "null" and blank commute dropped
    market = sv.requested_outputs("market_sql", {"query_resale_blocks_by_criteria": {}, "query_comparable_resale_transactions": {}}, c, "")
    assert list(market) == ["query_resale_blocks_by_criteria"]           # no comparable sales asked for
    assert sv.requested_outputs("financial", {"x": 1}, c, "") == {"x": 1}


def test_writer_is_told_to_list_every_match():
    blocks = [{"block": str(n), "street": "TOH GUAN RD"} for n in range(8)]
    _, rule = sv.synthesis_evidence({"variant": "C", "candidate_blocks_intersection": blocks})
    assert "List all 8 blocks" in rule
    _, rule = sv.synthesis_evidence({"variant": "C", "candidate_blocks_intersection": []})
    assert "List all" not in rule
