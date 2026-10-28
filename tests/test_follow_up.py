"""
Follow-up flow for missing financial facts (Member 4). The Supervisor asks for exactly what the calculators report
missing, remembers the earlier request, reads the reply, and completes the estimate. No model needed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import supervisor as sv  # noqa: E402
from agents import arch_3_specialist_5agent as C  # noqa: E402
from evaluation import architecture_metrics as M  # noqa: E402

ITEMS = {i["id"]: i for i in json.loads((ROOT / "evaluation" / "category_e_queries.json").read_text())}


def test_question_names_exactly_the_missing_facts():
    state = {"financial_breakdown": {
        "calculate_cpf_housing_grants": {"status": "ok", "total_grants": 90000.0},
        "calculate_maximum_affordability": {"status": "needs_information", "missing_inputs": ["cash_savings", "total_grants"]},
        "calculate_bsd_and_upfront_outlay": {"status": "needs_information", "missing_inputs": ["market_valuation"]}}}
    clar = sv.clarification_needed(state)
    assert clar["needed"] == ["cash_savings"]          # grants are known; upfront costs need a chosen flat
    assert "cash savings" in clar["question"] and clar["question"].startswith("To finish")
    assert sv.clarification_needed({"financial_breakdown": {}}) is None


def test_reply_is_read_into_the_right_calculator_fields():
    ans = sv.parse_clarification_answer("We are aged 32 and 30.", ["youngest_buyer_age", "average_applicant_age"],
                                        conversation="a married couple, both first-timers with no children")
    assert ans["grants"] == {"youngest_buyer_age": 30, "youngest_core_member_age": 30}
    assert ans["affordability"] == {"youngest_buyer_age": 30, "oldest_buyer_age": 32, "average_applicant_age": 31.0}
    cash = sv.parse_clarification_answer("We have $25,000 in cash savings.", ["cash_savings"])
    assert cash == {"affordability": {"cash_savings": 25000.0}}
    # without "no children", the youngest occupant is not assumed to be a buyer
    assert "youngest_core_member_age" not in sv.parse_clarification_answer("I am 38.", ["youngest_buyer_age"])["grants"]


def test_reply_keeps_the_earlier_request():
    prior = {"towns": ["TOA PAYOH"], "flat_types": ["4 ROOM"], "max_budget_sgd": 750000.0,
             "financial_inputs": {"affordability": {"monthly_income": 7800.0, "cpf_oa_balance": 80000.0}}}
    state = sv.initial_state("We have $25,000 in cash savings.", "C",
                             previous={"messages": [{"role": "user", "content": "..."}], "user_constraints": prior,
                                       "clarification": {"needed": ["cash_savings"], "question": "?"}})
    c = sv.extract_constraints_node(state)["user_constraints"]
    assert c["towns"] == ["TOA PAYOH"] and c["max_budget_sgd"] == 750000.0
    assert c["financial_inputs"]["affordability"] == {"monthly_income": 7800.0, "cpf_oa_balance": 80000.0, "cash_savings": 25000.0}
    assert len(state["messages"]) == 2


def _market_db_ready() -> bool:
    from agents.llm import load_config
    if load_config()["tools"]["use_mock"].get("query_resale_blocks_by_criteria") != "real":
        return False
    from dotenv import dotenv_values
    path = Path(os.getenv("HDB_COMPASS_DB_PATH") or dotenv_values(ROOT / ".env").get("HDB_COMPASS_DB_PATH")
                or "data/local/hdb_compass.duckdb")
    return (path if path.is_absolute() else ROOT / path).exists()


@pytest.mark.skipif(not _market_db_ready(), reason="needs the local market database (python -m data.build_market_db)")
@pytest.mark.parametrize("qid", ["E1-07", "E1-08"])
def test_two_turn_conversation_reaches_the_ground_truth(qid):
    item = ITEMS[qid]
    # Turn 1 as the model would extract it: the stated search and financial facts, one fact missing.
    first = C.graph.invoke({**sv.initial_state(item["query"], "C"),
                            "prior_constraints": {**item["stated_constraints"], "financial_inputs": item["financial_inputs"]}})
    clar = first["clarification"]
    assert clar and set(item["ground_truth_facts"]["missing_financial_inputs"]) <= set(clar["needed"])
    assert M.ground_truth_scores(item, first)["success"]          # asks, names the true blocks, invents no budget
    # Turn 2: the user answers; the earlier request and facts carry over.
    second = C.run(item["follow_up"]["answer"], previous=first)
    assert second["clarification"] is None
    fu = {"id": qid, "sub_use_case": "E1", "ground_truth_facts": item["follow_up"]["ground_truth_facts"]}
    r = M.ground_truth_scores(fu, second)
    assert r["blocks_exact"] and r["grants_correct"] and r["max_price_correct"], second["draft_response"]
