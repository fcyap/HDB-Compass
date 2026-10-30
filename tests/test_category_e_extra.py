"""
The extra Category E questions (evaluation/category_e_extra.json, Member 4): the file loads in the shape the runner
and scorer read, covers what it was written to cover, states in its words every fact its ground truth uses, and its
ground truth is what the real tools give today. No model needed.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import architecture_metrics as M  # noqa: E402
from evaluation.ground_category_e_extra import HOUSEHOLDS, SPECS, ground  # noqa: E402
from evaluation.run_category_e_prelim import dataset_path  # noqa: E402

EXTRA = ROOT / "evaluation" / "category_e_extra.json"
ITEMS = json.loads(EXTRA.read_text())
ORIGINAL = json.loads((ROOT / "evaluation" / "category_e_queries.json").read_text())


def test_file_loads_in_the_runner_shape():
    assert dataset_path("extra") == EXTRA
    assert [i["id"] for i in ITEMS] == [s["id"] for s in SPECS] and len({i["id"] for i in ITEMS}) == len(ITEMS)
    for item in ITEMS:
        assert item["category"] == "E" and item["sub_use_case"] in ("E1", "E2") and item["query"]
        truth = item["ground_truth_facts"]
        assert isinstance(truth["matching_block_keys"], list)
        assert (len(truth["matching_block_keys"]) > 0) == (item["sub_use_case"] == "E1")
        assert {c["tool"] for c in item["expected_mcp_calls"]} >= {"query_resale_blocks_by_criteria"}
        if item["sub_use_case"] == "E2":
            assert truth["best_single_change"]["unlocked_blocks"] > 0
        if item.get("follow_up"):
            assert item["follow_up"]["answer"] and item["follow_up"]["ground_truth_facts"]["matching_block_keys"]


def test_it_covers_what_the_original_set_does_not():
    old_towns = {t for i in ORIGINAL for t in i["stated_constraints"]["towns"]}
    new_towns = {i["stated_constraints"]["towns"][0] for i in ITEMS} - old_towns
    assert len(new_towns) >= 2
    tools = {c["tool"] for i in ITEMS for c in i["expected_mcp_calls"]}
    assert {"verify_primary_school_distance", "calculate_bsd_and_upfront_outlay"} <= tools
    omitted = {f for i in ITEMS for f in i.get("omitted_financial_facts") or []}
    assert omitted - {"cash", "ages"}                  # a follow-up that supplies something else (income, CPF)
    assert any(i["sub_use_case"] == "E2" and i["stated_constraints"]["towns"][0] in new_towns for i in ITEMS)


def _amounts(text: str) -> set:
    return {float(a.replace(",", "")) * (1000 if k else 1) for a, k in re.findall(r"\$(\d[\d,]*)(k)?", text)}


def test_the_words_state_every_fact_the_truth_uses():
    """The hand-written wording and the structured spec agree: each amount is said in the turn that supplies it."""
    for spec in SPECS:
        first, later = spec["query"], spec.get("answer", "")
        assert spec["town"].title() in first and spec["ft"].split()[0] + "-room" in first
        assert float(spec["budget"]) in _amounts(first) and f"{spec['lease']}" in first
        assert all(f"{metres}m" in first for metres in spec["radii"].values())
        if spec.get("school"):
            assert spec["school"] in first and "1km" in first
        for k, v in (spec.get("upfront") or {}).items():
            assert v == 0 or v in _amounts(first), k
        if "hh" not in spec:
            continue
        h, omit = HOUSEHOLDS[spec["hh"]], set(spec.get("omit", []))
        for fact, value in (("income", h["income"]), ("cpf", h["cpf"]), ("cash", h["cash"])):
            said_in = later if fact in omit else first
            assert float(value) in _amounts(said_in) and float(value) not in _amounts(first if fact in omit else later)
        assert all(str(age) in first for age in h["ages"]) and f"{spec['calc_lease']}" in first


def _market_db_ready() -> bool:
    from agents.llm import load_config
    if load_config()["tools"]["use_mock"].get("query_resale_blocks_by_criteria") != "real":
        return False
    from dotenv import dotenv_values
    path = Path(os.getenv("HDB_COMPASS_DB_PATH") or dotenv_values(ROOT / ".env").get("HDB_COMPASS_DB_PATH")
                or "data/local/hdb_compass.duckdb")
    return (path if path.is_absolute() else ROOT / path).exists()


@pytest.mark.skipif(not _market_db_ready(), reason="needs the local market database (python -m data.build_market_db)")
@pytest.mark.parametrize("spec", SPECS, ids=[s["id"] for s in SPECS])
def test_ground_truth_is_reproducible_from_the_tools(spec):
    item, problems = ground(spec)
    assert problems == []
    stored = {i["id"]: i for i in ITEMS}[spec["id"]]
    assert json.loads(json.dumps(item)) == stored     # rebuild with python -m evaluation.ground_category_e_extra


def test_stamp_duty_and_upfront_cash_are_scored_when_the_question_states_a_price():
    item = {i["id"]: i for i in ITEMS}["E1-X03"]
    truth = item["ground_truth_facts"]
    fin = truth["financial"]
    blocks = "\n".join(f"- Blk {k.split()[0]}" for k in truth["matching_block_keys"])
    money = (f"Grants ${fin['calculate_cpf_housing_grants']['total_grants']:,.0f}; max purchase price "
             f"${fin['calculate_maximum_affordability']['max_purchase_price']:,.2f}.")
    up = fin["calculate_bsd_and_upfront_outlay"]
    full = f"{blocks}\n{money} BSD ${up['buyer_stamp_duty']:,.0f}, cash required ${up['cash_required']:,.0f}."
    no_cash = f"{blocks}\n{money} BSD ${up['buyer_stamp_duty']:,.0f}."
    assert M.ground_truth_scores(item, {"draft_response": full})["success"]
    s = M.ground_truth_scores(item, {"draft_response": no_cash})
    assert s["grants_correct"] and s["max_price_correct"] and not s["upfront_correct"] and not s["success"]
    # questions without a stated price are scored as before
    plain = {i["id"]: i for i in ORIGINAL}["E1-01"]
    assert "upfront_correct" not in M.ground_truth_scores(plain, {"draft_response": "Blk 1"})
