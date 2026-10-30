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
from evaluation.ground_category_e_extra import (HOUSEHOLDS, SPECS, financial_facts, ground, lrt_only_blocks,  # noqa: E402
                                                school_problems)
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


# The yes/no facts the calculators read, and the words that state each one (a fact the truth uses must be said).
YES_NO_WORDS = {
    ("first_timer_status", True): r"\bfirst\b",
    ("eligibility_confirmed", True): r"(^|\.)(?=[^.]*\bconfirmed\b)(?=[^.]*\beligibility\b)(?=[^.]*\bgrants?\b)",
    ("loan_eligibility_confirmed", True): r"(^|\.)(?=[^.]*\bconfirmed\b)(?=[^.]*\beligibility\b)(?=[^.]*\bHDB loan\b)",
    ("continuous_employment_12_months", True): r"\b(working|worked|employed)\b[^.]*\b(past|last) 12 months\b",
    ("proximity", "none"): r"\b(no|don't qualify for the) proximity grant\b",
    ("monthly_debt_obligations", 0.0): r"\bno (car loan or )?(other )?(debts|loans)\b",
    ("citizenship_pair", "SC_SC"): r"\bboth (are )?Singapore citizens\b|\bare Singapore citizens\b",
    ("citizenship_pair", "SC_Single"): r"\bsingle\b[^.]*\bSingapore citizen\b",
}


def wording_problems(spec) -> list:
    """Where the hand-written wording and the structured spec disagree: every amount, lease and yes/no fact the ground
    truth uses is said in the turn that supplies it, as words (a lease figure inside a price does not count)."""
    first, later = spec["query"], spec.get("answer", "")
    problems = []
    if spec["town"].title() not in first or spec["ft"].split()[0] + "-room" not in first:
        problems.append("town or flat type")
    if float(spec["budget"]) not in _amounts(first):
        problems.append("budget")
    if not re.search(rf"\b(at least {spec['lease']}|{spec['lease']}\+) years\b[^.]*\blease\b", first):
        problems.append("search lease")
    if not all(f"{metres}m" in first for metres in spec["radii"].values()):
        problems.append("distance")
    if spec.get("school") and (spec["school"] not in first or "1km" not in first):
        problems.append("school")
    problems += [f"upfront {k}" for k, v in (spec.get("upfront") or {}).items() if v and v not in _amounts(first)]
    if "hh" not in spec:
        return problems
    h, omit = HOUSEHOLDS[spec["hh"]], set(spec.get("omit", []))
    for fact, value in (("income", h["income"]), ("cpf", h["cpf"]), ("cash", h["cash"])):
        said_in, not_in = (later, first) if fact in omit else (first, later)
        if float(value) not in _amounts(said_in) or float(value) in _amounts(not_in):
            problems.append(fact)
    if not all(re.search(rf"\b{age}\b", first) for age in h["ages"]):
        problems.append("ages")
    if not re.search(rf"\b(assume|using|on) (an? )?{spec['calc_lease']}[- ]years? (of )?remaining lease\b", first, re.I):
        problems.append("lease for the sums")
    facts = financial_facts(spec)
    for (name, value), words in YES_NO_WORDS.items():
        if any(sec.get(name) == value for sec in facts.values()) and not re.search(words, first, re.I):
            problems.append(name)
    return problems


def test_the_words_state_every_fact_the_truth_uses():
    for spec in SPECS:
        assert wording_problems(spec) == [], spec["id"]


@pytest.mark.parametrize("sid, cut, flagged", [
    ("E1-X02", ", with at least 60 years left on the lease", "search lease"),     # 60 is still inside "$600k"
    ("E1-X04", " with 70+ years of lease left", "search lease"),                  # 70 is still inside "$700,000"
    ("E1-X04", " on a 70-year remaining lease", "lease for the sums"),
    ("E1-X01", " We don't qualify for the proximity grant.", "proximity"),
    ("E1-X01", ", and HDB has confirmed our eligibility for the grants and an HDB loan", "eligibility_confirmed"),
    ("E1-X03", "both continuously employed for the last 12 months, ", "continuous_employment_12_months"),
])
def test_the_wording_check_catches_a_dropped_fact(sid, cut, flagged):
    spec = dict({s["id"]: s for s in SPECS}[sid])
    assert cut in spec["query"]
    spec["query"] = spec["query"].replace(cut, "")
    assert flagged in wording_problems(spec)


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


def test_the_school_guard_rejects_a_name_the_tool_matches_to_another_school():
    # the tool strips "primary" and "school" and takes the first name containing the rest: Punggol Cove, not Punggol
    assert school_problems("Meridian Primary School") == []
    assert "another school" in school_problems("Punggol Primary School")[0]
    assert "not in the local spatial index" in school_problems("Nowhere Primary School")[0]


def test_the_lrt_guard_finds_blocks_near_an_lrt_stop_only():
    sengkang = {i["id"]: i for i in ITEMS}["E1-X01"]["ground_truth_facts"]["matching_block_keys"]
    assert sorted(lrt_only_blocks(sengkang, 400)) == sorted(sengkang)      # every match is near an LRT stop only
    serangoon = {i["id"]: i for i in ITEMS}["E1-X04"]["ground_truth_facts"]["matching_block_keys"]
    assert lrt_only_blocks(serangoon, 400) == []


@pytest.mark.skipif(not _market_db_ready(), reason="needs the local market database (python -m data.build_market_db)")
def test_a_question_that_says_only_mrt_is_rejected_when_its_truth_counts_lrt_stops():
    spec = dict({s["id"]: s for s in SPECS}["E1-X01"])
    spec["query"] = spec["query"].replace("an MRT or LRT station", "an MRT station")
    assert "LRT" not in spec["query"]
    assert any("near an LRT stop but no MRT station" in p for p in ground(spec)[1])


@pytest.mark.skipif(not _market_db_ready(), reason="needs the local market database (python -m data.build_market_db)")
def test_a_relaxed_price_query_cut_off_at_the_result_limit_is_rejected():
    # Hougang 4-room under $700,000 with 70 years: relaxing the lease to 65 matches more blocks than the query returns
    spec = dict({s["id"]: s for s in SPECS}["E2-X01"], budget=700000, lease=70)
    assert "relaxed min_remaining_lease_years price query truncated at the result limit" in ground(spec)[1]


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
