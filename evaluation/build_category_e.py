"""
Category E golden queries (evaluation/build_category_e.py). Owned by Member 4
(Category E is Member 4's use-case category; Member 5 merges this into
golden_dataset_90.json and adds ground_truth_facts once real data lands).

15 queries: E1 full-stack synthesis + map sync (8), E2 zero-result relaxation (7).
expected_agents / expected_mcp_calls are derived from the supervisor's
deterministic plans, so they stay consistent with the routing rules under test.

    .venv/bin/python evaluation/build_category_e.py   -> evaluation/category_e_queries.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv

QUERIES = [
    # ---- E1: multi-objective synthesis + map ----
    ("E1-01", "E1", "We earn $9,000/month, have $90k CPF, want a 4-room flat in Tampines under $680k, within 400m of an MRT and 500m of a hawker centre, with 65+ years lease. Map the matches."),
    ("E1-02", "E1", "Couple earning $7,500/month with $60k CPF. Show 4-room flats in Bishan under $760k within 400m of an MRT with at least 60 years lease, and map them."),
    ("E1-03", "E1", "We are first-timer citizens earning $8,200/month. Find 5-room flats in Queenstown or Bukit Merah under $850k within 300m of an MRT with 60+ years lease and map the results."),
    ("E1-04", "E1", "Budget $720k, income $9,500/month, $110k CPF. 4-room in Bishan within 500m of an MRT and 600m of a hawker centre, 60+ years lease. Map the candidates."),
    ("E1-05", "E1", "Single citizen aged 36 earning $6,000/month. 4-room resale in Tampines under $620k within 400m of an MRT, 65+ years lease. Show them on the map."),
    ("E1-06", "E1", "We earn $10,000/month and have $150k CPF. Which 4-room flats in Bishan under $750k are within 400m of an MRT and have 60+ years lease? Map them and tell us our grants."),
    ("E1-07", "E1", "Income $8,800/month, CPF $95k. 4-room in Tampines under $650k within 300m of an MRT and 500m of a hawker centre with 65+ years lease. Map the matches."),
    ("E1-08", "E1", "Citizen + PR couple, $9,200/month, $80k CPF, buying a 4-room resale within 4km of parents in Tampines under $700k, within 500m of an MRT, 65+ years lease. Map it and calculate our grants."),
    # ---- E2: empty intersection -> relaxation ----
    ("E2-01", "E2", "Find a 5-room flat in Tiong Bahru under $600k within 200m of an MRT with 70+ years lease."),
    ("E2-02", "E2", "4-room flat in Bishan under $500k within 200m of an MRT and 100m of a hawker centre, 80+ years lease."),
    ("E2-03", "E2", "5-room in Queenstown under $650k within 150m of an MRT, 75+ years lease."),
    ("E2-04", "E2", "4-room in Tampines under $450k within 250m of an MRT and 200m of a hawker centre, 70+ years lease."),
    ("E2-05", "E2", "4-room in Bishan under $600k within 250m of an MRT, 90+ years lease."),
    ("E2-06", "E2", "We earn $7,000/month. 5-room in Bukit Merah under $700k within 200m of an MRT and 200m of a supermarket, 70+ years lease."),
    ("E2-07", "E2", "4-room in Tampines under $400k within 300m of an MRT, 75+ years lease."),
]


def stated_params(tool: str, args: dict, c: dict, q: str) -> dict:
    """Keep only arguments whose value the user's words determine. Defaults the planner filled in
    (citizenship when unstated, loan type, interest rate, a purchase price chosen from the budget)
    are not scored, so a model that picks another reasonable default is not penalised."""
    keep = set()
    if tool == "query_resale_blocks_by_criteria":
        keep |= {"towns"}
        keep |= {"flat_types"} if c.get("flat_types") else set()
        keep |= {"max_price"} if c.get("max_budget_sgd") else set()
        keep |= {"min_lease_years"} if c.get("min_remaining_lease_years") else set()
    elif tool == "compute_town_price_timeseries":
        keep |= {"towns"} | ({"flat_type"} if c.get("flat_types") else set())
    elif tool == "filter_blocks_by_amenity_proximity":
        keep |= {"towns", "amenity_constraints"}
    elif tool == "verify_primary_school_distance":
        keep |= {"school_name", "bracket"}
    elif tool == "compute_dual_transit_commute":
        keep |= {"work_dest_a", "work_dest_b"} | ({"max_minutes"} if c.get("max_commute_minutes") else set())
    elif tool == "calculate_cpf_housing_grants":
        keep.add("monthly_income")
        keep |= {"flat_category"} if re.search(r"\bresale\b|\bbto\b", q, re.I) else set()
        keep |= {"citizenship_pair"} if c.get("citizenship_scheme") else set()
        keep |= {"first_timer_status"} if re.search(r"first[- ]timer", q, re.I) else set()
        keep |= {"near_parents_4km"} if re.search(r"parents", q, re.I) else set()
    elif tool == "calculate_maximum_affordability":
        keep.add("monthly_income")
        keep |= {"cpf_oa_balance"} if c.get("cpf_oa_balance") else set()
        keep |= {"cash_savings"} if c.get("cash_savings") else set()
    return {k: v for k, v in args.items() if k in keep}


def build():
    items = []
    for qid, sub, q in QUERIES:
        c = sv.rule_based_extract(q).model_dump()
        agents = sv.decide_routing(c)
        calls = []
        for a in agents:
            for tool, args in sv.deterministic_plan(a, c, q):
                calls.append({"agent": a, "tool": tool, "required_params": stated_params(tool, args, c, q)})
        items.append({
            "id": qid, "category": "E", "sub_use_case": sub, "query": q,
            "initial_map_state": None,
            "expected_agents": agents,
            "expected_mcp_calls": calls,
            "expected_behaviour": ("non-empty candidate_blocks_intersection AND map_ui_payload.markers populated"
                                   if sub == "E1" else
                                   "candidate_blocks_intersection empty AND relaxation.best_single_change present"),
            "ground_truth_facts": {"_todo": "Member 5 fills from real DuckDB + spatial index: exact block IDs, grant $ amounts"},
        })
    out = PROJECT_ROOT / "evaluation" / "category_e_queries.json"
    out.write_text(json.dumps(items, indent=2))
    return items


if __name__ == "__main__":
    items = build()
    print(f"wrote {len(items)} Category E queries")
    for it in items:
        print(f"  {it['id']}: agents={it['expected_agents']} calls={[c['tool'] for c in it['expected_mcp_calls']]}")
