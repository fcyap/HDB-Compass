"""
Category E golden questions grounded on the real data (evaluation/ground_category_e.py). Owned by Member 4.

Replaces the fixture-era builder. Every question is generated from a structured spec, then checked with the
same tools the agents call:
  * Member 1's real resale query (local DuckDB snapshot) and financial calculators,
  * Member 2's real amenity-proximity filter,
  * the Supervisor's own join (agents.supervisor.compute_intersection), affordability cap and relaxation steps.
A spec is rejected unless it meets its definition:
  E1  at least one block meets every constraint after the affordability cap; price query not truncated;
      grant and affordability results have the status the question is designed for.
  E2  no block meets every constraint, the price query is not truncated, every amenity in the question has data
      in that town, and at least one single relaxation step unlocks a block (so a trade-off exists).

    .venv/bin/python -m evaluation.ground_category_e            # check and write evaluation/category_e_queries.json
    .venv/bin/python -m evaluation.ground_category_e --check    # check only, write nothing

The expected matches depend on the database snapshot and rule version recorded in each item; rebuild after
`python -m data.build_market_db --refresh`, a rules update or a spatial-index rebuild.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv  # noqa: E402
from agents.llm import load_config  # noqa: E402
from agents.schemas import block_key  # noqa: E402
from agents.tool_registry import ToolRunner  # noqa: E402

OUT = PROJECT_ROOT / "evaluation" / "category_e_queries.json"

# ---------------------------------------------------------------------------
# Households (financial facts stated explicitly, as Member 1's contract requires)
# ---------------------------------------------------------------------------
HOUSEHOLDS: Dict[str, Dict[str, Any]] = {
    "sc_sc_32_30": dict(text="We are a married Singapore citizen couple aged 32 and 30, both first-timers with no children",
                        pair="SC_SC", ages=(32, 30), income=9000, cpf=90000, cash=40000),
    "sc_sc_36_35": dict(text="We are a married Singapore citizen couple aged 36 and 35, both first-timers with no children",
                        pair="SC_SC", ages=(36, 35), income=8500, cpf=120000, cash=30000),
    "sc_pr_34_33": dict(text="We are a married couple aged 34 and 33, one Singapore citizen and one permanent resident, "
                             "both first-timers with no children",
                        pair="SC_PR", ages=(34, 33), income=7800, cpf=80000, cash=25000),
    "single_38": dict(text="I am a single Singapore citizen aged 38, a first-timer with no children",
                      pair="SC_Single", ages=(38, 38), income=5500, cpf=110000, cash=50000),
    "sc_sc_31_29": dict(text="We are a married Singapore citizen couple aged 31 and 29, both first-timers with no children",
                        pair="SC_SC", ages=(31, 29), income=11000, cpf=150000, cash=60000),
}

# ---------------------------------------------------------------------------
# Question specs. omit: financial facts deliberately left out (the assistant must ask for them).
# ---------------------------------------------------------------------------
SPECS: List[Dict[str, Any]] = [
    # ---- E1: matches exist; complete financial facts unless `omit` says otherwise
    dict(id="E1-01", sub="E1", town="JURONG EAST", ft="4 ROOM", budget=650000, lease=60, radii={"mrt": 400}, hh="sc_sc_36_35"),
    dict(id="E1-02", sub="E1", town="JURONG EAST", ft="5 ROOM", budget=850000, lease=66, radii={"mrt": 400}, hh="sc_sc_31_29"),
    dict(id="E1-03", sub="E1", town="ANG MO KIO", ft="4 ROOM", budget=750000, lease=65, radii={"mrt": 400, "hawker": 500}, hh="sc_sc_32_30"),
    dict(id="E1-04", sub="E1", town="BUKIT MERAH", ft="4 ROOM", budget=750000, lease=65, radii={"mrt": 400, "hawker": 500}, hh="sc_sc_32_30"),
    dict(id="E1-05", sub="E1", town="TAMPINES", ft="3 ROOM", budget=500000, lease=60, radii={"mrt": 400}, hh="single_38"),
    dict(id="E1-06", sub="E1", town="BEDOK", ft="5 ROOM", budget=850000, lease=66, radii={"mrt": 400}, hh="sc_sc_31_29"),
    dict(id="E1-07", sub="E1", town="TOA PAYOH", ft="4 ROOM", budget=750000, lease=65, radii={"mrt": 400}, hh="sc_pr_34_33",
         omit=["cash"]),
    dict(id="E1-08", sub="E1", town="JURONG EAST", ft="4 ROOM", budget=750000, lease=65, radii={"mrt": 400}, hh="sc_sc_32_30",
         omit=["ages"]),
    # ---- E2: no block meets everything; a single relaxation step unlocks one (search only, no finance)
    dict(id="E2-01", sub="E2", town="TAMPINES", ft="4 ROOM", budget=700000, lease=70, radii={"mrt": 400}),
    dict(id="E2-02", sub="E2", town="JURONG EAST", ft="4 ROOM", budget=750000, lease=65, radii={"mrt": 400, "hawker": 500}),
    dict(id="E2-03", sub="E2", town="BUKIT MERAH", ft="5 ROOM", budget=850000, lease=65, radii={"mrt": 300, "hawker": 300}),
    dict(id="E2-04", sub="E2", town="BEDOK", ft="5 ROOM", budget=850000, lease=65, radii={"mrt": 400, "hawker": 500}),
    dict(id="E2-05", sub="E2", town="ANG MO KIO", ft="4 ROOM", budget=650000, lease=60, radii={"mrt": 400}),
    dict(id="E2-06", sub="E2", town="TOA PAYOH", ft="5 ROOM", budget=850000, lease=65, radii={"mrt": 400, "hawker": 500}),
    dict(id="E2-07", sub="E2", town="JURONG EAST", ft="4 ROOM", budget=650000, lease=60, radii={"mrt": 300, "hawker": 300}),
]

AMENITY_WORDS = {"mrt": "an MRT station", "hawker": "a hawker centre", "supermarket": "a supermarket"}


def _title(town: str) -> str:
    return " ".join(w.capitalize() for w in town.split())


def _radii_text(radii: Dict[str, int]) -> str:
    parts = [f"{m}m of {AMENITY_WORDS[k]}" for k, m in radii.items()]
    return "within " + " and ".join(parts)


def financial_facts(spec: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """Structured facts the user states, in Member 1's financial_inputs shape (None for search-only questions)."""
    if "hh" not in spec:
        return None
    h, omit = HOUSEHOLDS[spec["hh"]], set(spec.get("omit", []))
    old, young = h["ages"]
    grants = dict(flat_category="resale", citizenship_pair=h["pair"], first_timer_status=True, monthly_income=float(h["income"]),
                  flat_type=spec["ft"], remaining_lease_years=float(spec["lease"]), eligibility_confirmed=True,
                  continuous_employment_12_months=True, proximity="none")
    aff = dict(monthly_income=float(h["income"]), cpf_oa_balance=float(h["cpf"]), loan_type="hdb",
               remaining_lease_years=float(spec["lease"]), monthly_debt_obligations=0.0, loan_eligibility_confirmed=True)
    if "cash" not in omit:
        aff["cash_savings"] = float(h["cash"])
    if "ages" not in omit:
        grants.update(youngest_buyer_age=young, youngest_core_member_age=young)
        aff.update(youngest_buyer_age=young, oldest_buyer_age=old, average_applicant_age=(old + young) / 2)
    return {"grants": grants, "affordability": aff}


def question_text(spec: Dict[str, Any]) -> str:
    town, ft = _title(spec["town"]), spec["ft"].split()[0] + "-room"
    search = (f"Find {ft} resale flats in {town} under ${spec['budget']:,} with at least {spec['lease']} years of lease left, "
              f"{_radii_text(spec['radii'])}.")
    if "hh" not in spec:
        return search
    h, omit = HOUSEHOLDS[spec["hh"]], set(spec.get("omit", []))
    who = h["text"]
    if "ages" in omit:  # drop the ages, keep the rest of the description
        who = who.replace(f" aged {h['ages'][0]} and {h['ages'][1]}", "").replace(f" aged {h['ages'][0]}", "")
    money = (f"Our HFE-assessed household income is ${h['income']:,} a month and we have ${h['cpf']:,} in CPF OA we can use"
             if h["pair"] != "SC_Single" else
             f"My HFE-assessed income is ${h['income']:,} a month and I have ${h['cpf']:,} in CPF OA I can use")
    if "cash" not in omit:
        money += f", plus ${h['cash']:,} in cash savings"
    we = "I" if h["pair"] == "SC_Single" else "we"
    our = "my" if h["pair"] == "SC_Single" else "our"
    facts = (f"{who}, buying a resale flat. {money}. {we.capitalize()} have no other debts, "
             f"{'have' if we == 'we' else 'have'} been continuously employed for the past 12 months, and {our} eligibility "
             f"for the grants and an HDB loan is confirmed. No proximity grant applies.")
    return (f"{facts} {search} Use {spec['lease']} years of remaining lease for the grant and loan calculation. "
            f"Map the matches and tell {'me' if we == 'I' else 'us'} {our} grants and maximum budget.")


# ---------------------------------------------------------------------------
# Ground truth with the real tools and the Supervisor's own rules
# ---------------------------------------------------------------------------
_RUN: Dict[str, ToolRunner] = {}


def tool(name: str) -> ToolRunner:
    if name not in _RUN:
        _RUN[name] = ToolRunner(name)
    return _RUN[name]


def search(town: str, ft: str, budget: float, lease: int, radii: Dict[str, int], cap: Optional[float] = None
           ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, List[str]]:
    sql = tool("query_resale_blocks_by_criteria")(towns=[town], flat_types=[ft], max_price=budget, min_lease_years=lease)
    if "error" in sql:
        raise RuntimeError(f"price query failed: {sql['error']}")
    spatial = tool("filter_blocks_by_amenity_proximity")(towns=[town], amenity_constraints=radii)
    if isinstance(spatial, dict) and "error" in spatial:
        raise RuntimeError(f"distance filter failed: {spatial['error']}")
    blocks = sv.compute_intersection(sql["blocks"], spatial, cap)
    return blocks, sql.get("metadata") or {}, len(spatial), [block_key(b["block"], b["street"]) for b in blocks]


def amenity_coverage(town: str, radii: Dict[str, int]) -> Dict[str, bool]:
    """True when the town has the amenity in the spatial index within a generous 2 km (so 'no match' is not 'no data')."""
    return {k: bool(tool("filter_blocks_by_amenity_proximity")(towns=[town], amenity_constraints={k: 2000})) for k in radii}


def relaxations(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = load_config()["architecture"]["relaxation_steps"]
    town, ft, budget, lease, radii = spec["town"], spec["ft"], spec["budget"], spec["lease"], spec["radii"]
    out = []
    relaxed_budget = round(budget * (1 + steps["max_budget_sgd_pct"] / 100), -3)
    out.append(("max_budget_sgd", budget, relaxed_budget, search(town, ft, relaxed_budget, lease, radii)[3]))
    for k, v in radii.items():
        r = {**radii, k: v + steps["amenity_radius_meters"]}
        out.append((f"amenity_radii_meters.{k}", v, v + steps["amenity_radius_meters"], search(town, ft, budget, lease, r)[3]))
    relaxed_lease = max(0, lease - steps["min_remaining_lease_years"])
    out.append(("min_remaining_lease_years", lease, relaxed_lease, search(town, ft, budget, relaxed_lease, radii)[3]))
    return [{"constraint": c, "original": o, "relaxed": n, "unlocked_blocks": len(keys), "unlocked_block_keys": keys}
            for c, o, n, keys in out]


def finance(facts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Member 1's real calculators, with the Supervisor's hand-off (a successful grant total feeds affordability)."""
    g = tool("calculate_cpf_housing_grants")(**facts["grants"])
    aff_args = dict(facts["affordability"])
    if g.get("status") == "ok" and aff_args.get("total_grants") is None:
        aff_args["total_grants"] = g["total_grants"]
    a = tool("calculate_maximum_affordability")(**aff_args)
    keep_g = ("status", "missing_inputs", "enhanced_housing_grant", "family_grant", "proximity_housing_grant", "total_grants", "rule_version")
    keep_a = ("status", "missing_inputs", "max_purchase_price", "max_loan_amount", "monthly_instalment", "binding_constraint",
              "grants_included", "rule_version")
    return {"calculate_cpf_housing_grants": {k: g.get(k) for k in keep_g},
            "calculate_maximum_affordability": {k: a.get(k) for k in keep_a}}


def expected_calls(spec: Dict[str, Any], facts: Optional[Dict[str, Dict[str, Any]]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    agents = ["market_sql", "spatial"] + (["financial"] if facts else [])
    calls = [
        {"agent": "market_sql", "tool": "query_resale_blocks_by_criteria",
         "required_params": {"towns": [spec["town"]], "flat_types": [spec["ft"]], "max_price": float(spec["budget"]),
                             "min_lease_years": spec["lease"]}},
        {"agent": "spatial", "tool": "filter_blocks_by_amenity_proximity",
         "required_params": {"towns": [spec["town"]], "amenity_constraints": dict(spec["radii"])}},
    ]
    if facts:
        # Every stated fact is scored; a fact the question leaves out is not expected (the agent should ask for it).
        calls.append({"agent": "financial", "tool": "calculate_cpf_housing_grants", "required_params": dict(facts["grants"])})
        calls.append({"agent": "financial", "tool": "calculate_maximum_affordability", "required_params": dict(facts["affordability"])})
    return agents, calls


def check_extraction(spec: Dict[str, Any], text: str) -> List[str]:
    """The deterministic extractor must recover the search constraints exactly from the generated wording."""
    c = sv.rule_based_extract(text)
    problems = []
    if c.towns != [spec["town"]]:
        problems.append(f"towns {c.towns}")
    if c.flat_types != [spec["ft"]]:
        problems.append(f"flat types {c.flat_types}")
    if c.max_budget_sgd != spec["budget"]:
        problems.append(f"budget {c.max_budget_sgd}")
    if c.min_remaining_lease_years != spec["lease"]:
        problems.append(f"lease {c.min_remaining_lease_years}")
    if c.amenity_radii_meters != spec["radii"]:
        problems.append(f"radii {c.amenity_radii_meters}")
    return problems


def ground(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    text = question_text(spec)
    problems = check_extraction(spec, text)
    facts = financial_facts(spec)
    fin = finance(facts) if facts else None
    cap = None
    if fin and fin["calculate_maximum_affordability"]["status"] == "ok":
        cap = min(float(spec["budget"]), fin["calculate_maximum_affordability"]["max_purchase_price"])
    blocks, meta, n_spatial, keys = search(spec["town"], spec["ft"], spec["budget"], spec["lease"], spec["radii"], cap)
    if meta.get("truncated"):
        problems.append("price query truncated at the result limit")
    coverage = amenity_coverage(spec["town"], spec["radii"])
    if not all(coverage.values()):
        problems.append(f"no amenity data in town: {coverage}")
    truth: Dict[str, Any] = {
        "data": {"snapshot_id": meta.get("snapshot_id"), "window": f"{meta.get('window_start_month')}..{meta.get('window_end_month')}",
                 "lease_reference_month": meta.get("lease_reference_month"),
                 "financial_rule_version": (fin or {}).get("calculate_maximum_affordability", {}).get("rule_version")},
        "price_query_blocks": meta.get("total_matching_blocks"),
        "distance_filter_blocks": n_spatial,
        "affordability_cap": cap,
        "matching_block_keys": keys,
    }
    omit = spec.get("omit", [])
    if spec["sub"] == "E1":
        if not keys:
            problems.append("E1 has no matching block")
        g_status = fin["calculate_cpf_housing_grants"]["status"]
        a_status = fin["calculate_maximum_affordability"]["status"]
        want_g = "needs_information" if "ages" in omit else "ok"
        want_a = "needs_information" if omit else "ok"
        if (g_status, a_status) != (want_g, want_a):
            problems.append(f"financial status grants={g_status} affordability={a_status}, expected {want_g}/{want_a}")
        truth["financial"] = fin
        behaviour = ("Recommend exactly the matching blocks (median price within the affordability cap), map them, and report the "
                     "grant total and maximum purchase price from the calculators.")
        if omit:
            missing = sorted(set(fin["calculate_cpf_housing_grants"]["missing_inputs"] or [])
                             | set(fin["calculate_maximum_affordability"]["missing_inputs"] or []))
            truth["missing_financial_inputs"] = missing
            behaviour = ("Recommend exactly the matching blocks and map them; do not invent a maximum budget. Say which facts are "
                         f"missing ({', '.join(missing)}) and ask for them.")
    else:
        if keys:
            problems.append(f"E2 has {len(keys)} matching block(s)")
        rel = relaxations(spec)
        truth["relaxations"] = rel
        best = max(rel, key=lambda r: r["unlocked_blocks"])
        truth["best_single_change"] = best if best["unlocked_blocks"] > 0 else None
        if truth["best_single_change"] is None:
            problems.append("no single relaxation step unlocks a block")
        behaviour = ("Say that no block meets every constraint, present nothing as a match, and offer the single change that "
                     "unlocks blocks with its number.")
    if omit:
        # The second turn: the user answers the follow-up question; ground truth after it uses the complete facts.
        h = HOUSEHOLDS[spec["hh"]]
        answer = " ".join(([f"We have ${h['cash']:,} in cash savings."] if "cash" in omit else [])
                          + ([f"We are aged {h['ages'][0]} and {h['ages'][1]}."] if "ages" in omit else []))
        full, full_problems = ground({k: v for k, v in spec.items() if k != "omit"})
        problems += [f"follow-up: {p}" for p in full_problems]
        full_truth = full["ground_truth_facts"]
        item_follow_up = {"answer": answer, "financial_inputs": full["financial_inputs"],
                          "ground_truth_facts": {k: full_truth[k] for k in ("matching_block_keys", "affordability_cap",
                                                                            "financial", "data")},
                          "expected_behaviour": "Keep the earlier search, complete the grant and budget estimate with the "
                                                "answer, and recommend exactly the matching blocks within the new cap."}
    agents, calls = expected_calls(spec, facts)
    item = {"id": spec["id"], "category": "E", "sub_use_case": spec["sub"], "query": text, "initial_map_state": None,
            "stated_constraints": {"towns": [spec["town"]], "flat_types": [spec["ft"]], "max_budget_sgd": float(spec["budget"]),
                                   "min_remaining_lease_years": spec["lease"], "amenity_radii_meters": dict(spec["radii"])},
            "financial_inputs": facts, "omitted_financial_facts": omit or None,
            "expected_agents": agents, "expected_mcp_calls": calls, "expected_behaviour": behaviour,
            "ground_truth_facts": truth}
    if omit:
        item["follow_up"] = item_follow_up
    return item, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify every spec, write nothing")
    args = ap.parse_args()
    items, failed = [], 0
    for spec in SPECS:
        item, problems = ground(spec)
        t = item["ground_truth_facts"]
        fin = t.get("financial") or {}
        g, a = fin.get("calculate_cpf_housing_grants", {}), fin.get("calculate_maximum_affordability", {})
        extra = (f" | grants {g.get('status')} {g.get('total_grants')} | afford {a.get('status')} {a.get('max_purchase_price')}"
                 if fin else f" | best change {(t.get('best_single_change') or {}).get('constraint')} "
                             f"-> {(t.get('best_single_change') or {}).get('unlocked_blocks')} block(s)")
        print(f"{'OK  ' if not problems else 'FAIL'} {item['id']} matches={len(t['matching_block_keys'])} "
              f"(price {t['price_query_blocks']}, distance {t['distance_filter_blocks']}, cap {t['affordability_cap']}){extra}")
        for p in problems:
            print(f"       - {p}")
        if item.get("follow_up"):
            fu = item["follow_up"]["ground_truth_facts"]
            fa = fu["financial"]["calculate_maximum_affordability"]
            print(f"       follow-up \"{item['follow_up']['answer']}\" -> matches={len(fu['matching_block_keys'])} "
                  f"cap {fu['affordability_cap']} | afford {fa['status']} {fa['max_purchase_price']}")
        failed += bool(problems)
        items.append(item)
    if failed:
        sys.exit(f"{failed} spec(s) failed; nothing written")
    if not args.check:
        OUT.write_text(json.dumps(items, indent=2) + "\n")
        print(f"wrote {len(items)} grounded questions to {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
