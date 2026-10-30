"""
Extra Category E questions, grounded on the real data (evaluation/ground_category_e_extra.py). Owned by Member 4.

The 15 grounded questions (ground_category_e.py) use six towns, one template and only four tools. These are a fresh
check that the agents were never tuned on: new towns, a primary-school distance limit, the upfront cost (stamp duty
and cash) of a stated purchase price, and follow-ups that supply the income or the CPF balance. Each question is
written by hand in the way a user would ask it; next to it the spec states, in structured form, exactly the facts
the question gives. Ground truth comes from the same real tools the agents call (Member 1's resale query and
calculators, Member 2's distance filter and school check), joined the way ground_category_e.py does it.

A spec is rejected unless it meets its definition (E1: at least one block meets everything, every calculator has
the status the question is designed for; E2: no block meets everything and one relaxation step unlocks one), or if
the truth could be measuring something other than what the words ask: a price query (first or relaxed) cut off at the
result limit, a school tool that measured from a different school, or a block counted as near "an MRT station" when
only an LRT stop is near it (Member 2's index files LRT stops under "mrt", so such questions say "MRT or LRT"). How
the no-model extractor reads the wording is printed as a note only: these questions must not be shaped to fit it.

    .venv/bin/python -m evaluation.ground_category_e_extra            # check and write evaluation/category_e_extra.json
    .venv/bin/python -m evaluation.ground_category_e_extra --check    # check only, write nothing
    LLM_PROFILE=none .venv/bin/python evaluation/run_category_e_prelim.py --dataset extra --tag extra1

Rebuild after a database refresh, a rules update or a spatial-index rebuild, as for ground_category_e.py.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv  # noqa: E402
from agents.schemas import block_key  # noqa: E402
from evaluation.ground_category_e import amenity_coverage, relaxations, tool  # noqa: E402

OUT = PROJECT_ROOT / "evaluation" / "category_e_extra.json"

# Households. omit in a spec names the facts the first message leaves out: "income", "cpf", "cash" or "ages".
HOUSEHOLDS: Dict[str, Dict[str, Any]] = {
    "sc_sc_29_28": dict(pair="SC_SC", ages=(29, 28), income=7200, cpf=70000, cash=30000),
    "sc_sc_33_31": dict(pair="SC_SC", ages=(33, 31), income=10500, cpf=140000, cash=70000),
    "sc_sc_31_30": dict(pair="SC_SC", ages=(31, 30), income=8000, cpf=85000, cash=35000),
    "single_40": dict(pair="SC_Single", ages=(40, 40), income=5000, cpf=95000, cash=40000),
}

SPECS: List[Dict[str, Any]] = [
    # (a) a town not used before; full financial facts, said the way a couple would say them. The stations near
    # Sengkang and Punggol blocks are mostly LRT stops, which the distance filter counts as "mrt", so these say LRT.
    dict(id="E1-X01", sub="E1", town="SENGKANG", ft="4 ROOM", budget=600000, lease=80, radii={"mrt": 400},
         hh="sc_sc_29_28", calc_lease=80,
         query="Hi, my wife and I are both Singapore citizens, 29 and 28, married, buying our first home, no kids yet. "
               "We earn $7,200 a month combined (that's the HFE-assessed figure), have $70,000 in our CPF ordinary "
               "accounts and $30,000 in cash. No car loan or other debts, we've both been working continuously for the "
               "past 12 months, and HDB has confirmed our eligibility for the grants and an HDB loan. We don't qualify "
               "for the proximity grant. We'd like a 4-room resale in Sengkang under $600,000, at least 80 years of "
               "lease left, no more than 400m from an MRT or LRT station. Assume an 80-year remaining lease for the sums. "
               "Which blocks work, what grants can we get and what's the most we can pay? Please put them on the map."),
    # (a) + (b) another new town, with a primary-school distance limit (search only)
    dict(id="E1-X02", sub="E1", town="PASIR RIS", ft="4 ROOM", budget=600000, lease=60, radii={},
         school="Meridian Primary School",
         query="Our son starts Primary 1 in 2028 and we want to live within 1km of Meridian Primary School for the "
               "ballot priority. We're looking at 4-room resale flats in Pasir Ris, max $600k, with at least 60 years "
               "left on the lease. Which blocks qualify? Show me on the map please."),
    # (c) upfront cost: stamp duty and cash for a stated purchase price, valuation, loan and fees
    dict(id="E1-X03", sub="E1", town="PUNGGOL", ft="5 ROOM", budget=750000, lease=80, radii={"mrt": 400},
         hh="sc_sc_33_31", calc_lease=80,
         upfront=dict(purchase_price=720000.0, market_valuation=710000.0, loan_amount=520000.0,
                      legal_and_valuation_fees=3500.0, additional_buyer_stamp_duty=0.0),
         query="We're a married couple, both Singapore citizens aged 33 and 31, buying our first flat, no children. "
               "Our combined HFE-assessed income is $10,500 a month, we have $140,000 in CPF OA between us and $70,000 "
               "cash in the bank. No other loans, both continuously employed for the last 12 months, and our grant and "
               "HDB loan eligibility has been confirmed. No proximity grant. We're looking for a 5-room resale in "
               "Punggol under $750,000 with at least 80 years of lease, within 400m of an MRT or LRT station. Say we buy one "
               "for $720,000 and it's valued at $710,000, with an HDB loan of $520,000, legal and valuation fees of "
               "$3,500 and no ABSD. Using 80 years of remaining lease, what grants do we get, what's our maximum "
               "budget, how much is the stamp duty and how much cash do we need upfront? Map the blocks too."),
    # (d) two-step: the first message leaves out the income; the follow-up supplies it
    dict(id="E1-X04", sub="E1", town="SERANGOON", ft="4 ROOM", budget=700000, lease=70, radii={"mrt": 400},
         hh="sc_sc_31_30", calc_lease=70, omit=["income"],
         query="My husband and I are Singapore citizens, 31 and 30, first-time buyers with no kids, looking at resale. "
               "We have $85,000 in CPF OA and $35,000 in cash, no other debts, both have worked continuously for the "
               "past 12 months, and HDB has confirmed our eligibility for the grants and an HDB loan. No proximity "
               "grant for us. Can you find 4-room resale flats in Serangoon below $700,000 with 70+ years of lease "
               "left and within 400m of an MRT? Work out the grants and loan on a 70-year remaining lease, show the "
               "blocks on the map, and tell us our grants and maximum budget.",
         answer="Sorry, forgot to say: our HFE-assessed household income is $8,000 a month."),
    # (d) two-step: the first message leaves out the CPF balance; the follow-up supplies it
    dict(id="E1-X05", sub="E1", town="BUKIT BATOK", ft="3 ROOM", budget=450000, lease=60, radii={"mrt": 400},
         hh="single_40", calc_lease=60, omit=["cpf"],
         query="I'm 40, single, a Singapore citizen and buying my first flat (no children). My HFE-assessed income is "
               "$5,000 a month and I've saved $40,000 in cash. I have no debts, I've been continuously employed for "
               "the past 12 months, and my eligibility for the grants and an HDB loan is confirmed. No proximity grant. "
               "I'm after a 3-room resale in Bukit Batok for less than $450,000, with at least 60 years of lease "
               "remaining and within 400m of an MRT station. Please assume a 60-year remaining lease for the grant "
               "and loan, map what you find, and tell me my grants and maximum budget.",
         answer="I have $95,000 in my CPF ordinary account."),
    # (e) a relaxation case in a new town (search only)
    dict(id="E2-X01", sub="E2", town="HOUGANG", ft="4 ROOM", budget=600000, lease=65, radii={"mrt": 400},
         query="Is there any 4-room resale in Hougang under $600,000 that still has at least 65 years of lease and is "
               "within 400m of an MRT station? If nothing fits, what should I compromise on?"),
]


# ---------------------------------------------------------------------------
# Stated facts, in Member 1's financial_inputs shape
# ---------------------------------------------------------------------------
def financial_facts(spec: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """The calculator facts the first message states (None for search-only questions)."""
    if "hh" not in spec:
        return None
    h, omit = HOUSEHOLDS[spec["hh"]], set(spec.get("omit", []))
    old, young = h["ages"]
    grants = dict(flat_category="resale", citizenship_pair=h["pair"], first_timer_status=True, flat_type=spec["ft"],
                  remaining_lease_years=float(spec["calc_lease"]), eligibility_confirmed=True,
                  continuous_employment_12_months=True, proximity="none")
    aff = dict(loan_type="hdb", remaining_lease_years=float(spec["calc_lease"]), monthly_debt_obligations=0.0,
               loan_eligibility_confirmed=True)
    if "income" not in omit:
        grants["monthly_income"] = aff["monthly_income"] = float(h["income"])
    if "cpf" not in omit:
        aff["cpf_oa_balance"] = float(h["cpf"])
    if "cash" not in omit:
        aff["cash_savings"] = float(h["cash"])
    if "ages" not in omit:
        grants.update(youngest_buyer_age=young, youngest_core_member_age=young)
        aff.update(youngest_buyer_age=young, oldest_buyer_age=old, average_applicant_age=(old + young) / 2)
    out = {"grants": grants, "affordability": aff}
    if spec.get("upfront"):
        out["upfront"] = dict(spec["upfront"], loan_type="hdb")
    return out


# ---------------------------------------------------------------------------
# Ground truth with the real tools
# ---------------------------------------------------------------------------
def school_blocks(school: str) -> List[str]:
    """Block keys inside 1 km of the school, read from the school tool's own lines ("Blk 123 STREET (456m)").
    Parsed here rather than with the Supervisor's parser, so a fault in that parser is not copied into the truth."""
    keys = []
    for line in tool("verify_primary_school_distance")(school_name=school, bracket="1km"):
        m = re.match(r"Blk\s+(\S+)\s+(.+?)\s*\(\d+m\)$", line)
        if m:
            keys.append(block_key(m.group(1), m.group(2)))
    return keys


def school_problems(school: str) -> List[str]:
    """The school tool falls back to an online geocoder for a name missing from the local index, and otherwise measures
    from the first school whose name contains the query less "primary" and "school" ("Punggol Primary School" finds
    Punggol Cove). Ground truth must depend on neither, so the tool's blocks must be exactly the blocks within 1 km of
    the named school's own entry in the index."""
    from mcp_server.spatial_onemap_tools import get_spatial_index
    index = get_spatial_index()["payload"]
    named = [s for s in index["schools"] if s["name"].upper() == school.upper()]
    if not named:
        return [f"{school} is not in the local spatial index"]
    sx, sy = named[0]["x"], named[0]["y"]
    near = {block_key(b["block"], b["street"]) for b in index["blocks"] if math.hypot(b["x"] - sx, b["y"] - sy) <= 1000}
    got = set(school_blocks(school))
    if got != near:
        return [f"the school tool's blocks differ from those within 1 km of {school} ({len(got ^ near)} differ): it "
                "measured from another school"]
    return []


def lrt_only_blocks(keys: List[str], radius: float) -> List[str]:
    """Keys of blocks that are within the radius of a station only because Member 2's index files LRT stops under
    "mrt". A question whose truth counts such blocks must say LRT, or the truth answers a different question."""
    from mcp_server.spatial_onemap_tools import get_spatial_index
    index = get_spatial_index()["payload"]
    mrt = [(a["x"], a["y"]) for a in index["amenities"] if a["category"].lower() == "mrt" and "LRT" not in a["name"].upper()]
    where = {block_key(b["block"], b["street"]): (b["x"], b["y"]) for b in index["blocks"]}
    return [k for k in keys if k in where and min(math.hypot(x - where[k][0], y - where[k][1]) for x, y in mrt) > radius]


def search(spec: Dict[str, Any], cap: Optional[float] = None) -> Tuple[List[str], Dict[str, Any], Optional[int]]:
    """Blocks meeting the price, lease, amenity and school limits, then the affordability cap. Returns
    (matching block keys, price-query metadata, blocks passing the distance and school checks or None)."""
    sql = tool("query_resale_blocks_by_criteria")(towns=[spec["town"]], flat_types=[spec["ft"]],
                                                  max_price=spec["budget"], min_lease_years=spec["lease"])
    if "error" in sql:
        raise RuntimeError(f"price query failed: {sql['error']}")
    near: Optional[set] = None
    if spec["radii"]:
        spatial = tool("filter_blocks_by_amenity_proximity")(towns=[spec["town"]], amenity_constraints=spec["radii"])
        if isinstance(spatial, dict) and "error" in spatial:
            raise RuntimeError(f"distance filter failed: {spatial['error']}")
        near = {block_key(b["block"], b["street"]) for b in spatial}
    if spec.get("school"):
        by_school = set(school_blocks(spec["school"]))
        near = by_school if near is None else near & by_school
    keys = [block_key(b["block"], b["street"]) for b in sql["blocks"]
            if (near is None or block_key(b["block"], b["street"]) in near)
            and (cap is None or b.get("median_resale_price") is None or b["median_resale_price"] <= cap)]
    return keys, sql.get("metadata") or {}, (len(near) if near is not None else None)


def finance(facts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Member 1's real calculators with the Supervisor's hand-off: a successful grant total feeds affordability and
    the upfront tool; affordability's LTV, the CPF balance and the cash feed the upfront tool."""
    g = tool("calculate_cpf_housing_grants")(**facts["grants"])
    aff_args = dict(facts["affordability"])
    if g.get("status") == "ok":
        aff_args["total_grants"] = g["total_grants"]
    a = tool("calculate_maximum_affordability")(**aff_args)
    keep_g = ("status", "missing_inputs", "enhanced_housing_grant", "family_grant", "proximity_housing_grant", "total_grants",
              "rule_version")
    keep_a = ("status", "missing_inputs", "max_purchase_price", "max_loan_amount", "monthly_instalment", "binding_constraint",
              "grants_included", "rule_version")
    out = {"calculate_cpf_housing_grants": {k: g.get(k) for k in keep_g},
           "calculate_maximum_affordability": {k: a.get(k) for k in keep_a}}
    if facts.get("upfront"):
        up_args = dict(facts["upfront"])
        for k in ("cpf_oa_balance", "cash_savings"):
            if k in facts["affordability"]:
                up_args[k] = facts["affordability"][k]
        if g.get("status") == "ok":
            up_args["total_grants"] = g["total_grants"]
        if a.get("status") == "ok":
            up_args["ltv_ratio"] = a["ltv_ratio"]
        u = tool("calculate_bsd_and_upfront_outlay")(**up_args)
        keep_u = ("status", "missing_inputs", "buyer_stamp_duty", "cash_required", "total_upfront", "cash_shortfall",
                  "cash_over_valuation", "cpf_oa_usable", "grants_used", "rule_version")
        out["calculate_bsd_and_upfront_outlay"] = {k: u.get(k) for k in keep_u}
    return out


def expected_calls(spec: Dict[str, Any], facts: Optional[Dict[str, Dict[str, Any]]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    agents = ["market_sql", "spatial"] + (["financial"] if facts else [])
    calls = [{"agent": "market_sql", "tool": "query_resale_blocks_by_criteria",
              "required_params": {"towns": [spec["town"]], "flat_types": [spec["ft"]], "max_price": float(spec["budget"]),
                                  "min_lease_years": spec["lease"]}}]
    if spec["radii"]:
        calls.append({"agent": "spatial", "tool": "filter_blocks_by_amenity_proximity",
                      "required_params": {"towns": [spec["town"]], "amenity_constraints": dict(spec["radii"])}})
    if spec.get("school"):
        calls.append({"agent": "spatial", "tool": "verify_primary_school_distance",
                      "required_params": {"school_name": spec["school"], "bracket": "1km"}})
    if facts:
        # every stated fact is scored; a fact the question leaves out is not expected (the agent should ask for it)
        calls.append({"agent": "financial", "tool": "calculate_cpf_housing_grants", "required_params": dict(facts["grants"])})
        calls.append({"agent": "financial", "tool": "calculate_maximum_affordability", "required_params": dict(facts["affordability"])})
        if facts.get("upfront"):
            calls.append({"agent": "financial", "tool": "calculate_bsd_and_upfront_outlay", "required_params": dict(facts["upfront"])})
    return agents, calls


def extraction_notes(spec: Dict[str, Any]) -> List[str]:
    """Where the no-model extractor misreads the wording. A note, not a rejection: the wording is the user's."""
    c = sv.rule_based_extract(spec["query"])
    radii = dict(spec["radii"], **({"school": 1000} if spec.get("school") else {}))
    notes = [f"{name} read as {got!r}, stated {want!r}" for name, got, want in (
        ("towns", c.towns, [spec["town"]]), ("flat types", c.flat_types, [spec["ft"]]),
        ("budget", c.max_budget_sgd, spec["budget"]), ("lease", c.min_remaining_lease_years, spec["lease"]),
        ("radii", c.amenity_radii_meters, radii)) if got != want]
    if spec.get("school") and (c.school_name or "").upper() != spec["school"].upper():
        notes.append(f"school read as {c.school_name!r}, stated {spec['school']!r}")
    want = financial_facts(spec)
    if want:
        got = sv.stated_financial_inputs(spec["query"])
        for sec, facts in want.items():
            wrong = sorted(k for k, v in facts.items() if (got.get(sec) or {}).get(k) != v)
            if wrong:
                notes.append(f"{sec} facts not read from the words: {', '.join(wrong)}")
    return notes


def ground(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    problems: List[str] = []
    omit = spec.get("omit", [])
    facts = financial_facts(spec)
    fin = finance(facts) if facts else None
    cap = None
    if fin and fin["calculate_maximum_affordability"]["status"] == "ok":
        cap = min(float(spec["budget"]), fin["calculate_maximum_affordability"]["max_purchase_price"])
    keys, meta, n_near = search(spec, cap)
    if meta.get("truncated"):
        problems.append("price query truncated at the result limit")
    if spec["radii"]:
        coverage = amenity_coverage(spec["town"], spec["radii"])
        if not all(coverage.values()):
            problems.append(f"no amenity data in town: {coverage}")
    if spec.get("school"):
        problems += school_problems(spec["school"])
    truth: Dict[str, Any] = {
        "data": {"snapshot_id": meta.get("snapshot_id"), "window": f"{meta.get('window_start_month')}..{meta.get('window_end_month')}",
                 "lease_reference_month": meta.get("lease_reference_month"),
                 "financial_rule_version": (fin or {}).get("calculate_maximum_affordability", {}).get("rule_version")},
        "price_query_blocks": meta.get("total_matching_blocks"),
        "distance_filter_blocks": n_near,
        "affordability_cap": cap,
        "matching_block_keys": keys,
    }
    if spec["sub"] == "E1":
        if not keys:
            problems.append("E1 has no matching block")
        behaviour = "Recommend exactly the matching blocks and map them."
        if fin:
            want = "needs_information" if omit else "ok"
            want_g = "needs_information" if set(omit) & {"income", "ages"} else "ok"
            got = {"grants": fin["calculate_cpf_housing_grants"]["status"],
                   "affordability": fin["calculate_maximum_affordability"]["status"]}
            wanted = {"grants": want_g, "affordability": want}
            if "calculate_bsd_and_upfront_outlay" in fin:
                got["upfront"], wanted["upfront"] = fin["calculate_bsd_and_upfront_outlay"]["status"], want
            if got != wanted:
                problems.append(f"financial status {got}, expected {wanted}")
            truth["financial"] = fin
            behaviour = ("Recommend exactly the matching blocks (median price within the affordability cap), map them, and report the "
                         "grant total and maximum purchase price from the calculators"
                         + (", and the buyer's stamp duty and cash required upfront." if facts.get("upfront") else "."))
            if omit:
                missing = sorted(set(fin["calculate_cpf_housing_grants"]["missing_inputs"] or [])
                                 | set(fin["calculate_maximum_affordability"]["missing_inputs"] or []))
                truth["missing_financial_inputs"] = missing
                behaviour = ("Recommend exactly the matching blocks and map them; do not invent a maximum budget. Say which facts are "
                             f"missing ({', '.join(missing)}) and ask for them.")
    else:
        if keys:
            problems.append(f"E2 has {len(keys)} matching block(s)")
        rel = relaxations(spec, problems)
        truth["relaxations"] = rel
        best = max(rel, key=lambda r: r["unlocked_blocks"])
        truth["best_single_change"] = best if best["unlocked_blocks"] > 0 else None
        if truth["best_single_change"] is None:
            problems.append("no single relaxation step unlocks a block")
        behaviour = ("Say that no block meets every constraint, present nothing as a match, and offer the single change that "
                     "unlocks blocks with its number.")
    if spec["radii"].get("mrt") and "LRT" not in spec["query"]:
        # a question that says only "MRT" must not count a block that is near an LRT stop and no MRT station
        mrt = spec["radii"]["mrt"]
        counted = [(keys, mrt)] + [(r["unlocked_block_keys"], r["relaxed"] if r["constraint"] == "amenity_radii_meters.mrt" else mrt)
                                   for r in truth.get("relaxations", [])]
        lrt = sorted({k for ks, radius in counted for k in lrt_only_blocks(ks, radius)})
        if lrt:
            problems.append(f"{len(lrt)} counted block(s) are near an LRT stop but no MRT station, e.g. {lrt[0]}: "
                            "say 'MRT or LRT station' in the question")
    agents, calls = expected_calls(spec, facts)
    item = {"id": spec["id"], "category": "E", "sub_use_case": spec["sub"], "query": spec["query"], "initial_map_state": None,
            "stated_constraints": {"towns": [spec["town"]], "flat_types": [spec["ft"]], "max_budget_sgd": float(spec["budget"]),
                                   "min_remaining_lease_years": spec["lease"],
                                   "amenity_radii_meters": dict(spec["radii"], **({"school": 1000} if spec.get("school") else {})),
                                   **({"school_name": spec["school"]} if spec.get("school") else {})},
            "financial_inputs": facts, "omitted_financial_facts": omit or None,
            "expected_agents": agents, "expected_mcp_calls": calls, "expected_behaviour": behaviour,
            "ground_truth_facts": truth}
    if omit:
        # the second turn: the user answers the follow-up question; ground truth after it uses the complete facts
        full, full_problems = ground({k: v for k, v in spec.items() if k not in ("omit", "answer")})
        problems += [f"follow-up: {p}" for p in full_problems]
        item["follow_up"] = {"answer": spec["answer"], "financial_inputs": full["financial_inputs"],
                             "ground_truth_facts": {k: full["ground_truth_facts"][k] for k in
                                                    ("matching_block_keys", "affordability_cap", "financial", "data")},
                             "expected_behaviour": "Keep the earlier search, complete the grant and budget estimate with the "
                                                   "answer, and recommend exactly the matching blocks within the new cap."}
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
        u = fin.get("calculate_bsd_and_upfront_outlay")
        extra = (f" | grants {g.get('status')} {g.get('total_grants')} | afford {a.get('status')} {a.get('max_purchase_price')}"
                 if fin else "")
        extra += f" | upfront {u['status']} BSD {u['buyer_stamp_duty']} cash {u['cash_required']}" if u else ""
        if spec["sub"] == "E2":
            best = t.get("best_single_change") or {}
            extra += f" | best change {best.get('constraint')} -> {best.get('unlocked_blocks')} block(s)"
        print(f"{'OK  ' if not problems else 'FAIL'} {item['id']} {spec['town']} matches={len(t['matching_block_keys'])} "
              f"(price {t['price_query_blocks']}, distance/school {t['distance_filter_blocks']}, cap {t['affordability_cap']}){extra}")
        for p in problems:
            print(f"       - {p}")
        for n in extraction_notes(spec):
            print(f"       note: no-model extractor: {n}")
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
