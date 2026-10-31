"""
Builder and ground-truth verifier for Category B (Spatial & Commute) benchmark queries (evaluation/build_category_b.py).
Owned by Member 2 (Categories B & E).

Category B benchmarks spatial reasoning across 3 distinct sub-use cases (15 queries total):
  * B1: MOE Primary School Priority Boundary (1km distance filters via verify_primary_school_distance)
  * B2: Multi-Amenity Proximity Filtering (conjunctions of MRT/hawker/supermarket via filter_blocks_by_amenity_proximity)
  * B3: Dual-Transit Commute Filtering (commute time thresholds to two distinct employment hubs via compute_dual_transit_commute)

Ground truth is evaluated using Member 2's real spatial and routing tools:
  - filter_blocks_by_amenity_proximity
  - verify_primary_school_distance
  - compute_dual_transit_commute
alongside Member 1's query_resale_blocks_by_criteria and supervisor intersection semantics.

Usage:
    python -m evaluation.build_category_b          # build and write evaluation/category_b_queries.json
    python -m evaluation.build_category_b --check  # dry-run verification
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv
from agents.schemas import block_key
from agents.tool_registry import ToolRunner

OUT = PROJECT_ROOT / "evaluation" / "category_b_queries.json"

_RUN: Dict[str, ToolRunner] = {}


def tool(name: str) -> ToolRunner:
    if name not in _RUN:
        _RUN[name] = ToolRunner(name)
    return _RUN[name]


# ---------------------------------------------------------------------------
# Category B Specs: 5 B1, 5 B2, 5 B3
# ---------------------------------------------------------------------------
SPECS: List[Dict[str, Any]] = [
    # ---- B1: Primary School Priority Boundary (1km)
    dict(
        id="B1-01",
        sub="B1",
        town="PASIR RIS",
        ft="4 ROOM",
        budget=600000,
        lease=60,
        school="Meridian Primary School",
        bracket="1km",
        query="We want to apply for Primary 1 ballot priority. Find 4-room resale flats in Pasir Ris under $600,000 with at least 60 years of lease left, within 1km of Meridian Primary School. Show the matching blocks on the map.",
        behaviour="Identify 4-room resale blocks in Pasir Ris under $600k with >=60 years lease within 1km of Meridian Primary School and sync them to the map."
    ),
    dict(
        id="B1-02",
        sub="B1",
        town="BISHAN",
        ft="4 ROOM",
        budget=800000,
        lease=55,
        school="Ai Tong School",
        bracket="1km",
        query="We are looking for 4-room resale flats in Bishan under $800,000 with at least 55 years of lease left, within 1km of Ai Tong School for P1 registration. Which blocks qualify? Please display them on the map.",
        behaviour="Filter 4-room blocks in Bishan under $800k with >=55 years lease within 1km of Ai Tong School and map the candidate blocks."
    ),
    dict(
        id="B1-03",
        sub="B1",
        town="SERANGOON",
        ft="4 ROOM",
        budget=600000,
        lease=60,
        school="Rosyth School",
        bracket="1km",
        query="Find 4-room resale flats in Serangoon under $600,000 with at least 60 years of lease left, within 1km of Rosyth School. Map the candidate blocks.",
        behaviour="Retrieve 4-room blocks in Serangoon under $600k with >=60 years lease within 1km of Rosyth School and map them."
    ),
    dict(
        id="B1-04",
        sub="B1",
        town="CLEMENTI",
        ft="5 ROOM",
        budget=900000,
        lease=50,
        school="Nan Hua Primary School",
        bracket="1km",
        query="Which 5-room resale flats in Clementi under $900,000 with at least 50 years of lease left are within 1km of Nan Hua Primary School? Map all qualifying blocks.",
        behaviour="Locate 5-room Clementi blocks under $900k with >=50 years lease within 1km of Nan Hua Primary School and sync to the interactive map."
    ),
    dict(
        id="B1-05",
        sub="B1",
        town="TOA PAYOH",
        ft="4 ROOM",
        budget=750000,
        lease=50,
        school="Pei Chun Public School",
        bracket="1km",
        query="Find 4-room resale flats in Toa Payoh under $750,000 with at least 50 years of lease left, within 1km of Pei Chun Public School. Please map the results.",
        behaviour="Find 4-room resale flats in Toa Payoh under $750k with >=50 years lease within 1km of Pei Chun Public School and present them on the map."
    ),

    # ---- B2: Multi-Amenity Proximity Filtering
    dict(
        id="B2-01",
        sub="B2",
        town="ANG MO KIO",
        ft="4 ROOM",
        budget=750000,
        lease=65,
        radii={"mrt": 400, "hawker": 500},
        query="Find 4-room resale flats in Ang Mo Kio under $750,000 with at least 65 years of lease left, within 400m of an MRT station and within 500m of a hawker centre. Put them on the map.",
        behaviour="Filter 4-room blocks in Ang Mo Kio under $750k with >=65 years lease satisfying both <=400m MRT and <=500m hawker centre proximity constraints."
    ),
    dict(
        id="B2-02",
        sub="B2",
        town="BISHAN",
        ft="4 ROOM",
        budget=800000,
        lease=60,
        radii={"mrt": 400, "supermarket": 300},
        query="Show me 4-room resale flats in Bishan under $800,000 with at least 60 years of lease left, within 400m of an MRT station and within 300m of a supermarket. Map the matching blocks.",
        behaviour="Filter 4-room blocks in Bishan under $800k with >=60 years lease within 400m of MRT and 300m of a supermarket, then sync to map."
    ),
    dict(
        id="B2-03",
        sub="B2",
        town="BEDOK",
        ft="4 ROOM",
        budget=700000,
        lease=60,
        radii={"mrt": 400, "hawker": 500},
        query="We want convenience in Bedok. Find 4-room resale flats under $700,000 with at least 60 years of lease left, within 400m of an MRT station and within 500m of a hawker centre. Display them on the map.",
        behaviour="Retrieve 4-room Bedok blocks under $700k with >=60 years lease within 400m of an MRT and 500m of a hawker centre."
    ),
    dict(
        id="B2-04",
        sub="B2",
        town="TAMPINES",
        ft="4 ROOM",
        budget=650000,
        lease=60,
        radii={"mrt": 400, "supermarket": 300},
        query="Find 4-room resale flats in Tampines under $650,000 with at least 60 years of lease left, within 400m of an MRT station and within 300m of a supermarket. Show the results on the map.",
        behaviour="Filter 4-room Tampines blocks under $650k with >=60 years lease satisfying MRT <=400m and supermarket <=300m."
    ),
    dict(
        id="B2-05",
        sub="B2",
        town="CLEMENTI",
        ft="4 ROOM",
        budget=800000,
        lease=50,
        radii={"mrt": 400, "hawker": 500},
        query="Which 4-room resale flats in Clementi under $800,000 with at least 50 years of lease left are within 400m of an MRT station and within 500m of a hawker centre? Map the matches.",
        behaviour="Locate 4-room Clementi blocks under $800k with >=50 years lease within 400m of both an MRT station and a hawker centre."
    ),

    # ---- B3: Dual-Transit Commute Filtering
    dict(
        id="B3-01",
        sub="B3",
        town="QUEENSTOWN",
        ft="4 ROOM",
        budget=950000,
        lease=60,
        work_a="Raffles Place",
        work_b="one-north",
        max_mins=25,
        radii={"mrt": 400},
        query="I work at Raffles Place and work at one-north. Find 4-room resale flats in Queenstown under $950,000 with at least 60 years of lease left within 400m of an MRT station, with commute under 25 minutes to each workplace. Map the qualifying blocks.",
        behaviour="Filter 4-room Queenstown blocks under $950k with >=60 years lease near MRT whose public transit commute to both Raffles Place and one-north is under 25 minutes."
    ),
    dict(
        id="B3-02",
        sub="B3",
        town="TOA PAYOH",
        ft="4 ROOM",
        budget=850000,
        lease=60,
        work_a="Marina Bay",
        work_b="Orchard",
        max_mins=25,
        radii={"mrt": 400},
        query="We work at Marina Bay and work at Orchard. Find 4-room resale flats in Toa Payoh under $850,000 with at least 60 years of lease left within 400m of an MRT station, with commute under 25 minutes to each workplace. Show them on the map.",
        behaviour="Evaluate 4-room Toa Payoh blocks under $850k with >=60 years lease near MRT and verify joint transit commute under 25 minutes to Marina Bay and Orchard."
    ),
    dict(
        id="B3-03",
        sub="B3",
        town="BUKIT MERAH",
        ft="4 ROOM",
        budget=850000,
        lease=60,
        work_a="Raffles Place",
        work_b="Jurong East",
        max_mins=30,
        radii={"mrt": 400},
        query="We work at Raffles Place and work at Jurong East. Find 4-room resale flats in Bukit Merah under $850,000 with at least 60 years of lease left within 400m of an MRT station, with commute under 30 minutes to both offices. Map the candidates.",
        behaviour="Identify 4-room Bukit Merah blocks under $850k with >=60 years lease near MRT with transit commute under 30 minutes to both Raffles Place and Jurong East."
    ),
    dict(
        id="B3-04",
        sub="B3",
        town="KALLANG/WHAMPOA",
        ft="4 ROOM",
        budget=850000,
        lease=60,
        work_a="City Hall",
        work_b="Bugis",
        max_mins=25,
        radii={"mrt": 400},
        query="We work at City Hall and work at Bugis. Which 4-room resale flats in Kallang/Whampoa under $850,000 with at least 60 years of lease left are within 400m of an MRT station with commute under 25 minutes to both locations? Map the matches.",
        behaviour="Retrieve 4-room Kallang/Whampoa blocks under $850k with >=60 years lease near MRT with under 25 minutes transit commute to City Hall and Bugis."
    ),
    dict(
        id="B3-05",
        sub="B3",
        town="BISHAN",
        ft="4 ROOM",
        budget=850000,
        lease=60,
        work_a="Raffles Place",
        work_b="one-north",
        max_mins=35,
        radii={"mrt": 400},
        query="I work at Raffles Place and work at one-north. Find 4-room resale flats in Bishan under $850,000 with at least 60 years of lease left within 400m of an MRT station, with commute under 35 minutes to both destinations. Map the qualifying blocks.",
        behaviour="Filter 4-room Bishan blocks under $850k with >=60 years lease near MRT with transit commute under 35 minutes to Raffles Place and one-north."
    ),
]


def ground_spec(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    sub = spec["sub"]
    town = spec["town"]
    ft = spec["ft"]
    budget = float(spec["budget"])
    lease = int(spec["lease"])
    problems = []

    expected_agents = ["market_sql", "spatial"]
    expected_calls: List[Dict[str, Any]] = [
        {
            "agent": "market_sql",
            "tool": "query_resale_blocks_by_criteria",
            "required_params": {
                "towns": [town],
                "flat_types": [ft],
                "max_price": budget,
                "min_lease_years": lease,
            }
        }
    ]

    sql_res = tool("query_resale_blocks_by_criteria")(
        towns=[town],
        flat_types=[ft],
        max_price=budget,
        min_lease_years=lease
    )
    if "error" in sql_res:
        problems.append(f"price query failed: {sql_res['error']}")
        sql_blocks = []
    else:
        sql_blocks = sql_res.get("blocks", [])

    spatial_outputs: Dict[str, Any] = {}

    if sub == "B1":
        school = spec["school"]
        bracket = spec.get("bracket", "1km")
        expected_calls.append({
            "agent": "spatial",
            "tool": "verify_primary_school_distance",
            "required_params": {"school_name": school, "bracket": bracket}
        })
        lines = tool("verify_primary_school_distance")(school_name=school, bracket=bracket)
        if isinstance(lines, dict) and "error" in lines:
            problems.append(f"school tool error: {lines['error']}")
            lines = []
        spatial_outputs["verify_primary_school_distance"] = lines

    elif sub == "B2":
        radii = spec["radii"]
        expected_calls.append({
            "agent": "spatial",
            "tool": "filter_blocks_by_amenity_proximity",
            "required_params": {"towns": [town], "amenity_constraints": dict(radii)}
        })
        blocks = tool("filter_blocks_by_amenity_proximity")(towns=[town], amenity_constraints=radii)
        if isinstance(blocks, dict) and "error" in blocks:
            problems.append(f"proximity tool error: {blocks['error']}")
            blocks = []
        spatial_outputs["filter_blocks_by_amenity_proximity"] = blocks

    elif sub == "B3":
        work_a = spec["work_a"]
        work_b = spec["work_b"]
        max_mins = spec["max_mins"]
        radii = spec.get("radii", {"mrt": 400})
        expected_calls.append({
            "agent": "spatial",
            "tool": "filter_blocks_by_amenity_proximity",
            "required_params": {"towns": [town], "amenity_constraints": dict(radii)}
        })
        expected_calls.append({
            "agent": "spatial",
            "tool": "compute_dual_transit_commute",
            "required_params": {"work_dest_a": work_a, "work_dest_b": work_b, "max_minutes": max_mins}
        })
        mrt_blocks = tool("filter_blocks_by_amenity_proximity")(towns=[town], amenity_constraints=radii)
        spatial_outputs["filter_blocks_by_amenity_proximity"] = mrt_blocks

        cand_strs = [f"Blk {b['block']} {b['street']}" for b in mrt_blocks]
        res = tool("compute_dual_transit_commute")(
            candidate_blocks=cand_strs,
            work_dest_a=work_a,
            work_dest_b=work_b,
            max_minutes=max_mins
        )
        if isinstance(res, dict) and "error" in res:
            problems.append(f"commute tool error: {res['error']}")
            res = []
        spatial_outputs["compute_dual_transit_commute"] = res

    slots = sv.slot_outputs(spatial_outputs, [town])
    spatial_cands = slots.get("spatial_block_candidates") or []
    intersection = sv.compute_intersection(sql_blocks, spatial_cands, budget)

    matching_keys = [block_key(b["block"], b["street"]) for b in intersection]

    if not matching_keys:
        problems.append(f"zero matching blocks found for {spec['id']}")

    truth: Dict[str, Any] = {
        "price_query_blocks": len(sql_blocks),
        "spatial_candidate_blocks": len(spatial_cands),
        "matching_block_keys": matching_keys,
    }

    stated = {
        "towns": [town],
        "flat_types": [ft],
        "max_budget_sgd": budget,
        "min_remaining_lease_years": lease,
        **({"amenity_radii_meters": dict(spec["radii"])} if spec.get("radii") else {}),
        **({"school_name": spec["school"]} if spec.get("school") else {}),
        **({"work_destinations": [spec["work_a"], spec["work_b"]], "max_commute_minutes": spec["max_mins"]} if spec.get("work_a") else {}),
    }

    item = {
        "id": spec["id"],
        "category": "B",
        "sub_use_case": sub,
        "query": spec["query"],
        "initial_map_state": None,
        "stated_constraints": stated,
        "expected_agents": expected_agents,
        "expected_mcp_calls": expected_calls,
        "expected_behaviour": spec["behaviour"],
        "ground_truth_facts": truth
    }
    return item, problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Category B benchmark questions")
    parser.add_argument("--check", action="store_true", help="Dry run check only without writing file")
    args = parser.parse_args()

    items = []
    failed = 0
    for spec in SPECS:
        item, problems = ground_spec(spec)
        t = item["ground_truth_facts"]
        n_match = len(t["matching_block_keys"])
        status = "OK  " if not problems else "FAIL"
        print(f"{status} {item['id']} ({item['sub_use_case']}) {spec['town']}: {n_match} matching blocks (price: {t['price_query_blocks']}, spatial: {t['spatial_candidate_blocks']})")
        for p in problems:
            print(f"      - {p}")
        failed += bool(problems)
        items.append(item)

    if failed:
        sys.exit(f"{failed} spec(s) failed; aborting")

    if not args.check:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {len(items)} Category B queries to {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
