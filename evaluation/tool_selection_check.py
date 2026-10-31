"""
Live tool-selection check over the 12-tool inventory (evaluation/tool_selection_check.py). Member 4.

Member 1 asked for a live-model check after adding the comparable-sales tool: one plain request per tool
(the map tool is a UI side effect and is scored separately elsewhere), run through each architecture, recording
whether the intended tool was called successfully and which other tools were called. Category E exercises only
the search and finance tools, so this complements it.

    LLM_PROFILE=vllm .venv/bin/python -m evaluation.tool_selection_check            # A, B and C
    LLM_PROFILE=vllm .venv/bin/python -m evaluation.tool_selection_check --variants C
Writes evaluation/results/tool_selection_<profile>.jsonl (gitignored) and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm import active_profile  # noqa: E402
from evaluation.run_category_e_prelim import load_variant  # noqa: E402

FACTS = ("We are a married Singapore citizen couple aged 32 and 30, both first-timers with no children, buying a resale "
         "4-room flat. Our HFE-assessed household income is $9,000 a month and we have $90,000 in CPF OA we can use, plus "
         "$40,000 in cash savings. We have no other debts, have been continuously employed for the past 12 months, and our "
         "eligibility for the grants and an HDB loan is confirmed. No proximity grant applies. Assume 65 years of remaining lease.")

PROMPTS: List[Dict[str, str]] = [
    {"tool": "query_resale_blocks_by_criteria",
     "query": "List 4-room resale blocks in Bedok under $600,000 with at least 60 years of lease left."},
    {"tool": "compute_town_price_timeseries",
     "query": "How have 4-room resale prices in Tampines and Bedok trended over the last 3 years?"},
    {"tool": "query_comparable_resale_transactions",
     "query": "What did 4-room flats sell for within 500m of latitude 1.3329, longitude 103.7436 over the last 12 months?"},
    {"tool": "compare_bto_launch_vs_resale",
     "query": "Compare the indicative launch prices of the Kallang Breeze BTO with resale flats within 500m of it."},
    {"tool": "filter_blocks_by_amenity_proximity",
     "query": "Which HDB blocks in Ang Mo Kio are within 250m of an MRT station and 150m of a hawker centre?"},
    {"tool": "verify_primary_school_distance",
     "query": "Which HDB blocks fall within 1km of Ai Tong School for Primary 1 registration priority?"},
    {"tool": "compute_dual_transit_commute",
     "query": "My partner works at Raffles Place and I work at one-north. Which blocks near Buona Vista give us both a "
              "public transport commute under 35 minutes?"},
    {"tool": "calculate_cpf_housing_grants",
     "query": FACTS + " Which CPF housing grants can we get, and how much in total?"},
    {"tool": "calculate_maximum_affordability",
     "query": FACTS + " What is the maximum purchase price we can afford with an HDB loan?"},
    {"tool": "calculate_bsd_and_upfront_outlay",
     "query": "We are buying a resale flat for $600,000, valued at $600,000, with an HDB loan of $450,000 at 80% LTV. We have "
              "$100,000 in CPF OA and $50,000 in cash, quoted legal and valuation fees of $3,000, no grants, and no ABSD "
              "as first-time Singapore citizen buyers. What is our buyer's stamp duty and total upfront cash and CPF?"},
    {"tool": "retrieve_hdb_policy_knowledge",
     "query": "What is the minimum occupation period for a Prime flat, and can it be rented out whole after that?"},
]


def run(variants: List[str]) -> List[Dict[str, Any]]:
    profile = active_profile()
    out_path = PROJECT_ROOT / "evaluation" / "results" / f"tool_selection_{profile['name']}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w") as f:
        for v in variants:
            mod = load_variant(v)
            for p in PROMPTS:
                t0 = time.perf_counter()
                try:
                    state = mod.run(p["query"])
                    calls = [(r["tool"], bool(r.get("ok", True))) for r in state.get("tool_call_log", []) if r.get("agent") != "supervisor"]
                    called_ok = any(t == p["tool"] and ok for t, ok in calls)
                    others = sorted({t for t, _ in calls if t not in (p["tool"], "sync_interactive_map_ui")})
                    row = {"variant": v, "tool": p["tool"], "hit": called_ok, "other_tools": others,
                           "failed_calls": [t for t, ok in calls if not ok], "errors": state.get("errors", []),
                           "latency_s": round(time.perf_counter() - t0, 1), "draft": (state.get("draft_response") or "")[:600]}
                except Exception as e:  # noqa: BLE001
                    row = {"variant": v, "tool": p["tool"], "hit": False, "error": f"{type(e).__name__}: {e}",
                           "latency_s": round(time.perf_counter() - t0, 1)}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{v}] {p['tool']:38s} {'hit ' if row['hit'] else 'MISS'} others={row.get('other_tools')} "
                      f"{row['latency_s']}s", flush=True)
    print("\n| Tool | " + " | ".join(variants) + " |\n|---|" + "---|" * len(variants))
    for p in PROMPTS:
        cells = []
        for v in variants:
            r = next(r for r in rows if r["variant"] == v and r["tool"] == p["tool"])
            cells.append(("hit" if r["hit"] else "miss") + (f" (+{len(r.get('other_tools') or [])} other)" if r.get("other_tools") else ""))
        print(f"| {p['tool']} | " + " | ".join(cells) + " |")
    for v in variants:
        hits = sum(r["hit"] for r in rows if r["variant"] == v)
        print(f"{v}: {hits}/{len(PROMPTS)} intended tools called successfully")
    print(f"results -> {out_path.relative_to(PROJECT_ROOT)}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C")
    run([v.strip() for v in ap.parse_args().variants.split(",") if v.strip()])
