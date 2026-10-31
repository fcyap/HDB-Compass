"""
Architecture Variant C — Proposed 5-Agent Specialist Graph + Runtime Auditor
(agents/arch_3_specialist_5agent.py). Owned by Member 4. Proposal §4.1, §6.2.

    input_guardrail ──(blocked)──────────────────────────────────────┐
          │                                                          │
    extract_constraints → route ─┬→ market_sql_agent ─┐              │
                                 ├→ spatial_agent ────┤ (parallel    │
                                 ├→ financial_agent ──┤  Send fan-out)│
                                 └→ policy_rag_agent ─┘              │
                                                      ↓              │
                                           intersect_candidates      │
                                                      ↓              │
                                 (empty?) relax_constraints          │
                                                      ↓              │
                                                 synthesize ←──┐     │
                                                      ↓        │     │
                                              runtime_auditor ─┘ (fail, ≤2 retries)
                                                      ↓              │
                                                  finalize ←─────────┘

Member 2's geo_agent subgraph is kept as `invoke_spatial_agent_subgraph` for the
map-first demo path; inside Variant C the spatial specialist calls Member 2's four
MCP tools directly so that Tool Selection F1 is measured the same way for every agent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv
from agents.state import HDBCompassState
from agents.tracing import run_config

VARIANT = "C"


def build_graph():
    g = StateGraph(HDBCompassState)
    g.add_node("input_guardrail", sv.input_guardrail_node)
    g.add_node("extract_constraints", sv.extract_constraints_node)
    g.add_node("route", sv.route_node)
    for agent in sv.SPECIALISTS:
        g.add_node(f"{agent}_agent", sv.make_specialist_node(agent))
    g.add_node("intersect_candidates", sv.intersect_node)
    g.add_node("relax_constraints", sv.relax_node)
    g.add_node("synthesize", sv.synthesize_node)
    g.add_node("runtime_auditor", sv.audit_node)
    g.add_node("finalize", sv.finalize_node)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", sv.guardrail_gate, ["extract_constraints", "finalize"])
    g.add_edge("extract_constraints", "route")
    g.add_conditional_edges("route", sv.dispatch, [f"{a}_agent" for a in sv.SPECIALISTS] + ["synthesize"])
    for agent in sv.SPECIALISTS:
        g.add_edge(f"{agent}_agent", "intersect_candidates")
    g.add_conditional_edges("intersect_candidates", sv.should_relax, ["relax_constraints", "synthesize"])
    g.add_edge("relax_constraints", "synthesize")
    g.add_edge("synthesize", "runtime_auditor")
    g.add_conditional_edges("runtime_auditor", sv.audit_gate, ["synthesize", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


graph = build_graph()


def run(query: str, map_state: Optional[Dict[str, Any]] = None,
        previous: Optional[Dict[str, Any]] = None, question_id: Optional[str] = None) -> HDBCompassState:
    """Uniform entry point for the eval runner (docs/INTERFACES.md §6). previous: the last turn's final state.
    question_id names the benchmark question in the LangSmith trace (ignored when tracing is off)."""
    return graph.invoke(sv.initial_state(query, VARIANT, map_state, previous),
                        config=run_config(VARIANT, question_id=question_id))


# ---------------------------------------------------------------------------
# Member 2's adapter, preserved from their commit (used by the map-first demo).
# ---------------------------------------------------------------------------
def invoke_spatial_agent_subgraph(state: HDBCompassState) -> Dict[str, Any]:
    from agents.spatial_agent import geo_agent as spatial_agent_graph
    query = sv.last_user_query(state)
    constraints = state.get("user_constraints", {})
    radii = constraints.get("amenity_radii_meters", {})
    sub_res = spatial_agent_graph.invoke({"query": query, "radius_meters": float(radii.get("school", radii.get("mrt", 1000)))})
    return {"spatial_block_candidates": sub_res.get("spatial_block_candidates", []),
            "map_ui_payload": sub_res.get("map_ui_payload", {}),
            "draft_response": sub_res.get("summary", "")}


if __name__ == "__main__":
    import json
    q = ("We earn $9,000/month, have $90k CPF, want a 4-room flat in Tampines under $680k, "
         "within 300m of an MRT and hawker centre, with 65+ years lease. Map the matches.")
    out = run(q)
    print("active_subagents:", out["active_subagents"])
    print("tool calls:", [(r["agent"], r["tool"]) for r in out["tool_call_log"]])
    print("candidates:", len(out["candidate_blocks_intersection"]))
    print("auditor:", out["auditor_verdict"])
    print("errors:", out["errors"])
    print("\n" + out["draft_response"])
    print("\nmap payload keys:", list(out["map_ui_payload"].keys()))
