"""
Architecture Variant B — Coarse 2-Agent Split (agents/arch_2_coarse.py).
Owned by Member 4. Proposal §6.2 Experiment 1, Variant B.

A supervisor routes between:
  data_agent   — combined Market SQL + Geospatial + Financial agent (11 tools)
  policy_agent — Policy RAG agent (1 tool)
No intersection node and no runtime auditor: the combined agent must reconcile
SQL and spatial results on its own, which is the specific weakness this
variant is meant to expose relative to C.

    START → extract_constraints → route_coarse ─┬→ data_agent ───┐
                                                └→ policy_agent ─┤ → synthesize → finalize → END
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv
from agents.schemas import TOOL_OWNER
from agents.state import HDBCompassState

VARIANT = "B"
DATA_TOOLS = [t for t, a in TOOL_OWNER.items() if a != "policy_rag"]
POLICY_TOOLS = [t for t, a in TOOL_OWNER.items() if a == "policy_rag"]

DATA_SYSTEM = (
    "You are the HDB data agent: you have market SQL, geospatial proximity and financial calculator tools. "
    "Use whichever tools the user's constraints need, with exact towns/flat types/money, then reconcile the "
    "results yourself (e.g. keep only blocks that appear in BOTH the SQL and spatial results) and summarise."
)


def route_coarse_node(state: HDBCompassState) -> Dict[str, Any]:
    routed = sv.route_node(state)  # also adds a clarifying question when nothing can be routed
    fine = routed["active_subagents"]
    active: List[str] = []
    if any(a in fine for a in ("market_sql", "spatial", "financial")):
        active.append("data")
    if "policy_rag" in fine:
        active.append("policy")
    return {**routed, "active_subagents": active}


def dispatch_coarse(state: HDBCompassState) -> List[Send]:
    active = state.get("active_subagents", [])
    return [Send(f"{a}_agent", state) for a in active] if active else [Send("synthesize", state)]


def data_agent_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    outputs, log, errors, final = sv.run_react("data", DATA_TOOLS, DATA_SYSTEM, sv.last_user_query(state), c,
                                               max_iter=sv.load_config()["architecture"]["subagent_max_tool_iterations"] * 2,
                                               plan_agents=["market_sql", "spatial", "financial"])
    update = sv.slot_outputs(outputs, c.get("towns") or [])
    # No join in code in B (by design): the candidate list stays empty and the writer reconciles the price and
    # distance results itself, from the data agent's own summary plus the raw tool outputs (supervisor.synthesis_evidence).
    update["candidate_blocks_intersection"] = []
    update.update({"tool_call_log": log, "errors": errors,
                   "subagent_notes": [{"agent": "data", "text": (final or "")[:4000]}] if final else []})
    return update


def policy_agent_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    outputs, log, errors, final = sv.run_react("policy", POLICY_TOOLS, sv.AGENT_SYSTEM["policy_rag"], sv.last_user_query(state), c,
                                               plan_agents=["policy_rag"])
    update = sv.slot_outputs(outputs, c.get("towns") or [])
    update.update({"tool_call_log": log, "errors": errors,
                   "subagent_notes": [{"agent": "policy", "text": (final or "")[:4000]}] if final else []})
    return update


def build_graph():
    g = StateGraph(HDBCompassState)
    g.add_node("extract_constraints", sv.extract_constraints_node)
    g.add_node("route_coarse", route_coarse_node)
    g.add_node("data_agent", data_agent_node)
    g.add_node("policy_agent", policy_agent_node)
    g.add_node("synthesize", sv.synthesize_node)
    g.add_node("finalize", sv.finalize_node)
    g.add_edge(START, "extract_constraints")
    g.add_edge("extract_constraints", "route_coarse")
    g.add_conditional_edges("route_coarse", dispatch_coarse, ["data_agent", "policy_agent", "synthesize"])
    g.add_edge("data_agent", "synthesize")
    g.add_edge("policy_agent", "synthesize")
    g.add_edge("synthesize", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


graph = build_graph()


def run(query: str, map_state: Optional[Dict[str, Any]] = None) -> HDBCompassState:
    return graph.invoke(sv.initial_state(query, VARIANT, map_state))


if __name__ == "__main__":
    out = run("We are comparing a Prime BTO vs a Plus resale flat in Bishan under $750k. What are the exact differences in MOP and subsidy clawback?")
    print("active:", out["active_subagents"], "| tool calls:", [(r["agent"], r["tool"]) for r in out["tool_call_log"]])
    print(out["draft_response"][:600])
