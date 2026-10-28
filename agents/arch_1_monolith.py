"""
Architecture Variant A — Baseline Monolithic ReAct Agent (agents/arch_1_monolith.py).
Owned by Member 4. Proposal §6.2 Experiment 1, Variant A.

One ReAct agent, all 12 tools in a flat list, naive RAG (same retrieval tool
without the reranker stage: pipeline hint RAG-1), no constraint extraction, no
supervisor, no intersection step, no runtime auditor. The model must plan,
call tools and combine results on its own — this is the baseline whose
tool-selection and multi-objective failures Variant C is meant to fix.

    START → monolith_agent → finalize → END

Deterministic mode (LLM_PROFILE=none): runs every specialist's plan in one loop
(what an ideal monolith would call) and a template answer, so the eval pipeline
can be exercised without a model. Routing metrics for A are therefore trivial in
that mode — the LLM run is the one that matters.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv
from agents.schemas import TOOL_OWNER
from agents.state import HDBCompassState

VARIANT = "A"

MONOLITH_SYSTEM = (
    "You are HDB-Compass, a Singapore public-housing assistant with tools for resale market data, geospatial "
    "proximity, grant/loan calculators and HDB policy retrieval. Decide which tools answer the user's request, "
    "call them with exact parameters (HDB town names upper case, flat types like '4 ROOM', money as numbers), "
    "combine the results yourself, then answer in markdown using only numbers returned by tools. "
    "Cite policy passages by source_url. Do not speculate on future prices."
)


def monolith_node(state: HDBCompassState) -> Dict[str, Any]:
    query = sv.conversation_text(state)  # the monolith sees the whole conversation, as a chat agent would
    # No extraction step by design; the regex pass only feeds the deterministic fallback plan, and reads the
    # user's own messages so numbers quoted in an earlier assistant reply are never taken as constraints.
    users = " ".join(m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user")
    c = sv.rule_based_extract(users).model_dump()
    outputs, log, errors, final_text = sv.run_react(
        "monolith", list(TOOL_OWNER), MONOLITH_SYSTEM, query, c,
        max_iter=sv.load_config()["architecture"]["subagent_max_tool_iterations"] * 2,
        plan_agents=sv.SPECIALISTS)
    update = sv.slot_outputs(outputs, c.get("towns") or [])
    update["user_constraints"] = c
    update["active_subagents"] = ["monolith"]
    update["tool_call_log"] = log
    update["errors"] = errors
    if final_text.strip():
        update["draft_response"] = final_text
    else:
        # template answer needs a candidate list; the monolith has no intersection node, so use raw SQL rows
        sql_blocks = sv._sql_blocks({**state, **update}) or update.get("spatial_block_candidates") or []
        update["candidate_blocks_intersection"] = sql_blocks
        update["draft_response"] = sv.template_response({**state, **update})
    return update


def build_graph():
    g = StateGraph(HDBCompassState)
    g.add_node("monolith_agent", monolith_node)
    g.add_node("finalize", sv.finalize_node)
    g.add_edge(START, "monolith_agent")
    g.add_edge("monolith_agent", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


graph = build_graph()


def run(query: str, map_state: Optional[Dict[str, Any]] = None,
        previous: Optional[Dict[str, Any]] = None) -> HDBCompassState:
    """Uniform entry point for the eval runner (docs/INTERFACES.md §6). previous: the last turn's final state."""
    return graph.invoke(sv.initial_state(query, VARIANT, map_state, previous))


if __name__ == "__main__":
    out = run("Find 5-room flats in Queenstown or Bukit Merah under $850k with at least 65 years of remaining lease within 300m of an MRT.")
    print("tool calls:", [(r["agent"], r["tool"]) for r in out["tool_call_log"]])
    print(out["draft_response"][:800])
