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

Add-on tools (agents/addons): under addons.offer_in: all, A gets the same tools as B and C and the same "when to use"
line for each (supervisor.addon_prompt). It does not get the analysis-mode data guide that C's market agent and B's
data agent get for a data question no specific intent covers: that mode is set by the extraction step, which A does
not have by design, and the guide tells the model to end with one sentence and no numbers because code writes the
figures, whereas A's model writes its own answer. Giving it to A would either need an extraction step or cost A its
figures, so the difference is part of the architecture being measured.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import addons
from agents import supervisor as sv
from agents.schemas import TOOL_OWNER
from agents.state import HDBCompassState
from agents.tracing import run_config

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
    c["requested_tools"] = sv.requested_tools(users)   # the fallback plan runs upfront costs and comparables only when asked
    # Same tools as B and C (add-on tools only under addons.offer_in: all), with the same "when to use" lines C's
    # specialists get for them, and the same coordinates a tool call may use (the map pin and the user's own).
    outputs, log, errors, final_text = sv.run_react(
        "monolith", addons.visible_tools(list(TOOL_OWNER), VARIANT),
        MONOLITH_SYSTEM + sv.addon_prompt(sv.SPECIALISTS, VARIANT), query, c,
        max_iter=sv.load_config()["architecture"]["subagent_max_tool_iterations"] * 2,
        plan_agents=sv.SPECIALISTS, known_points=sv.trusted_points(state))
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
        previous: Optional[Dict[str, Any]] = None, question_id: Optional[str] = None) -> HDBCompassState:
    """Uniform entry point for the eval runner (docs/INTERFACES.md §6). previous: the last turn's final state.
    question_id names the benchmark question in the LangSmith trace (ignored when tracing is off)."""
    return graph.invoke(sv.initial_state(query, VARIANT, map_state, previous),
                        config=run_config(VARIANT, question_id=question_id))


if __name__ == "__main__":
    out = run("Find 5-room flats in Queenstown or Bukit Merah under $850k with at least 65 years of remaining lease within 300m of an MRT.")
    print("tool calls:", [(r["agent"], r["tool"]) for r in out["tool_call_log"]])
    print(out["draft_response"][:800])
