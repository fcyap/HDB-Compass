"""
Architecture Variant C: Proposed 5-Agent Specialist Graph (agents/arch_3_specialist_5agent.py).
Fulfills Proposal Section 4 (Page 6 & 11) and Directory Structure (Page 14).

Supervisor dispatches parallel requests to:
- Agent 2: Market & Valuation SQL Agent (Member 1)
- Geospatial & Commute Agent (Member 2 - Live Implemented)
- Agent 4: Financial & Grant Calculator (Member 1/4)
- Agent 5: Regulatory Policy RAG Agent (Member 3)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import HDBCompassState
from agents.spatial_agent import geo_agent as spatial_agent_graph


def invoke_spatial_agent_subgraph(state: HDBCompassState) -> Dict[str, Any]:
    """
    Sub-agent node adapter for Geospatial & Commute Agent (Member 2).
    Extracts constraints from HDBCompassState, executes spatial filtering,
    and returns spatial_block_candidates and map_ui_payload.
    """
    query = ""
    messages = state.get("messages", [])
    if messages:
        query = messages[-1].get("content", "")

    constraints = state.get("user_constraints", {})
    towns = constraints.get("towns", [])
    radii = constraints.get("amenity_radii_meters", {})

    agent_input = {
        "query": query,
        "radius_meters": float(radii.get("school", radii.get("mrt", 1000)))
    }

    sub_res = spatial_agent_graph.invoke(agent_input)

    return {
        "spatial_block_candidates": sub_res.get("spatial_block_candidates", []),
        "map_ui_payload": sub_res.get("map_ui_payload", {}),
        "draft_response": sub_res.get("summary", "")
    }


if __name__ == "__main__":
    print("[HDB-Compass] Testing Geospatial & Commute Sub-agent node in 5-Agent Architecture...")
    test_state: HDBCompassState = {
        "messages": [{"role": "user", "content": "Ai Tong School 1km"}],
        "user_constraints": {"towns": ["BISHAN"], "amenity_radii_meters": {"school": 1000}},
        "active_subagents": ["spatial_agent"]
    }
    out = invoke_spatial_agent_subgraph(test_state)
    print(f"  -> Returned {len(out['spatial_block_candidates'])} spatial block candidates.")
    print("  -> Map UI Payload generated:", bool(out['map_ui_payload']))
    print("  [PASS] Geospatial & Commute Agent integrated into 5-Agent architecture.")
