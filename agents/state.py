"""
Shared LangGraph State Schema (agents/state.py).
Directly implements Section 4.2 (Page 7) of the HDB-Compass Project Proposal.
Enables inspection in LangSmith and live synchronization with the interactive map.
"""
from __future__ import annotations

import operator
from typing import TypedDict, List, Dict, Optional, Annotated, Any


class UserConstraints(TypedDict, total=False):
    towns: List[str]
    flat_types: List[str]
    max_budget_sgd: Optional[float]
    min_remaining_lease_years: Optional[int]
    amenity_radii_meters: Dict[str, int]   # e.g. {"mrt": 300, "hawker": 200, "school": 1000}
    household_income: Optional[float]
    citizenship_scheme: Optional[str]      # e.g. "SC_SC_FirstTimer", "SC_Single_35"


class HDBCompassState(TypedDict, total=False):
    """
    Global HDB-Compass typed state shared across:
    - Agent 1: Supervisor & Constraint Negotiator
    - Agent 2: Market & Valuation SQL Agent
    - Geospatial & Commute Agent (Member 2)
    - Agent 4: Financial & Grant Calculator Agent
    - Agent 5: Regulatory Policy RAG Agent
    - Runtime Auditor / LLM-as-a-Judge Node
    """
    messages: Annotated[List[dict], operator.add]
    user_constraints: UserConstraints
    active_subagents: List[str]
    sql_market_results: Optional[Dict[str, Any]]
    spatial_block_candidates: Optional[List[Dict[str, Any]]]   # Owned by Member 2
    financial_breakdown: Optional[Dict[str, Any]]
    rag_policy_passages: Optional[List[Dict[str, Any]]]
    candidate_blocks_intersection: List[Dict[str, Any]]
    map_ui_payload: Dict[str, Any]                             # Synced to frontend map (GeoJSON markers + buffer rings)
    draft_response: str
    auditor_verdict: Dict[str, Any]                            # {"passed": bool, "critique": str, "retry_count": int}
