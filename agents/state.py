"""
Shared LangGraph State Schema (agents/state.py).
Directly implements Section 4.2 (Page 7) of the HDB-Compass Project Proposal.
Enables inspection in LangSmith and live synchronization with the interactive map.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class UserConstraints(TypedDict, total=False):
    towns: List[str]
    flat_types: List[str]
    max_budget_sgd: Optional[float]
    min_remaining_lease_years: Optional[int]
    amenity_radii_meters: Dict[str, int]   # e.g. {"mrt": 300, "hawker": 200, "school": 1000}
    household_income: Optional[float]
    citizenship_scheme: Optional[str]      # e.g. "SC_SC_FirstTimer", "SC_Single_35"
    financial_inputs: Optional[Dict[str, Any]]
    comparable_resale: Optional[Dict[str, Any]]


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

    # --- Member 4 additions (optional keys; additive, nothing above changes) ---
    tool_call_log: Annotated[List[Dict[str, Any]], operator.add]   # schemas.ToolCallRecord dicts -> Tool F1 / Param EM
    relaxation: Optional[Dict[str, Any]]                           # schemas.RelaxUnsatisfiedConstraintsOutput (use case E2)
    variant: str                                                   # "A" | "B" | "C" — which architecture produced this run
    errors: Annotated[List[str], operator.add]                     # non-fatal sub-agent errors surfaced to the UI/eval
    subagent_notes: Annotated[List[Dict[str, Any]], operator.add]  # {"agent", "text"}: each sub-agent's own final summary
    # Follow-up flow for missing financial facts (Member 4). clarification: what this turn asks the user for;
    # prior_constraints / pending_clarification: carried in from the previous turn by initial_state(previous=...).
    clarification: Optional[Dict[str, Any]]          # {"needed": [calculator fields], "question": str}
    prior_constraints: Optional[Dict[str, Any]]
    pending_clarification: Optional[Dict[str, Any]]
