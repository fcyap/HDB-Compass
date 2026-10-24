"""
Pydantic contract for every MCP tool input/output and for the supervisor's
structured outputs (agents/schemas.py). Owned by Member 4.

This is the single source of truth for what crosses the LangGraph <-> MCP
boundary. Members 1, 3 and 5 implement real tools against these models;
mcp_server/mock_tools.py returns fixtures that validate against them.

Naming: <ToolName>Input / <ToolName>Output. Block identity everywhere is the
string key  f"{block} {street}".upper()  (see block_key()).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------
def block_key(block: str, street: str) -> str:
    """Canonical identity for an HDB block across SQL, spatial and map payloads."""
    return f"{block} {street}".strip().upper()


class BlockRef(BaseModel):
    """Minimum fields every tool that returns blocks must include."""
    block: str = Field(..., examples=["123"])
    street: str = Field(..., examples=["BISHAN ST 13"])
    town: str = Field(..., examples=["BISHAN"])
    postal: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

    @property
    def key(self) -> str:
        return block_key(self.block, self.street)


# ---------------------------------------------------------------------------
# Agent 1 — Supervisor structured outputs (LLM -> graph)
# ---------------------------------------------------------------------------
class ExtractedConstraints(BaseModel):
    """What the constraint-extraction node must produce from natural language.
    Mirrors agents/state.py::UserConstraints, with the extra fields the router needs."""
    towns: List[str] = Field(default_factory=list)
    flat_types: List[str] = Field(default_factory=list, examples=[["4 ROOM"]])
    max_budget_sgd: Optional[float] = None
    min_remaining_lease_years: Optional[int] = None
    amenity_radii_meters: Dict[str, int] = Field(default_factory=dict, examples=[{"mrt": 300, "hawker": 200}])
    household_income: Optional[float] = None
    citizenship_scheme: Optional[Literal["SC_SC_FirstTimer", "SC_PR_FirstTimer", "SC_Single_35", "SC_SC_SecondTimer"]] = Field(
        None, description="Household scheme; leave null unless the user states citizenship or household type")
    cpf_oa_balance: Optional[float] = None
    cash_savings: Optional[float] = None
    school_name: Optional[str] = None
    work_destinations: List[str] = Field(default_factory=list)
    max_commute_minutes: Optional[int] = None
    policy_question: Optional[str] = Field(None, description="Regulatory question, if any, for Agent 5")
    wants_price_trend: bool = False
    wants_bto_comparison: bool = False


class RoutingDecision(BaseModel):
    """Which sub-agents the supervisor dispatches (in parallel)."""
    active_subagents: List[Literal["market_sql", "spatial", "financial", "policy_rag"]]
    rationale: str = ""


class RelaxationSuggestion(BaseModel):
    constraint: str = Field(..., examples=["max_budget_sgd"])
    original: Any
    relaxed: Any
    unlocked_blocks: int


class RelaxUnsatisfiedConstraintsOutput(BaseModel):
    """Output of the deterministic zero-result relaxation node (use case E2)."""
    original_constraints: Dict[str, Any]
    suggestions: List[RelaxationSuggestion]
    best_single_change: Optional[RelaxationSuggestion] = None


# ---------------------------------------------------------------------------
# Agent 2 — Market & Valuation SQL (Member 1)
# ---------------------------------------------------------------------------
class QueryResaleBlocksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    towns: List[str]
    flat_types: List[str]
    max_price: float = Field(..., ge=0, allow_inf_nan=False)
    min_lease_years: int = Field(..., ge=0, le=99)

    @field_validator("towns", "flat_types")
    @classmethod
    def normalise_filters(cls, values: List[str]) -> List[str]:
        result = [" ".join(value.upper().split()) for value in values]
        if any(not value for value in result):
            raise ValueError("Filter values must not be blank")
        return list(dict.fromkeys(result))


class ResaleBlockRecord(BlockRef):
    flat_type: str
    median_resale_price: float
    median_price_per_sqm: float
    median_remaining_lease_years: float
    transaction_count: int
    last_transaction_month: str = Field(..., examples=["2026-09"])


class MarketQueryMetadata(BaseModel):
    snapshot_id: str
    dataset_id: str
    source_url: str
    window_start_month: str
    window_end_month: str
    latest_source_month: str
    lease_reference_month: str
    total_matching_blocks: int
    returned_blocks: int
    result_limit: int
    truncated: bool
    notes: List[str] = Field(default_factory=list)


class QueryResaleBlocksOutput(BaseModel):
    blocks: List[ResaleBlockRecord]
    sql: str = Field("", description="The parameterised SQL that ran (for auditing)")
    metadata: Optional[MarketQueryMetadata] = None


class ComputeTownPriceTimeseriesInput(BaseModel):
    towns: List[str]
    flat_type: str
    lookback_years: int = 3


class TownQuarterlyPoint(BaseModel):
    town: str
    quarter: str = Field(..., examples=["2025Q4"])
    median_price: float
    median_price_per_sqm: float
    transaction_count: int


class TownTrendSummary(BaseModel):
    town: str
    latest_median_price: float
    yoy_appreciation_pct: float
    median_price_per_sqm: float


class ComputeTownPriceTimeseriesOutput(BaseModel):
    series: List[TownQuarterlyPoint]
    summary: List[TownTrendSummary]


class CompareBtoLaunchVsResaleInput(BaseModel):
    bto_project_name: str
    radius_meters: int = 500


class CompareBtoLaunchVsResaleOutput(BaseModel):
    bto_project_name: str
    bto_indicative_price_range: Dict[str, float] = Field(..., examples=[{"min": 420000, "max": 560000}])
    resale_median_nearby: float
    resale_sample_size: int
    price_gap_pct: float


# ---------------------------------------------------------------------------
# Agent 3 — Geospatial & Commute (Member 2, already implemented)
# Models below describe Member 2's existing return shapes so other code can validate them.
# ---------------------------------------------------------------------------
class FilterBlocksByAmenityProximityInput(BaseModel):
    towns: List[str]
    amenity_constraints: Dict[str, int] = Field(
        ..., description="Max walking distance in metres per amenity; keys: mrt, hawker, supermarket",
        examples=[{"mrt": 400, "hawker": 500}])


class SpatialBlockRecord(BlockRef):
    distances: Dict[str, float] = Field(default_factory=dict, examples=[{"mrt_dist_m": 240.5}])


class VerifyPrimarySchoolDistanceInput(BaseModel):
    school_name: str
    bracket: Literal["1km", "2km"] = "1km"


class ComputeDualTransitCommuteInput(BaseModel):
    candidate_blocks: List[str]
    work_dest_a: str
    work_dest_b: str
    max_minutes: int = 40


class SyncInteractiveMapUiInput(BaseModel):
    candidate_blocks: List[str]
    buffer_radii: Dict[str, int]
    center_latlon: List[float]


# ---------------------------------------------------------------------------
# Agent 4 — Financial & Grant Calculator (Member 1)
# ---------------------------------------------------------------------------
class CalculateCpfHousingGrantsInput(BaseModel):
    flat_category: Literal["bto", "resale"]
    citizenship_pair: Literal["SC_SC", "SC_PR", "SC_Single"] = Field(
        ..., description="SC_SC: two Singapore citizens; SC_PR: citizen + permanent resident; SC_Single: single citizen")
    first_timer_status: bool
    monthly_income: float
    near_parents_4km: bool = False


class CalculateCpfHousingGrantsOutput(BaseModel):
    enhanced_housing_grant: float
    family_grant: float
    proximity_housing_grant: float
    total_grants: float
    eligibility_notes: List[str] = Field(default_factory=list)


class CalculateMaximumAffordabilityInput(BaseModel):
    monthly_income: float
    cpf_oa_balance: float
    cash_savings: float
    loan_type: Literal["hdb", "bank"]
    interest_rate: float = 2.6
    tenure_years: int = 25


class CalculateMaximumAffordabilityOutput(BaseModel):
    max_loan_amount: float
    max_purchase_price: float
    monthly_instalment: float
    msr_cap_monthly: float
    ltv_ratio: float
    binding_constraint: Literal["MSR", "LTV", "TDSR", "DOWNPAYMENT"]


class CalculateBsdAndUpfrontOutlayInput(BaseModel):
    purchase_price: float
    loan_type: Literal["hdb", "bank"]
    total_grants: float = 0.0


class CalculateBsdAndUpfrontOutlayOutput(BaseModel):
    buyer_stamp_duty: float
    min_cash_downpayment: float
    cpf_oa_usable: float
    legal_and_valuation_fees: float
    total_upfront: float
    cash_required: float


# ---------------------------------------------------------------------------
# Agent 5 — Regulatory Policy RAG (Member 3)
# ---------------------------------------------------------------------------
class RetrieveHdbPolicyKnowledgeInput(BaseModel):
    query: str
    policy_domain: Optional[str] = Field(None, examples=["classification", "eligibility", "grants", "eip"])


class PolicyPassage(BaseModel):
    text: str
    source_url: str
    section: str = ""
    policy_category: str = ""
    flat_classification: Optional[str] = None
    effective_date: Optional[str] = None
    rerank_score: float = 0.0


class RetrieveHdbPolicyKnowledgeOutput(BaseModel):
    passages: List[PolicyPassage]
    pipeline: Literal["RAG-1", "RAG-2", "RAG-3"] = "RAG-3"


# ---------------------------------------------------------------------------
# Auditor (Member 5) and evaluation bookkeeping
# ---------------------------------------------------------------------------
class AuditorVerdict(BaseModel):
    passed: bool
    critique: str = ""
    retry_count: int = 0
    ungrounded_claims: List[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Appended to state['tool_call_log'] by every variant; feeds Tool Selection F1
    and Parameter Exact-Match in evaluation/."""
    agent: str
    tool: str
    arguments: Dict[str, Any]
    ok: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Canonical tool inventory (resolves the 11-vs-12 count in the proposal)
# ---------------------------------------------------------------------------
TOOL_OWNER: Dict[str, str] = {
    # Agent 2 — Member 1
    "query_resale_blocks_by_criteria": "market_sql",
    "compute_town_price_timeseries": "market_sql",
    "compare_bto_launch_vs_resale": "market_sql",
    # Agent 3 — Member 2
    "filter_blocks_by_amenity_proximity": "spatial",
    "verify_primary_school_distance": "spatial",
    "compute_dual_transit_commute": "spatial",
    "sync_interactive_map_ui": "spatial",
    # Agent 4 — Member 1
    "calculate_cpf_housing_grants": "financial",
    "calculate_maximum_affordability": "financial",
    "calculate_bsd_and_upfront_outlay": "financial",
    # Agent 5 — Member 3
    "retrieve_hdb_policy_knowledge": "policy_rag",
}
# Variant A (monolith) gets all 11. Variant B: combined agent gets the 10 non-RAG tools,
# the policy agent gets 1. relax_unsatisfied_constraints is a deterministic supervisor
# node, not an LLM-callable tool, so it is not counted.
assert len(TOOL_OWNER) == 11

TOOL_INPUT_MODEL: Dict[str, type[BaseModel]] = {
    "query_resale_blocks_by_criteria": QueryResaleBlocksInput,
    "compute_town_price_timeseries": ComputeTownPriceTimeseriesInput,
    "compare_bto_launch_vs_resale": CompareBtoLaunchVsResaleInput,
    "filter_blocks_by_amenity_proximity": FilterBlocksByAmenityProximityInput,
    "verify_primary_school_distance": VerifyPrimarySchoolDistanceInput,
    "compute_dual_transit_commute": ComputeDualTransitCommuteInput,
    "sync_interactive_map_ui": SyncInteractiveMapUiInput,
    "calculate_cpf_housing_grants": CalculateCpfHousingGrantsInput,
    "calculate_maximum_affordability": CalculateMaximumAffordabilityInput,
    "calculate_bsd_and_upfront_outlay": CalculateBsdAndUpfrontOutlayInput,
    "retrieve_hdb_policy_knowledge": RetrieveHdbPolicyKnowledgeInput,
}
