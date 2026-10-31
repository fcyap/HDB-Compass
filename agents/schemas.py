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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    nearby_schools: Optional["FindSchoolsNearAddressInput"] = Field(
        None, description="Primary schools near a residential address, not blocks near a named school. Preserve the user's address; never invent one."
    )
    work_destinations: List[str] = Field(default_factory=list)
    max_commute_minutes: Optional[int] = None
    policy_question: Optional[str] = Field(None, description="Regulatory question, if any, for Agent 5")
    wants_price_trend: bool = False
    wants_bto_comparison: bool = False
    comparable_resale: Optional["QueryComparableResaleInput"] = Field(
        None, description="Comparable-sale criteria with verified coordinates; never invent a project location"
    )
    financial_inputs: Optional["FinancialScenario"] = Field(
        None, description="Explicit financial facts only. Never invent eligibility confirmations, citizenship, age, funds, fees or zero debts/grants."
    )
    # --- research intents (Member 1; see agents/research_intents.py) ---
    addresses: List[str] = Field(default_factory=list, description="Blocks, postal codes or named places exactly as the user wrote them")
    asking_price_sgd: Optional[float] = Field(None, description="A specific flat's asking/offer price the user wants checked")
    bto_project_name: Optional[str] = Field(None, description="A named BTO launch project, e.g. 'Berlayar Rise'")
    trend_years: Optional[int] = Field(None, ge=1, le=30, description="Years for 'over the last N years' questions")
    wants_block_history: bool = False
    wants_market_snapshot: bool = False
    wants_town_ranking: bool = False
    wants_option_comparison: bool = False
    wants_price_assessment: bool = False
    wants_nearby_amenities: bool = False
    wants_area_statistics: bool = False
    area_radius_meters: Optional[int] = Field(None, ge=50, le=3000, description="Radius for an area question, e.g. 'within 300m of Blk 277'")
    assumptions: List[str] = Field(default_factory=list, description="Defaults applied to vague requests; shown to the user")
    clarifications: List[str] = Field(default_factory=list, description="Questions the user must answer; never guess these")


class AmenityRadii(BaseModel):
    """Typed radii for model extraction: strict JSON-schema providers reject open-ended dicts."""
    mrt: Optional[int] = None
    hawker: Optional[int] = None
    supermarket: Optional[int] = None
    school: Optional[int] = None
    bus_stop: Optional[int] = None
    park: Optional[int] = None


class LLMExtractedConstraints(ExtractedConstraints):
    """The schema given to the model; converted to ExtractedConstraints by the supervisor."""
    amenity_radii_meters: AmenityRadii = Field(default_factory=AmenityRadii,
                                               description="Distance limits in metres the user stated; null when not stated")

    def to_constraints(self) -> "ExtractedConstraints":
        data = self.model_dump(exclude={"amenity_radii_meters"})
        data["amenity_radii_meters"] = {k: v for k, v in self.amenity_radii_meters.model_dump().items() if v}
        return ExtractedConstraints(**data)


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
    block_keys: Optional[List[str]] = Field(
        None, description="Restrict to these 'BLOCK STREET' keys (e.g. '123 BISHAN ST 13'); used to price blocks from spatial results")

    @field_validator("towns", "flat_types", "block_keys")
    @classmethod
    def normalise_filters(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return None
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
    model_config = ConfigDict(extra="forbid")
    towns: List[str]
    flat_type: str
    lookback_years: int = Field(3, ge=1, le=30, strict=True)

    @field_validator("towns")
    @classmethod
    def normalise_towns(cls, values: List[str]) -> List[str]:
        result = [" ".join(value.upper().split()) for value in values]
        if any(not value for value in result):
            raise ValueError("Town filters must not be blank")
        return list(dict.fromkeys(result))

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: str) -> str:
        result = " ".join(value.upper().split())
        if not result:
            raise ValueError("Flat type must not be blank")
        return result


class TownQuarterlyPoint(BaseModel):
    town: str
    quarter: str = Field(..., examples=["2025Q4"])
    median_price: float
    median_price_per_sqm: float
    transaction_count: int
    is_complete: Optional[bool] = Field(
        None, description="Whether source coverage spans the full calendar quarter"
    )


class TownTrendSummary(BaseModel):
    town: str
    latest_median_price: float
    yoy_appreciation_pct: Optional[float] = None
    median_price_per_sqm: float
    latest_quarter: Optional[str] = None
    transaction_count: Optional[int] = None
    yoy_comparison_quarter: Optional[str] = None
    yoy_unavailable_reason: Optional[
        Literal["missing_prior_year_quarter", "incomplete_prior_year_quarter"]
    ] = None
    period_start_quarter: Optional[str] = Field(None, description="Earliest complete quarter in the display window")
    period_change_pct: Optional[float] = Field(None, description="Latest vs earliest complete quarter median over the window")
    period_cagr_pct: Optional[float] = Field(None, description="Compound annual growth rate over that span")


class TownPriceTimeseriesMetadata(BaseModel):
    snapshot_id: str
    dataset_id: str
    source_url: str
    flat_type: str
    lookback_years: int
    requested_start_month: str
    window_start_month: str
    window_end_month: str
    first_source_month: str
    latest_source_month: str
    history_truncated: bool
    towns_without_data: List[str] = Field(default_factory=list)
    towns_without_complete_quarters: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ComputeTownPriceTimeseriesOutput(BaseModel):
    series: List[TownQuarterlyPoint]
    summary: List[TownTrendSummary]
    metadata: Optional[TownPriceTimeseriesMetadata] = None


class QueryComparableResaleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    flat_type: str
    radius_meters: int = Field(500, ge=1, le=10000, strict=True)
    lookback_months: int = Field(12, ge=1, le=120, strict=True)
    end_month: Optional[str] = Field(None, description="YYYY-MM; defaults to the snapshot's last completed analysis month")
    min_floor_area_sqm: Optional[float] = Field(None, gt=0)
    max_floor_area_sqm: Optional[float] = Field(None, gt=0)
    min_lease_years: Optional[float] = Field(None, ge=0, le=99)
    max_lease_years: Optional[float] = Field(None, ge=0, le=99)
    min_storey: Optional[int] = Field(None, ge=1, le=99, strict=True)
    max_storey: Optional[int] = Field(None, ge=1, le=99, strict=True)
    min_sample_size: int = Field(5, ge=1, le=1000, strict=True)
    sample_limit: int = Field(20, ge=1, le=100, strict=True)

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: str) -> str:
        result = " ".join(value.upper().split())
        if not result:
            raise ValueError("Flat type must not be blank")
        return result

    @field_validator("end_month")
    @classmethod
    def validate_month(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            from datetime import date

            parsed = date.fromisoformat(value + "-01")
            if parsed.strftime("%Y-%m") != value:
                raise ValueError("Use YYYY-MM for end_month")
        return value

    @model_validator(mode="after")
    def ordered_ranges(self):
        for lower, upper in (("min_floor_area_sqm", "max_floor_area_sqm"),
                             ("min_lease_years", "max_lease_years"),
                             ("min_storey", "max_storey")):
            low, high = getattr(self, lower), getattr(self, upper)
            if low is not None and high is not None and low > high:
                raise ValueError(f"{lower} must not exceed {upper}")
        return self


class ComparablePriceDistribution(BaseModel):
    minimum: float
    lower_quartile: float
    median: float
    upper_quartile: float
    maximum: float


class ComparableResaleSummary(BaseModel):
    transaction_count: int
    block_count: int
    price: ComparablePriceDistribution
    price_per_sqm: ComparablePriceDistribution
    median_floor_area_sqm: float
    median_remaining_lease_years: float
    first_transaction_month: str
    last_transaction_month: str


class ComparableResaleTransaction(BlockRef):
    source_row_number: int
    month: str
    flat_type: str
    floor_area_sqm: float
    storey_range: str
    remaining_lease_years: float
    resale_price: float
    price_per_sqm: float
    distance_meters: float


class ComparableResaleMetadata(BaseModel):
    snapshot_id: str
    dataset_id: str
    source_url: str
    retrieved_at: str
    first_source_month: str
    latest_source_month: str
    window_start_month: str
    window_end_month: str
    lease_reference_month: str
    history_truncated: bool
    transactions_without_coordinates: int
    returned_transactions: int
    sample_truncated: bool
    notes: List[str] = Field(default_factory=list)


class QueryComparableResaleOutput(BaseModel):
    status: Literal["ok", "insufficient_sample", "no_matches", "no_coverage"]
    criteria: QueryComparableResaleInput
    summary: Optional[ComparableResaleSummary] = None
    transactions: List[ComparableResaleTransaction]
    metadata: ComparableResaleMetadata
    sql: str


class CompareBtoLaunchVsResaleInput(BaseModel):
    bto_project_name: str
    radius_meters: int = Field(500, description="Kept for compatibility; launch sites have no published coordinates")
    flat_type: str = Field("4 ROOM", description="e.g. '4 ROOM', '3 ROOM', '2-ROOM FLEXI (TYPE 2)'")


class CompareBtoLaunchVsResaleOutput(BaseModel):
    """Real output adds status/provenance fields; the legacy mock fills only the original five."""
    status: Literal["ok", "not_found", "flat_type_not_offered"] = "ok"
    bto_project_name: str
    bto_indicative_price_range: Optional[Dict[str, float]] = Field(None, examples=[{"min": 420000, "max": 560000}])
    resale_median_nearby: Optional[float] = None
    resale_sample_size: Optional[int] = None
    price_gap_pct: Optional[float] = None
    flat_type: Optional[str] = None
    town: Optional[str] = None
    launch_month: Optional[str] = None
    classification: Optional[Literal["Standard", "Plus", "Prime"]] = None
    floor_area_sqm: Optional[float] = None
    units: Optional[int] = None
    waiting_time_months: Optional[str] = None
    hdb_resale_comparable_range: Optional[Dict[str, float]] = Field(None, description="HDB's own 'resale comparables nearby' from the launch annex")
    resale_basis: Optional[str] = None
    available_projects: List[str] = Field(default_factory=list)
    available_flat_types: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


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
    bracket: Literal["1km", "2km", "1-2km"] = "1km"


class FindSchoolsNearAddressInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    address: str = Field(..., min_length=1, max_length=300,
                         description="User-supplied block and street, block and town, or six-digit postal code. No inferred addresses.")
    radius_meters: float = Field(2000, gt=0, le=10000,
                                description="Straight-line search radius; default 2000 m, disclosed to the user.")


class NearbySchoolRecord(BaseModel):
    name: str
    address: str
    lat: float
    lng: float
    distance_meters: float


class FindSchoolsNearAddressOutput(BaseModel):
    status: Literal["ok", "needs_information", "unavailable"]
    message: str
    radius_meters: float
    location: Optional[BlockRef] = None
    address_candidates: List[BlockRef] = Field(default_factory=list)
    schools: List[NearbySchoolRecord] = Field(default_factory=list)
    distance_method: Literal["straight_line_svy21"] = "straight_line_svy21"
    source: str = "Local spatial index; primary schools only; freshness not verified"


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
class FinancialInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FinancialResult(BaseModel):
    status: Literal["ok", "needs_information", "unsupported", "ineligible"] = "ok"
    missing_inputs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    rule_version: Optional[str] = None
    verified_on: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CalculateCpfHousingGrantsInput(FinancialInput):
    flat_category: Literal["bto", "resale"] = "resale"
    citizenship_pair: Optional[Literal["SC_SC", "SC_PR", "SC_Single"]] = None
    first_timer_status: Optional[bool] = None
    monthly_income: Optional[float] = Field(None, ge=0, description="HFE-assessed average monthly income")
    near_parents_4km: bool = False
    flat_type: Optional[Literal["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]] = None
    youngest_buyer_age: Optional[int] = Field(None, ge=21, le=100)
    youngest_core_member_age: Optional[int] = Field(None, ge=0, le=100, description="Youngest core buyer or core occupier for EHG lease coverage")
    remaining_lease_years: Optional[float] = Field(None, gt=0, le=99)
    eligibility_confirmed: Optional[bool] = Field(None, description="Confirm standard first-timer scheme, unclassified/Standard resale flat, citizenship and property-ownership prerequisites; see financial documentation")
    continuous_employment_12_months: Optional[bool] = None
    proximity: Optional[Literal["none", "near", "with"]] = None
    proximity_eligibility_confirmed: Optional[bool] = Field(None, description="Confirm eligible relatives/residence, no previous PHG and all PHG conditions")


class CalculateCpfHousingGrantsOutput(FinancialResult):
    enhanced_housing_grant: Optional[float] = None
    family_grant: Optional[float] = None
    proximity_housing_grant: Optional[float] = None
    total_grants: Optional[float] = None
    eligibility_notes: List[str] = Field(default_factory=list)


class CalculateMaximumAffordabilityInput(FinancialInput):
    monthly_income: Optional[float] = Field(None, gt=0)
    cpf_oa_balance: Optional[float] = Field(None, ge=0, description="CPF OA available after retention and CPF eligibility limits")
    cash_savings: Optional[float] = Field(None, ge=0)
    loan_type: Literal["hdb", "bank"] = "hdb"
    interest_rate: Optional[float] = Field(None, ge=0, le=100, description="Annual percent; bank input must include highest applicable thereafter rate")
    tenure_years: int = Field(25, ge=1, le=30, strict=True)
    average_applicant_age: Optional[float] = Field(None, ge=21, lt=65, description="HDB average age; for bank loans use the lender-assessed income-weighted age")
    oldest_buyer_age: Optional[int] = Field(None, ge=21, le=100)
    youngest_buyer_age: Optional[int] = Field(None, ge=21, le=100)
    remaining_lease_years: Optional[float] = Field(None, gt=20, le=99)
    monthly_debt_obligations: Optional[float] = Field(None, ge=0)
    total_grants: Optional[float] = Field(None, ge=0, description="Confirmed grant estimate; supply explicit zero if none")
    loan_eligibility_confirmed: Optional[bool] = Field(None, description="Confirm requested loan eligibility and no outstanding housing loans; approval still required")


class CalculateMaximumAffordabilityOutput(FinancialResult):
    max_loan_amount: Optional[float] = None
    borrowing_ceiling: Optional[float] = None
    max_purchase_price: Optional[float] = None
    monthly_instalment: Optional[float] = None
    msr_cap_monthly: Optional[float] = None
    ltv_ratio: Optional[float] = None
    binding_constraint: Optional[Literal["MSR", "LTV", "TDSR", "DOWNPAYMENT"]] = None
    assessment_interest_rate: Optional[float] = None
    interest_rate: Optional[float] = None
    tenure_months: Optional[int] = None
    grants_included: bool = False
    purchase_costs_included: bool = False


class CalculateBsdAndUpfrontOutlayInput(FinancialInput):
    purchase_price: float = Field(..., gt=0)
    loan_type: Literal["hdb", "bank"] = "hdb"
    total_grants: Optional[float] = Field(None, ge=0)
    market_valuation: Optional[float] = Field(None, gt=0)
    loan_amount: Optional[float] = Field(None, ge=0, description="Actual approved or modelled loan, not borrowing ceiling")
    cpf_oa_balance: Optional[float] = Field(None, ge=0, description="CPF OA confirmed usable for purchase and eligible fees")
    cash_savings: Optional[float] = Field(None, ge=0)
    legal_and_valuation_fees: Optional[float] = Field(None, ge=0, description="Quoted costs; conservatively cash-funded")
    additional_buyer_stamp_duty: Optional[float] = Field(None, ge=0, description="Separately assessed ABSD; explicit zero if none")
    ltv_ratio: Optional[Literal[0.75, 0.55]] = None


class CalculateBsdAndUpfrontOutlayOutput(FinancialResult):
    buyer_stamp_duty: Optional[float] = None
    min_cash_downpayment: Optional[float] = None
    cpf_oa_usable: Optional[float] = None
    legal_and_valuation_fees: Optional[float] = None
    total_upfront: Optional[float] = None
    cash_required: Optional[float] = None
    cash_shortfall: Optional[float] = None
    cash_over_valuation: Optional[float] = None
    grants_used: Optional[float] = None
    unused_grants: Optional[float] = None
    additional_buyer_stamp_duty: Optional[float] = None


class FinancialScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grants: Optional[CalculateCpfHousingGrantsInput] = None
    affordability: Optional[CalculateMaximumAffordabilityInput] = None
    upfront: Optional[CalculateBsdAndUpfrontOutlayInput] = None


ExtractedConstraints.model_rebuild()
LLMExtractedConstraints.model_rebuild()


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
    # "model": the model chose this call in its tool loop; "code": a deterministic plan, the supervisor (relaxation,
    # map sync) or a guard made it. Tool Selection F1 can then be split by who chose the call.
    source: Literal["model", "code"] = "code"


# ---------------------------------------------------------------------------
# Tool inventory: the 12 core tools. The add-on packs (agents/addons) register the research tools
# (agents/research_schemas.py) and the address-centred school lookup at the end of this module.
# ---------------------------------------------------------------------------

TOOL_OWNER: Dict[str, str] = {
    # Agent 2 — Member 1
    "query_resale_blocks_by_criteria": "market_sql",
    "compute_town_price_timeseries": "market_sql",
    "query_comparable_resale_transactions": "market_sql",
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
CORE_TOOLS = tuple(TOOL_OWNER)
# The 12 core tools. Add-on packs (agents/addons: research and school lookup) register theirs at the end of this
# module; which runs are offered them is addons.offer_in. relax_unsatisfied_constraints is a deterministic supervisor
# node, not an LLM-callable tool, so it is not counted.
assert len(CORE_TOOLS) == 12

TOOL_INPUT_MODEL: Dict[str, type[BaseModel]] = {
    "query_resale_blocks_by_criteria": QueryResaleBlocksInput,
    "compute_town_price_timeseries": ComputeTownPriceTimeseriesInput,
    "query_comparable_resale_transactions": QueryComparableResaleInput,
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


# Add-on packs: their tools' owners and input models (agents/addons). Kept last: the packs name models defined above.
from agents import addons as _addons  # noqa: E402

_addons.register_tools(TOOL_OWNER, TOOL_INPUT_MODEL)
