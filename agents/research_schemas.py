"""
Contracts for the research tools (agents/research_schemas.py). Owned by Member 1,
with the two location tools shared with Member 2.

These tools answer questions the original block filter could not: where a place
is, what is near it, what a block has sold for, how the market stands now, which
town is cheapest for a flat type, how specific options compare, whether an asking
price is in line with nearby sales, and how a BTO launch compares with resale.

Every tool that needs a location takes the user's address text and resolves it
itself, so a model never has to supply coordinates. agents/schemas.py re-exports
these models and registers them in TOOL_OWNER / TOOL_INPUT_MODEL.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AmenityCategory = Literal["mrt", "hawker", "supermarket", "school"]


def _normalise(values: List[str]) -> List[str]:
    result = [" ".join(v.upper().split()) for v in values]
    if any(not v for v in result):
        raise ValueError("Filter values must not be blank")
    return list(dict.fromkeys(result))


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ResearchMetadata(BaseModel):
    """Provenance shared by the market research tools."""
    snapshot_id: str
    source_url: str
    window_start_month: Optional[str] = None
    window_end_month: Optional[str] = None
    latest_source_month: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Location resolution and nearby amenities (shared with Member 2)
# ---------------------------------------------------------------------------
class LocationCandidate(BaseModel):
    kind: Literal["hdb_block", "mrt", "hawker", "supermarket", "school", "onemap"]
    name: str
    block: Optional[str] = None
    street: Optional[str] = None
    town: Optional[str] = None
    postal: Optional[str] = None
    lat: float
    lng: float
    source: str
    score: float = Field(..., ge=0, le=1, description="1.0 is an exact identifier match")


class ResolveLocationInput(StrictInput):
    query: str = Field(..., min_length=1, max_length=200,
                       description="User-supplied place: 'Blk 288B Jurong East St 21', a postal code, 'Bishan MRT', a school or hawker centre")
    limit: int = Field(5, ge=1, le=10, strict=True)


class ResolveLocationOutput(BaseModel):
    status: Literal["ok", "ambiguous", "not_found", "unavailable"]
    query: str
    candidates: List[LocationCandidate] = Field(default_factory=list)
    message: str


class GetNearbyAmenitiesInput(StrictInput):
    address: Optional[str] = Field(None, min_length=1, max_length=200,
                                   description="User-supplied address or place name; preferred over coordinates")
    latitude: Optional[float] = Field(None, ge=1.1, le=1.5, description="Only coordinates from resolve_location or the user's map pin")
    longitude: Optional[float] = Field(None, ge=103.5, le=104.1)
    radius_meters: int = Field(1000, ge=50, le=5000, strict=True)
    categories: List[AmenityCategory] = Field(default_factory=list, description="Empty means all categories")
    max_per_category: int = Field(5, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def location_given(self):
        has_coordinates = self.latitude is not None and self.longitude is not None
        if not self.address and not has_coordinates:
            raise ValueError("Provide an address, or both latitude and longitude")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Provide both latitude and longitude")
        return self


class NearbyAmenity(BaseModel):
    name: str
    category: AmenityCategory
    lat: float
    lng: float
    distance_meters: float
    walking_minutes: int


class GetNearbyAmenitiesOutput(BaseModel):
    status: Literal["ok", "ambiguous", "not_found", "unavailable"]
    message: str
    location: Optional[LocationCandidate] = None
    candidates: List[LocationCandidate] = Field(default_factory=list)
    radius_meters: int
    counts: Dict[str, int] = Field(default_factory=dict)
    nearest: Dict[str, Optional[NearbyAmenity]] = Field(default_factory=dict,
                                                        description="Nearest of each category even when outside the radius")
    amenities: List[NearbyAmenity] = Field(default_factory=list)
    distance_method: Literal["straight_line_svy21"] = "straight_line_svy21"


# ---------------------------------------------------------------------------
# Market research (Member 1)
# ---------------------------------------------------------------------------
class GetBlockTransactionHistoryInput(StrictInput):
    address: Optional[str] = Field(None, min_length=1, max_length=200, description="User-supplied block and street or postal code")
    area: Optional["AreaFilter"] = Field(None, description="Instead of address: every resale block within the circle")
    block_keys: Optional[List[str]] = Field(None, min_length=1, max_length=200,
                                            description="Instead of address: exact 'BLOCK STREET' keys")
    flat_type: Optional[str] = Field(None, description="e.g. '4 ROOM'; omit for every flat type")
    lookback_months: int = Field(60, ge=1, le=240, strict=True)

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: Optional[str]) -> Optional[str]:
        return _normalise([value])[0] if value is not None else None

    @field_validator("block_keys")
    @classmethod
    def normalise_keys(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _normalise(value) if value is not None else None

    @model_validator(mode="after")
    def one_scope(self) -> "GetBlockTransactionHistoryInput":
        if sum(v is not None for v in (self.address, self.area, self.block_keys)) != 1:
            raise ValueError("Provide exactly one of address, area or block_keys")
        return self


class BlockTransaction(BaseModel):
    month: str
    block: Optional[str] = None
    street: Optional[str] = None
    distance_meters: Optional[float] = Field(None, description="From the area centre, for area queries")
    flat_type: str
    storey_range: str
    floor_area_sqm: float
    flat_model: str
    remaining_lease_years: float
    resale_price: float
    price_per_sqm: float


class BlockYearSummary(BaseModel):
    year: int
    flat_type: str
    transaction_count: int
    median_price: float
    median_price_per_sqm: float


class GetBlockTransactionHistoryOutput(BaseModel):
    status: Literal["ok", "ambiguous", "not_found", "no_transactions"]
    message: str
    scope: Literal["block", "area", "block_keys"] = "block"
    block_count: int = 1
    location: Optional[LocationCandidate] = Field(None, description="The block, or the area centre")
    candidates: List[LocationCandidate] = Field(default_factory=list)
    transaction_count: int = 0
    transactions: List[BlockTransaction] = Field(default_factory=list, description="Most recent first, at most 50")
    yearly: List[BlockYearSummary] = Field(default_factory=list)
    metadata: Optional[ResearchMetadata] = None


class GetMarketSnapshotInput(StrictInput):
    towns: List[str] = Field(default_factory=list, description="Empty means island-wide only")
    flat_types: List[str] = Field(default_factory=list, description="Empty means 3 ROOM, 4 ROOM, 5 ROOM and EXECUTIVE")

    @field_validator("towns", "flat_types")
    @classmethod
    def normalise_filters(cls, values: List[str]) -> List[str]:
        return _normalise(values)


class MarketSnapshotRow(BaseModel):
    town: str = Field(..., description="'ALL TOWNS' for the island-wide row")
    flat_type: str
    quarter: str
    median_price: float
    median_price_per_sqm: float
    transaction_count: int
    qoq_change_pct: Optional[float] = None
    yoy_change_pct: Optional[float] = None
    change_3y_pct: Optional[float] = None
    change_5y_pct: Optional[float] = None


class GetMarketSnapshotOutput(BaseModel):
    status: Literal["ok", "no_data"]
    latest_complete_quarter: Optional[str] = None
    rows: List[MarketSnapshotRow] = Field(default_factory=list)
    metadata: ResearchMetadata


class RankTownsByValueInput(StrictInput):
    flat_type: str
    max_budget: Optional[float] = Field(None, gt=0)
    min_lease_years: Optional[int] = Field(None, ge=0, le=99)
    lookback_months: int = Field(12, ge=3, le=36, strict=True)
    min_transactions: int = Field(10, ge=1, le=500, strict=True)
    rank_by: Literal["price_per_sqm", "median_price", "share_within_budget"] = "price_per_sqm"

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: str) -> str:
        return _normalise([value])[0]


class TownValueRow(BaseModel):
    rank: int
    town: str
    median_price: float
    median_price_per_sqm: float
    median_remaining_lease_years: float
    transaction_count: int
    share_within_budget_pct: Optional[float] = None


class RankTownsByValueOutput(BaseModel):
    status: Literal["ok", "no_data"]
    flat_type: str
    ranking_basis: str
    rows: List[TownValueRow] = Field(default_factory=list)
    excluded_low_sample_towns: List[str] = Field(default_factory=list)
    metadata: ResearchMetadata


class CompareResaleOptionsInput(StrictInput):
    addresses: List[str] = Field(..., min_length=2, max_length=4, description="User-supplied blocks/addresses to compare")
    flat_type: str
    lookback_months: int = Field(24, ge=3, le=120, strict=True)

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: str) -> str:
        return _normalise([value])[0]


class ResaleOption(BaseModel):
    address: str
    status: Literal["ok", "ambiguous", "not_found", "no_transactions"]
    location: Optional[LocationCandidate] = None
    candidates: List[LocationCandidate] = Field(default_factory=list)
    transaction_count: int = 0
    median_price: Optional[float] = None
    median_price_per_sqm: Optional[float] = None
    median_floor_area_sqm: Optional[float] = None
    median_remaining_lease_years: Optional[float] = None
    last_transaction_month: Optional[str] = None
    last_price: Optional[float] = None
    nearest: Dict[str, Optional[NearbyAmenity]] = Field(default_factory=dict)


class CompareResaleOptionsOutput(BaseModel):
    status: Literal["ok", "partial", "needs_information"]
    flat_type: str
    options: List[ResaleOption]
    metadata: ResearchMetadata


class AssessResalePriceInput(StrictInput):
    address: str = Field(..., min_length=1, max_length=200, description="User-supplied block and street or postal code")
    flat_type: str
    asking_price: Optional[float] = Field(None, gt=0)
    radius_meters: int = Field(500, ge=100, le=3000, strict=True)
    lookback_months: int = Field(12, ge=3, le=60, strict=True)
    floor_area_sqm: Optional[float] = Field(None, gt=0, description="Only if the user states it; restricts comparables to ±10 sqm")

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: str) -> str:
        return _normalise([value])[0]


class PriceDistribution(BaseModel):
    transaction_count: int
    minimum: float
    lower_quartile: float
    median: float
    upper_quartile: float
    maximum: float
    median_price_per_sqm: float


class AssessResalePriceOutput(BaseModel):
    status: Literal["ok", "ambiguous", "not_found", "insufficient_sample", "no_matches"]
    message: str
    location: Optional[LocationCandidate] = None
    candidates: List[LocationCandidate] = Field(default_factory=list)
    flat_type: str
    same_block: Optional[PriceDistribution] = None
    nearby: Optional[PriceDistribution] = None
    asking_price: Optional[float] = None
    asking_percentile: Optional[float] = Field(None, description="Share of nearby sales priced below the asking price, 0-100")
    asking_vs_nearby_median_pct: Optional[float] = None
    position: Optional[Literal["below_lower_quartile", "within_middle_50", "above_upper_quartile"]] = None
    remaining_lease_years: Optional[float] = Field(None, description="The block's remaining lease now, from its latest sale (any flat type)")
    nearby_median_lease_years: Optional[float] = Field(None, description="Median remaining lease of the nearby sales used, at the end of the window")
    metadata: Optional[ResearchMetadata] = None


# ---------------------------------------------------------------------------
# Area research: find the resale blocks near a place, then run set statistics
# ---------------------------------------------------------------------------
class AreaFilter(StrictInput):
    """A circle around a place. The tool resolves the address itself."""
    address: Optional[str] = Field(None, min_length=1, max_length=200,
                                   description="User-supplied block, postal code or named place; preferred over coordinates")
    latitude: Optional[float] = Field(None, ge=1.1, le=1.5, description="Only from resolve_location, a tool result or the map pin")
    longitude: Optional[float] = Field(None, ge=103.5, le=104.1)
    radius_meters: int = Field(500, ge=50, le=3000, strict=True)

    @model_validator(mode="after")
    def place_given(self) -> "AreaFilter":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Provide both latitude and longitude")
        if not self.address and self.latitude is None:
            raise ValueError("Provide an address, or both latitude and longitude")
        return self


class BlockSetInput(StrictInput):
    """Statistics over a set of blocks: an area, or explicit block keys (e.g. from find_resale_blocks_near)."""
    area: Optional[AreaFilter] = Field(None, description="Every resale block within the circle")
    block_keys: Optional[List[str]] = Field(None, min_length=1, max_length=200,
                                            description="Exact 'BLOCK STREET' keys, e.g. '277 TOH GUAN RD'")

    @field_validator("block_keys")
    @classmethod
    def normalise_keys(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _normalise(value) if value is not None else None

    @model_validator(mode="after")
    def one_scope(self) -> "BlockSetInput":
        if (self.area is None) == (self.block_keys is None):
            raise ValueError("Provide exactly one of area or block_keys")
        return self


class FindResaleBlocksNearInput(AreaFilter):
    flat_type: Optional[str] = Field(None, description="e.g. '4 ROOM'; omit for every flat type")
    lookback_months: int = Field(12, ge=1, le=120, strict=True)
    limit: int = Field(60, ge=1, le=200, strict=True)

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: Optional[str]) -> Optional[str]:
        return _normalise([value])[0] if value is not None else None


class NearbyResaleBlock(BaseModel):
    block_key: str
    block: str
    street: str
    town: str
    lat: float
    lng: float
    distance_meters: Optional[float] = None
    transaction_count: int = Field(..., description="Sales in the window (and flat type, if given)")
    median_price: Optional[float] = None
    median_price_per_sqm: Optional[float] = None
    median_remaining_lease_years: Optional[float] = Field(None, description="At the end of the window")
    last_sale_month: Optional[str] = None
    is_target: bool = Field(False, description="The block the user named or pinned")


class FindResaleBlocksNearOutput(BaseModel):
    status: Literal["ok", "ambiguous", "not_found", "no_blocks"]
    message: str
    location: Optional[LocationCandidate] = None
    candidates: List[LocationCandidate] = Field(default_factory=list)
    radius_meters: int
    flat_type: Optional[str] = None
    block_count: int = Field(0, description="Resale blocks within the radius, with or without sales in the window")
    blocks_with_sales: int = 0
    transaction_count: int = 0
    blocks: List[NearbyResaleBlock] = Field(default_factory=list, description="Nearest first; blocks without sales last")
    metadata: Optional[ResearchMetadata] = None


class GetAreaResaleStatisticsInput(BlockSetInput):
    flat_type: Optional[str] = Field(None, description="e.g. '4 ROOM'; omit for every flat type")
    lookback_months: int = Field(12, ge=1, le=120, strict=True)

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: Optional[str]) -> Optional[str]:
        return _normalise([value])[0] if value is not None else None


class FlatTypeStatistics(BaseModel):
    flat_type: str
    block_count: int = Field(..., description="Blocks with at least one sale of this type")
    distribution: PriceDistribution
    median_floor_area_sqm: float
    median_remaining_lease_years: float


class GetAreaResaleStatisticsOutput(BaseModel):
    status: Literal["ok", "ambiguous", "not_found", "no_transactions"]
    message: str
    location: Optional[LocationCandidate] = None
    candidates: List[LocationCandidate] = Field(default_factory=list)
    scope: Literal["area", "block_keys"]
    radius_meters: Optional[int] = None
    block_count: int = Field(0, description="Blocks in the set")
    unknown_block_keys: List[str] = Field(default_factory=list)
    by_flat_type: List[FlatTypeStatistics] = Field(default_factory=list, description="Pooled over every sale, not a median of medians")
    target_block_key: Optional[str] = None
    target_by_flat_type: List[FlatTypeStatistics] = Field(default_factory=list, description="The named block alone, for comparison")
    metadata: Optional[ResearchMetadata] = None


class GetAreaPriceTrendInput(BlockSetInput):
    flat_type: str
    lookback_years: int = Field(3, ge=1, le=20, strict=True)

    @field_validator("flat_type")
    @classmethod
    def normalise_flat_type(cls, value: str) -> str:
        return _normalise([value])[0]


class AreaTrendPoint(BaseModel):
    quarter: str
    transaction_count: int
    median_price: float
    median_price_per_sqm: float
    is_complete: bool


class GetAreaPriceTrendOutput(BaseModel):
    status: Literal["ok", "ambiguous", "not_found", "no_transactions"]
    message: str
    location: Optional[LocationCandidate] = None
    candidates: List[LocationCandidate] = Field(default_factory=list)
    scope: Literal["area", "block_keys"]
    radius_meters: Optional[int] = None
    block_count: int = 0
    flat_type: str
    points: List[AreaTrendPoint] = Field(default_factory=list)
    period_start_quarter: Optional[str] = Field(None, description="First complete quarter with enough sales")
    period_end_quarter: Optional[str] = None
    period_change_pct: Optional[float] = Field(None, description="Median price change between those quarters")
    period_price_per_sqm_change_pct: Optional[float] = Field(None, description="Less sensitive to flat-size mix")
    period_cagr_pct: Optional[float] = None
    min_quarter_sales: int = Field(..., description="Quarters with fewer sales are shown but not used as endpoints")
    metadata: Optional[ResearchMetadata] = None


# ---------------------------------------------------------------------------
# Grouped resale statistics: island-wide / town / station / distance bands (mcp_server/aggregate_resale_tools.py)
# ---------------------------------------------------------------------------
# The allowed values are listed here, in the tool's descriptions and in describe_resale_data. The inputs stay plain
# strings on purpose: a wrong value comes back from the tool as a structured error that lists the allowed values, so a
# model can correct itself, instead of a validation exception that ends the call.
STATS_DIMENSIONS = ("distance_band", "town", "station", "street", "flat_type", "year", "storey_band", "lease_band")
STATS_NEAR = ("mrt", "mrt_lrt", "lrt", "hawker", "supermarket", "school")
STATS_METRICS = ("count", "median_price", "p25", "p75", "median_psm", "change_1y_pct", "change_period_pct")
STATS_SORT = ("median_price", "count", "median_psm", "change")


class QueryResaleStatisticsInput(StrictInput):
    """Grouped resale statistics over every matching past sale (not listings)."""
    group_by: List[str] = Field(default_factory=lambda: ["distance_band"], min_length=1, max_length=2,
                                description="1-2 of: " + ", ".join(STATS_DIMENSIONS))
    towns: List[str] = Field(default_factory=list, description="Only these towns, e.g. ['BISHAN']")
    flat_types: List[str] = Field(default_factory=list, description="Only these flat types, e.g. ['4 ROOM']")
    station: Optional[str] = Field(None, description="Only blocks whose nearest station is this one, e.g. 'Bishan MRT'")
    near: Optional[str] = Field(None, description="Distance to measure: one of " + ", ".join(STATS_NEAR)
                                + ". mrt leaves LRT out; mrt_lrt counts both")
    max_distance_m: Optional[int] = Field(None, ge=50, le=3000, strict=True,
                                          description="Keep only blocks within this many metres (straight line)")
    bands: Optional[List[int]] = Field(None, min_length=1, max_length=6,
                                       description="Upper edges in metres for distance_band, e.g. [200, 500, 1000]")
    lookback_years: Optional[int] = Field(None, ge=1, le=10, strict=True, description="Default 1 year")
    start_month: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}$", description="YYYY-MM; overrides lookback_years")
    end_month: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}$", description="YYYY-MM; default the last complete month")
    metrics: Optional[List[str]] = Field(None, description="Subset of: " + ", ".join(STATS_METRICS) + "; default all")
    sort_by: Optional[str] = Field(None, description="One of " + ", ".join(STATS_SORT)
                                   + "; default: by band/year order, else median_price")
    order: Literal["desc", "asc"] = "desc"
    top_n: int = Field(10, ge=1, le=50, strict=True)
    include_trend: bool = Field(False, description="Add each group's median per calendar year")
    min_sales: int = Field(10, ge=1, le=1000, strict=True, description="Groups with fewer sales are left out")


class DescribeResaleDataInput(StrictInput):
    include_stations: bool = Field(True, description="List the MRT and LRT station names")


class StatPeriod(BaseModel):
    start_month: str
    end_month: str
    months: int


class StatTrendPoint(BaseModel):
    year: int
    count: int
    median_price: float


class StatGroup(BaseModel):
    key: Dict[str, str]
    label: str
    count: int
    median_price: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    median_psm: Optional[float] = None
    change_1y_pct: Optional[float] = Field(None, description="Median of the last 12 months vs the 12 months before")
    change_period_pct: Optional[float] = Field(None, description="Last 12 months vs first 12 months of the period")
    trend: Optional[List[StatTrendPoint]] = None
    lat: Optional[float] = Field(None, description="Station or town centre, for the map")
    lng: Optional[float] = None
    drill: Optional[Dict[str, Any]] = Field(None, description="The next-level call: {tool, params}")


class StatBaseline(BaseModel):
    label: str
    count: int
    median_price: Optional[float] = None
    median_psm: Optional[float] = None


class StatMapBlock(BaseModel):
    block_key: str
    town: str
    lat: float
    lng: float
    group: str
    count: int
    median_price: float


class QueryResaleStatisticsOutput(BaseModel):
    status: Literal["ok", "no_data", "invalid_request"]
    message: str
    allowed: Optional[Dict[str, List[str]]] = Field(None, description="Valid values, when the request was invalid")
    group_by: List[str] = Field(default_factory=list)
    near: Optional[str] = None
    max_distance_m: Optional[int] = None
    bands: List[int] = Field(default_factory=list)
    station: Optional[str] = None
    towns: List[str] = Field(default_factory=list)
    flat_types: List[str] = Field(default_factory=list)
    period: Optional[StatPeriod] = None
    groups: List[StatGroup] = Field(default_factory=list)
    baseline: Optional[StatBaseline] = None
    total_groups: int = 0
    truncated: bool = False
    excluded_small_groups: int = 0
    min_sales: int = 10
    unlocated_sales_excluded: int = 0
    distance_definition: Optional[str] = None
    sort_by: Optional[str] = None
    order: str = "desc"
    include_trend: bool = False
    map_blocks: List[StatMapBlock] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    snapshot_id: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class DescribeResaleDataOutput(BaseModel):
    status: Literal["ok"] = "ok"
    snapshot_id: str
    period_covered: Dict[str, str]
    dimensions: Dict[str, str]
    towns: List[str]
    flat_types: List[str]
    near_options: Dict[str, str]
    mrt_stations: List[str] = Field(default_factory=list)
    lrt_stations: List[str] = Field(default_factory=list)
    metrics: Dict[str, str]
    definitions: List[str]
    example_calls: List[Dict[str, object]] = Field(default_factory=list)


GetBlockTransactionHistoryInput.model_rebuild()
