"""Research tools: location resolution, nearby amenities, block history, market snapshot,
town ranking, option comparison, asking-price assessment and BTO comparison.

A synthetic DuckDB snapshot and spatial index keep every case deterministic and offline.
"""

from __future__ import annotations

import csv
import pickle
from datetime import datetime, timezone

import pytest

from agents.research_schemas import (
    AssessResalePriceOutput,
    CompareResaleOptionsOutput,
    GetBlockTransactionHistoryOutput,
    GetMarketSnapshotOutput,
    GetNearbyAmenitiesInput,
    RankTownsByValueOutput,
    ResolveLocationOutput,
)
from agents.schemas import CompareBtoLaunchVsResaleOutput
from data.build_market_db import SOURCE_COLUMNS, build_database
from mcp_server import location_tools
from mcp_server.bto_tools import compare_bto_launch_vs_resale, load_launches
from mcp_server.location_tools import display_name, get_nearby_amenities, resolve_location
from mcp_server.market_research_tools import (
    assess_resale_price,
    compare_resale_options,
    get_block_transaction_history,
    get_market_snapshot,
    rank_towns_by_value,
)
from mcp_server.market_sql_tools import compute_town_price_timeseries, query_resale_blocks_by_criteria

BLOCKS = [
    {"block": "1", "street": "BISHAN ST 1", "town": "BISHAN", "postal": "570001", "lat": 1.3500, "lng": 103.8500},
    {"block": "2", "street": "BISHAN ST 1", "town": "BISHAN", "postal": "570002", "lat": 1.3510, "lng": 103.8500},
    {"block": "2", "street": "BISHAN ST 2", "town": "BISHAN", "postal": "570202", "lat": 1.3520, "lng": 103.8510},
    {"block": "5", "street": "TAMPINES ST 1", "town": "TAMPINES", "postal": "520005", "lat": 1.3540, "lng": 103.9450},
    {"block": "9", "street": "STREET IN BISHAN", "town": "BISHAN", "postal": "570009", "lat": 1.3530, "lng": 103.8520},
]
AMENITIES = [
    {"name": "BISHAN MRT STATION", "category": "mrt", "lat": 1.3505, "lng": 103.8505},
    {"name": "Bishan Food Centre", "category": "hawker", "lat": 1.3600, "lng": 103.8500},
    {"name": "SHENG SIONG SUPERMARKET PTE LTD (#01-02 BISHAN)", "category": "supermarket", "lat": 1.3502, "lng": 103.8502},
]
SCHOOLS = [
    {"name": "AI TONG SCHOOL", "alias": "Ai Tong", "town": "BISHAN", "lat": 1.3550, "lng": 103.8500},
    {"name": "CHIJ OUR LADY OF GOOD COUNSEL", "alias": "", "town": "SERANGOON", "lat": 1.3600, "lng": 103.8700},
]


def sale(month, price, block="1", street="BISHAN ST 1", town="BISHAN", flat_type="4 ROOM", area=100, lease="70 years"):
    return {"month": month, "town": town, "flat_type": flat_type, "block": block, "street_name": street,
            "storey_range": "04 TO 06", "floor_area_sqm": str(area), "flat_model": "Model A",
            "lease_commence_date": "1995", "remaining_lease": lease, "resale_price": str(price)}


ROWS = [
    # Block 1 BISHAN ST 1: history across years, latest complete quarter is 2026Q3.
    sale("2021-08", 400000), sale("2023-08", 500000), sale("2025-08", 600000),
    sale("2026-04", 640000), sale("2026-08", 660000), sale("2026-09", 700000),
    sale("2026-08", 300000, flat_type="3 ROOM", area=70),
    # Block 2 BISHAN ST 1: nearby comparables for assessment.
    sale("2026-07", 620000, block="2"), sale("2026-08", 680000, block="2"), sale("2026-09", 720000, block="2"),
    # Tampines: cheaper per sqm, enough sales to rank.
    *[sale(f"2026-0{m}", 450000 + 1000 * m, block="5", street="TAMPINES ST 1", town="TAMPINES") for m in range(1, 10)],
    sale("2021-08", 300000, block="5", street="TAMPINES ST 1", town="TAMPINES"),
    sale("2026-10", 9_000_000),  # in-progress month: excluded everywhere
]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("research")
    source, database, locations = root / "sales.csv", root / "market.duckdb", root / "locations.pkl"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(ROWS)
    locations.write_bytes(pickle.dumps({"payload": {"blocks": BLOCKS}}))
    build_database(source, database, spatial_path=locations, retrieved_at=datetime(2026, 10, 15, tzinfo=timezone.utc))
    return database


@pytest.fixture
def research_env(snapshot, monkeypatch):
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(snapshot))
    monkeypatch.delenv("ONEMAP_EMAIL", raising=False)
    payload = {"blocks": BLOCKS, "amenities": AMENITIES, "schools": SCHOOLS}
    monkeypatch.setattr(location_tools, "get_spatial_index", lambda: {"payload": payload})
    return snapshot


# ---------------------------------------------------------------------------
# Location resolution and nearby amenities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, status, name", [
    ("Blk 1 Bishan Street 1", "ok", "Blk 1 BISHAN ST 1"),
    ("570202", "ok", "Blk 2 BISHAN ST 2"),
    ("Bishan MRT", "ok", "Bishan Mrt Station"),
    ("Ai Tong School", "ok", "Ai Tong School"),
    ("Blk 2 Bishan", "ambiguous", None),          # two streets share block 2
    ("good school", "ambiguous", None),           # weak partial match needs confirmation
    ("Blk 77 Nowhere Rd", "not_found", None),
    ("Blk 9 Bishan", "not_found", None),          # placeholder street is never resolved
])
def test_resolve_location(research_env, query, status, name):
    result = ResolveLocationOutput.model_validate(resolve_location(query))
    assert result.status == status
    if name:
        assert [c.name for c in result.candidates] == [name]
    if status == "ambiguous":
        assert result.candidates


def test_nearby_amenities_counts_nearest_and_validation(research_env):
    result = get_nearby_amenities(address="Blk 1 Bishan St 1", radius_meters=300)
    assert result["status"] == "ok" and result["location"]["name"] == "Blk 1 BISHAN ST 1"
    assert result["counts"] == {"mrt": 1, "hawker": 0, "supermarket": 1, "school": 0}
    assert result["nearest"]["hawker"]["name"] == "Bishan Food Centre"  # reported even beyond the radius
    assert result["nearest"]["supermarket"]["name"] == "Sheng Siong"  # unit-number branch text is dropped
    assert all(a["distance_meters"] <= 300 for a in result["amenities"])
    assert get_nearby_amenities(address="Blk 2 Bishan")["status"] == "ambiguous"
    with pytest.raises(ValueError):
        GetNearbyAmenitiesInput(latitude=1.35)


def test_supermarket_display_names_drop_units_and_legal_suffixes():
    assert display_name("COLD STORAGE SINGAPORE (1983) PTE LTD (ALTEZ (UNITS: #01-05))", "supermarket") == "Cold Storage (Altez)"
    assert display_name("ISETAN (SINGAPORE) LIMITED? (WESTGATE)", "supermarket") == "Isetan (Westgate)"
    assert display_name("U STARS SUPERMARKET PTE. LTD.", "supermarket") == "U Stars Supermarket"


# ---------------------------------------------------------------------------
# Market research tools
# ---------------------------------------------------------------------------
def test_block_history_orders_sales_and_summarises_years(research_env):
    result = GetBlockTransactionHistoryOutput.model_validate(
        get_block_transaction_history("Blk 1 Bishan St 1", "4 room", lookback_months=120))
    assert result.status == "ok" and result.transaction_count == 6
    assert [t.month for t in result.transactions][:2] == ["2026-09", "2026-08"]
    assert [(y.year, y.median_price) for y in result.yearly] == [
        (2021, 400000), (2023, 500000), (2025, 600000), (2026, 660000)]
    assert all(t.resale_price < 9_000_000 for t in result.transactions)  # in-progress month excluded
    assert get_block_transaction_history("Blk 2 Bishan")["status"] == "ambiguous"
    assert get_block_transaction_history("Bishan MRT")["status"] == "ambiguous"  # not an HDB block


def test_market_snapshot_island_row_ignores_town_filter(research_env):
    result = GetMarketSnapshotOutput.model_validate(get_market_snapshot(["BISHAN"], ["4 ROOM"]))
    assert result.latest_complete_quarter == "2026Q3"
    rows = {r.town: r for r in result.rows}
    assert set(rows) == {"ALL TOWNS", "BISHAN"}
    assert rows["ALL TOWNS"].transaction_count > rows["BISHAN"].transaction_count  # includes Tampines
    bishan = rows["BISHAN"]
    assert bishan.median_price == 680000  # 2026Q3: 660k, 700k, 620k, 680k, 720k
    assert bishan.yoy_change_pct == pytest.approx((680000 / 600000 - 1) * 100, abs=0.01)
    assert bishan.change_3y_pct == pytest.approx((680000 / 500000 - 1) * 100, abs=0.01)
    # The 5-year baseline (2021Q3) starts before source coverage (2021-08), so it is not compared.
    assert bishan.change_5y_pct is None
    with pytest.raises(ValueError, match="Unknown town"):
        get_market_snapshot(["ATLANTIS"])


def test_rank_towns_by_value_orders_and_excludes_low_samples(research_env):
    result = RankTownsByValueOutput.model_validate(rank_towns_by_value("4 ROOM", lookback_months=12, min_transactions=6))
    assert [r.town for r in result.rows] == ["TAMPINES", "BISHAN"]  # lower price per sqm first
    strict = rank_towns_by_value("4 ROOM", min_transactions=9)
    assert [r["town"] for r in strict["rows"]] == ["TAMPINES"] and strict["excluded_low_sample_towns"] == ["BISHAN"]
    budget = rank_towns_by_value("4 ROOM", max_budget=650000, min_transactions=1, rank_by="share_within_budget")
    shares = {r["town"]: r["share_within_budget_pct"] for r in budget["rows"]}
    assert shares["TAMPINES"] == 100 and budget["rows"][0]["town"] == "TAMPINES"
    with pytest.raises(ValueError, match="requires max_budget"):
        rank_towns_by_value("4 ROOM", rank_by="share_within_budget")


def test_compare_options_reports_unresolved_addresses(research_env):
    result = CompareResaleOptionsOutput.model_validate(
        compare_resale_options(["Blk 1 Bishan St 1", "Blk 5 Tampines St 1", "Blk 77 Nowhere Rd"], "4 ROOM"))
    assert result.status == "partial"
    first, second, missing = result.options
    assert first.status == "ok" and first.median_price and first.nearest["mrt"].name == "Bishan Mrt Station"
    assert second.location.town == "TAMPINES" and second.transaction_count == 9
    assert missing.status == "not_found" and missing.median_price is None


def test_assess_resale_price_percentile_and_position(research_env):
    result = AssessResalePriceOutput.model_validate(
        assess_resale_price("Blk 1 Bishan St 1", "4 ROOM", asking_price=700000, radius_meters=500, lookback_months=6))
    # Nearby 4-room sales 2026-04..2026-09 within 500 m: 620k-720k; quartiles 645k and 695k.
    assert result.nearby.transaction_count == 6 and result.nearby.median == 670000
    assert (result.nearby.lower_quartile, result.nearby.upper_quartile) == (645000, 695000)
    assert result.asking_percentile == pytest.approx(100 * 4 / 6, abs=0.1)
    assert result.position == "above_upper_quartile"
    assert result.asking_vs_nearby_median_pct == pytest.approx((700000 / 670000 - 1) * 100, abs=0.01)
    assert assess_resale_price("Blk 1 Bishan St 1", "4 ROOM", asking_price=660000, lookback_months=6)["position"] == "within_middle_50"
    assert result.same_block.transaction_count == 3
    assert result.status == "ok"
    assert assess_resale_price("Blk 2 Bishan", "4 ROOM")["status"] == "ambiguous"


def test_town_trend_period_change_and_cagr(research_env):
    trend = compute_town_price_timeseries(["BISHAN"], "4 ROOM", lookback_years=5)
    summary = trend["summary"][0]
    # Earliest complete quarter with sales in the window is 2023Q3 (500k); latest 2026Q3 median 680k.
    assert summary["period_start_quarter"] == "2023Q3" and summary["latest_quarter"] == "2026Q3"
    assert summary["period_change_pct"] == pytest.approx(36.0)
    assert summary["period_cagr_pct"] == pytest.approx((1.36 ** (1 / 3) - 1) * 100)


def test_block_keys_filter_prices_spatial_results(research_env):
    result = query_resale_blocks_by_criteria([], ["4 ROOM"], 5_000_000, 0, block_keys=["2 bishan st 1", "77 NOWHERE RD"])
    assert [(b["block"], b["street"]) for b in result["blocks"]] == [("2", "BISHAN ST 1")]


# ---------------------------------------------------------------------------
# BTO launches (official HDB annex data)
# ---------------------------------------------------------------------------
def test_bto_dataset_rows_are_complete_and_sourced():
    rows = load_launches()
    assert {r["launch_month"] for r in rows} == {"2026-02", "2026-06"}
    for r in rows:
        assert r["source_url"].startswith("https://www.hdb.gov.sg/") and r["classification"] in ("Standard", "Plus", "Prime")
        assert 0 < float(r["price_min"]) <= float(r["price_max"])


def test_bto_comparison_uses_official_range_and_town_resale(research_env):
    result = CompareBtoLaunchVsResaleOutput.model_validate(
        compare_bto_launch_vs_resale("lakeview cascadia", flat_type="4-room"))
    assert result.status == "ok" and result.town == "BISHAN" and result.classification == "Prime"
    assert result.bto_indicative_price_range == {"min": 534000.0, "max": 742000.0}
    assert result.hdb_resale_comparable_range == {"min": 840000.0, "max": 950000.0}
    assert result.resale_sample_size == 6  # Bishan 4-room sales 2025-10..2026-09 in the synthetic snapshot
    assert result.price_gap_pct == pytest.approx(((534000 + 742000) / 2 / result.resale_median_nearby - 1) * 100, abs=0.01)
    assert compare_bto_launch_vs_resale("Tengah Garden")["status"] == "not_found"
    assert compare_bto_launch_vs_resale("Lakeview Cascadia", flat_type="5 ROOM")["status"] == "flat_type_not_offered"


# ---------------------------------------------------------------------------
# Area research: find blocks near a place, then pooled statistics over the set
# ---------------------------------------------------------------------------
from agents.research_schemas import (  # noqa: E402
    BlockSetInput,
    FindResaleBlocksNearOutput,
    GetAreaPriceTrendOutput,
    GetAreaResaleStatisticsOutput,
)
from mcp_server.area_resale_tools import (  # noqa: E402
    find_resale_blocks_near,
    get_area_price_trend,
    get_area_resale_statistics,
)

AREA = {"address": "Blk 1 Bishan St 1", "radius_meters": 200}  # Blk 2 Bishan St 1 is ~111 m away


def test_find_resale_blocks_near_lists_blocks_nearest_first(research_env):
    out = FindResaleBlocksNearOutput.model_validate(find_resale_blocks_near(**AREA, flat_type="4 ROOM"))
    assert out.status == "ok" and [b.block_key for b in out.blocks] == ["1 BISHAN ST 1", "2 BISHAN ST 1"]
    assert out.blocks[0].is_target and out.blocks[0].distance_meters == 0 and not out.blocks[1].is_target
    assert [b.transaction_count for b in out.blocks] == [3, 3] and out.transaction_count == 6
    assert out.blocks[1].distance_meters == pytest.approx(111, abs=2)
    by_pin = find_resale_blocks_near(latitude=1.3540, longitude=103.9450, radius_meters=100)
    assert [b["block_key"] for b in by_pin["blocks"]] == ["5 TAMPINES ST 1"] and not by_pin["blocks"][0]["is_target"]
    assert find_resale_blocks_near(address="Blk 2 Bishan")["status"] == "ambiguous"


def test_area_statistics_pool_every_sale(research_env):
    out = GetAreaResaleStatisticsOutput.model_validate(get_area_resale_statistics(area=AREA))
    four = next(f for f in out.by_flat_type if f.flat_type == "4 ROOM")
    # Pooled 4-room sales 2025-10..2026-09: 620, 640, 660, 680, 700, 720k -> median 670k (not a median of block medians).
    assert four.distribution.transaction_count == 6 and four.distribution.median == 670000 and four.block_count == 2
    assert [f.flat_type for f in out.by_flat_type] == ["3 ROOM", "4 ROOM"]
    target = next(f for f in out.target_by_flat_type if f.flat_type == "4 ROOM")
    assert out.target_block_key == "1 BISHAN ST 1" and target.distribution.median == 660000


def test_area_statistics_accept_block_keys_and_report_unknown(research_env):
    out = get_area_resale_statistics(block_keys=["2 bishan st 1", "999 NOWHERE RD"], flat_type="4 ROOM")
    assert out["scope"] == "block_keys" and out["unknown_block_keys"] == ["999 NOWHERE RD"]
    assert out["by_flat_type"][0]["distribution"]["median"] == 680000
    assert get_area_resale_statistics(block_keys=["999 NOWHERE RD"])["status"] == "not_found"
    with pytest.raises(ValueError):
        BlockSetInput(area=AREA, block_keys=["1 BISHAN ST 1"])


def test_area_trend_never_uses_thin_quarters_as_endpoints(research_env):
    out = GetAreaPriceTrendOutput.model_validate(get_area_price_trend("4 ROOM", area=AREA, lookback_years=5))
    counts = {p.quarter: p.transaction_count for p in out.points}
    assert counts["2026Q3"] == 5 and counts["2023Q3"] == 1
    assert out.period_change_pct is None and any("Too few quarters" in n for n in out.metadata.notes)


def test_block_history_over_an_area(research_env):
    out = GetBlockTransactionHistoryOutput.model_validate(
        get_block_transaction_history(area=AREA, flat_type="4 ROOM", lookback_months=12))
    assert out.scope == "area" and out.block_count == 2 and out.transaction_count == 6
    assert {(t.block, t.street) for t in out.transactions} == {("1", "BISHAN ST 1"), ("2", "BISHAN ST 1")}
    assert next(t for t in out.transactions if t.block == "2").distance_meters == pytest.approx(111, abs=2)
