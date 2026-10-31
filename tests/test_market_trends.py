"""Independent quarterly aggregation, comparison and integration cases."""

from __future__ import annotations

import csv
import re
import json
from datetime import datetime, timezone

import anyio
import pytest

from agents.schemas import (
    ComputeTownPriceTimeseriesInput,
    ComputeTownPriceTimeseriesOutput,
)
from data.build_market_db import SOURCE_COLUMNS, build_database
from mcp_server.market_sql_tools import compute_town_price_timeseries


def transaction(month, price, area=100, town="BISHAN", flat_type="4 ROOM"):
    return {
        "month": month,
        "town": town,
        "flat_type": flat_type,
        "block": "123",
        "street_name": f"{town} ST 1",
        "storey_range": "01 TO 03",
        "floor_area_sqm": str(area),
        "flat_model": "Model A",
        "lease_commence_date": "1990",
        "remaining_lease": "60 years",
        "resale_price": str(price),
    }


def create_snapshot(root, rows, retrieved_at):
    source, database = root / "resale.csv", root / "market.duckdb"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    profile = build_database(source, database, retrieved_at=retrieved_at)
    return database, profile


@pytest.fixture(scope="module")
def trend_snapshot(tmp_path_factory):
    rows = [
        transaction("2024-02", 200000),
        transaction("2024-04", 300000),
        transaction("2024-07", 300000),
        transaction("2024-10", 500000),
        transaction("2025-01", 400000),
        transaction("2025-04", 400000, 100),
        transaction("2025-05", 600000, 200),
        transaction("2025-06", 800000, 100),
        transaction("2025-10", 800000),
        transaction("2026-01", 750000),
        transaction("2026-04", 900000, 150),
        transaction("2026-05", 1200000, 200),
        transaction("2026-06", 1500000, 100),
        transaction("2026-08", 100000000),  # Incomplete Q3: excluded from summary.
        transaction("2026-09", 200000000),  # In-progress month: excluded entirely.
        transaction("2024-04", 200000, town="HOUGANG"),
        transaction("2025-01", 500000, town="HOUGANG"),
        transaction("2026-04", 400000, town="HOUGANG"),  # No 2025Q2 comparison.
        transaction("2026-08", 600000, town="TAMPINES"),  # Only a partial quarter.
        transaction("2026-05", 100000, town="BEDOK", flat_type="5 ROOM"),
        transaction("2026-06", 100000, flat_type="5 ROOM"),
    ]
    return create_snapshot(
        tmp_path_factory.mktemp("trends"),
        rows,
        datetime(2026, 9, 15, tzinfo=timezone.utc),
    )


@pytest.fixture
def trend_database(trend_snapshot, monkeypatch):
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(trend_snapshot[0]))
    return trend_snapshot


def test_quarterly_medians_counts_and_price_per_sqm(trend_database):
    result = compute_town_price_timeseries([" bishan ", "BISHAN"], "4 room")
    points = {point["quarter"]: point for point in result["series"]}
    # Median of [400k, 600k, 800k] is 600k; median of [4000, 3000, 8000] is 4000.
    # Dividing the median price by median area would incorrectly give 6000/sqm.
    assert points["2025Q2"]["median_price"] == 600000
    assert points["2025Q2"]["median_price_per_sqm"] == 4000
    assert points["2025Q2"]["transaction_count"] == 3
    assert points["2025Q2"]["is_complete"] is True
    assert points["2026Q2"]["median_price"] == 1200000
    assert points["2026Q2"]["median_price_per_sqm"] == 6000
    assert points["2026Q2"]["transaction_count"] == 3
    assert {point["town"] for point in result["series"]} == {"BISHAN"}
    assert [point["quarter"] for point in result["series"]] == sorted(points)
    ComputeTownPriceTimeseriesOutput.model_validate(result)


def test_summary_uses_complete_quarter_and_same_quarter_last_year(trend_database):
    result = compute_town_price_timeseries(["BISHAN"], "4 ROOM")
    summary = result["summary"][0]
    assert summary["latest_quarter"] == "2026Q2"
    assert summary["latest_median_price"] == 1200000
    assert summary["transaction_count"] == 3
    assert summary["yoy_comparison_quarter"] == "2025Q2"
    assert summary["yoy_appreciation_pct"] == 100
    assert summary["yoy_unavailable_reason"] is None


def test_one_year_display_fetches_baseline_outside_display(trend_database):
    result = compute_town_price_timeseries(["BISHAN"], "4 ROOM", lookback_years=1)
    assert [point["quarter"] for point in result["series"]] == [
        "2025Q4",
        "2026Q1",
        "2026Q2",
        "2026Q3",
    ]
    assert result["metadata"]["requested_start_month"] == "2025-10"
    assert result["summary"][0]["yoy_appreciation_pct"] == 100


def test_partial_quarters_are_labelled_and_current_month_is_excluded(trend_database):
    result = compute_town_price_timeseries(["BISHAN"], "4 ROOM")
    points = {point["quarter"]: point for point in result["series"]}
    assert points["2024Q1"]["is_complete"] is False  # Source starts in February.
    assert points["2026Q3"]["is_complete"] is False  # Analysis ends in August.
    assert points["2026Q3"]["median_price"] == 100000000
    assert points["2026Q3"]["transaction_count"] == 1
    assert result["metadata"]["window_end_month"] == "2026-08"
    assert result["metadata"]["latest_source_month"] == "2026-09"
    assert result["metadata"]["history_truncated"] is True
    assert result["metadata"]["snapshot_id"] == trend_database[1]["snapshot_id"]


def test_missing_same_quarter_is_not_replaced_with_neighbouring_observation(trend_database):
    result = compute_town_price_timeseries(["HOUGANG"], "4 ROOM")
    summary = result["summary"][0]
    assert summary["latest_quarter"] == "2026Q2"
    assert summary["yoy_comparison_quarter"] == "2025Q2"
    assert summary["yoy_appreciation_pct"] is None
    assert summary["yoy_unavailable_reason"] == "missing_prior_year_quarter"
    assert "2025Q2" not in {point["quarter"] for point in result["series"]}


def test_incomplete_prior_year_quarter_has_no_yoy(tmp_path, monkeypatch):
    database, _ = create_snapshot(
        tmp_path,
        [
            transaction("2024-02", 200000),
            transaction("2025-01", 400000),
            transaction("2025-03", 400000),
        ],
        datetime(2025, 4, 15, tzinfo=timezone.utc),
    )
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(database))
    result = compute_town_price_timeseries(["BISHAN"], "4 ROOM", 1)
    summary = result["summary"][0]
    assert summary["latest_quarter"] == "2025Q1"
    assert summary["yoy_appreciation_pct"] is None
    assert summary["yoy_unavailable_reason"] == "incomplete_prior_year_quarter"


def test_partial_only_and_empty_town_results_are_explicit(trend_database):
    result = compute_town_price_timeseries(["TAMPINES", "BEDOK"], "4 ROOM")
    assert result["summary"] == []
    assert len(result["series"]) == 1 and result["series"][0]["is_complete"] is False
    assert result["metadata"]["towns_without_data"] == ["BEDOK"]
    assert result["metadata"]["towns_without_complete_quarters"] == ["TAMPINES"]


def test_all_towns_and_flat_type_filters(trend_database):
    result = compute_town_price_timeseries([], "5 ROOM")
    assert {point["town"] for point in result["series"]} == {"BEDOK", "BISHAN"}
    assert result["metadata"]["towns_without_data"] == ["HOUGANG", "TAMPINES"]
    assert all(point["median_price"] == 100000 for point in result["series"])


@pytest.mark.parametrize("years", [0, -1, 31, 1.5, True, "3"])
def test_invalid_lookback_is_rejected(years):
    with pytest.raises(ValueError):
        ComputeTownPriceTimeseriesInput(towns=["BISHAN"], flat_type="4 ROOM", lookback_years=years)


@pytest.mark.parametrize("towns,flat_type", [([" "], "4 ROOM"), (["BISHAN"], " ")])
def test_blank_filters_are_rejected(towns, flat_type):
    with pytest.raises(ValueError):
        ComputeTownPriceTimeseriesInput(towns=towns, flat_type=flat_type)


def test_unknown_filters_and_sql_injection_are_rejected(trend_database):
    with pytest.raises(ValueError, match="Unknown town"):
        compute_town_price_timeseries(["BISHAN'; DROP TABLE resale_transactions; --"], "4 ROOM")
    with pytest.raises(ValueError, match="Unknown flat_type"):
        compute_town_price_timeseries(["BISHAN"], "99 ROOM")
    assert compute_town_price_timeseries(["BISHAN"], "4 ROOM")["series"]


def test_missing_database_has_setup_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(tmp_path / "missing.duckdb"))
    with pytest.raises(FileNotFoundError, match="python -m data.build_market_db"):
        compute_town_price_timeseries(["BISHAN"], "4 ROOM")


def test_template_handles_unavailable_yoy_and_partial_only_results(trend_database):
    from agents.supervisor import template_response

    trend = compute_town_price_timeseries(["HOUGANG", "TAMPINES", "BEDOK"], "4 ROOM")
    text = template_response({"sql_market_results": {"compute_town_price_timeseries": trend}})
    assert "YoY unavailable" in text and "2026Q2" in text
    assert "only incomplete quarters" in text and "BEDOK: no transactions" in text
    assert "No block satisfies" not in text
    # An unavailable comparison must never be rendered as a zero change.
    assert not re.search(r"YoY [+-]?0\.0%", text)


def test_agent_calls_real_trend_tool(trend_database, monkeypatch):
    from agents import arch_3_specialist_5agent as C
    from agents.llm import load_config

    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setitem(load_config()["tools"]["use_mock"], "compute_town_price_timeseries", "real")
    monkeypatch.setitem(
        load_config()["tools"]["use_mock"], "query_resale_blocks_by_criteria", "real"
    )
    result = C.run("Show 4-room price trends in Bishan.")
    trend = result["sql_market_results"]["compute_town_price_timeseries"]
    assert trend["metadata"]["snapshot_id"] == trend_database[1]["snapshot_id"]
    assert "2026Q2" in result["draft_response"] and "+100.0%" in result["draft_response"]
    calls = [
        call for call in result["tool_call_log"] if call["tool"] == "compute_town_price_timeseries"
    ]
    assert len(calls) == 1 and calls[0]["ok"]
    assert not result["errors"]


def test_mcp_server_discovers_and_executes_trend_tool(trend_database):
    from fastmcp import Client

    from mcp_server.server import mcp

    async def exercise():
        async with Client(mcp, timeout=10) as client:
            tools = await client.list_tools()
            tool = next(tool for tool in tools if tool.name == "compute_town_price_timeseries")
            assert tool.inputSchema["properties"]["lookback_years"]["minimum"] == 1
            result = await client.call_tool(
                tool.name,
                {
                    "towns": ["BISHAN"],
                    "flat_type": "4 ROOM",
                    "lookback_years": 1,
                },
            )
            payload = json.loads(result.content[0].text)
            assert not result.is_error
            assert payload["summary"][0]["yoy_appreciation_pct"] == 100
            ComputeTownPriceTimeseriesOutput.model_validate(payload)

    anyio.run(exercise)
