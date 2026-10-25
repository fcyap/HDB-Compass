"""Independent comparable-sale statistics, matching boundaries and integration."""

from __future__ import annotations

import csv
import math
import pickle
from datetime import datetime, timezone

import anyio
import pytest
from pydantic import ValidationError

from agents.schemas import QueryComparableResaleInput, QueryComparableResaleOutput
from data.build_market_db import SOURCE_COLUMNS, build_database
from mcp_server.comparable_resale_tools import (
    query_comparable_resale_transactions as compare,
)


def record(**changes):
    return {
        "month": "2026-09",
        "town": "BISHAN",
        "flat_type": "4 ROOM",
        "block": "1",
        "street_name": "BISHAN ST 1",
        "storey_range": "01 TO 03",
        "floor_area_sqm": "100",
        "flat_model": "Model A",
        "lease_commence_date": "1992",
        "remaining_lease": "65 years",
        "resale_price": "200000",
        **changes,
    }


def make_snapshot(root, rows, geocoded=True):
    source, database, locations = (
        root / "sales.csv",
        root / "market.duckdb",
        root / "locations.pkl",
    )
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    locations.write_bytes(
        pickle.dumps(
            {
                "payload": {
                    "blocks": [
                        {
                            "block": "1",
                            "street": "BISHAN ST 1",
                            "lat": 1.35,
                            "lng": 103.85,
                        },
                        {
                            "block": "2",
                            "street": "BISHAN ST 1",
                            "lat": 1.351,
                            "lng": 103.85,
                        },
                        {
                            "block": "3",
                            "street": "BISHAN ST 1",
                            "lat": 1.36,
                            "lng": 103.85,
                        },
                    ]
                }
            }
        )
    )
    build_database(
        source,
        database,
        spatial_path=locations if geocoded else None,
        retrieved_at=datetime(2026, 10, 25, tzinfo=timezone.utc),
    )
    return database


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    rows = [
        record(),
        record(
            resale_price="400000",
            month="2026-08",
            storey_range="04 TO 06",
            remaining_lease="65 years 01 month",
        ),
        record(
            block="2",
            resale_price="600000",
            floor_area_sqm="200",
            storey_range="07 TO 09",
        ),
        record(
            block="2",
            resale_price="800000",
            floor_area_sqm="200",
            storey_range="10 TO 12",
        ),
        record(block="3", resale_price="9000000", storey_range="04 TO 06"),
        record(block="4", resale_price="1000000", storey_range="04 TO 06"),
        record(flat_type="3 ROOM", resale_price="1"),
        record(month="2025-08", resale_price="300000"),
        record(month="2025-09", resale_price="100000"),
        record(block="2", month="2025-09", resale_price="700000"),
        record(month="2026-10", resale_price="50000000"),
    ]
    return make_snapshot(tmp_path_factory.mktemp("comparables"), rows)


@pytest.fixture
def database(snapshot, monkeypatch):
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(snapshot))
    return snapshot


def inputs(**changes):
    return {"latitude": 1.35, "longitude": 103.85, "flat_type": "4 room", **changes}


def test_quartiles_and_summary_use_all_sales_before_sample_limit(database):
    r = compare(**inputs(sample_limit=1))
    assert r["status"] == "insufficient_sample"
    assert r["summary"]["transaction_count"] == 4
    assert r["summary"]["block_count"] == 2
    assert r["summary"]["price"] == {
        "minimum": 200000,
        "lower_quartile": 350000,
        "median": 500000,
        "upper_quartile": 650000,
        "maximum": 800000,
    }
    assert r["summary"]["price_per_sqm"] == {
        "minimum": 2000,
        "lower_quartile": 2750,
        "median": 3500,
        "upper_quartile": 4000,
        "maximum": 4000,
    }
    assert r["summary"]["median_floor_area_sqm"] == 150
    assert r["summary"]["median_remaining_lease_years"] == 65
    assert len(r["transactions"]) == 1 and r["metadata"]["sample_truncated"]
    assert r["transactions"][0]["resale_price"] == 200000
    assert r["metadata"]["transactions_without_coordinates"] == 1
    assert r["metadata"]["window_start_month"] == "2025-10"
    assert r["metadata"]["window_end_month"] == "2026-09"
    assert r["metadata"]["history_truncated"] is False
    QueryComparableResaleOutput.model_validate(r)


def test_distance_units_radius_boundary_and_stable_order(database):
    r = compare(**inputs(radius_meters=112, min_sample_size=4))
    assert r["status"] == "ok"
    assert [row["resale_price"] for row in r["transactions"]] == [
        200000,
        400000,
        600000,
        800000,
    ]
    expected = math.radians(0.001) * 6371008.8
    assert r["transactions"][2]["distance_meters"] == pytest.approx(expected, abs=1e-6)
    assert compare(**inputs(radius_meters=111))["summary"]["transaction_count"] == 2
    assert r == compare(**inputs(radius_meters=112, min_sample_size=4))


@pytest.mark.parametrize(
    "criteria,prices",
    [
        ({"min_floor_area_sqm": 150}, [600000, 800000]),
        ({"max_floor_area_sqm": 100}, [200000, 400000]),
        ({"min_storey": 4, "max_storey": 9}, [400000, 600000]),
        ({"min_storey": 5, "max_storey": 6}, []),
        (
            {"min_lease_years": 65, "max_lease_years": 65},
            [200000, 400000, 600000, 800000],
        ),
        ({"max_lease_years": 64.99}, []),
    ],
)
def test_attribute_matching_and_no_silent_relaxation(database, criteria, prices):
    r = compare(**inputs(**criteria))
    assert [row["resale_price"] for row in r["transactions"]] == prices
    if not prices:
        assert r["status"] == "no_matches" and r["summary"] is None


def test_historical_window_and_lease_reference(database):
    r = compare(**inputs(end_month="2025-09", lookback_months=2))
    assert r["summary"]["transaction_count"] == 3
    assert r["summary"]["price"]["median"] == 300000
    assert r["metadata"]["lease_reference_month"] == "2025-09"
    r = compare(**inputs(end_month="2026-08", lookback_months=1))
    assert r["summary"]["median_remaining_lease_years"] == pytest.approx(65 + 1 / 12)
    assert r["summary"]["price"]["median"] == 400000
    assert r["status"] == "insufficient_sample"
    assert r["summary"]["transaction_count"] == 1
    assert set(r["summary"]["price"].values()) == {400000}
    assert compare(**inputs(lookback_months=120))["metadata"]["history_truncated"]


def test_no_coverage_unknown_type_and_unavailable_end_month(database):
    r = compare(**inputs(end_month="2024-12"))
    assert (
        r["status"] == "no_coverage"
        and r["summary"] is None
        and r["transactions"] == []
    )
    with pytest.raises(ValueError, match="last completed analysis month"):
        compare(**inputs(end_month="2026-10"))
    with pytest.raises(ValueError, match="Unknown flat_type"):
        compare(**inputs(flat_type="4 ROOM'; DROP TABLE resale_transactions; --"))
    assert compare(**inputs())["summary"]["transaction_count"] == 4


def test_all_geocodes_missing_are_explicit(tmp_path, monkeypatch):
    path = make_snapshot(tmp_path, [record()], geocoded=False)
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(path))
    r = compare(**inputs())
    assert (
        r["status"] == "no_matches"
        and r["metadata"]["transactions_without_coordinates"] == 1
    )


def test_duplicate_looking_source_sales_are_preserved(tmp_path, monkeypatch):
    path = make_snapshot(tmp_path, [record(), record()])
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(path))
    r = compare(**inputs())
    assert r["summary"]["transaction_count"] == 2
    assert r["summary"]["price"]["lower_quartile"] == 200000
    assert len({row["source_row_number"] for row in r["transactions"]}) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"latitude": float("nan")},
        {"longitude": 181},
        {"radius_meters": 0},
        {"radius_meters": 500.5},
        {"radius_meters": True},
        {"lookback_months": 121},
        {"sample_limit": 101},
        {"min_sample_size": 0},
        {"flat_type": " "},
        {"end_month": "2026-13"},
        {"end_month": "2026-9"},
        {"end_month": "2026-09-01"},
        {"min_floor_area_sqm": 101, "max_floor_area_sqm": 100},
        {"min_lease_years": 66, "max_lease_years": 65},
        {"min_storey": 7, "max_storey": 6},
        {"min_floor_area_sqm": float("inf")},
        {"unknown": 1},
    ],
)
def test_invalid_inputs_are_rejected(changes):
    with pytest.raises(ValidationError):
        QueryComparableResaleInput(**inputs(**changes))


def test_missing_database_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(tmp_path / "missing.duckdb"))
    with pytest.raises(FileNotFoundError, match="build_market_db"):
        compare(**inputs())


def test_longitude_distance_uses_latitude_correction(database):
    r = compare(**inputs(longitude=103.851, radius_meters=112))
    assert r["summary"]["transaction_count"] == 2
    expected = (
        2
        * 6371008.8
        * math.asin(math.cos(math.radians(1.35)) * math.sin(math.radians(0.001) / 2))
    )
    assert r["transactions"][0]["distance_meters"] == pytest.approx(expected, abs=1e-6)
    assert (
        compare(**inputs(longitude=103.851, radius_meters=111))["status"]
        == "no_matches"
    )


def test_real_registry_and_graph_keep_comparable_evidence(database, monkeypatch):
    from agents import arch_3_specialist_5agent as C
    from agents import supervisor as sv
    from agents.schemas import ExtractedConstraints
    from agents.tool_registry import ToolRunner

    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setitem(
        sv.load_config()["tools"]["use_mock"],
        "query_comparable_resale_transactions",
        "real",
    )
    runner = ToolRunner("query_comparable_resale_transactions")
    assert "latitude" in runner.as_langchain_tool().args
    assert runner(**inputs())["summary"]["transaction_count"] == 4
    assert "error" in runner(**inputs(radius_meters=-1))
    monkeypatch.setattr(
        sv,
        "rule_based_extract",
        lambda query: ExtractedConstraints(
            comparable_resale=inputs(), max_budget_sgd=900000
        ),
    )
    state = C.run("Compare sales around the supplied coordinates.")
    assert state["active_subagents"] == ["market_sql"]
    result = state["sql_market_results"]["query_comparable_resale_transactions"]
    assert result["summary"]["transaction_count"] == 4
    assert state["tool_call_log"][0]["ok"]
    assert (
        "500,000" in state["draft_response"]
        and "below the requested minimum" in state["draft_response"]
    )
    assert "No block satisfies" not in state["draft_response"]
    assert not state.get("relaxation")


def test_templates_explain_absent_coverage_and_missing_locations(database):
    from agents.supervisor import template_response

    for args, expected in (
        (inputs(end_month="2024-12"), "predates"),
        (inputs(min_floor_area_sqm=400), "filters have not been widened"),
    ):
        r = compare(**args)
        assert expected in template_response(
            {"sql_market_results": {"query_comparable_resale_transactions": r}}
        )


@pytest.mark.parametrize(
    "server_module", ["mcp_server.server", "mcp_server.market_sql_tools"]
)
def test_mcp_discovers_validates_and_executes(database, server_module):
    import importlib

    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    mcp = importlib.import_module(server_module).mcp

    async def check():
        async with Client(mcp, timeout=10) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            schema = tools["query_comparable_resale_transactions"].inputSchema
            assert schema["additionalProperties"] is False
            assert schema["properties"]["radius_meters"]["maximum"] == 10000
            response = await client.call_tool(
                "query_comparable_resale_transactions", inputs()
            )
            parsed = QueryComparableResaleOutput.model_validate(
                response.structured_content
            )
            assert parsed.summary.transaction_count == 4
            with pytest.raises(ToolError):
                await client.call_tool(
                    "query_comparable_resale_transactions", inputs(radius_meters=0)
                )

    anyio.run(check)
