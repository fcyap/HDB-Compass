"""Independent numeric and integration checks using tiny synthetic resale data."""

from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime, timezone

import anyio
import pytest

from agents.schemas import QueryResaleBlocksInput, QueryResaleBlocksOutput
from data.build_market_db import SOURCE_COLUMNS, build_database, parse_lease_months
from data.market_database import market_settings, open_market_database
from mcp_server.market_sql_tools import query_resale_blocks_by_criteria

CAPTURED_AT = datetime(2026, 10, 25, tzinfo=timezone.utc)


def record(**overrides):
    return {
        "month": "2026-09",
        "town": "BISHAN",
        "flat_type": "4 ROOM",
        "block": "123",
        "street_name": "BISHAN ST 13",
        "storey_range": "01 TO 03",
        "floor_area_sqm": "100",
        "flat_model": "Model A",
        "lease_commence_date": "1992",
        "remaining_lease": "65 years 01 month",
        "resale_price": "400000",
        **overrides,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def market_snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    source, database, spatial = root / "resale.csv", root / "market.duckdb", root / "spatial.pkl"
    rows = [
        record(),
        record(resale_price="800000"),
        record(month="2025-01", resale_price="100000"),  # Outside the 12-month window.
        record(month="2026-10", resale_price="100000"),  # Incomplete current month.
        record(block="151", month="2026-08", remaining_lease="65 years", resale_price="500000"),
        record(town="TAMPINES", street_name="TAMPINES ST 21", resale_price="700000"),
        record(block="9B", resale_price="550000"),
        record(block="9B", resale_price="550000"),
    ]
    write_csv(source, rows)
    spatial.write_bytes(
        pickle.dumps(
            {
                "payload": {
                    "blocks": [
                        {
                            "block": "123",
                            "street": "BISHAN ST 13",
                            "postal": "570123",
                            "lat": 1.35,
                            "lng": 103.85,
                        },
                    ]
                }
            }
        )
    )
    profile = build_database(source, database, spatial_path=spatial, retrieved_at=CAPTURED_AT)
    return source, database, profile


@pytest.fixture
def market_database(market_snapshot, monkeypatch):
    monkeypatch.setenv("HDB_COMPASS_DB_PATH", str(market_snapshot[1]))
    return market_snapshot[1]


@pytest.mark.parametrize(
    "raw,expected",
    [("61 years 04 months", 736), ("65 years", 780), ("1 year 1 month", 13), ("8 months", 8)],
)
def test_lease_parsing(raw, expected):
    assert parse_lease_months(raw) == expected


@pytest.mark.parametrize("raw", ["", "NIL", "65 years 12 months", "65 years garbage", "-1 year"])
def test_invalid_leases_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_lease_months(raw)


def test_snapshot_retains_duplicates_and_parses_types(market_snapshot):
    _, database, profile = market_snapshot
    assert profile["row_count"] == 8 and profile["duplicate_looking_rows_retained"] == 1
    assert profile["distinct_blocks"] == 4 and profile["blocks_with_coordinates"] == 1
    assert profile["analysis_end_month"] == "2026-09"
    with open_market_database(database) as con:
        assert (
            con.execute("SELECT count(*) FROM resale_transactions WHERE block='9B'").fetchone()[0]
            == 2
        )
        assert (
            con.execute(
                "SELECT remaining_lease_months FROM resale_transactions WHERE source_row_number=1"
            ).fetchone()[0]
            == 781
        )
        assert (
            con.execute("SELECT typeof(resale_price) FROM resale_transactions LIMIT 1").fetchone()[
                0
            ]
            == "DECIMAL(14,2)"
        )


def test_median_filters_do_not_discard_expensive_transactions(market_database):
    # Two sales: 400k and 800k. The median is 600k, not 400k after filtering.
    result = query_resale_blocks_by_criteria([" bishan "], ["4 room"], 500000, 65)
    assert result["blocks"] == []
    result = query_resale_blocks_by_criteria(["BISHAN"], ["4 ROOM"], 600000, 65)
    row = next(row for row in result["blocks"] if row["block"] == "123")
    assert row["median_resale_price"] == 600000
    assert row["median_price_per_sqm"] == 6000
    assert row["transaction_count"] == 2
    assert row["median_remaining_lease_years"] == pytest.approx(65 + 1 / 12)
    assert row["postal"] == "570123" and row["lat"] == 1.35
    assert result["metadata"]["window_start_month"] == "2025-10"
    assert result["metadata"]["window_end_month"] == "2026-09"
    QueryResaleBlocksOutput.model_validate(result)


def test_lease_is_adjusted_to_the_shared_reference_month(market_database):
    # August's 65-year lease is 64y11m in September and fails a 65-year limit.
    result = query_resale_blocks_by_criteria(["BISHAN"], ["4 ROOM"], 1000000, 65)
    assert "151" not in {row["block"] for row in result["blocks"]}
    assert next(row for row in result["blocks"] if row["block"] == "9B")["lat"] is None


def test_unknown_filters_and_invalid_arguments_fail_explicitly(market_database):
    with pytest.raises(ValueError, match="Unknown town"):
        query_resale_blocks_by_criteria(
            ["BISHAN'); DROP TABLE resale_transactions; --"], [], 1000000, 0
        )
    with pytest.raises(ValueError, match="Unknown flat_type"):
        query_resale_blocks_by_criteria([], ["99 ROOM"], 1000000, 0)
    for price in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            QueryResaleBlocksInput(towns=[], flat_types=[], max_price=price, min_lease_years=0)


def test_result_limit_is_reported_without_losing_total_count(market_database, monkeypatch):
    from mcp_server import market_sql_tools

    monkeypatch.setattr(market_sql_tools, "RESULT_LIMIT", 1)
    result = query_resale_blocks_by_criteria(["BISHAN"], ["4 ROOM"], 600000, 65)
    assert len(result["blocks"]) == 1 and result["blocks"][0]["block"] == "9B"
    assert result["metadata"]["total_matching_blocks"] == 2
    assert result["metadata"]["returned_blocks"] == 1 and result["metadata"]["truncated"]


def test_rebuild_is_idempotent_and_bad_input_keeps_previous_database(market_snapshot, tmp_path):
    source, _, _ = market_snapshot
    target = tmp_path / "snapshot.duckdb"
    first = build_database(source, target, retrieved_at=CAPTURED_AT)
    second = build_database(source, target, retrieved_at=CAPTURED_AT)
    assert first["snapshot_id"] == second["snapshot_id"]
    invalid = tmp_path / "invalid.csv"
    write_csv(invalid, [record(remaining_lease="unknown")])
    with pytest.raises(ValueError, match="database unchanged"):
        build_database(invalid, target, retrieved_at=CAPTURED_AT)
    with open_market_database(target) as con:
        assert con.execute("SELECT count(*) FROM resale_transactions").fetchone()[0] == 8


def test_missing_database_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m data.build_market_db"):
        with open_market_database(tmp_path / "missing.duckdb"):
            pass


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_GOV_SG_API_KEY", raising=False)
    monkeypatch.delenv("HDB_COMPASS_DB_PATH", raising=False)
    (tmp_path / ".env").write_text(
        "DATA_GOV_SG_API_KEY=local\nHDB_COMPASS_DB_PATH=data/local/market.duckdb\n"
    )
    settings = market_settings(tmp_path)
    assert settings["api_key"] == "local"
    assert settings["db_path"] == tmp_path / "data/local/market.duckdb"
    monkeypatch.setenv("DATA_GOV_SG_API_KEY", "environment")
    assert market_settings(tmp_path)["api_key"] == "environment"


def test_agent_uses_real_query_and_keeps_evidence(market_database, monkeypatch):
    from agents import arch_3_specialist_5agent as C
    from agents.llm import load_config

    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setitem(
        load_config()["tools"]["use_mock"], "query_resale_blocks_by_criteria", "real"
    )
    state = C.run("Find 4-room flats in Bishan under $600k with at least 65 years lease.")
    calls = [
        call for call in state["tool_call_log"] if call["tool"] == "query_resale_blocks_by_criteria"
    ]
    assert calls and all(call["ok"] for call in calls)
    assert {row["block"] for row in state["candidate_blocks_intersection"]} == {"123", "9B"}
    assert state["sql_market_results"]["query_resale_blocks_by_criteria"]["metadata"]["snapshot_id"]
    assert "600,000" in state["draft_response"]


def test_unified_mcp_server_exposes_and_executes_real_tool(market_database):
    from fastmcp import Client

    from mcp_server.server import mcp

    async def exercise():
        async with Client(mcp, timeout=10) as client:
            tools = await client.list_tools()
            tool = next(tool for tool in tools if tool.name == "query_resale_blocks_by_criteria")
            assert "max_price" in tool.inputSchema["properties"]
            result = await client.call_tool(
                tool.name,
                {
                    "towns": ["BISHAN"],
                    "flat_types": ["4 ROOM"],
                    "max_price": 600000,
                    "min_lease_years": 65,
                },
            )
            payload = json.loads(result.content[0].text)
            assert not result.is_error
            assert {row["block"] for row in payload["blocks"]} == {"123", "9B"}
            QueryResaleBlocksOutput.model_validate(payload)

    anyio.run(exercise)
