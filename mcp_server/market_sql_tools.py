"""Read-only resale queries over local DuckDB snapshots."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Dict, List

from fastmcp import FastMCP
from pydantic import Field

from agents.schemas import (
    ComputeTownPriceTimeseriesInput,
    ComputeTownPriceTimeseriesOutput,
    QueryComparableResaleInput,
    QueryComparableResaleOutput,
    QueryResaleBlocksInput,
    QueryResaleBlocksOutput,
)
from data.market_database import open_market_database
from mcp_server.comparable_resale_tools import query_comparable_resale_transactions

LOOKBACK_MONTHS = 12
RESULT_LIMIT = 100


def _validate_market_filters(con, towns: List[str], flat_types: List[str]) -> None:
    for column, requested in (("town", towns), ("flat_type", flat_types)):
        available = {
            row[0]
            for row in con.execute(f"SELECT DISTINCT {column} FROM resale_transactions").fetchall()
        }
        unknown = set(requested) - available
        if unknown:
            raise ValueError(
                f"Unknown {column} filter(s): {sorted(unknown)}. Available values: {sorted(available)}"
            )


# Town, type and date filters are applied before aggregation. Budget and lease
# thresholds are applied to the resulting medians, never to individual sales.
RESALE_QUERY = """
WITH matching_transactions AS (
    SELECT *, greatest(remaining_lease_months - date_diff('month', month, ?::DATE), 0) / 12.0
        AS lease_years_at_reference
    FROM resale_transactions
    WHERE month BETWEEN ?::DATE AND ?::DATE
      AND (len(?::VARCHAR[]) = 0 OR town IN (SELECT unnest(?::VARCHAR[])))
      AND (len(?::VARCHAR[]) = 0 OR flat_type IN (SELECT unnest(?::VARCHAR[])))
), aggregates AS (
    SELECT block_key, block, street, town, flat_type,
           median(resale_price) AS median_resale_price,
           median(resale_price / floor_area_sqm) AS median_price_per_sqm,
           median(lease_years_at_reference) AS median_remaining_lease_years,
           count(*) AS transaction_count, strftime(max(month), '%Y-%m') AS last_transaction_month
    FROM matching_transactions
    GROUP BY block_key, block, street, town, flat_type
    HAVING median(resale_price) <= ? AND median(lease_years_at_reference) >= ?
)
SELECT a.block, a.street, a.town, l.postal, l.lat, l.lng, a.flat_type,
       a.median_resale_price, a.median_price_per_sqm, a.median_remaining_lease_years,
       a.transaction_count, a.last_transaction_month, count(*) OVER () AS total_matches
FROM aggregates a LEFT JOIN block_locations l USING (block_key)
ORDER BY a.median_resale_price, a.town, a.street, a.block, a.flat_type
LIMIT ?
""".strip()


def query_resale_blocks_by_criteria(
    towns: List[str],
    flat_types: List[str],
    max_price: Annotated[float, Field(ge=0, allow_inf_nan=False)],
    min_lease_years: Annotated[int, Field(ge=0, le=99)],
) -> Dict[str, Any]:
    """Find block/flat-type resale medians over the snapshot's last 12 calendar months.

    The end month excludes the calendar month in progress when data was downloaded.
    Empty town/type lists mean all. Use exact HDB town names and flat types such as
    BISHAN and '4 ROOM'. Budget and lease limits apply after aggregation, not to
    individual sales. Lease is a month-resolution estimate at the window end.
    Returns at most 100 groups, ordered by median price, with truncation metadata.
    Results are historical transaction summaries, not available-flat listings.
    Coordinates come from the existing index and may be unavailable. No API calls.
    """
    args = QueryResaleBlocksInput(
        towns=towns, flat_types=flat_types, max_price=max_price, min_lease_years=min_lease_years
    )
    with open_market_database() as con:
        snapshot, dataset, url, first, latest, end = con.execute(
            "SELECT snapshot_id, dataset_id, source_url, first_month, latest_month, analysis_end_month FROM source_snapshot"
        ).fetchone()
        end_index = end.year * 12 + end.month - 1
        start_index = end_index - LOOKBACK_MONTHS + 1
        start = max(first, date(start_index // 12, start_index % 12 + 1, 1))
        _validate_market_filters(con, args.towns, args.flat_types)
        cursor = con.execute(
            RESALE_QUERY,
            [
                end,
                start,
                end,
                args.towns,
                args.towns,
                args.flat_types,
                args.flat_types,
                Decimal(str(args.max_price)),
                args.min_lease_years,
                RESULT_LIMIT,
            ],
        )
        columns = [item[0] for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    total = rows[0]["total_matches"] if rows else 0
    for row in rows:
        row.pop("total_matches")
    result = QueryResaleBlocksOutput.model_validate(
        {
            "blocks": rows,
            "sql": RESALE_QUERY,
            "metadata": {
                "snapshot_id": snapshot,
                "dataset_id": dataset,
                "source_url": url,
                "window_start_month": start.strftime("%Y-%m"),
                "window_end_month": end.strftime("%Y-%m"),
                "latest_source_month": latest.strftime("%Y-%m"),
                "lease_reference_month": end.strftime("%Y-%m"),
                "total_matching_blocks": total,
                "returned_blocks": len(rows),
                "result_limit": RESULT_LIMIT,
                "truncated": total > len(rows),
                "notes": [
                    "One result per block and flat type; historical transactions, not listings.",
                    "Budget and remaining-lease filters apply to group medians.",
                    "Remaining lease is estimated to the window end at month resolution, not today's date.",
                    "Single-transaction groups are included; inspect transaction_count.",
                    "Null coordinates mean the block did not match the existing spatial index.",
                ],
            },
        }
    )
    return result.model_dump(mode="json")


TOWN_PRICE_QUERY = """
SELECT town, date_trunc('quarter', month)::DATE AS quarter_start,
       median(resale_price) AS median_price,
       median(resale_price / floor_area_sqm) AS median_price_per_sqm,
       count(*) AS transaction_count
FROM resale_transactions
WHERE month BETWEEN ?::DATE AND ?::DATE
  AND (len(?::VARCHAR[]) = 0 OR town IN (SELECT unnest(?::VARCHAR[])))
  AND flat_type = ?
GROUP BY town, quarter_start
ORDER BY town, quarter_start
""".strip()


def _shift_months(month: date, offset: int) -> date:
    index = month.year * 12 + month.month - 1 + offset
    return date(index // 12, index % 12 + 1, 1)


def _quarter_label(start: date) -> str:
    return f"{start.year}Q{(start.month - 1) // 3 + 1}"


def compute_town_price_timeseries(
    towns: List[str],
    flat_type: str,
    lookback_years: Annotated[int, Field(ge=1, le=30, strict=True)] = 3,
) -> Dict[str, Any]:
    """Calculate quarterly resale medians and same-quarter year-on-year changes.

    Select exact HDB towns and one flat type, such as BISHAN and '4 ROOM'. Empty
    towns select all. Return up to lookback_years * 4 calendar quarters ending
    in the snapshot's analysis month; no current-month data or network calls.
    Each point includes transaction count, median price, median individual price
    per sqm, and whether source coverage spans the complete quarter. Missing
    observations are omitted, never filled with zero. Summaries use each town's
    latest complete quarter within the returned window. YoY compares only the
    same complete quarter one year earlier, even when outside the display window;
    missing/incomplete baselines return null with a reason. These are descriptive
    transaction medians, not a mix-adjusted price index or a price forecast.
    """
    args = ComputeTownPriceTimeseriesInput(
        towns=towns, flat_type=flat_type, lookback_years=lookback_years
    )
    with open_market_database() as con:
        snapshot, dataset, url, first, latest, end = con.execute(
            "SELECT snapshot_id, dataset_id, source_url, first_month, latest_month, "
            "analysis_end_month FROM source_snapshot"
        ).fetchone()
        _validate_market_filters(con, args.towns, [args.flat_type])
        selected_towns = args.towns or [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT town FROM resale_transactions ORDER BY town"
            ).fetchall()
        ]
        end_quarter = date(end.year, (end.month - 1) // 3 * 3 + 1, 1)
        requested_start = _shift_months(end_quarter, -(args.lookback_years * 4 - 1) * 3)
        start = max(first, requested_start)
        # Fetch one extra year solely for same-quarter comparisons, not display.
        cursor = con.execute(
            TOWN_PRICE_QUERY,
            [
                max(first, _shift_months(requested_start, -12)),
                end,
                args.towns,
                args.towns,
                args.flat_type,
            ],
        )
        columns = [item[0] for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    by_period = {(row["town"], row["quarter_start"]): row for row in rows}
    series, summaries = [], []
    displayed_by_town: Dict[str, List[dict]] = {}
    for row in rows:
        quarter = row["quarter_start"]
        row["is_complete"] = first <= quarter and _shift_months(quarter, 2) <= end
        if quarter < requested_start:
            continue
        series.append(
            {
                "town": row["town"],
                "quarter": _quarter_label(quarter),
                "median_price": row["median_price"],
                "median_price_per_sqm": row["median_price_per_sqm"],
                "transaction_count": row["transaction_count"],
                "is_complete": row["is_complete"],
            }
        )
        displayed_by_town.setdefault(row["town"], []).append(row)

    without_complete = []
    for town in sorted(displayed_by_town):
        complete = [row for row in displayed_by_town[town] if row["is_complete"]]
        if not complete:
            without_complete.append(town)
            continue
        current = complete[-1]
        baseline_quarter = _shift_months(current["quarter_start"], -12)
        baseline = by_period.get((town, baseline_quarter))
        change, reason = None, None
        if baseline is None:
            reason = "missing_prior_year_quarter"
        elif not baseline["is_complete"]:
            reason = "incomplete_prior_year_quarter"
        else:
            change = float((current["median_price"] / baseline["median_price"] - 1) * 100)
        summaries.append(
            {
                "town": town,
                "latest_median_price": current["median_price"],
                "median_price_per_sqm": current["median_price_per_sqm"],
                "latest_quarter": _quarter_label(current["quarter_start"]),
                "transaction_count": current["transaction_count"],
                "yoy_appreciation_pct": change,
                "yoy_comparison_quarter": _quarter_label(baseline_quarter),
                "yoy_unavailable_reason": reason,
            }
        )

    result = ComputeTownPriceTimeseriesOutput.model_validate(
        {
            "series": series,
            "summary": summaries,
            "metadata": {
                "snapshot_id": snapshot,
                "dataset_id": dataset,
                "source_url": url,
                "flat_type": args.flat_type,
                "lookback_years": args.lookback_years,
                "requested_start_month": requested_start.strftime("%Y-%m"),
                "window_start_month": start.strftime("%Y-%m"),
                "window_end_month": end.strftime("%Y-%m"),
                "first_source_month": first.strftime("%Y-%m"),
                "latest_source_month": latest.strftime("%Y-%m"),
                "history_truncated": first > requested_start,
                "towns_without_data": sorted(set(selected_towns) - displayed_by_town.keys()),
                "towns_without_complete_quarters": without_complete,
                "notes": [
                    "Series span at most four calendar quarters per requested year; missing observations are omitted.",
                    "Quarter completeness describes snapshot date coverage, not transaction volume or reporting finality.",
                    "Summaries use the latest complete observed quarter in the display window for each town.",
                    "YoY is the percentage change in same-quarter median prices, not matched-flat appreciation.",
                    "Prior-year evidence may precede the display window; unavailable comparisons return null.",
                ],
            },
        }
    )
    return result.model_dump(mode="json")


mcp = FastMCP("HDB-Compass-Market-Server")
# Keep the plain Python function importable for ToolRunner and unit tests.
mcp.tool(output_schema=QueryResaleBlocksOutput.model_json_schema())(query_resale_blocks_by_criteria)
mcp.tool(output_schema=ComputeTownPriceTimeseriesOutput.model_json_schema())(
    compute_town_price_timeseries
)
comparable_tool = mcp.tool(output_schema=QueryComparableResaleOutput.model_json_schema())(
    query_comparable_resale_transactions
)
comparable_tool.parameters = QueryComparableResaleInput.model_json_schema()


if __name__ == "__main__":
    mcp.run()
