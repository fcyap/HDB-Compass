"""Read-only resale queries over local DuckDB snapshots."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Dict, List

from fastmcp import FastMCP
from pydantic import Field

from agents.schemas import QueryResaleBlocksInput, QueryResaleBlocksOutput
from data.market_database import open_market_database

LOOKBACK_MONTHS = 12
RESULT_LIMIT = 100

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
        for column, requested in (("town", args.towns), ("flat_type", args.flat_types)):
            available = {
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT {column} FROM resale_transactions"
                ).fetchall()
            }
            unknown = set(requested) - available
            if unknown:
                raise ValueError(
                    f"Unknown {column} filter(s): {sorted(unknown)}. Available values: {sorted(available)}"
                )
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


mcp = FastMCP("HDB-Compass-Market-Server")
# Keep the plain Python function importable for ToolRunner and unit tests.
mcp.tool(output_schema=QueryResaleBlocksOutput.model_json_schema())(query_resale_blocks_by_criteria)


if __name__ == "__main__":
    mcp.run()
