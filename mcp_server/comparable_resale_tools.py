"""Read-only transaction comparables with explicit matching and coverage evidence."""

from __future__ import annotations

from datetime import date

from agents.schemas import QueryComparableResaleInput, QueryComparableResaleOutput
from data.market_database import open_market_database

COMPARABLE_CTE = """
WITH dated AS (
    SELECT t.*, greatest(remaining_lease_months - date_diff('month', month, $end::DATE), 0) / 12.0
        AS lease_years_at_reference
    FROM resale_transactions t
    WHERE month BETWEEN $start::DATE AND $end::DATE AND flat_type = $flat_type
      AND ($min_area::DOUBLE IS NULL OR floor_area_sqm >= $min_area)
      AND ($max_area::DOUBLE IS NULL OR floor_area_sqm <= $max_area)
      AND ($min_storey::INTEGER IS NULL OR storey_min >= $min_storey)
      AND ($max_storey::INTEGER IS NULL OR storey_max <= $max_storey)
), located AS (
    SELECT t.*, l.postal, l.lat, l.lng
    FROM dated t LEFT JOIN block_locations l USING (block_key)
    WHERE ($min_lease::DOUBLE IS NULL OR lease_years_at_reference >= $min_lease)
      AND ($max_lease::DOUBLE IS NULL OR lease_years_at_reference <= $max_lease)
), distances AS (
    SELECT *, 2 * 6371008.8 * asin(sqrt(least(1.0, greatest(0.0,
        pow(sin(radians(lat - $latitude) / 2), 2)
        + cos(radians($latitude)) * cos(radians(lat))
        * pow(sin(radians(lng - $longitude) / 2), 2))))) AS distance_meters
    FROM located WHERE lat IS NOT NULL AND lng IS NOT NULL
), nearby AS (
    SELECT * FROM distances WHERE distance_meters <= $radius
)
""".strip()

SUMMARY_SQL = (
    COMPARABLE_CTE
    + """
SELECT count(*) AS transaction_count, count(DISTINCT block_key) AS block_count,
       min(resale_price), quantile_cont(resale_price, 0.25), median(resale_price),
       quantile_cont(resale_price, 0.75), max(resale_price),
       min(resale_price / floor_area_sqm), quantile_cont(resale_price / floor_area_sqm, 0.25),
       median(resale_price / floor_area_sqm), quantile_cont(resale_price / floor_area_sqm, 0.75),
       max(resale_price / floor_area_sqm), median(floor_area_sqm), median(lease_years_at_reference),
       strftime(min(month), '%Y-%m'), strftime(max(month), '%Y-%m'),
       (SELECT count(*) FROM located WHERE lat IS NULL OR lng IS NULL) AS unlocated_count
FROM nearby
"""
)

SAMPLE_SQL = (
    COMPARABLE_CTE
    + """
SELECT source_row_number, strftime(month, '%Y-%m') AS month, block, street, town,
       postal, lat, lng, flat_type, floor_area_sqm, storey_range_raw AS storey_range,
       lease_years_at_reference AS remaining_lease_years, resale_price,
       resale_price / floor_area_sqm AS price_per_sqm, distance_meters
FROM nearby
ORDER BY distance_meters, month DESC, town, street, block, resale_price, source_row_number
LIMIT $sample_limit
"""
)


def query_comparable_resale_transactions(
    latitude: float,
    longitude: float,
    flat_type: str,
    radius_meters: int = 500,
    lookback_months: int = 12,
    end_month: str | None = None,
    min_floor_area_sqm: float | None = None,
    max_floor_area_sqm: float | None = None,
    min_lease_years: float | None = None,
    max_lease_years: float | None = None,
    min_storey: int | None = None,
    max_storey: int | None = None,
    min_sample_size: int = 5,
    sample_limit: int = 20,
) -> dict:
    """Compare historical resale transactions around verified WGS84 coordinates.

    Match flat type, straight-line radius and optional area, lease and storey ranges.
    Supply verified coordinates; never substitute a guessed town centre. Optional
    end_month (YYYY-MM) supports historical launch comparisons and cannot exceed
    the snapshot's last completed analysis month. Lease filters use that end month.
    Storey filters require the entire published storey band to fit. Summary statistics
    use ALL matching sales; only the displayed sample is limited. No automatic widening.
    Low samples, missing geocodes and absent source coverage are explicit. These are
    historical comparisons, not listings, matched-unit valuations or financial approval.
    """
    args = QueryComparableResaleInput(**locals())
    with open_market_database() as con:
        snapshot, dataset, url, retrieved_at, first, latest, analysis_end = con.execute(
            "SELECT snapshot_id, dataset_id, source_url, "
            "strftime(retrieved_at AT TIME ZONE 'UTC', '%Y-%m-%dT%H:%M:%SZ'), first_month, "
            "latest_month, analysis_end_month FROM source_snapshot"
        ).fetchone()
        known_types = {
            row[0]
            for row in con.execute(
                "SELECT DISTINCT flat_type FROM resale_transactions"
            ).fetchall()
        }
        if args.flat_type not in known_types:
            raise ValueError(
                f"Unknown flat_type {args.flat_type!r}. Available values: {sorted(known_types)}"
            )
        end = (
            date.fromisoformat(args.end_month + "-01")
            if args.end_month
            else analysis_end
        )
        if end > analysis_end:
            raise ValueError(
                f"end_month exceeds the snapshot's last completed analysis month {analysis_end:%Y-%m}. Refresh the database or choose an earlier month."
            )
        index = end.year * 12 + end.month - args.lookback_months
        start = date(index // 12, index % 12 + 1, 1)
        params = {
            "start": start,
            "end": end,
            "flat_type": args.flat_type,
            "min_area": args.min_floor_area_sqm,
            "max_area": args.max_floor_area_sqm,
            "min_storey": args.min_storey,
            "max_storey": args.max_storey,
            "min_lease": args.min_lease_years,
            "max_lease": args.max_lease_years,
            "latitude": args.latitude,
            "longitude": args.longitude,
            "radius": args.radius_meters,
        }
        stats = con.execute(SUMMARY_SQL, params).fetchone()
        cursor = con.execute(SAMPLE_SQL, {**params, "sample_limit": args.sample_limit})
        columns = [item[0] for item in cursor.description]
        transactions = [dict(zip(columns, row)) for row in cursor.fetchall()]

    count, blocks = stats[:2]
    distribution_keys = (
        "minimum",
        "lower_quartile",
        "median",
        "upper_quartile",
        "maximum",
    )
    summary = (
        None
        if not count
        else {
            "transaction_count": count,
            "block_count": blocks,
            "price": dict(zip(distribution_keys, stats[2:7])),
            "price_per_sqm": dict(zip(distribution_keys, stats[7:12])),
            "median_floor_area_sqm": stats[12],
            "median_remaining_lease_years": stats[13],
            "first_transaction_month": stats[14],
            "last_transaction_month": stats[15],
        }
    )
    status = (
        "no_coverage"
        if end < first
        else "no_matches"
        if not count
        else "insufficient_sample"
        if count < args.min_sample_size
        else "ok"
    )
    return QueryComparableResaleOutput.model_validate(
        {
            "status": status,
            "criteria": args,
            "summary": summary,
            "transactions": transactions,
            "sql": SUMMARY_SQL,
            "metadata": {
                "snapshot_id": snapshot,
                "dataset_id": dataset,
                "source_url": url,
                "retrieved_at": retrieved_at,
                "first_source_month": f"{first:%Y-%m}",
                "latest_source_month": f"{latest:%Y-%m}",
                "window_start_month": f"{start:%Y-%m}",
                "window_end_month": f"{end:%Y-%m}",
                "lease_reference_month": f"{end:%Y-%m}",
                "history_truncated": start < first,
                "transactions_without_coordinates": stats[16],
                "returned_transactions": len(transactions),
                "sample_truncated": count > len(transactions),
                "notes": [
                    "Statistics use every matching source row, including legitimate duplicate-looking sales; sample_limit only limits displayed transactions.",
                    "Quartiles use continuous linear interpolation; price per sqm statistics use individual transaction ratios.",
                    "Distances are straight-line haversine distances to block coordinates, not walking distances or exact unit locations.",
                    "Missing-coordinate count covers otherwise matching type/date/area/lease/storey sales across the snapshot; their proximity is unknown.",
                    "Lease is estimated at the requested end month. Entire reported storey bands must fit any storey bounds.",
                    "Unspecified area/lease/storey criteria remain unrestricted; filters are never widened automatically.",
                    "Sample threshold is a configurable warning, not a statistical confidence guarantee. No matched-unit valuation or BTO financial eligibility is established.",
                    "Source coverage does not guarantee every transaction has been reported or that the snapshot is current today.",
                ],
            },
        }
    ).model_dump(mode="json")
