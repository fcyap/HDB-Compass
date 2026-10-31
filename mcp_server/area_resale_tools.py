"""Resale research over a set of blocks (mcp_server/area_resale_tools.py). Owned by Member 1.

Composable area questions ("what do flats around Blk 277 Toh Guan Rd sell for?"):

  find_resale_blocks_near     which resale blocks lie within a radius of a place, with per-block medians
  get_area_resale_statistics  pooled price distribution per flat type for an area or a block set
  get_area_price_trend        pooled quarterly medians and period change for an area or a block set

get_block_transaction_history (market_research_tools) also takes an area or block set.
Every set statistic is computed in SQL over the pooled sales, never as a median of
per-block medians. Places are resolved from the user's own text; coordinates are
accepted only from resolve_location, a tool result or the map pin.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from agents.research_schemas import (
    AreaFilter,
    FindResaleBlocksNearInput,
    FindResaleBlocksNearOutput,
    GetAreaPriceTrendInput,
    GetAreaPriceTrendOutput,
    GetAreaResaleStatisticsInput,
    GetAreaResaleStatisticsOutput,
)
from data.market_database import open_market_database
from mcp_server.market_research_tools import (
    _check_values,
    _lease_at,
    _metadata,
    _quarter_label,
    _shift_months,
    _snapshot,
    resolve_block_set,
)

MIN_QUARTER_SALES = 5
_SET = "block_key IN (SELECT unnest(?::VARCHAR[]))"
_FLAT = "(?::VARCHAR IS NULL OR flat_type = ?)"

PER_BLOCK_SQL = f"""
SELECT block_key, count(*) AS transaction_count, median(resale_price) AS median_price,
       median(resale_price / floor_area_sqm) AS median_price_per_sqm,
       median({_lease_at()}) AS median_remaining_lease_years, strftime(max(month), '%Y-%m') AS last_sale_month
FROM resale_transactions
WHERE {_SET} AND month BETWEEN ?::DATE AND ?::DATE AND {_FLAT}
GROUP BY block_key
""".strip()

STATS_SQL = f"""
SELECT flat_type, count(DISTINCT block_key) AS block_count, count(*) AS transaction_count,
       min(resale_price) AS minimum, quantile_cont(resale_price, 0.25) AS lower_quartile,
       median(resale_price) AS median, quantile_cont(resale_price, 0.75) AS upper_quartile,
       max(resale_price) AS maximum, median(resale_price / floor_area_sqm) AS median_price_per_sqm,
       median(floor_area_sqm) AS median_floor_area_sqm, median({_lease_at()}) AS median_remaining_lease_years
FROM resale_transactions
WHERE {_SET} AND month BETWEEN ?::DATE AND ?::DATE AND {_FLAT}
GROUP BY flat_type ORDER BY flat_type
""".strip()

TREND_SQL = f"""
SELECT date_trunc('quarter', month)::DATE AS quarter_start, count(*) AS transaction_count,
       median(resale_price) AS median_price, median(resale_price / floor_area_sqm) AS median_price_per_sqm
FROM resale_transactions
WHERE {_SET} AND flat_type = ? AND month BETWEEN ?::DATE AND ?::DATE
GROUP BY 1 ORDER BY 1
""".strip()


def _rows(cursor) -> List[Dict[str, Any]]:
    columns = [c[0] for c in cursor.description]
    return [{k: float(v) if isinstance(v, Decimal) else v for k, v in zip(columns, r)} for r in cursor.fetchall()]


def _flat_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    dist = {k: row[k] for k in ("transaction_count", "minimum", "lower_quartile", "median", "upper_quartile",
                                "maximum", "median_price_per_sqm")}
    return {"flat_type": row["flat_type"], "block_count": row["block_count"], "distribution": dist,
            "median_floor_area_sqm": row["median_floor_area_sqm"],
            "median_remaining_lease_years": round(row["median_remaining_lease_years"], 1)}


def _scope_failure(model, found: Dict[str, Any], **extra) -> Dict[str, Any]:
    return model(status=found["status"] if found["status"] != "no_blocks" else "no_transactions",
                 message=found["message"], candidates=found["candidates"], location=found["location"],
                 scope=found["scope"], radius_meters=found["radius"], **extra).model_dump(mode="json")


def find_resale_blocks_near(address: Optional[str] = None, latitude: Optional[float] = None,
                            longitude: Optional[float] = None, radius_meters: int = 500,
                            flat_type: Optional[str] = None, lookback_months: int = 12, limit: int = 60) -> Dict[str, Any]:
    """List the HDB resale blocks within radius_meters of a place, nearest first, with each block's sales.

    Pass the user's address or place name as written (block and street, postal code,
    MRT station, school...); coordinates only from resolve_location or the map pin.
    Each block reports sales count, median price, median price per sqm and median
    remaining lease over the last lookback_months (optionally one flat_type). The
    returned block_keys can be passed to get_area_resale_statistics,
    get_area_price_trend or get_block_transaction_history. Historical sales, not listings.
    """
    args = FindResaleBlocksNearInput(address=address, latitude=latitude, longitude=longitude, radius_meters=radius_meters,
                                     flat_type=flat_type, lookback_months=lookback_months, limit=limit)
    area = AreaFilter(address=args.address, latitude=args.latitude, longitude=args.longitude,
                      radius_meters=args.radius_meters)
    with open_market_database() as con:
        found = resolve_block_set(con, area=area)
        if found["status"] in ("ambiguous", "not_found"):
            return FindResaleBlocksNearOutput(status=found["status"], message=found["message"], candidates=found["candidates"],
                                              radius_meters=args.radius_meters, flat_type=args.flat_type).model_dump(mode="json")
        snapshot, url, first, latest, end = _snapshot(con)
        if args.flat_type:
            _check_values(con, "flat_type", [args.flat_type])
        start = max(first, _shift_months(end, -(args.lookback_months - 1)))
        keys = [b["block_key"] for b in found["blocks"]]
        stats = {r["block_key"]: r for r in _rows(con.execute(PER_BLOCK_SQL, [end, keys, start, end, args.flat_type, args.flat_type]))}
    blocks = []
    for b in found["blocks"]:
        s = stats.get(b["block_key"], {})
        blocks.append({**b, "distance_meters": round(b["distance_meters"], 1),
                       "transaction_count": s.get("transaction_count", 0), "median_price": s.get("median_price"),
                       "median_price_per_sqm": s.get("median_price_per_sqm"),
                       "median_remaining_lease_years": round(s["median_remaining_lease_years"], 1) if s else None,
                       "last_sale_month": s.get("last_sale_month"), "is_target": b["block_key"] == found["target_key"]})
    blocks.sort(key=lambda b: (b["transaction_count"] == 0, b["distance_meters"]))
    with_sales = sum(1 for b in blocks if b["transaction_count"])
    total = sum(b["transaction_count"] for b in blocks)
    label = f"{args.flat_type} " if args.flat_type else ""
    notes = ["Straight-line distance from the place to each block's geocode.",
             "Historical transactions, not current listings.",
             f"Lease is the median remaining at {end:%Y-%m}."]
    if len(blocks) > args.limit:
        notes.append(f"Showing {args.limit} of {len(blocks)} blocks (blocks with sales first, nearest first).")
    return FindResaleBlocksNearOutput(
        status="ok" if blocks else "no_blocks",
        message=(f"{len(blocks)} resale block(s) within {args.radius_meters} m of {found['location'].name}; "
                 f"{with_sales} had {total} {label}sale(s) from {start:%Y-%m} to {end:%Y-%m}."
                 if blocks else f"No resale blocks within {args.radius_meters} m of {found['location'].name}."),
        location=found["location"], radius_meters=args.radius_meters, flat_type=args.flat_type,
        block_count=len(blocks), blocks_with_sales=with_sales, transaction_count=total, blocks=blocks[:args.limit],
        metadata=_metadata(snapshot, url, start, end, latest, notes),
    ).model_dump(mode="json")


def get_area_resale_statistics(area: Optional[Dict[str, Any]] = None, block_keys: Optional[List[str]] = None,
                               flat_type: Optional[str] = None, lookback_months: int = 12) -> Dict[str, Any]:
    """Pooled resale price statistics per flat type for an area or a set of blocks.

    Give exactly one of area ({address or latitude/longitude, radius_meters}) or
    block_keys (e.g. from find_resale_blocks_near). For each flat type: sales,
    blocks, price range and quartiles, median price per sqm, floor area and
    remaining lease, pooled over every sale in the last lookback_months. When the
    area's address is an HDB block, that block's own figures are returned
    alongside for comparison. Historical sales, not listings or valuations.
    """
    args = GetAreaResaleStatisticsInput(area=area, block_keys=block_keys, flat_type=flat_type, lookback_months=lookback_months)
    with open_market_database() as con:
        found = resolve_block_set(con, args.area, args.block_keys)
        if found["status"] != "ok":
            return _scope_failure(GetAreaResaleStatisticsOutput, found, unknown_block_keys=found["unknown"])
        snapshot, url, first, latest, end = _snapshot(con)
        if args.flat_type:
            _check_values(con, "flat_type", [args.flat_type])
        start = max(first, _shift_months(end, -(args.lookback_months - 1)))
        keys = [b["block_key"] for b in found["blocks"]]
        rows = _rows(con.execute(STATS_SQL, [end, keys, start, end, args.flat_type, args.flat_type]))
        target = found["target_key"]
        target_rows = _rows(con.execute(STATS_SQL, [end, [target], start, end, args.flat_type, args.flat_type])) if target else []
    where = (f"{len(keys)} block(s) within {found['radius']} m of {found['location'].name}" if found["scope"] == "area"
             else f"{len(keys)} selected block(s)")
    total = sum(r["transaction_count"] for r in rows)
    notes = ["Statistics pool every sale in the set; they are not medians of block medians.",
             f"Lease is the median remaining at {end:%Y-%m}.", "Historical transactions, not current listings."]
    if found["unknown"]:
        notes.append(f"Not in the resale data: {', '.join(found['unknown'])}.")
    return GetAreaResaleStatisticsOutput(
        status="ok" if rows else "no_transactions",
        message=(f"{total} sale(s) in {where} from {start:%Y-%m} to {end:%Y-%m}." if rows
                 else f"No sales in {where} from {start:%Y-%m} to {end:%Y-%m}."),
        location=found["location"], scope=found["scope"], radius_meters=found["radius"], block_count=len(keys),
        unknown_block_keys=found["unknown"], by_flat_type=[_flat_stats(r) for r in rows],
        target_block_key=target, target_by_flat_type=[_flat_stats(r) for r in target_rows],
        metadata=_metadata(snapshot, url, start, end, latest, notes),
    ).model_dump(mode="json")


def get_area_price_trend(flat_type: str, area: Optional[Dict[str, Any]] = None, block_keys: Optional[List[str]] = None,
                         lookback_years: int = 3) -> Dict[str, Any]:
    """Quarterly pooled resale medians for one flat type in an area or a set of blocks.

    Give exactly one of area ({address or latitude/longitude, radius_meters}) or
    block_keys. Returns up to lookback_years * 4 quarters ending at the snapshot's
    last complete month, with sales count, median price and median price per sqm.
    period_change_pct compares the first and last complete quarters that have at
    least 5 sales (thinner quarters are shown but never used as endpoints); the
    per-sqm change is less sensitive to which flat sizes happened to sell.
    """
    args = GetAreaPriceTrendInput(area=area, block_keys=block_keys, flat_type=flat_type, lookback_years=lookback_years)
    with open_market_database() as con:
        found = resolve_block_set(con, args.area, args.block_keys)
        if found["status"] != "ok":
            return _scope_failure(GetAreaPriceTrendOutput, found, flat_type=args.flat_type, min_quarter_sales=MIN_QUARTER_SALES)
        snapshot, url, first, latest, end = _snapshot(con)
        _check_values(con, "flat_type", [args.flat_type])
        end_quarter = date(end.year, (end.month - 1) // 3 * 3 + 1, 1)
        start = max(first, _shift_months(end_quarter, -(args.lookback_years * 4 - 1) * 3))
        keys = [b["block_key"] for b in found["blocks"]]
        rows = _rows(con.execute(TREND_SQL, [keys, args.flat_type, start, end]))
    points = [{"quarter": _quarter_label(r["quarter_start"]), "transaction_count": r["transaction_count"],
               "median_price": r["median_price"], "median_price_per_sqm": r["median_price_per_sqm"],
               "is_complete": first <= r["quarter_start"] and _shift_months(r["quarter_start"], 2) <= end,
               "_start": r["quarter_start"]} for r in rows]
    usable = [p for p in points if p["is_complete"] and p["transaction_count"] >= MIN_QUARTER_SALES]
    change = psm_change = cagr = None
    period_start = period_end = None
    if len(usable) >= 2:
        a, b = usable[0], usable[-1]
        period_start, period_end = a["quarter"], b["quarter"]
        change = round(100 * (b["median_price"] / a["median_price"] - 1), 1)
        psm_change = round(100 * (b["median_price_per_sqm"] / a["median_price_per_sqm"] - 1), 1)
        years = ((b["_start"].year - a["_start"].year) * 12 + b["_start"].month - a["_start"].month) / 12
        cagr = round(100 * ((b["median_price"] / a["median_price"]) ** (1 / years) - 1), 1) if years >= 1 else None
    for p in points:
        del p["_start"]
    where = (f"{len(keys)} block(s) within {found['radius']} m of {found['location'].name}" if found["scope"] == "area"
             else f"{len(keys)} selected block(s)")
    notes = ["Pooled transaction medians, not a mix-adjusted index; small areas are noisy.",
             f"Quarters with fewer than {MIN_QUARTER_SALES} sales are not used as endpoints."]
    if points and len(usable) < 2:
        notes.append("Too few quarters with enough sales to measure a change; widen the radius or the period.")
    return GetAreaPriceTrendOutput(
        status="ok" if points else "no_transactions",
        message=(f"{sum(p['transaction_count'] for p in points)} {args.flat_type} sale(s) in {where}, "
                 f"{points[0]['quarter']} to {points[-1]['quarter']}." if points
                 else f"No {args.flat_type} sales in {where} in the period."),
        location=found["location"], scope=found["scope"], radius_meters=found["radius"], block_count=len(keys),
        flat_type=args.flat_type, points=points, period_start_quarter=period_start, period_end_quarter=period_end,
        period_change_pct=change, period_price_per_sqm_change_pct=psm_change, period_cagr_pct=cagr,
        min_quarter_sales=MIN_QUARTER_SALES, metadata=_metadata(snapshot, url, start, end, latest, notes),
    ).model_dump(mode="json")
