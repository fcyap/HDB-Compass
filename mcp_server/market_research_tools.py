"""Read-only market research over the local DuckDB resale snapshot (Member 1).

Each tool answers one kind of buyer question from transaction evidence:

  get_block_transaction_history  what has this block sold for?
  get_market_snapshot            where is the market now (latest quarter, 1/3/5-year change)?
  rank_towns_by_value            which town is cheapest for this flat type?
  compare_resale_options         how do these specific blocks compare?
  assess_resale_price            is this asking price in line with nearby sales?

Locations are always resolved from the user's own address text
(mcp_server.location_tools.resolve_place); no tool accepts guessed coordinates.
Results are historical transactions, not listings or valuations.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from agents.research_schemas import (
    AssessResalePriceInput,
    AssessResalePriceOutput,
    CompareResaleOptionsInput,
    CompareResaleOptionsOutput,
    GetBlockTransactionHistoryInput,
    GetBlockTransactionHistoryOutput,
    GetMarketSnapshotInput,
    GetMarketSnapshotOutput,
    LocationCandidate,
    RankTownsByValueInput,
    RankTownsByValueOutput,
)
from agents.schemas import block_key
from data.market_database import open_market_database
from mcp_server.comparable_resale_tools import COMPARABLE_CTE, query_comparable_resale_transactions
from mcp_server.location_tools import CATEGORIES, amenities_near, resolve_place

DEFAULT_SNAPSHOT_TYPES = ["3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
HISTORY_LIMIT = 50
ALL_TOWNS = "ALL TOWNS"


def _shift_months(month: date, offset: int) -> date:
    index = month.year * 12 + month.month - 1 + offset
    return date(index // 12, index % 12 + 1, 1)


def _quarter_label(start: date) -> str:
    return f"{start.year}Q{(start.month - 1) // 3 + 1}"


def _snapshot(con) -> Tuple[str, str, date, date, date]:
    return con.execute(
        "SELECT snapshot_id, source_url, first_month, latest_month, analysis_end_month FROM source_snapshot"
    ).fetchone()


def _metadata(snapshot: str, url: str, start: Optional[date], end: date, latest: date, notes: List[str]) -> Dict[str, Any]:
    return {"snapshot_id": snapshot, "source_url": url,
            "window_start_month": f"{start:%Y-%m}" if start else None,
            "window_end_month": f"{end:%Y-%m}", "latest_source_month": f"{latest:%Y-%m}", "notes": notes}


def _check_values(con, column: str, requested: List[str]) -> None:
    available = {row[0] for row in con.execute(f"SELECT DISTINCT {column} FROM resale_transactions").fetchall()}
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"Unknown {column} value(s): {sorted(unknown)}. Available values: {sorted(available)}")


def _resolve_block(address: str) -> Tuple[str, Optional[LocationCandidate], List[Dict[str, Any]], str]:
    """(status, location, candidates, message); only HDB blocks carry resale history."""
    resolved = resolve_place(address)
    if resolved["status"] != "ok":
        return resolved["status"], None, resolved["candidates"], resolved["message"]
    location = LocationCandidate(**resolved["candidates"][0])
    if location.kind != "hdb_block":
        return ("ambiguous", None, resolved["candidates"],
                f"'{address}' resolved to {location.name}, which is not an HDB block. Ask for the block number and street.")
    return "ok", location, [], resolved["message"]


# ---------------------------------------------------------------------------
# Block sets: an area around a place, or explicit block keys
# ---------------------------------------------------------------------------
AREA_BLOCKS_SQL = """
SELECT * FROM (
    SELECT b.block_key, b.block, b.street, b.town, l.lat, l.lng,
           2 * 6371008.8 * asin(sqrt(least(1.0, greatest(0.0,
               pow(sin(radians(l.lat - $lat) / 2), 2)
               + cos(radians($lat)) * cos(radians(l.lat)) * pow(sin(radians(l.lng - $lng) / 2), 2))))) AS distance_meters
    FROM blocks b JOIN block_locations l USING (block_key)
    WHERE l.lat BETWEEN $lat - $deg AND $lat + $deg
) WHERE distance_meters <= $radius
ORDER BY distance_meters, block_key
""".strip()

KEY_BLOCKS_SQL = """
SELECT b.block_key, b.block, b.street, b.town, l.lat, l.lng, NULL::DOUBLE AS distance_meters
FROM blocks b LEFT JOIN block_locations l USING (block_key)
WHERE b.block_key IN (SELECT unnest(?::VARCHAR[]))
ORDER BY b.block_key
""".strip()


def resolve_block_set(con, area: Optional[Any] = None, block_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Blocks for a set query. Distances match query_comparable_resale_transactions
    (great-circle from the resolved centre to each block's geocode)."""
    if block_keys is not None:
        cursor = con.execute(KEY_BLOCKS_SQL, [block_keys])
        blocks = [dict(zip([c[0] for c in cursor.description], r)) for r in cursor.fetchall()]
        found = {b["block_key"] for b in blocks}
        unknown = [k for k in block_keys if k not in found]
        status = "ok" if blocks else "not_found"
        return {"status": status, "scope": "block_keys", "blocks": blocks, "unknown": unknown, "location": None,
                "candidates": [], "target_key": None, "radius": None,
                "message": f"{len(blocks)} of {len(block_keys)} block key(s) found." if blocks else
                           "None of the block keys are in the resale data; use 'BLOCK STREET', e.g. '277 TOH GUAN RD'."}
    target = None
    if area.address:
        resolved = resolve_place(area.address)
        if resolved["status"] != "ok":
            return {"status": resolved["status"], "scope": "area", "blocks": [], "unknown": [], "location": None,
                    "candidates": resolved["candidates"], "target_key": None, "radius": area.radius_meters,
                    "message": resolved["message"]}
        location = LocationCandidate(**resolved["candidates"][0])
        if location.kind == "hdb_block":
            target = block_key(location.block, location.street)
    else:
        location = LocationCandidate(kind="onemap", name=f"Pinned location ({area.latitude:.5f}, {area.longitude:.5f})",
                                     lat=area.latitude, lng=area.longitude, source="supplied coordinates", score=1.0)
    cursor = con.execute(AREA_BLOCKS_SQL, {"lat": location.lat, "lng": location.lng, "radius": area.radius_meters,
                                           "deg": area.radius_meters / 111_000 + 0.001})
    blocks = [dict(zip([c[0] for c in cursor.description], r)) for r in cursor.fetchall()]
    return {"status": "ok" if blocks else "no_blocks", "scope": "area", "blocks": blocks, "unknown": [],
            "location": location, "candidates": [], "target_key": target, "radius": area.radius_meters,
            "message": f"{len(blocks)} resale block(s) within {area.radius_meters} m of {location.name}."}


def _lease_at(end_param: str = "?::DATE") -> str:
    return f"greatest(remaining_lease_months - date_diff('month', month, {end_param}), 0) / 12.0"


# ---------------------------------------------------------------------------
# Block transaction history
# ---------------------------------------------------------------------------
HISTORY_SQL = """
SELECT strftime(month, '%Y-%m') AS month, block, street, block_key, flat_type, storey_range_raw AS storey_range,
       floor_area_sqm, flat_model, remaining_lease_months / 12.0 AS remaining_lease_years,
       resale_price, resale_price / floor_area_sqm AS price_per_sqm
FROM resale_transactions
WHERE block_key IN (SELECT unnest(?::VARCHAR[])) AND month BETWEEN ?::DATE AND ?::DATE AND (?::VARCHAR IS NULL OR flat_type = ?)
ORDER BY month DESC, resale_price DESC, source_row_number
""".strip()

YEARLY_SQL = """
SELECT year(month) AS year, flat_type, count(*) AS transaction_count,
       median(resale_price) AS median_price, median(resale_price / floor_area_sqm) AS median_price_per_sqm
FROM resale_transactions
WHERE block_key IN (SELECT unnest(?::VARCHAR[])) AND month BETWEEN ?::DATE AND ?::DATE AND (?::VARCHAR IS NULL OR flat_type = ?)
GROUP BY ALL ORDER BY year, flat_type
""".strip()


def get_block_transaction_history(address: Optional[str] = None, flat_type: Optional[str] = None,
                                  lookback_months: int = 60, area: Optional[Dict[str, Any]] = None,
                                  block_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """List resale transactions and per-year medians for one block, an area, or a block set.

    Give exactly one of: address (the user's block and street or postal code, as
    written), area ({address or latitude/longitude, radius_meters}: every resale
    block within the circle) or block_keys (e.g. from find_resale_blocks_near).
    Ambiguous places return candidates rather than a guess. Optional flat_type such
    as '4 ROOM'. Up to 50 most recent sales are listed; yearly medians pool every
    sale in the window. Lease shown is at the time of each sale.
    """
    args = GetBlockTransactionHistoryInput(address=address, area=area, block_keys=block_keys,
                                           flat_type=flat_type, lookback_months=lookback_months)
    distances: Dict[str, Optional[float]] = {}
    if args.address:
        status, location, candidates, message = _resolve_block(args.address)
        if status != "ok":
            return GetBlockTransactionHistoryOutput(status=status, message=message, candidates=candidates).model_dump()
        keys, scope, where = [block_key(location.block, location.street)], "block", f"at {location.name}"
    with open_market_database() as con:
        if not args.address:
            found = resolve_block_set(con, args.area, args.block_keys)
            if found["status"] in ("ambiguous", "not_found") and not found["blocks"]:
                return GetBlockTransactionHistoryOutput(status=found["status"], message=found["message"],
                                                        candidates=found["candidates"], scope=found["scope"],
                                                        block_count=0).model_dump()
            location, scope = found["location"], found["scope"]
            keys = [b["block_key"] for b in found["blocks"]]
            distances = {b["block_key"]: b["distance_meters"] for b in found["blocks"]}
            where = (f"in {len(keys)} block(s) within {found['radius']} m of {location.name}" if scope == "area"
                     else f"in {len(keys)} selected block(s)")
        snapshot, url, first, latest, end = _snapshot(con)
        if args.flat_type:
            _check_values(con, "flat_type", [args.flat_type])
        start = max(first, _shift_months(end, -(args.lookback_months - 1)))
        params = [keys, start, end, args.flat_type, args.flat_type]
        cursor = con.execute(HISTORY_SQL, params)
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
        cursor = con.execute(YEARLY_SQL, params)
        columns = [c[0] for c in cursor.description]
        yearly = [dict(zip(columns, r)) for r in cursor.fetchall()]
    for r in rows:
        r["distance_meters"] = round(distances[r["block_key"]], 1) if distances.get(r["block_key"]) is not None else None
        del r["block_key"]
    notes = ["Historical transactions, not current listings.",
             "Remaining lease is as recorded at each sale.",
             f"Window ends at the snapshot's last complete month ({end:%Y-%m})."]
    if scope != "block":
        notes.append("Yearly medians pool every sale in the set; they shift with the mix of flats sold.")
    if len(rows) > HISTORY_LIMIT:
        notes.append(f"Showing the {HISTORY_LIMIT} most recent of {len(rows)} sales; yearly medians use all of them.")
    return GetBlockTransactionHistoryOutput(
        status="ok" if rows else "no_transactions", scope=scope, block_count=len(keys),
        message=(f"{len(rows)} resale transaction(s) {where} from {start:%Y-%m} to {end:%Y-%m}."
                 if rows else f"No resale transactions {where} in the window."),
        location=location, transaction_count=len(rows), transactions=rows[:HISTORY_LIMIT], yearly=yearly,
        metadata=_metadata(snapshot, url, start, end, latest, notes),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------
SNAPSHOT_SQL = """
SELECT coalesce(town, '{all}') AS town, flat_type, date_trunc('quarter', month)::DATE AS quarter_start,
       median(resale_price) AS median_price, median(resale_price / floor_area_sqm) AS median_price_per_sqm,
       count(*) AS transaction_count
FROM resale_transactions
WHERE date_trunc('quarter', month)::DATE IN (SELECT unnest(?::DATE[]))
  AND flat_type IN (SELECT unnest(?::VARCHAR[]))
GROUP BY GROUPING SETS ((town, flat_type, quarter_start), (flat_type, quarter_start))
""".strip().replace("{all}", ALL_TOWNS)


def _pct(current: float, base: Optional[Dict[str, Any]]) -> Optional[float]:
    return round((current / base["median_price"] - 1) * 100, 2) if base else None


def get_market_snapshot(towns: Optional[List[str]] = None, flat_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Describe where the resale market stands now.

    For the latest complete calendar quarter, return median price, median price
    per sqm and volume island-wide (town 'ALL TOWNS') and for any requested towns,
    with changes versus the previous quarter, one, three and five years earlier
    (same quarter). Empty flat_types covers 3 ROOM, 4 ROOM, 5 ROOM and EXECUTIVE.
    Changes compare transaction medians, not a quality-adjusted price index.
    """
    args = GetMarketSnapshotInput(towns=towns or [], flat_types=flat_types or [])
    with open_market_database() as con:
        snapshot, url, first, latest, end = _snapshot(con)
        _check_values(con, "town", args.towns)
        if args.flat_types:
            _check_values(con, "flat_type", args.flat_types)
            types = args.flat_types
        else:  # defaults cover only the types this snapshot actually contains
            available = {r[0] for r in con.execute("SELECT DISTINCT flat_type FROM resale_transactions").fetchall()}
            types = [t for t in DEFAULT_SNAPSHOT_TYPES if t in available]
        quarter = date(end.year, (end.month - 1) // 3 * 3 + 1, 1)
        if _shift_months(quarter, 2) > end:
            quarter = _shift_months(quarter, -3)
        offsets = {"qoq": -3, "yoy": -12, "3y": -36, "5y": -60}
        quarters = [quarter] + [_shift_months(quarter, o) for o in offsets.values()]
        # Town rows are filtered afterwards so the island-wide row always covers every town.
        cursor = con.execute(SNAPSHOT_SQL, [quarters, types])
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
    by_key = {(r["town"], r["flat_type"], r["quarter_start"]): r for r in rows
              if r["quarter_start"] >= first}  # quarters before source coverage are incomplete
    out = []
    for (town, flat_type, q), r in sorted(by_key.items()):
        if q != quarter or (town != ALL_TOWNS and town not in args.towns):
            continue
        base = {name: by_key.get((town, flat_type, _shift_months(quarter, o))) for name, o in offsets.items()}
        out.append({"town": town, "flat_type": flat_type, "quarter": _quarter_label(quarter),
                    "median_price": r["median_price"], "median_price_per_sqm": r["median_price_per_sqm"],
                    "transaction_count": r["transaction_count"],
                    "qoq_change_pct": _pct(r["median_price"], base["qoq"]),
                    "yoy_change_pct": _pct(r["median_price"], base["yoy"]),
                    "change_3y_pct": _pct(r["median_price"], base["3y"]),
                    "change_5y_pct": _pct(r["median_price"], base["5y"])})
    out.sort(key=lambda r: (r["town"] != ALL_TOWNS, r["town"], r["flat_type"]))
    return GetMarketSnapshotOutput(
        status="ok" if out else "no_data", latest_complete_quarter=_quarter_label(quarter), rows=out,
        metadata=_metadata(snapshot, url, quarters[-1], end, latest, [
            "Latest complete calendar quarter within the snapshot; the in-progress month is excluded.",
            "Changes compare same-quarter transaction medians and are affected by the mix of flats sold.",
            "Null changes mean the comparison quarter is outside source coverage or had no sales.",
        ]),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Town value ranking
# ---------------------------------------------------------------------------
RANK_SQL = """
WITH sales AS (
    SELECT town, resale_price, resale_price / floor_area_sqm AS ppsm,
           greatest(remaining_lease_months - date_diff('month', month, ?::DATE), 0) / 12.0 AS lease_years
    FROM resale_transactions
    WHERE month BETWEEN ?::DATE AND ?::DATE AND flat_type = ?
)
SELECT town, median(resale_price) AS median_price, median(ppsm) AS median_price_per_sqm,
       median(lease_years) AS median_remaining_lease_years, count(*) AS transaction_count,
       CASE WHEN ?::DOUBLE IS NULL THEN NULL ELSE 100.0 * avg((resale_price <= ?::DOUBLE)::INT) END AS share_within_budget_pct
FROM sales WHERE ?::INTEGER IS NULL OR lease_years >= ?::INTEGER
GROUP BY town
""".strip()


def rank_towns_by_value(flat_type: str, max_budget: Optional[float] = None, min_lease_years: Optional[int] = None,
                        lookback_months: int = 12, min_transactions: int = 10,
                        rank_by: str = "price_per_sqm") -> Dict[str, Any]:
    """Rank every HDB town for one flat type, for 'which town is best value/cheapest' questions.

    Uses resale transactions in the last lookback_months (optionally only flats with
    at least min_lease_years remaining). rank_by: price_per_sqm (default, lower first),
    median_price (lower first) or share_within_budget (needs max_budget; higher first).
    Towns with fewer than min_transactions sales are listed separately, not ranked.
    'Value' here means price paid per sqm; it ignores location, condition and amenities.
    """
    args = RankTownsByValueInput(flat_type=flat_type, max_budget=max_budget, min_lease_years=min_lease_years,
                                 lookback_months=lookback_months, min_transactions=min_transactions, rank_by=rank_by)
    if args.rank_by == "share_within_budget" and args.max_budget is None:
        raise ValueError("rank_by='share_within_budget' requires max_budget")
    with open_market_database() as con:
        snapshot, url, first, latest, end = _snapshot(con)
        _check_values(con, "flat_type", [args.flat_type])
        start = max(first, _shift_months(end, -(args.lookback_months - 1)))
        cursor = con.execute(RANK_SQL, [end, start, end, args.flat_type, args.max_budget, args.max_budget,
                                        args.min_lease_years, args.min_lease_years])
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
    ranked = [r for r in rows if r["transaction_count"] >= args.min_transactions]
    excluded = sorted(r["town"] for r in rows if r["transaction_count"] < args.min_transactions)
    sort_key = {"price_per_sqm": lambda r: (r["median_price_per_sqm"], r["town"]),
                "median_price": lambda r: (r["median_price"], r["town"]),
                "share_within_budget": lambda r: (-(r["share_within_budget_pct"] or 0), r["median_price"])}[args.rank_by]
    ranked.sort(key=sort_key)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    basis = {"price_per_sqm": "median price per sqm, lowest first",
             "median_price": "median resale price, lowest first",
             "share_within_budget": "share of sales at or under the budget, highest first"}[args.rank_by]
    return RankTownsByValueOutput(
        status="ok" if ranked else "no_data", flat_type=args.flat_type, ranking_basis=basis, rows=ranked,
        excluded_low_sample_towns=excluded,
        metadata=_metadata(snapshot, url, start, end, latest, [
            f"{args.flat_type} resale transactions from {start:%Y-%m} to {end:%Y-%m}.",
            "Remaining lease is estimated at the window end.",
            "Price per sqm does not account for location, floor level, condition or nearby amenities.",
        ]),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Option comparison and asking-price assessment
# ---------------------------------------------------------------------------
BLOCK_STATS_SQL = """
SELECT count(*) AS transaction_count, median(resale_price) AS median_price,
       median(resale_price / floor_area_sqm) AS median_price_per_sqm, median(floor_area_sqm) AS median_floor_area_sqm,
       median(greatest(remaining_lease_months - date_diff('month', month, ?::DATE), 0) / 12.0) AS median_remaining_lease_years,
       strftime(max(month), '%Y-%m') AS last_transaction_month, arg_max(resale_price, month) AS last_price,
       min(resale_price) AS minimum, quantile_cont(resale_price, 0.25) AS lower_quartile,
       quantile_cont(resale_price, 0.75) AS upper_quartile, max(resale_price) AS maximum
FROM resale_transactions
WHERE block_key = ? AND flat_type = ? AND month BETWEEN ?::DATE AND ?::DATE
""".strip()


def _block_stats(con, key: str, flat_type: str, start: date, end: date) -> Dict[str, Any]:
    cursor = con.execute(BLOCK_STATS_SQL, [end, key, flat_type, start, end])
    return dict(zip([c[0] for c in cursor.description], cursor.fetchone()))


def compare_resale_options(addresses: List[str], flat_type: str, lookback_months: int = 24) -> Dict[str, Any]:
    """Compare 2-4 specific HDB blocks side by side for one flat type.

    Pass the user's addresses as written; each is resolved independently and
    ambiguous or unknown ones are reported for clarification. For each block:
    sales count, median price, price per sqm, floor area, remaining lease (at the
    window end), latest sale, and the nearest MRT/LRT, hawker centre, supermarket
    and primary school (straight-line).
    """
    args = CompareResaleOptionsInput(addresses=addresses, flat_type=flat_type, lookback_months=lookback_months)
    options: List[Dict[str, Any]] = []
    with open_market_database() as con:
        snapshot, url, first, latest, end = _snapshot(con)
        _check_values(con, "flat_type", [args.flat_type])
        start = max(first, _shift_months(end, -(args.lookback_months - 1)))
        for address in args.addresses:
            status, location, candidates, _ = _resolve_block(address)
            if status != "ok":
                options.append({"address": address, "status": status, "candidates": candidates})
                continue
            stats = _block_stats(con, block_key(location.block, location.street), args.flat_type, start, end)
            _, nearest, _ = amenities_near(location.lat, location.lng, 1000, list(CATEGORIES), 1)
            option = {"address": address, "location": location, "nearest": nearest,
                      "status": "ok" if stats["transaction_count"] else "no_transactions"}
            option.update({k: stats[k] for k in ("transaction_count", "median_price", "median_price_per_sqm",
                                                 "median_floor_area_sqm", "median_remaining_lease_years",
                                                 "last_transaction_month", "last_price")})
            options.append(option)
    resolved = sum(o["status"] in ("ok", "no_transactions") for o in options)
    status = "ok" if resolved == len(options) else "partial" if resolved >= 2 else "needs_information"
    return CompareResaleOptionsOutput(
        status=status, flat_type=args.flat_type, options=options,
        metadata=_metadata(snapshot, url, start, end, latest, [
            f"Block statistics use {args.flat_type} sales from {start:%Y-%m} to {end:%Y-%m}.",
            "Nearest amenities are straight-line distances from the block's indexed point.",
            "Options with status ambiguous or not_found need the user to confirm the address.",
        ]),
    ).model_dump(mode="json")


PERCENTILE_SQL = COMPARABLE_CTE + "\nSELECT count(*) FILTER (WHERE resale_price < $asking) AS below, count(*) AS total FROM nearby"


def assess_resale_price(address: str, flat_type: str, asking_price: Optional[float] = None, radius_meters: int = 500,
                        lookback_months: int = 12, floor_area_sqm: Optional[float] = None) -> Dict[str, Any]:
    """Judge whether a resale asking price is in line with recent sales, for 'is $X fair?' questions.

    Resolves the user's address, then compares against (a) the same block's sales
    and (b) all sales of the flat type within radius_meters in the last
    lookback_months (optionally ±10 sqm of a stated floor area). Reports the price
    distribution, the asking price's percentile among nearby sales and its gap to
    the nearby median. This is evidence for negotiation, not an HDB valuation.
    """
    args = AssessResalePriceInput(address=address, flat_type=flat_type, asking_price=asking_price,
                                  radius_meters=radius_meters, lookback_months=lookback_months,
                                  floor_area_sqm=floor_area_sqm)
    status, location, candidates, message = _resolve_block(args.address)
    if status != "ok":
        return AssessResalePriceOutput(status=status, message=message, candidates=candidates,
                                       flat_type=args.flat_type, asking_price=args.asking_price).model_dump()
    area = args.floor_area_sqm
    comparable = query_comparable_resale_transactions(
        latitude=location.lat, longitude=location.lng, flat_type=args.flat_type, radius_meters=args.radius_meters,
        lookback_months=args.lookback_months, sample_limit=1,
        min_floor_area_sqm=area - 10 if area else None, max_floor_area_sqm=area + 10 if area else None)
    criteria, meta = comparable["criteria"], comparable["metadata"]
    start = date.fromisoformat(meta["window_start_month"] + "-01")
    end = date.fromisoformat(meta["window_end_month"] + "-01")
    with open_market_database() as con:
        snapshot, url, first, latest, _ = _snapshot(con)
        block = _block_stats(con, block_key(location.block, location.street), args.flat_type, start, end)
        percentile = None
        if args.asking_price is not None and comparable.get("summary"):
            params = {"start": start, "end": end, "flat_type": args.flat_type,
                      "min_area": criteria["min_floor_area_sqm"], "max_area": criteria["max_floor_area_sqm"],
                      "min_storey": None, "max_storey": None, "min_lease": None, "max_lease": None,
                      "latitude": location.lat, "longitude": location.lng, "radius": args.radius_meters,
                      "asking": args.asking_price}
            below, total = con.execute(PERCENTILE_SQL, params).fetchone()
            percentile = round(100 * below / total, 1) if total else None
    nearby = None
    if comparable.get("summary"):
        s = comparable["summary"]
        nearby = {"transaction_count": s["transaction_count"], **s["price"],
                  "median_price_per_sqm": s["price_per_sqm"]["median"]}
    same_block = None
    if block["transaction_count"]:
        same_block = {k: block[k] for k in ("transaction_count", "minimum", "lower_quartile", "upper_quartile", "maximum")}
        same_block.update(median=block["median_price"], median_price_per_sqm=block["median_price_per_sqm"])
    position = gap = None
    if args.asking_price is not None and nearby:
        gap = round((args.asking_price / nearby["median"] - 1) * 100, 2)
        position = ("below_lower_quartile" if args.asking_price < nearby["lower_quartile"]
                    else "above_upper_quartile" if args.asking_price > nearby["upper_quartile"] else "within_middle_50")
    comp_status = comparable["status"]
    status = {"ok": "ok", "insufficient_sample": "insufficient_sample"}.get(comp_status, "no_matches")
    notes = [f"Nearby means {args.flat_type} sales within {args.radius_meters} m straight-line, "
             f"{meta['window_start_month']} to {meta['window_end_month']}.",
             "Prices vary with floor level, condition, facing and remaining lease; this is not an HDB valuation."]
    if area:
        notes.append(f"Comparables restricted to {area - 10:g}-{area + 10:g} sqm.")
    if meta.get("transactions_without_coordinates"):
        notes.append("Some sales lack coordinates and could not be included.")
    return AssessResalePriceOutput(
        status=status, location=location, flat_type=args.flat_type, same_block=same_block, nearby=nearby,
        asking_price=args.asking_price, asking_percentile=percentile, asking_vs_nearby_median_pct=gap,
        position=position,
        message={"ok": f"Compared with {nearby['transaction_count'] if nearby else 0} nearby sales around {location.name}.",
                 "insufficient_sample": "Few nearby sales match; treat the comparison as limited evidence.",
                 "no_matches": "No nearby sales match these criteria; filters were not widened."}[status],
        metadata=_metadata(snapshot, url, start, end, latest, notes),
    ).model_dump(mode="json")
