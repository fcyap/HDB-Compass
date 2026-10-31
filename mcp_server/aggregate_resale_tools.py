"""Grouped resale statistics (mcp_server/aggregate_resale_tools.py). Owned by Member 1.

The area tools answer "around one place". These answer the broad questions: how do sales near MRT stations compare
with the rest, which towns or stations are dearest, how has a group moved. Two tools:

  query_resale_statistics  count, median, quartiles, price per sqm and change for groups of past sales, grouped by
                           1-2 of distance band, town, station, street, flat type, year, storey band, lease band.
                           Distance comes from mcp_server/block_features.py. Each group carries the call that goes
                           one level deeper (island -> town -> station -> area around the station).
  describe_resale_data     the dimensions, valid values, period and definitions, so a model can plan a call.

Every figure is computed in SQL over the pooled sales (never a median of medians). Wrong values come back as a
structured "invalid_request" result that lists the allowed values. Historical sales, not live listings.
"""
from __future__ import annotations

import difflib
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from agents.research_schemas import (
    STATS_DIMENSIONS,
    STATS_METRICS,
    STATS_NEAR,
    STATS_SORT,
    DescribeResaleDataInput,
    DescribeResaleDataOutput,
    QueryResaleStatisticsInput,
    QueryResaleStatisticsOutput,
)
from data.market_database import open_market_database
from mcp_server.block_features import get_block_features, station_identity
from mcp_server.market_research_tools import _shift_months, _snapshot

DEFAULT_BANDS = [200, 500, 1000]
DEFAULT_RADIUS_M = 500
MIN_WINDOW_SALES = 5          # a median over fewer sales is not used for a change
# near -> (distance column, nearest-station column)
NEAR_COLUMNS = {"mrt": ("mrt_m", "mrt_station"), "mrt_lrt": ("rail_m", "rail_station"), "lrt": ("lrt_m", "lrt_station"),
                "hawker": ("hawker_m", None), "supermarket": ("supermarket_m", None), "school": ("school_m", None)}
NEAR_TEXT = {"mrt": "an MRT station (LRT stations excluded)", "mrt_lrt": "an MRT or LRT station",
             "lrt": "an LRT station", "hawker": "a hawker centre", "supermarket": "a supermarket",
             "school": "a primary school"}
DIMENSION_TEXT = {
    "distance_band": "distance from each block to its nearest amenity (set by near), in bands",
    "town": "HDB town", "station": "the nearest MRT/LRT station (needs near mrt, mrt_lrt or lrt)",
    "street": "street name", "flat_type": "flat type such as 4 ROOM", "year": "calendar year of sale",
    "storey_band": "storey: 1-5, 6-10, 11-15, 16-20, 21-30, 31+",
    "lease_band": "remaining lease at the time of sale: under 60, 60-70, 70-80, 80-90, 90+ years",
}
METRIC_TEXT = {
    "count": "number of sales", "median_price": "median resale price (SGD)", "p25": "25th percentile price",
    "p75": "75th percentile price", "median_psm": "median price per square metre",
    "change_1y_pct": "median of the last 12 months of the period vs the 12 months before it",
    "change_period_pct": "median of the last 12 months vs the first 12 months of a period of 2+ years",
}
STOREY_LABELS = ["1-5", "6-10", "11-15", "16-20", "21-30", "31+"]
STOREY_CASE = "CASE WHEN t.storey_min <= 5 THEN 0 WHEN t.storey_min <= 10 THEN 1 WHEN t.storey_min <= 15 THEN 2 " \
              "WHEN t.storey_min <= 20 THEN 3 WHEN t.storey_min <= 30 THEN 4 ELSE 5 END"
LEASE_LABELS = ["under 60 years", "60-70 years", "70-80 years", "80-90 years", "90+ years"]
LEASE_CASE = "CASE WHEN t.remaining_lease_months < 720 THEN 0 WHEN t.remaining_lease_months < 840 THEN 1 " \
             "WHEN t.remaining_lease_months < 960 THEN 2 WHEN t.remaining_lease_months < 1080 THEN 3 ELSE 4 END"
ORDERED = ("distance_band", "year", "storey_band", "lease_band")


def _invalid(message: str, allowed: Optional[Dict[str, List[str]]] = None, **extra: Any) -> Dict[str, Any]:
    return QueryResaleStatisticsOutput(status="invalid_request", message=message, allowed=allowed,
                                       **extra).model_dump(mode="json")


def band_labels(edges: List[int]) -> List[str]:
    def m(v: int) -> str:
        return f"{v:,}"
    labels = [f"0-{m(edges[0])} m"]
    labels += [f"{m(a)}-{m(b)} m" for a, b in zip(edges, edges[1:])]
    return labels + [f">{m(edges[-1])} m"]


def _norm_flat(value: str) -> str:
    v = " ".join(value.upper().replace("-", " ").split())
    v = re.sub(r"^(\d)\s*(?:RM|RMS|ROOMS?)$", r"\1 ROOM", v)
    return {"EXEC": "EXECUTIVE", "MULTI GENERATION": "MULTI-GENERATION"}.get(v, v)


def find_station(text: str, stations: Dict[str, Dict[str, Any]], prefer: str = "MRT") -> Tuple[Optional[str], List[str]]:
    """(display name, close names). 'Bishan', 'Bishan MRT' and 'BISHAN MRT STATION' are the same station."""
    key, kind, _ = station_identity(text)
    base = key.split("|")[0]
    explicit = bool(re.search(r"\b(?:MRT|LRT)\b", text.upper()))
    matches = [name for name, info in stations.items() if station_identity(name)[0].split("|")[0] == base
               and (not explicit or info["kind"] == kind)]
    if matches:
        matches.sort(key=lambda n: stations[n]["kind"] != prefer)
        return matches[0], []
    close = difflib.get_close_matches(base, sorted({station_identity(n)[0].split("|")[0] for n in stations}), n=5, cutoff=0.6)
    return None, [n for n in sorted(stations) if station_identity(n)[0].split("|")[0] in close][:6]


def _period(args: QueryResaleStatisticsInput, first: date, last: date) -> Tuple[date, date, List[str]]:
    notes: List[str] = []

    def month(text: str) -> date:
        return date(int(text[:4]), int(text[5:7]), 1)
    end = min(max(month(args.end_month), first), last) if args.end_month else last
    if args.end_month and month(args.end_month) > last:
        notes.append(f"Sales are complete only up to {last:%Y-%m}, so the period ends there.")
    if args.start_month:
        start = max(month(args.start_month), first)
    else:
        start = max(first, _shift_months(end, -(12 * (args.lookback_years or 1) - 1)))
    return start, end, notes


def _months(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + b.month - a.month + 1


def query_resale_statistics(group_by: Optional[List[str]] = None, towns: Optional[List[str]] = None,
                            flat_types: Optional[List[str]] = None, station: Optional[str] = None,
                            near: Optional[str] = None, max_distance_m: Optional[int] = None,
                            bands: Optional[List[int]] = None, lookback_years: Optional[int] = None,
                            start_month: Optional[str] = None, end_month: Optional[str] = None,
                            metrics: Optional[List[str]] = None, sort_by: Optional[str] = None, order: str = "desc",
                            top_n: int = 10, include_trend: bool = False, min_sales: int = 10) -> Dict[str, Any]:
    """Grouped statistics of past HDB resale sales: count, median, quartiles, price per sqm, change.

    group_by takes 1-2 of distance_band, town, station, street, flat_type, year, storey_band, lease_band.
    near (mrt, mrt_lrt, lrt, hawker, supermarket, school) sets the distance that distance_band measures and that
    max_distance_m limits; mrt leaves LRT stations out. bands are upper edges in metres (default 200, 500, 1000).
    Period: lookback_years (default 1) or start_month/end_month as YYYY-MM. Each group's `drill` is the call one
    level deeper. Every figure is computed over the pooled sales. Historical sales, not live listings. A wrong
    value returns status 'invalid_request' with the allowed values.
    """
    try:
        args = QueryResaleStatisticsInput(
            group_by=group_by or ["distance_band"], towns=towns or [], flat_types=flat_types or [], station=station,
            near=near, max_distance_m=max_distance_m, bands=bands, lookback_years=lookback_years,
            start_month=start_month, end_month=end_month, metrics=metrics, sort_by=sort_by, order=order,
            top_n=top_n, include_trend=include_trend, min_sales=min_sales)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return _invalid(f"Invalid arguments: {problems}",
                        {"group_by": list(STATS_DIMENSIONS), "near": list(STATS_NEAR), "sort_by": list(STATS_SORT)})

    bad = [g for g in args.group_by if g not in STATS_DIMENSIONS]
    if bad:
        return _invalid(f"Unknown group_by value(s) {bad}.", {"group_by": list(STATS_DIMENSIONS)})
    if len(set(args.group_by)) != len(args.group_by):
        return _invalid("group_by lists the same dimension twice.", {"group_by": list(STATS_DIMENSIONS)})
    if args.near is not None and args.near not in STATS_NEAR:
        return _invalid(f"Unknown near value {args.near!r}.", {"near": list(STATS_NEAR)})
    if args.sort_by is not None and args.sort_by not in STATS_SORT:
        return _invalid(f"Unknown sort_by value {args.sort_by!r}.", {"sort_by": list(STATS_SORT)})
    bad = [m for m in (args.metrics or []) if m not in STATS_METRICS]
    if bad:
        return _invalid(f"Unknown metric(s) {bad}.", {"metrics": list(STATS_METRICS)})
    if args.bands is not None:
        if args.bands != sorted(set(args.bands)) or args.bands[0] < 50 or args.bands[-1] > 3000:
            return _invalid("bands must be increasing upper edges in metres between 50 and 3000, e.g. [200, 500, 1000].")
    if "year" in args.group_by and args.include_trend:
        args = args.model_copy(update={"include_trend": False})

    assumptions: List[str] = []
    notes: List[str] = []
    features = get_block_features()
    with open_market_database() as con:
        snapshot, _, first, latest, last = _snapshot(con)
        for column, requested, norm in (("town", args.towns, lambda v: " ".join(v.upper().split())),
                                        ("flat_type", args.flat_types, _norm_flat)):
            have = [r[0] for r in con.execute(f"SELECT DISTINCT {column} FROM resale_transactions ORDER BY 1").fetchall()]
            wanted = [norm(v) for v in requested]
            unknown = [v for v in wanted if v not in have]
            if unknown:
                return _invalid(f"Unknown {column} value(s) {unknown}.", {column: have})
            if column == "town":
                towns_f = wanted
            else:
                flats_f = wanted

        near_kind = args.near
        station_col_name: Optional[str] = None
        station_name: Optional[str] = None
        if args.station:
            station_name, close = find_station(args.station, features.stations,
                                               "LRT" if near_kind == "lrt" else "MRT")
            if not station_name:
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                return _invalid(f"Unknown station {args.station!r}.{hint}",
                                {"station": sorted(features.stations)[:60]})
            if near_kind is None:
                near_kind = "lrt" if features.stations[station_name]["kind"] == "LRT" else "mrt"
        if near_kind is None and ("distance_band" in args.group_by or "station" in args.group_by):
            near_kind = "mrt"
            assumptions.append("No amenity was named, so distance was measured to the nearest MRT station "
                               "(LRT stations excluded).")
        if "station" in args.group_by and (near_kind not in ("mrt", "mrt_lrt", "lrt")):
            return _invalid("group_by station needs near to be mrt, mrt_lrt or lrt.", {"near": ["mrt", "mrt_lrt", "lrt"]})
        dist_col, station_col = NEAR_COLUMNS[near_kind] if near_kind else (None, None)
        if station_name:
            kind = features.stations[station_name]["kind"]
            station_col_name = ("rail_station" if near_kind == "mrt_lrt" else "lrt_station" if kind == "LRT"
                                else "mrt_station")
            if near_kind == "mrt" and kind == "LRT":
                return _invalid(f"{station_name} is an LRT station; use near lrt or mrt_lrt.", {"near": ["lrt", "mrt_lrt"]})
        max_m = args.max_distance_m
        if max_m is None and ("station" in args.group_by or station_name):
            max_m = DEFAULT_RADIUS_M
            assumptions.append(f"Blocks were counted only within {DEFAULT_RADIUS_M} m of their nearest station "
                               f"(say a different distance to change this).")
        edges = list(args.bands or DEFAULT_BANDS)
        if "distance_band" in args.group_by and max_m is not None and not args.bands:
            edges = [e for e in DEFAULT_BANDS if e < max_m] + [max_m]
        labels_band = band_labels(edges)
        if max_m is not None and "distance_band" in args.group_by:
            labels_band = labels_band[:-1]       # nothing lies beyond the limit
        needs_features = bool(near_kind)

        start, end, period_notes = _period(args, first, last)
        notes += period_notes
        if start > end:
            return _invalid("start_month is after end_month.")
        span = _months(start, end)
        cur_from, prev_from, prev_to = _shift_months(end, -11), _shift_months(end, -23), _shift_months(end, -12)
        want_change = "year" not in args.group_by
        lower = min(start, prev_from) if want_change else start
        in_period = f"t.month >= DATE '{start}'"
        d = lambda v: f"DATE '{v}'"    # noqa: E731

        dim_sql: List[str] = []
        for g in args.group_by:
            if g == "distance_band":
                cases = " ".join(f"WHEN f.{dist_col} <= {e} THEN {i}" for i, e in enumerate(edges))
                dim_sql.append(f"CASE {cases} ELSE {len(edges)} END")
            elif g == "town":
                dim_sql.append("t.town")
            elif g == "station":
                dim_sql.append(f"f.{station_col}")
            elif g == "street":
                dim_sql.append("t.street")
            elif g == "flat_type":
                dim_sql.append("t.flat_type")
            elif g == "year":
                dim_sql.append("year(t.month)")
            elif g == "storey_band":
                dim_sql.append(STOREY_CASE)
            else:
                dim_sql.append(LEASE_CASE)

        where = [f"t.month BETWEEN {d(lower)} AND {d(end)}"]
        params: List[Any] = []
        if towns_f:
            where.append("t.town IN (SELECT unnest(?::VARCHAR[]))")
            params.append(towns_f)
        if flats_f:
            where.append("t.flat_type IN (SELECT unnest(?::VARCHAR[]))")
            params.append(flats_f)
        base_where = " AND ".join(where)
        near_where = []
        if near_kind:
            near_where.append(f"f.{dist_col} IS NOT NULL")
            if max_m is not None:
                near_where.append(f"f.{dist_col} <= {int(max_m)}")
        if station_name:
            near_where.append(f"f.{station_col_name} = ?")
        if "station" in args.group_by:
            near_where.append(f"f.{station_col} IS NOT NULL")
        full_where = " AND ".join([base_where] + near_where)
        station_params = [station_name] if station_name else []
        frm = "resale_transactions t" + (" JOIN block_features f ON f.block_key = t.block_key" if needs_features else "")
        if needs_features:
            features.load_into(con)

        select_dims = ", ".join(f"{e} AS d{i}" for i, e in enumerate(dim_sql))
        group_cols = ", ".join(f"d{i}" for i in range(len(dim_sql)))
        P = in_period
        sql = f"""
            SELECT {select_dims}, count(*) FILTER (WHERE {P}) AS n,
                   median(t.resale_price) FILTER (WHERE {P}) AS med,
                   quantile_cont(t.resale_price, 0.25) FILTER (WHERE {P}) AS p25,
                   quantile_cont(t.resale_price, 0.75) FILTER (WHERE {P}) AS p75,
                   median(t.resale_price / t.floor_area_sqm) FILTER (WHERE {P}) AS psm,
                   count(*) FILTER (WHERE t.month >= {d(cur_from)}) AS cur_n,
                   median(t.resale_price) FILTER (WHERE t.month >= {d(cur_from)}) AS cur_med,
                   count(*) FILTER (WHERE t.month BETWEEN {d(prev_from)} AND {d(prev_to)}) AS prev_n,
                   median(t.resale_price) FILTER (WHERE t.month BETWEEN {d(prev_from)} AND {d(prev_to)}) AS prev_med,
                   count(*) FILTER (WHERE t.month BETWEEN {d(start)} AND {d(_shift_months(start, 11))}) AS first_n,
                   median(t.resale_price) FILTER (WHERE t.month BETWEEN {d(start)} AND {d(_shift_months(start, 11))}) AS first_med
            FROM {frm} WHERE {full_where} GROUP BY {group_cols}
        """
        rows = con.execute(sql, params + station_params).fetchall()
        rows = [r for r in rows if r[len(dim_sql)]]            # n > 0 inside the period

        base_sql = f"""
            SELECT count(*) FILTER (WHERE {P}), median(t.resale_price) FILTER (WHERE {P}),
                   median(t.resale_price / t.floor_area_sqm) FILTER (WHERE {P})
                   {', count(*) FILTER (WHERE ' + P + ' AND f.' + dist_col + ' IS NULL)' if needs_features else ', 0'}
            FROM resale_transactions t {'LEFT JOIN block_features f ON f.block_key = t.block_key' if needs_features else ''}
            WHERE {base_where}
        """
        b = con.execute(base_sql, params).fetchone()

        def trend_for() -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
            tsql = f"""SELECT {select_dims}, year(t.month) AS y, count(*), median(t.resale_price)
                       FROM {frm} WHERE {full_where} AND {P} GROUP BY {group_cols}, y ORDER BY y"""
            out: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
            for r in con.execute(tsql, params + station_params).fetchall():
                k = tuple(str(x) for x in r[:len(dim_sql)])
                out.setdefault(k, []).append({"year": int(r[len(dim_sql)]), "count": int(r[len(dim_sql) + 1]),
                                              "median_price": round(float(r[len(dim_sql) + 2]))})
            return out
        trend = trend_for() if args.include_trend else {}

        map_blocks: List[Dict[str, Any]] = []
        if args.group_by[0] in ("distance_band", "station") and needs_features:
            msql = f"""SELECT t.block_key, t.town, {dim_sql[0]} AS g, count(*), median(t.resale_price)
                       FROM {frm} WHERE {full_where} AND {P} GROUP BY t.block_key, t.town, g HAVING count(*) >= 3
                       ORDER BY count(*) DESC, t.block_key"""
            seen: Dict[str, int] = {}
            for key, town, g, n, med in con.execute(msql, params + station_params).fetchall():
                row = features.row(key)
                label = (labels_band[min(int(g), len(labels_band) - 1)] if args.group_by[0] == "distance_band" else str(g))
                if row and seen.get(label, 0) < 12 and len(map_blocks) < 60:
                    seen[label] = seen.get(label, 0) + 1
                    map_blocks.append({"block_key": key, "town": town, "lat": row["lat"], "lng": row["lng"],
                                       "group": label, "count": int(n), "median_price": round(float(med))})

    # ---- assemble groups --------------------------------------------------------------------------------------
    def label_of(i: int, value: Any) -> str:
        g = args.group_by[i]
        if g == "distance_band":
            return labels_band[min(int(value), len(labels_band) - 1)]
        if g == "storey_band":
            return f"storey {STOREY_LABELS[int(value)]}"
        if g == "lease_band":
            return LEASE_LABELS[int(value)]
        return str(value)

    def pct(cur: Any, prev: Any, n1: int, n2: int) -> Optional[float]:
        if cur is None or prev is None or n1 < MIN_WINDOW_SALES or n2 < MIN_WINDOW_SALES:
            return None
        return round((float(cur) / float(prev) - 1) * 100, 1)

    nd = len(dim_sql)
    groups: List[Dict[str, Any]] = []
    small = 0
    for r in rows:
        if r[nd] < args.min_sales:
            small += 1
            continue
        dvals = list(r[:nd])
        key = {args.group_by[i]: label_of(i, v) for i, v in enumerate(dvals)}
        g = {"key": key, "label": " · ".join(key.values()), "count": int(r[nd]),
             "median_price": round(float(r[nd + 1])), "p25": round(float(r[nd + 2])), "p75": round(float(r[nd + 3])),
             "median_psm": round(float(r[nd + 4]), 1),
             "change_1y_pct": pct(r[nd + 6], r[nd + 8], r[nd + 5], r[nd + 7]) if want_change else None,
             "change_period_pct": (pct(r[nd + 6], r[nd + 10], r[nd + 5], r[nd + 9])
                                   if want_change and span >= 24 else None),
             "_sort": dvals}
        if args.include_trend:
            g["trend"] = trend.get(tuple(str(x) for x in dvals), [])
        groups.append(g)

    sort_by = args.sort_by
    if sort_by is None and args.group_by[0] in ORDERED:
        groups.sort(key=lambda g: tuple(str(x) if not isinstance(x, (int, float)) else x for x in g["_sort"]))
    else:
        field = {"median_price": "median_price", "count": "count", "median_psm": "median_psm",
                 "change": "change_1y_pct"}[sort_by or "median_price"]
        sort_by = sort_by or "median_price"
        have = [g for g in groups if g[field] is not None]
        have.sort(key=lambda g: (g[field], g["label"]), reverse=(args.order == "desc"))
        groups = have + [g for g in groups if g[field] is None]
    total = len(groups)
    shown = groups[:args.top_n]

    flat_one = flats_f[0] if len(flats_f) == 1 else None
    for g in shown:
        g.pop("_sort", None)
        g["lat"], g["lng"] = None, None
        g["drill"] = _drill(g, args, near_kind, max_m, edges, towns_f, flats_f, flat_one, features)
    for g in groups[args.top_n:]:
        g.pop("_sort", None)

    if args.metrics:
        keep = set(args.metrics) | {"count", "key", "label", "lat", "lng", "drill", "trend"}
        for g in shown:
            for k in ("median_price", "p25", "p75", "median_psm", "change_1y_pct", "change_period_pct"):
                if k not in keep:
                    g[k] = None

    if near_kind:
        what = NEAR_TEXT[near_kind]
        definition = (f"Straight-line distance from each block to {what}. Blocks are matched to the nearest one.")
        if max_m is not None and "distance_band" not in args.group_by:
            definition = f"Blocks within {max_m:,} m straight-line of {what}."
    else:
        definition = None
    if near_kind == "mrt":
        notes.append("LRT stations are not counted as MRT stations here.")
    if near_kind == "mrt_lrt":
        notes.append("LRT stations are counted together with MRT stations.")
    if len(period_label := f"{start:%Y-%m}") and span < 12:
        notes.append(f"The period is only {span} month(s); changes need 12-month windows and may be missing.")

    if towns_f:
        scope = "in " + ", ".join(t.title() for t in towns_f)
    else:
        scope = "island-wide"
    flat_text = ", ".join(flats_f) if flats_f else "flat"
    baseline = {"label": f"all {flat_text} sales {scope}" + (", any distance" if near_kind else ""),
                "count": int(b[0] or 0), "median_price": round(float(b[1])) if b[1] is not None else None,
                "median_psm": round(float(b[2]), 1) if b[2] is not None else None}
    if not shown:
        return QueryResaleStatisticsOutput(
            status="no_data", message=f"No group had at least {args.min_sales} sales in {start:%Y-%m} to {end:%Y-%m}.",
            group_by=args.group_by, near=near_kind, max_distance_m=max_m, towns=towns_f, flat_types=flats_f,
            station=station_name, period={"start_month": f"{start:%Y-%m}", "end_month": f"{end:%Y-%m}", "months": span},
            baseline=baseline, excluded_small_groups=small, min_sales=args.min_sales, assumptions=assumptions,
            snapshot_id=snapshot, notes=notes).model_dump(mode="json")
    station_pts = features.stations
    for g in shown:
        if "station" in g["key"] and g["key"]["station"] in station_pts:
            g["lat"], g["lng"] = station_pts[g["key"]["station"]]["lat"], station_pts[g["key"]["station"]]["lng"]
        elif "town" in g["key"] and len(g["key"]) == 1:
            idx = [i for i, t in enumerate(features.towns) if t == g["key"]["town"]]
            if idx:
                g["lat"], g["lng"] = float(features.lat[idx].mean()), float(features.lng[idx].mean())
    unlocated = int(b[3] or 0)
    if needs_features and unlocated:
        notes.append(f"{unlocated:,} sales at blocks without a known location are left out of the distance groups.")
    return QueryResaleStatisticsOutput(
        status="ok",
        message=(f"{total} group(s) of {args.group_by} from {start:%Y-%m} to {end:%Y-%m}; "
                 f"baseline {baseline['count']:,} sales."),
        group_by=args.group_by, near=near_kind, max_distance_m=max_m, bands=edges if "distance_band" in args.group_by else [],
        station=station_name, towns=towns_f, flat_types=flats_f,
        period={"start_month": f"{start:%Y-%m}", "end_month": f"{end:%Y-%m}", "months": span},
        groups=shown, baseline=baseline, total_groups=total, truncated=total > len(shown),
        excluded_small_groups=small, min_sales=args.min_sales, unlocated_sales_excluded=unlocated,
        distance_definition=definition, sort_by=sort_by, order=args.order, include_trend=args.include_trend,
        map_blocks=map_blocks, assumptions=assumptions, snapshot_id=snapshot,
        notes=notes + ["Historical transactions, not current listings."]).model_dump(mode="json")


def _drill(g: Dict[str, Any], args: QueryResaleStatisticsInput, near_kind: Optional[str], max_m: Optional[int],
           edges: List[int], towns: List[str], flats: List[str], flat_one: Optional[str], features) -> Optional[Dict[str, Any]]:
    """The call one level deeper than this group: band -> its towns, town -> its stations, station -> the area."""
    carry = {"flat_types": flats, "lookback_years": args.lookback_years, "start_month": args.start_month,
             "end_month": args.end_month}
    carry = {k: v for k, v in carry.items() if v}
    key = g["key"]
    if len(args.group_by) == 1 and args.group_by[0] == "distance_band":
        labels = band_labels(edges)
        if g["label"] in labels[:-1] and near_kind:
            upper = edges[labels.index(g["label"])]
            return {"tool": "query_resale_statistics",
                    "params": {**carry, "group_by": ["town"], "towns": towns, "near": near_kind, "max_distance_m": upper}}
    if len(args.group_by) == 1 and args.group_by[0] == "town":
        kind = near_kind if near_kind in ("mrt", "mrt_lrt", "lrt") else "mrt"
        return {"tool": "query_resale_statistics",
                "params": {**carry, "group_by": ["station"], "towns": [key["town"]], "near": kind,
                           "max_distance_m": max_m or DEFAULT_RADIUS_M}}
    if len(args.group_by) == 1 and args.group_by[0] == "station":
        return {"tool": "get_area_resale_statistics",
                "params": {"area": {"address": key["station"], "radius_meters": max_m or DEFAULT_RADIUS_M},
                           "flat_type": flat_one, "lookback_months": 12 * (args.lookback_years or 1)}}
    return None


# ---------------------------------------------------------------------------
# describe_resale_data
# ---------------------------------------------------------------------------
def describe_resale_data(include_stations: bool = True) -> Dict[str, Any]:
    """Describe what the resale data holds and the valid values for query_resale_statistics.

    Lists the dimensions, valid towns and flat types, MRT and LRT station names, the amenity types distance can be
    measured to, the period covered, the metrics and how each is defined. Call it before building a grouped query
    when unsure of a name. Historical sales, not live listings.
    """
    args = DescribeResaleDataInput(include_stations=include_stations)
    features = get_block_features()
    with open_market_database() as con:
        snapshot, _, first, latest, last = _snapshot(con)
        towns = [r[0] for r in con.execute("SELECT DISTINCT town FROM resale_transactions ORDER BY 1").fetchall()]
        flats = [r[0] for r in con.execute("SELECT DISTINCT flat_type FROM resale_transactions ORDER BY 1").fetchall()]
    mrt = sorted(n for n, s in features.stations.items() if s["kind"] == "MRT") if args.include_stations else []
    lrt = sorted(n for n, s in features.stations.items() if s["kind"] == "LRT") if args.include_stations else []
    return DescribeResaleDataOutput(
        snapshot_id=snapshot, period_covered={"first_month": f"{first:%Y-%m}", "last_complete_month": f"{last:%Y-%m}"},
        dimensions=DIMENSION_TEXT, towns=towns, flat_types=flats,
        near_options={"mrt": "nearest MRT station, LRT excluded", "mrt_lrt": "nearest MRT or LRT station",
                      "lrt": "nearest LRT station", "hawker": "nearest hawker centre",
                      "supermarket": "nearest supermarket", "school": "nearest primary school"},
        mrt_stations=mrt, lrt_stations=lrt, metrics=METRIC_TEXT,
        definitions=["Past resale transactions only; there are no live listings.",
                     "Distances are straight-line metres from the block's geocode to the amenity.",
                     "Medians and quartiles are over all pooled sales in the group, not an average of block medians.",
                     "Default period: the last 12 complete months. Groups with fewer than min_sales (10) sales are left out.",
                     "Interchange stations are merged into one station."],
        example_calls=[
            {"question": "Resale prices near MRT stations vs far from them",
             "call": {"tool": "query_resale_statistics", "params": {"group_by": ["distance_band"], "near": "mrt"}}},
            {"question": "Which MRT stations have the dearest 4-room flats nearby",
             "call": {"tool": "query_resale_statistics",
                      "params": {"group_by": ["station"], "near": "mrt", "max_distance_m": 500,
                                 "flat_types": ["4 ROOM"], "sort_by": "median_price"}}},
        ]).model_dump(mode="json")
