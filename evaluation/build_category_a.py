"""Category A seed: broad market questions with aggregate answers (evaluation/build_category_a.py). Owned by Member 1.

Builds evaluation/category_a_queries.json in the same format as the other categories. The ground truth is worked out
here from the raw tables with plain SQL and a brute-force distance search, without importing query_resale_statistics,
mcp_server/block_features.py or any agent code, so a test can hold the tool to it. Only the expected tool call
(the arguments the question should produce) is written from the spec below.

    .venv/bin/python -m evaluation.build_category_a            # write evaluation/category_a_queries.json
    .venv/bin/python -m evaluation.build_category_a --check    # compare the file with the current database, write nothing

Medians depend on the database snapshot recorded in each item; rebuild after `python -m data.build_market_db --refresh`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.market_database import open_market_database  # noqa: E402

OUT = PROJECT_ROOT / "evaluation" / "category_a_queries.json"
MIN_SALES = 10
BANDS = [200, 500, 1000]

SPECS: List[Dict[str, Any]] = [
    dict(id="A1-01", query="What are the historical statistics of HDBs sold near MRT stations?", years=10,
         params={"group_by": ["distance_band"], "near": "mrt", "lookback_years": 10, "include_trend": True}),
    dict(id="A1-02", query="How do resale prices compare for flats near MRT vs far from MRT?", years=1,
         params={"group_by": ["distance_band"], "near": "mrt"}),
    dict(id="A1-03", query="Resale prices near hawker centres", years=1,
         params={"group_by": ["distance_band"], "near": "hawker"}),
    dict(id="A1-04", query="Resale prices within 300m of an MRT", years=1,
         params={"group_by": ["distance_band"], "near": "mrt", "bands": [300]}),
    dict(id="A1-05", query="What's the price premium for 5-room flats close to MRT in the east?", years=1,
         params={"group_by": ["distance_band"], "towns": ["BEDOK", "PASIR RIS", "TAMPINES"], "flat_types": ["5 ROOM"],
                 "near": "mrt"}),
    dict(id="A2-01", query="Which MRT stations have the most expensive resale flats nearby?", years=1,
         params={"group_by": ["station"], "near": "mrt", "max_distance_m": 500, "sort_by": "median_price", "order": "desc"}),
    dict(id="A2-02", query="Show me 4-room resale prices by town near MRT stations over the last 5 years", years=5, flat="4 ROOM",
         params={"group_by": ["town"], "near": "mrt", "max_distance_m": 500, "flat_types": ["4 ROOM"], "lookback_years": 5,
                 "include_trend": True, "sort_by": "median_price", "order": "desc"}),
    dict(id="A2-03", query="Which towns have the cheapest flats near MRT stations?", years=1,
         params={"group_by": ["town"], "near": "mrt", "max_distance_m": 500, "sort_by": "median_price", "order": "asc"}),
    dict(id="A2-04", query="Resale statistics near MRT stations in the north-east region", years=1,
         params={"group_by": ["station"], "towns": ["ANG MO KIO", "HOUGANG", "PUNGGOL", "SENGKANG", "SERANGOON"],
                 "near": "mrt", "max_distance_m": 500, "sort_by": "median_price", "order": "desc"}),
    dict(id="A2-05", query="Resale prices within 1km of MRT stations by town", years=1,
         params={"group_by": ["town"], "near": "mrt", "max_distance_m": 1000, "sort_by": "median_price", "order": "desc"}),
    dict(id="A3-01", query="Sales statistics across Singapore", years=1,
         params={"group_by": ["town"], "sort_by": "median_price", "order": "desc"}),
    dict(id="A3-02", query="Resale prices by storey in Tampines", years=1,
         params={"group_by": ["storey_band"], "towns": ["TAMPINES"]}),
    dict(id="A3-03", query="Average resale price by year over the last 5 years", years=5,
         params={"group_by": ["year"], "lookback_years": 5, "include_trend": True}),
    dict(id="A3-04", query="Which flat types sell for the most per square metre?", years=1,
         params={"group_by": ["flat_type"], "sort_by": "median_psm", "order": "desc"}),
    dict(id="A3-05", query="Resale prices by remaining lease in the east", years=1,
         params={"group_by": ["lease_band"], "towns": ["BEDOK", "PASIR RIS", "TAMPINES"]}),
]

# Band definitions restated from the tool's documented labels (storey = lowest storey of the range; lease = months left).
STOREY_BANDS = [("storey 1-5", 0, 5), ("storey 6-10", 6, 10), ("storey 11-15", 11, 15), ("storey 16-20", 16, 20),
                ("storey 21-30", 21, 30), ("storey 31+", 31, 999)]
LEASE_BANDS = [("under 60 years", 0, 720), ("60-70 years", 720, 840), ("70-80 years", 840, 960),
               ("80-90 years", 960, 1080), ("90+ years", 1080, 10 ** 6)]


def _station_key(name: str) -> Tuple[str, bool]:
    up = " ".join(re.sub(r"[^A-Z0-9 ]", " ", name.upper()).split())
    lrt = " LRT " in f" {up} "
    core = " ".join(w for w in up.split() if w not in ("MRT", "LRT", "STATION", "INTERCHANGE"))
    return core.title() + (" LRT" if lrt else " MRT"), lrt


def _load(con) -> Dict[str, Any]:
    index = pickle.loads((PROJECT_ROOT / "data" / "spatial_index.pkl").read_bytes())["payload"]
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)
    rows = con.execute("SELECT b.block_key, b.town, l.lat, l.lng FROM blocks b JOIN block_locations l USING (block_key)").fetchall()
    keys = [r[0] for r in rows]
    bx, by = to_xy.transform([r[3] for r in rows], [r[2] for r in rows])
    blocks = np.column_stack([bx, by])

    def nearest(points: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
        pts = np.array([[p["x"], p["y"]] for p in points], dtype=float)
        d = np.sqrt(((blocks[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        return d.min(axis=1), list(d.argmin(axis=1))

    out: Dict[str, Any] = {"keys": keys}
    mrt_points = [a for a in index["amenities"] if a["category"] == "mrt" and not _station_key(a["name"])[1]]
    d, idx = nearest(mrt_points)
    out["mrt"] = (dict(zip(keys, d)), {k: _station_key(mrt_points[i]["name"])[0] for k, i in zip(keys, idx)})
    hawkers = [a for a in index["amenities"] if a["category"] == "hawker"]
    d, _ = nearest(hawkers)
    out["hawker"] = (dict(zip(keys, d)), {})
    return out


def _period(con, years: int) -> Tuple[str, str]:
    first, last = con.execute("SELECT first_month, analysis_end_month FROM source_snapshot").fetchone()
    y, m = last.year, last.month - (12 * years - 1)
    while m < 1:
        y, m = y - 1, m + 12
    start = max(first, type(first)(y, m, 1))
    return f"{start:%Y-%m-%d}", f"{last:%Y-%m-%d}"


def _sales(con, keys: List[str] | None, start: str, end: str, flats: List[str], towns: List[str]) -> List[Dict[str, Any]]:
    """Raw sales in the period as dicts: price, price per sqm, town, flat type, lowest storey, months of lease left, year."""
    sql = ("SELECT resale_price, resale_price / floor_area_sqm, town, flat_type, storey_min, remaining_lease_months, year(month) "
           "FROM resale_transactions WHERE month BETWEEN ? AND ?")
    params: List[Any] = [start, end]
    if keys is not None:
        sql += " AND block_key IN (SELECT unnest(?::VARCHAR[]))"
        params.append(keys)
    if flats:
        sql += " AND flat_type IN (SELECT unnest(?::VARCHAR[]))"
        params.append(flats)
    if towns:
        sql += " AND town IN (SELECT unnest(?::VARCHAR[]))"
        params.append(towns)
    cols = ("price", "psm", "town", "flat_type", "storey", "lease", "year")
    return [dict(zip(cols, (float(r[0]), float(r[1])) + tuple(r[2:]))) for r in con.execute(sql, params).fetchall()]


def _group(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any] | None:
    if len(rows) < MIN_SALES:
        return None
    return {"label": label, "count": len(rows), "median_price": statistics.median(r["price"] for r in rows),
            "median_psm": round(statistics.median(r["psm"] for r in rows), 1)}


def truth(con, loaded: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    p = spec["params"]
    flats, towns_f = p.get("flat_types", []), p.get("towns", [])
    start, end = _period(con, spec["years"])
    near, by_dim = p.get("near"), p["group_by"][0]
    town_of = dict(con.execute("SELECT block_key, town FROM blocks").fetchall())
    allowed = [k for k in loaded["keys"] if not towns_f or town_of[k] in towns_f]
    groups: List[Dict[str, Any]] = []
    if by_dim == "distance_band":
        dist = loaded[near][0]
        edges = p.get("bands") or BANDS
        bounds = [(0, edges[0])] + list(zip(edges, edges[1:])) + [(edges[-1], None)]
        for lo, hi in bounds:
            keys = [k for k in allowed if (dist[k] > lo or lo == 0) and (hi is None or dist[k] <= hi)]
            label = f"0-{hi:,} m" if lo == 0 else f">{lo:,} m" if hi is None else f"{lo:,}-{hi:,} m"
            g = _group(_sales(con, keys, start, end, flats, towns_f), label)
            if g:
                groups.append(g)
    elif by_dim in ("station", "town"):
        by: Dict[str, List[str]] = {}
        if by_dim == "station":
            dist, station = loaded["mrt"]
            for k in allowed:
                if dist[k] <= p["max_distance_m"]:
                    by.setdefault(station[k], []).append(k)
        else:
            dist = loaded["mrt"][0] if near else None
            for k in (allowed if near else town_of):
                if near is None or dist[k] <= p["max_distance_m"]:
                    by.setdefault(town_of[k], []).append(k)
        for name, keys in by.items():
            g = _group(_sales(con, keys, start, end, flats, towns_f), name)
            if g:
                groups.append(g)
        groups.sort(key=lambda g: (g["median_price"], g["label"]), reverse=p.get("order", "desc") == "desc")
        groups = groups[:3]
    else:
        rows = _sales(con, None, start, end, flats, towns_f)
        if by_dim == "storey_band":
            buckets = [(n, [r for r in rows if lo <= r["storey"] <= hi]) for n, lo, hi in STOREY_BANDS]
        elif by_dim == "lease_band":
            buckets = [(n, [r for r in rows if lo <= r["lease"] < hi]) for n, lo, hi in LEASE_BANDS]
        elif by_dim == "year":
            buckets = [(str(y), [r for r in rows if r["year"] == y]) for y in sorted({r["year"] for r in rows})]
        else:                                                   # flat_type
            buckets = [(t, [r for r in rows if r["flat_type"] == t]) for t in sorted({r["flat_type"] for r in rows})]
        groups = [g for g in (_group(rs, n) for n, rs in buckets) if g]
        if p.get("sort_by") == "median_psm":
            groups.sort(key=lambda g: (g["median_psm"], g["label"]), reverse=p.get("order", "desc") == "desc")
    return {"period": f"{start[:7]}..{end[:7]}", "groups": groups[:10]}


def build() -> List[Dict[str, Any]]:
    items = []
    with open_market_database() as con:
        snapshot = con.execute("SELECT snapshot_id FROM source_snapshot").fetchone()[0]
        loaded = _load(con)
        for spec in SPECS:
            facts = truth(con, loaded, spec)
            items.append({
                "id": spec["id"], "category": "A", "sub_use_case": spec["id"][:4], "query": spec["query"],
                "initial_map_state": None, "expected_agents": ["market_sql"],
                "expected_mcp_calls": [{"agent": "market_sql", "tool": "query_resale_statistics", "required_params": spec["params"]}],
                "ground_truth_facts": {"snapshot_id": snapshot, "min_sales_per_group": MIN_SALES, **facts},
            })
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    built = build()
    if args.check:
        same = json.loads(OUT.read_text(encoding="utf-8")) == json.loads(json.dumps(built))
        print("category_a_queries.json matches the database" if same else "category_a_queries.json is out of date")
        sys.exit(0 if same else 1)
    OUT.write_text(json.dumps(built, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(built)} questions to {OUT}")
