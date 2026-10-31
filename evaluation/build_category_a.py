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
    dict(id="A2-01", query="Which MRT stations have the most expensive resale flats nearby?", years=1,
         params={"group_by": ["station"], "near": "mrt", "max_distance_m": 500, "sort_by": "median_price", "order": "desc"}),
    dict(id="A2-02", query="Show me 4-room resale prices by town near MRT stations over the last 5 years", years=5, flat="4 ROOM",
         params={"group_by": ["town"], "near": "mrt", "max_distance_m": 500, "flat_types": ["4 ROOM"], "lookback_years": 5,
                 "include_trend": True, "sort_by": "median_price", "order": "desc"}),
    dict(id="A2-03", query="Which towns have the cheapest flats near MRT stations?", years=1,
         params={"group_by": ["town"], "near": "mrt", "max_distance_m": 500, "sort_by": "median_price", "order": "asc"}),
    dict(id="A3-01", query="Sales statistics across Singapore", years=1,
         params={"group_by": ["town"], "sort_by": "median_price", "order": "desc"}),
]


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


def _prices(con, keys: List[str], start: str, end: str, flat: str | None) -> List[float]:
    sql = "SELECT resale_price FROM resale_transactions WHERE block_key IN (SELECT unnest(?::VARCHAR[])) AND month BETWEEN ? AND ?"
    params: List[Any] = [keys, start, end]
    if flat:
        sql += " AND flat_type = ?"
        params.append(flat)
    return [float(r[0]) for r in con.execute(sql, params).fetchall()]


def truth(con, loaded: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    p, flat = spec["params"], spec.get("flat")
    start, end = _period(con, spec["years"])
    near = p.get("near")
    groups: List[Dict[str, Any]] = []
    if p["group_by"] == ["distance_band"]:
        dist = loaded[near][0]
        edges = p.get("bands") or BANDS
        bounds = [(0, edges[0])] + list(zip(edges, edges[1:])) + [(edges[-1], None)]
        for lo, hi in bounds:
            keys = [k for k, d in dist.items() if (d > lo or lo == 0) and (hi is None or d <= hi)]
            prices = _prices(con, keys, start, end, flat)
            label = f"0-{hi:,} m" if lo == 0 else f">{lo:,} m" if hi is None else f"{lo:,}-{hi:,} m"
            if len(prices) >= MIN_SALES:
                groups.append({"label": label, "count": len(prices), "median_price": statistics.median(prices)})
    else:
        town_of = dict(con.execute("SELECT block_key, town FROM blocks").fetchall())
        by: Dict[str, List[str]] = {}
        if p["group_by"] == ["station"]:
            dist, station = loaded["mrt"]
            for k, d in dist.items():
                if d <= p["max_distance_m"]:
                    by.setdefault(station[k], []).append(k)
        else:
            dist = loaded["mrt"][0] if near else None
            for k in loaded["keys"] if near else town_of:
                if near is None or dist[k] <= p["max_distance_m"]:
                    by.setdefault(town_of[k], []).append(k)
        for name, keys in by.items():
            prices = _prices(con, keys, start, end, flat)
            if len(prices) >= MIN_SALES:
                groups.append({"label": name, "count": len(prices), "median_price": statistics.median(prices)})
        groups.sort(key=lambda g: (g["median_price"], g["label"]), reverse=p.get("order", "desc") == "desc")
        groups = groups[:3]
    return {"period": f"{start[:7]}..{end[:7]}", "groups": groups}


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
