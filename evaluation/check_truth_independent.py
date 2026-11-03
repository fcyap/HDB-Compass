"""
Independent check of the stored expected blocks for Category E and B (Member 4).

The stored answers (evaluation/ground_category_e.py, evaluation/build_category_b.py) are built with the same tools and
the same join (agents.supervisor.compute_intersection) that Variant C uses, so a fault there would be in both the
answer and its expected answer. This script rebuilds each expected set without any of that code:
  * prices and leases: the raw sales rows in the market database, grouped and medianed here in Python
    (12 calendar months ending at the snapshot's analysis month; lease counted down to that month);
  * distances: straight-line SVY21 distances computed here from the raw block, station, hawker, supermarket and
    school coordinates in the spatial index file;
  * the join: block and street, normalised here, then the price cap.
It uses the same definitions the questions use (median over the window, straight-line distance, MRT including LRT
stops, a school's 1 km as 1,000 m from its point), so a difference means an implementation fault, not a definition.
A difference already recorded in architecture_metrics.KNOWN_WRONG_BLOCK_TRUTH (scored with the correction) is
reported as known; any other difference fails the check.
Commute limits (B3) are an estimate with no independent source here: for B3 the stored blocks are checked to meet
every other limit, and the commute part is reported as not checked.

    .venv/bin/python -m evaluation.check_truth_independent          # all E and B files; exit 1 on any mismatch
    .venv/bin/python -m evaluation.check_truth_independent --json   # one JSON line per question
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.market_database import open_market_database  # noqa: E402  (opens the file only; no query logic)
from evaluation.architecture_metrics import KNOWN_WRONG_BLOCK_TRUTH, expected_block_keys  # noqa: E402

FILES = ["category_e_queries.json", "category_e_extra.json", "category_e_varied.json", "category_b_queries.json"]
SPATIAL_INDEX = ROOT / "data" / "spatial_index.pkl"
WINDOW_MONTHS = 12


def norm_key(block: Any, street: Any) -> str:
    """'288b', 'Jurong East St 21 ' -> '288B JURONG EAST ST 21'."""
    return " ".join(f"{block} {street}".upper().split())


def norm_stored(key: str) -> str:
    return " ".join(str(key).upper().split())


def month_index(d: date) -> int:
    return d.year * 12 + d.month - 1


def window(end: date, first: date) -> Tuple[date, date]:
    i = month_index(end) - WINDOW_MONTHS + 1
    return max(first, date(i // 12, i % 12 + 1, 1)), end


# ---------------------------------------------------------------------------
# Prices and leases from the raw sales
# ---------------------------------------------------------------------------
def price_groups(rows: Iterable[Tuple[Any, ...]], ref: date) -> Dict[str, Dict[str, float]]:
    """Raw sales (block, street, month, price, remaining_lease_months at sale) -> per block: median price and
    median lease left at the reference month."""
    prices: Dict[str, List[float]] = {}
    leases: Dict[str, List[float]] = {}
    for block, street, month, price, lease_months in rows:
        k = norm_key(block, street)
        elapsed = month_index(ref) - month_index(month)
        prices.setdefault(k, []).append(float(price))
        leases.setdefault(k, []).append(max(int(lease_months) - elapsed, 0) / 12.0)
    return {k: {"price": statistics.median(prices[k]), "lease": statistics.median(leases[k])} for k in prices}


def within_price_and_lease(groups: Dict[str, Dict[str, float]], max_price: float, min_lease: float) -> Set[str]:
    return {k for k, g in groups.items() if g["price"] <= max_price + 1e-6 and g["lease"] >= min_lease - 1e-9}


# ---------------------------------------------------------------------------
# Distances from the raw coordinates
# ---------------------------------------------------------------------------
def _is_lrt(a: Dict[str, Any]) -> bool:
    return str(a.get("rail_type") or "").upper() == "LRT" or " LRT " in f" {str(a.get('name') or '').upper()} "


def amenity_points(amenities: List[Dict[str, Any]], kind: str) -> List[Tuple[float, float]]:
    kind = kind.lower()
    if kind in ("mrt", "mrt_lrt"):
        pool = [a for a in amenities if a["category"].lower() in ("mrt", "lrt")]
    elif kind == "mrt_only":
        pool = [a for a in amenities if a["category"].lower() == "mrt" and not _is_lrt(a)]
    elif kind == "lrt":
        pool = [a for a in amenities if a["category"].lower() in ("mrt", "lrt") and _is_lrt(a)]
    else:
        pool = [a for a in amenities if a["category"].lower() == kind]
    return [(float(a["x"]), float(a["y"])) for a in pool]


def nearest(x: float, y: float, points: List[Tuple[float, float]]) -> float:
    return min((math.hypot(px - x, py - y) for px, py in points), default=math.inf)


def blocks_near(blocks: List[Dict[str, Any]], towns: List[str], radii: Dict[str, int],
                amenities: List[Dict[str, Any]]) -> Set[str]:
    """Blocks in the towns whose nearest amenity of every kind is within its limit."""
    pools = {k: amenity_points(amenities, k) for k in radii}
    want = {t.upper() for t in towns}
    out = set()
    for b in blocks:
        if want and b["town"].upper() not in want:
            continue
        if all(nearest(float(b["x"]), float(b["y"]), pools[k]) <= r for k, r in radii.items()):
            out.add(norm_key(b["block"], b["street"]))
    return out


def find_school(schools: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    want = re.sub(r"[^A-Z0-9 ]", " ", name.upper()).split()
    for s in schools:
        have = re.sub(r"[^A-Z0-9 ]", " ", f"{s['name']} {s.get('alias') or ''}".upper()).split()
        if all(w in have for w in want):
            return s
    return None


def blocks_near_school(blocks: List[Dict[str, Any]], school: Dict[str, Any], metres: float = 1000.0) -> Set[str]:
    sx, sy = float(school["x"]), float(school["y"])
    return {norm_key(b["block"], b["street"]) for b in blocks if math.hypot(float(b["x"]) - sx, float(b["y"]) - sy) <= metres}


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------
def check_item(item: Dict[str, Any], sales_for, spatial: Dict[str, Any]) -> Dict[str, Any]:
    c = item["stated_constraints"]
    truth = item["ground_truth_facts"]
    raw = {norm_stored(k) for k in truth["matching_block_keys"]}
    stored = {norm_stored(k) for k in expected_block_keys(item)}      # with any recorded correction
    towns, flat_types = c.get("towns") or [], c.get("flat_types") or []
    cap = truth.get("affordability_cap")
    max_price = min(float(c["max_budget_sgd"]), float(cap)) if cap is not None else float(c["max_budget_sgd"])
    priced = within_price_and_lease(sales_for(towns, flat_types), max_price, float(c.get("min_remaining_lease_years") or 0))

    blocks, amenities = spatial["blocks"], spatial["amenities"]
    if c.get("school_name"):
        school = find_school(spatial["schools"], c["school_name"])
        if school is None:
            return {"id": item["id"], "status": "error", "note": f"school not found: {c['school_name']}"}
        near = blocks_near_school(blocks, school)
    else:
        near = blocks_near(blocks, towns, c.get("amenity_radii_meters") or {}, amenities)
    independent = priced & near                      # the join, done here on normalised block keys

    out: Dict[str, Any] = {"id": item["id"], "stored": len(raw), "independent": len(independent)}
    if stored != raw:
        out["known_correction"] = KNOWN_WRONG_BLOCK_TRUTH[str(item["id"]).split("~")[0]]["note"]
    if c.get("work_destinations"):
        # commute is an estimate with no independent source: the stored blocks must meet every other limit
        extra = sorted(stored - independent)
        out.update(status="ok" if not extra else "mismatch", commute_checked=False,
                   stored_not_meeting_other_limits=extra,
                   dropped_by_commute=len(independent - stored))
        return out
    missing, extra = sorted(independent - stored), sorted(stored - independent)
    out.update(status="ok" if not missing and not extra else "mismatch",
               missing_from_stored=missing, stored_but_failing=extra)
    return out


def load_spatial() -> Dict[str, Any]:
    with open(SPATIAL_INDEX, "rb") as fh:
        return pickle.load(fh)["payload"]


def sales_reader(con):
    first, end = con.execute("SELECT first_month, analysis_end_month FROM source_snapshot").fetchone()
    start, end = window(end, first)
    cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Dict[str, Dict[str, float]]] = {}

    def sales_for(towns: List[str], flat_types: List[str]) -> Dict[str, Dict[str, float]]:
        key = (tuple(sorted(towns)), tuple(sorted(flat_types)))
        if key not in cache:
            rows = con.execute(
                "SELECT block, street, month, resale_price, remaining_lease_months FROM resale_transactions "
                "WHERE month BETWEEN ? AND ? AND list_contains(?, town) AND list_contains(?, flat_type)",
                [start, end, list(towns), list(flat_types)]).fetchall()
            cache[key] = price_groups(rows, end)
        return cache[key]

    return sales_for, (start, end)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="one JSON line per question")
    args = ap.parse_args(argv)
    spatial = load_spatial()
    bad = 0
    with open_market_database() as con:
        sales_for, (start, end) = sales_reader(con)
        if not args.json:
            print(f"window {start:%Y-%m}..{end:%Y-%m}")
        for name in FILES:
            path = ROOT / "evaluation" / name
            if not path.is_file():
                continue
            results = [check_item(it, sales_for, spatial) for it in json.loads(path.read_text())]
            bad += sum(r["status"] != "ok" for r in results)
            if args.json:
                for r in results:
                    print(json.dumps({"file": name, **r}))
                continue
            ok = sum(r["status"] == "ok" for r in results)
            for r in results:
                if r["status"] == "ok" and r.get("known_correction"):
                    print(f"  {r['id']}: stored {r['stored']} blocks, data shows {r['independent']}; known, corrected in "
                          f"scoring ({r['known_correction']})")
            b3 = sum(1 for r in results if r.get("commute_checked") is False)
            print(f"{name}: {ok}/{len(results)} match" + (f" ({b3} commute questions: other limits only)" if b3 else ""))
            for r in results:
                if r["status"] != "ok":
                    print(f"  {r['id']}: {json.dumps({k: v for k, v in r.items() if k != 'id'})}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
