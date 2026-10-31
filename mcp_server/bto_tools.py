"""BTO launch prices against resale evidence (mcp_server/bto_tools.py). Owned by Member 1.

Launch data comes from data/bto/bto_launches.csv, transcribed from HDB's official
launch annexes (Table A(1)(a) supply and pricing, and HDB's own "resale comparables
nearby"). Each row keeps its source URL. Add a launch by appending rows from its
annex; never estimate a price range.
"""

from __future__ import annotations

import csv
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from agents.schemas import CompareBtoLaunchVsResaleInput, CompareBtoLaunchVsResaleOutput
from data.market_database import open_market_database

BTO_CSV = Path(__file__).resolve().parent.parent / "data" / "bto" / "bto_launches.csv"
RESALE_LOOKBACK_MONTHS = 12


@lru_cache(maxsize=1)
def load_launches() -> List[Dict[str, Any]]:
    with BTO_CSV.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _norm(value: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9()]+", " ", value.upper()).split())


def _flat_type(value: str) -> str:
    v = _norm(value)
    m = re.fullmatch(r"([1-5]) ?ROOM", v.replace("-", " "))
    return f"{m.group(1)} ROOM" if m else v


def _money(value: str):
    return float(value) if value else None


def compare_bto_launch_vs_resale(bto_project_name: str, radius_meters: int = 500, flat_type: str = "4 ROOM") -> Dict[str, Any]:
    """Compare a recent BTO launch's official indicative prices with resale prices.

    Use the BTO project name (e.g. 'Berlayar Rise', 'Kim Keat Crest'). Returns HDB's
    indicative price range (99-year lease, before grants), classification, waiting
    time and HDB's own published nearby resale comparables, plus the town-wide resale
    median for the same flat type over the last 12 months from the local database.
    Covers launches in data/bto/bto_launches.csv only; unknown projects list what is covered.
    """
    args = CompareBtoLaunchVsResaleInput(bto_project_name=bto_project_name, radius_meters=radius_meters, flat_type=flat_type)
    launches = load_launches()
    wanted = _norm(args.bto_project_name)
    projects = sorted({r["project"] for r in launches})
    rows = [r for r in launches if _norm(r["project"]) == wanted] or \
           [r for r in launches if wanted and (wanted in _norm(r["project"]) or _norm(r["project"]) in wanted)]
    if not rows or len({r["project"] for r in rows}) > 1:
        return CompareBtoLaunchVsResaleOutput(
            status="not_found", bto_project_name=args.bto_project_name,
            available_projects=sorted({r["project"] for r in rows}) or projects,
            notes=["Only launches transcribed from HDB annexes are covered."]).model_dump()
    project = rows[0]["project"]
    offered = sorted({_flat_type(r["flat_type"]) for r in rows})
    flat = _flat_type(args.flat_type)
    match = [r for r in rows if _flat_type(r["flat_type"]) == flat]
    if not match:
        return CompareBtoLaunchVsResaleOutput(
            status="flat_type_not_offered", bto_project_name=project, flat_type=flat, town=rows[0]["town"],
            launch_month=rows[0]["launch_month"], classification=rows[0]["classification"],
            available_flat_types=offered, source_url=rows[0]["source_url"]).model_dump()
    r = match[0]
    price = {"min": float(r["price_min"]), "max": float(r["price_max"])}
    hdb_comp = ({"min": float(r["hdb_resale_comparable_min"]), "max": float(r["hdb_resale_comparable_max"])}
                if r["hdb_resale_comparable_min"] else None)
    resale_median = sample = gap = None
    basis = None
    notes = ["BTO prices are HDB's indicative range for 99-year leases, before grants.",
             "Prime and Plus flats carry a 10-year MOP and subsidy recovery on resale; see HDB policy before comparing."]
    resale_type = flat if re.fullmatch(r"[1-5] ROOM|EXECUTIVE", flat) else None
    if resale_type:
        try:
            with open_market_database() as con:
                end = con.execute("SELECT analysis_end_month FROM source_snapshot").fetchone()[0]
                index = end.year * 12 + end.month - RESALE_LOOKBACK_MONTHS
                start = date(index // 12, index % 12 + 1, 1)
                resale_median, sample = con.execute(
                    "SELECT median(resale_price), count(*) FROM resale_transactions "
                    "WHERE town = ? AND flat_type = ? AND month BETWEEN ?::DATE AND ?::DATE",
                    [r["town"], resale_type, start, end]).fetchone()
            basis = f"{r['town']} town-wide {resale_type} resale median, {start:%Y-%m} to {end:%Y-%m}"
        except FileNotFoundError:
            notes.append("Market database unavailable; town-wide resale median omitted.")
    if resale_median:
        gap = round(((price["min"] + price["max"]) / 2 / float(resale_median) - 1) * 100, 2)
        notes.append("price_gap_pct compares the BTO range midpoint with the town-wide resale median; "
                     "resale flats differ in age, lease and size.")
    if hdb_comp:
        notes.append(f"HDB's annex quotes nearby resale comparables of ${hdb_comp['min']:,.0f}-${hdb_comp['max']:,.0f} "
                     f"({r['hdb_resale_comparable_area_sqm']} sqm).")
    return CompareBtoLaunchVsResaleOutput(
        status="ok", bto_project_name=project, bto_indicative_price_range=price,
        resale_median_nearby=float(resale_median) if resale_median else None,
        resale_sample_size=int(sample) if sample is not None else None, price_gap_pct=gap,
        flat_type=flat, town=r["town"], launch_month=r["launch_month"], classification=r["classification"],
        floor_area_sqm=_money(r["floor_area_sqm"]), units=int(r["units"]), waiting_time_months=r["waiting_time_months"],
        hdb_resale_comparable_range=hdb_comp, resale_basis=basis, available_flat_types=offered,
        source_url=r["source_url"], notes=notes,
    ).model_dump()

