"""Per-block location features for resale analysis (mcp_server/block_features.py). Owned by Member 1.

"Resale prices near an MRT" needs, for every block that has sales, how far its nearest station is. This module
works that out once per market snapshot and keeps it, so the analysis tools can group sales by distance band, by
nearest station or by town with one SQL join instead of a spatial search per question.

What each block gets (straight-line metres in SVY21, the spatial index's own projection):
  nearest MRT station and distance      LRT stations are left out of this on purpose: an LRT stop is not what most
                                        people mean by "near the MRT"
  nearest LRT station and distance      kept apart so "MRT or LRT" and "LRT" questions can use them
  nearest rail (MRT or LRT) station     the nearer of the two
  nearest hawker centre / supermarket / primary school   distance only
  primary schools within 1 km           a count

Interchanges appear in the index more than once (one point per line or exit, slightly apart). Stations are merged by
normalised name, and a block's distance is to the nearest point of any of them, so "Serangoon" is one station.

Block coordinates come from the market database (block_locations), the same geocodes the area tools use; amenities
come from the spatial index, read through location_tools so tests can swap it.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from data.market_database import open_market_database
from mcp_server import location_tools

SCHOOL_RADIUS_M = 1000
_CACHE: Dict[Tuple[Any, ...], "BlockFeatures"] = {}

_STATION_WORDS = re.compile(r"\b(?:MRT|LRT|INTERCHANGE|STATION)\b")


def station_identity(raw: str) -> Tuple[str, str, str]:
    """(merge key, kind, display name) for an index entry such as 'SERANGOON INTERCHANGE MRT STATION'."""
    upper = " ".join(re.sub(r"[^A-Z0-9 ]", " ", (raw or "").upper()).split())
    kind = "LRT" if re.search(r"\bLRT\b", upper) else "MRT"
    base = " ".join(_STATION_WORDS.sub(" ", upper).split()) or upper
    return f"{base}|{kind}", kind, f"{base.title()} {kind}"


def _project(lats: List[float], lngs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    from mcp_server.spatial_onemap_tools import _TO_SVY21   # lazy: pyproj set-up is not needed to import this module
    x, y = _TO_SVY21.transform(np.asarray(lngs, dtype=float), np.asarray(lats, dtype=float))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _xy(records: List[Dict[str, Any]]) -> np.ndarray:
    """SVY21 coordinates of index records: the stored x, y, or projected from lat/lng when absent."""
    out = np.empty((len(records), 2))
    missing = [i for i, r in enumerate(records) if r.get("x") is None or r.get("y") is None]
    for i, r in enumerate(records):
        if i not in missing:
            out[i] = (float(r["x"]), float(r["y"]))
    if missing:
        x, y = _project([records[i]["lat"] for i in missing], [records[i]["lng"] for i in missing])
        out[missing] = np.column_stack([x, y])
    return out


class BlockFeatures:
    """Column-oriented features for every located block that has sales."""

    def __init__(self, snapshot_id: str, keys: List[str], towns: List[str], lat: np.ndarray, lng: np.ndarray,
                 stations: Dict[str, Dict[str, Any]]):
        self.snapshot_id = snapshot_id
        self.keys, self.towns, self.lat, self.lng = keys, towns, lat, lng
        self.stations = stations                       # display name -> {kind, lat, lng, points}
        self.index = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        self.columns: Dict[str, Any] = {name: [None] * n for name in ("mrt_station", "lrt_station", "rail_station")}
        for name in ("mrt_m", "lrt_m", "rail_m", "hawker_m", "supermarket_m", "school_m"):
            self.columns[name] = np.full(n, np.nan)
        self.columns["schools_1km"] = np.zeros(n, dtype=int)
        self.build_seconds = 0.0

    def row(self, key: str) -> Optional[Dict[str, Any]]:
        i = self.index.get(key)
        if i is None:
            return None
        out = {"block_key": key, "town": self.towns[i], "lat": float(self.lat[i]), "lng": float(self.lng[i])}
        for name, col in self.columns.items():
            v = col[i]
            out[name] = (None if isinstance(v, float) and np.isnan(v) else
                         round(float(v), 1) if name.endswith("_m") else v if isinstance(v, str) or v is None else int(v))
        return out

    def load_into(self, con) -> None:
        """Expose the features to SQL as the temporary table block_features (one row per block)."""
        c = self.columns
        con.execute("DROP TABLE IF EXISTS block_features")
        con.execute(
            "CREATE TEMP TABLE block_features AS SELECT unnest(?::VARCHAR[]) AS block_key, unnest(?::VARCHAR[]) AS fe_town, "
            "unnest(?::VARCHAR[]) AS mrt_station, unnest(?::DOUBLE[]) AS mrt_m, "
            "unnest(?::VARCHAR[]) AS lrt_station, unnest(?::DOUBLE[]) AS lrt_m, "
            "unnest(?::VARCHAR[]) AS rail_station, unnest(?::DOUBLE[]) AS rail_m, "
            "unnest(?::DOUBLE[]) AS hawker_m, unnest(?::DOUBLE[]) AS supermarket_m, unnest(?::DOUBLE[]) AS school_m, "
            "unnest(?::INTEGER[]) AS schools_1km",
            [self.keys, self.towns, c["mrt_station"], _nan_none(c["mrt_m"]), c["lrt_station"], _nan_none(c["lrt_m"]),
             c["rail_station"], _nan_none(c["rail_m"]), _nan_none(c["hawker_m"]), _nan_none(c["supermarket_m"]),
             _nan_none(c["school_m"]), [int(v) for v in c["schools_1km"]]])


def _nan_none(arr: np.ndarray) -> List[Optional[float]]:
    return [None if np.isnan(v) else float(v) for v in arr]   # full precision: a band edge must not move by rounding


def _fingerprint(payload: Dict[str, Any]) -> Tuple[Any, ...]:
    am, sc = payload.get("amenities") or [], payload.get("schools") or []
    edge = lambda rows: (rows[0].get("lat"), rows[-1].get("lat")) if rows else ()    # noqa: E731
    return (len(am), len(sc), edge(am), edge(sc))


def _nearest(tree: Optional[cKDTree], xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if tree is None:
        return np.full(len(xy), np.nan), np.full(len(xy), -1, dtype=int)
    dist, idx = tree.query(xy)
    return dist, idx


def _station_tree(records: List[Dict[str, Any]], kind: str, stations: Dict[str, Dict[str, Any]]):
    """KD-tree over every point of the stations of one kind, and the station name behind each point."""
    names: List[str] = []
    points: List[Dict[str, Any]] = []
    for r in records:
        if r.get("category") != "mrt" or not r.get("name"):
            continue
        _, k, display = station_identity(r["name"])
        if k != kind:
            continue
        names.append(display)
        points.append(r)
    if not points:
        return None, names
    xy = _xy(points)
    for display in set(names):
        mine = [p for n, p in zip(names, points) if n == display]
        stations[display] = {"kind": kind, "lat": float(np.mean([p["lat"] for p in mine])),
                             "lng": float(np.mean([p["lng"] for p in mine])), "points": len(mine)}
    return cKDTree(xy), names


def build_block_features(con, payload: Dict[str, Any], snapshot_id: str) -> BlockFeatures:
    started = time.perf_counter()
    rows = con.execute("SELECT b.block_key, b.town, l.lat, l.lng FROM blocks b JOIN block_locations l USING (block_key) "
                       "WHERE l.lat IS NOT NULL AND l.lng IS NOT NULL ORDER BY b.block_key").fetchall()
    keys, towns = [r[0] for r in rows], [r[1] for r in rows]
    lat, lng = np.array([r[2] for r in rows], dtype=float), np.array([r[3] for r in rows], dtype=float)
    stations: Dict[str, Dict[str, Any]] = {}
    feats = BlockFeatures(snapshot_id, keys, towns, lat, lng, stations)
    if not keys:
        return feats
    x, y = _project(list(lat), list(lng))
    xy = np.column_stack([x, y])
    amenities = payload.get("amenities") or []

    mrt_tree, mrt_names = _station_tree(amenities, "MRT", stations)
    lrt_tree, lrt_names = _station_tree(amenities, "LRT", stations)
    c = feats.columns
    d, i = _nearest(mrt_tree, xy)
    c["mrt_m"] = d
    c["mrt_station"] = [mrt_names[j] if j >= 0 else None for j in i]
    d2, i2 = _nearest(lrt_tree, xy)
    c["lrt_m"] = d2
    c["lrt_station"] = [lrt_names[j] if j >= 0 else None for j in i2]
    use_lrt = np.nan_to_num(d2, nan=np.inf) < np.nan_to_num(d, nan=np.inf)
    c["rail_m"] = np.where(use_lrt, d2, d)
    c["rail_station"] = [c["lrt_station"][n] if use_lrt[n] else c["mrt_station"][n] for n in range(len(keys))]

    for category, column in (("hawker", "hawker_m"), ("supermarket", "supermarket_m")):
        pts = [a for a in amenities if a.get("category") == category]
        if pts:
            c[column], _ = _nearest(cKDTree(_xy(pts)), xy)
    schools = payload.get("schools") or []
    if schools:
        school_tree = cKDTree(_xy(schools))
        c["school_m"], _ = _nearest(school_tree, xy)
        counts = school_tree.query_ball_point(xy, r=SCHOOL_RADIUS_M, return_length=True)
        c["schools_1km"] = np.asarray(counts, dtype=int)
    feats.build_seconds = time.perf_counter() - started
    return feats


def get_block_features(con=None) -> BlockFeatures:
    """The features for the current snapshot and spatial index, built on first use and then kept."""
    payload = (location_tools.get_spatial_index() or {}).get("payload") or {}

    def make(connection) -> BlockFeatures:
        snapshot_id = connection.execute("SELECT snapshot_id FROM source_snapshot").fetchone()[0]
        key = (snapshot_id, _fingerprint(payload))
        if key not in _CACHE:
            _CACHE.clear()      # one snapshot is live at a time
            _CACHE[key] = build_block_features(connection, payload, snapshot_id)
        return _CACHE[key]

    if con is not None:
        return make(con)
    with open_market_database() as connection:
        return make(connection)
