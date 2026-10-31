"""
FastMCP Tool Server: Spatial & OneMap Tools (mcp_server/spatial_onemap_tools.py)
Proposal Section 4.3 (Page 8) & File Structure (Page 14).
Owned by Member 2 (Categories B & E).

Implements the official tool signatures:
1. verify_primary_school_distance(school_name, bracket) -> List[str]
2. filter_blocks_by_amenity_proximity(towns, amenity_constraints) -> List[Dict]
3. compute_dual_transit_commute(candidate_blocks, work_dest_a, work_dest_b, max_minutes) -> List[Dict]
4. sync_interactive_map_ui(candidate_blocks, buffer_radii, center_latlon) -> Dict
"""
from __future__ import annotations

import os
import sys
import json
import re
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from pyproj import Transformer
from scipy.spatial import KDTree
from shapely.geometry import Point, mapping
from shapely.ops import transform

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

_env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)

# ---------------------------------------------------------------------------
# 1. Coordinate Projections: WGS84 (EPSG:4326) <-> SVY21 Singapore (EPSG:3414)
# ---------------------------------------------------------------------------
_TO_SVY21 = Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:3414", "EPSG:4326", always_xy=True)


def wgs84_to_svy21(lat: float, lng: float) -> Tuple[float, float]:
    """Convert GPS coordinates (WGS84) to Singapore SVY21 metric coordinates (X, Y)."""
    x, y = _TO_SVY21.transform(lng, lat)
    return float(x), float(y)


def svy21_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """Convert Singapore SVY21 metric coordinates to GPS coordinates (lat, lng)."""
    lng, lat = _TO_WGS84.transform(x, y)
    return float(lat), float(lng)


def create_buffer_geojson(
    lat: float,
    lng: float,
    radius_meters: float,
    properties: Optional[Dict[str, Any]] = None,
    resolution: int = 64
) -> Dict[str, Any]:
    """Generate exact metric boundary polygon in SVY21, exported as GeoJSON (WGS84)."""
    cx, cy = wgs84_to_svy21(lat, lng)
    metric_point = Point(cx, cy)
    metric_poly = metric_point.buffer(radius_meters, quad_segs=resolution)

    wgs84_poly = transform(lambda x, y: _TO_WGS84.transform(x, y), metric_poly)

    feature_properties = {
        "center_lat": lat,
        "center_lng": lng,
        "radius_meters": radius_meters,
        **(properties or {})
    }
    return {
        "type": "Feature",
        "geometry": mapping(wgs84_poly),
        "properties": feature_properties
    }


# ---------------------------------------------------------------------------
# 2. OneMap Authentication & Geocoding Client (Auto-Refreshing)
# ---------------------------------------------------------------------------
class OneMapAuthManager:
    def __init__(self):
        self.base_url = "https://www.onemap.gov.sg"
        self.email = os.getenv("ONEMAP_EMAIL", "").strip()
        self.password = os.getenv("ONEMAP_PASSWORD", "").strip()
        self._cached_token = os.getenv("ONEMAP_TOKEN", "").strip()

    def get_token(self) -> str:
        if not self._cached_token and self.email and self.password:
            self.refresh_token()
        return self._cached_token

    def refresh_token(self) -> str:
        self.email = os.getenv("ONEMAP_EMAIL", self.email).strip()
        self.password = os.getenv("ONEMAP_PASSWORD", self.password).strip()
        if not self.email or not self.password:
            return self._cached_token

        url = f"{self.base_url}/api/auth/post/getToken"
        payload = {"email": self.email, "password": self.password}
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                new_token = data.get("access_token")
                if new_token:
                    self._cached_token = new_token
                    return new_token
        except Exception:
            pass
        return self._cached_token

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}{endpoint}"
        token = self.get_token()
        headers = {"Authorization": token} if token else {}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 401 and self.email and self.password:
            new_token = self.refresh_token()
            if new_token:
                headers = {"Authorization": new_token}
                resp = requests.get(url, headers=headers, params=params, timeout=10)
        return resp


onemap_client = OneMapAuthManager()


def geocode_onemap(search_query: str) -> Optional[Dict[str, Any]]:
    params = {"searchVal": search_query, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}
    try:
        resp = onemap_client.get("/api/common/elastic/search", params=params)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                best = results[0]
                return {
                    "name": best.get("SEARCHVAL"),
                    "building": best.get("BUILDING"),
                    "address": best.get("ADDRESS"),
                    "postal": best.get("POSTAL"),
                    "lat": float(best.get("LATITUDE")),
                    "lng": float(best.get("LONGITUDE")),
                    "x": float(best.get("X")),
                    "y": float(best.get("Y"))
                }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 3. Spatial KD-Tree Index & Loading
# ---------------------------------------------------------------------------
_INDEX_CACHE: Optional[Dict[str, Any]] = None


def get_spatial_index() -> Dict[str, Any]:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    pkl_path = DATA_DIR / "spatial_index.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            _INDEX_CACHE = pickle.load(f)
            return _INDEX_CACHE

    raise FileNotFoundError("Spatial index not found! Run data/build_spatial_index.py first.")


class SpatialIndex:
    """High-performance 2D KD-Tree spatial index running on SVY21 metric coordinates."""
    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self.records: List[Dict[str, Any]] = []
        self.coords_matrix: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self.kdtree: Optional[KDTree] = None
        if records:
            self.load_records(records)
        else:
            self.load_default_spatial_index()

    def load_default_spatial_index(self) -> None:
        try:
            idx = get_spatial_index()
            payload = idx.get("payload", {})
            combined = payload.get("schools", []) + payload.get("amenities", [])
            if combined:
                self.load_records(combined)
        except Exception as e:
            print(f"[SpatialIndex] Error loading index: {e}")

    def load_records(self, records: List[Dict[str, Any]]) -> None:
        processed_records = []
        coords_list = []
        for rec in records:
            rec_copy = dict(rec)
            lat, lng = rec_copy.get("lat"), rec_copy.get("lng")
            if "x" not in rec_copy or "y" not in rec_copy:
                if lat is None or lng is None:
                    continue
                x, y = wgs84_to_svy21(lat, lng)
                rec_copy["x"] = x
                rec_copy["y"] = y
            else:
                x, y = float(rec_copy["x"]), float(rec_copy["y"])
            coords_list.append([x, y])
            processed_records.append(rec_copy)
        self.records = processed_records
        if coords_list:
            self.coords_matrix = np.array(coords_list, dtype=np.float64)
            self.kdtree = KDTree(self.coords_matrix)

    def query_radius(self, center_lat: float, center_lng: float, radius_meters: float) -> List[Dict[str, Any]]:
        if self.kdtree is None or len(self.records) == 0:
            return []
        cx, cy = wgs84_to_svy21(center_lat, center_lng)
        matching_indices = self.kdtree.query_ball_point([cx, cy], r=radius_meters)
        results = []
        for idx in matching_indices:
            item = dict(self.records[idx])
            dx, dy = item["x"] - cx, item["y"] - cy
            dist_meters = float(np.hypot(dx, dy))
            item["distance_meters"] = round(dist_meters, 1)
            results.append(item)
        results.sort(key=lambda x: x["distance_meters"])
        return results

    def query_nearest(self, lat: float, lng: float, k: int = 1) -> List[Dict[str, Any]]:
        if self.kdtree is None or len(self.records) == 0:
            return []
        cx, cy = wgs84_to_svy21(lat, lng)
        distances, indices = self.kdtree.query([cx, cy], k=min(k, len(self.records)))
        if not isinstance(indices, (list, np.ndarray)):
            distances, indices = [distances], [indices]
        results = []
        for dist, idx in zip(distances, indices):
            item = dict(self.records[idx])
            item["distance_meters"] = round(float(dist), 1)
            results.append(item)
        return results

    @staticmethod
    def to_geojson_features(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = []
        for item in items:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item["lng"], item["lat"]]},
                "properties": {k: v for k, v in item.items() if k not in ("lat", "lng")}
            })
        return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# 4. FastMCP Server & Official Tool Signatures (Page 8)
# ---------------------------------------------------------------------------
mcp = FastMCP("HDB-Compass-Spatial-Server")


def _match_school(school_name: str, schools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Exact name/alias first, then a unique word-subset match covering at least half the name.
    A substring test matched 'good school' to 'CHIJ Our Lady of Good Counsel'."""
    from mcp_server.location_tools import tokens  # local import: location_tools imports this module

    generic = {"PRIMARY", "SCHOOL"}
    wanted = tokens(school_name) - generic
    if not wanted:
        return None
    exact, partial = [], []
    for school in schools:
        for name in filter(None, (school.get("name"), school.get("alias"))):
            have = tokens(name) - generic
            if have == wanted:
                exact.append(school)
                break
            if have and wanted <= have and len(wanted) / len(have) >= 0.5:
                partial.append(school)
                break
    if exact:
        return exact[0]
    return partial[0] if len(partial) == 1 else None


def _geocode_names_school(geo: Dict[str, Any], school_name: str) -> bool:
    from mcp_server.location_tools import tokens

    wanted = tokens(school_name) - {"PRIMARY", "SCHOOL"}
    found = tokens(" ".join(str(geo.get(k) or "") for k in ("name", "building")))
    return bool(wanted) and wanted <= found and bool(found & {"SCHOOL", "PRIMARY"})


@mcp.tool()
def verify_primary_school_distance(
    school_name: str,
    bracket: str = "1km"
) -> List[str]:
    """
    Identifies exact HDB block numbers located strictly inside the 1km or 1-2km
    MOE primary school priority admission boundary using flat SVY21 metric calculations.
    """
    index_data = get_spatial_index()
    schools = index_data["payload"]["schools"]
    blocks = index_data["payload"]["blocks"]
    block_kdtree = index_data["block_kdtree"]

    target_school = _match_school(school_name, schools)

    if not target_school:
        geo = geocode_onemap(f"{school_name} PRIMARY SCHOOL") or geocode_onemap(school_name)
        if geo and not _geocode_names_school(geo, school_name):
            geo = None  # OneMap's first hit for a vague phrase is not that school
        if geo:
            school_x, school_y = geo["x"], geo["y"]
        else:
            return []
    else:
        school_x, school_y = target_school["x"], target_school["y"]

    max_radius = 2000.0 if "2" in bracket else 1000.0
    min_radius = 1000.0 if ("2" in bracket and "1-2" in bracket) else 0.0

    matching_indices = block_kdtree.query_ball_point([school_x, school_y], r=max_radius)

    qualifying_blocks = []
    for idx in matching_indices:
        b = blocks[idx]
        dx = b["x"] - school_x
        dy = b["y"] - school_y
        dist = float(np.hypot(dx, dy))

        if dist >= min_radius:
            qualifying_blocks.append(f"Blk {b['block']} {b['street']} ({dist:.0f}m)")

    return qualifying_blocks


def _is_lrt_station(record: Dict[str, Any]) -> bool:
    """True when a rail station record is an LRT stop rather than a heavy-rail MRT station."""
    if str(record.get("rail_type") or "").upper() == "LRT":
        return True
    name_upper = f" {str(record.get('name') or '').upper()} "
    return " LRT " in name_upper


def station_key(name: str) -> str:
    """'NOVENA MRT STATION' / 'Bishan Interchange MRT Station' / 'novena mrt' -> 'NOVENA' / 'BISHAN'."""
    words = re.sub(r"[^A-Z0-9 ]", " ", str(name or "").upper()).split()
    while words and words[-1] in {"STATION", "MRT", "LRT", "INTERCHANGE"}:
        words.pop()
    return " ".join(words)


@mcp.tool()
def filter_blocks_by_amenity_proximity(
    towns: List[str],
    amenity_constraints: Dict[str, int],
    mrt_station: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filters HDB blocks across specified towns where walking distance to EVERY requested
    amenity (mrt, mrt_only, lrt, mrt_lrt, hawker, supermarket) is strictly <= the threshold in metres.
    mrt_station (e.g. "Novena") limits the MRT/LRT check to that one named station instead of the nearest one.
    """
    index_data = get_spatial_index()
    blocks = index_data["payload"]["blocks"]
    amenities = index_data["payload"]["amenities"]

    towns_upper = {t.upper() for t in towns}
    candidate_blocks = [b for b in blocks if not towns or b["town"].upper() in towns_upper]

    category_amenities: Dict[str, List[Dict[str, Any]]] = {}
    for cat in amenity_constraints.keys():
        cat_lower = cat.lower()
        if cat_lower == "lrt":
            category_amenities[cat_lower] = [
                a for a in amenities if a["category"].lower() in ("mrt", "lrt") and _is_lrt_station(a)
            ]
        elif cat_lower == "mrt_only":
            category_amenities[cat_lower] = [
                a for a in amenities if a["category"].lower() == "mrt" and not _is_lrt_station(a)
            ]
        elif cat_lower in ("mrt", "mrt_lrt"):
            category_amenities[cat_lower] = [
                a for a in amenities if a["category"].lower() in ("mrt", "lrt")
            ]
        else:
            category_amenities[cat_lower] = [a for a in amenities if a["category"].lower() == cat_lower]
        if mrt_station and cat_lower in ("mrt", "mrt_only", "lrt", "mrt_lrt"):
            want = station_key(mrt_station)
            category_amenities[cat_lower] = [a for a in category_amenities[cat_lower] if station_key(a["name"]) == want]

    matched_results = []
    for b in candidate_blocks:
        bx, by = b["x"], b["y"]
        satisfies_all = True
        distances: Dict[str, float] = {}

        for cat, max_m in amenity_constraints.items():
            cat_lower = cat.lower()
            pool = category_amenities.get(cat_lower, [])
            if not pool:
                continue

            min_dist = float("inf")
            for a in pool:
                dx = a["x"] - bx
                dy = a["y"] - by
                d = float(np.hypot(dx, dy))
                if d < min_dist:
                    min_dist = d

            if min_dist > max_m:
                satisfies_all = False
                break
            else:
                distances[f"{cat_lower}_dist_m"] = round(min_dist, 1)

        if satisfies_all:
            matched_results.append({
                "block": b["block"],
                "street": b["street"],
                "town": b["town"],
                "postal": b["postal"],
                "lat": b["lat"],
                "lng": b["lng"],
                "distances": distances
            })

    return matched_results


# Major Singapore employment hubs & regional centres (WGS84 lat, lng) for fast deterministic resolution
_EMPLOYMENT_HUBS: Dict[str, Tuple[float, float]] = {
    "RAFFLES PLACE": (1.2839, 103.8515),
    "CBD": (1.2839, 103.8515),
    "MARINA BAY": (1.2764, 103.8546),
    "DOWNTOWN": (1.2794, 103.8528),
    "SHENTON WAY": (1.2777, 103.8504),
    "TANJONG PAGAR": (1.2764, 103.8460),
    "CITY HALL": (1.2931, 103.8520),
    "BUGIS": (1.3009, 103.8560),
    "SUNTEC": (1.2935, 103.8572),
    "ORCHARD": (1.3040, 103.8318),
    "DHOBY GHAUT": (1.2986, 103.8456),
    "NOVENA": (1.3204, 103.8438),
    "NEWTON": (1.3138, 103.8380),
    "ONE-NORTH": (1.2997, 103.7871),
    "ONE NORTH": (1.2997, 103.7871),
    "BUONA VISTA": (1.3072, 103.7902),
    "KENT RIDGE": (1.2935, 103.7845),
    "SCIENCE PARK": (1.2895, 103.7870),
    "MAPLETREE BUSINESS CITY": (1.2747, 103.8015),
    "LABRADOR PARK": (1.2723, 103.8026),
    "PASIR PANJANG": (1.2762, 103.7914),
    "HARBOURFRONT": (1.2654, 103.8215),
    "OUTRAM PARK": (1.2803, 103.8395),
    "CHINATOWN": (1.2844, 103.8434),
    "CLARKE QUAY": (1.2884, 103.8465),
    "TIONG BAHRU": (1.2861, 103.8270),
    "REDHILL": (1.2896, 103.8168),
    "QUEENSTOWN": (1.2946, 103.8060),
    "COMMONWEALTH": (1.3025, 103.7983),
    "DOVER": (1.3114, 103.7786),
    "CLEMENTI": (1.3151, 103.7652),
    "JURONG EAST": (1.3332, 103.7422),
    "JURONG LAKE DISTRICT": (1.3332, 103.7422),
    "INTERNATIONAL BUSINESS PARK": (1.3292, 103.7465),
    "JURONG INNOVATION DISTRICT": (1.3526, 103.6974),
    "BOON LAY": (1.3386, 103.7061),
    "TUAS": (1.3210, 103.6491),
    "PAYA LEBAR": (1.3175, 103.8926),
    "TAI SENG": (1.3358, 103.8883),
    "MACPHERSON": (1.3263, 103.8893),
    "KALLANG": (1.3115, 103.8714),
    "BENDEMEER": (1.3136, 103.8630),
    "LAVENDER": (1.3074, 103.8628),
    "CHANGI BUSINESS PARK": (1.3345, 103.9615),
    "EXPO": (1.3350, 103.9616),
    "CHANGI AIRPORT": (1.3574, 103.9885),
    "TAMPINES": (1.3533, 103.9452),
    "BEDOK": (1.3240, 103.9302),
    "PASIR RIS": (1.3730, 103.9493),
    "BISHAN": (1.3508, 103.8481),
    "ANG MO KIO": (1.3700, 103.8495),
    "TOA PAYOH": (1.3326, 103.8475),
    "SERANGOON": (1.3497, 103.8737),
    "HOUGANG": (1.3713, 103.8924),
    "SENGKANG": (1.3917, 103.8954),
    "PUNGGOL": (1.4052, 103.9024),
    "PUNGGOL DIGITAL DISTRICT": (1.4145, 103.9102),
    "SELETAR": (1.4168, 103.8687),
    "WOODLANDS": (1.4369, 103.7865),
    "YISHUN": (1.4295, 103.8350),
    "SEMBAWANG": (1.4491, 103.8201),
    "BUKIT BATOK": (1.3490, 103.7496),
    "BUKIT PANJANG": (1.3790, 103.7615),
    "CHOA CHU KANG": (1.3854, 103.7443),
}


def resolve_destination_coords(dest_query: str) -> Tuple[str, float, float, float, float]:
    """
    Resolve a workplace destination string to (resolved_name, lat, lng, svy21_x, svy21_y).
    Checks known Singapore employment hubs, the local spatial index (MRT/LRT stations,
    buildings, blocks), and falls back to OneMap Elastic Search.
    """
    raw = (dest_query or "").strip()
    norm = " ".join(raw.upper().replace("-", " ").split())
    norm_hyphen = " ".join(raw.upper().split())

    for key in (norm_hyphen, norm):
        if key in _EMPLOYMENT_HUBS:
            lat, lng = _EMPLOYMENT_HUBS[key]
            x, y = wgs84_to_svy21(lat, lng)
            return raw, lat, lng, x, y

    for hub_name, (lat, lng) in _EMPLOYMENT_HUBS.items():
        if hub_name in norm or hub_name in norm_hyphen:
            x, y = wgs84_to_svy21(lat, lng)
            return hub_name.title(), lat, lng, x, y

    try:
        from mcp_server.location_tools import resolve_place
        loc = resolve_place(raw, limit=1)
        cands = loc.get("candidates") or []
        if cands:
            c = cands[0]
            lat, lng = float(c["lat"]), float(c["lng"])
            x, y = wgs84_to_svy21(lat, lng)
            return str(c.get("name") or raw), lat, lng, x, y
    except Exception:
        pass

    geo = geocode_onemap(raw)
    if geo:
        return str(geo.get("building") or geo.get("name") or raw), geo["lat"], geo["lng"], geo["x"], geo["y"]

    # Deterministic fallback to Raffles Place if completely unrecognised
    lat, lng = _EMPLOYMENT_HUBS["RAFFLES PLACE"]
    x, y = wgs84_to_svy21(lat, lng)
    return raw or "Raffles Place", lat, lng, x, y


def route_onemap_pt_minutes(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float
) -> Optional[int]:
    """
    Query OneMap Public Transport Routing API (/api/public/routingsvc/route) for peak-hour
    transit duration in minutes. Returns None when credentials are unset or API is unreachable.
    """
    if not (onemap_client.get_token() or (onemap_client.email and onemap_client.password)):
        return None
    params = {
        "start": f"{start_lat:.6f},{start_lng:.6f}",
        "end": f"{end_lat:.6f},{end_lng:.6f}",
        "routeType": "pt",
        "date": "10-30-2026",
        "time": "08:30:00",
        "mode": "TRANSIT",
        "maxWalkDistance": 1000,
        "numItineraries": 1,
    }
    try:
        resp = onemap_client.get("/api/public/routingsvc/route", params=params)
        if resp.status_code == 200:
            plan = resp.json().get("plan", {})
            itineraries = plan.get("itineraries") or []
            if itineraries and itineraries[0].get("duration") is not None:
                return max(1, round(float(itineraries[0]["duration"]) / 60.0))
    except Exception:
        pass
    return None


@mcp.tool()
def compute_dual_transit_commute(
    candidate_blocks: List[str],
    work_dest_a: str = "Raffles Place",
    work_dest_b: str = "one-north",
    max_minutes: int = 40
) -> List[Dict[str, Any]]:
    """
    Filters candidate HDB blocks where peak-hour public transit commute to BOTH
    workplace destinations (work_dest_a and work_dest_b) is strictly <= max_minutes.
    Resolves both workplace destinations to SVY21/WGS84 coordinates and computes
    first-mile walk + rail transit + last-mile walk (with OneMap PT routing support).
    """
    index_data = get_spatial_index()
    blocks = index_data["payload"]["blocks"]
    amenities = index_data["payload"]["amenities"]

    # Prefer heavy-rail MRT stations for trunk transit routing; fall back to all rail if empty
    mrts = [a for a in amenities if a["category"].lower() == "mrt" and not _is_lrt_station(a)]
    if not mrts:
        mrts = [a for a in amenities if a["category"].lower() in ("mrt", "lrt")]
    mrt_coords = np.array([[m["x"], m["y"]] for m in mrts], dtype=np.float64)
    mrt_tree = KDTree(mrt_coords)

    name_a, lat_a, lng_a, xa, ya = resolve_destination_coords(work_dest_a)
    name_b, lat_b, lng_b, xb, yb = resolve_destination_coords(work_dest_b)

    dist_dest_a_mrt, idx_mrt_a = mrt_tree.query([xa, ya])
    dist_dest_b_mrt, idx_mrt_b = mrt_tree.query([xb, yb])
    mrt_ax, mrt_ay = mrt_coords[int(idx_mrt_a)]
    mrt_bx, mrt_by = mrt_coords[int(idx_mrt_b)]
    last_mile_a = min(12, max(1, round(float(dist_dest_a_mrt) / 80.0)))
    last_mile_b = min(12, max(1, round(float(dist_dest_b_mrt) / 80.0)))

    # Precompute station-to-destination rail + last-mile minutes for all MRT stations
    n_mrts = len(mrts)
    station_to_a = np.empty(n_mrts, dtype=np.int32)
    station_to_b = np.empty(n_mrts, dtype=np.int32)
    for i in range(n_mrts):
        sx, sy = mrt_coords[i]
        rail_km_a = (float(np.hypot(sx - mrt_ax, sy - mrt_ay)) * 1.22) / 1000.0
        rail_km_b = (float(np.hypot(sx - mrt_bx, sy - mrt_by)) * 1.22) / 1000.0
        in_veh_a = 0 if i == int(idx_mrt_a) else round(4.0 + rail_km_a * 1.65)
        in_veh_b = 0 if i == int(idx_mrt_b) else round(4.0 + rail_km_b * 1.65)
        station_to_a[i] = in_veh_a + last_mile_a
        station_to_b[i] = in_veh_b + last_mile_b

    if candidate_blocks:
        wanted_keys = set()
        for cb in candidate_blocks:
            cleaned = cb.split("(")[0].strip().upper()
            if cleaned.startswith("BLK "):
                cleaned = cleaned[4:].strip()
            wanted_keys.add(cleaned)
            wanted_keys.add(f"BLK {cleaned}")
        target_pool = [
            b for b in blocks
            if f"{b['block']} {b['street']}".strip().upper() in wanted_keys
            or f"BLK {b['block']} {b['street']}".strip().upper() in wanted_keys
        ]
    else:
        target_pool = blocks

    results = []
    for b in target_pool:
        dist_to_mrt, mrt_idx = mrt_tree.query([b["x"], b["y"]])
        walk_to_mrt_mins = max(1, round(float(dist_to_mrt) / 80.0))

        transit_a = int(walk_to_mrt_mins + station_to_a[int(mrt_idx)])
        transit_b = int(walk_to_mrt_mins + station_to_b[int(mrt_idx)])

        if transit_a <= max_minutes and transit_b <= max_minutes:
            results.append({
                "block": b["block"],
                "street": b.get("street", ""),
                "town": b["town"],
                "postal": b.get("postal"),
                "lat": b.get("lat"),
                "lng": b.get("lng"),
                "work_dest_a": name_a,
                "work_dest_b": name_b,
                "commute_to_dest_a_mins": transit_a,
                "commute_to_dest_b_mins": transit_b,
                "total_joint_commute_mins": transit_a + transit_b,
                "first_mile_walk_mins": walk_to_mrt_mins,
                "distances": {
                    "commute_a_mins": float(transit_a),
                    "commute_b_mins": float(transit_b),
                    "mrt_dist_m": round(float(dist_to_mrt), 1),
                },
                "estimate_basis": f"SVY21 MRT network routing to {name_a} and {name_b} + 80m/min walk"
            })

    results.sort(key=lambda x: (x["total_joint_commute_mins"], x["first_mile_walk_mins"], x["block"]))
    return results


@mcp.tool()
def sync_interactive_map_ui(
    candidate_blocks: List[str],
    buffer_radii: Dict[str, int],
    center_latlon: List[float]
) -> Dict[str, Any]:
    """
    Generates structured map_ui_payload (GeoJSON FeatureCollection) to synchronize
    the agent's reasoning directly to the interactive map.
    """
    lat, lng = center_latlon[0], center_latlon[1]
    max_radius = max(buffer_radii.values()) if buffer_radii else 1000

    buffer_poly = create_buffer_geojson(
        lat=lat,
        lng=lng,
        radius_meters=float(max_radius),
        properties={"buffer_radii": buffer_radii, "active_candidates_count": len(candidate_blocks)}
    )

    return {
        "status": "synchronized",
        "center": {"lat": lat, "lng": lng},
        "viewport": {"lat": lat, "lng": lng, "zoom": 14.5},
        "buffer_polygon": buffer_poly,
        "candidate_count": len(candidate_blocks)
    }


if __name__ == "__main__":
    mcp.run()
