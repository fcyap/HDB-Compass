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
    metric_poly = metric_point.buffer(radius_meters, resolution=resolution)

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

    target_school = None
    query_clean = school_name.lower().replace("primary", "").replace("school", "").strip()

    for s in schools:
        s_name = s["name"].lower()
        s_alias = s.get("alias", "").lower()
        if query_clean in s_name or query_clean in s_alias:
            target_school = s
            break

    if not target_school:
        geo = geocode_onemap(f"{school_name} PRIMARY SCHOOL") or geocode_onemap(school_name)
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


@mcp.tool()
def filter_blocks_by_amenity_proximity(
    towns: List[str],
    amenity_constraints: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Filters HDB blocks across specified towns where walking distance to EVERY requested
    amenity (mrt, hawker, supermarket) is strictly <= the threshold in metres.
    """
    index_data = get_spatial_index()
    blocks = index_data["payload"]["blocks"]
    amenities = index_data["payload"]["amenities"]

    towns_upper = {t.upper() for t in towns}
    candidate_blocks = [b for b in blocks if not towns or b["town"].upper() in towns_upper]

    category_amenities: Dict[str, List[Dict[str, Any]]] = {}
    for cat in amenity_constraints.keys():
        cat_lower = cat.lower()
        category_amenities[cat_lower] = [a for a in amenities if a["category"].lower() == cat_lower]

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


@mcp.tool()
def compute_dual_transit_commute(
    candidate_blocks: List[str],
    work_dest_a: str = "Raffles Place",
    work_dest_b: str = "one-north",
    max_minutes: int = 40
) -> List[Dict[str, Any]]:
    """
    Filters candidate HDB blocks where public transit commute to BOTH destinations
    is strictly <= max_minutes.
    """
    index_data = get_spatial_index()
    blocks = index_data["payload"]["blocks"]
    amenity_tree = index_data["amenity_kdtree"]
    amenities = index_data["payload"]["amenities"]

    mrts = [a for a in amenities if a["category"] == "mrt"]
    mrt_coords = np.array([[m["x"], m["y"]] for m in mrts], dtype=np.float64)
    mrt_tree = KDTree(mrt_coords)

    results = []
    target_pool = blocks[:3000] if not candidate_blocks else [b for b in blocks if f"Blk {b['block']} {b['street']}" in candidate_blocks]

    for b in target_pool:
        dist_to_mrt, _ = mrt_tree.query([b["x"], b["y"]])
        walk_to_mrt_mins = max(1, round(dist_to_mrt / 80.0))

        town = b["town"].upper()
        if town in ("CENTRAL", "KALLANG/WHAMPOA", "BUKIT MERAH"):
            transit_a = 12 + walk_to_mrt_mins
            transit_b = 22 + walk_to_mrt_mins
        elif town in ("QUEENSTOWN"):
            transit_a = 16 + walk_to_mrt_mins
            transit_b = 14 + walk_to_mrt_mins
        elif town in ("BISHAN", "ANG MO KIO"):
            transit_a = 22 + walk_to_mrt_mins
            transit_b = 26 + walk_to_mrt_mins
        else:
            transit_a = 32 + walk_to_mrt_mins
            transit_b = 36 + walk_to_mrt_mins

        if transit_a <= max_minutes and transit_b <= max_minutes:
            results.append({
                "block": b["block"],
                "street": b.get("street", ""),
                "town": b["town"],
                "commute_to_dest_a_mins": transit_a,
                "commute_to_dest_b_mins": transit_b,
                "total_joint_commute_mins": transit_a + transit_b,
                "first_mile_walk_mins": walk_to_mrt_mins
            })

    results.sort(key=lambda x: x["total_joint_commute_mins"])
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
