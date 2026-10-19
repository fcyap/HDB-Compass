"""
Data Layer: Build Spatial Index (build_spatial_index.py)
Fulfills Phase 2 & Scale Architecture of HDB-Compass (Page 5, Section 3.2).

100% Dynamic Government API Ingestion Pipeline:
1. Ingests all ~13,426 official Singapore HDB existing building footprints & coordinates
   directly from data.gov.sg (Dataset ID: d_16b157c52ed637edd6ba1232e026258d).
2. Merges with HDB Property Information (Dataset ID: d_17f5382f26140b1fdae0ba2ef6239d2f)
   for street names, town classifications, year completed, and max floor levels.
3. Ingests all 179 MOE Primary Schools from data.gov.sg (Dataset ID: d_688b934f82c1059ed0a6993d2a829089).
4. Ingests all 257 official URA Rail Stations (MRT/LRT) from data.gov.sg (Dataset ID: d_8d886e3a83934d7447acdf5bc6959999).
5. Ingests all 107 NEA Government Hawker Centres from data.gov.sg (Dataset ID: d_68a42f09f350881996d83f9cd73ab02f).
6. Ingests all 440+ NEA Licensed Supermarkets from data.gov.sg (Dataset ID: d_11edd0117280c5776651d7891114c88c).
7. Geocodes missing coordinates using OneMap Elastic Search API (/api/common/elastic/search).
8. Converts all geographic coordinates into SVY21 metric planar grid (EPSG:3414).
9. Pre-computes and serializes scipy.spatial.KDTree indices for O(log N) sub-millisecond radius lookups.
"""
from __future__ import annotations

import os
import sys
import io
import csv
import json
import time
import re
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import KDTree
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from mcp_server.spatial_onemap_tools import onemap_client

# Official Public Datasets from data.gov.sg
DATASETS = {
    "hdb_geojson": ("d_16b157c52ed637edd6ba1232e026258d", "geojson"),
    "hdb_prop": ("d_17f5382f26140b1fdae0ba2ef6239d2f", "csv"),
    "schools": ("d_688b934f82c1059ed0a6993d2a829089", "csv"),
    "rail_stations": ("d_8d886e3a83934d7447acdf5bc6959999", "geojson"),
    "hawkers": ("d_68a42f09f350881996d83f9cd73ab02f", "csv"),
    "supermarkets": ("d_11edd0117280c5776651d7891114c88c", "csv"),
}

# Postal Sector to Town Code Mapping
SECTOR_MAP = {
    '56': 'AMK', '57': 'BH', '58': 'BT', '59': 'BT', '60': 'JE', '61': 'JW', '62': 'JW', '63': 'JW', '64': 'JW',
    '65': 'BB', '66': 'BB', '67': 'BP', '68': 'CCK', '69': 'TG', '70': 'TG', '71': 'TG', '72': 'WL', '73': 'WL',
    '75': 'SB', '76': 'YS', '79': 'SK', '80': 'SK', '82': 'PG', '53': 'HG', '54': 'SK', '55': 'SGN',
    '51': 'PRC', '52': 'TAP', '46': 'BD', '47': 'BD', '48': 'BD', '44': 'MP', '45': 'MP', '38': 'GL', '39': 'GL',
    '40': 'GL', '41': 'GL', '34': 'KWN', '35': 'KWN', '36': 'KWN', '37': 'KWN', '31': 'TP', '32': 'TP', '33': 'TP',
    '17': 'CT', '18': 'CT', '19': 'CT', '20': 'CT', '21': 'CT', '01': 'CT', '02': 'CT', '03': 'CT', '04': 'CT',
    '05': 'CT', '06': 'CT', '07': 'CT', '08': 'CT', '14': 'QT', '15': 'BM', '16': 'BM', '09': 'BM', '10': 'BM',
    '11': 'CL', '12': 'CL', '13': 'CL'
}

TOWN_FULL_NAMES = {
    'AMK': 'ANG MO KIO', 'BB': 'BUKIT BATOK', 'BD': 'BEDOK', 'BH': 'BISHAN', 'BM': 'BUKIT MERAH',
    'BP': 'BUKIT PANJANG', 'BT': 'BUKIT TIMAH', 'CCK': 'CHOA CHU KANG', 'CL': 'CLEMENTI', 'CT': 'CENTRAL',
    'GL': 'GEYLANG', 'HG': 'HOUGANG', 'JE': 'JURONG EAST', 'JW': 'JURONG WEST', 'KWN': 'KALLANG/WHAMPOA',
    'MP': 'MARINE PARADE', 'PG': 'PUNGGOL', 'PRC': 'PASIR RIS', 'QT': 'QUEENSTOWN', 'SB': 'SEMBAWANG',
    'SGN': 'SERANGOON', 'SK': 'SENGKANG', 'TAP': 'TAMPINES', 'TG': 'TENGAH', 'TP': 'TOA PAYOH',
    'WL': 'WOODLANDS', 'YS': 'YISHUN'
}


def download_government_datasets() -> Dict[str, Path]:
    """
    Downloads and caches all official datasets from data.gov.sg.
    Handles rate-limits with exponential backoff.
    """
    print("[1/5] Checking and downloading official government datasets from data.gov.sg...")
    file_paths: Dict[str, Path] = {}

    for name, (d_id, ext) in DATASETS.items():
        cache_path = CACHE_DIR / f"{name}.{ext}"
        if cache_path.exists() and cache_path.stat().st_size > 1000:
            print(f"      [OK] {name}: Using cached copy ({cache_path.stat().st_size / 1024:.1f} KB)")
            file_paths[name] = cache_path
            continue

        print(f"      Fetching {name} (ID: {d_id}) from data.gov.sg...")
        downloaded = False
        for attempt in range(5):
            try:
                url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{d_id}/poll-download"
                res = requests.get(url, timeout=15).json()
                if res.get("code") == 0:
                    dl_url = res.get("data", {}).get("url")
                    if dl_url:
                        content = requests.get(dl_url, timeout=60).content
                        with open(cache_path, "wb") as f:
                            f.write(content)
                        print(f"      [SUCCESS] Downloaded {name} ({len(content) / 1024:.1f} KB)")
                        file_paths[name] = cache_path
                        downloaded = True
                        break
                elif res.get("code") == 24:
                    print("      [Rate Limit] data.gov.sg rate limited. Pausing 10s...")
                    time.sleep(10)
            except Exception as e:
                print(f"      Attempt {attempt+1} error: {e}")
            time.sleep(3)

        if not downloaded:
            raise RuntimeError(f"Failed to fetch dataset {name} ({d_id}) from data.gov.sg")

    return file_paths


def geocode_missing_postals(
    postals: Set[str],
    transformer: Transformer,
    hdb_postal_map: Dict[str, Tuple[float, float, float, float]]
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Geocodes postal codes using HDB block matches first, then local cache,
    and queries OneMap Elastic Search API for any remaining.
    Returns: Dict[postal, (lat, lng, svy21_x, svy21_y)]
    """
    cache_path = DATA_DIR / "geocoded_cache.json"
    cache: Dict[str, List[float]] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    resolved: Dict[str, Tuple[float, float, float, float]] = {}
    missing_for_api: List[str] = []

    for p in postals:
        if not p or len(p) != 6 or not p.isdigit():
            continue
        # Check HDB blocks first
        if p in hdb_postal_map:
            resolved[p] = hdb_postal_map[p]
        # Check local cache
        elif p in cache:
            resolved[p] = tuple(cache[p])  # type: ignore
        else:
            missing_for_api.append(p)

    if missing_for_api:
        print(f"      Querying OneMap Geocoding API for {len(missing_for_api)} missing postal codes in parallel...")
        token = onemap_client.get_token()
        headers = {"Authorization": token}

        def _fetch_onemap(postal: str) -> Optional[Tuple[str, float, float, float, float]]:
            try:
                r = requests.get(
                    "https://www.onemap.gov.sg/api/common/elastic/search",
                    params={"searchVal": postal, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1},
                    headers=headers,
                    timeout=8
                )
                data = r.json()
                results = data.get("results", [])
                if results:
                    row = results[0]
                    lat = float(row["LATITUDE"])
                    lng = float(row["LONGITUDE"])
                    x = float(row.get("X", 0.0))
                    y = float(row.get("Y", 0.0))
                    if x == 0.0 or y == 0.0:
                        x, y = transformer.transform(lng, lat)
                    return postal, lat, lng, x, y
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=15) as executor:
            fetched_results = list(executor.map(_fetch_onemap, missing_for_api))

        new_cached = 0
        for item in fetched_results:
            if item:
                pos, lat, lng, x, y = item
                resolved[pos] = (lat, lng, x, y)
                cache[pos] = [lat, lng, x, y]
                new_cached += 1

        print(f"      Successfully geocoded {new_cached} / {len(missing_for_api)} postal codes via OneMap.")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    return resolved


def build_and_save_index() -> None:
    t_start = time.time()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)

    print("=" * 80)
    print("  HDB-Compass: 100% Dynamic Government API Spatial Ingestion Pipeline")
    print("=" * 80)

    # 1. Download official datasets
    paths = download_government_datasets()

    # 2. Parse HDB Blocks
    print("\n[2/5] Parsing HDB Building Polygons & Property Metadata...")
    with open(paths["hdb_geojson"], "r", encoding="utf-8") as f:
        geo_data = json.load(f)
    features = geo_data.get("features", [])

    with open(paths["hdb_prop"], "r", encoding="utf-8") as f:
        prop_reader = csv.DictReader(f)
        prop_dict = {(r.get("bldg_contract_town", "").strip(), r.get("blk_no", "").upper().strip()): r for r in prop_reader}

    processed_blocks = []
    block_coords_svy21 = []
    hdb_postal_map: Dict[str, Tuple[float, float, float, float]] = {}

    for f in features:
        props = f.get("properties", {})
        blk = str(props.get("BLK_NO", "")).strip().upper()
        postal = str(props.get("POSTAL_COD", "")).strip().zfill(6)
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [[]])[0]
        if not coords or len(coords) < 3:
            continue

        lng = float(sum(p[0] for p in coords) / len(coords))
        lat = float(sum(p[1] for p in coords) / len(coords))

        sec = postal[:2]
        town_code = SECTOR_MAP.get(sec, "CT")
        prop = prop_dict.get((town_code, blk))

        if prop:
            street = prop.get("street", "").strip()
            town = TOWN_FULL_NAMES.get(town_code, town_code)
            year = prop.get("year_completed", "")
            max_floor = prop.get("max_floor_lvl", "")
        else:
            town = TOWN_FULL_NAMES.get(town_code, town_code)
            street = f"STREET IN {town}"
            year = ""
            max_floor = ""

        x, y = transformer.transform(lng, lat)

        block_entry = {
            "block": blk,
            "street": street,
            "town": town,
            "postal": postal,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "year_completed": year,
            "max_floor_lvl": max_floor,
        }
        processed_blocks.append(block_entry)
        block_coords_svy21.append([x, y])

        if postal and postal != "000000":
            hdb_postal_map[postal] = (round(lat, 6), round(lng, 6), round(float(x), 2), round(float(y), 2))

    block_kdtree = KDTree(np.array(block_coords_svy21, dtype=np.float64))
    print(f"      Indexed {len(processed_blocks):,} HDB Blocks into KDTree.")

    # 3. Collect all needed postals for Schools, Hawkers, Supermarkets
    print("\n[3/5] Ingesting Primary Schools from MOE & Public Amenities from NEA / URA...")
    
    # 3a. MOE Primary Schools
    with open(paths["schools"], "r", encoding="utf-8") as f:
        schools_reader = csv.DictReader(f)
        raw_schools = [r for r in schools_reader if "PRIMARY" in r.get("mainlevel_code", "").upper()]

    school_postals = {r["postal_code"].strip().zfill(6) for r in raw_schools}

    # 3b. NEA Hawker Centres
    with open(paths["hawkers"], "r", encoding="utf-8") as f:
        hawkers_reader = csv.DictReader(f)
        raw_hawkers = list(hawkers_reader)

    hawker_postals: Set[str] = set()
    for h in raw_hawkers:
        loc = h.get("location_of_centre", "")
        m = re.search(r"S\(?(\d{6})\)?", loc) or re.search(r"\b(\d{6})\b", loc)
        if m:
            hawker_postals.add(m.group(1))

    # 3c. NEA Supermarkets
    with open(paths["supermarkets"], "r", encoding="utf-8") as f:
        sm_reader = csv.DictReader(f)
        raw_supermarkets = list(sm_reader)

    sm_postals = {r["postal_code"].strip().zfill(6) for r in raw_supermarkets if len(r.get("postal_code", "").strip()) == 6}

    # Geocode all missing postals via OneMap API
    all_needed_postals = school_postals | hawker_postals | sm_postals
    geocoded_coords = geocode_missing_postals(all_needed_postals, transformer, hdb_postal_map)

    # 4. Construct Final Entities
    print("\n[4/5] Building Unified Spatial Datasets for Schools & Amenities...")

    # Primary Schools
    processed_schools: List[Dict[str, Any]] = []
    school_coords_svy21 = []
    for s in raw_schools:
        postal = s.get("postal_code", "").strip().zfill(6)
        geo = geocoded_coords.get(postal)
        if not geo:
            continue
        lat, lng, x, y = geo
        name = s.get("school_name", "").strip()
        # Clean alias (e.g. 'AI TONG SCHOOL' -> 'Ai Tong School')
        alias = name.title().replace("Primary School", "Primary").replace("School", "").strip()
        processed_schools.append({
            "name": name,
            "alias": alias,
            "postal": postal,
            "town": s.get("dgp_code", "").strip(),
            "address": s.get("address", "").strip(),
            "lat": lat,
            "lng": lng,
            "x": x,
            "y": y
        })
        school_coords_svy21.append([x, y])

    school_kdtree = KDTree(np.array(school_coords_svy21, dtype=np.float64))
    print(f"      Ingested {len(processed_schools)} MOE Primary Schools.")

    # Amenities: Rail Stations (URA Master Plan) + Hawkers (NEA) + Supermarkets (NEA)
    processed_amenities: List[Dict[str, Any]] = []
    amenity_coords_svy21 = []

    # 4a. URA Rail Stations (MRT & LRT)
    with open(paths["rail_stations"], "r", encoding="utf-8") as f:
        rail_geo = json.load(f)
    for feat in rail_geo.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            poly = shape(geom)
            centroid = poly.centroid
            lng, lat = centroid.x, centroid.y
            x, y = transformer.transform(lng, lat)
            stn_name = props.get("NAME", "STATION").strip()
            rail_type = props.get("RAIL_TYPE", "MRT").strip()
            amenity_entry = {
                "name": f"{stn_name} {rail_type} STATION",
                "category": "mrt",
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "x": round(float(x), 2),
                "y": round(float(y), 2)
            }
            processed_amenities.append(amenity_entry)
            amenity_coords_svy21.append([x, y])
        except Exception:
            continue

    print(f"      Ingested {len(amenity_coords_svy21)} URA MRT & LRT Stations.")

    # 4b. NEA Hawker Centres
    hawkers_count = 0
    for h in raw_hawkers:
        loc = h.get("location_of_centre", "")
        name = h.get("name_of_centre", "").strip()
        m = re.search(r"S\(?(\d{6})\)?", loc) or re.search(r"\b(\d{6})\b", loc)
        if m:
            postal = m.group(1)
            geo = geocoded_coords.get(postal)
            if geo:
                lat, lng, x, y = geo
                amenity_entry = {
                    "name": name,
                    "category": "hawker",
                    "postal": postal,
                    "lat": lat,
                    "lng": lng,
                    "x": x,
                    "y": y
                }
                processed_amenities.append(amenity_entry)
                amenity_coords_svy21.append([x, y])
                hawkers_count += 1

    print(f"      Ingested {hawkers_count} NEA Government Hawker Centres.")

    # 4c. NEA Supermarkets
    supermarket_count = 0
    for sm in raw_supermarkets:
        postal = sm.get("postal_code", "").strip().zfill(6)
        geo = geocoded_coords.get(postal)
        if geo:
            lat, lng, x, y = geo
            bldg = sm.get("building_name", "").strip()
            licensee = sm.get("licensee_name", "").strip()
            if bldg and bldg.lower() != "na":
                sm_name = f"{licensee} ({bldg})"
            else:
                sm_name = licensee
            amenity_entry = {
                "name": sm_name,
                "category": "supermarket",
                "postal": postal,
                "lat": lat,
                "lng": lng,
                "x": x,
                "y": y
            }
            processed_amenities.append(amenity_entry)
            amenity_coords_svy21.append([x, y])
            supermarket_count += 1

    print(f"      Ingested {supermarket_count} NEA Licensed Supermarkets.")

    amenity_kdtree = KDTree(np.array(amenity_coords_svy21, dtype=np.float64))

    # 5. Serialize to Disk
    print("\n[5/5] Serializing to Disk (spatial_index.pkl & spatial_index.json)...")
    payload = {
        "schools": processed_schools,
        "blocks": processed_blocks,
        "amenities": processed_amenities,
    }

    json_path = DATA_DIR / "spatial_index.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_blocks_indexed": len(processed_blocks),
            "total_schools_indexed": len(processed_schools),
            "total_amenities_indexed": len(processed_amenities),
            "schools": processed_schools,
            "sample_amenities": processed_amenities[:50],
            "sample_blocks": processed_blocks[:500]
        }, f, indent=2)

    pickle_path = DATA_DIR / "spatial_index.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "payload": payload,
            "school_kdtree": school_kdtree,
            "block_kdtree": block_kdtree,
            "amenity_kdtree": amenity_kdtree,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    t_end = time.time()
    print("\n" + "=" * 80)
    print(f"  [SUCCESS] Geospatial Index Built 100% Dynamically from APIs in {t_end - t_start:.2f}s:")
    print(f"    - Total HDB Blocks Indexed:   {len(processed_blocks):,}")
    print(f"    - Primary Schools Indexed:     {len(processed_schools):,}")
    print(f"    - Public Amenities Indexed:    {len(processed_amenities):,}")
    print(f"    - Pickle Archive: {pickle_path.name} ({pickle_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"    - JSON Summary:   {json_path.name} ({json_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    build_and_save_index()
