"""
Google Places API Integration (mcp_server/google_places_tools.py)
Fetches private and commercial food establishments (food courts, coffeeshops,
restaurants, cafes) using Google Places Nearby Search API.
"""
from __future__ import annotations

import os
import math
import requests
from typing import Any, Dict, List, Optional
from mcp_server.spatial_onemap_tools import wgs84_to_svy21

_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# Seeded commercial food places for Bishan & key hubs so food appears even before an external key is configured
SEEDED_COMMERCIAL_FOOD: List[Dict[str, Any]] = [
    {
        "name": "Food Junction (Junction 8)",
        "category": "hawker",
        "place_type": "food_court",
        "rating": 4.1,
        "lat": 1.350481,
        "lng": 103.848760,
        "address": "9 Bishan Pl, #04-01 Junction 8, Singapore 579837",
    },
    {
        "name": "Kopitiam (Bishan Central)",
        "category": "hawker",
        "place_type": "coffeeshop",
        "rating": 3.9,
        "lat": 1.351210,
        "lng": 103.849100,
        "address": "Blk 512 Bishan St 13, Singapore 570512",
    },
    {
        "name": "S-11 Food House (Bishan 504)",
        "category": "hawker",
        "place_type": "coffeeshop",
        "rating": 4.0,
        "lat": 1.349820,
        "lng": 103.849450,
        "address": "Blk 504 Bishan St 13, Singapore 570504",
    },
    {
        "name": "Kim San Leng Coffeeshop (Bishan)",
        "category": "hawker",
        "place_type": "coffeeshop",
        "rating": 3.8,
        "lat": 1.351890,
        "lng": 103.848200,
        "address": "Blk 511 Bishan St 13, Singapore 570511",
    },
    {
        "name": "Bishan North Shopping Mall Food Court",
        "category": "hawker",
        "place_type": "food_court",
        "rating": 3.9,
        "lat": 1.358920,
        "lng": 103.846500,
        "address": "Blk 282 Bishan St 22, Singapore 570282",
    },
    {
        "name": "Broadway Food Centre (Sin Ming)",
        "category": "hawker",
        "place_type": "coffeeshop",
        "rating": 4.1,
        "lat": 1.355150,
        "lng": 103.837900,
        "address": "Blk 23 Sin Ming Rd, Singapore 570023",
    },
]


def get_google_maps_api_key() -> str:
    """Retrieve Google Places / Maps API key from environment."""
    return (
        os.getenv("GOOGLE_PLACES_API_KEY")
        or os.getenv("GOOGLE_MAPS_API_KEY")
        or ""
    ).strip()


def fetch_google_places_food(
    lat: float,
    lng: float,
    radius_meters: float = 1500.0,
) -> List[Dict[str, Any]]:
    """
    Queries Google Places Nearby Search for food courts, restaurants, cafes,
    and coffeeshops within radius_meters.
    Falls back gracefully to seeded commercial eating places if key is unavailable.
    """
    api_key = get_google_maps_api_key()
    cx, cy = wgs84_to_svy21(lat, lng)
    results: List[Dict[str, Any]] = []

    # 1. Check in-memory cache (keyed to ~50m grid)
    cache_key = f"{round(lat, 4)}_{round(lng, 4)}_{int(radius_meters)}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # 2. If API Key is configured, perform live Google Places Nearby Search
    if api_key:
        try:
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{lat},{lng}",
                "radius": int(radius_meters),
                "type": "restaurant",
                "key": api_key,
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    p_lat = item["geometry"]["location"]["lat"]
                    p_lng = item["geometry"]["location"]["lng"]
                    px, py = wgs84_to_svy21(p_lat, p_lng)
                    dist = float(math.hypot(px - cx, py - cy))
                    if dist <= radius_meters:
                        # Categorize specific food type
                        types = item.get("types", [])
                        ptype = "restaurant"
                        if "cafe" in types:
                            ptype = "cafe"
                        elif "bakery" in types:
                            ptype = "bakery"
                        elif "meal_takeaway" in types:
                            ptype = "takeaway"

                        results.append({
                            "name": item.get("name", ""),
                            "category": "hawker",
                            "place_type": ptype,
                            "rating": item.get("rating"),
                            "user_ratings_total": item.get("user_ratings_total"),
                            "lat": p_lat,
                            "lng": p_lng,
                            "x": px,
                            "y": py,
                            "distance_meters": round(dist, 1),
                            "walking_mins": max(1, round(dist / 80.0)),
                            "address": item.get("vicinity", ""),
                        })
        except Exception as exc:
            print(f"[Warning] Google Places API query failed: {exc}")

    # 3. Augment with seeded commercial food hubs within radius
    for seed in SEEDED_COMMERCIAL_FOOD:
        s_lat = seed["lat"]
        s_lng = seed["lng"]
        sx, sy = wgs84_to_svy21(s_lat, s_lng)
        dist = float(math.hypot(sx - cx, sy - cy))
        if dist <= radius_meters:
            # Avoid duplicate if already pulled by Google Places
            if not any(seed["name"].lower() in r["name"].lower() for r in results):
                results.append({
                    **seed,
                    "x": sx,
                    "y": sy,
                    "distance_meters": round(dist, 1),
                    "walking_mins": max(1, round(dist / 80.0)),
                })

    _CACHE[cache_key] = results
    return results

