"""
Geospatial & Commute Agent (agents/spatial_agent.py).
Owned by Member 2 (Categories B & E).
Implements a LangGraph cyclic state machine that resolves location entities,
constructs SVY21 metric buffer polygons, performs KD-Tree proximity searches,
and emits GeoJSON state updates to sync the interactive MapLibre GL map.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
import numpy as np
from langgraph.graph import StateGraph, END

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.spatial_onemap_tools import (
    SpatialIndex,
    create_buffer_geojson,
    wgs84_to_svy21,
    svy21_to_wgs84,
    geocode_onemap,
    onemap_client,
    get_spatial_index,
    verify_primary_school_distance,
    filter_blocks_by_amenity_proximity,
)
from agents.state import HDBCompassState, UserConstraints

spatial_index = SpatialIndex()


class GeoAgentState(TypedDict, total=False):
    """
    State schema for Geospatial & Commute Agent.
    Sub-state of HDBCompassState with map synchronization keys.
    """
    query: str
    step: str
    target_name: Optional[str]
    school_name: Optional[str]
    town: Optional[str]
    lat: Optional[float]
    lng: Optional[float]
    svy21_x: Optional[float]
    svy21_y: Optional[float]
    radius_meters: float
    qualifying_blocks: List[str]
    spatial_block_candidates: Optional[List[Any]]   # Proposal Page 7 key
    geojson_buffer: Optional[Dict[str, Any]]
    nearby_amenities: List[Dict[str, Any]]
    geojson_amenities: Optional[Dict[str, Any]]
    map_ui_payload: Optional[Dict[str, Any]]        # Proposal Page 7 key
    summary: str
    logs: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Node 1: Resolve Location Coordinates & Entity Extraction
# ---------------------------------------------------------------------------
def resolve_location_node(state: GeoAgentState) -> Dict[str, Any]:
    query = state.get("query", "").strip()
    logs = list(state.get("logs", []))
    radius = state.get("radius_meters", 1000.0)

    # 1. Extract radius if specified (e.g. "500m", "1km", "300 meters")
    match_radius = re.search(r"(\d+(?:\.\d+)?)\s*(km|m|meters|kilometer)", query, re.IGNORECASE)
    if match_radius:
        val = float(match_radius.group(1))
        unit = match_radius.group(2).lower()
        radius = val * 1000.0 if "km" in unit else val

    # 2. Check if user pinned exact coordinates (e.g. from map click)
    lat = state.get("lat")
    lng = state.get("lng")
    if lat is not None and lng is not None:
        x, y = wgs84_to_svy21(lat, lng)
        logs.append({
            "step": "RESOLVE_LOCATION",
            "message": f"Coordinates pinned: ({lat:.5f}, {lng:.5f}) -> SVY21 (X={x:.1f}m, Y={y:.1f}m)",
            "status": "success"
        })
        return {
            "step": "LOCATION_RESOLVED",
            "target_name": f"Selected Coordinate ({lat:.4f}, {lng:.4f})",
            "lat": lat,
            "lng": lng,
            "svy21_x": x,
            "svy21_y": y,
            "radius_meters": radius,
            "logs": logs
        }

    # 3. Detect MOE Primary Schools dynamically from official MOE schools dataset
    if get_spatial_index:
        try:
            s_idx = get_spatial_index()
            query_lower = query.lower()
            for s in s_idx["payload"]["schools"]:
                s_name = s["name"].lower()
                s_clean = s_name.replace("primary school", "").replace("primary", "").replace("school", "").strip()
                s_alias = s.get("alias", "").lower().replace("primary", "").replace("school", "").strip()

                # Dynamic matching against school name or alias
                matched = False
                if len(s_clean) >= 3 and s_clean in query_lower:
                    matched = True
                elif len(s_alias) >= 3 and s_alias in query_lower:
                    matched = True

                if matched:
                    lat, lng = s["lat"], s["lng"]
                    x, y = s["x"], s["y"]
                    school_full = s.get("alias") or s["name"]
                    logs.append({
                        "step": "RESOLVE_LOCATION",
                        "message": f"Identified MOE Primary School: {school_full} in {s.get('town')} (Lat: {lat:.5f}, Lng: {lng:.5f})",
                        "status": "success"
                    })
                    return {
                        "step": "LOCATION_RESOLVED",
                        "target_name": school_full,
                        "school_name": school_full,
                        "town": s.get("town"),
                        "lat": lat,
                        "lng": lng,
                        "svy21_x": x,
                        "svy21_y": y,
                        "radius_meters": radius if radius != 300 else 1000.0,
                        "logs": logs
                    }
        except Exception as e:
            print(f"School lookup error: {e}")

    # 4. Extract location keywords (Postal codes, landmarks, MRTs)
    postal_match = re.search(r"\b\d{6}\b", query)
    if postal_match:
        search_term = postal_match.group(0)
    else:
        # Strip conversational filler to find candidate landmark
        search_term = re.sub(
            r"\b(we|make|earn|want|find|search|around|near|within|\$?\d+[km]?|budget|room|flat|resale|under|over|lease|years?|mrt|hawker|school|amenities|of|an|and|the|a|for|in)\b",
            "",
            query,
            flags=re.IGNORECASE
        ).strip()
        if not search_term:
            search_term = "Orchard"

    # Call OneMap geocoder
    geo = geocode_onemap(search_term)
    if not geo and "mrt" in query.lower() and "mrt" not in search_term.lower():
        geo = geocode_onemap(f"{search_term} MRT")

    if geo:
        lat, lng = geo["lat"], geo["lng"]
        x, y = geo["x"], geo["y"]
        resolved_name = geo["building"] if geo["building"] and geo["building"] != "NIL" else geo["address"]
        logs.append({
            "step": "RESOLVE_LOCATION",
            "message": f"OneMap Geocoded '{search_term}': {resolved_name} (SVY21: X={x:.1f}m, Y={y:.1f}m)",
            "status": "success"
        })
    else:
        # Fallback to Singapore central (Bishan)
        lat, lng = 1.3508, 103.8481
        x, y = wgs84_to_svy21(lat, lng)
        resolved_name = search_term or "Bishan"
        logs.append({
            "step": "RESOLVE_LOCATION",
            "message": f"Geocoding fallback applied for '{search_term}'. Centered at Bishan.",
            "status": "fallback"
        })

    is_school = "school" in resolved_name.lower() or "primary" in resolved_name.lower()
    detected_school_name = resolved_name if is_school else None

    return {
        "step": "LOCATION_RESOLVED",
        "target_name": resolved_name,
        "school_name": detected_school_name,
        "lat": lat,
        "lng": lng,
        "svy21_x": x,
        "svy21_y": y,
        "radius_meters": radius if (not is_school or radius != 300) else 1000.0,
        "logs": logs
    }


# ---------------------------------------------------------------------------
# Node 2: Compute Metric Buffer Boundary (SVY21 -> GeoJSON Polygon)
# ---------------------------------------------------------------------------
def compute_buffer_node(state: GeoAgentState) -> Dict[str, Any]:
    lat = state["lat"]
    lng = state["lng"]
    radius = state.get("radius_meters", 1000.0)
    target_name = state.get("target_name", "Target Location")
    logs = list(state.get("logs", []))

    buffer_poly = create_buffer_geojson(
        lat=lat,
        lng=lng,
        radius_meters=radius,
        properties={
            "target_name": target_name,
            "radius_meters": radius,
            "layer_type": "moe_school_admission_boundary" if state.get("school_name") else "agent_spatial_buffer"
        }
    )

    logs.append({
        "step": "COMPUTE_BUFFER",
        "message": f"Constructed exact {radius:.0f}m SVY21 metric boundary circle for {target_name}.",
        "status": "success"
    })

    return {
        "step": "BUFFER_COMPUTED",
        "geojson_buffer": buffer_poly,
        "logs": logs
    }


# ---------------------------------------------------------------------------
# Node 3: Query KD-Tree Spatial Index & MOE School Priority Filter
# ---------------------------------------------------------------------------
def query_kdtree_node(state: GeoAgentState) -> Dict[str, Any]:
    lat = state["lat"]
    lng = state["lng"]
    radius = state.get("radius_meters", 1000.0)
    school_name = state.get("school_name")
    logs = list(state.get("logs", []))

    qualifying_blocks = []
    amenity_features = []

    # If query is targeting an MOE Primary School, run verify_primary_school_distance
    if school_name and verify_primary_school_distance:
        bracket = "1km" if radius <= 1000 else "2km"
        blocks_matched = verify_primary_school_distance(school_name, bracket=bracket)
        qualifying_blocks = blocks_matched
        logs.append({
            "step": "QUERY_KDTREE",
            "message": f"MOE School Priority: Found {len(blocks_matched)} HDB blocks strictly within {bracket} of {school_name}.",
            "status": "success",
            "count": len(blocks_matched)
        })

    # Also query general KD-Tree for nearby transit and amenities
    matches = spatial_index.query_radius(lat, lng, radius)

    # If qualifying blocks exist, map them as features for MapLibre
    if qualifying_blocks and get_spatial_index:
        s_idx = get_spatial_index()
        block_keys = {qb.split(" (")[0].strip().upper(): qb for qb in qualifying_blocks}
        rendered_count = 0

        for b in s_idx["payload"]["blocks"]:
            b_key = f"BLK {b['block']} {b['street']}".upper()
            if b_key in block_keys:
                amenity_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [b["lng"], b["lat"]]},
                    "properties": {
                        "name": f"Blk {b['block']} {b['street']}",
                        "category": "hdb_candidate",
                        "priority": "Inside 1km MOE Priority"
                    }
                })
                rendered_count += 1
                if rendered_count >= 60:
                    break
    elif not qualifying_blocks and get_spatial_index:
        s_idx = get_spatial_index()
        block_tree = s_idx.get("block_kdtree")
        if block_tree:
            x, y = state.get("svy21_x"), state.get("svy21_y")
            if x is not None and y is not None:
                nearby_idxs = block_tree.query_ball_point([x, y], r=radius)
                blocks_list = s_idx["payload"]["blocks"]
                for idx in nearby_idxs:
                    b = blocks_list[idx]
                    dx = b["x"] - x
                    dy = b["y"] - y
                    d = float(np.hypot(dx, dy))
                    qualifying_blocks.append(f"Blk {b['block']} {b['street']} ({d:.0f}m)")

                qualifying_blocks.sort(key=lambda s: float(s.split("(")[-1].replace("m)", "")))

                for idx in nearby_idxs[:60]:
                    b = blocks_list[idx]
                    amenity_features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [b["lng"], b["lat"]]},
                        "properties": {
                            "name": f"Blk {b['block']} {b['street']}",
                            "category": "hdb_candidate",
                            "priority": f"Within {radius:.0f}m Buffer"
                        }
                    })

    # Add matched MRTs and amenities
    for m in matches:
        amenity_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [m["lng"], m["lat"]]},
            "properties": {
                "name": m["name"],
                "category": m.get("category", "amenity"),
                "distance_meters": m.get("distance_meters")
            }
        })

    geojson_amenities = {
        "type": "FeatureCollection",
        "features": amenity_features
    }

    map_ui_payload = {
        "buffer": state.get("geojson_buffer"),
        "amenities": geojson_amenities,
        "center": [lat, lng],
        "radius_meters": radius
    }

    return {
        "step": "KDTREE_QUERIED",
        "qualifying_blocks": qualifying_blocks,
        "spatial_block_candidates": qualifying_blocks,
        "nearby_amenities": matches,
        "geojson_amenities": geojson_amenities,
        "map_ui_payload": map_ui_payload,
        "logs": logs
    }


# ---------------------------------------------------------------------------
# Node 4: Synthesize Geospatial & Commute Intelligence
# ---------------------------------------------------------------------------
def synthesize_response_node(state: GeoAgentState) -> Dict[str, Any]:
    target = state.get("target_name", "Specified Location")
    school = state.get("school_name")
    radius = state.get("radius_meters", 1000.0)
    qualifying = state.get("qualifying_blocks", [])
    matches = state.get("nearby_amenities", [])
    logs = list(state.get("logs", []))

    lines = []
    if school:
        lines.append(f"**MOE Primary 1 Priority & Commute Analysis:**")
        lines.append(f"- Target School: **{school}** ({state.get('town', 'Singapore')}).")
        lines.append(f"- Official Priority Circle: **{radius:.0f} meters** (SVY21 metric boundary).")
        if qualifying:
            lines.append(f"- **Qualifying HDB Resale Blocks (< {radius:.0f}m):**")
            for q in qualifying[:5]:
                lines.append(f"  * {q}")
        else:
            lines.append(f"- No HDB block candidates indexed within this exact boundary.")
    else:
        lines.append(f"**Geospatial & Commute Analysis for {target}:**")
        lines.append(f"- Search Buffer: **{radius:.0f} meters** (SVY21 metric boundary).")
        if qualifying:
            lines.append(f"- **Matching HDB Resale Blocks (< {radius:.0f}m):** Found {len(qualifying)} blocks.")
            for q in qualifying[:5]:
                lines.append(f"  * {q}")

    if matches:
        closest = matches[0]
        walk_mins = max(1, round(closest["distance_meters"] / 80.0))
        lines.append(f"- Nearest Transit: **{closest['name']}** at **{closest['distance_meters']:.0f}m** (~{walk_mins} min walk).")

    summary = "\n".join(lines)

    logs.append({
        "step": "SYNTHESIZE",
        "message": "Synthesized proximity and commute analysis.",
        "status": "completed"
    })

    return {
        "step": "COMPLETE",
        "summary": summary,
        "logs": logs
    }


# ---------------------------------------------------------------------------
# Construct the LangGraph State Machine
# ---------------------------------------------------------------------------
def create_geo_agent_workflow():
    workflow = StateGraph(GeoAgentState)

    workflow.add_node("resolve_location", resolve_location_node)
    workflow.add_node("compute_buffer", compute_buffer_node)
    workflow.add_node("query_kdtree", query_kdtree_node)
    workflow.add_node("synthesize", synthesize_response_node)

    workflow.set_entry_point("resolve_location")
    workflow.add_edge("resolve_location", "compute_buffer")
    workflow.add_edge("compute_buffer", "query_kdtree")
    workflow.add_edge("query_kdtree", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


geo_agent = create_geo_agent_workflow()
