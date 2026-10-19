"""
HDB-Compass Backend API & WebSocket Server (backend/main.py).
Connects the split-screen web frontend to the Geospatial & Commute Agent.
Streams real-time LangGraph state transitions directly to the MapLibre GL map.
"""
from __future__ import annotations

import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.spatial_onemap_tools import (
    create_buffer_geojson,
    wgs84_to_svy21,
    geocode_onemap,
    get_spatial_index,
)
from agents.spatial_agent import geo_agent, GeoAgentState

app = FastAPI(
    title="HDB-Compass Geospatial & Multi-Agent API",
    description="Agentic Geospatial Engine combining SVY21 metric calculations, KD-Tree proximity, and LangGraph WebSocket sync.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RadiusQueryRequest(BaseModel):
    lat: float
    lng: float
    radius_meters: float = Field(1000.0, ge=10.0, le=10000.0)
    include_buffer_polygon: bool = True


@app.get("/")
def read_root():
    return {
        "app": "HDB-Compass API",
        "role": "Member 2: Geospatial Engine & Interactive Map Sync",
        "status": "healthy"
    }


@app.get("/health")
def health_check():
    idx = get_spatial_index()
    return {
        "status": "healthy",
        "total_blocks": len(idx["payload"]["blocks"]),
        "total_schools": len(idx["payload"]["schools"]),
        "total_amenities": len(idx["payload"]["amenities"])
    }


@app.post("/api/spatial/radius")
def query_radius(req: RadiusQueryRequest):
    idx = get_spatial_index()
    amenity_tree = idx.get("amenity_kdtree")
    amenities = idx["payload"]["amenities"]

    cx, cy = wgs84_to_svy21(req.lat, req.lng)
    matches = []
    if amenity_tree:
        idxs = amenity_tree.query_ball_point([cx, cy], r=req.radius_meters)
        for i in idxs:
            a = dict(amenities[i])
            import numpy as np
            dist = float(np.hypot(a["x"] - cx, a["y"] - cy))
            a["distance_meters"] = round(dist, 1)
            matches.append(a)
        matches.sort(key=lambda x: x["distance_meters"])

    features = []
    for item in matches:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [item["lng"], item["lat"]]},
            "properties": {k: v for k, v in item.items() if k not in ("lat", "lng")}
        })

    response = {
        "query": {"center": [req.lat, req.lng], "radius_meters": req.radius_meters},
        "count": len(matches),
        "results": matches,
        "geojson": {"type": "FeatureCollection", "features": features}
    }

    if req.include_buffer_polygon:
        response["buffer_polygon"] = create_buffer_geojson(
            lat=req.lat,
            lng=req.lng,
            radius_meters=req.radius_meters
        )

    return response


@app.get("/api/spatial/geocode")
def geocode_location(q: str):
    res = geocode_onemap(q)
    if not res:
        raise HTTPException(status_code=404, detail="Location not found via OneMap API")
    return res


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected to HDB-Compass Live Sync.")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            query_text = message.get("query", "").strip()
            pinned_lat = message.get("lat")
            pinned_lng = message.get("lng")
            radius = float(message.get("radius_meters", 1000.0))

            if not query_text and pinned_lat is None:
                continue

            init_state: GeoAgentState = {
                "query": query_text,
                "step": "START",
                "lat": float(pinned_lat) if pinned_lat is not None else None,
                "lng": float(pinned_lng) if pinned_lng is not None else None,
                "radius_meters": radius,
                "qualifying_blocks": [],
                "nearby_amenities": [],
                "logs": []
            }

            try:
                # Stream each LangGraph node transition to the React frontend
                async for chunk in geo_agent.astream(init_state):
                    for node_name, node_output in chunk.items():
                        logs = node_output.get("logs", [])
                        step_log = logs[-1] if logs else {"step": node_name, "message": f"Completed {node_name}"}

                        lat = node_output.get("lat")
                        lng = node_output.get("lng")
                        viewport = None
                        if lat is not None and lng is not None:
                            viewport = {"lat": float(lat), "lng": float(lng), "zoom": 14.5}

                        await websocket.send_json({
                            "type": "STEP_UPDATE",
                            "node": node_name,
                            "geojson_buffer": node_output.get("geojson_buffer"),
                            "geojson_amenities": node_output.get("geojson_amenities"),
                            "viewport": viewport,
                            "logs": [step_log],
                            "summary": node_output.get("summary"),
                            "qualifying_blocks": node_output.get("qualifying_blocks"),
                            "nearby_amenities": node_output.get("nearby_amenities"),
                        })
                        await asyncio.sleep(0.05)

                await websocket.send_json({"type": "AGENT_FINISHED"})

            except Exception as e:
                print(f"[WebSocket] LangGraph error: {e}")
                await websocket.send_json({
                    "type": "ERROR",
                    "error": str(e)
                })

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Exception: {e}")
