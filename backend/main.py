"""
FastAPI Server for HDB-Compass Geospatial & Multi-Agent Engine.
Combines OneMap REST API, SVY21 Planar KD-Trees, and LangGraph Multi-Agent Streaming.
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CHAT_LOG_DIR = PROJECT_ROOT / "data" / "local" / "chat_logs"   # gitignored

# The last finished state per page load (session id from the frontend), so a reply such as "we have $25,000 in cash"
# completes the earlier request even if the WebSocket dropped and reconnected in between. Held in this process only:
# a backend restart (including --reload on a file save) forgets every conversation, and a page reload starts a new
# one. Oldest sessions are dropped beyond SESSION_LIMIT.
SESSION_LIMIT = 200
SESSIONS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def remember(session_id: Optional[str], state: Optional[Dict[str, Any]]) -> None:
    if not session_id:
        return
    if state is None:
        SESSIONS.pop(session_id, None)
        return
    SESSIONS[session_id] = state
    SESSIONS.move_to_end(session_id)
    while len(SESSIONS) > SESSION_LIMIT:
        SESSIONS.popitem(last=False)


def chip_clarification(c: Dict[str, Any]) -> Dict[str, Any]:
    """The open follow-up question as the page shows it: kind, question and options that each carry a label to show and
    the text to send when clicked (menu options have both; location options only a label, which is also what to send)."""
    options = [{"label": str(o.get("label") or o.get("text") or ""), "text": str(o.get("text") or o.get("label") or "")}
               for o in c.get("options") or [] if isinstance(o, dict)]
    return {"kind": c.get("kind"), "question": c.get("question"), "options": [o for o in options if o["label"]]}


def log_chat_turn(question: str, state: Dict[str, Any], seconds: float, turn: int) -> None:
    """One JSON line per chat turn, so a wrong answer seen in the UI can be traced: what was asked, how it was read,
    where it was routed, which tools ran and whether they failed, internal errors, and the answer. Never breaks a chat."""
    import time
    try:
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_profile": os.getenv("LLM_PROFILE", ""),
            "turn": turn,
            "question": question,
            "constraints": state.get("user_constraints"),
            "routing": state.get("active_subagents"),
            "tool_calls": [{"agent": t.get("agent"), "tool": t.get("tool"), "ok": t.get("ok", True),
                            "arguments": t.get("arguments")} for t in state.get("tool_call_log") or []],
            "errors": state.get("errors") or [],
            "clarification": state.get("clarification"),
            "matching_blocks": [f"{b.get('block')} {b.get('street')}" for b in state.get("candidate_blocks_intersection") or []
                                if isinstance(b, dict)],
            "answer": state.get("draft_response"),
            "seconds": round(seconds, 1),
        }
        with (CHAT_LOG_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[chat log] not written: {e}")

from mcp_server.spatial_onemap_tools import (
    create_buffer_geojson,
    wgs84_to_svy21,
    geocode_onemap,
    get_spatial_index,
)
from agents.spatial_agent import geo_agent, GeoAgentState
from agents.arch_3_specialist_5agent import graph as specialist_5agent_graph, sv
from agents.llm import model_status

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
    category: Optional[str] = None
    include_buffer_polygon: bool = True


NODE_INFO = {
    "input_guardrail": ("Input Guardrail (Member 5)", "Validating query safety and Singapore housing domain scope"),
    "extract_constraints": ("Constraint Extractor", "Extracting town, flat-type, financial, and amenity criteria"),
    "route": ("Supervisor Router", "Routing sub-tasks across specialist agents in parallel"),
    "market_sql_agent": ("Market SQL Agent (Member 1)", "Querying DuckDB resale transactions for price medians & remaining lease"),
    "spatial_agent": ("Geospatial Agent (Member 2)", "Filtering island-wide SVY21 KD-Tree for amenity and transit proximity"),
    "financial_agent": ("Financial Calculator (Member 1)", "Calculating CPF housing grants and maximum affordability thresholds"),
    "policy_rag_agent": ("Policy RAG Agent (Member 3)", "Retrieving HDB MOP rules, Prime/Plus classification, and resale policies"),
    "intersect_candidates": ("Candidate Intersector", "Computing cross-agent intersection of price, lease, and spatial bounds"),
    "relax_constraints": ("Constraint Relaxation (Category E2)", "Performing sensitivity sweep across budget and radius parameters"),
    "synthesize": ("Response Synthesizer", "Generating grounded executive summary and interactive map payload"),
    "runtime_auditor": ("Runtime Auditor (Member 5)", "Auditing draft response against evidence to prevent hallucinations"),
    "finalize": ("Completed", "Multi-agent decision support results finalized."),
}


def honest_step(node_name: str, node_output: Dict[str, Any], desc: str) -> str:
    """The trace must not claim a check that never ran. Member 5's guardrail and auditor fall back to pass-throughs
    until agents/guardrails.py exists, and the policy search is a stand-in while it is mocked; say so."""
    if node_name == "input_guardrail" and sv._INPUT_GUARDRAIL is None:
        return "Not installed yet (pass-through): the message is not safety-checked."
    if node_name == "runtime_auditor":
        if sv._AUDITOR is None:
            return "Not installed yet (pass-through): the answer is not independently checked."
        verdict = node_output.get("auditor_verdict") or {}
        if verdict.get("passed", True):
            return "Answer check passed."
        return f"Answer check asked for a rewrite: {verdict.get('critique') or 'no reason given'}"
    if node_name == "policy_rag_agent" and \
            sv.load_config().get("tools", {}).get("use_mock", {}).get("retrieve_hdb_policy_knowledge") == "mock":
        return "Stand-in only (the real policy search is not built yet); no rules are quoted."
    return desc


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


@app.get("/api/model/status")
def get_model_status(check: bool = False):
    return model_status(check=check)


import re


def clean_amenity_name(raw_name: str, category: str) -> str:
    """Format corporate registry names to clean consumer-facing names."""
    if not raw_name:
        return ""
    name = raw_name.strip()
    upper = name.upper()

    if category == "supermarket":
        if "FAIRPRICE" in upper:
            branch_match = re.search(r"\((.*?)\)", name)
            branch = f" ({branch_match.group(1).title()})" if branch_match and "CO-OPERATIVE" not in branch_match.group(1).upper() else ""
            return f"NTUC FairPrice{branch}"
        if "COLD STORAGE" in upper:
            branch_match = re.search(r"\((.*?)\)", name)
            branch = f" ({branch_match.group(1).title()})" if branch_match and "1983" not in branch_match.group(1) else ""
            return f"Cold Storage{branch}"
        if "SHENG SIONG" in upper:
            return "Sheng Siong Supermarket"
        if "PRIME" in upper:
            return "Prime Supermarket"
        if "GIANT" in upper:
            return "Giant Supermarket"
        if "DON DON DONKI" in upper:
            return "Don Don Donki"
        # Generic cleanup of corporate legal suffixes
        cleaned = re.sub(r"\s*(?:SINGAPORE|\(1983\)|CO-OPERATIVE|PTE\.?\s*LTD\.?|LIMITED)\s*", " ", name, flags=re.I).strip()
        return cleaned.title()

    if category == "hawker":
        return name.title()
    if category == "mrt":
        return name.title()
    return name.title()


@app.post("/api/spatial/radius")
def query_radius(req: RadiusQueryRequest):
    """
    Official OneMap & Singapore Government Spatial Proximity Endpoint.
    Uses metric SVY21 planar KD-Trees for 100% sovereign, sub-millisecond local queries.
    """
    import numpy as np
    idx = get_spatial_index()
    amenity_tree = idx.get("amenity_kdtree")
    amenities = idx["payload"]["amenities"]
    school_tree = idx.get("school_kdtree")
    schools = idx["payload"].get("schools", [])

    cx, cy = wgs84_to_svy21(req.lat, req.lng)
    matches = []
    seen_names = set()
    cat_lower = (req.category or "all").lower()

    # 1. Query Official Singapore Public Amenities (MRT / LRT, Hawkers, Supermarkets)
    if cat_lower in ("all", "mrt", "hawker", "supermarket") and amenity_tree:
        idxs = amenity_tree.query_ball_point([cx, cy], r=req.radius_meters)
        for i in idxs:
            a = dict(amenities[i])
            item_cat = a.get("category", "").lower()
            if cat_lower != "all" and item_cat != cat_lower:
                continue
            dist = float(np.hypot(a["x"] - cx, a["y"] - cy))
            a["distance_meters"] = round(dist, 1)
            # Standard Singapore pedestrian walking speed: 80 m/min (~4.8 km/h)
            a["walking_mins"] = max(1, round(dist / 80.0))
            a["name"] = clean_amenity_name(a.get("name", ""), item_cat)
            norm_name = a["name"].strip().upper()
            if norm_name not in seen_names:
                seen_names.add(norm_name)
                matches.append(a)

    # 2. Query Official MOE Primary Schools (Full SVY21 ground-truth for Phase 2C)
    if cat_lower in ("all", "school") and school_tree and schools:
        s_idxs = school_tree.query_ball_point([cx, cy], r=req.radius_meters)
        for i in s_idxs:
            s = dict(schools[i])
            dist = float(np.hypot(s["x"] - cx, s["y"] - cy))
            s["category"] = "school"
            s["distance_meters"] = round(dist, 1)
            s["walking_mins"] = max(1, round(dist / 80.0))
            s["name"] = s.get("name", "").strip().title()
            norm_name = s["name"].strip().upper()
            if norm_name not in seen_names:
                seen_names.add(norm_name)
                matches.append(s)

    matches.sort(key=lambda x: x["distance_meters"])

    features = []
    for item in matches:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [item["lng"], item["lat"]]},
            "properties": {k: v for k, v in item.items() if k not in ("lat", "lng")}
        })

    response = {
        "query": {"center": [req.lat, req.lng], "radius_meters": req.radius_meters, "category": req.category},
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


@app.get("/api/spatial/inspect_point")
def inspect_point(lat: float, lng: float):
    """
    Reverse inspects a clicked map coordinate.
    If an HDB block is within 150m, returns its details (block, street, town, postal).
    Otherwise returns the clicked location coordinates.
    """
    import numpy as np
    idx = get_spatial_index()
    block_tree = idx.get("block_kdtree")
    blocks = idx["payload"].get("blocks", [])

    cx, cy = wgs84_to_svy21(lat, lng)
    if block_tree and blocks:
        dist, i = block_tree.query([cx, cy])
        if dist <= 140.0:  # within 140m of an HDB block
            b = blocks[i]
            block_name = f"Blk {b.get('block', '')} {b.get('street', '')}".strip()
            return {
                "is_hdb": True,
                "name": block_name,
                "block": b.get("block"),
                "street": b.get("street"),
                "town": b.get("town"),
                "postal": b.get("postal"),
                "lat": float(b.get("lat", lat)),
                "lng": float(b.get("lng", lng)),
                "distance_to_click": round(float(dist), 1)
            }

    return {
        "is_hdb": False,
        "name": f"Pinned Location ({lat:.4f}° N, {lng:.4f}° E)",
        "lat": lat,
        "lng": lng
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected to HDB-Compass Live Sync.")
    # The last finished state, passed back on the next message so a reply completes the earlier request
    # (docs/INTERFACES.md §7). Kept per browser tab (?session=), so it survives a reconnect; Clear resets it.
    session_id = websocket.query_params.get("session")
    previous: Optional[Dict[str, Any]] = SESSIONS.get(session_id) if session_id else None
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("reset"):
                previous = None
                remember(session_id, None)
                continue
            query_text = message.get("query", "").strip()
            pinned_lat = message.get("lat")
            pinned_lng = message.get("lng")
            radius = float(message.get("radius_meters", 500.0))

            if not query_text and pinned_lat is None:
                continue

            await websocket.send_json({"type": "AGENT_STARTED"})

            try:
                # -------------------------------------------------------------
                # 1. Full Multi-Agent Mode (Variant C Specialist Graph)
                # -------------------------------------------------------------
                if query_text:
                    model_info = await asyncio.to_thread(model_status)
                    await websocket.send_json({"type": "MODEL_STATUS", **model_info})
                    if model_info["status"] == "error":
                        await websocket.send_json({"type": "ERROR", "error": model_info["message"]})
                        continue
                    map_state = {}
                    if pinned_lat is not None and pinned_lng is not None:
                        map_state["center"] = [float(pinned_lat), float(pinned_lng)]
                        map_state["radius_meters"] = radius

                    # fast_path: the app answers fully recognised questions without a model call; previous: the last
                    # turn's final state, so a reply completes an open follow-up question (fast path is skipped then).
                    st = sv.initial_state(query=query_text, variant="C", map_state=map_state, previous=previous,
                                          fast_path=True)
                    started = asyncio.get_event_loop().time()

                    current_buffer = None
                    current_markers = None
                    current_viewport = None
                    current_summary = None
                    current_qualifying: List[str] = []
                    current_school_lookup = None
                    current_clarification = None   # kind, question, options of an open follow-up, for the answer chips
                    active_agents: List[str] = []

                    final_state: Optional[Dict[str, Any]] = None
                    async for mode, chunk in specialist_5agent_graph.astream(st, stream_mode=["updates", "values"]):
                        if mode == "values":
                            final_state = chunk  # the full state after each step; the last one is the answer's
                            continue
                        for node_name, node_output in chunk.items():
                            if node_output is None and node_name in ("input_guardrail", "runtime_auditor"):
                                node_output = {}      # a pass-through stand-in returns nothing; still show its step
                            if not isinstance(node_output, dict):
                                continue

                            if node_output.get("clarification"):
                                current_clarification = chip_clarification(node_output["clarification"])
                            elif node_output.get("suggestions"):
                                # drill-down and radius chips are whole questions, not answers to one: they travel in the same
                                # chip field but are no pending clarification, so a click takes the fast path
                                current_clarification = chip_clarification({"kind": "drilldown", "question": None,
                                                                            "options": node_output["suggestions"]})

                            node_errors = node_output.get("errors") or []
                            if any(
                                any(marker in str(error) for marker in ("LLM", "rule-based fallback used", "deterministic plan used"))
                                for error in node_errors
                            ):
                                await websocket.send_json({
                                    "type": "MODEL_STATUS", **model_info,
                                    "status": "fallback",
                                    "message": "A model step failed or produced no valid tool call; rule-based processing was used for part of this answer.",
                                })

                            step_title, step_desc = NODE_INFO.get(
                                node_name, (node_name, f"Completed {node_name}")
                            )
                            step_desc = honest_step(node_name, node_output, step_desc)

                            if node_name == "extract_constraints":
                                c = node_output.get("user_constraints", {})
                                t_str = ", ".join(c.get("towns", [])) or "Any"
                                b_val = c.get("max_budget_sgd")
                                b_str = f"${b_val:,}" if b_val else "Any"
                                step_desc = f"Extracted criteria: Towns={t_str} | Budget={b_str} | Radii={c.get('amenity_radii_meters', {})}"

                            elif node_name == "route":
                                active_agents = node_output.get("active_subagents", [])
                                step_desc = f"Supervisor dispatched parallel tasks to: {', '.join(active_agents)}"

                            elif node_name == "spatial_agent" and "nearby_school_results" in node_output:
                                current_school_lookup = node_output["nearby_school_results"]
                                step_desc = f"Address-centred school lookup: {current_school_lookup.get('status', 'unavailable')}."

                            elif node_name == "intersect_candidates":
                                has_spatial_or_market = any(a in active_agents for a in ("spatial", "market_sql"))
                                inter = node_output.get("candidate_blocks_intersection", [])
                                if current_school_lookup is not None:
                                    step_desc = "Skipped block intersection; this request asks for nearby schools."
                                    current_qualifying = []
                                elif not has_spatial_or_market:
                                    step_desc = "Skipped (no spatial/market candidates requested; policy/financial query)."
                                    current_qualifying = None
                                elif len(inter) == 0:
                                    step_desc = "Intersected 0 blocks satisfying all criteria (proceeding to constraint relaxation)."
                                    current_qualifying = []
                                    current_markers = {"type": "FeatureCollection", "features": []}
                                    current_buffer = None
                                else:
                                    step_desc = f"Intersected {len(inter)} blocks satisfying both market price and spatial requirements."
                                    current_qualifying = [
                                        (f"Blk {b.get('block', '')} {b.get('street', '')}".strip() if isinstance(b, dict)
                                         else (str(b) if str(b).startswith("Blk ") else f"Blk {b}"))
                                        for b in inter
                                    ]

                            elif node_name == "synthesize":
                                mp = node_output.get("map_ui_payload", {})
                                if mp.get("buffer_polygon"):
                                    current_buffer = mp["buffer_polygon"]
                                if mp.get("markers"):
                                    current_markers = mp["markers"]
                                if mp.get("viewport"):
                                    current_viewport = mp["viewport"]
                                current_summary = node_output.get("draft_response")
                                if mp.get("candidate_blocks") and not current_qualifying:
                                    current_qualifying = [
                                        (f"Blk {b.get('block', '')} {b.get('street', '')}".strip() if isinstance(b, dict)
                                         else (str(b) if str(b).startswith("Blk ") else f"Blk {b}"))
                                        for b in mp["candidate_blocks"]
                                    ]

                                has_spatial = "spatial" in active_agents
                                if not has_spatial and not mp.get("markers") and not mp.get("buffer_polygon"):
                                    step_desc = "Map unchanged (no spatial agent invoked)."
                                else:
                                    n_markers = len(mp.get("markers", {}).get("features", [])) if mp.get("markers") else 0
                                    step_desc = f"Cross-agent synthesis complete with {n_markers} synchronized map markers and buffer."

                            elif node_name == "runtime_auditor":
                                step_desc = honest_step(node_name, node_output, step_desc)

                            elif node_name == "finalize":
                                msgs = node_output.get("messages", [])
                                if msgs:
                                    current_summary = msgs[-1].get("content", current_summary)
                                step_desc = "Multi-agent decision support results finalized."

                            await websocket.send_json({
                                "type": "STEP_UPDATE",
                                "node": node_name,
                                "geojson_buffer": current_buffer,
                                "geojson_amenities": current_markers,
                                "viewport": current_viewport,
                                "logs": [{
                                    "step": step_title,
                                    "message": step_desc,
                                    "status": "completed"
                                }],
                                "summary": current_summary,
                                "qualifying_blocks": current_qualifying if current_qualifying else None,
                                "school_lookup": current_school_lookup,
                                "clarification": current_clarification,
                            })
                            await asyncio.sleep(0.04)
                    if final_state is not None:
                        previous = final_state
                        remember(session_id, final_state)
                        turn = sum(1 for m in final_state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user")
                        log_chat_turn(query_text, final_state, asyncio.get_event_loop().time() - started, turn)

                # -------------------------------------------------------------
                # 2. Solo Spatial Fallback (Direct Map Click / Pin)
                # -------------------------------------------------------------
                else:
                    init_geo: GeoAgentState = {
                        "query": "",
                        "step": "START",
                        "lat": float(pinned_lat),
                        "lng": float(pinned_lng),
                        "radius_meters": radius,
                        "qualifying_blocks": [],
                        "nearby_amenities": [],
                        "logs": []
                    }

                    async for chunk in geo_agent.astream(init_geo):
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
                print(f"[WebSocket] Execution error: {e}")
                await websocket.send_json({
                    "type": "ERROR",
                    "error": str(e)
                })

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Exception: {e}")
