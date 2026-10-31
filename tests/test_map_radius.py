"""The map's radius control: a pin question uses the radius set on the map unless the text states a distance
(stated > map control > default), the answer says which radius it used, and radius chips re-ask with another one."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect

from agents import research_intents as ri
from agents import supervisor as sv
from tests.test_research_tools import research_env, snapshot  # noqa: F401  (shared fixtures)

PIN = (1.35001, 103.85001)   # Blk 1 BISHAN ST 1 in the synthetic index


@pytest.fixture
def app_mode(research_env, monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")
    from agents import arch_3_specialist_5agent as C
    return C


def ask(C, query, radius=None, pin=PIN, previous=None):
    map_state = {"center": list(pin), **({"radius_meters": radius} if radius else {})} if pin else {}
    return C.graph.invoke(sv.initial_state(query, "C", map_state=map_state, previous=previous, fast_path=True))


def constraints(query, radius=None, pin=PIN):
    c = sv.rule_based_extract(query).model_dump()
    ri.resolve_place(c, query, pin, [], None, radius_meters=radius)
    return c


def test_pin_radius_is_the_area_radius_when_no_distance_is_stated(research_env):
    c = constraints("What are resale prices around here?", radius=300)
    assert c["area_radius_meters"] == 300 and c["area_radius_from_pin"]
    notes = []
    arg = ri._area_arg(c, notes)
    assert arg["radius_meters"] == 300
    assert any("300 m" in n and "your pin" in n and "set on the map" in n for n in notes)
    assert not any("'Around' was taken" in n for n in notes)


def test_stated_distance_beats_the_map_radius(research_env):
    c = constraints("What are resale prices within 700m of here?", radius=300)
    assert c["area_radius_meters"] == 700 and not c.get("area_radius_from_pin")
    assert ri._area_arg(c, [])["radius_meters"] == 700


def test_no_map_radius_keeps_the_default_assumption(research_env):
    c = constraints("What are resale prices around here?", radius=None)
    notes = []
    assert ri._area_arg(c, notes)["radius_meters"] == ri.VAGUE_RADIUS_METERS
    assert any("'Around' was taken" in n for n in notes)


def test_out_of_range_radius_is_clamped_and_said(research_env):
    c = constraints("What are resale prices around here?", radius=9000)
    notes = []
    assert ri._area_arg(c, notes)["radius_meters"] == 3000
    assert any("9000 m is outside" in n for n in notes)
    c = constraints("What are resale prices within 5km of here?", radius=300)
    notes = []
    assert ri._area_arg(c, notes)["radius_meters"] == 3000 and any("5000 m is outside" in n for n in notes)


def test_nearby_uses_map_radius_and_stated_distance_overrides(research_env):
    c = constraints("What's near here?", radius=300)
    assert dict(ri.research_plan("spatial", c, "What's near here?"))["get_nearby_amenities"]["radius_meters"] == 300
    c = constraints("What's near here within 800m?", radius=300)
    assert dict(ri.research_plan("spatial", c, "x"))["get_nearby_amenities"]["radius_meters"] == 800


def test_pin_answers_state_the_radius(app_mode):
    nearby = ask(app_mode, "What's near here?", radius=300)["draft_response"]
    assert "within 300 m" in nearby and "as set on the map" in nearby
    area = ask(app_mode, "What are resale prices around here?", radius=300)["draft_response"]
    assert "within 300 m of" in area and "set on the map" in area


def test_radius_chips_leave_out_the_current_radius(research_env):
    c = constraints("What are resale prices within 500m of Blk 1 Bishan St 1?", pin=None)
    chips = ri.radius_chips(c)
    assert [x["label"] for x in chips] == ["Within 200 m", "Within 1 km"]
    assert chips[0]["text"] == "What are resale prices within 200m of Blk 1 Bishan St 1?"
    c = constraints("What's near here?", radius=1000)
    assert [x["label"] for x in ri.radius_chips(c)] == ["Within 200 m", "Within 500 m"]
    assert ri.radius_chips(constraints("What is the BTO grant?", pin=None)) == []


@pytest.mark.parametrize("query", [
    "What are resale prices around Blk 1 Bishan St 1?",
    "What are resale prices within 700m of Blk 1 Bishan St 1?",
    "What's near here?",
])
def test_every_chip_text_round_trips_through_the_fast_path(app_mode, query):
    out = ask(app_mode, query, radius=500)
    assert out["suggestions"] and not out["clarification"]   # chips are not a pending question
    radius = {"Within 200 m": 200, "Within 500 m": 500, "Within 1 km": 1000}
    chips = [c for c in out["suggestions"] if c["label"] in radius]   # area answers also carry drill-down chips (blocks)
    assert chips
    for chip in chips:
        wanted = radius[chip["label"]]
        again = ask(app_mode, chip["text"], radius=500, previous=out)
        assert again["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE
        assert f"within {wanted} m" in again["draft_response"]


def test_websocket_pin_radius_300_gives_a_300_m_answer(app_mode, monkeypatch):
    from backend import main
    monkeypatch.setattr(main, "model_status", lambda: {"status": "disabled", "mode": "deterministic"})

    class Socket:
        def __init__(self):
            self.sent, self.queue = [], [json.dumps({"query": "What's near here?", "lat": PIN[0], "lng": PIN[1],
                                                     "radius_meters": 300})]
            self.query_params = {"session": "test-map-radius"}

        async def accept(self):
            pass

        async def receive_text(self):
            if not self.queue:
                raise WebSocketDisconnect()
            return self.queue.pop(0)

        async def send_json(self, message):
            self.sent.append(message)

    socket = Socket()
    asyncio.run(main.websocket_endpoint(socket))
    assert socket.sent[-1]["type"] == "AGENT_FINISHED"
    final = [m for m in socket.sent if m["type"] == "STEP_UPDATE"][-1]
    assert "within 300 m" in final["summary"]
    chip = final["clarification"]
    assert chip["kind"] == "drilldown" and [o["label"] for o in chip["options"]] == ["Within 200 m", "Within 500 m", "Within 1 km"]
