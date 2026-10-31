"""Address resolution, tool discovery and the chat transport must share school evidence."""

import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect
from fastmcp import Client

from agents import supervisor as sv
from agents.arch_3_specialist_5agent import run
from agents.schemas import FindSchoolsNearAddressOutput
from agents.tool_registry import ToolRunner
from mcp_server import school_lookup_tools as lookup


@pytest.fixture
def school_index(monkeypatch):
    payload = {
        "blocks": [
            {"block": "147", "street": "BISHAN ST 11", "town": "BISHAN",
             "postal": "570147", "lat": 1.34, "lng": 103.85},
            {"block": "147", "street": "STREET IN KALLANG/WHAMPOA", "town": "KALLANG/WHAMPOA",
             "postal": "350147", "lat": 1.33, "lng": 103.86},
            {"block": "148", "street": "BISHAN ST 11", "town": "BISHAN",
             "postal": "570148", "lat": 1.341, "lng": 103.85},
        ],
        "schools": [
            {"name": "Near Primary", "address": "10 SCHOOL RD", "lat": 1.341, "lng": 103.85},
            {"name": "Far Primary", "address": "20 SCHOOL RD", "lat": 1.354, "lng": 103.85},
            {"name": "Outside Primary", "address": "30 SCHOOL RD", "lat": 1.39, "lng": 103.85},
        ],
    }
    monkeypatch.setattr(lookup, "get_spatial_index", lambda: {"payload": payload})
    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setattr(sv, "_INPUT_GUARDRAIL", None)
    monkeypatch.setattr(sv, "_AUDITOR", None)
    return payload


@pytest.mark.parametrize("address", ["Blk 147 Bishan Street 11", "147 BISHAN", "570147"])
def test_resolved_address_returns_sorted_measured_primary_schools(school_index, address):
    result = lookup.find_primary_schools_near_address(address)
    FindSchoolsNearAddressOutput(**result)
    assert result["status"] == "ok"
    assert result["location"]["postal"] == "570147"
    assert [s["name"] for s in result["schools"]] == ["Near Primary", "Far Primary"]
    assert 110 < result["schools"][0]["distance_meters"] < 112
    assert 1500 < result["schools"][1]["distance_meters"] < 1600


@pytest.mark.parametrize("address", ["147", "999 Kallang", "Kallang", "147 street in kallang", "350147", "11 Bishan St 147"])
def test_unresolved_addresses_never_produce_schools(school_index, address):
    result = lookup.find_primary_schools_near_address(address)
    assert result["status"] == "needs_information"
    assert result["location"] is None
    assert result["schools"] == []


def test_same_number_and_town_still_requires_street_if_ambiguous(school_index):
    school_index["blocks"].append({**school_index["blocks"][0], "street": "BISHAN ST 12", "postal": "571147"})
    result = lookup.find_primary_schools_near_address("147 Bishan")
    assert result["status"] == "needs_information"
    assert len(result["address_candidates"]) == 2


def test_empty_school_coverage_is_not_no_matches(school_index):
    school_index["schools"] = []
    assert lookup.find_primary_schools_near_address("570147")["status"] == "unavailable"


def test_missing_index_is_unavailable(monkeypatch):
    def missing():
        raise FileNotFoundError()
    monkeypatch.setattr(lookup, "get_spatial_index", missing)
    assert lookup.find_primary_schools_near_address("570147")["status"] == "unavailable"


def test_empty_radius_result_does_not_expand_search(school_index):
    result = lookup.find_primary_schools_near_address("570147", radius_meters=50)
    assert result["status"] == "ok"
    assert result["schools"] == []
    assert result["radius_meters"] == 50


@pytest.mark.parametrize("radius", [0, -1, 10001, float("nan")])
def test_registry_rejects_invalid_radius(school_index, radius):
    runner = ToolRunner("find_primary_schools_near_address")
    assert "error" in runner(address="570147", radius_meters=radius)
    assert not runner.log[-1].ok


@pytest.mark.parametrize("query, status, school_count", [
    ("schools near blk 147 street in kallang", "needs_information", 0),
    ("Find primary schools near Blk 147 Bishan Street 11", "ok", 2),
    ("Which schools are near 570147?", "ok", 2),
    ("schools within 1km of 570147", "ok", 1),
    ("schools near 570147 within 1km", "ok", 1),
    ("schools near 147", "needs_information", 0),
])
def test_graph_resolves_location_without_market_or_llm(school_index, monkeypatch, query, status, school_count):
    monkeypatch.setattr(sv, "get_llm", lambda: pytest.fail("Address lookup must not depend on model inference"))
    result = run(query)
    assert result["active_subagents"] == ["spatial"]
    assert [r["tool"] for r in result["tool_call_log"]] == ["find_primary_schools_near_address"]
    assert result["nearby_school_results"]["status"] == status
    assert len(result["nearby_school_results"]["schools"]) == school_count
    assert result["candidate_blocks_intersection"] == []
    assert "No block" not in result["draft_response"]
    assert "HFE" not in result["draft_response"]
    markers = result["map_ui_payload"]["markers"]["features"]
    assert len(markers) == (school_count + 1 if status == "ok" else 0)


def test_mcp_discovery_and_execution(school_index):
    from mcp_server.server import mcp

    async def check():
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert "address" in tools["find_primary_schools_near_address"].inputSchema["properties"]
            result = await client.call_tool("find_primary_schools_near_address", {"address": "570147"})
            assert not result.is_error
            data = result.structured_content
            assert data["status"] == "ok"
            assert len(data["schools"]) == 2
    asyncio.run(check())


@pytest.mark.parametrize("query, expected_status", [
    ("schools near blk 147 street in kallang", "needs_information"),
    ("schools near 570147", "ok"),
])
def test_websocket_transports_schools_without_resale_cards(school_index, monkeypatch, query, expected_status):
    from backend import main

    monkeypatch.setattr(main, "model_status", lambda: {"status": "disabled", "mode": "deterministic"})

    class Socket:
        sent = []
        received = False

        async def accept(self):
            pass

        async def receive_text(self):
            if self.received:
                raise WebSocketDisconnect()
            self.received = True
            return json.dumps({"query": query})

        async def send_json(self, message):
            self.sent.append(message)

    socket = Socket()
    asyncio.run(main.websocket_endpoint(socket))
    assert not any(m["type"] == "ERROR" for m in socket.sent)
    updates = [m for m in socket.sent if m["type"] == "STEP_UPDATE"]
    final = updates[-1]
    assert final["school_lookup"]["status"] == expected_status
    assert not any(m.get("qualifying_blocks") for m in updates)
    assert not any("zero hallucinations" in json.dumps(m) for m in updates)
    assert "not installed" in main.audit_description({"passed": True})
    assert socket.sent[-1]["type"] == "AGENT_FINISHED"
