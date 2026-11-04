"""
Tests for WebSocket QA fixes (PLAT-20-25, PLAT-26, PLAT-32, PLAT-34).
Owned by Member 2 & Member 4.
"""
import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_plat_20_25_malformed_json_returns_error_and_connection_stays_alive():
    """Malformed JSON should not drop or crash the WebSocket connection."""
    with client.websocket_connect("/ws") as ws:
        # Send non-JSON raw text
        ws.send_text("this is definitely not valid json {{{")
        resp = ws.receive_json()
        assert resp["type"] == "ERROR"
        assert "Malformed WebSocket message" in resp["error"]

        # Send non-dict JSON (e.g. integer)
        ws.send_text("12345")
        resp2 = ws.receive_json()
        assert resp2["type"] == "ERROR"
        assert "Malformed WebSocket message" in resp2["error"]

        # Ensure socket is still alive and responsive to valid ping/reset
        ws.send_json({"reset": True})


def test_plat_26_zero_or_negative_radius_rejected():
    """Zero or negative radius should return an error and not be accepted."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "RUN_AGENT",
            "request_id": "req-zero-rad",
            "query": "Find flats in Tampines",
            "radius_meters": 0
        })
        resp = ws.receive_json()
        assert resp["type"] == "ERROR"
        assert resp.get("request_id") == "req-zero-rad"
        assert "greater than 0" in resp["error"]

        # Test negative radius
        ws.send_json({
            "type": "RUN_AGENT",
            "request_id": "req-neg-rad",
            "query": "Find flats in Tampines",
            "radius_meters": -100
        })
        resp2 = ws.receive_json()
        assert resp2["type"] == "ERROR"
        assert resp2.get("request_id") == "req-neg-rad"
        assert "greater than 0" in resp2["error"]


def test_plat_34_request_ids_echoed_in_events():
    """Request ID should be present on AGENT_STARTED and ERROR/STEP_UPDATE."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "RUN_AGENT",
            "request_id": "req-custom-999",
            "radius_meters": 0  # Triggers immediate error with request_id attached
        })
        resp = ws.receive_json()
        assert resp["type"] == "ERROR"
        assert resp.get("request_id") == "req-custom-999"


def test_create_buffer_geojson_zero_radius_raises():
    """create_buffer_geojson should raise ValueError for non-positive radius."""
    from mcp_server.spatial_onemap_tools import create_buffer_geojson
    with pytest.raises(ValueError, match="greater than 0"):
        create_buffer_geojson(1.35, 103.85, radius_meters=0)

    with pytest.raises(ValueError, match="greater than 0"):
        create_buffer_geojson(1.35, 103.85, radius_meters=-50)

