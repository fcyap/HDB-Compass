"""backend/output_schema.py: what the server sends to the browser is checked and cleaned."""
import asyncio
import json
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

from backend import output_schema as out

EVAL = Path(__file__).resolve().parent.parent / "evaluation"

STEP = {
    "type": "STEP_UPDATE", "node": "synthesize",
    "geojson_buffer": {"type": "Feature", "properties": {"radius": 500}}, "geojson_amenities": {"type": "FeatureCollection", "features": []},
    "viewport": {"lat": 1.35, "lng": 103.82, "zoom": 14.5},
    "logs": [{"step": "Synthesizer", "message": "Wrote the answer", "status": "completed"}],
    "summary": "**4 candidate block(s)** match all stated constraints:\n- Blk 288B JURONG EAST ST 21 — median $598,888, 73 yrs lease",
    "qualifying_blocks": ["288B", "266"], "school_lookup": None, "clarification": None, "flat_type": "4 ROOM",
    "some_new_field_from_another_member": {"anything": [1, 2, 3]},
}


def test_a_well_formed_message_passes_unchanged():
    assert out.clean_message(STEP) == STEP


@pytest.mark.parametrize("message", [
    {"type": "AGENT_STARTED"}, {"type": "AGENT_FINISHED"}, {"type": "CONNECTION_ESTABLISHED"},
    {"type": "MODEL_STATUS", "status": "configured", "mode": "model", "model": "qwen3", "message": "Model configured."},
])
def test_simple_messages_pass_unchanged(message):
    assert out.clean_message(message) == message


@pytest.mark.parametrize("field, bad", [
    ("summary", ["not", "a", "string"]), ("logs", "not a list"), ("viewport", {"lat": "north", "lng": 103.8}),
    ("qualifying_blocks", "288B"), ("clarification", "a string"),
])
def test_a_field_with_the_wrong_type_is_dropped_and_the_rest_survives(field, bad):
    cleaned = out.clean_message({**STEP, field: bad})
    assert field not in cleaned
    assert cleaned["type"] == "STEP_UPDATE" and cleaned["node"] == "synthesize" and cleaned["flat_type"] == "4 ROOM"
    if field != "summary":
        assert cleaned["summary"] == STEP["summary"]


def test_a_message_that_is_not_a_dict_becomes_a_plain_error():
    assert out.clean_message("oops")["type"] == "ERROR"


def test_identifiers_are_removed_from_the_text_people_read_but_not_from_map_data():
    msg = {**STEP, "summary": "Call 9123 4567 about S1234567D", "financial_summary": "Pay into account 0123456789",
           "logs": [{"step": "x", "message": "email a.b@example.com", "status": "completed"}],
           "clarification": {"kind": "financial", "question": "Is your NRIC S7654321B?",
                             "options": [{"label": "Phone 8123 4567", "text": "Phone 8123 4567"}]},
           "geojson_amenities": {"type": "FeatureCollection", "features": [{"properties": {"name": "Hawker 91234567"}}]}}
    cleaned = out.clean_message(msg)
    text = json.dumps({k: cleaned[k] for k in ("summary", "financial_summary", "logs", "clarification")})
    for raw in ("9123", "S1234567D", "0123456789", "a.b@example.com", "S7654321B", "8123"):
        assert raw not in text
    assert "[PHONE]" in cleaned["summary"] and "[NRIC]" in cleaned["summary"]
    assert cleaned["geojson_amenities"] == msg["geojson_amenities"]       # map data is the other members' and is left alone


def test_script_like_html_and_javascript_links_are_removed_markdown_stays():
    dirty = "Hello <script>alert(1)</script> **bold** [click](javascript:alert(1)) <iframe src=x></iframe> <b onclick=\"x()\">b</b>"
    cleaned = out.clean_text(dirty)
    assert "<script" not in cleaned.lower() and "alert(1)</" not in cleaned and "<iframe" not in cleaned.lower()
    assert "javascript:" not in cleaned.lower() and "onclick" not in cleaned
    assert "**bold**" in cleaned and "[click](#)" in cleaned


@pytest.mark.parametrize("exc, expected", [
    (RuntimeError("Error code: 429 - Rate limit reached for model in organization org_01m3x0h7cnermrq24sy49jrjsh"), out.BUSY),
    (ConnectionError("Connection refused to http://10.193.104.102:8001 key gsk_abcdef"), out.UNREACHABLE),
    (TimeoutError("timed out after 90 s"), out.UNREACHABLE),
    (KeyError("/Users/zenia/Desktop/HDB-Compass/agents/supervisor.py line 4211"), out.GENERIC),
])
def test_an_error_reaches_the_browser_as_a_plain_sentence_never_as_the_exception(exc, expected):
    message = out.safe_error_message(exc)
    assert message == expected
    for leak in ("org_", "gsk_", "10.193", "/Users", "supervisor.py", "429"):
        assert leak not in message


def test_send_clean_sends_the_cleaned_message():
    sent = []

    class Socket:
        async def send_json(self, message):
            sent.append(message)

    asyncio.run(out.send_clean(Socket(), {"type": "ERROR", "error": "Reach me on 9123 4567"}))
    assert sent == [{"type": "ERROR", "error": "Reach me on [PHONE]"}]


def test_real_answers_are_never_changed_by_the_cleaning(monkeypatch):
    """Over a spread of the benchmark's real answers, cleaning must not touch a word (no false redactions)."""
    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setenv("HDB_ADDONS", "off")
    from agents.arch_3_specialist_5agent import run
    items = [i for i in json.loads((EVAL / "benchmark_hdb_90.json").read_text(encoding="utf-8")) if i["category"] != "F"][::9]
    assert len(items) >= 8
    for it in items:
        answer = run(it["query"])["draft_response"]
        assert out.clean_text(answer) == answer, it["id"]


def _socket(queries):
    class Socket:
        def __init__(self):
            self.sent, self.queue = [], [json.dumps({"query": q}) for q in queries]
            self.query_params = {"session": "test-output-schema"}

        async def accept(self):
            pass

        async def receive_text(self):
            if not self.queue:
                raise WebSocketDisconnect()
            return self.queue.pop(0)

        async def send_json(self, message):
            self.sent.append(message)
    return Socket()


def test_a_failing_request_sends_a_plain_error_not_the_exception(monkeypatch):
    from backend import main
    monkeypatch.setattr(main, "model_status", lambda: {"status": "disabled", "mode": "deterministic"})

    def boom(*a, **k):
        raise RuntimeError("Error code: 429 - organization org_01m3x0h7cnermrq24sy49jrjsh key gsk_secret123")
    monkeypatch.setattr(main.sv, "initial_state", boom)
    socket = _socket(["Find 4-room flats in Bishan under 850k"])
    asyncio.run(main.websocket_endpoint(socket))
    errors = [m for m in socket.sent if m["type"] == "ERROR"]
    assert len(errors) == 1 and errors[0]["error"] == out.BUSY
    assert "org_" not in json.dumps(socket.sent) and "gsk_" not in json.dumps(socket.sent)


def test_a_normal_question_still_reaches_the_browser_in_the_shape_the_page_reads(monkeypatch):
    from backend import main
    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setattr(main, "model_status", lambda: {"status": "disabled", "mode": "deterministic"})
    socket = _socket(["Find 4-room flats in Bishan under 850k"])
    asyncio.run(main.websocket_endpoint(socket))
    kinds = [m["type"] for m in socket.sent]
    assert kinds[0] == "MODEL_STATUS" or kinds[0] == "AGENT_STARTED"
    assert kinds[-1] == "AGENT_FINISHED" and "ERROR" not in kinds
    final = [m for m in socket.sent if m["type"] == "STEP_UPDATE"][-1]
    assert isinstance(final["summary"], str) and final["summary"]
    assert all(isinstance(m.get("logs", []), list) for m in socket.sent)
