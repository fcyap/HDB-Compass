"""The conversational app over the websocket (HDB_ANSWER_STYLE=conversational, the app on Claude): the reply streams a
checked sentence at a time before the final answer, steps carry plain names, the cards carry their own figures, and
with the switch unset the messages are as before. A stand-in model; no network."""
from __future__ import annotations

import os

import pytest

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from fastapi.testclient import TestClient  # noqa: E402

from agents import reply_composer as rc  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from backend import main  # noqa: E402

QUERY = "Find 4-room resale flats in Ang Mo Kio under $650,000 within 400m of an MRT station."
REPLY = ("Yes, there are 40 blocks that fit. Blk 216 is the cheapest at $438,000, while Blk 207 is the nearest to the "
         "MRT at 88m. The full list is in the details below and on the map.")


class Model:
    def invoke(self, messages):
        return type("Reply", (), {"content": REPLY})()

    def stream(self, messages):
        for i in range(0, len(REPLY), 9):
            yield type("Chunk", (), {"content": REPLY[i:i + 9]})()


def run(query=QUERY):
    with TestClient(main.app).websocket_connect("/ws?session=test-conversational") as ws:
        ws.send_json({"reset": True})
        ws.send_json({"type": "RUN_AGENT", "request_id": "r1", "query": query, "radius_meters": 500})
        out = []
        while True:
            msg = ws.receive_json()
            out.append(msg)
            if msg["type"] in ("AGENT_FINISHED", "ERROR"):
                return out


@pytest.fixture
def talk(monkeypatch):
    monkeypatch.setenv("HDB_ANSWER_STYLE", "conversational")
    monkeypatch.setenv("HDB_AUDITOR_LLM", "off")
    monkeypatch.setattr(sv, "get_llm", lambda **k: Model())


def test_the_reply_streams_before_the_final_answer(talk):
    msgs = run()
    deltas = [m for m in msgs if m["type"] == "REPLY_DELTA"]
    assert deltas and "".join(d["text"] for d in deltas).strip() == REPLY
    final = [m for m in msgs if m["type"] == "STEP_UPDATE" and m.get("summary")][-1]
    assert msgs.index(deltas[-1]) < msgs.index(final)
    assert final["summary"].startswith(REPLY) and rc.DETAILS_SEP in final["summary"]
    assert {m.get("answer_style") for m in msgs if m["type"] == "MODEL_STATUS"} == {"conversational"}


def test_steps_have_plain_names_and_cards_their_own_figures(talk):
    msgs = run()
    titles = {log["step"] for m in msgs if m["type"] == "STEP_UPDATE" for log in m.get("logs") or []}
    assert "Writing the reply" in titles and not any("Member" in t for t in titles)
    cards = [m["candidate_cards"] for m in msgs if m.get("candidate_cards")][-1]
    assert cards[0]["name"].startswith("Blk ") and cards[0]["price"] and cards[0]["lease"]


def test_with_the_switch_unset_nothing_streams_and_steps_keep_their_names(monkeypatch):
    monkeypatch.delenv("HDB_ANSWER_STYLE", raising=False)
    msgs = run()
    assert not [m for m in msgs if m["type"] in ("REPLY_DELTA", "REPLY_RESET")]
    assert not any("candidate_cards" in m for m in msgs)
    titles = {log["step"] for m in msgs if m["type"] == "STEP_UPDATE" for log in m.get("logs") or []}
    assert any("Member" in t or t == "Response Synthesizer" for t in titles)


def test_the_footer_counts_only_real_model_failures():
    assert not sv.real_model_failure(["financial: no valid tool call from model; deterministic plan used"])
    assert sv.real_model_failure(["synthesize: LLM failed, template used: APITimeoutError"])
    assert sv.real_model_failure(["extract_constraints: model extraction failed; rule-based fallback used"])


# --- the model dropdown: each chat's choice reaches every step of its turns ------------------------------------------
def run_with(model, query=QUERY):
    with TestClient(main.app).websocket_connect("/ws?session=test-dropdown") as ws:
        ws.send_json({"reset": True, "model": model})
        ws.send_json({"type": "RUN_AGENT", "request_id": "r1", "query": query, "radius_meters": 500, "model": model})
        out = []
        while True:
            msg = ws.receive_json()
            out.append(msg)
            if msg["type"] in ("AGENT_FINISHED", "ERROR"):
                return out


def test_choosing_claude_gives_its_style_inside_the_graph(monkeypatch):
    # the backend was started offline (LLM_PROFILE=none, no answer style); the chat picks Claude
    monkeypatch.delenv("HDB_ANSWER_STYLE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.setenv("HDB_AUDITOR_LLM", "off")
    monkeypatch.setattr(sv, "get_llm", lambda **k: Model())
    msgs = run_with("claude")
    status = [m for m in msgs if m["type"] == "MODEL_STATUS"][0]
    assert status["profile"] == "claude" and status["answer_style"] == "conversational"
    final = [m for m in msgs if m["type"] == "STEP_UPDATE" and m.get("summary")][-1]
    assert final["summary"].startswith(REPLY) and rc.DETAILS_SEP in final["summary"]   # composed: the choice ran in the nodes


def test_choosing_offline_or_nothing_keeps_the_started_model(monkeypatch):
    monkeypatch.delenv("HDB_ANSWER_STYLE", raising=False)
    for model in ("none", None, "no-such-model"):
        msgs = run_with(model)
        status = [m for m in msgs if m["type"] == "MODEL_STATUS"][0]
        assert status["mode"] == "deterministic" and status["answer_style"] == "template", model
        assert not [m for m in msgs if m["type"] == "REPLY_DELTA"]


def test_the_model_list_says_what_can_answer(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    data = TestClient(main.app).get("/api/models").json()
    by = {o["id"]: o for o in data["options"]}
    assert by["claude"]["available"] and by["none"]["available"]
    assert not by["gemini"]["available"] and "GEMINI_API_KEY" in by["gemini"]["reason"]
    assert data["default"] == "none"                      # the model the backend was started with
    status = TestClient(main.app).get("/api/model/status", params={"model": "claude"}).json()
    assert status["profile"] == "claude" and status["answer_style"] == "conversational"
