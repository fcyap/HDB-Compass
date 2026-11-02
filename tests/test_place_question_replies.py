"""
Replies to the open question about a place (tests/test_place_question_replies.py). Owned by Member 4.

In the app (2 Nov), "Tampines blk 884" got "What would you like to know about Tampines blk 884? For example: its past
sales, ...", and the reply "resale price" got the same question again, every time. The reply is now read as the
option it picks ("resale price" and "past sales" are the block's past sales), "Tampines blk 884" is the block
"Blk 884 Tampines", and a reply no option matches gets the block's past sales instead of the question again.

    LLM_PROFILE=none HDB_ADDONS=app .venv/bin/python -m pytest tests/test_place_question_replies.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ["LLM_PROFILE"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents.schemas import LLMExtractedConstraints  # noqa: E402
from tests.test_research_orchestration import FakeLLM  # noqa: E402


@pytest.fixture(autouse=True)
def app_mode(monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "app")


def app(query, previous=None, **reading):
    """One app turn (Variant C, fast path) with a model whose reading is given."""
    with patch.object(sv, "get_llm", return_value=FakeLLM(extraction=LLMExtractedConstraints(**reading))):
        return C.graph.invoke(sv.initial_state(query=query, variant="C", map_state={}, previous=previous, fast_path=True))


@pytest.fixture
def asked():
    first = app("Tampines blk 884", addresses=["Tampines blk 884"])
    assert (first.get("clarification") or {}).get("kind") == "place"
    return first


@pytest.mark.parametrize("reply,intent,tool", [
    ("resale price", "wants_block_history", "get_block_transaction_history"),
    ("past sales", "wants_block_history", "get_block_transaction_history"),
    ("is $600k fair?", "wants_price_assessment", "assess_resale_price"),
    ("price trend", "wants_price_trend", None),
    ("amenities", "wants_nearby_amenities", "get_nearby_amenities"),
    ("resale prices around it", "wants_area_statistics", "get_area_resale_statistics"),
])
def test_a_reply_picks_the_option_and_is_answered(asked, reply, intent, tool):
    out = app(reply, asked)
    assert not out.get("clarification"), "the same question must not be asked again"
    assert out["user_constraints"][intent]
    if tool:
        assert tool in [t["tool"] for t in out["tool_call_log"]]


@pytest.mark.parametrize("reply", ["ok", "hmm not sure", "yes"])
def test_a_reply_that_picks_nothing_gets_past_sales_not_the_question_again(asked, reply):
    out = app(reply, asked)
    assert not out.get("clarification")
    assert out["user_constraints"]["wants_block_history"]
    assert "get_block_transaction_history" in [t["tool"] for t in out["tool_call_log"]]


def test_town_first_block_is_read_as_the_block():
    c = {"addresses": ["Tampines blk 884"], "clarifications": ["What would you like to know about Tampines blk 884? ..."]}
    sv.answer_place_question(c, "past sales")
    assert c["addresses"] == ["Blk 884 Tampines"] and c["wants_block_history"] and not c["clarifications"]


def test_a_place_on_its_own_still_gets_the_question_first():
    # only a reply to the open question is read as an option; a first message naming only a place is asked about
    out = app("Tampines blk 884", addresses=["Tampines blk 884"])
    assert (out.get("clarification") or {}).get("kind") == "place"
    assert not any(v for k, v in out["user_constraints"].items() if k.startswith("wants_"))
