"""Feature: 'Is this price fair?' for one named block (fast path, no model)."""
import warnings

import pytest

warnings.filterwarnings("ignore")
from agents.arch_3_specialist_5agent import build_graph  # noqa: E402
import agents.supervisor as sv  # noqa: E402


@pytest.fixture(scope="module")
def graph():
    return build_graph()


def ask(graph, q):
    return graph.invoke(sv.initial_state(query=q, variant="C", previous=None, fast_path=True))


def tools(out):
    return [l.get("tool") for l in out.get("tool_call_log") or []]


CARD = "Is the price fair for a 4 ROOM at Blk 231 TAMPINES ST 21?"


@pytest.mark.parametrize("q", [
    CARD,
    "What should I offer for a 4-room at Blk 231 Tampines St 21?",
    "How much is a 4-room at Blk 231 Tampines St 21 worth?",
])
def test_routes_to_price_assessment_without_asking_price(graph, q):
    out = ask(graph, q)
    assert tools(out) == ["assess_resale_price"]
    assert not out.get("clarification")
    text = out["draft_response"]
    assert "A fair range for a 4 ROOM" in text
    assert "nearby median" in text and "Lease:" in text and "nearby sales of 4 ROOM" in text
    assert "2025" in text or "2026" in text   # the sample window


def test_asking_price_leads_with_verdict(graph):
    out = ask(graph, "Is $650,000 a fair price for a 4-room at Blk 231 Tampines St 21?")
    lines = out["draft_response"].splitlines()
    assert lines[1].startswith("- **Verdict: $650,000 is")
    assert "higher than" in lines[1] and "middle 50%" in lines[2]


def test_follow_up_chips(graph):
    out = ask(graph, CARD)
    chips = {c["label"]: c["text"] for c in out["suggestions"]}
    assert chips["Can we afford it?"] == "Can we afford a 4 ROOM at Blk 231 TAMPINES ST 21?"
    assert "Blk 231 TAMPINES ST 21 sell for" in chips["Past sales"]


def test_chip_questions_route(graph):
    chips = {c["label"]: c["text"] for c in ask(graph, CARD)["suggestions"]}
    assert "get_block_transaction_history" in tools(ask(graph, chips["Past sales"]))


def test_block_without_sales_still_gives_range_and_no_lease_crash(graph):
    out = ask(graph, "Is the price fair for a 4 ROOM at Blk 201 TAMPINES ST 21?")
    assert tools(out) == ["assess_resale_price"]
    assert "fair range" in out["draft_response"]


def test_unknown_block_has_no_chips(graph):
    out = ask(graph, "Is the price fair for a 4 ROOM at Blk 99999 NOWHERE ST 1?")
    assert "fair range" not in out["draft_response"]
    assert not [c for c in out.get("suggestions") or [] if c["label"] == "Past sales"]


def test_lease_gap_is_called_out():
    import agents.research_intents as ri
    meta = {"window_start_month": "2025-10", "window_end_month": "2026-09", "notes": ["within 500 m"]}
    a = {"status": "ok", "message": "m", "location": {"name": "Blk 1 X ST"}, "flat_type": "4 ROOM", "asking_price": None,
         "nearby": {"transaction_count": 20, "median": 600000, "lower_quartile": 580000, "upper_quartile": 620000,
                    "minimum": 500000, "maximum": 700000},
         "same_block": None, "remaining_lease_years": 50.0, "nearby_median_lease_years": 75.0, "metadata": meta}
    text = "\n".join(ri.price_check_lines(a))
    assert "50 years left versus 75" in text and "lower price" in text
