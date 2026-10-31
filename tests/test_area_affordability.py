"""
"Can I afford a resale near Novena MRT?": a named MRT station is the one the distance is measured to (not whichever
station is nearest each block), and an afford question about an area runs the calculators, shows the household form,
and once the facts are in, keeps only the area's blocks within the purchase ceiling. No model needed.
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("LANGSMITH_TRACING", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import agents.research_intents as ri  # noqa: E402
import agents.supervisor as sv  # noqa: E402
from mcp_server import spatial_onemap_tools as spatial  # noqa: E402

HOUSEHOLD = ("We are both Singapore citizens. We are first-time buyers. We are aged 32 and 30. We have no children. "
             "Our household income is $9,000 a month. Our CPF OA balance is $80,000. We have $40,000 in cash savings. "
             "We have no monthly debt. We have both worked continuously for the past 12 months. We are not buying near "
             "our parents. Our grant eligibility is confirmed. Our HDB loan eligibility is confirmed.")


def _filter(*args, **kwargs):
    f = spatial.filter_blocks_by_amenity_proximity
    return (getattr(f, "fn", None) or f)(*args, **kwargs)


# --- named stations ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("query,station", [
    ("can i afford a resale near novena mrt", "Novena"),
    ("resale flats within 1km of Toa Payoh MRT station", "Toa Payoh"),
    ("4 room near Bishan Interchange", "Bishan"),
    ("near one-north mrt", "One North"),
    ("near an mrt", None),
    ("near the mrt station", None),
    ("flats near mrt in bishan", None),
])
def test_named_station_is_read(query, station):
    assert ri.named_mrt_station(query) == station


def test_station_key_strips_rail_words():
    assert spatial.station_key("BISHAN INTERCHANGE MRT STATION") == "BISHAN"
    assert spatial.station_key("novena mrt") == "NOVENA"


def test_filter_measures_to_the_named_station_only():
    index = spatial.get_spatial_index()["payload"]["amenities"]
    novena = next(a for a in index if spatial.station_key(a["name"]) == "NOVENA")
    rows = _filter([], {"mrt": 1000}, mrt_station="Novena")
    any_station = _filter([], {"mrt": 1000})
    assert rows and len(rows) < len(any_station) / 10
    for b in rows:
        x, y = spatial.wgs84_to_svy21(b["lat"], b["lng"])
        assert math.hypot(x - novena["x"], y - novena["y"]) <= 1001


def test_extraction_keeps_the_station_and_a_disclosed_default_radius():
    c = sv.rule_based_extract("can i afford a resale near novena mrt").model_dump()
    assert c["mrt_station"] == "Novena"
    assert c["amenity_radii_meters"].get("mrt") == ri.VAGUE_RADIUS_METERS
    assert any("Novena MRT station" in a for a in c["assumptions"])
    assert not any("not a specific named station" in a for a in c["assumptions"])


def test_generic_mrt_has_no_station():
    c = sv.rule_based_extract("4 room resale near an mrt in bishan").model_dump()
    assert not c.get("mrt_station")


# --- area affordability --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("can i afford a resale near novena mrt", True),
    ("can we afford a 4 room in bishan", True),
    ("Can we afford a 4 ROOM at Blk 231 Tampines St 21?", False),   # one flat: the flat verdict instead
    ("4 room resale in bishan", False),
])
def test_area_affordability_detection(query, expected):
    c = sv.rule_based_extract(query).model_dump()
    assert bool(c.get("wants_area_affordability")) is expected


@pytest.fixture(scope="module")
def graph():
    from agents.arch_3_specialist_5agent import build_graph
    return build_graph()


def _run(graph, *turns):
    prev = None
    for q in turns:
        prev = graph.invoke(sv.initial_state(query=q, variant="C", previous=prev, fast_path=True))
    return prev


def test_first_turn_asks_household_facts_and_shows_prices(graph):
    out = _run(graph, "can we afford a 4 room in bishan")
    text = out["draft_response"]
    assert "Can you afford a 4 room flat in Bishan?" in text
    assert "Recent median prices in Bishan" in text
    clar = out.get("clarification") or {}
    assert clar.get("kind") == "financial"
    assert "remaining_lease_years" not in clar.get("needed", [])   # assumed from the buyers' ages, never asked


def test_household_reply_filters_blocks_by_the_ceiling(graph):
    out = _run(graph, "can we afford a 4 room in bishan", HOUSEHOLD)
    cap = sv._affordable_price(out)
    assert cap and cap > 0
    blocks = out["candidate_blocks_intersection"]
    assert blocks and all(b["median_resale_price"] <= cap for b in blocks)
    text = out["draft_response"]
    assert f"${cap:,.0f}" in text and f"{len(blocks)} block(s)" in text
    assert "at least 65 years of lease" in text   # youngest buyer 30 -> lease to 95
    assert not out.get("clarification")


def test_novena_reply_uses_the_station_not_any_mrt(graph):
    out = _run(graph, "can i afford a resale within 1km of novena mrt", HOUSEHOLD)
    calls = [l for l in out["tool_call_log"] if l.get("tool") == "filter_blocks_by_amenity_proximity"]
    assert calls and all((l.get("arguments") or {}).get("mrt_station") == "Novena" for l in calls)
    assert "near Novena MRT" in out["draft_response"]


# --- "how much can we afford?" with no flat chosen -----------------------------------------------------------------
BUDGET = ("How much can we afford? We are both Singapore citizens. We are first-time buyers. We are aged 32 and 30. "
          "We have no children. Our household income is $9,000 a month. Our CPF OA balance is $80,000. We have $50,000 "
          "in cash savings. We have no monthly debt. We have both worked continuously for the past 12 months. We are "
          "buying within 4 km of our parents. Our grant eligibility is confirmed. Our proximity grant eligibility is "
          "confirmed. Our HDB loan eligibility is confirmed.")


@pytest.mark.parametrize("text,expected", [
    ("Our household income is $9,000 a month.", False),
    ("Our CPF OA balance is $80,000.", False),
    ("My pay is $5,000 per month", False),
    ("Blk 123 Bedok North St 2 is $650k, fair?", True),
    ("is $600,000 reasonable for blk 1", True),
])
def test_income_amounts_are_not_a_price_check(text, expected):
    assert ri._has("assessment", text) is expected


def test_budget_question_answers_without_asking_lease_or_flat_type(graph):
    st = sv.initial_state(query=BUDGET, variant="C", fast_path=True,
                          map_state={"center": [1.3199, 103.84659], "radius_meters": 500})
    out = graph.invoke(st)
    assert not out.get("clarification")
    text = out["draft_response"]
    assert "Max purchase price" in text and "4-room flat" in text and "at least 65 years of lease" in text
    assert "map pin" not in text   # a budget question is not about the pinned place


def test_stated_flat_type_and_lease_win(graph):
    first = graph.invoke(sv.initial_state(query=BUDGET, variant="C", fast_path=True))
    out = graph.invoke(sv.initial_state(query="5 room, assume 80 year lease", variant="C", previous=first, fast_path=True))
    g = out["financial_breakdown"]["calculate_cpf_housing_grants"]
    assert g["family_grant"] == 50000   # 5-room Family Grant, not the 4-room default
    assert out["user_constraints"]["financial_inputs"]["grants"]["remaining_lease_years"] == 80
