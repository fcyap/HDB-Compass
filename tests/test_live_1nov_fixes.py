"""
Fixes for the 1 Nov live run (tests/test_live_1nov_fixes.py). Owned by Member 4.

  * A model's MRT station is kept only when the user's words name that station: the live model filled in the town's
    station ("ANG MO KIO" for "within 400m of an MRT station" in Ang Mo Kio), "not specified" or a school, and every
    block was then measured to that one station (E1-03, E1-07, E1-X02, B1-01, B1-02, B1-03, B1-05 and 8 varied rows).
  * A model's commute call that names no block the spatial filters kept ("QUEENSTOWN", "BLK 123") is given the
    filtered blocks, so the commute check times real blocks instead of nothing (B3-01..B3-05).

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_live_1nov_fixes.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

os.environ["LLM_PROFILE"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import supervisor as sv  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "evaluation"
CAT_E = {q["id"]: q["query"] for q in json.load(open(EVAL / "category_e_queries.json"))}
CAT_B = {q["id"]: q["query"] for q in json.load(open(EVAL / "category_b_queries.json"))}


def model_path(query, **reading):
    """A model reading put through the same finalize step as extract_constraints_node."""
    st = {"messages": [{"role": "user", "content": query}], "variant": "C"}
    c = {**sv.rule_based_extract(query).model_dump(), **reading}
    errors = []
    return sv._finalize_constraints(c, query, st, errors, from_model=True), errors


# --- 1. the model's MRT station ----------------------------------------------------------------------------------
@pytest.mark.parametrize("query,station", [
    (CAT_E["E1-03"], "ANG MO KIO"),
    (CAT_E["E1-07"], "TOA PAYOH"),
    (CAT_B["B1-01"], "not specified"),
    (CAT_B["B1-03"], "ROSYTH SCHOOL"),
    ("4rm in amk within 400m of mrt under 600k", "ang mo kio"),
])
def test_a_station_the_user_did_not_name_is_dropped(query, station):
    c, errors = model_path(query, mrt_station=station)
    assert not c.get("mrt_station")
    plan = dict(sv.deterministic_plan("spatial", c, query))
    assert "mrt_station" not in plan.get("filter_blocks_by_amenity_proximity", {})


@pytest.mark.parametrize("query,station,kept", [
    ("4-room in Toa Payoh within 400m of Novena MRT under $700k", "Novena MRT Station", "Novena"),
    ("flats near Ang Mo Kio MRT under 600k", "ANG MO KIO", "Ang Mo Kio"),
    ("4-room within 300m of Bishan station", "bishan", "Bishan"),
])
def test_a_station_the_user_named_is_kept(query, station, kept):
    c, errors = model_path(query, mrt_station=station)
    assert c["mrt_station"] == kept
    assert not [e for e in errors if "MRT station" in e]


def test_a_missing_model_station_takes_the_one_the_user_named():
    c, _ = model_path("4-room in Toa Payoh within 400m of Novena MRT under $700k", mrt_station="")
    assert c["mrt_station"] == "Novena"


# --- 2. the model's commute call ---------------------------------------------------------------------------------
class CallsSpatialTools:
    """One turn: the amenity filter, then the commute check with the given candidate blocks."""

    def __init__(self, c, candidate_blocks):
        self.c, self.candidate_blocks, self.called = c, candidate_blocks, False

    def bind_tools(self, tools, tool_choice=None):
        return self

    def invoke(self, msgs):
        if self.called:
            return AIMessage(content="done")
        self.called = True
        work = self.c["work_destinations"]
        return AIMessage(content="", tool_calls=[
            {"name": "filter_blocks_by_amenity_proximity", "id": "c1",
             "args": {"towns": self.c["towns"], "amenity_constraints": self.c["amenity_radii_meters"]}},
            {"name": "compute_dual_transit_commute", "id": "c2",
             "args": {"candidate_blocks": self.candidate_blocks, "work_dest_a": work[0], "work_dest_b": work[1],
                      "max_minutes": self.c["max_commute_minutes"]}},
        ])


def run_spatial(query, candidate_blocks):
    c = sv.rule_based_extract(query).model_dump()
    state = {"messages": [{"role": "user", "content": query}], "variant": "C", "user_constraints": c}
    with patch.object(sv, "get_llm", return_value=CallsSpatialTools(c, candidate_blocks)):
        return sv.make_specialist_node("spatial")(state)


@pytest.mark.parametrize("given", [["QUEENSTOWN"], ["BLK 123", "BLK 456"], []])
def test_a_commute_call_without_filtered_blocks_times_the_filtered_blocks(given):
    out = run_spatial(CAT_B["B3-01"], given)
    commute = [r for r in out["tool_call_log"] if r["tool"] == "compute_dual_transit_commute" and r["source"] == "model"]
    assert commute and commute[0]["arguments"]["candidate_blocks"]
    assert all(b.startswith("Blk ") for b in commute[0]["arguments"]["candidate_blocks"])
    assert out["spatial_block_candidates"], "B3-01 has blocks within the commute limit"
    if given:
        assert [e for e in out["errors"] if "commute blocks that no filter kept were replaced" in e]


def test_a_commute_call_with_filtered_blocks_is_left_alone():
    query = CAT_B["B3-01"]
    c = sv.rule_based_extract(query).model_dump()
    kept = sv._commute_candidates({"filter_blocks_by_amenity_proximity": sv.ToolRunner("filter_blocks_by_amenity_proximity")(
        towns=c["towns"], amenity_constraints=c["amenity_radii_meters"])})
    out = run_spatial(query, kept[:3])
    commute = [r for r in out["tool_call_log"] if r["tool"] == "compute_dual_transit_commute" and r["source"] == "model"]
    assert commute[0]["arguments"]["candidate_blocks"] == kept[:3]
    assert not [e for e in out["errors"] if "commute blocks" in e]
