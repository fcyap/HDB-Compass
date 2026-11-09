"""
Independent check of the expected answers, and the fixes it led to (tests/test_independent_truth_check.py). Owned by
Member 4.

  * evaluation/check_truth_independent.py rebuilds Category E and B expected blocks from the raw sales and raw
    coordinates, with its own medians, distances and join (not the tools or the Supervisor's join under test).
  * It found B2-04's stored answer built from a price search cut off at 100 blocks: the scorer applies the recorded
    correction, and the Supervisor now prices the distance filter's blocks directly when the town-wide search is cut off.
  * Category C answers also get a score on the clauses that are still current (known-wrong clauses left out).

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_independent_truth_check.py -q
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from evaluation import architecture_metrics as M  # noqa: E402
from evaluation import check_truth_independent as T  # noqa: E402

HAS_DATA = (ROOT / "data" / "spatial_index.pkl").is_file() and any((ROOT / "data").rglob("*.duckdb"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="needs the local market database and spatial index")


def _items(name):
    return {i["id"]: i for i in json.loads((ROOT / "evaluation" / name).read_text())}


# --- the checker's own arithmetic -----------------------------------------------------------------------------------
def test_price_groups_take_medians_and_count_the_lease_down_to_the_reference_month():
    ref = date(2026, 9, 1)
    rows = [("12", "Bedok  North Rd", date(2026, 9, 1), 500_000, 720),     # 60 years left at the reference month
            ("12", "BEDOK NORTH RD", date(2026, 3, 1), 520_000, 726),     # 726 - 6 months = 720
            ("12", "BEDOK NORTH RD", date(2025, 10, 1), 540_000, 600)]    # 600 - 11 = 589
    g = T.price_groups(rows, ref)
    assert list(g) == ["12 BEDOK NORTH RD"]
    assert g["12 BEDOK NORTH RD"]["price"] == 520_000
    assert g["12 BEDOK NORTH RD"]["lease"] == 60.0
    assert T.within_price_and_lease(g, 520_000, 60) == {"12 BEDOK NORTH RD"}
    assert T.within_price_and_lease(g, 519_999, 60) == set()


def test_distances_count_lrt_stops_as_mrt_and_need_every_amenity():
    blocks = [{"block": "1", "street": "A ST", "town": "T", "x": 0, "y": 0},
              {"block": "2", "street": "A ST", "town": "T", "x": 0, "y": 1000},
              {"block": "3", "street": "A ST", "town": "OTHER", "x": 0, "y": 0}]
    amenities = [{"name": "X LRT STATION", "category": "mrt", "x": 300, "y": 0},
                 {"name": "H", "category": "hawker", "x": 0, "y": 450}]
    assert T.blocks_near(blocks, ["T"], {"mrt": 400}, amenities) == {"1 A ST"}
    assert T.blocks_near(blocks, ["T"], {"mrt": 400, "hawker": 400}, amenities) == set()
    assert T.blocks_near(blocks, ["T"], {"mrt_only": 400}, amenities) == set()


def test_check_item_joins_on_block_keys_and_applies_the_affordability_cap():
    item = {"id": "X1", "stated_constraints": {"towns": ["T"], "flat_types": ["4 ROOM"], "max_budget_sgd": 700_000,
                                               "min_remaining_lease_years": 60, "amenity_radii_meters": {"mrt": 400}},
            "ground_truth_facts": {"matching_block_keys": ["1 A ST"], "affordability_cap": 600_000}}
    spatial = {"blocks": [{"block": "1", "street": "A ST", "town": "T", "x": 0, "y": 0},
                          {"block": "2", "street": "A ST", "town": "T", "x": 10, "y": 0}],
               "amenities": [{"name": "M", "category": "mrt", "x": 0, "y": 0}], "schools": []}
    groups = {"1 A ST": {"price": 590_000, "lease": 70}, "2 A ST": {"price": 650_000, "lease": 70}}
    assert T.check_item(item, lambda towns, fts: groups, spatial)["status"] == "ok"
    item["ground_truth_facts"]["affordability_cap"] = None             # without the cap block 2 also qualifies
    r = T.check_item(item, lambda towns, fts: groups, spatial)
    assert r["status"] == "mismatch" and r["missing_from_stored"] == ["2 A ST"]


# --- the recorded correction and the scorer ---------------------------------------------------------------------------
def test_b2_04_is_scored_against_the_corrected_blocks():
    item = _items("category_b_queries.json")["B2-04"]
    assert "122 SIMEI ST 1" not in item["ground_truth_facts"]["matching_block_keys"]   # fcyap's file, unchanged
    assert "122 SIMEI ST 1" in M.expected_block_keys(item)
    old = "Matches: Blk 235 Tampines St 21, Blk 458 Tampines St 42, Blk 459 Tampines St 42."
    new = old.replace("Matches:", "Matches: Blk 122 Simei St 1,")
    assert M.ground_truth_scores(item, {"draft_response": old})["success"] is False
    s = M.ground_truth_scores(item, {"draft_response": new})
    assert s["truth_corrected"] is True and s["blocks_exact"] is True


def test_other_questions_are_not_corrected():
    item = _items("category_e_queries.json")["E1-03"]
    assert M.expected_block_keys(item) == item["ground_truth_facts"]["matching_block_keys"]


def test_policy_answers_also_get_a_score_on_current_clauses(monkeypatch):
    # the mechanism, on the legacy truth it was built for: the live truth (official pages) has no known-wrong clauses
    item = _items("category_c_queries.json")["C1-03"]
    item = {**item, "ground_truth_facts": item["legacy_ground_truth"]}
    monkeypatch.setattr(M, "KNOWN_WRONG_CLAUSES", {"C1-03": {"3% to 6% subsidy recovery": "Plus subsidy recovery given "
                                                             "as 3% to 6%; the current HDB range is 6% to 8%."}})
    wrong = {M._clause_key(w["clause"]) for w in M.known_wrong_clauses(item)}
    assert wrong
    clauses = item["ground_truth_facts"]["key_clauses"]
    current = [c for c in clauses if M._clause_key(c) not in wrong]
    answer = " ".join(" ".join(c) if isinstance(c, list) else c for c in current)
    s = M.policy_clause_scores(item, {"draft_response": answer})
    assert s["success"] is False and s["success_current"] is True and s["clause_coverage_current"] == 1.0
    plain = M.policy_clause_scores(_items("category_c_queries.json")["C2-01"], {"draft_response": ""})
    assert plain["success_current"] == plain["success"]


# --- the Supervisor no longer loses blocks to the price search's limit ------------------------------------------------
def test_truncation_flag():
    assert sv._price_query_truncated({"blocks": [], "metadata": {"truncated": True}})
    assert not sv._price_query_truncated({"blocks": [], "metadata": {"truncated": False}})
    assert not sv._price_query_truncated(None)


@needs_data
def test_b2_04_lists_the_block_beyond_the_price_search_limit():
    item = _items("category_b_queries.json")["B2-04"]
    st = C.graph.invoke(sv.initial_state(query=item["query"], variant="C", map_state={}, fast_path=True))
    got = {sv.block_key(b["block"], b["street"]) for b in st["candidate_blocks_intersection"]}
    assert got == {k.upper() for k in M.expected_block_keys(item)}
    assert "Blk 122 Simei St 1".upper() in st["draft_response"].upper()


@needs_data
def test_every_stored_e_and_b_answer_matches_the_raw_data():
    assert T.main([]) == 0
