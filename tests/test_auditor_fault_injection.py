"""evaluation/auditor_fault_injection.py: the corruption helpers and the measurement on a synthetic answer (no model)."""
import json

import pytest

from agents import guardrails as g
from evaluation import auditor_fault_injection as fi

DRAFT = ("- Blk 336 CLEMENTI AVE 4 — median $748,000, 72 yrs lease — mrt 420m\n"
         "- Blk 340 CLEMENTI AVE 4 — median $802,000, 69 yrs lease — mrt 515m\n")
STATE = {"draft_response": DRAFT, "candidate_blocks_intersection": [
    {"block": "336", "street": "CLEMENTI AVE 4", "median_price": 748000.0, "median_remaining_lease_years": 72.4, "nearest_mrt_m": 420.0},
    {"block": "340", "street": "CLEMENTI AVE 4", "median_price": 802000.0, "median_remaining_lease_years": 69.1, "nearest_mrt_m": 515.0}]}
ITEMS = [{"id": "T-01", "category": "B", "query": "4-room in Clementi"}]


def test_a_clean_answer_passes_its_own_evidence():
    assert g.audit_text(DRAFT, g.evidence_text(STATE), "4-room in Clementi")["passed"]


@pytest.mark.parametrize("severity", ["gross", "subtle"])
@pytest.mark.parametrize("kind", fi.KINDS)
def test_every_corruption_changes_one_figure_to_a_value_the_evidence_lacks(kind, severity):
    evidence = g.evidence_text(STATE)
    pool = fi._pool(evidence)
    span = fi._spans(DRAFT)[kind][0]
    bad = fi.corrupt(DRAFT, kind, span, pool, severity)
    assert bad is not None and bad != DRAFT
    assert not g.audit_text(bad, evidence)["passed"]


def test_corruption_keeps_the_whitespace_after_the_figure():
    text = "Income ceiling $7,000 to $14,000."
    span = fi._spans(text)["price"][0]
    bad = fi.corrupt(text, "price", span, [])
    assert " to " in bad


def test_measurement_is_seeded_and_reports_zero_false_rejections():
    states = {"T-01": STATE}
    a = fi.measure_figures(ITEMS, states, "gross")
    b = fi.measure_figures(ITEMS, states, "gross")
    assert a == b
    assert a["faults"] == 4 and a["caught"] == 4 and a["clean_rejected"] == 0


def test_misattributed_figures_are_not_detectable_by_value():
    out = fi.measure_misattribution(ITEMS, {"T-01": STATE})
    assert out["faults"] >= 1 and out["caught"] == 0       # documented limitation, see the module docstring


def test_the_benchmark_has_90_questions_and_the_measurement_uses_75():
    assert len(json.loads((fi.EVAL / "benchmark_hdb_90.json").read_text())) == 90
    assert len(fi.load_questions()) == 75
