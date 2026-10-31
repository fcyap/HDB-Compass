"""
Unit and integration tests for Category B Spatial & Commute Benchmark (tests/test_category_b.py).
Owned by Member 2 (Categories B & E).
"""
import json
import pytest
from pathlib import Path

from evaluation.build_category_b import SPECS, ground_spec
from agents import arch_3_specialist_5agent as arch3
from evaluation import architecture_metrics as M

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATEGORY_B_PATH = PROJECT_ROOT / "evaluation" / "category_b_queries.json"


def test_category_b_file_exists_and_has_15_queries():
    assert CATEGORY_B_PATH.exists(), "category_b_queries.json does not exist"
    data = json.loads(CATEGORY_B_PATH.read_text(encoding="utf-8"))
    assert len(data) == 15, f"Expected 15 queries, found {len(data)}"
    
    sub_counts = {}
    for item in data:
        sub_counts[item["sub_use_case"]] = sub_counts.get(item["sub_use_case"], 0) + 1
    assert sub_counts.get("B1") == 5, f"Expected 5 B1 questions, got {sub_counts.get('B1')}"
    assert sub_counts.get("B2") == 5, f"Expected 5 B2 questions, got {sub_counts.get('B2')}"
    assert sub_counts.get("B3") == 5, f"Expected 5 B3 questions, got {sub_counts.get('B3')}"


def test_all_category_b_specs_ground_successfully():
    for spec in SPECS:
        item, problems = ground_spec(spec)
        assert not problems, f"Spec {spec['id']} failed with problems: {problems}"
        truth = item["ground_truth_facts"]
        assert len(truth["matching_block_keys"]) > 0, f"Spec {spec['id']} has no matching blocks"


def test_variant_c_passes_all_15_category_b_queries(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")
    data = json.loads(CATEGORY_B_PATH.read_text(encoding="utf-8"))
    
    for item in data:
        state = arch3.run(item["query"])
        scores = M.score_item(item, state)
        assert scores["task_success"] is True, f"Item {item['id']} task_success failed"
        assert scores["routing"]["exact"] is True, f"Item {item['id']} routing failed: {scores['routing']}"
        assert scores["tool_selection"]["f1"] == 1.0, f"Item {item['id']} tool F1 < 1.0: {scores['tool_selection']}"
        gt = scores.get("ground_truth") or {}
        assert gt.get("blocks_exact") is True, f"Item {item['id']} blocks_exact failed: {gt}"

