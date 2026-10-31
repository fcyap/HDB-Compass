"""School-distance prompts must not fall through to unrelated policy fixtures."""

import pytest

from agents import supervisor as sv
from agents import arch_3_specialist_5agent as C


@pytest.mark.parametrize("query", [
    "School Priority: Check 1km and 1–2km MOE Primary 1 registration boundaries.",
    "Check primary school distance within 1km.",
    "Show MOE Primary 1 registration boundaries.",
])
def test_missing_school_asks_before_model_or_tool_calls(monkeypatch, query):
    monkeypatch.setattr(sv, "_INPUT_GUARDRAIL", None)
    monkeypatch.setattr(sv, "get_llm", lambda: pytest.fail("Clarification must precede inference"))
    result = C.run(query)
    assert "Which primary school" in result["draft_response"]
    assert result["tool_call_log"] == []
    assert not result.get("rag_policy_passages")
    assert "HFE" not in result["draft_response"]


@pytest.mark.parametrize("query, school, bracket", [
    ("Ai Tong School 1km", "Ai Tong School", "1km"),
    ("Catholic High School 1km", "Catholic High School", "1km"),
    ("Show HDB blocks within 1km of Ai Tong School.", "Ai Tong School", "1km"),
    ("show hdb blocks within 1–2km of catholic high school.", "catholic high school", "1-2km"),
    ("Catholic High School 2km", "Catholic High School", "2km"),
])
def test_named_school_routes_to_distance_tool(query, school, bracket):
    c = sv.rule_based_extract(query).model_dump()
    assert c["school_name"] == school
    assert sv.decide_routing(c) == ["spatial"]
    plan = sv.deterministic_plan("spatial", c, query)
    assert plan == [("verify_primary_school_distance", {"school_name": school, "bracket": bracket})]


def test_school_clarification_does_not_override_a_safety_block(monkeypatch):
    monkeypatch.setattr(sv, "_INPUT_GUARDRAIL", lambda state: {"draft_response": "Blocked"})
    assert sv.input_guardrail_node({"messages": [{"role": "user", "content": "School priority within 1km"}]})["draft_response"] == "Blocked"


def test_named_school_graph_uses_real_local_spatial_index(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setattr(sv, "_INPUT_GUARDRAIL", None)
    result = C.run("Ai Tong School 1km")
    assert result["active_subagents"] == ["spatial"]
    assert any(call["tool"] == "verify_primary_school_distance" and call["ok"] for call in result["tool_call_log"])
    assert result["spatial_block_candidates"]
    assert not result.get("rag_policy_passages")
