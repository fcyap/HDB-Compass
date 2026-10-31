"""
Contract tests (tests/test_contract.py). Owned by Member 4.
Runs WITHOUT an LLM and without network: proves the tool contract, the mock
server and the registry agree with agents/schemas.py.

    .venv/bin/pytest tests/test_contract.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import schemas as S  # noqa: E402
from agents.tool_registry import ToolRunner, build_runners, tools_for_agent  # noqa: E402
from mcp_server.mock_tools import MOCK_TOOLS  # noqa: E402

pytestmark = pytest.mark.usefixtures("mock_domain_tools")

EXAMPLE_INPUTS = {
    "query_resale_blocks_by_criteria": dict(towns=["BISHAN", "TAMPINES"], flat_types=["4 ROOM"], max_price=750000, min_lease_years=60),
    "compute_town_price_timeseries": dict(towns=["TAMPINES", "BEDOK"], flat_type="4 ROOM", lookback_years=3),
    "compare_bto_launch_vs_resale": dict(bto_project_name="Kallang Breeze", radius_meters=500),
    "calculate_cpf_housing_grants": dict(flat_category="resale", citizenship_pair="SC_PR", first_timer_status=True, monthly_income=7800, near_parents_4km=True),
    "calculate_maximum_affordability": dict(monthly_income=8500, cpf_oa_balance=120000, cash_savings=30000, loan_type="hdb"),
    "calculate_bsd_and_upfront_outlay": dict(purchase_price=680000, loan_type="hdb", total_grants=110000),
    "retrieve_hdb_policy_knowledge": dict(query="Prime vs Plus MOP and clawback", policy_domain=None),
}
OUTPUT_MODELS = {
    "query_resale_blocks_by_criteria": S.QueryResaleBlocksOutput,
    "compute_town_price_timeseries": S.ComputeTownPriceTimeseriesOutput,
    "compare_bto_launch_vs_resale": S.CompareBtoLaunchVsResaleOutput,
    "calculate_cpf_housing_grants": S.CalculateCpfHousingGrantsOutput,
    "calculate_maximum_affordability": S.CalculateMaximumAffordabilityOutput,
    "calculate_bsd_and_upfront_outlay": S.CalculateBsdAndUpfrontOutlayOutput,
    "retrieve_hdb_policy_knowledge": S.RetrieveHdbPolicyKnowledgeOutput,
}


def test_inventory_includes_comparable_analytics():
    assert len(S.TOOL_OWNER) == 23
    assert set(S.TOOL_OWNER) == set(S.TOOL_INPUT_MODEL)
    assert len(tools_for_agent("market_sql")) == 12
    assert len(tools_for_agent("spatial")) == 7
    assert len(tools_for_agent("financial")) == 3
    assert len(tools_for_agent("policy_rag")) == 1


def test_every_mocked_tool_is_in_the_inventory():
    assert set(MOCK_TOOLS) <= set(S.TOOL_OWNER)


@pytest.mark.parametrize("tool_name", sorted(EXAMPLE_INPUTS))
def test_mock_output_validates_against_schema(tool_name):
    runner = ToolRunner(tool_name)
    out = runner(**EXAMPLE_INPUTS[tool_name])
    assert "error" not in out, out
    OUTPUT_MODELS[tool_name](**out)  # raises if the shape drifts
    assert runner.log[-1].ok and runner.log[-1].tool == tool_name


def test_invalid_arguments_are_rejected_at_the_boundary_not_in_the_tool():
    runner = ToolRunner("calculate_maximum_affordability")
    out = runner(monthly_income="eight thousand", cpf_oa_balance=1, cash_savings=1, loan_type="hdb")
    assert "error" in out
    assert runner.log[-1].ok is False


def test_mock_is_deterministic():
    r = ToolRunner("query_resale_blocks_by_criteria")
    a = r(**EXAMPLE_INPUTS["query_resale_blocks_by_criteria"])
    b = r(**EXAMPLE_INPUTS["query_resale_blocks_by_criteria"])
    assert a == b


def test_block_key_is_shared_identity_between_sql_and_spatial_shapes():
    sql = S.ResaleBlockRecord(block="123", street="Bishan St 13", town="BISHAN", flat_type="4 ROOM",
                              median_resale_price=1, median_price_per_sqm=1, median_remaining_lease_years=1,
                              transaction_count=1, last_transaction_month="2026-09")
    spatial = S.SpatialBlockRecord(block="123", street="BISHAN ST 13", town="BISHAN")
    assert sql.key == spatial.key == "123 BISHAN ST 13"


def test_langchain_tool_wrapping_exposes_schema():
    lc = ToolRunner("retrieve_hdb_policy_knowledge").as_langchain_tool()
    assert lc.name == "retrieve_hdb_policy_knowledge"
    assert "query" in lc.args


def test_registry_builds_all_tools_including_real_spatial_ones():
    runners = build_runners()
    assert set(runners) == set(S.TOOL_OWNER)


def test_vllm_profile_sends_no_think_and_key_env(monkeypatch):
    from agents import llm as L
    monkeypatch.setenv("LLM_PROFILE", "vllm")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    p = L.active_profile()
    assert p["model"] == "Qwen/Qwen3-30B-A3B-GPTQ-Int4" and p["api_key"] == "test-key"
    model = L.get_llm()
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_strip_think_removes_reasoning_blocks():
    from agents.supervisor import strip_think
    assert strip_think("<think>\nplan\n</think>\n\nAnswer") == "Answer"
    assert strip_think("No tags") == "No tags"
