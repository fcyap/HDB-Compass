"""
Tool selection and robustness tests (tests/test_tool_gating.py). Owned by Member 4.

  * Upfront costs and comparable sales run only when the user asks for them, on every path that can call them
    (the no-model plans, the real financial sequence, the evidence gate and Variant C's tool lists), and a user who
    does ask still gets them.
  * A tool module that fails to import makes only its own tools unavailable, with one loud warning.
  * Member 5's hook module: absent is quiet, present but broken is loud and recorded in the answer and run settings.
  * HDB_CODE_WRITTEN_FACTS overrides architecture.code_written_facts for one run, and the run settings record it.

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_tool_gating.py -q
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv  # noqa: E402
from agents import tool_registry as tr  # noqa: E402
from agents import arch_1_monolith as A  # noqa: E402
from agents import arch_2_coarse as B  # noqa: E402
from agents import arch_3_specialist_5agent as C  # noqa: E402

UPFRONT = "calculate_bsd_and_upfront_outlay"
COMPARABLES = "query_comparable_resale_transactions"
ITEMS = {i["id"]: i for i in json.loads((PROJECT_ROOT / "evaluation" / "category_e_queries.json").read_text())}
E1 = ITEMS["E1-01"]["query"]   # grants and maximum budget: no upfront costs asked for
E1_ASKS = E1.replace("tell us our grants and maximum budget.",
                     "tell us our grants and maximum budget, and how much stamp duty and cash we need upfront.")
MOCK_E2 = "Find a 4-room flat in Tampines under $680k within 400m of an MRT with 65+ years lease."
COORDS = {"latitude": 1.35, "longitude": 103.85, "flat_type": "4 ROOM"}


def tools_called(state):
    return [r["tool"] for r in state["tool_call_log"]]


# --- 1. what counts as an ask -----------------------------------------------------------------------------------
@pytest.mark.parametrize("text,asked", [
    ("How much cash do we need upfront?", [UPFRONT]),
    ("What is the stamp duty on a $650k flat?", [UPFRONT]),
    ("and the BSD please", [UPFRONT]),
    ("what's the downpayment?", [UPFRONT]),
    ("how much cash will we need to pay?", [UPFRONT]),
    ("We are buying a 4-room in Bishan for $650,000.", [UPFRONT]),
    ("What did similar 4-room flats sell for?", [COMPARABLES]),
    ("Show me recent transactions near Blk 123.", [COMPARABLES]),
    ("Any comparables around these coordinates?", [COMPARABLES]),
    ("Compare sales around the supplied coordinates.", [COMPARABLES]),
    # not an ask: the words the Category E questions use for facts and searches
    ("We have $30,000 in cash savings and $120,000 in CPF OA.", []),
    ("Tell us our grants and maximum budget.", []),
    ("We are comparing a Prime BTO vs. a Plus resale flat.", []),
    ("How does the EIP quota affect a resale transaction?", []),
    ("Show the price trend over the last 3 years.", []),
])
def test_requested_tools_reads_only_explicit_asks(text, asked):
    assert sv.requested_tools(text) == asked


def test_no_category_e_question_asks_for_the_optional_tools():
    varied = json.loads((PROJECT_ROOT / "evaluation" / "category_e_varied.json").read_text())
    for item in list(ITEMS.values()) + varied:
        assert sv.requested_tools(item["query"]) == [], item["id"]
        assert not {c["tool"] for c in item.get("expected_mcp_calls", [])} & set(sv.OPTIONAL_TOOLS)


# --- 1. the no-model plans ---------------------------------------------------------------------------------------
def test_financial_plan_costs_the_budget_only_when_asked():
    c = {"household_income": 9000.0, "max_budget_sgd": 680000.0}
    assert UPFRONT not in [t for t, _ in sv.deterministic_plan("financial", c, "We earn $9,000 a month.")]
    plan = dict(sv.deterministic_plan("financial", c, "What stamp duty would we pay?"))
    assert plan[UPFRONT]["purchase_price"] == 680000.0
    # what the extraction step recorded from every user message wins over the latest message alone
    assert UPFRONT not in dict(sv.deterministic_plan("financial", {**c, "requested_tools": []}, "stamp duty?"))
    assert UPFRONT in dict(sv.deterministic_plan("financial", {**c, "requested_tools": [UPFRONT]}, "ok"))


def test_a_stated_purchase_price_is_an_ask_and_is_the_price_costed():
    c = {"household_income": 9000.0, "max_budget_sgd": 700000.0, "requested_tools": [],
         "financial_inputs": {"upfront": {"purchase_price": 600000.0}}}
    assert dict(sv.deterministic_plan("financial", c, ""))[UPFRONT]["purchase_price"] == 600000.0


def test_comparables_need_coordinates_and_an_ask():
    c = {"towns": ["TAMPINES"], "comparable_resale": COORDS}
    assert COMPARABLES not in dict(sv.deterministic_plan("market_sql", c, "Find 4-room flats in Tampines."))
    assert COMPARABLES in dict(sv.deterministic_plan("market_sql", c, "What did similar flats sell for recently?"))
    assert COMPARABLES not in dict(sv.deterministic_plan("market_sql", {"towns": ["TAMPINES"]}, "recent sales?"))


# --- 1. the real financial sequence and the evidence gate -------------------------------------------------------
@pytest.fixture
def real_financial_tools(monkeypatch):
    for name in tr.tools_for_agent("financial"):
        monkeypatch.setitem(sv.load_config()["tools"]["use_mock"], name, "real")
    return tr.build_runners(tr.tools_for_agent("financial"))


def test_real_financial_sequence_never_costs_the_budget_unasked(real_financial_tools):
    c = {"max_budget_sgd": 650000.0, "financial_inputs": {"affordability": {"monthly_income": 8500.0}}}
    assert UPFRONT not in sv.run_financial_sequence(real_financial_tools, c, "Tell us our maximum budget.")
    out = sv.run_financial_sequence(real_financial_tools, c, "And the stamp duty on our budget?")
    assert real_financial_tools[UPFRONT].log[-1].arguments["purchase_price"] == 650000.0 and UPFRONT in out


def test_evidence_gate_drops_unasked_upfront_costs():
    outputs = {"calculate_cpf_housing_grants": {"status": "ok"}, UPFRONT: {"status": "ok"}}
    assert list(sv.requested_outputs("financial", outputs, {"requested_tools": []}, "")) == ["calculate_cpf_housing_grants"]
    assert list(sv.requested_outputs("financial", outputs, {"requested_tools": [UPFRONT]}, "")) == list(outputs)
    market = {"query_resale_blocks_by_criteria": {}, COMPARABLES: {}}
    asked = {"towns": ["TAMPINES"], "comparable_resale": COORDS, "requested_tools": [COMPARABLES]}
    assert list(sv.requested_outputs("market_sql", market, asked, "")) == list(market)


# --- 1. Variant C's specialists: the model is not offered a tool nobody asked for --------------------------------
class ToolListModel:
    """Records the tools it is offered and makes no call, so the specialist falls back to its plan."""

    def __init__(self):
        self.offered = []

    def bind_tools(self, tools):
        self.offered = [t.name for t in tools]
        return self

    def invoke(self, msgs):
        return AIMessage(content="done")


@pytest.mark.usefixtures("mock_domain_tools")
@pytest.mark.parametrize("agent,optional,ask", [("financial", UPFRONT, "how much stamp duty do we pay?"),
                                                ("market_sql", COMPARABLES, "what did similar flats sell for?")])
def test_variant_c_specialist_offers_optional_tools_only_when_asked(agent, optional, ask):
    c = {"towns": ["TAMPINES"], "household_income": 9000.0, "max_budget_sgd": 680000.0, "comparable_resale": COORDS}
    model = ToolListModel()
    node = sv.make_specialist_node(agent)
    with patch.object(sv, "get_llm", return_value=model):
        quiet = node({"messages": [{"role": "user", "content": "Find flats."}], "user_constraints": {**c, "requested_tools": []}})
        assert optional not in model.offered and model.offered
        assert optional not in [r["tool"] for r in quiet["tool_call_log"]]
        asked = node({"messages": [{"role": "user", "content": ask}],
                      "user_constraints": {**c, "requested_tools": sv.requested_tools(ask)}})
        assert optional in model.offered
    assert optional in [r["tool"] for r in asked["tool_call_log"]]


# --- 1. end to end on the real tools: no unasked section, and the asker still gets it ---------------------------
@pytest.mark.parametrize("mod", [A, B, C])
def test_e1_question_makes_no_upfront_call_unless_asked(mod):
    plain = mod.run(E1)
    assert UPFRONT not in tools_called(plain)
    assert "Upfront" not in plain["draft_response"]
    asked = mod.run(E1_ASKS)
    assert UPFRONT in tools_called(asked)
    assert {"calculate_cpf_housing_grants", "calculate_maximum_affordability"} <= set(tools_called(asked))


def test_variant_c_fully_right_answer_keeps_its_upfront_section_when_asked():
    state = C.run(E1_ASKS)
    assert state["user_constraints"]["requested_tools"] == [UPFRONT]
    assert "Upfront funding" in state["draft_response"] and "Grants:" in state["draft_response"]


def test_an_ask_in_the_first_turn_survives_the_follow_up_reply():
    state = {"messages": [{"role": "user", "content": "What stamp duty would a $600k 4-room in Tampines cost us?"},
                          {"role": "assistant", "content": "To finish, please tell me your cash savings."},
                          {"role": "user", "content": "We have $40,000 in cash."}]}
    assert sv.extract_constraints_node(state)["user_constraints"]["requested_tools"] == [UPFRONT]


# --- 2. a tool module that fails to import ----------------------------------------------------------------------
@pytest.fixture
def broken_market_module(tmp_path, monkeypatch):
    (tmp_path / "broken_market_tools_fake.py").write_text('raise ImportError("simulated: duckdb is not installed")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(tr.REAL_IMPL, "query_resale_blocks_by_criteria",
                        ("broken_market_tools_fake", "query_resale_blocks_by_criteria"))
    monkeypatch.setitem(sv.load_config()["tools"]["use_mock"], "query_resale_blocks_by_criteria", "real")
    monkeypatch.setattr(tr, "_WARNED", set())


@pytest.mark.usefixtures("mock_domain_tools", "broken_market_module")
def test_broken_tool_import_gives_an_unavailable_runner_and_one_warning(capsys):
    runners = tr.build_runners(["query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity"])
    runners = tr.build_runners(["query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity"])
    warning = capsys.readouterr().err
    assert warning.count("WARNING: broken_market_tools_fake failed to import") == 1
    assert "simulated: duckdb is not installed" in warning
    out = runners["query_resale_blocks_by_criteria"](towns=["TAMPINES"])
    assert out["status"] == "unavailable" and "simulated" in out["error"]
    rec = runners["query_resale_blocks_by_criteria"].log[-1]
    assert not rec.ok and rec.error.startswith("unavailable:")      # still counted as a selected tool
    assert runners["query_resale_blocks_by_criteria"].as_langchain_tool().name == "query_resale_blocks_by_criteria"


@pytest.mark.usefixtures("mock_domain_tools", "broken_market_module")
def test_variant_a_still_answers_when_one_tool_module_is_broken():
    state = A.run(MOCK_E2)
    log = {r["tool"]: r for r in state["tool_call_log"]}
    assert not log["query_resale_blocks_by_criteria"]["ok"]
    assert log["filter_blocks_by_amenity_proximity"]["ok"]           # the other tools still ran
    assert state["draft_response"] and "sql_market_results" not in state   # the error never became evidence


# --- 3. Member 5's hook module ----------------------------------------------------------------------------------
def hook_module(tmp_path, monkeypatch, name, source):
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_absent_hook_module_is_quiet(capsys):
    assert sv._load_member5_hooks("agents.no_such_guardrails_module") == (None, None, [])
    assert capsys.readouterr().err == ""


def test_hook_module_that_fails_to_import_is_loud_and_recorded(tmp_path, monkeypatch, capsys):
    name = hook_module(tmp_path, monkeypatch, "guardrails_fake_broken", "import a_package_nobody_installed_xyz\n")
    guardrail, auditor, problems = sv._load_member5_hooks(name)
    assert guardrail is None and auditor is None
    assert len(problems) == 1 and "ModuleNotFoundError" in problems[0] and "a_package_nobody_installed_xyz" in problems[0]
    assert "WARNING: Member 5 hooks" in capsys.readouterr().err


def test_hook_module_missing_a_node_installs_the_other_and_is_loud(tmp_path, monkeypatch, capsys):
    name = hook_module(tmp_path, monkeypatch, "guardrails_fake_partial",
                       "def input_guardrail_node(state):\n    return {}\nruntime_auditor_node = None\n")
    guardrail, auditor, problems = sv._load_member5_hooks(name)
    assert callable(guardrail) and auditor is None
    assert problems == [f"{name} has no callable runtime_auditor_node"]
    assert "runtime_auditor_node" in capsys.readouterr().err
    name = hook_module(tmp_path, monkeypatch, "guardrails_fake_complete",
                       "def input_guardrail_node(state):\n    return {}\ndef runtime_auditor_node(state):\n    return {}\n")
    assert sv._load_member5_hooks(name)[2] == [] and capsys.readouterr().err == ""


@pytest.mark.usefixtures("mock_domain_tools")
def test_hook_load_problem_shows_in_the_answer_and_the_run_settings(monkeypatch):
    from evaluation.run_category_e_prelim import lost_model, model_fallbacks, run_settings
    assert "hook_load_problems" not in run_settings({"name": "none", "model": None})   # nothing to record
    monkeypatch.setattr(sv, "HOOK_LOAD_PROBLEMS", ["agents.guardrails failed to import (SyntaxError: x)"])
    monkeypatch.setattr(sv, "_INPUT_GUARDRAIL", None)
    monkeypatch.setattr(sv, "_AUDITOR", None)
    state = C.run(MOCK_E2)
    notes = [e for e in state["errors"] if "Member 5 hook could not be loaded" in e]
    assert [n.split(":")[0] for n in notes] == ["input_guardrail", "runtime_auditor"]
    assert state["auditor_verdict"]["critique"] == "auditor failed to load (pass-through)"
    assert not lost_model(state) and model_fallbacks(state) == []       # never mistaken for a model failure
    assert run_settings({"name": "none", "model": None})["hook_load_problems"] == sv.HOOK_LOAD_PROBLEMS


# --- 4. the code-written facts switch ---------------------------------------------------------------------------
def test_code_written_facts_env_override(monkeypatch):
    arch = sv.load_config()["architecture"]
    monkeypatch.delenv("HDB_CODE_WRITTEN_FACTS", raising=False)
    monkeypatch.setitem(arch, "code_written_facts", True)
    assert sv.code_written_facts_on() and sv.writes_facts_in_code({"variant": "C"})
    monkeypatch.setenv("HDB_CODE_WRITTEN_FACTS", "0")
    assert not sv.code_written_facts_on() and not sv.writes_facts_in_code({"variant": "C"})
    monkeypatch.setitem(arch, "code_written_facts", False)
    monkeypatch.setenv("HDB_CODE_WRITTEN_FACTS", "1")
    assert sv.writes_facts_in_code({"variant": "C"})
    assert not sv.writes_facts_in_code({"variant": "B"})                 # B keeps the model as the writer by design
    monkeypatch.setenv("HDB_CODE_WRITTEN_FACTS", "")
    assert not sv.code_written_facts_on()                                # empty: the config decides
    monkeypatch.setenv("HDB_CODE_WRITTEN_FACTS", "off")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        sv.code_written_facts_on()


def test_run_settings_record_the_code_written_facts_override(monkeypatch):
    from evaluation.run_category_e_prelim import run_settings
    monkeypatch.setitem(sv.load_config()["architecture"], "code_written_facts", True)
    monkeypatch.setenv("HDB_CODE_WRITTEN_FACTS", "0")
    assert run_settings({"name": "none", "model": None})["code_written_facts"] is False
    monkeypatch.delenv("HDB_CODE_WRITTEN_FACTS")
    assert run_settings({"name": "none", "model": None})["code_written_facts"] is True
