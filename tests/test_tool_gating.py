"""
Tool selection and robustness tests (tests/test_tool_gating.py). Owned by Member 4.

  * Upfront costs and comparable sales run only when the user asks for them, on every path that can call them
    (the no-model plans, the real financial sequence, the evidence gate and Variant C's tool lists), and a user who
    does ask still gets them, in whatever common words they ask. No model prompt names the asks.
  * A tool module that fails to import makes only its own tools unavailable, with one loud warning, and the answer
    says what could not be checked.
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


# asks the first version of the gate missed (the review of 31 Oct): each lost a section the user had before the gate
@pytest.mark.parametrize("text,asked", [
    ("How much did 4-room flats near latitude 1.3329, longitude 103.7436 sell for in the last 12 months?", [COMPARABLES]),
    ("What have 4-room flats within 500m of 1.3329, 103.7436 been selling for?", [COMPARABLES]),
    ("Show resale prices within 500m of latitude 1.3329, longitude 103.7436 in the last 12 months.", [COMPARABLES]),
    ("What prices did nearby 4-room flats fetch over the past year?", [COMPARABLES]),
    ("What are 4-room flats around 1.3329, 103.7436 going for these days?", [COMPARABLES]),
    ("How much did Blk 123 sell for?", [COMPARABLES]),
    ("Find a 4-room flat in Tampines under $600k. We earn $9,000 a month. "
     "How much would we need in cash and CPF to complete the purchase?", [UPFRONT]),
    ("How much do I need to pay in cash for a $600k flat?", [UPFRONT]),
    ("What's the total cash and CPF needed for the purchase?", [UPFRONT]),
    ("What are the initial costs of buying a $600k 4-room?", [UPFRONT]),
    ("We are buying a 4-room in Bishan for $650k, near MRT.", [UPFRONT]),
    ("We don't have a car, so what is the BSD on $600k?", [UPFRONT]),   # the refusal is about something else
    ("Don't forget the stamp duty.", [UPFRONT]),
])
def test_requested_tools_reads_other_phrasings_of_an_ask(text, asked):
    assert sv.requested_tools(text) == asked


@pytest.mark.parametrize("text", [
    "how much cash savings should I keep as buffer?",
    "How much cash can I use from savings?",
    "Looking to buy in Tampines for $600k or less, near MRT",            # a budget, not a purchase price
    "Looking to buy a flat for $600k max",
    "What are the ABSD rules for PR couples?",                           # a policy question
    "Does stamp duty apply to a resale flat?",
    "we don't care about stamp duty, just find flats in Tampines",
    "Forget the comparables, just find flats in Tampines.",
    "How much would I need to pay monthly from CPF?",                    # the loan, which affordability covers
    "How much do we need to earn to afford a $600k flat with CPF?",
    "How have prices in Tampines changed recently? Sales have been slow lately",
    "Tampines flats near MRT, with sales near the MRT",
    "Which flats should we go for?",
    "Find flats that go for under $600k in Bishan",
    "Find 4-room flats with prices around 600k in Bishan",
    "Find 4-room flats with sales near 1 million in Bishan",
])
def test_requested_tools_ignores_words_that_are_not_an_ask(text):
    assert sv.requested_tools(text) == []


def test_a_budget_is_not_read_as_a_purchase_price():
    assert "upfront" not in sv.stated_financial_inputs("Looking to buy in Tampines for $600k or less, near MRT")
    got = sv.stated_financial_inputs("We are buying a 4-room in Bishan for $650k, near MRT.")
    assert got["upfront"]["purchase_price"] == 650000.0


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


class ReadsCoordinates:
    """A model whose constraint reading fills comparable_resale with the given coordinates."""

    def __init__(self, where):
        self.where = where

    def with_structured_output(self, schema, method=None, include_raw=False):
        return self

    def invoke(self, _msgs):
        from agents.schemas import ExtractedConstraints
        return {"parsed": ExtractedConstraints(comparable_resale=self.where)}


@pytest.mark.parametrize("where,asked", [
    ({"latitude": 1.3329, "longitude": 103.7436, "flat_type": "4 ROOM"}, [COMPARABLES]),   # the user's own numbers
    ({"latitude": 1.35, "longitude": 103.85, "flat_type": "4 ROOM"}, []),                  # numbers the model made up
])
def test_coordinates_the_user_typed_are_an_ask_for_comparables(where, asked):
    # words the patterns do not know, so only the coordinates can make it an ask
    text = "4-room flats at 1.3329, 103.7436: what's the damage over the last year?"
    assert sv.requested_tools(text) == []
    with patch.object(sv, "get_llm", return_value=ReadsCoordinates(where)):
        out = sv.extract_constraints_node({"messages": [{"role": "user", "content": text}]})
    assert out["user_constraints"]["requested_tools"] == asked
    assert not sv.coordinates_stated({"latitude": 1.3329, "longitude": None}, text)


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
@pytest.mark.parametrize("agent,optional,ask", [
    ("financial", UPFRONT, "how much stamp duty do we pay?"),
    ("financial", UPFRONT, "How much would we need in cash and CPF to complete the purchase?"),
    ("market_sql", COMPARABLES, "what did similar flats sell for?"),
    ("market_sql", COMPARABLES, "How much did 4-room flats near latitude 1.35, longitude 103.85 sell for in the last 12 months?"),
])
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


# --- 1. requested_tools is code's reading for code: no model is shown it ------------------------------------------
class PromptRecorder(ToolListModel):
    """Records every prompt it is sent, reads no constraints (code's reading fills them) and calls no tool."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def with_structured_output(self, schema, method=None, include_raw=False):
        return ReadsCoordinates(None)

    def invoke(self, msgs):
        self.prompts.append(" ".join(str(m.content) for m in msgs))
        return AIMessage(content="done")


@pytest.mark.usefixtures("mock_domain_tools")
@pytest.mark.parametrize("mod", [A, B, C])
def test_no_model_prompt_names_the_requested_tools(mod):
    model = PromptRecorder()
    with patch.object(sv, "get_llm", return_value=model):
        state = mod.run("Find a 4-room flat in Tampines under $680k. What is the stamp duty?")
    assert state["user_constraints"]["requested_tools"] == [UPFRONT]    # code still has it
    tool_prompts = [p for p in model.prompts if "Extracted constraints (JSON)" in p]
    assert tool_prompts and not [p for p in model.prompts if "requested_tools" in p]
    evidence, _ = sv.synthesis_evidence(state)
    assert "requested_tools" not in evidence["user_constraints"]


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


def test_variant_c_costs_the_budget_for_a_cash_and_cpf_ask():
    # the review's case: the ask is several words from "cash", and the budget is the price to cost
    state = C.run("Find a 4-room flat in Tampines under $600k. We earn $9,000 a month. "
                  "How much would we need in cash and CPF to complete the purchase?")
    assert [r["arguments"]["purchase_price"] for r in state["tool_call_log"] if r["tool"] == UPFRONT] == [600000.0]
    assert "Upfront funding" in state["draft_response"]


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


@pytest.mark.usefixtures("mock_domain_tools", "broken_market_module")
@pytest.mark.parametrize("mod", [A, C])
def test_answer_says_what_a_broken_search_could_not_check(mod):
    # the blocks come from the distance search alone, so they are not said to match every stated constraint
    draft = mod.run(MOCK_E2)["draft_response"]
    assert "match all stated constraints" not in draft and "constraints that could be checked" in draft
    assert "Prices, remaining leases and flat types could not be checked: that tool is unavailable right now." in draft


@pytest.fixture
def broken_financial_module(tmp_path, monkeypatch):
    (tmp_path / "broken_financial_tools_fake.py").write_text('raise ImportError("simulated: a syntax error")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in tr.tools_for_agent("financial"):
        monkeypatch.setitem(tr.REAL_IMPL, name, ("broken_financial_tools_fake", name))
        monkeypatch.setitem(sv.load_config()["tools"]["use_mock"], name, "real")
    monkeypatch.setattr(tr, "_WARNED", set())


@pytest.mark.usefixtures("broken_financial_module")
def test_answer_says_which_calculations_a_broken_module_could_not_do():
    state = C.run(E1_ASKS)
    draft = state["draft_response"]
    for what in ("Upfront costs", "Housing grants", "The maximum budget"):
        assert f"{what} could not be checked: that tool is unavailable right now." in draft
    assert "match all stated constraints" in draft      # the calculators filter no block, so the search stands
    evidence, _ = sv.synthesis_evidence(state)           # the model writing the opening is told as well
    assert len(evidence["could_not_check"]) == 3
    ran = {"user_constraints": {}, "tool_call_log": [{"tool": UPFRONT, "ok": False, "error": "invalid arguments: x"}]}
    assert "could_not_check" not in sv.synthesis_evidence(ran)[0]   # an ordinary failed call changes no prompt


@pytest.mark.usefixtures("mock_domain_tools", "broken_market_module")
def test_model_writing_the_opening_is_told_what_could_not_be_checked():
    model = PromptRecorder()
    with patch.object(sv, "get_llm", return_value=model):
        C.run(MOCK_E2)
    final = model.prompts[-1]
    assert "never say the blocks meet every constraint" in final
    assert "Prices, remaining leases and flat types could not be checked" in final
    assert '"blocks_meeting_the_constraints_that_could_be_checked"' in final and "every_constraint\"" not in final


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
