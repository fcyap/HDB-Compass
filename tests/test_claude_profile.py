"""The claude profile (LLM_PROFILE=claude): Anthropic's Claude through langchain-anthropic, for the app only. No network:
the request payloads are built and checked, never sent. Experiment 1 refuses the profile (open models only)."""
from __future__ import annotations

import os
import subprocess

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

pytest.importorskip("langchain_anthropic")

from agents import llm as L  # noqa: E402


@pytest.fixture
def claude(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    for k in ("CLAUDE_MODEL", "LLM_MODEL", "LLM_BASE_URL", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")          # Claude Code's own setting: ignored
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/proxy")   # likewise: the key must not go there
    monkeypatch.delenv("HDB_READING_CONTEXT", raising=False)
    for k in ():
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@tool
def calculate_maximum_affordability(monthly_income: float) -> str:
    """Work out the maximum budget for a household income."""
    return "ok"


def payload(llm, **kwargs):
    return llm._get_request_payload([HumanMessage(content="hi")], **kwargs)


def test_the_profile_defaults_to_haiku_with_no_sampling_parameters(claude):
    p = L.active_profile()
    assert p["name"] == "claude" and p["model"] == "claude-haiku-5-5" and L.provider_of(p) == "anthropic"
    llm = L.get_llm()
    body = payload(llm)
    assert body["model"] == "claude-haiku-5-5" and body["max_tokens"] == 4096
    assert body["thinking"] == {"type": "disabled"}
    assert "temperature" not in body and "seed" not in body and "top_p" not in body
    assert llm.default_request_timeout == 60 and llm.max_retries == 2
    assert llm.anthropic_api_url == "https://api.anthropic.com"


def test_the_demo_timeouts_reach_the_client(claude):
    claude.setenv("LLM_TIMEOUT_S", "30")
    claude.setenv("LLM_MAX_RETRIES", "1")
    llm = L.get_llm()
    assert llm.default_request_timeout == 30 and llm.max_retries == 1


def test_a_missing_key_is_a_clear_error(claude):
    claude.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(L.ModelConfigurationError, match="ANTHROPIC_API_KEY"):
        L.active_profile()


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"])
def test_models_that_cannot_turn_thinking_off_are_refused(claude, model):
    claude.setenv("CLAUDE_MODEL", model)
    with pytest.raises(L.ModelConfigurationError, match="claude-haiku-5-5"):
        L.get_llm()
    status = L.model_status(check=True)        # the backend's model status and --check say so too
    assert status["status"] == "error" and "claude-haiku-5-5" in str(status)


def test_a_forced_tool_call_is_sent_as_asked(claude):
    # the policy agent forces its one tool on the first turn (supervisor.run_specialist); Haiku 5.5 accepts that
    bound = L.get_llm().bind_tools([calculate_maximum_affordability], tool_choice="calculate_maximum_affordability")
    body = payload(bound.bound, **bound.kwargs)
    assert body["tool_choice"] == {"type": "tool", "name": "calculate_maximum_affordability"}
    assert body["tools"][0]["name"] == "calculate_maximum_affordability"


def test_structured_reading_uses_output_config_format(claude):
    from agents.supervisor import LLMExtractedConstraints
    structured = L.get_llm().with_structured_output(LLMExtractedConstraints, method="json_schema", include_raw=True)
    bound = structured.first if hasattr(structured, "first") else structured
    found = []

    def walk(r, depth=0):
        if depth > 8 or r is None:
            return
        kw = getattr(r, "kwargs", None)
        if isinstance(kw, dict) and "output_config" in kw:
            found.append(kw["output_config"])
        for attr in ("bound", "first", "middle", "last", "steps", "runnable", "mapper", "default"):
            v = getattr(r, attr, None)
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if x is not None and x is not r:
                    walk(x, depth + 1)
        for x in (getattr(r, "steps__", None) or {}).values():
            walk(x, depth + 1)

    walk(structured)
    assert found and found[0]["format"]["type"] == "json_schema", bound


def test_replies_are_plain_text_and_keep_their_tool_calls(claude, monkeypatch):
    from langchain_anthropic import ChatAnthropic
    reply = AIMessage(content=[{"type": "text", "text": "Let me check. "},
                               {"type": "tool_use", "id": "toolu_1", "name": "calculate_maximum_affordability",
                                "input": {"monthly_income": 8500}}],
                      tool_calls=[{"id": "toolu_1", "name": "calculate_maximum_affordability",
                                   "args": {"monthly_income": 8500}}])
    monkeypatch.setattr(ChatAnthropic, "_generate",
                        lambda self, *a, **k: ChatResult(generations=[ChatGeneration(message=reply)]))
    out = L.get_llm().invoke([HumanMessage(content="hi")])
    assert out.content == "Let me check. "
    assert out.tool_calls[0]["args"] == {"monthly_income": 8500}


def test_the_model_check_uses_anthropics_models_api(claude, monkeypatch):
    import anthropic
    seen = {}

    class FakeModels:
        def retrieve(self, model):
            seen["model"] = model
            return type("Info", (), {"id": model})()

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)
            self.models = FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    ok, detail = L._probe_profile(L.active_profile())
    assert ok and seen["model"] == "claude-haiku-5-5" and seen["max_retries"] == 0
    assert seen["base_url"] == "https://api.anthropic.com"


def test_the_evaluation_refuses_the_claude_profile(claude):
    from evaluation.run_category_e_prelim import require_open_model_profile
    with pytest.raises(SystemExit, match="hosted"):
        require_open_model_profile()


# --- cluster/demo_start.sh --claude ---------------------------------------------------------------------------------
def test_demo_start_claude_needs_no_cluster_and_asks_for_the_key(tmp_path):
    from tests.test_demo_scripts import _stub, calls, run, sandbox
    env = sandbox(tmp_path, vite_engines=">=1.0.0")
    env.pop("ANTHROPIC_API_KEY", None)
    # the stand-in python answers the profile check the way agents.llm does when the key is missing
    _stub(tmp_path / ".venv" / "bin" / "python",
          'case "$*" in *active_profile*) grep -qE "^ANTHROPIC_API_KEY=." .env || '
          '{ echo "agents.llm.ModelConfigurationError: Set ANTHROPIC_API_KEY in .env for the claude profile." >&2; exit 1; };; esac; exit 0')
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=\n")
    out = run(tmp_path, env, "demo_start.sh", "--dry-run", "--claude")
    assert "Set ANTHROPIC_API_KEY in .env for the claude profile." in out.stdout and "fix the problems above first" in out.stdout
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-test-not-a-real-key\n")
    out = run(tmp_path, env, "demo_start.sh", "--dry-run", "--claude")
    assert out.returncode == 0 and "fix the problems" not in out.stdout, out.stdout + out.stderr
    assert "LLM_PROFILE=claude LLM_TIMEOUT_S=60 LLM_MAX_RETRIES=1 HDB_TRUST_MODEL_READING=0 HDB_READING_CONTEXT=1" in out.stdout
    assert "CLUSTER_LOGIN" not in out.stdout and "GPU job" not in out.stdout and calls(tmp_path, "ssh") == []
    traced = subprocess.run(["bash", "-x", str(tmp_path / "cluster" / "demo_start.sh"), "--dry-run", "--claude"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert "sk-ant-test" not in traced.stdout + traced.stderr   # not even bash -x shows the key


def test_demo_start_still_parses():
    assert subprocess.run(["bash", "-n", os.path.join(os.path.dirname(__file__), "..", "cluster", "demo_start.sh")]).returncode == 0


def test_the_constraint_reading_goes_straight_to_a_forced_tool_call(claude, monkeypatch):
    # json_schema would be refused: Anthropic allows 16 union-type fields per schema and LLMExtractedConstraints has ~70
    from agents import supervisor as sv
    llm = L.get_llm()
    assert llm._llm_type.startswith("anthropic")
    tried = []
    from langchain_anthropic import ChatAnthropic

    def fake(self, schema, method="function_calling", include_raw=False, **kw):
        tried.append(method)
        raise RuntimeError("stop here")

    monkeypatch.setattr(ChatAnthropic, "with_structured_output", fake)
    sv.read_constraints_with_model(llm, [sv.sys_msg("Read the constraints."), HumanMessage(content="4-room in Bedok")])
    assert tried == ["function_calling"]


def test_a_reading_with_one_bad_field_keeps_the_rest(claude, monkeypatch):
    # the review (6 Nov): a forced tool call carries its reading in tool_calls; trend_years 35 (max 30) had thrown it all
    # away and sent a blind repair turn
    from agents import supervisor as sv
    from langchain_anthropic import ChatAnthropic
    args = {"towns": ["BEDOK"], "flat_types": ["4 ROOM"], "max_budget_sgd": 600000, "trend_years": 35}
    raw = AIMessage(content="", tool_calls=[{"id": "toolu_1", "name": "LLMExtractedConstraints", "args": args}])

    class Bound:
        def invoke(self, _):
            return {"raw": raw, "parsed": None, "parsing_error": ValueError("trend_years")}

    monkeypatch.setattr(ChatAnthropic, "with_structured_output", lambda self, *a, **k: Bound())
    monkeypatch.setattr(ChatAnthropic, "invoke", lambda self, *a, **k: pytest.fail("no repair turn"))
    c, notes = sv.read_constraints_with_model(L.get_llm(), [sv.sys_msg("Read."), HumanMessage(content="4-room Bedok")])
    assert c.towns == ["BEDOK"] and c.flat_types == ["4 ROOM"] and c.max_budget_sgd == 600000
    assert any("dropped invalid fields trend_years" in n for n in notes)


def test_a_reply_that_is_not_the_models_api_is_not_available(claude, monkeypatch):
    import anthropic

    class FakeClient:
        def __init__(self, **kw):
            self.models = type("M", (), {"retrieve": lambda self, model: "<html>portal</html>"})()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    ok, detail = L._probe_profile(L.active_profile())
    assert not ok and "invalid" in detail


# --- HDB_TRUST_MODEL_READING: the model's reading stands (testing Claude without code's checks) ------------------------
def _fake_reading(monkeypatch, args):
    from langchain_anthropic import ChatAnthropic
    seen = {}

    class Bound:
        def invoke(self, _):
            return {"raw": AIMessage(content="", tool_calls=[{"id": "t1", "name": "TrustedModelReading", "args": args}]),
                    "parsed": None, "parsing_error": None}

    def wso(self, schema, *a, **k):
        seen["schema"] = schema
        return Bound()

    monkeypatch.setattr(ChatAnthropic, "with_structured_output", wso)
    return seen


def _read(query):
    from agents import supervisor as sv
    state = sv.initial_state(query=query, variant="C", map_state={}, previous=None, fast_path=False)
    return sv.extract_constraints_node(state)


def test_trusted_the_model_says_when_the_user_wants_their_own_grant(claude, monkeypatch):
    from agents import supervisor as sv
    claude.setenv("HDB_TRUST_MODEL_READING", "1")
    query = "Can you sort out the money side for us first?"          # no wording code reads as a grant question
    assert not sv._MONEY_AMOUNT_ASK.search(query)
    seen = _fake_reading(monkeypatch, {"wants_money_figures": True, "assumptions": ["Assumed a 4-room resale flat."]})
    out = _read(query)
    c = out["user_constraints"]
    assert seen["schema"] is sv.TrustedModelReading and "wants_money_figures" in sv.TrustedModelReading.model_json_schema()["properties"]
    assert c["money_intent"] and "model_money_intent" not in c
    assert c["assumptions"] == ["Assumed a 4-room resale flat."]             # the model's own text stands
    assert any("model reading trusted" in e for e in out["errors"])
    assert not any("replaced by code's" in e for e in out["errors"])


def test_trusted_the_grounding_checks_are_skipped_and_off_they_run(claude, monkeypatch):
    query = "Find 4-room flats under $600k near my office."
    _fake_reading(monkeypatch, {"towns": ["BEDOK"], "flat_types": ["4 ROOM"], "max_budget_sgd": 600000})
    claude.setenv("HDB_TRUST_MODEL_READING", "1")
    assert _read(query)["user_constraints"]["towns"] == ["BEDOK"]          # kept as the model read it
    claude.setenv("HDB_TRUST_MODEL_READING", "0")
    out = _read(query)
    assert out["user_constraints"]["towns"] == [] and any("not named in the message were dropped" in e for e in out["errors"])


def test_off_by_default_the_model_reading_has_no_money_field(claude, monkeypatch):
    from agents import supervisor as sv
    seen = _fake_reading(monkeypatch, {"wants_money_figures": True})
    out = _read("Can you sort out the money side for us first?")
    assert seen["schema"] is sv.LLMExtractedConstraints and not out["user_constraints"]["money_intent"]


# --- the earlier turns of the conversation, given to the model's reading under HDB_TRUST_MODEL_READING ---------------
AMK_ANSWER = ("Only Blk 216 meets every limit you set.\n**1 candidate block(s)** match all stated constraints:\n"
              "- Blk 216 ANG MO KIO AVE 1 (ANG MO KIO) — median $438,000, 49 yrs lease — mrt 241m\n")
TAMPINES_ANSWER = "**Near Blk 458 TAMPINES ST 42 (within 1000 m):**\n" + "\n".join(
    f"- Blk {i} TAMPINES ST {i % 90} (TAMPINES) — median $500,000, 60 yrs lease" for i in range(100, 200))
LATEST = "compare between 458 and the ang mo kio one i mentioned earlier"


def _history_state(latest=LATEST):
    from agents import supervisor as sv
    prev = {"messages": [{"role": "user", "content": "Find 4-room resale flats in Ang Mo Kio for us"},
                         {"role": "assistant", "content": AMK_ANSWER},
                         {"role": "user", "content": "within tampines"},
                         {"role": "assistant", "content": TAMPINES_ANSWER}],
            "user_constraints": {"towns": ["TAMPINES"], "flat_types": ["4 ROOM"]}}
    return sv.initial_state(query=latest, variant="C", map_state={}, previous=prev, fast_path=False)


def _record_messages(monkeypatch):
    from langchain_anthropic import ChatAnthropic
    sent = []

    class Bound:
        def invoke(self, messages):
            sent.append(messages)
            return {"raw": AIMessage(content="", tool_calls=[{"id": "t1", "name": "x", "args": {}}]),
                    "parsed": None, "parsing_error": None}

    monkeypatch.setattr(ChatAnthropic, "with_structured_output", lambda self, *a, **k: Bound())
    return sent


def test_trusted_the_reading_sees_the_blocks_named_earlier(claude, monkeypatch):
    from agents import supervisor as sv
    claude.setenv("HDB_TRUST_MODEL_READING", "1")
    sent = _record_messages(monkeypatch)
    out = sv.extract_constraints_node(_history_state())
    msgs = sent[0]
    assert "mentioned earlier" in msgs[0].content.lower() or "earlier turns" in msgs[0].content   # the context rule
    context = msgs[1].content
    assert "Blk 216 ANG MO KIO AVE 1" in context and "Blk 458 TAMPINES ST 42" in context
    assert "User: Find 4-room resale flats in Ang Mo Kio for us" in context and "User: within tampines" in context
    assert msgs[-1].content == f"Latest message (read this one):\n{LATEST}"
    assert LATEST not in context                                       # the latest message is not also in the history
    assert any("earlier turns given to the model (2 turns)" in e for e in out["errors"])


def test_the_context_is_capped(claude):
    from agents import supervisor as sv
    text, turns = sv.reading_context(_history_state())
    assert "(and 96 more)" in text and len(text) <= sv.READING_CONTEXT_CHARS
    many = {"messages": [{"role": "user", "content": f"question {i} " + "x" * 400} for i in range(20)]
            + [{"role": "user", "content": "latest"}]}
    text, turns = sv.reading_context(many)
    assert turns <= sv.READING_CONTEXT_TURNS and len(text) <= sv.READING_CONTEXT_CHARS and "question 19" in text


def test_off_the_reading_is_the_latest_message_alone(claude, monkeypatch):
    from agents import supervisor as sv
    claude.setenv("HDB_TRUST_MODEL_READING", "0")
    sent = _record_messages(monkeypatch)
    out = sv.extract_constraints_node(_history_state())
    assert len(sent[0]) == 2 and sent[0][1].content == LATEST
    assert not any("earlier turns given" in e for e in out["errors"])


def test_the_evaluation_runs_claude_only_when_asked_and_checks_it_through_anthropic(claude, monkeypatch):
    from agents import llm as llm_mod
    from evaluation import run_category_e_prelim as ev
    claude.setenv("HDB_EVAL_ALLOW_HOSTED", "claude")
    assert ev.require_open_model_profile() == "claude"
    claude.setenv("HDB_EVAL_ALLOW_HOSTED", "auto")
    claude.setenv("LLM_PROFILE", "auto")
    with pytest.raises(SystemExit):
        ev.require_open_model_profile()                   # auto can fall back to anything: never
    claude.setenv("LLM_PROFILE", "claude")
    monkeypatch.setattr(llm_mod, "_probe_anthropic", lambda p: (True, "advertised"))
    assert ev.server_answers(L.active_profile()) == (True, "advertised")



def test_context_with_the_checks_on_the_default_for_claude(claude, monkeypatch):
    # the setup chosen after the 6 Nov full run: earlier turns and Claude's grant judgement, code's checks still on
    from agents import supervisor as sv
    claude.setenv("HDB_READING_CONTEXT", "1")
    claude.setenv("HDB_TRUST_MODEL_READING", "0")
    sent = _record_messages(monkeypatch)
    out = sv.extract_constraints_node(_history_state())
    assert "Blk 216 ANG MO KIO AVE 1" in sent[0][1].content
    assert not any("model reading trusted" in e for e in out["errors"])
    seen = _fake_reading(monkeypatch, {"towns": ["BEDOK"], "wants_money_figures": True,
                                       "assumptions": ["Assumed a 4-room resale flat."]})
    out = _read("Can you sort out the money side for us first?")
    c = out["user_constraints"]
    assert seen["schema"] is sv.TrustedModelReading and c["money_intent"]          # Claude's judgement counts
    assert c["towns"] == [] and any("not named in the message were dropped" in e for e in out["errors"])  # checks on
    assert any("replaced by code's" in e for e in out["errors"])                     # code's assumptions
