"""The auditor's self-correction loop through the real Variant C graph, with a scripted stand-in for the writer model.

The graph is run end to end (extraction, tools, synthesize, auditor, retry gate, finalize). Only the writer's text is
scripted, so the auditor decides on real state. No network, no key: the model-only paths fall back to code as they do
when a provider fails.
"""
import json
from pathlib import Path

import pytest

from agents import supervisor as sv
from agents.arch_3_specialist_5agent import run

EVAL = Path(__file__).resolve().parent.parent / "evaluation"
QUERY = next(i["query"] for i in json.loads((EVAL / "category_e_queries.json").read_text(encoding="utf-8")) if i["id"] == "E1-01")

# a true comparison (so synthesize's own checks keep it) that also promises a gain (so the auditor must refuse it)
BAD = "Blk 288B is cheaper and nearer the MRT station than Blk 266, so it is guaranteed to appreciate."
CLEAN = "Blk 266 is cheaper than Blk 286D, though Blk 286D is nearer an MRT station."


class _Reply:
    def __init__(self, content):
        self.content = content


def _writer(script, seen):
    class Stub:
        def invoke(self, msgs, *args, **kwargs):
            critique = [str(m.content) for m in msgs if "rejected by the auditor" in str(getattr(m, "content", ""))]
            seen.append(critique[0] if critique else None)
            return _Reply(script(bool(critique)))

        def __getattr__(self, name):          # structured output / tool binding: not available, code takes over
            raise AttributeError(name)
    return Stub()


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")
    monkeypatch.setenv("HDB_AUDITOR_LLM", "off")        # the rule layer decides; the judge needs a real model
    monkeypatch.setenv("HDB_ADDONS", "off")

    def install(script):
        seen = []
        monkeypatch.setattr(sv, "get_llm", lambda *a, **k: _writer(script, seen))
        return seen
    return install


def test_a_rejected_sentence_is_rewritten_with_the_critique_and_then_passes(scripted):
    seen = scripted(lambda saw_critique: CLEAN if saw_critique else BAD)
    state = run(QUERY)
    assert len(seen) == 2 and seen[0] is None                      # the writer ran twice, the second time told why
    assert "speculative claim" in seen[1] and "guaranteed" in seen[1]
    verdict = state["auditor_verdict"]
    assert verdict["passed"] is True and verdict["retry_count"] == 1
    assert "guaranteed" not in state["draft_response"]


def test_a_writer_that_never_complies_is_stopped_after_two_rewrites(scripted):
    seen = scripted(lambda saw_critique: BAD)
    state = run(QUERY)
    assert len(seen) == 3                                           # the first draft and the two rewrites
    verdict = state["auditor_verdict"]
    assert verdict["passed"] is False and verdict["retry_count"] == 3


def test_a_refused_sentence_is_never_shipped_and_the_facts_stay(scripted):
    scripted(lambda saw_critique: BAD)
    state = run(QUERY)
    answer = state["draft_response"]
    assert "guaranteed" not in answer and "appreciate" not in answer
    assert state["model_opening"] == ""
    assert "Blk 288B" in answer and "candidate block" in answer     # code's answer and the facts section are untouched
    assert any("model sentences removed" in e for e in state["errors"])


def test_a_clean_writer_costs_one_call_and_no_rejection(scripted):
    seen = scripted(lambda saw_critique: CLEAN)
    state = run(QUERY)
    assert len(seen) == 1
    assert state["auditor_verdict"]["passed"] is True and state["auditor_verdict"]["retry_count"] == 0


def test_without_sentences_removes_only_the_models_text():
    from agents import guardrails as g
    draft = "Code answer. " + BAD + "\n\n**4 candidate block(s)** match."
    out = g._without_sentences(draft, BAD)
    assert BAD not in out and out.startswith("Code answer.") and "candidate block(s)" in out
