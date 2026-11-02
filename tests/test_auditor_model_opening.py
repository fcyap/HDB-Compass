"""
What the runtime auditor judges: state["model_opening"] (tests/test_auditor_model_opening.py). Owned by Member 4.

On 2 Nov, Member 5's auditor judged the whole opening of every Variant C answer, including code's own answer sentence
("Blk 228 is the cheapest, while Blk 288B is nearest an MRT station"), and its model judge rejected 15 of 15 correct
Category E answers three times each. synthesize now records the model's own sentences, exactly as kept by its checks,
in model_opening; docs/INTERFACES.md section 3 tells the auditor to judge only that.

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_auditor_model_opening.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ["LLM_PROFILE"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import supervisor as sv  # noqa: E402
from tests.test_code_opening import Writer, state_for  # noqa: E402


def synthesize(state, model_text):
    with patch.object(sv, "get_llm", return_value=Writer(model_text)):
        return sv.synthesize_node({**state, "draft_response": ""})


def test_only_the_models_kept_sentences_are_the_model_opening():
    s = state_for("E1-01")
    facts = sv.template_response(s)
    kept = "Consider which matters more to you, the price or the walk to the station."
    out = synthesize(s, "Blk 228 costs $532,500. " + kept)   # the first sentence states a figure and is dropped
    assert out["model_opening"] == kept
    code_answer = sv.code_opening(s, facts)[0]
    assert code_answer and code_answer not in out["model_opening"]
    assert out["draft_response"].startswith(code_answer) and kept in out["draft_response"]


def test_a_model_whose_sentences_are_all_dropped_has_no_opening_to_judge():
    out = synthesize(state_for("E1-01"), "Blk 228 costs $532,500. The cheapest is $532,500.")
    assert out["model_opening"] == ""
    assert out["draft_response"]   # the answer is still code's opening and the facts


def test_no_model_means_no_model_opening():
    s = state_for("E1-01")
    assert s["model_opening"] == ""
    assert sv.synthesize_node({**s, "draft_response": ""})["model_opening"] == ""


def test_a_new_turn_starts_with_an_empty_model_opening():
    assert sv.initial_state("4-room in Tampines", "C")["model_opening"] == ""


def test_with_code_written_facts_off_the_whole_draft_is_the_models():
    s = state_for("E1-01")
    text = "Here are some blocks in Jurong East that fit your budget."
    with patch.object(sv, "code_written_facts_on", return_value=False):
        out = synthesize(s, text)
    assert out["model_opening"] == text and out["draft_response"].startswith(text)


def test_the_contract_names_model_opening():
    doc = (Path(__file__).resolve().parent.parent / "docs" / "INTERFACES.md").read_text()
    assert 'state["model_opening"]' in doc and "Never judge code's answer sentence" in doc
