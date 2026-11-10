"""HDB_ANSWER_STYLE=conversational: the model writes the reply from code's checked facts, verify_reply checks it, and a
reply that fails twice gives today's answer. A stand-in model; no network."""
from __future__ import annotations

import os

import pytest

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import reply_composer as rc  # noqa: E402
from agents import supervisor as sv  # noqa: E402

SEARCH = "Find 4-room resale flats in Ang Mo Kio under $650,000 within 400m of an MRT station."


@pytest.fixture(scope="module")
def searched():
    state = C.graph.invoke(sv.initial_state(query=SEARCH, variant="C", map_state={}, previous=None, fast_path=True))
    return state, sv.template_response(state)


def problems(reply, searched, clar=None):
    state, facts = searched
    return rc.verify_reply(reply, state, facts, clar)


GOOD = ("Yes, there are 40 blocks that fit. Blk 216 is the cheapest at $438,000, while Blk 207 is the nearest to the "
        "MRT at 88m, so it comes down to price or convenience. The full list is in the details below and on the map. "
        "Would you like me to check what you can afford?")


def test_a_reply_with_the_facts_figures_passes(searched):
    assert problems(GOOD, searched) == []
    assert problems(GOOD.replace("$438,000", "$438k"), searched) == []          # the same figure, abbreviated


@pytest.mark.parametrize("bad, why", [
    ("Blk 216 is the cheapest at $450,000.", "not in the checked facts"),         # a price code never gave
    ("Blk 216 costs $505,000, the cheapest.", "not Blk 216's figure"),             # another block's price
    ("Blk 999 is a good option near the MRT.", "Blk 999"),                         # a block nobody listed
    ("There are 55 blocks that fit.", "not in the checked facts"),                 # a count code never gave
    ("Prices rose 7.5% last year.", "not in the checked facts"),                   # a percentage code never gave
    ("## Results\nBlk 216 is the cheapest.", "headings"),
    ("Blk 216 is cheapest. Want more? Or another town?", "more than one question"),
])
def test_a_reply_that_states_something_else_is_refused(searched, bad, why):
    assert any(why in p for p in problems(bad, searched)), problems(bad, searched)


def test_flat_types_and_street_numbers_are_not_figures(searched):
    assert problems("The 4-room flats at Blk 216 ANG MO KIO AVE 1 are the cheapest.", searched) == []


def test_only_codes_question_while_one_is_open(searched):
    clar = {"kind": "financial", "question": "To work out your grant, could you tell me X?"}
    assert problems("Happy to help with your grant. To work out your grant, could you tell me X?", searched, clar) == []
    assert any("ask only the question" in p for p in
               problems("What is your income? To work out your grant, could you tell me X?", searched, clar))
    assert any("ask only the question" in p for p in problems("Happy to help with your grant.", searched, clar))


class Model:
    """Hands back the given replies in turn and counts the calls."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, messages):
        self.calls += 1
        return type("Reply", (), {"content": self.replies.pop(0)})()


def test_a_bad_reply_is_written_again_and_a_second_failure_gives_todays_answer(searched):
    state, facts = searched
    model = Model("Blk 216 is $450,000.", GOOD)
    reply, notes = rc.compose(model, state, facts, {}, None, None, "")
    assert reply == GOOD and model.calls == 2 and "second try" in notes[0]
    model = Model("Blk 216 is $450,000.", "Blk 999 is best.")
    reply, notes = rc.compose(model, state, facts, {}, None, None, "")
    assert reply is None and model.calls == 2 and "checked facts used" in notes[0]
    assert not any(w in notes[0] for w in ("LLM", "rule-based fallback used", "deterministic plan used"))


def test_the_synthesis_shows_the_reply_with_the_facts_as_details(searched, monkeypatch):
    state, facts = searched
    monkeypatch.setenv("HDB_ANSWER_STYLE", "conversational")
    monkeypatch.setattr(sv, "get_llm", lambda **k: Model(GOOD))
    out = sv.synthesize_node(state)
    assert out["answer_style"] == "composed" and out["model_opening"] == GOOD
    assert out["draft_response"].startswith(GOOD) and rc.DETAILS_SEP in out["draft_response"]
    assert out["draft_response"].split(rc.DETAILS_SEP)[1].startswith(facts[:60])
    assert out["fallback_draft"] and GOOD not in out["fallback_draft"]


def test_two_bad_replies_give_todays_answer(searched, monkeypatch):
    state, _ = searched
    monkeypatch.setenv("HDB_ANSWER_STYLE", "conversational")
    monkeypatch.setattr(sv, "get_llm", lambda **k: Model("Blk 999 is best.", "Blk 998 is best."))
    out = sv.synthesize_node(state)
    assert out["answer_style"] == "template" and rc.DETAILS_SEP not in out["draft_response"]
    assert "candidate block(s)" in out["draft_response"]


def test_a_refused_composed_reply_is_not_composed_again(searched, monkeypatch):
    state, _ = searched
    monkeypatch.setenv("HDB_ANSWER_STYLE", "conversational")
    monkeypatch.setattr(sv, "get_llm", lambda **k: pytest.fail("no model call after a refusal"))
    refused = {**state, "answer_style": "composed", "auditor_verdict": {"passed": False, "critique": "x", "retry_count": 1}}
    out = sv.synthesize_node(refused)
    assert out["answer_style"] == "template" and rc.DETAILS_SEP not in out["draft_response"]


def test_the_auditor_cutting_a_composed_reply_gives_todays_answer(monkeypatch):
    monkeypatch.setattr(sv, "_AUDITOR", lambda st: {"draft_response": "cut", "auditor_verdict": {"passed": False}})
    out = sv.audit_node({"answer_style": "composed", "fallback_draft": "today's answer"})
    assert out["draft_response"] == "today's answer" and out["answer_style"] == "template"
    out = sv.audit_node({"answer_style": "template", "fallback_draft": ""})
    assert out["draft_response"] == "cut"                                         # the template path is the auditor's own


def test_template_and_focused_never_call_the_composer(searched, monkeypatch):
    state, _ = searched
    monkeypatch.setattr(rc, "compose", lambda *a, **k: pytest.fail("composer called"))
    for style in ("template", "focused"):
        monkeypatch.setenv("HDB_ANSWER_STYLE", style)
        out = sv.synthesize_node(state)
        assert out["answer_style"] == "template"


class StreamingModel(Model):
    def stream(self, messages):
        self.calls += 1
        text = self.replies.pop(0)
        for i in range(0, len(text), 7):
            yield type("Chunk", (), {"content": text[i:i + 7]})()


def test_the_reply_streams_one_checked_sentence_at_a_time(searched, monkeypatch):
    state, facts = searched
    sent = []
    monkeypatch.setattr(rc, "stream_writer", lambda: sent.append)
    reply, _ = rc.compose(StreamingModel(GOOD), state, facts, {}, None, None, "")
    assert reply == GOOD
    deltas = [s["reply_delta"] for s in sent if "reply_delta" in s]
    assert len(deltas) >= 3 and "".join(deltas).strip() == GOOD


def test_a_wrong_sentence_stops_the_stream_and_resets_it(searched, monkeypatch):
    state, facts = searched
    sent = []
    monkeypatch.setattr(rc, "stream_writer", lambda: sent.append)
    bad = "Blk 216 is the cheapest at $438,000. Blk 999 is also nice. Blk 207 is near the MRT."
    reply, _ = rc.compose(StreamingModel(bad, GOOD), state, facts, {}, None, None, "")
    assert reply == GOOD
    first = sent[:sent.index({"reply_reset": True})]
    assert [s["reply_delta"].strip() for s in first] == ["Blk 216 is the cheapest at $438,000."]   # stopped at Blk 999


def test_nothing_streams_while_the_auditors_model_judge_is_on(monkeypatch):
    monkeypatch.setenv("HDB_AUDITOR_LLM", "on")
    assert rc.stream_writer() is None


@pytest.mark.parametrize("sentence", [
    "There are 40 blocks that fit, and the full list is in the details below.",   # a count, no block ranked
    "Blk 209 is one of the closest options.",                                     # a hedge
    "I can't say how close each block is to a hawker centre.",                    # no distance stated
])
def test_a_reply_sentence_that_ranks_nothing_stands(searched, sentence):
    state, facts = searched
    assert rc.verify_reply(sentence, state, facts, shape=False) == []


def test_a_wrong_superlative_about_a_named_block_is_still_refused(searched):
    state, facts = searched
    assert any("comparison" in p for p in rc.verify_reply("Blk 113 is the nearest to the MRT.", state, facts, shape=False))


def test_a_figure_from_an_earlier_answer_may_be_repeated(searched):
    state, facts = searched
    earlier = {**state, "messages": [{"role": "user", "content": "x"},
                                     {"role": "assistant", "content": "Raising the budget to $715,000 unlocks 2 blocks."},
                                     {"role": "user", "content": "cheaper?"}]}
    assert rc.verify_reply("As before, raising the budget to $715,000 unlocks 2 blocks.", earlier, facts, shape=False) == []


def test_a_blocks_lease_and_distance_must_be_as_the_facts_write_them(searched):
    state, facts = searched
    assert rc.verify_reply("Blk 216 is at $438,000 with 49 years left, 241m from the MRT.", state, facts, shape=False) == []
    assert any("not Blk 216's figure" in p for p in
               rc.verify_reply("Blk 216 is at $438,000 with 48 years left.", state, facts, shape=False))
    assert any("not Blk 216's figure" in p for p in rc.verify_reply("Blk 216 is 250m from the MRT.", state, facts, shape=False))


def test_the_number_of_matching_blocks_is_the_total_not_the_listed():
    facts = "**The 100 cheapest of 121 matching blocks** (narrow the search), meeting all stated constraints:"
    state = {"messages": [{"role": "user", "content": "4-room in Tampines"}]}
    assert any("121 blocks match" in p for p in rc.verify_reply("There are 100 four-room blocks that fit.", state, facts, shape=False))
    assert not any("blocks match" in p for p in rc.verify_reply("There are 121 blocks that match.", state, facts, shape=False))


def test_a_check_that_ran_is_not_called_unchecked(searched):
    state, facts = searched
    assert any("every check ran" in p for p in
               rc.verify_reply("The MRT distance hasn't been checked yet.", state, facts, shape=False))


def test_block_numbers_are_not_a_distance(searched):
    state, facts = searched
    assert rc.verify_reply("Two things would help: the asking prices for Blk 216 and Blk 207, and whether being near "
                           "the MRT matters to you.", state, facts, shape=False) == []
