"""The runtime auditor reads the conversation, not only the latest turn.

Before: a true reference to an earlier turn ("Blk 174D, which you saw at $520,000", "with your $8,000 income") was
rejected as ungrounded, so after two retries the model's sentences were stripped from follow-up answers. Earlier turns'
tool facts (prior_block_facts), the user's earlier constraints and every user message now count as evidence; earlier
assistant replies never do. HDB_AUDITOR_HISTORY=off restores the turn-by-turn check (ablation).
"""
import pytest

from agents import guardrails as g


@pytest.fixture(autouse=True)
def _rules_only(monkeypatch):
    monkeypatch.setenv("HDB_AUDITOR_LLM", "off")     # measure the rule check; never call a judge model
    monkeypatch.delenv("HDB_AUDITOR", raising=False)
    monkeypatch.delenv("HDB_AUDITOR_HISTORY", raising=False)


def _state(opening: str, **extra):
    base = {
        "messages": [
            {"role": "user", "content": "4 room in Punggol under 600k, my income is $8,000"},
            {"role": "assistant", "content": "Blk 174D Edgedale Plains: $520,000. Blk 999 Fake St: $480,000."},
            {"role": "user", "content": "which of those is near a good primary school?"},
        ],
        "prior_block_facts": [{"name": "Blk 174D Edgedale Plains", "price": 520000, "lease": 76.2, "dist": 350}],
        "prior_constraints": {"towns": ["PUNGGOL"], "max_price": 600000},
        "user_constraints": {"towns": ["PUNGGOL"]},
        "candidate_blocks_intersection": [{"block": "301", "street": "Punggol Cent", "median_resale_price": 585000}],
        "model_opening": opening,
        "draft_response": opening,
    }
    base.update(extra)
    return base


def _verdict(state):
    return g.runtime_auditor_node(state)["auditor_verdict"]


def test_a_true_reference_to_an_earlier_turns_block_passes():
    v = _verdict(_state("Blk 174D, which you saw earlier at $520,000, is not near a school."))
    assert v["passed"], v["ungrounded_claims"]


def test_a_figure_the_user_gave_in_an_earlier_turn_may_be_repeated():
    v = _verdict(_state("With your $8,000 income, Blk 301 fits."))
    assert v["passed"], v["ungrounded_claims"]


def test_an_invented_figure_is_still_rejected_mid_conversation():
    v = _verdict(_state("Blk 174D now costs $455,000."))
    assert not v["passed"] and "$455,000" in v["ungrounded_claims"]


def test_an_earlier_assistant_reply_is_not_evidence():
    # Blk 999 and $480,000 appear only in the assistant's own earlier message, never in a tool result
    v = _verdict(_state("Blk 999 at $480,000 is the cheapest."))
    assert not v["passed"]
    assert "Blk 999" in v["ungrounded_claims"] and "$480,000" in v["ungrounded_claims"]


def test_history_switch_off_restores_the_turn_by_turn_check(monkeypatch):
    monkeypatch.setenv("HDB_AUDITOR_HISTORY", "off")
    v = _verdict(_state("Blk 174D, which you saw earlier at $520,000, is not near a school."))
    assert not v["passed"]
    assert g.history_evidence(_state("x")) == ""


def test_a_first_turn_is_judged_exactly_as_before():
    st = _state("Blk 301 Punggol Cent costs $585,000.",
                messages=[{"role": "user", "content": "4 room in Punggol"}], prior_block_facts=None, prior_constraints=None)
    assert g.history_evidence(st) == ""
    assert g.user_text(st) == "4 room in Punggol"
    assert _verdict(st)["passed"]


def test_the_judge_reads_earlier_results_ahead_of_the_long_data():
    data = g._judge_data(_state("x"))
    assert "EARLIER RESULTS" in data and "Edgedale Plains" in data
    assert data.index("EARLIER RESULTS") < data.index("Punggol Cent")
