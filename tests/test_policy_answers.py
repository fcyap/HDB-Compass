"""
Policy answers in the supervisor (Member 4). Member 3's policy search returns raw markdown chunks; the facts section
must show them as readable lines taken from the passages (never invented), and an opening about policy alone may state
a figure only as the policy lines shown write it. No model needed: the real search runs offline (read-only) and the
chunks come from Member 3's data/policy_chunks.json.

    LLM_PROFILE=none .venv/bin/pytest tests/test_policy_answers.py -q
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

from agents import supervisor as sv  # noqa: E402
from agents.tool_registry import resolve_callable  # noqa: E402

QUERIES = {q["id"]: q["query"] for q in json.loads((ROOT / "evaluation" / "category_c_queries.json").read_text())}
CHUNKS = json.loads((ROOT / "data" / "policy_chunks.json").read_text())
URL = "https://www.hdb.gov.sg/residential/buying-a-flat/new-flats/flat-classification"


def chunk(title: str) -> dict:
    """The chunk of Member 3's corpus whose section ends with this title and has text under its header."""
    return next(c for c in CHUNKS if c["section"].endswith(title) and sv._policy_lines(c["text"]))


def search(qid: str) -> list:
    return resolve_callable("retrieve_hdb_policy_knowledge")(query=QUERIES[qid])["passages"]


def words(text: str) -> set:
    return set(re.findall(r"[A-Za-z]+|\d+", text))


# --- the facts section ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("qid", ["C1-01", "C1-02", "C2-02"])
def test_real_passages_read_as_a_clean_section(qid):
    passages = search(qid)
    lines = sv.policy_fact_lines(passages, QUERIES[qid])
    source = "\n".join(p["text"] for p in passages)
    titles = [i for i, line in enumerate(lines) if line.startswith("- **")]
    assert 1 <= len(titles) <= 3
    for i, line in enumerate(lines):
        body = line.lstrip(" -")
        assert not body.startswith(("[", "#", "|", "---", "$$")) and "|" not in body and ":---" not in body, line
        if i in titles:
            assert re.fullmatch(r"- \*\*[^*\[\]]+\*\*(?: \([^*]+\))?", line), line     # the bold section title
        else:
            assert line.startswith("  - ") and "**" not in line and len(body) <= sv.POLICY_LINE_CHARS + 1, line
        if not body.startswith("Source: "):
            assert words(body) <= words(source), line                     # every word comes from the passages
    for start, end in zip(titles, titles[1:] + [len(lines)]):
        shown = lines[start + 1:end]
        assert shown[-1].startswith("  - Source: https://")
        assert 2 <= len(shown) - 1 <= sv.POLICY_LINES_SHOWN, shown
    answer = sv.template_response({"messages": [{"role": "user", "content": QUERIES[qid]}], "rag_policy_passages": passages})
    assert answer.startswith("**Policy:**\n- **") and "HFE" in answer


def test_header_only_chunks_are_skipped_and_a_section_is_shown_once():
    header_only = {"text": "[HDB Flat Classification Framework: Standard, Plus, and Prime Flats]\n---", "source_url": URL,
                   "section": "HDB Flat Classification Framework: Standard, Plus, and Prime Flats"}
    mop = chunk("4.2 10-Year Minimum Occupation Period (MOP)")
    shown = sv.shown_policy_passages([header_only, mop, dict(mop), chunk("4.3 Lifelong Whole-Flat Rental Prohibition")],
                                     "How long is the MOP for a Plus flat?")
    assert [s["title"] for s in shown] == ["10-Year Minimum Occupation Period (MOP)", "Lifelong Whole-Flat Rental Prohibition"]
    assert shown[0]["parent"] == "Deep Dive: Plus Flats Regulations"
    assert shown[0]["lines"][0] == "Plus flat owners are subject to the same 10-year MOP as Prime flats."
    assert sv.policy_fact_lines([header_only]) == ["- The policy search found no passage that answers this. Please check with HDB."]


def test_at_most_three_sections_are_shown():
    titles = ["4.1 Subsidy Recovery (Clawback) Rate", "4.2 10-Year Minimum Occupation Period (MOP)",
              "4.3 Lifelong Whole-Flat Rental Prohibition", "5.1 Baseline Parameters"]
    shown = sv.shown_policy_passages([chunk(t) for t in titles], "Plus flat rules")
    assert len(shown) == 3 and shown[-1]["title"] == "Lifelong Whole-Flat Rental Prohibition"


def test_table_rows_read_as_text_and_separators_are_dropped():
    matrix = next(c for c in CHUNKS if c["section"].endswith("2. Comprehensive Classification Comparison Matrix") and "|" in c["text"])
    lines = sv._policy_lines(matrix["text"])
    assert "Minimum Occupation Period (MOP) — Standard Flats: 5 years; Plus Flats: 10 years; Prime Flats: 10 years" in lines
    assert not any("---" in line or "|" in line or "Policy Dimension —" in line for line in lines)
    ehg = next(c for c in CHUNKS if c["section"].endswith("2.1 EHG for First-Timer Families") and "|" in c["text"])
    assert "Up to $1,500: $120,000" in sv._policy_lines(ehg["text"])


def test_a_bare_lead_in_goes_in_front_of_the_lines_under_it():
    lines = sv._policy_lines(chunk("4. Singles Grant for Resale Flats")["text"])
    assert "Grant Amounts — 5-room Resale Flat: $25,000" in lines
    assert "Grant Amounts" not in lines and "Monthly Income Ceiling: Gross monthly income not exceeding $7,000." in lines


def test_the_lines_shown_answer_each_part_of_the_question():
    # "minimum age and income ceiling": one line for each, although longer lines mention the flat type more often
    shown = sv.shown_policy_passages([chunk("1.3 Single Singapore Citizen (SSC) Scheme")], QUERIES["C2-02"])[0]
    assert any(line.startswith("Minimum Age Requirement:") for line in shown["lines"])
    assert "BTO Income Ceiling: Average gross monthly income must not exceed $7,000/month." in shown["lines"]
    assert 2 <= len(shown["lines"]) <= sv.POLICY_LINES_SHOWN


def test_long_lines_are_cut_at_a_sentence_or_word():
    line = "First sentence is here. " + "word " * 80
    assert sv._trim_line(line) == "First sentence is here."
    cut = sv._trim_line("x" * 10 + " longword" * 60)
    assert cut.endswith("…") and len(cut) <= sv.POLICY_LINE_CHARS + 1 and "longwor…" not in cut


# --- the opening ----------------------------------------------------------------------------------------------------
PLUS = [chunk("4.2 10-Year Minimum Occupation Period (MOP)"), chunk("4.1 Subsidy Recovery (Clawback) Rate"),
        chunk("5.1 Baseline Parameters")]


@pytest.mark.parametrize("sentence", [
    "Plus flats have a 10-year MOP.",
    "Plus flats have a 10 year MOP and a 3% to 6% clawback on the first resale.",
    "Standard flats have a 5-year MOP and 0% clawback.",
    "Neither Prime nor Plus flats can be rented out whole."])
def test_an_opening_may_state_a_figure_the_policy_lines_give(sentence):
    assert sv.keep_supported_sentences(sentence, "", QUERIES["C1-01"], PLUS) == (sentence, 0)


@pytest.mark.parametrize("sentence", [
    "Prime flats carry a 6% to 9% clawback.",          # not in these passages
    "The clawback is 3% to 9%.",                       # both ends appear, the range does not
    "Plus flats have a ten-year MOP.",                 # spelled out: not as the passages write it
    "Blk 123 has a 10-year MOP.",                      # names a block
    "Plus flats have a 10-year MOP and cost $1,200,000.",
    "A Plus flat resells for about $900,000.",         # only in the passage's worked example
    "The clawback on a $900k sale is 6%."])            # an amount not written as the passages write it
def test_an_opening_figure_not_in_the_policy_lines_is_dropped(sentence):
    assert sv.keep_supported_sentences(sentence, "", QUERIES["C1-01"], PLUS) == ("", 1)


def test_without_policy_passages_every_figure_is_still_dropped():
    assert sv.keep_supported_sentences("Plus flats have a 10-year MOP.", "") == ("", 1)
    assert sv.keep_supported_sentences("Plus flats have a 10-year MOP.", "", "", []) == ("", 1)


def test_policy_figures_are_allowed_only_in_an_answer_about_policy_alone():
    state = {**sv.initial_state(QUERIES["C1-01"], "C"), "rag_policy_passages": PLUS}
    assert sv.policy_answer_passages(state) == PLUS
    for extra in ({"financial_breakdown": {"calculate_cpf_housing_grants": {"total_grants": 80000}}},
                  {"sql_market_results": {"compute_town_price_timeseries": {}}},
                  {"candidate_blocks_intersection": [{"block": "123", "street": "X"}]}):
        assert sv.policy_answer_passages({**state, **extra}) is None
    assert sv.policy_answer_passages({**state, "rag_policy_passages": []}) is None


@pytest.mark.usefixtures("mock_domain_tools")
def test_no_policy_figures_while_the_policy_search_is_mocked():
    state = {**sv.initial_state(QUERIES["C1-01"], "C"), "rag_policy_passages": PLUS,
             "tool_call_log": [{"tool": "retrieve_hdb_policy_knowledge"}]}
    assert sv.policy_answer_passages(state) is None


def test_synthesis_keeps_a_policy_opening_and_the_code_written_policy_lines():
    state = {**sv.initial_state(QUERIES["C1-01"], "C"), "rag_policy_passages": PLUS}
    seen = {}

    class Writer:
        def invoke(self, msgs):
            seen["prompt"] = msgs[-1].content
            return AIMessage(content="Plus flats have a 10-year MOP, while Standard flats have a 5-year MOP. "
                                     "Plus flats resell for about $900k.")

    with patch.object(sv, "get_llm", return_value=Writer()):
        out = sv.synthesize_node(state)
    facts = sv.template_response(state)
    assert out["draft_response"] == ("Plus flats have a 10-year MOP, while Standard flats have a 5-year MOP.\n\n" + facts)
    assert out["errors"] == ["synthesize: dropped 1 opening sentence(s) that stated figures"]
    assert "exactly as the Policy lines above write it" in seen["prompt"]
    with patch.object(sv, "get_llm", return_value=Writer()):
        mixed = sv.synthesize_node({**state, "financial_breakdown": {"calculate_cpf_housing_grants": {"status": "ok",
                                    "enhanced_housing_grant": 0, "family_grant": 80000, "proximity_housing_grant": 0,
                                    "total_grants": 80000}}})
    assert "10-year MOP, while" not in mixed["draft_response"]           # grants shown too: the opening states no figure
    assert "write no numbers except block names" in seen["prompt"]
