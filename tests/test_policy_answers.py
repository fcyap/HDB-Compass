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
# The policy corpus as indexed: verbatim official pages (data/policy_sources/, data/ingest_hdb_policy_rag.py).
CHUNKS = json.loads((ROOT / "data" / "policy_chunks.json").read_text())
URL = "https://www.hdb.gov.sg/residential/buying-a-flat/new-flats/flat-classification"


def chunk(title: str, containing: str = "") -> dict:
    """The chunk of the corpus whose section ends with this title, has text under its header and contains the text."""
    return next(c for c in CHUNKS if c["section"].endswith(title) and containing in c["text"]
                and sv._policy_lines(c["text"]))


FRAMEWORK = "Key features of Standard, Plus, and Prime flats"
MOP = chunk("Minimum occupation period (MOP)", "the MOP is 10 years")              # HDB, Conditions After Buying a New Flat
WHOLE_FLAT = chunk("Renting out whole flat")
TABLE = chunk(FRAMEWORK, "| Classification of flats |")                          # the Standard/Plus/Prime comparison
RECOVERY = chunk(FRAMEWORK, "subsidy recovery percentage is commensurate")


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
    assert 1 <= len(titles) <= sv.POLICY_PASSAGES_SHOWN
    for i, line in enumerate(lines):
        body = line.lstrip(" -")
        assert not body.startswith(("[", "#", "|", "---", "$$")) and "|" not in body and ":---" not in body, line
        if i in titles:
            assert re.fullmatch(r"- \*\*[^*\[\]]+\*\*(?: \([^*]+\))?", line), line     # the bold section title
        else:
            assert line.startswith("  - ") and "**" not in line and len(body) <= sv.POLICY_LINE_MAX + 1, line
        if not body.startswith("Source: "):
            assert words(body) <= words(source), line                     # every word comes from the passages
    for start, end in zip(titles, titles[1:] + [len(lines)]):
        shown = lines[start + 1:end]
        assert shown[-1].startswith("  - Source: https://")
        # 2-4 lines when the section has them (POLICY_LINES_SHOWN); the official pages often state a rule in one
        # paragraph ("You may rent out your whole flat after the MOP ... Owners of Plus and Prime flats are not
        # allowed to rent out the whole flat."), which is then the section's one line
        assert 1 <= len(shown) - 1 <= sv.POLICY_LINES_SHOWN, shown
    answer = sv.template_response({"messages": [{"role": "user", "content": QUERIES[qid]}], "rag_policy_passages": passages})
    assert answer.startswith("**Policy:**\n- **") and "HFE" in answer


def test_header_only_chunks_are_skipped_and_a_section_is_shown_once():
    header_only = {"text": "[Conditions After Buying a New Flat]\n---", "source_url": URL,
                   "section": "Conditions After Buying a New Flat"}
    shown = sv.shown_policy_passages([header_only, MOP, dict(MOP), WHOLE_FLAT], "How long is the MOP for a Plus flat?")
    titles = [s["title"] for s in shown]
    assert titles.count("Minimum occupation period (MOP)") == 1                    # the repeated chunk is one section
    assert "For new and resale Plus and Prime flats, the MOP is 10 years." in shown[titles.index(
        "Minimum occupation period (MOP)")]["lines"]
    assert sv.policy_fact_lines([header_only]) == ["- The policy search found no passage that answers this. Please check with HDB."]


def test_at_most_four_sections_are_shown():
    passages = [TABLE, RECOVERY, MOP, WHOLE_FLAT, chunk("Renting out room(s)"),
                chunk("Solemnisation of marriage", "within 3 months")]
    shown = sv.shown_policy_passages(passages, "Flat classification rules")
    assert sv.POLICY_PASSAGES_SHOWN == 4
    assert len(shown) <= 4


def test_table_rows_read_as_text_and_separators_are_dropped():
    lines = sv._policy_lines(TABLE["text"])
    assert "Minimum occupation period (MOP) — Standard: 5 years; Plus: 10 years; Prime: 10 years" in lines
    assert not any("---" in line or "|" in line for line in lines)
    assert "Renting out of whole flat — Standard: Allowed after MOP; Plus: Not allowed; Prime: Not allowed" in lines


def test_a_bare_lead_in_goes_in_front_of_the_lines_under_it():
    # a parsing rule, shown on a small passage written for the test (the official pages rarely indent under a short
    # lead-in): "Grant Amounts:" is not a line of its own but heads the lines indented under it
    text = ("[Test page > Grants]\nGrant Amounts:\n  - 2- to 4-room resale flat: $40,000\n"
            "  - 5-room resale flat: $25,000\nIncome ceiling: $8,000")
    lines = sv._policy_lines(text)
    assert "Grant Amounts — 5-room resale flat: $25,000" in lines
    assert "Grant Amounts" not in lines and "Income ceiling: $8,000" in lines


def test_the_lines_shown_answer_each_part_of_the_question():
    # "minimum age and income ceiling": a line for each, from HDB's singles eligibility table
    singles = next(c for c in CHUNKS if c["section"] == "Singles > Eligibility conditions"
                   and "Monthly household income ceiling" in c["text"])
    shown = sv.shown_policy_passages([singles], QUERIES["C2-02"])[0]
    assert any(line.startswith("Age") and "At least 35 years old" in line for line in shown["lines"])
    assert any(line.startswith("Monthly household income ceiling") and "$8,000" in line for line in shown["lines"])
    assert 2 <= len(shown["lines"]) <= sv.POLICY_LINES_SHOWN


def test_long_lines_keep_whole_sentences():
    line = "First sentence is here. " + "word " * 80
    assert sv._trim_line(line) == "First sentence is here."
    cut = sv._trim_line("x" * 10 + " longword" * 60)         # one sentence longer than any real line: cut at a word
    assert cut.endswith("…") and len(cut) <= sv.POLICY_LINE_MAX + 1 and "longwor…" not in cut


# Expected lines are HDB's own wording from the captured official pages (data/policy_sources/). The legacy corpus's
# lines ("must wait 15 months ...", "the transaction is ALWAYS PERMITTED ...") are not what HDB's current pages say.
@pytest.mark.parametrize("qid, kept", [
    # HDB removed the 15-month wait-out on 27 July 2026 (data/policy_documents/46_hdb_press_wait_out_removal_2026.md)
    ("C3-01", "previously exempted from the 15-month wait-out period to buy a 4-room or smaller"),
    # the 2026 removal page answers first, with the 30-month wait that still applies to subsidised purchases and loans
    ("C3-02", "the existing requirement to wait out a period of 30 months after the disposal"),
    ("C3-04", "You and the flat seller are of the same ethnic group and household citizenship"),
    ("C2-01", "At least 35 years old"),
    # a Plus question is shown the Plus/Prime subsidy-recovery rule, including that open-market resale buyers do not pay it
    ("C1-03", "If you buy a resale Plus or Prime flat on the open market, the subsidy recovery does not apply when you "
              "sell the flat.")])
def test_the_line_that_answers_keeps_its_figure_unit_and_condition(qid, kept):
    lines = sv.policy_fact_lines(search(qid), QUERIES[qid])
    assert any(kept in line for line in lines), lines
    assert not any(re.search(r"\d…|\w…$", line) for line in lines), lines     # no line ends mid-sentence


def test_no_line_of_the_corpus_is_cut_inside_a_sentence():
    for line in {line for c in CHUNKS for line in sv._policy_lines(c["text"])}:
        whole = set(sv._split_opening(line))
        for question in ("", "How long must I wait after selling my private property?"):
            trimmed = sv._trim_line(line, question)
            assert len(trimmed) <= sv.POLICY_LINE_MAX
            for piece in sv._split_opening(trimmed.replace(" … ", " ").removesuffix(" …")):
                bare = piece.rstrip(";,:. ")
                if any(w.rstrip(";,:. ") == bare for w in whole):
                    continue
                # a table row with a long label keeps its rule whole and shortens the label ("Label start…: rule")
                if "…: " in piece:
                    head, rule = piece.split("…: ", 1)
                    assert line.startswith(head) and rule.rstrip("…").rstrip() in line, (piece, line)
                    continue
                # a single sentence longer than POLICY_LINE_MAX (HDB/CPF run some formulas and notes together) is cut
                # by design, at a comma or a word: the piece is then the start of that sentence, marked "…"
                assert piece.endswith("…") and any(w.startswith(piece.rstrip("…").rstrip()) for w in whole), (piece, line)


def test_a_later_sentence_that_answers_the_question_is_kept_after_a_gap():
    line = ("Single Parent — Min Age: 21; " + "; ".join(f"Rule {n}: something about the scheme here" for n in range(8))
            + "; Wait-Out After Selling Private Property: 30 months (BTO) / 15 months (Resale)")
    trimmed = sv._trim_line(line, "How long must I wait after selling my private property?")
    assert trimmed.startswith("Single Parent — Min Age: 21;") and " … Wait-Out After Selling Private Property: 30 months " \
        "(BTO) / 15 months (Resale)" in trimmed and len(trimmed) <= sv.POLICY_LINE_MAX
    assert "Wait-Out" not in sv._trim_line(line) and sv._trim_line(line).endswith(" …")


def test_a_sentence_too_long_to_show_is_cut_at_a_comma_or_before_a_number():
    assert sv._cut_sentence("a " * (sv.POLICY_LINE_MAX // 2 - 10) + "must wait 15 months from the date").endswith("must wait…")
    assert sv._cut_sentence("word " * 60 + "thing, " + "word " * 40).endswith("thing…")


# --- the opening ----------------------------------------------------------------------------------------------------
PLUS = [TABLE, RECOVERY, MOP, WHOLE_FLAT]          # HDB's pages on Plus/Prime flats, as the search returns them

# The range rules ("3% to 6%" may be written "between 3% and 6%", but the ends of two ranges are not a new range) need a
# passage that states ranges. HDB's pages give no fixed Plus/Prime rate, so this passage is written for the test; its
# figures are test data, not policy.
RANGES = [{"text": "[Test page > Subsidy recovery]\nPlus flats: a subsidy recovery of 3% to 6% on the first resale.\n"
                   "Prime flats: a subsidy recovery of 6% to 9% on the first resale.",
           "section": "Test page > Subsidy recovery", "source_url": URL}]


@pytest.mark.parametrize("sentence", [
    "Plus flats have a 10-year MOP.",
    "Plus and Prime flats have a 10 years MOP.",
    "Neither Prime nor Plus flats can be rented out whole."])
def test_an_opening_may_state_a_figure_the_policy_lines_give(sentence):
    assert sv.keep_supported_sentences(sentence, "", QUERIES["C1-01"], PLUS) == (sentence, 0)


def test_a_figure_from_a_section_on_a_class_not_asked_about_is_dropped():
    # C1-01 compares Prime and Plus: the table's Standard cells are not shown (SEM-07), so their figures are not allowed;
    # for a question that asks about Standard flats too they are
    sentence = "Standard flats have a 5 years MOP."
    assert sv.keep_supported_sentences(sentence, "", QUERIES["C1-01"], PLUS) == ("", 1)
    assert sv.keep_supported_sentences(sentence, "", "What is the MOP for Standard, Plus and Prime flats?", PLUS) == (sentence, 0)


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


@pytest.mark.parametrize("sentence", [
    "Plus flats carry a clawback between 3% and 9%.",   # the ends of two ranges, joined by "between ... and"
    "The Plus clawback ranges from 3% up to 9%.",
    "Plus flat clawback is 3%/9%.",
    "The Plus clawback is 3% to 9%.",
    "Plus flats carry a 3% clawback.",                  # one end of "3% to 6%" is not a figure on its own
    "Prime flats carry a 9% clawback."])
def test_a_range_is_checked_as_a_range_however_it_is_written(sentence):
    assert sv.keep_supported_sentences(sentence, "", QUERIES["C1-03"], RANGES) == ("", 1)


@pytest.mark.parametrize("sentence", [
    "Plus flats carry a clawback of between 3% and 6% on the first resale.",
    # (no longer "..., and Prime from 6% to 9%": C1-03 asks about Plus flats, so the Prime section is not shown)
    "The Plus clawback ranges from 3% up to 6%."])
def test_a_range_the_passages_give_may_be_written_another_way(sentence):
    assert sv.keep_supported_sentences(sentence, "", QUERIES["C1-03"], RANGES) == (sentence, 0)


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
            # (no Standard-flat clause: C1-01 asks about Prime and Plus, and a sentence on Standard flats is now dropped)
            return AIMessage(content="Plus flats have a 10-year MOP, while Prime flats also have a 10-year MOP. "
                                     "Plus flats resell for about $900k.")

    with patch.object(sv, "get_llm", return_value=Writer()):
        out = sv.synthesize_node(state)
    facts = sv.template_response(state)
    assert out["draft_response"] == ("Plus flats have a 10-year MOP, while Prime flats also have a 10-year MOP.\n\n" + facts)
    assert out["errors"] == ["synthesize: dropped 1 opening sentence(s) that stated figures"]
    assert "exactly as the Policy lines above write it" in seen["prompt"]
    with patch.object(sv, "get_llm", return_value=Writer()):
        mixed = sv.synthesize_node({**state, "financial_breakdown": {"calculate_cpf_housing_grants": {"status": "ok",
                                    "enhanced_housing_grant": 0, "family_grant": 80000, "proximity_housing_grant": 0,
                                    "total_grants": 80000}}})
    assert "10-year MOP, while" not in mixed["draft_response"]           # grants shown too: the opening states no figure
    assert "write no numbers at all except block names" in seen["prompt"]
