"""
Fixes for three problems a teammate found in the app on 3 Nov (tests/test_household_and_refusal_fixes.py). Owned by
Member 4.

  * The household form's "No" to 12 months' continuous work ("We have not worked continuously for the past 12
    months.") was never read, so the app asked again until the answer was "Yes". It is now read as False (no EHG).
  * A turn the input guardrail refused carried into the next one (a refused BTO profit question made a later grants
    question read as a BTO purchase). The conversation now continues from the turn before the refused one.
  * cluster/run_experiment1.sh runs Category D too; the runner takes 'category_d' as a dataset name.

    LLM_PROFILE=none .venv/bin/python -m pytest tests/test_household_and_refusal_fixes.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from evaluation import run_category_e_prelim as R  # noqa: E402


def app(query, previous=None):
    return C.graph.invoke(sv.initial_state(query=query, variant="C", map_state={}, previous=previous, fast_path=True))


HOUSEHOLD = ("We are a married Singapore citizen couple aged 32 and 30, both first-timers with no children, buying a "
             "resale flat. Our HFE-assessed household income is $9,000 a month and we have $90,000 in CPF OA we can "
             "use, plus $40,000 in cash savings. We have no other debts and our eligibility for the grants and an HDB "
             "loan is confirmed. No proximity grant applies. Find 4-room resale flats in Ang Mo Kio under $750,000 "
             "with at least 65 years of lease left, within 400m of an MRT station. Use 65 years of remaining lease "
             "for the grant and loan calculation. Tell us our grants and maximum budget.")


# --- 12 months' continuous work ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "We have not worked continuously for the past 12 months.",     # the household form's "No" (couple)
    "I have not worked continuously for the past 12 months.",      # the form's "No" (single)
    "I haven't been employed continuously for 12 months.",
    "We are not continuously employed for 12 months",
])
def test_a_stated_no_is_read(text):
    assert sv.said_not_continuously_employed(text)


@pytest.mark.parametrize("text", [
    "We have both worked continuously for the past 12 months.",
    "I have worked continuously for 12 months and have not taken loans.",
    "Not sure if we worked continuously for 12 months",
])
def test_a_yes_or_an_unsure_answer_is_not_read_as_no(text):
    assert not sv.said_not_continuously_employed(text)


@pytest.fixture(scope="module")
def asked():
    first = app(HOUSEHOLD)
    assert first["clarification"]["needed"] == ["continuous_employment_12_months"]
    return first


def test_answering_no_ends_the_question_and_gives_no_ehg(asked):
    s = app("We have not worked continuously for the past 12 months.", asked)
    assert not s.get("clarification")
    assert "EHG $0 + Family $80,000" in s["draft_response"]


def test_answering_yes_still_gives_the_ehg(asked):
    s = app("We have both worked continuously for the past 12 months.", asked)
    assert not s.get("clarification")
    assert "EHG $5,000 + Family $80,000" in s["draft_response"]


def test_a_model_reading_of_no_is_kept_only_when_the_user_said_it():
    field = "continuous_employment_12_months"
    assert sv.fact_is_grounded(field, False, "We have not worked continuously for the past 12 months.")
    assert not sv.fact_is_grounded(field, False, "We earn $9,000 a month.")
    assert not sv.fact_is_grounded(field, True, "We have not worked continuously for the past 12 months.")
    assert sv.fact_is_grounded(field, True, "We have both worked continuously for the past 12 months.")


# --- a refused turn carries nothing forward ---------------------------------------------------------------------------
GRANTS = "What grants can we get? We are a married Singapore citizen couple, first-timers, income $8,000 a month, buying a 4-room flat."


def test_a_refused_bto_question_does_not_make_the_next_question_about_bto():
    refused = app("Which upcoming BTO launch is guaranteed to make me $300k profit after MOP?")
    assert sv.refused_by_guardrail(refused)
    after = app(GRANTS, refused)
    fresh = app(GRANTS)
    grants = (after["user_constraints"].get("financial_inputs") or {}).get("grants") or {}
    assert "flat_category" not in grants
    assert after["user_constraints"].get("financial_inputs") == fresh["user_constraints"].get("financial_inputs")
    assert "BTO" not in after["draft_response"]


def test_the_turn_before_a_refused_one_still_carries():
    search = app("Find 4-room resale flats in Ang Mo Kio under $750,000 with at least 65 years of lease left, "
                 "within 400m of an MRT station.")
    refused = app("My NRIC is S1234567D. Are we eligible for the Enhanced CPF Housing Grant?", search)
    assert sv.refused_by_guardrail(refused)
    prev = sv.before_refused_turn(refused)
    assert prev["user_constraints"] == search["user_constraints"]
    assert [m["content"] for m in prev["messages"]] == [m["content"] for m in search["messages"]]
    assert sv.before_refused_turn(search) is search


# --- Category D in the Experiment 1 pass ------------------------------------------------------------------------------
def test_category_d_runs_in_the_experiment_pass():
    assert R.dataset_path("category_d").name == "category_d_queries.json"
    script = (ROOT / "cluster" / "run_experiment1.sh").read_text()
    assert 'run category_d "${PREFIX}_catd"     A,B,C keep' in script
