"""Category D (financial) benchmark questions (evaluation/build_category_d.py): schema, ground truth from the real
calculators, the app's no-model answers scored, and the 90-question benchmark totals. Tools: Member 1; scoring: Member 5."""
import json
from pathlib import Path

import pytest

from agents import arch_3_specialist_5agent as arch3
from agents import supervisor as sv
from evaluation import architecture_metrics as M
from evaluation import build_benchmark_90 as B
from evaluation import build_category_d as D

EVAL = Path(__file__).resolve().parent.parent / "evaluation"
ITEMS = json.loads((EVAL / "category_d_queries.json").read_text(encoding="utf-8"))
BY_ID = {i["id"]: i for i in ITEMS}


def test_fifteen_questions_with_the_benchmark_schema():
    assert len(ITEMS) == 15 and len(BY_ID) == 15
    assert {i["sub_use_case"] for i in ITEMS} == {"D1", "D2", "D3"}
    for i in ITEMS:
        assert i["category"] == "D" and i["id"].startswith(i["sub_use_case"] + "-")
        assert not B.problems(i), (i["id"], B.problems(i))
        assert i["expected_agents"] == ["financial"]
        assert i["expected_mcp_calls"] and all(c["agent"] == "financial" for c in i["expected_mcp_calls"])
        assert i["ground_truth_facts"]["target_tool"] in D.TOOLS


def test_spread_covers_answers_figures_missing_facts_and_declines():
    statuses = {i["ground_truth_facts"]["expected_status"] for i in ITEMS}
    assert statuses == {"ok", "needs_information", "unsupported"}
    for i in ITEMS:
        gt = i["ground_truth_facts"]
        assert gt["give_figure"] == (gt["expected_status"] == "ok")


def test_ground_truth_is_reproducible_from_the_tools():
    assert json.loads(json.dumps(D.build())) == ITEMS


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i["id"])
def test_the_extractor_reads_back_every_stated_fact(item):
    assert D.check_extraction(item["financial_inputs"], item["query"]) == []


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i["id"])
def test_expected_calls_are_the_tools_the_supervisor_runs(item):
    read = sv.stated_financial_inputs(item["query"])
    for call in item["expected_mcp_calls"]:
        section = D.SECTION[call["tool"]]
        assert section in read
        assert all(read[section].get(k) == v for k, v in call["required_params"].items()
                   if k not in ("flat_category", "flat_type"))


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i["id"])
def test_the_apps_no_model_answer_scores_right(item, monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "none")
    state = arch3.run(item["query"])
    scores = M.score_item(item, state)
    assert scores["routing"]["exact"] is True, scores["routing"]
    assert scores["tool_selection"]["f1"] == 1.0, scores["tool_selection"]
    gt = scores["ground_truth"]
    assert gt["success"] is True, gt


def test_scorer_rejects_a_wrong_figure_and_an_invented_one():
    ok = BY_ID["D2-01"]
    assert M.financial_scores(ok, {"draft_response": "Max purchase price $700,000 (loan $500,000)"})["success"] is False
    assert M.financial_scores(ok, {"draft_response": "Max purchase price $774,366 (loan $569,366)"})["success"] is True
    missing = BY_ID["D2-05"]
    assert M.financial_scores(missing, {"draft_response": "Max purchase price $800,000. Tell me your cash savings."})["success"] is False
    declined = BY_ID["D1-06"]
    assert M.financial_scores(declined, {"draft_response": "Grants: EHG $5,000 + Family $80,000 = **$85,000**"})["success"] is False


def test_benchmark_has_ninety_questions_fifteen_per_category():
    built = B.build()
    assert all(r["kept"] == 15 and not r["issues"] for r in built["report"].values())
    assert len(built["items"]) == 90
    assert len({i["id"] for i in built["items"]}) == 90
    assert json.loads((EVAL / "benchmark_hdb_90.json").read_text(encoding="utf-8")) == built["items"]
    assert [i["category"] for i in built["items"]][::15] == list("ABCDEF")


def test_the_input_guardrail_lets_every_financial_question_through():
    from agents import guardrails
    assert [i["id"] for i in ITEMS if guardrails.classify_input(i["query"])[0] != "ok"] == []
    from evaluation import run_category_f
    assert not any(x.startswith("D") for x in run_category_f.false_blocks())
