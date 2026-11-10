"""Category A (historical resale statistics) answers scored against their ground-truth groups (Member 4).

Synthetic answers in the formats a model writes (tables, bullets, prose, "$515k", "S$515,000"): the score needs every
asked figure of every group beside that group's label, and no group figure that disagrees with the truth."""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROFILE"] = "none"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import architecture_metrics as M  # noqa: E402

ITEMS = {i["id"]: i for i in json.loads(
    (Path(__file__).resolve().parent.parent / "evaluation" / "category_a_queries.json").read_text())}
TRUTH_SNAPSHOT = ITEMS["A1-01"]["ground_truth_facts"]["snapshot_id"]
STALE_SNAPSHOT = "00000000000000000000"   # any id other than the truth's: the test must not depend on which snapshot the truth was built on

A1_01_TABLE = """**Resale statistics in Singapore, 2017-01 to 2026-09**
Flats within 200 m of an MRT station sold at a median of $515,000, 5.5% above flats more than 1,000 m away ($488,000).
| Distance | Sales | Median price | Middle 50% | Per sqm | Last 12 mths vs year before |
|---|---|---|---|---|---|
| 0-200 m | 10,693 | $515,000 | $407,000 - $652,800 | $5,435 | +0.0% |
| 200-500 m | 65,309 | $525,000 | $410,000 - $670,888 | $5,403 | +1.9% |
| 500-1,000 m | 103,487 | $492,000 | $388,000 - $628,000 | $5,179 | +0.0% |
| >1,000 m | 39,543 | $488,000 | $395,000 - $600,000 | $5,192 | +0.3% |
| _all flat sales island-wide, any distance_ | 241,701 | $505,000 |  | $5,323 |  |
**Median price by year** (first and last year may be part-years):
- 0-200 m: 2017 $440,800 → 2018 $430,000 → 2026 $630,000
_Period: 2017-01 to 2026-09 (117 months)._"""


def _state(draft, snapshot=TRUTH_SNAPSHOT):
    smr = {"query_resale_statistics": {"status": "ok", "snapshot_id": snapshot}} if snapshot else {}
    return {"variant": "C", "draft_response": draft, "sql_market_results": smr, "tool_call_log": []}


def _score(qid, draft, **kw):
    return M.ground_truth_scores(ITEMS[qid], _state(draft, **kw))


def test_correct_table_answer_is_fully_right():
    gt = _score("A1-01", A1_01_TABLE)
    assert gt["kind"] == "statistics" and gt["measures_asked"] == ["median_price", "count"]   # "historical statistics"
    assert gt["success"] and gt["figures_correct"] == gt["figures_total"] == 8 and gt["figure_accuracy"] == 1.0
    assert gt["figures_wrong"] == 0 and gt["groups_missing"] == []
    assert all(g["median_psm"] == "right" for g in gt["per_group"])   # stated, though not asked, and right


def test_one_wrong_figure_fails_and_shows_in_the_accuracy():
    gt = _score("A1-01", A1_01_TABLE.replace("| 0-200 m | 10,693 | $515,000", "| 0-200 m | 10,693 | $520,000")
                .replace("a median of $515,000", "a median of $520,000"))
    assert not gt["success"] and gt["figures_wrong"] == 1
    assert gt["figures_correct"] == 7 and gt["figure_accuracy"] == pytest.approx(7 / 8)
    assert gt["per_group"][0]["median_price"] == "wrong" and gt["per_group"][0]["count"] == "right"


def test_swapped_groups_fail_even_though_every_figure_appears():
    swapped = A1_01_TABLE.replace("| 0-200 m | 10,693 | $515,000", "| 0-200 m | 65,309 | $525,000") \
                         .replace("| 200-500 m | 65,309 | $525,000", "| 200-500 m | 10,693 | $515,000") \
                         .replace("a median of $515,000", "a median of $525,000")
    gt = _score("A1-01", swapped)
    assert not gt["success"] and gt["figures_wrong"] >= 4
    assert [g["median_price"] for g in gt["per_group"][:2]] == ["wrong", "wrong"]


def test_a_missing_group_is_missing_not_wrong():
    lines = [l for l in A1_01_TABLE.splitlines() if not l.startswith("| >1,000 m")]
    gt = _score("A1-01", "\n".join(lines).replace(", 5.5% above flats more than 1,000 m away ($488,000)", ""))
    assert not gt["success"] and gt["figures_wrong"] == 0
    assert gt["groups_missing"] == [">1,000 m"] and gt["figure_accuracy"] == pytest.approx(6 / 8)


def test_thousands_rounding_and_currency_prefixes_are_read():
    bullets = ("Here is what flats sold for over the last ten years:\n"
               "- **0-200m from an MRT**: median S$515k across 10,693 sales\n"
               "- **200 to 500 metres**: median SGD 525,000 (65,309 transactions)\n"
               "- **500 m to 1 km**: $492K median, 103,487 sales\n"
               "- **Over 1 km**: median $488k from 39,543 sales")
    gt = _score("A1-01", bullets)
    assert gt["success"], gt["per_group"]
    # $515k is right to the nearest thousand only: $516k is a wrong figure, not a rounding
    gt = _score("A1-01", bullets.replace("S$515k", "S$516k"))
    assert not gt["success"] and gt["per_group"][0]["median_price"] == "wrong"


def test_prose_figures_go_to_the_group_named_in_their_clause():
    right = ("Flats within 300 m of an MRT station sold at a median of S$650,000, while those more than 300 m away "
             "went for $625,000.")
    assert _score("A1-04", right)["success"]
    wrong = ("Flats within 300 m of an MRT station sold at a median of $625,000, while those more than 300 m away "
             "went for $650,000.")
    gt = _score("A1-04", wrong)
    assert not gt["success"] and gt["figures_wrong"] == 2
    # figure before the label: "$500,000 for 2021 and $630,000 for 2026"
    item = copy.deepcopy(ITEMS["A3-03"])
    item["ground_truth_facts"]["groups"] = [g for g in item["ground_truth_facts"]["groups"] if g["label"] in ("2021", "2026")]
    gt = M.ground_truth_scores(item, _state("The median was $500,000 for 2021 and $630,000 for 2026 (+26.0%)."))
    assert gt["success"], gt["per_group"]


def test_one_paragraph_naming_every_group():
    para = ("The 0-200 m band ($515,000; 10,693 sales) is $27,000 higher than the >1,000 m band ($488,000; 39,543 "
            "sales). Between 200 and 500 m the median was 525,000 from 65,309 sales, and between 500 m and 1 km it "
            "was $492,000 (n = 103,487).")
    gt = _score("A1-01", para)
    assert gt["success"], gt["per_group"]
    gt = _score("A1-01", para.replace("39,543", "39,000"))
    assert not gt["success"] and gt["per_group"][3]["count"] == "wrong" and gt["figures_wrong"] == 1


def test_storey_and_lease_bands_in_other_wordings():
    storeys = ("| Storey range | Median | Transactions |\n|---|---|---|\n| 01 TO 05 | $668,000 | 968 |\n"
               "| 06 TO 10 | $735,000 | 856 |\n| 11 TO 15 | $802,444 | 150 |\n| 16 TO 20 | $859,444 | 32 |")
    assert _score("A3-02", storeys)["success"]
    lease = ("| Lease left | Median price |\n|---|---|\n| <60 years | $508,888 |\n| 60-70 | $720,000 |\n"
             "| 70-80 | $750,000 |\n| 80-90 | $795,000 |\n| 90+ | $804,888 |")
    assert _score("A3-05", lease)["success"]
    gt = _score("A3-05", lease.replace("| 70-80 | $750,000 |", "| 70-80 | $795,000 |"))
    assert not gt["success"] and gt["per_group"][2]["median_price"] == "wrong"


def test_station_names_millions_and_vague_figures():
    answer = ("| Station | Median price |\n|---|---|\n| Outram Park | $1.45M |\n| Havelock | $1.20M |\n"
              "| Beauty World | $1,134,000 |")
    assert _score("A2-01", answer)["success"]
    # "$1.2M" has two significant figures: too vague to count either way, so Havelock is missing, not wrong
    gt = _score("A2-01", answer.replace("$1.20M", "$1.2M"))
    assert not gt["success"] and gt["figures_wrong"] == 0 and gt["per_group"][1]["median_price"] == "missing"


def test_per_sqm_is_needed_when_the_question_asks_per_square_metre():
    rows = [("2 ROOM", 372000, 8085.1), ("4 ROOM", 630000, 6611.1), ("3 ROOM", 440000, 6388.1),
            ("EXECUTIVE", 915000, 6301.4), ("5 ROOM", 740000, 6250.0)]
    without = "\n".join(f"- {t.title()}: median ${p:,.0f}" for t, p, _ in rows)
    gt = _score("A3-04", without)
    assert gt["measures_asked"] == ["median_price", "median_psm"]
    assert not gt["success"] and gt["figure_accuracy"] == 0.5 and gt["figures_wrong"] == 0
    with_psm = "\n".join(f"- {t.replace(' ROOM', '-room')}: ${psm:,.0f} per sqm (median ${p:,.0f})" for t, p, psm in rows)
    assert _score("A3-04", with_psm)["success"]


def test_headings_transposed_tables_and_grant_lines():
    headed = "\n".join(f"### {lab}\n\n- Median price: ${p:,}\n- Sales: {n:,}" for lab, p, n in
                       [("0-200 m", 515000, 10693), ("200-500 m", 525000, 65309), ("500-1,000 m", 492000, 103487),
                        (">1,000 m", 488000, 39543)])
    assert _score("A1-01", headed)["success"]
    transposed = ("| | 0-200 m | 200-500 m | 500-1,000 m | >1,000 m |\n|---|---|---|---|---|\n"
                  "| Median price | $515,000 | $525,000 | $492,000 | $488,000 |\n"
                  "| Sales | 10,693 | 65,309 | 103,487 | 39,543 |")
    assert _score("A1-01", transposed)["success"]
    # money the buyer has or borrows is not a statistic of a group, even beside a group's name
    grants = A1_01_TABLE + "\n- Flats within 200 m still qualify for the CPF housing grant of up to $120,000."
    assert _score("A1-01", grants)["success"]


def test_snapshot_is_recorded_so_a_stale_truth_is_visible(monkeypatch):
    gt = _score("A1-01", A1_01_TABLE)
    assert gt["snapshot_matches"] is True and gt["snapshot_source"] == "tool"
    gt = _score("A1-01", A1_01_TABLE, snapshot=STALE_SNAPSHOT)
    assert gt["snapshot_matches"] is False and gt["success"]       # figures are still compared, the mismatch shown
    monkeypatch.setattr(M, "database_snapshot", lambda: None)
    gt = _score("A1-01", A1_01_TABLE, snapshot=None)
    assert gt["snapshot_matches"] is None and gt["snapshot_source"] is None


def test_task_success_and_aggregate_use_the_figures():
    item = ITEMS["A1-01"]
    right = M.score_item(item, _state(A1_01_TABLE))
    wrong = M.score_item(item, dict(_state(A1_01_TABLE.replace("$525,000", "$530,000")), variant="A"))
    assert right["category"] == "A" and right["task_success"] and not wrong["task_success"]
    agg = M.aggregate([right, wrong])
    assert agg["C"]["stat_figure_accuracy"] == 1.0 and agg["A"]["stat_figure_accuracy"] == pytest.approx(7 / 8)
    assert agg["C"]["stat_snapshot_matches"] == 1.0


def test_report_and_run_comparison_show_figures_and_a_stale_snapshot():
    from evaluation import analyze_category_e as AN, compare_runs as CR
    item = ITEMS["A1-01"]
    rows = []
    for v, draft in (("A", A1_01_TABLE.replace("$525,000", "$530,000")), ("C", A1_01_TABLE)):
        st = dict(_state(draft, snapshot=STALE_SNAPSHOT), variant=v)
        rows.append({"variant": v, "id": item["id"], "sub_use_case": item["sub_use_case"], "status": "ok",
                     "scores": M.score_item(item, st)})
    table = AN.statistics_table(rows)
    assert "| A | 1 | 0/1" in table and "| 7/8 (88%) | 1 | 0 |" in table and table.rstrip().endswith("| 0/1 |")
    assert "| C | 1 | 1/1" in table and "| 8/8 (100%) | 0 | 0 |" in table
    caveat = AN.snapshot_caveat(rows)
    assert "be71fe16c4e7e7548eb7" in caveat and TRUTH_SNAPSHOT in caveat and "2 of 2" in caveat
    assert CR.counts(rows, "C")["fully right"] == (1, 1) and CR.counts(rows, "C")["A answer on the truth's snapshot"] == (0, 1)


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_every_variant_answer_is_read_from_its_own_format(variant, monkeypatch):
    """Each variant's real no-model answer (add-ons offered to all, as Category A runs) scores fully right against the
    figures its own statistics tool returned, so the reading works on the formats the variants actually write
    whatever database snapshot this machine has."""
    if M.database_snapshot() is None:
        pytest.skip("no market database on this machine")
    monkeypatch.setenv("HDB_ADDONS", "all")
    from evaluation.run_category_e_prelim import load_variant
    mod = load_variant(variant)
    for qid in ("A1-01", "A2-01", "A3-02", "A3-03", "A3-04", "A3-05"):
        state = mod.run(ITEMS[qid]["query"], None, question_id=qid)
        res = (state.get("sql_market_results") or {}).get("query_resale_statistics") or {}
        got = {g["label"]: g for g in res.get("groups") or []}
        item = copy.deepcopy(ITEMS[qid])
        groups = [dict(g, **{k: got[g["label"]][k] for k in M.STAT_MEASURES}) for g in item["ground_truth_facts"]["groups"]
                  if g["label"] in got]
        assert groups, f"{variant} {qid}: the statistics tool returned none of the truth's groups"
        item["ground_truth_facts"] = dict(item["ground_truth_facts"], groups=groups)
        gt = M.ground_truth_scores(item, state)
        assert gt["success"], (variant, qid, gt["per_group"])
