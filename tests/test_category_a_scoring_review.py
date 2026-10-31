"""Category A scoring after review (Member 4): the answer phrasings the first scorer read wrongly.

A figure in a comparison ("(vs $650,000 for >300 m)", "(up from $525,000 in 2021)", ", compared with $650,000 for
flats further out") is not the line's own group's figure, so a swapped answer no longer passes; a median column whose
header names the period is read; "most affordable" is no grant line; and the usual storey and lease band wordings are
labels."""
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


def _score(qid, draft):
    state = {"variant": "C", "draft_response": draft, "sql_market_results": {}, "tool_call_log": []}
    return M.ground_truth_scores(ITEMS[qid], state)


# --- swapped groups hidden behind a comparison with the other group ------------------------------------------------

def test_swapped_groups_with_a_vs_aside_naming_the_other_group_fail():
    swapped = "- 0-300 m: $625,000 (vs $650,000 for >300 m)\n- >300 m: $650,000 (vs $625,000 for 0-300 m)"
    gt = _score("A1-04", swapped)
    assert not gt["success"] and gt["figures_wrong"] == 2 and gt["figure_accuracy"] == 0.0
    right = "- 0-300 m: $650,000 (vs $625,000 for >300 m)\n- >300 m: $625,000 (vs $650,000 for 0-300 m)"
    gt = _score("A1-04", right)
    assert gt["success"] and gt["figures_wrong"] == 0


def test_a_trend_off_by_one_year_with_up_from_asides_fails():
    off_by_one = ("- 2021: $500,000\n- 2022: $550,000 (up from $525,000 in 2021)\n"
                  "- 2023: $590,000 (up from $550,000 in 2022)\n- 2024: $628,000 (up from $590,000 in 2023)\n"
                  "- 2025: $630,000 (up from $628,000 in 2024)\n- 2026: $630,000 (flat vs 2025)")
    gt = _score("A3-03", off_by_one)
    assert not gt["success"] and gt["figures_wrong"] == 4
    assert [g["median_price"] for g in gt["per_group"]] == ["right", "wrong", "wrong", "wrong", "wrong", "right"]
    right = ("- 2021: $500,000\n- 2022: $525,000 (up from $500,000 in 2021)\n"
             "- 2023: $550,000 (up from $525,000 in 2022)\n- 2024: $590,000 (up from $550,000 in 2023)\n"
             "- 2025: $628,000 (up from $590,000 in 2024)\n- 2026: $630,000 (flat vs 2025)")
    assert _score("A3-03", right)["success"]


def test_a_compared_with_clause_about_an_unnamed_group_is_not_the_lines_figure():
    swapped = ("- Within 300 m of an MRT: $625,000, compared with $650,000 for flats further out\n"
               "- More than 300 m away: $650,000, compared with $625,000 nearer the station")
    gt = _score("A1-04", swapped)
    assert not gt["success"] and gt["figures_wrong"] == 2
    right = ("- Within 300 m of an MRT: $650,000, compared with $625,000 for flats further out\n"
             "- More than 300 m away: $625,000, compared with $650,000 nearer the station")
    assert _score("A1-04", right)["success"]


def test_a_right_figure_in_another_groups_aside_does_not_rescue_a_wrong_one():
    gt = _score("A2-03", "- Jurong East: $514,000 (vs $545,000 in Bedok)\n- Yishun: $517,944\n- Bedok: $514,000")
    assert not gt["success"] and gt["per_group"][2]["median_price"] == "wrong"
    # with no figure of its own, a group's figure in another line's comparison still counts
    assert _score("A2-03", "- Jurong East: $514,000 (vs $545,000 in Bedok)\n- Yishun: $517,944")["success"]


def test_a_figure_written_before_its_label_in_an_arrow_chain():
    chain = ("Median resale price by year: $500,000 (2021) → $525,000 (2022) → $550,000 (2023) → $590,000 (2024) → "
             "$628,000 (2025) → $630,000 (2026)")
    gt = _score("A3-03", chain)
    assert gt["success"] and gt["figures_wrong"] == 0, gt["per_group"]
    assert _score("A3-03", "Median resale price by year: 2021 $500,000 → 2022 $525,000 → 2023 $550,000 → "
                           "2024 $590,000 → 2025 $628,000 → 2026 $630,000")["success"]
    assert _score("A3-03", "Prices rose from $500,000 in 2021 to $525,000 in 2022, $550,000 in 2023, $590,000 in "
                           "2024, $628,000 in 2025 and $630,000 in 2026.")["success"]
    # a label that opens its own figure keeps it: "(2026: $630,000)" is not the 2025 line's figure for 2026
    assert _score("A3-03", "- 2021: $500,000\n- 2022: $525,000\n- 2023: $550,000\n- 2024: $590,000\n"
                           "- 2025: $628,000 (2026: $630,000)")["success"]


def test_more_than_a_number_is_no_comparison():
    bullets = ("- 0-200 m: more than 10,693 sales at $515,000\n- 200-500 m: 65,309 sales at $525,000\n"
               "- 500-1,000 m: 103,487 sales at $492,000\n- >1,000 m: 39,543 sales at $488,000")
    assert _score("A1-01", bullets)["success"]


# --- median and sales columns whose header names the period --------------------------------------------------------

@pytest.mark.parametrize("header", ["Median price (Oct 2025 - Sep 2026)", "Median price (past year)",
                                    "Median price (since 2025)"])
def test_a_median_column_naming_the_period_is_read(header):
    table = (f"| Distance | {header} | Sales |\n|---|---|---|\n| 0-300 m | $650,000 | 2,658 |\n"
             "| >300 m | $625,000 | 19,526 |")
    gt = _score("A1-04", table)
    assert gt["success"] and [g["median_price"] for g in gt["per_group"]] == ["right", "right"]


def test_period_and_flats_sold_headers_are_read():
    gt = _score("A2-02", "| Town | Median 4-room price (last 5 years) | Sales |\n|---|---|---|\n"
                         "| Bukit Merah | $900,000 | 446 |\n| Toa Payoh | $889,444 | 580 |\n| Queenstown | $859,000 | 212 |")
    assert gt["success"] and gt["figure_accuracy"] == 1.0
    gt = _score("A3-01", "| Town | Median price | Resales (2025-26) |\n|---|---|---|\n"
                         "| Bukit Timah | $1,010,000 | 58 |\n| Queenstown | $850,000 | 751 |\n| Bishan | $848,000 | 410 |")
    assert gt["success"] and all(g["count"] == "right" for g in gt["per_group"])
    gt = _score("A1-01", "| Distance | Median price | Flats sold |\n|---|---|---|\n| 0-200 m | $515,000 | 10,693 |\n"
                         "| 200-500 m | $525,000 | 65,309 |\n| 500-1,000 m | $492,000 | 103,487 |\n"
                         "| >1,000 m | $488,000 | 39,543 |")
    assert gt["success"] and gt["figure_accuracy"] == 1.0


def test_one_year_and_quartile_columns_are_still_not_the_median():
    # "Median 2021" and "(first year)" are one part of the period; "25th" / "Q1" are quartiles
    for extra in ("Median 2021", "Median (first year)", "25th percentile", "Q1"):
        table = (f"| Distance | {extra} | Median price | Sales |\n|---|---|---|---|\n"
                 "| 0-300 m | $500,000 | $650,000 | 2,658 |\n| >300 m | $480,000 | $625,000 | 19,526 |")
        gt = _score("A1-04", table)
        assert gt["success"] and gt["figures_wrong"] == 0, extra


# --- "affordable" is a statistics word, not a grant line -----------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "Jurong East is the most affordable, with a median of $514,000.\nYishun follows at $517,944.\n"
    "Bedok is third at $545,000.",
    "- **Jurong East** – $514,000 (most affordable)\n- **Yishun** – $517,944\n- **Bedok** – $545,000",
    "| Town | Median price | Note |\n|---|---|---|\n| Jurong East | $514,000 | most affordable |\n"
    "| Yishun | $517,944 | |\n| Bedok | $545,000 | |",
])
def test_most_affordable_town_keeps_its_median(answer):
    gt = _score("A2-03", answer)
    assert gt["success"] and gt["per_group"][0]["median_price"] == "right"


def test_an_income_clause_drops_only_its_own_figures():
    answer = ("- Jurong East: $514,000, the cheapest; buyers under the $14,000 income ceiling qualify\n"
              "- Yishun: $517,944\n- Bedok: $545,000")
    assert _score("A2-03", answer)["success"]
    # a grant line beside a group is still not read
    grants = ("### 4-room grants\n- Enhanced CPF Housing Grant: up to $120,000\nBukit Merah: $900,000\n"
              "Toa Payoh: $889,444\nQueenstown: $859,000")
    assert _score("A2-02", grants)["success"]


# --- storey and lease band wordings --------------------------------------------------------------------------------

@pytest.mark.parametrize("top", ["90 years and above", "90 years or more"])
def test_lease_band_90_years_and_above(top):
    answer = f"- lease <60 yrs: ~$509k\n- 60-70 yrs: $720k\n- 70-80 yrs: $750k\n- 80-90 yrs: $795k\n- {top}: $805k"
    gt = _score("A3-05", answer)
    assert gt["success"] and gt["per_group"][4]["median_price"] == "right"


def test_storey_bands_as_low_mid_high_floors_and_ordinals():
    floors = ("- Low floors (1-5): $668,000\n- Mid floors (6-10): $735,000\n- High floors (11-15): $802,444\n"
              "- Very high (16-20): $859,444")
    gt = _score("A3-02", floors)
    assert gt["success"] and gt["figure_accuracy"] == 1.0
    ordinals = ("| Floors | Median price |\n|---|---|\n| 1st-5th | $668,000 |\n| 6th-10th | $735,000 |\n"
                "| 11th-15th | $802,444 |\n| 16th-20th | $859,444 |")
    assert _score("A3-02", ordinals)["success"]
    gt = _score("A3-02", floors.replace("(6-10): $735,000", "(6-10): $802,444"))
    assert not gt["success"] and gt["per_group"][1]["median_price"] == "wrong"
