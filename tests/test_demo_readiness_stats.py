"""Demo-readiness fixes to analytics follow-ups (Member 4, 4 Nov): a short follow-up to a grouped-statistics answer keeps
that analysis and changes only what it states (flat type, town, order, minimum group size, period); price statistics
around an address list no unasked town block shortlist; "Use a 1km area instead." widens the same area question; a
message pointing back at an earlier turn is read with it. Each test uses the tester's wording
(data/local/qa/gpt_issues.json, gpt_records.json)."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402

LOWEST_4RM = "Which towns have the lowest median price for a 4-room resale over the last year?"
PSM_3RM = "Compare median price per sqm across towns for 3-room resale."
AREA_500 = "Show the price distribution for 4-room within 500m of Blk 510 Bedok North St 3."


def conversation(*turns):
    """Each turn's final state, every turn carrying the last (as the app does)."""
    states, prev = [], None
    for text in turns:
        prev = C.graph.invoke(sv.initial_state(query=text, variant="C", map_state={}, previous=prev, fast_path=True))
        states.append(prev)
    return states


def calls(state, tool=None):
    return [r for r in state.get("tool_call_log") or [] if tool is None or r["tool"] == tool]


def stats_args(state):
    found = calls(state, "query_resale_statistics")
    assert found, [r["tool"] for r in calls(state)]
    return found[-1]["arguments"]


# ---------------------------------------------------------------------------
# RC1 (SEM-04, SPAT-42/45/46, SEM-31): a follow-up keeps the analysis
# ---------------------------------------------------------------------------
def test_flat_type_follow_up_keeps_the_island_wide_town_ranking():
    _, t2 = conversation(LOWEST_4RM, "And for 5-room instead?")   # DEMO-09/t2
    args = stats_args(t2)
    assert args["group_by"] == ["town"] and args["flat_types"] == ["5 ROOM"]
    assert args["sort_by"] == "median_price" and args["order"] == "asc"
    assert "Which town or area" not in t2["draft_response"]
    assert "lowest median price" in t2["draft_response"]


def test_flat_type_follow_up_keeps_the_mrt_distance_bands():
    _, t2 = conversation("Do 4-room flats near MRT cost more? Show median prices by 200m, 500m and 1km distance bands.",
                         "Now use 5-room sales.")   # STATS-01/t2 (SPAT-42)
    args = stats_args(t2)
    assert args["group_by"] == ["distance_band"] and args["near"] == "mrt" and args["flat_types"] == ["5 ROOM"]


def test_town_follow_up_keeps_the_storey_trend_since_2020():
    _, t2 = conversation("How have 4-room prices changed by storey since 2020?", "Limit it to Tampines.")   # SPAT-45
    args = stats_args(t2)
    assert args["group_by"] == ["storey_band"] and args["towns"] == ["TAMPINES"]
    assert args["start_month"] == "2020-01" and args["include_trend"]
    assert not calls(t2, "query_resale_blocks_by_criteria")


def test_cheapest_first_reverses_the_order_and_keeps_price_per_sqm():
    t1, t2 = conversation(PSM_3RM, "Sort from cheapest to most expensive.")   # STATS-04/t2 (SPAT-46)
    assert stats_args(t1)["order"] == "desc"
    args = stats_args(t2)
    assert args["sort_by"] == "median_psm" and args["order"] == "asc" and args["flat_types"] == ["3 ROOM"]
    assert sv.UNCHANGED_NOTE not in t2["draft_response"]
    assert "lowest price per sqm" in t2["draft_response"]


def test_minimum_group_size_and_period_are_applied_and_kept():
    _, t2, t3, t4 = conversation(LOWEST_4RM, "Only groups with at least 20 sales.",   # STATS-02/t2's wording
                                 "How have those prices changed over three years?", PSM_3RM)
    assert stats_args(t2)["min_sales"] == 20 and stats_args(t2)["group_by"] == ["town"]
    args = stats_args(t3)
    assert args["min_sales"] == 20 and args["lookback_years"] == 3 and args["include_trend"]
    assert args["order"] == "asc" and args["flat_types"] == ["4 ROOM"]
    # a new statistics question starts afresh: the tool's own minimum, not the last analysis's
    assert "min_sales" not in stats_args(t4) or stats_args(t4)["min_sales"] == 10


def test_a_new_search_or_another_question_ends_the_analysis():
    _, t2 = conversation(LOWEST_4RM, "Find 4-room flats in Tampines under $600k.")
    assert not calls(t2, "query_resale_statistics") and calls(t2, "query_resale_blocks_by_criteria")
    _, t2 = conversation(PSM_3RM, "Is Blk 510 Bedok North St 3 fairly priced?")
    assert not calls(t2, "query_resale_statistics") and calls(t2, "assess_resale_price")


def test_the_models_reading_of_the_follow_up_does_not_end_the_analysis():
    """A live model read "And for 5-room instead?" as a market snapshot; code's own reading decides."""
    prior = {"wants_group_stats": True, "stats_group_by": ["town"], "stats_sort": "median_price", "stats_order": "asc",
             "flat_types": ["4 ROOM"]}
    fresh = {"flat_types": ["5 ROOM"], "wants_market_snapshot": True}
    c = {**fresh, "clarifications": [sv.WHICH_TOWN]}
    assert sv._carry_analysis(c, prior, fresh, "And for 5-room instead?")
    assert c["wants_group_stats"] and c["stats_group_by"] == ["town"] and c["stats_order"] == "asc"
    assert not c["wants_market_snapshot"] and c["clarifications"] == []


# ---------------------------------------------------------------------------
# RC2 (SPAT-49): no unasked town block shortlist before area statistics
# ---------------------------------------------------------------------------
def test_area_price_distribution_lists_no_town_shortlist():
    (t1,) = conversation(AREA_500)   # STATS-06/t1
    assert calls(t1, "get_area_resale_statistics") and not calls(t1, "query_resale_blocks_by_criteria")
    assert "100 cheapest" not in t1["draft_response"]
    c = t1["user_constraints"]
    assert not sv.needed_for_task("query_resale_blocks_by_criteria", c)               # nor offered to a live model
    assert sv.needed_for_task("query_resale_blocks_by_criteria", {**c, "max_budget_sgd": 600000})


# ---------------------------------------------------------------------------
# RC3 (SPAT-50): a new radius for the same area question
# ---------------------------------------------------------------------------
def test_a_1km_area_widens_the_same_area_statistics():
    _, t2 = conversation(AREA_500, "Use a 1km area instead.")   # STATS-06/t2
    near = calls(t2, "find_resale_blocks_near")
    assert near and near[0]["arguments"]["radius_meters"] == 1000
    assert near[0]["arguments"]["address"] == "Blk 510 Bedok North St 3"
    assert calls(t2, "get_area_resale_statistics") and not calls(t2, "query_resale_blocks_by_criteria")
    assert "within 1000 m of Blk 510" in t2["draft_response"]


def test_area_radius_wording():
    for text in ("Use a 1km area instead.", "Try 800m radius.", "make it 2 km area then", "What about a 1.5km circle?"):
        assert sv._AREA_RADIUS.search(text), text
    assert not sv._AREA_RADIUS.search("Find 4-room in Bedok within 1km of MRT.")


# ---------------------------------------------------------------------------
# RC4 (SPAT-48, partial): a back-reference is read with the earlier turn
# ---------------------------------------------------------------------------
def test_back_reference_is_not_read_alone():
    # after a statistics answer, "those prices" is read with the earlier turn
    _, t2 = conversation('Which towns have the lowest median price for a 4-room resale over the last year?',
                         "How have those prices changed over three years?")
    assert t2["user_constraints"]["extraction_source"] != sv.FAST_PATH_SOURCE


def test_back_reference_after_a_block_search_keeps_the_fast_path():
    # STATS-05: "within 300m of Novena MRT" is still read as a block search (named-station statistics are the
    # teammate's research_intents change); reading "those prices" with it would carry the 300 m limit into a trend
    # question and give an empty search, so the trend question is read on its own until then
    _, t2 = conversation("What are resale prices within 300m of Novena MRT?",
                         "How have those prices changed over three years?")
    assert t2["user_constraints"]["extraction_source"] == sv.FAST_PATH_SOURCE
