"""Demo-readiness fixes to how a flat search is read (Member 4, 4 Nov): budgets, leases and distances in the tester's
own words, follow-ups that change one limit, LRT-only and MRT-only limits, short town names, commute workplaces, and
what the answer says about limits it could not check. Each test uses the tester's wording (data/local/qa/gpt_issues.json,
diag_search.json). Fixes that only the live model path shows use fake model doubles."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents.tool_registry import tools_for_agent  # noqa: E402


def rb(query):
    return sv.rule_based_extract(query)


def conversation(*turns):
    """Each turn's final state, every turn carrying the last (as the app does)."""
    states, prev = [], None
    for text in turns:
        prev = C.run(text, previous=prev)
        states.append(prev)
    return states


def tools_run(state):
    return [r["tool"] for r in state.get("tool_call_log") or []]


# ---------------------------------------------------------------------------
# 1. Budgets (SEM-02, SPAT-03, SPAT-56, SPAT-57)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, budget", [
    ("Any 4-room blocks in Bishan under a million?", 1_000_000),
    ("Bishan four room under one million", 1_000_000),
    ("4-room Queenstown below a million, commute under 40 minutes to Raffles Place and one-north.", 1_000_000),
    ("Find 4 room resale in Yishun <= 550000", 550_000),
    ("Help us find a 3-room in Bedok, ceiling $500,000.", 500_000),
    ("Put back a ceiling of 750k.", 750_000),
    ("Raise the top end to 700k, keep the lower end.", 700_000),
])
def test_budget_words_are_read(query, budget):
    assert rb(query).max_budget_sgd == budget


def test_half_a_million_is_a_budget():
    assert rb("3-room in Bedok under half a million").max_budget_sgd == 500_000


def test_a_bare_budget_follow_up_changes_the_budget():
    first, second = conversation("Any 4-room blocks in Bishan under a million?", "Actually only $850k.")
    assert first["user_constraints"]["max_budget_sgd"] == 1_000_000
    c = second["user_constraints"]
    assert c["max_budget_sgd"] == 850_000 and c["towns"] == ["BISHAN"]
    assert sv.UNCHANGED_NOTE not in (c.get("assumptions") or [])
    assert all(b["median_resale_price"] <= 850_000 for b in second["candidate_blocks_intersection"])


def test_a_ceiling_follow_up_puts_the_budget_back():
    *_, last = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.",
                            "Forget the budget cap, keep the town and MRT distance.", "Put back a ceiling of 750k.")
    c = last["user_constraints"]
    assert c["max_budget_sgd"] == 750_000 and c["amenity_radii_meters"] == {"mrt": 500}


def test_raising_the_top_end_keeps_the_lower_end():
    *_, last = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.",
                            "Budget between 600k and 650k.", "Raise the top end to 700k, keep the lower end.")
    c = last["user_constraints"]
    assert c["max_budget_sgd"] == 700_000 and c["min_price_sgd"] == 600_000


# ---------------------------------------------------------------------------
# 2. Leases (SEM-03, SPAT-15, SPAT-55)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, lease", [
    ("Search Pasir Ris 5-room below $850k, at least 65-year lease.", 65),
    ("5-room Sengkang maximum $800k with 70 years remaining.", 70),
    ("At least 70 years left.", 70),
    ("at least seventy years lease", 70),
    ("Only places with 65 years of lease left or more.", 65),
])
def test_lease_words_are_read(query, lease):
    assert rb(query).min_remaining_lease_years == lease


@pytest.mark.parametrize("query", [
    "4-room in Tampines under $600k, 30 years left before retirement.",
    "4-room in Tampines under $600k, at least 60-year-old flat is fine.",
])
def test_a_working_life_or_flat_age_is_not_a_lease(query):
    assert rb(query).min_remaining_lease_years is None


def test_a_bare_lease_follow_up_changes_the_lease():
    *_, last = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.", "At least 70 years left.",
                            "Actually 60 years will do.")
    c = last["user_constraints"]
    assert c["min_remaining_lease_years"] == 60 and c["amenity_radii_meters"] == {"mrt": 500}
    assert sv.UNCHANGED_NOTE not in (c.get("assumptions") or [])


def test_seventy_years_after_a_search_keeps_the_search():
    first, second = conversation("Bishan four room under one million", "at least seventy years lease")
    c = second["user_constraints"]
    assert c["max_budget_sgd"] == 1_000_000 and c["min_remaining_lease_years"] == 70
    assert all(b["median_remaining_lease_years"] >= 70 for b in second["candidate_blocks_intersection"])


# ---------------------------------------------------------------------------
# 3. Distance limits and removals (SEM-16, SPAT-22, SPAT-62, SPAT-63)
# ---------------------------------------------------------------------------
def test_mrt_limit_wording_changes_the_distance():
    _, second, third = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.",
                                    "Make the MRT limit 900m.", "Tighten it to 300m.")
    assert second["user_constraints"]["amenity_radii_meters"] == {"mrt": 900}
    assert third["user_constraints"]["amenity_radii_meters"] == {"mrt": 300}
    assert all(b["distances"]["mrt_dist_m"] <= 300 for b in third["candidate_blocks_intersection"])


def test_mrt_less_than_800_metres_is_read():
    assert rb("mrt less than 800 metres").amenity_radii_meters == {"mrt": 800}
    _, second = conversation("jurong east 4-room max 0.7m", "mrt less than 800 metres")
    c = second["user_constraints"]
    assert c["amenity_radii_meters"] == {"mrt": 800} and c["max_budget_sgd"] == 700_000


def test_remove_just_the_hawker_condition():
    *_, last = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.", "Add hawker centres within 500m.",
                            "Remove just the hawker condition.")
    assert last["user_constraints"]["amenity_radii_meters"] == {"mrt": 500}


# ---------------------------------------------------------------------------
# 4. Adding or switching towns (SEM-15, SPAT-18)
# ---------------------------------------------------------------------------
def test_include_adds_a_town_and_switch_replaces_them():
    _, second, third = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.",
                                    "Switch to Tampines, same requirements.", "Include Pasir Ris too.")
    assert second["user_constraints"]["towns"] == ["TAMPINES"]
    c = third["user_constraints"]
    assert c["towns"] == ["TAMPINES", "PASIR RIS"] and c["amenity_radii_meters"] == {"mrt": 500}
    assert {b["town"] for b in third["candidate_blocks_intersection"]} == {"TAMPINES", "PASIR RIS"}


# ---------------------------------------------------------------------------
# 5. LRT-only and MRT-only limits (SEM-01, SPAT-02, SPAT-16, SPAT-17, SPAT-38)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, radii", [
    ("4-room in Punggol below 750k; LRT must be within 300m.", {"lrt": 300}),
    ("4-room Choa Chu Kang below 580k within 350m of LRT.", {"lrt": 350}),
    ("Find blocks in Punggol within 250m of LRT, 4-room below 750k.", {"lrt": 250}),
    ("Find 4-room resale in Bedok under $700k within 500m of MRT.", {"mrt": 500}),   # MRT or LRT, as before
    ("4-room in Punggol within 300m of an MRT station only, below 750k", {"mrt_only": 300}),
])
def test_rail_kind_is_read(query, radii):
    assert rb(query).amenity_radii_meters == radii


def test_lrt_limit_is_kept_by_a_follow_up_and_measured_to_lrt():
    first, second = conversation("4-room in Punggol below 750k; LRT must be within 300m.",
                                 "Raise budget to $800k but keep the LRT limit.")
    for st in (first, second):
        assert st["user_constraints"]["amenity_radii_meters"] == {"lrt": 300}
        blocks = st["candidate_blocks_intersection"]
        assert blocks and all(b["distances"]["lrt_dist_m"] <= 300 for b in blocks)
        assert "lrt " in st["draft_response"] and "an LRT station" in st["draft_response"]
    assert second["user_constraints"]["max_budget_sgd"] == 800_000


def test_mrt_stations_only_follow_up_changes_the_rail_kind():
    first, second = conversation("Find blocks in Punggol within 250m of LRT, 4-room below 750k.",
                                 "Use MRT stations only instead.")
    assert second["user_constraints"]["amenity_radii_meters"] == {"mrt_only": 250}
    blocks = second["candidate_blocks_intersection"]
    assert len(blocks) < len(first["candidate_blocks_intersection"])
    assert all(b["distances"]["mrt_only_dist_m"] <= 250 for b in blocks)


def test_rail_labels_keep_the_auditor_form():
    # the answer and the auditor read "mrt 250m" / "lrt 300m"; mrt_only is shown as mrt
    assert sv._distance_label("lrt_dist_m", 240.4) == "lrt 240m"
    assert sv._distance_label("mrt_only_dist_m", 180) == "mrt 180m"
    assert sv._distance_label("mrt_dist_m", 379) == "mrt 379m"


def test_removing_the_lrt_limit_removes_any_rail_limit():
    *_, last = conversation("4-room in Punggol below 750k; LRT must be within 300m.", "Remove the LRT limit.")
    assert not last["user_constraints"].get("amenity_radii_meters")


def test_lrt_comparison_is_checked_against_the_lrt_distance():
    listed = [{"block": "1", "street": "A", "distances": {"lrt_dist_m": 100.0}},
              {"block": "2", "street": "B", "distances": {"lrt_dist_m": 300.0}}]
    assert sv.comparisons_hold("Blk 1 is nearer an LRT station.", listed)
    assert not sv.comparisons_hold("Blk 2 is nearer an LRT station.", listed)


# ---------------------------------------------------------------------------
# 6. Short town names, and no island-wide distance search without a place (SEM-33, SPAT-51, SPAT-52, LIVE-WS-01)
# ---------------------------------------------------------------------------
def test_tamp_is_tampines():
    assert rb("4rm tamp under 650k can?").towns == ["TAMPINES"]
    assert rb("4rm amk under 650k").towns == ["ANG MO KIO"]
    first, second = conversation("4rm tamp under 650k can?", "near mrt 700m also")
    assert first["user_constraints"]["towns"] == ["TAMPINES"]
    assert second["user_constraints"]["towns"] == ["TAMPINES"]
    assert {b["town"] for b in second["candidate_blocks_intersection"]} == {"TAMPINES"}


def test_a_distance_with_no_place_asks_which_town():
    _, second = conversation("4rm under 650k can?", "near mrt 700m also")
    assert sv.WHICH_TOWN in second["draft_response"]
    assert "filter_blocks_by_amenity_proximity" not in tools_run(second)
    assert not second["candidate_blocks_intersection"]


def test_an_island_wide_search_stays_island_wide():
    _, second = conversation("Find 4-room anywhere in Singapore under $550k.", "Within 700m of MRT as well.")
    c = second["user_constraints"]
    assert c["islandwide"] and c["amenity_radii_meters"] == {"mrt": 700}
    assert sv.WHICH_TOWN not in second["draft_response"] and second["candidate_blocks_intersection"]


# ---------------------------------------------------------------------------
# 7. Commutes (SEM-27, SEM-28, SPAT-33)
# ---------------------------------------------------------------------------
def test_two_workplaces_joined_by_and_are_both_read():
    c = rb("4-room Queenstown below a million, commute under 40 minutes to Raffles Place and one-north.")
    assert c.work_destinations == ["Raffles Place", "one-north"] and c.max_commute_minutes == 40
    assert c.towns == ["QUEENSTOWN"]


def test_a_town_named_only_as_a_workplace_is_not_the_search_town():
    c = rb("Change the workplaces to Changi Business Park and Tampines.")
    assert c.work_destinations == ["Changi Business Park", "Tampines"] and not c.towns
    _, second = conversation("4-room Queenstown below a million, commute under 40 minutes to Raffles Place and one-north.",
                             "Change the workplaces to Changi Business Park and Tampines.")
    c2 = second["user_constraints"]
    assert c2["towns"] == ["QUEENSTOWN"] and c2["work_destinations"] == ["Changi Business Park", "Tampines"]


def test_below_30_minutes_is_a_commute_limit_and_is_said_to_be_unchecked():
    c = rb("4-room Bedok under 700k, below 30 minutes to Raffles Place.")
    assert c.work_destinations == ["Raffles Place"] and c.max_commute_minutes == 30
    state = C.run("4-room Bedok under 700k, below 30 minutes to Raffles Place.")
    assert "every limit you set" not in state["draft_response"]
    assert sv.unchecked_commute_line(state) in state["draft_response"]


@pytest.mark.parametrize("words", ["no more than", "at most", "up to", "<"])
def test_minutes_limit_words(words):
    assert sv.commute_limits(f"4-room Bedok under 700k, commute {words} 35 minutes to Raffles Place") == [35]


# ---------------------------------------------------------------------------
# 8. A workplace nobody can locate is not timed (SPAT-64, SPAT-65)
# ---------------------------------------------------------------------------
def test_unknown_workplace_is_named_and_not_timed():
    first, second = conversation(
        "4-room Bishan under 900k, both commutes under 35 minutes to Unknown Workplace QZX and one-north.",
        "I meant a building called QZX; I have no address yet.")
    for st in (first, second):
        text = st["draft_response"]
        assert "compute_dual_transit_commute" not in tools_run(st)
        assert "I could not find “Unknown Workplace QZX”" in text and "address" in text
        assert "every limit you set" not in text and "all stated constraints" not in text


def test_known_workplaces_are_known():
    assert sv.workplace_known("Raffles Place") and sv.workplace_known("one-north")
    assert not sv.workplace_known("Unknown Workplace QZX")


# ---------------------------------------------------------------------------
# 9. A distance to a place with no location data makes the answer "partly checked" (SEM-11, SPAT-12)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    "4-room Bedok below 650k, within 200m of a park.",
    "Find Bedok 4-room under 650k within 100m of a bus stop.",
])
def test_unmeasurable_place_is_not_promised(query):
    state = C.run(query)
    assert sv.partly_checked(state)
    text = state["draft_response"]
    assert "every limit you set" not in text and "all stated constraints" not in text
    assert "I can't measure distances" in text


# ---------------------------------------------------------------------------
# 10. The map count (SEM-26, SPAT-01)
# ---------------------------------------------------------------------------
def test_map_count_says_how_many_blocks_are_on_the_map():
    state = C.run("4-room resale in Tampines below $650k, please.")
    blocks = state["candidate_blocks_intersection"]
    mapped = sum(1 for b in blocks if b.get("lat") and b.get("lng"))
    assert len(blocks) > sv.LISTED_BLOCKS and mapped < len(blocks)
    text = state["draft_response"]
    assert "all shown on the map" not in text
    assert f"{mapped} of the {len(blocks)} blocks are on the map ({len(blocks) - mapped} " in text


# ---------------------------------------------------------------------------
# Live model path (fake model doubles): readings the code check used to drop, and the tool loop's guards
# ---------------------------------------------------------------------------
class ReadingModel:
    """Structured output returns the given reading; the tool loop calls nothing."""

    def __init__(self, reading):
        self.reading = reading

    def with_structured_output(self, schema, method=None, include_raw=False):
        reading = self.reading

        class Runner:
            def invoke(self, _messages):
                return {"parsed": schema(**reading), "raw": None}
        return Runner()

    def bind_tools(self, _tools, tool_choice=None):
        return self

    def invoke(self, _messages):
        return AIMessage(content="")


class ToolModel(ReadingModel):
    """The tool loop makes the given calls once, then stops."""

    def __init__(self, calls):
        super().__init__({})
        self.calls, self.n = calls, 0

    def invoke(self, _messages):
        self.n += 1
        return AIMessage(content="", tool_calls=self.calls if self.n == 1 else [])


@pytest.mark.parametrize("query, reading, key, value", [
    ("Any 4-room blocks in Bishan under a million?",
     dict(towns=["BISHAN"], flat_types=["4 ROOM"], max_budget_sgd=1_000_000), "max_budget_sgd", 1_000_000),
    ("Find 4 room resale in Yishun <= 550000",
     dict(towns=["YISHUN"], flat_types=["4 ROOM"], max_budget_sgd=550_000), "max_budget_sgd", 550_000),
    ("Bishan four room, at least seventy years lease",
     dict(towns=["BISHAN"], flat_types=["4 ROOM"], min_remaining_lease_years=70), "min_remaining_lease_years", 70),
])
def test_model_reading_of_spelled_or_symbol_limits_is_kept(monkeypatch, query, reading, key, value):
    # code's check on the model's reading dropped these, since "a million" / "<= 550000" / "seventy" were not read
    monkeypatch.setattr(sv, "get_llm", lambda: ReadingModel(reading))
    out = sv.extract_constraints_node(sv.initial_state(query, "C"))
    assert out["user_constraints"][key] == value
    assert not any("dropped" in e and key in e for e in out["errors"])


def test_model_reading_of_an_lrt_limit_is_lrt_only(monkeypatch):
    reading = dict(towns=["PUNGGOL"], flat_types=["4 ROOM"], max_budget_sgd=750_000, amenity_radii_meters={"mrt": 300})
    monkeypatch.setattr(sv, "get_llm", lambda: ReadingModel(reading))
    out = sv.extract_constraints_node(sv.initial_state("4-room in Punggol below 750k; LRT must be within 300m.", "C"))
    assert out["user_constraints"]["amenity_radii_meters"] == {"lrt": 300}


def test_commute_with_a_town_as_its_blocks_times_the_search_blocks(monkeypatch):
    # LIVE-SPATIAL-01: the model passed ["QUEENSTOWN"] as the blocks to time, with no other spatial filter run
    calls = [{"name": "compute_dual_transit_commute", "id": "t1", "args": {
        "candidate_blocks": ["QUEENSTOWN"], "work_dest_a": "Raffles Place", "work_dest_b": "one-north", "max_minutes": 40}}]
    monkeypatch.setattr(sv, "get_llm", lambda: ToolModel(calls))
    c = {"towns": ["QUEENSTOWN"], "flat_types": ["4 ROOM"], "work_destinations": ["Raffles Place", "one-north"],
         "max_commute_minutes": 40}
    outputs, log, errors, _ = sv.run_react("spatial", ["filter_blocks_by_amenity_proximity", "compute_dual_transit_commute"],
                                           "sys", "q", c)
    commute = [r for r in log if r["tool"] == "compute_dual_transit_commute"]
    assert commute and commute[0]["arguments"]["candidate_blocks"] == []
    assert outputs["compute_dual_transit_commute"]
    assert any("no filter ran" in e for e in errors)


def test_block_search_the_model_skipped_is_run_by_the_supervisor(monkeypatch):
    # LIVE-SEARCH-01: the model fetched statistics only and answered "I could not find data" for a block search
    calls = [{"name": "query_resale_statistics", "id": "a", "args": {
        "towns": ["QUEENSTOWN", "CLEMENTI"], "flat_types": ["3 ROOM"], "group_by": ["town"]}}]
    monkeypatch.setattr(sv, "get_llm", lambda: ToolModel(calls))
    q = "Compare available historical block summaries for 3-room in Queenstown and Clementi under 750k."
    c = dict(towns=["QUEENSTOWN", "CLEMENTI"], flat_types=["3 ROOM"], max_budget_sgd=750_000)
    outputs, log, errors, _ = sv.run_react("market_sql", tools_for_agent("market_sql"), "sys", q, c)
    added = [r for r in log if r["tool"] == "query_resale_blocks_by_criteria"]
    assert added and added[0]["agent"] == "supervisor"
    assert any("added query_resale_blocks_by_criteria" in e for e in errors)
    # the model's own calls are still logged as the model's
    assert [r["agent"] for r in log if r["tool"] == "query_resale_statistics"] == ["market_sql"]
