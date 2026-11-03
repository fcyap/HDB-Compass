"""Demo-readiness fixes to follow-ups and places (Member 4, 4 Nov): a radius on its own changes the pin's, the school
lookup's or the last place's radius; a question about the last answer (live listings, straight-line distances, what can
be measured) is answered by code instead of running the search again; "Where is <block>?" says what is near it; a
block with no street, an unpicked school and an unfound workplace get a focused question; the "same search" note only
when it is; one line per MRT station. Each test uses the tester's wording (data/local/qa/gpt_issues.json,
gpt_records.json); map-pin turns carry the pin the app sends."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402

PIN = {"center": [1.3508, 103.8481], "radius_meters": 500}


def conversation(*turns):
    """Each turn's final state, every turn carrying the last (as the app does). A turn is a message or (message, map)."""
    states, prev = [], None
    for turn in turns:
        text, ms = (turn, {}) if isinstance(turn, str) else turn
        prev = C.graph.invoke(sv.initial_state(query=text, variant="C", map_state=ms, previous=prev, fast_path=True))
        states.append(prev)
    return states


def calls(state, tool=None):
    return [r for r in state.get("tool_call_log") or [] if tool is None or r["tool"] == tool]


def question(state):
    return (state.get("clarification") or {}).get("question") or ""


# ---------------------------------------------------------------------------
# RC1 (SEM-09, SPAT-08/10/11/29/30): a radius on its own keeps the pin, the address and the school
# ---------------------------------------------------------------------------
def test_pin_radius_expands_with_and_without_the_pin_resent():
    for second in (("Expand to 1km.", PIN), "Expand to 1km."):
        _, t2 = conversation(("What is around this pin within 500m?", PIN), second)
        nearby = calls(t2, "get_nearby_amenities")
        assert nearby and nearby[0]["arguments"]["radius_meters"] == 1000
        assert nearby[0]["arguments"]["latitude"] == 1.3508
        assert "within 1000 m" in t2["draft_response"]


def test_school_lookup_radius_narrows_for_the_same_address():
    _, t2 = conversation("Which primary schools are within 2km of Blk 510 Bedok North St 3?", "Only within 1km, please.")
    assert t2["nearby_school_results"]["radius_meters"] == 1000
    assert "Search radius: 1000 m" in t2["draft_response"]
    assert "What would you like to do" not in t2["draft_response"]


def test_school_distance_expands_for_the_school_already_named():
    _, t2 = conversation("Find 4-room Bishan below 950k within 1km of Ai Tong School.", "Expand the school distance to 2km.")
    assert "Which primary school" not in t2["draft_response"]
    school = calls(t2, "verify_primary_school_distance")
    assert school and school[0]["arguments"]["school_name"] == "Ai Tong School"
    assert school[0]["arguments"]["bracket"] == "2km"
    assert t2["user_constraints"]["amenity_radii_meters"] == {"school": 2000}


def test_bare_distance_reads_prefixes_and_politeness():
    for text in ("Expand to 1km.", "Only within 1km, please.", "Expand the school distance to 2km.", "Tighten it to 300m."):
        assert sv._BARE_DISTANCE.search(text), text
    assert not sv._BARE_DISTANCE.search("Find 4-room in Bedok within 1km of MRT.")


# ---------------------------------------------------------------------------
# RC2 (SEM-14, SPAT-13/14/37/59, SEM-30): questions about the last answer get an answer, not the same search
# ---------------------------------------------------------------------------
def test_live_listing_question_is_answered_not_searched():
    _, t2 = conversation("3-room Yishun below 450k, at least 60 years left.", "Are these live flats being sold now?")
    assert t2["active_subagents"] == [] and not calls(t2, "query_resale_blocks_by_criteria")
    assert t2["draft_response"].startswith("These are not live listings")
    assert "past resale transactions" in t2["draft_response"]


def test_straight_line_question_is_answered():
    _, t2 = conversation("Find 4-room Bedok under 700k within 400m of a supermarket and 600m of MRT.",
                         "Is that actual walking-route distance or a straight line?")
    assert t2["active_subagents"] == []
    assert "**straight-line**" in t2["draft_response"] and "candidate block" not in t2["draft_response"]


def test_coverage_question_says_the_pool_limit_cannot_be_measured():
    _, t2 = conversation("I need a 4-room in Tampines under 750k next to a swimming pool.",
                         "Can you actually measure swimming-pool proximity?")
    assert t2["draft_response"].startswith("No. I can't measure distances to swimming pools")
    assert "not applied to the list" in t2["draft_response"]
    assert "meet every limit" not in t2["draft_response"]


def test_a_new_search_in_the_same_message_wins_and_the_search_carries_on():
    t1, t2 = conversation("3-room Yishun below 450k, at least 60 years left.",
                          "Are there live listings for 4-room in Bedok under 600k?")
    assert not t2["user_constraints"].get("about_results")
    assert calls(t2, "query_resale_blocks_by_criteria")
    # after an about-the-answer turn the next refinement still changes the last search
    _, _, t3 = conversation("Find 4-room Bedok under 700k within 400m of a supermarket and 600m of MRT.",
                            "Is that actual walking-route distance or a straight line?", "Remove the supermarket limit.")
    assert t3["user_constraints"]["amenity_radii_meters"] == {"mrt": 600}
    assert t3["candidate_blocks_intersection"]


def test_about_results_needs_an_earlier_search_or_place():
    assert sv.about_last_answer("Are these live flats being sold now?", None) is None
    assert sv.about_last_answer("Are these live flats being sold now?", {"towns": ["YISHUN"]}) == sv.ABOUT_LIVE


# ---------------------------------------------------------------------------
# RC4 (SEM-29, SPAT-36): "centres are" is not Central Area; a distance after one place's nearby answer
# ---------------------------------------------------------------------------
def test_two_words_are_not_a_misspelt_town():
    assert sv.misspelt_towns("Which hawker centres are within 700m?") == []
    assert sv.misspelt_towns("4-room in Tampenis") == [("Tampenis", "TAMPINES")]


def test_hawker_radius_after_where_is_block_is_that_blocks_neighbourhood():
    t1, t2 = conversation("Where is Blk 288B Jurong East St 21?", "Which hawker centres are within 700m?")
    assert calls(t1, "get_nearby_amenities")
    assert not calls(t2, "filter_blocks_by_amenity_proximity")
    nearby = calls(t2, "get_nearby_amenities")
    assert nearby and nearby[0]["arguments"]["address"] == "Blk 288B Jurong East St 21"
    assert nearby[0]["arguments"]["radius_meters"] == 700
    assert "Central Area" not in t2["draft_response"] and "No block" not in t2["draft_response"]


# ---------------------------------------------------------------------------
# RC5 (SEM-30, SPAT-40, SPAT-58): focused questions for a bare block, an unpicked school, an unfound workplace
# ---------------------------------------------------------------------------
def test_block_without_street_asks_where_it_is():
    t1, t2 = conversation("Where is Blk 123 in Singapore?", "I don't know the street or postal code.")
    assert question(t1).startswith("Which Blk 123 do you mean?") and "postal code" in question(t1)
    assert "drop a pin" in question(t1)
    assert not calls(t2) and "can't tell which Blk 123" in question(t2)
    assert "past sales" not in t2["draft_response"]


def test_unpicked_school_gets_the_ranking_answer():
    t1, t2 = conversation("Show flats near a good primary school.", "I have not picked a school yet.")
    assert not calls(t2, "get_nearby_amenities")
    assert question(t2) == sv.NO_SCHOOL_PICKED
    assert (t2["clarification"] or {}).get("kind") == "school"


def test_unfound_workplace_is_asked_for_and_a_reply_is_timed():
    t1, t2 = conversation("4-room Bishan under 900k, both commutes under 35 minutes to Unknown Workplace QZX and one-north.",
                          "I meant a building called QZX; I have no address yet.")
    assert (t1["clarification"] or {}).get("kind") == "workplace" and question(t1).startswith("Where is “Unknown Workplace QZX”?")
    assert (t2["clarification"] or {}).get("kind") == "workplace" and question(t2).startswith("Without an address")
    assert "meet every limit" not in t2["draft_response"]
    _, t2b = conversation("4-room Bishan under 900k, both commutes under 35 minutes to Unknown Workplace QZX and one-north.",
                          "Raffles Place")
    assert t2b["user_constraints"]["work_destinations"] == ["one-north", "Raffles Place"]
    assert calls(t2b, "compute_dual_transit_commute") and not t2b.get("clarification")


# ---------------------------------------------------------------------------
# RC3 (SEM-17): the "same search" note only when the turn is the same search
# ---------------------------------------------------------------------------
def test_conditional_followup_has_no_unchanged_note():
    *_, t3 = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.", "Now only blocks within 200m of MRT.",
                          "If nothing fits, show which one change would help.")
    assert sv.UNCHANGED_NOTE not in (t3["user_constraints"].get("assumptions") or [])
    assert "couldn't tell what to change" not in t3["draft_response"]


def test_cheapest_of_those_has_no_unchanged_note():
    _, t2 = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.", "Which of those blocks is cheapest?")
    assert "couldn't tell what to change" not in t2["draft_response"]


# ---------------------------------------------------------------------------
# RC6 (SEM-32, SPAT-09): one line per MRT station
# ---------------------------------------------------------------------------
def test_pin_lists_bishan_interchange_once():
    (t1,) = conversation(("What is around this pin within 500m?", PIN))
    mrt = next(line for line in t1["draft_response"].splitlines() if line.startswith("- MRT/LRT"))
    assert mrt.count("Bishan Interchange") == 1 and mrt.startswith("- MRT/LRT (1):")


def test_station_dedupe_keeps_other_kinds_and_recounts():
    result = {"status": "ok", "counts": {"mrt": 3, "supermarket": 2}, "amenities": [
        {"category": "mrt", "name": "Bishan Interchange Mrt Station", "distance_meters": 135.0},
        {"category": "mrt", "name": "Bishan Interchange Mrt Station", "distance_meters": 6.0},
        {"category": "mrt", "name": "Marymount Mrt Station", "distance_meters": 993.0},
        {"category": "supermarket", "name": "NTUC FairPrice", "distance_meters": 82.0},
        {"category": "supermarket", "name": "NTUC FairPrice", "distance_meters": 281.0}]}
    out = sv._dedupe_stations(result)
    assert out["counts"] == {"mrt": 2, "supermarket": 2}
    assert [a["distance_meters"] for a in out["amenities"] if a["category"] == "mrt"] == [6.0, 993.0]
    assert len([a for a in out["amenities"] if a["category"] == "supermarket"]) == 2
