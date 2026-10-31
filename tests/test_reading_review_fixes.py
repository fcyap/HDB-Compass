"""Second review of the supervisor reading fixes (1 Nov): amenity minutes are not commute limits, the regex merge
never adds a policy route the model did not choose, commute places typed in lower case still count, "move to" and
"open to" name search towns, floors are not schools, and "Bt" is Bukit. Each test uses the reviewer's phrasing."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"

import pytest  # noqa: E402

from agents import supervisor as sv  # noqa: E402
from tests.test_reading_regressions import model_path, run_extract  # noqa: E402


def dual_commute_minutes(c, query):
    return [args["max_minutes"] for name, args in sv.deterministic_plan("spatial", c, query)
            if name == "compute_dual_transit_commute"]


# ---------------------------------------------------------------------------
# 1. Minutes about an amenity or family are not the commute limit, so the stricter-limit rule keeps the real one
# ---------------------------------------------------------------------------
AMENITY_AND_COMMUTE = [
    ("My husband works at Raffles Place and I work at Jurong East. Find 4-room flats in Bukit Merah with MRT within "
     "10 mins, commute under 45 mins", 45),
    ("I work at Raffles Place and my wife works at Changi Airport; we want a flat in Tampines with max 50 min commute, "
     "near clinic within 10 min", 50),
    ("I work at Raffles Place and my wife works at one-north. 4-room in Queenstown within 10 mins of my parents, both "
     "of us under 40 min commute", 40),
    ("I work at Raffles Place and my wife works at one-north; 4-room in Bishan within 15 mins by bus to MRT, commute "
     "under 40 min", 40),
]


@pytest.mark.parametrize("query, minutes", AMENITY_AND_COMMUTE)
def test_amenity_minutes_do_not_replace_the_commute_limit(query, minutes):
    c = sv.rule_based_extract(query).model_dump()
    assert c["max_commute_minutes"] == minutes
    assert dual_commute_minutes(c, query) == [minutes]


def test_hawker_minutes_before_a_commute_limit():
    c = sv.rule_based_extract("Hawker centre within 10 mins and under 45 min commute to one-north, 4-room in Clementi")
    assert c.max_commute_minutes == 45 and c.work_destinations == ["one-north"]


def test_amenity_minutes_alone_are_no_commute_limit():
    assert sv.commute_limits("4-room in Tampines with MRT within 10 mins") == []
    # two commute limits still keep the stricter one
    assert sv.commute_limits("Commute to Raffles Place within 30 mins and to Jurong East within 40 mins") == [30, 40]


def test_model_path_keeps_the_stated_commute_limit_when_the_model_leaves_it_empty(monkeypatch):
    q = AMENITY_AND_COMMUTE[0][0]
    out = run_extract(monkeypatch, q, towns=["BUKIT MERAH"], flat_types=["4 ROOM"],
                      work_destinations=["Raffles Place", "Jurong East"])
    c = out["user_constraints"]
    assert c["max_commute_minutes"] == 45 and dual_commute_minutes(c, q) == [45]


# ---------------------------------------------------------------------------
# 2. The regex merge fills an empty policy_question, but only the model's own policy reading is kept on wording alone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, reading", [
    ("Can I buy a 4-room flat in Tampines under $600k?",
     dict(towns=["TAMPINES"], flat_types=["4 ROOM"], max_budget_sgd=600000)),
    ("Can we buy a 5-room in Punggol with 700k budget?",
     dict(towns=["PUNGGOL"], flat_types=["5 ROOM"], max_budget_sgd=700000)),
])
def test_plain_search_the_model_did_not_flag_gets_no_policy_route(monkeypatch, query, reading):
    out = run_extract(monkeypatch, query, **reading)
    c = out["user_constraints"]
    assert not c["policy_question"]
    # a search, so the market agent runs; main's affordability flow may add the financial agent, never policy
    routing = sv.decide_routing(c)
    assert "market_sql" in routing and "policy_rag" not in routing


def test_rule_question_the_model_flagged_still_reaches_policy(monkeypatch):
    q = "Can I buy a resale flat after selling my condo?"
    c = run_extract(monkeypatch, q, policy_question=q)["user_constraints"]
    assert c["policy_question"] == q and sv.decide_routing(c) == ["policy_rag"]


# ---------------------------------------------------------------------------
# 3. In the commute clause a place typed in lower case is a destination; "Raffles Place MRT" is Raffles Place
# ---------------------------------------------------------------------------
def test_lower_case_second_destination_keeps_the_dual_commute_check():
    q = "4rm in amk, commute to one-north and to jurong east under 45 min"
    c = sv.rule_based_extract(q).model_dump()
    assert c["work_destinations"] == ["one-north", "jurong east"] and c["max_commute_minutes"] == 45
    assert "spatial" in sv.decide_routing(c)
    assert dual_commute_minutes(c, q) == [45]


@pytest.mark.parametrize("query, work", [
    ("max 45 min commute to raffles place, 4-room in Bishan", ["raffles place"]),
    ("4-room in Bishan, commute to Raffles Place MRT under 45 mins", ["Raffles Place"]),
    ("4-room in Bishan, my commute to work under 45 mins", []),
    ("Find flats in Bishan within 10 min walk to Bishan MRT with short commute", []),
])
def test_commute_clause_destinations(query, work):
    assert sv.rule_based_extract(query).work_destinations == work


# ---------------------------------------------------------------------------
# 4. "move to X" and "open to X" name search towns, not workplaces
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, towns", [
    ("Planning to move to Sengkang, commute to Raffles Place under 45 mins, 4-room", ["SENGKANG"]),
    ("Open to Bishan or AMK, commute to Raffles Place under 45 min", ["BISHAN", "ANG MO KIO"]),
])
def test_move_to_and_open_to_keep_the_search_town(query, towns):
    c = sv.rule_based_extract(query)
    assert c.work_destinations == ["Raffles Place"] and c.towns == towns


# ---------------------------------------------------------------------------
# 5. A high floor is not a school; a name after "the" is still the user's school
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    "Find 4-room flats near MRT high-floor unit in Tampines",
    "4rm near MRT pref high flr Tampines under 600k",
    "4-room near MRT high up",
    "Find 4-room flats near MRT High floor in Tampines",
])
def test_high_floor_wording_is_not_a_school(query):
    c = sv.rule_based_extract(query)
    assert c.school_name is None and "school" not in c.amenity_radii_meters


@pytest.mark.parametrize("query, school", [
    ("4-room within 1km of the Henry Park Primary School", "Henry Park Primary School"),
    ("4-room near the Rosyth School in Serangoon", "Rosyth School"),
    ("4-room near Catholic High", "Catholic High"),
    ("4-room within 1km of catholic high school", "catholic high school"),
])
def test_school_names_still_read(query, school):
    c = sv.rule_based_extract(query)
    assert c.school_name == school and c.amenity_radii_meters["school"] == 1000


def test_lower_case_phrases_about_some_school_are_still_not_names():
    for q in ("Find flats near my kids school", "Find flats close to our old school", "near the nearest school"):
        assert sv.rule_based_extract(q).school_name is None, q


# ---------------------------------------------------------------------------
# 6. "Bt" is Bukit, for the regex reading and for the check on the model's reading
# ---------------------------------------------------------------------------
def test_bt_batok_read_by_the_model_is_kept():
    c, errors = model_path("Bt Batok 4rm under 500k", towns=["BUKIT BATOK"])
    assert c["towns"] == ["BUKIT BATOK"] and not any("not named in the message" in e for e in errors)
    assert sv.named_towns("4-room in Bt. Merah") == ["BUKIT MERAH"]
    assert sv.named_towns("I bought a BTO") == []
