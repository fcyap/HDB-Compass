"""Supervisor reading and routing regressions found in the 1 Nov review: what code reads from the user's words, and
what it keeps of a model's reading. Each test uses the reproduction phrasing from the review."""
from __future__ import annotations

import json
import os

os.environ["LLM_PROFILE"] = "none"

import pytest  # noqa: E402

from agents import research_intents as ri  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents.schemas import LLMExtractedConstraints  # noqa: E402
from tests.test_research_orchestration import FakeLLM  # noqa: E402


def _queries(path):
    data = json.load(open(os.path.join(os.path.dirname(__file__), "..", "evaluation", path)))
    items = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
    return {item.get("query_id") or item.get("id"): item for item in items}


VARIED = _queries("category_e_varied.json")
EXTRA = _queries("category_e_extra.json")
CAT_C = _queries("category_c_queries.json")


def model_path(query, **reading):
    """A model reading of the query put through the same finalize step as extract_constraints_node."""
    st = {"messages": [{"role": "user", "content": query}], "variant": "C"}
    c = {**sv.rule_based_extract(query).model_dump(), **reading}
    errors = []
    return sv._finalize_constraints(c, query, st, errors, from_model=True), errors


def run_extract(monkeypatch, query, **reading):
    monkeypatch.setattr(sv, "get_llm", lambda: FakeLLM(extraction=LLMExtractedConstraints(**reading)))
    return sv.extract_constraints_node(sv.initial_state(query, "C"))


# ---------------------------------------------------------------------------
# 1. Short town names the prompt teaches (JE, AMK, TPY) survive the check on the model's reading
# ---------------------------------------------------------------------------
SHORTHAND = ["E1-01~shorthand", "E1-02~shorthand", "E1-03~shorthand", "E1-07~shorthand", "E1-08~shorthand",
             "E2-02~shorthand", "E2-05~shorthand", "E2-06~shorthand", "E2-07~shorthand"]


@pytest.mark.parametrize("qid", SHORTHAND)
def test_model_read_shorthand_town_is_kept(qid):
    query = VARIED[qid]["query"]
    towns = sv.rule_based_extract(query).towns
    assert len(towns) == 1
    c, errors = model_path(query, towns=towns)
    assert c["towns"] == towns and not any("not named in the message" in e for e in errors)
    assert "market_sql" in sv.decide_routing(c)


def test_je_reproduction_keeps_jurong_east_and_the_market_search():
    q = "4rm resale JE <750k within 500m of MRT"
    c, errors = model_path(q, towns=["JURONG EAST"])
    assert c["towns"] == ["JURONG EAST"] and not errors
    assert sv.decide_routing(c) == ["market_sql", "spatial"]


def test_short_town_check_is_a_whole_word_and_still_drops_invented_towns():
    c, _ = model_path("4-room flats in amk under 600k", towns=["ANG MO KIO"])
    assert c["towns"] == ["ANG MO KIO"]                       # lower case is fine once the model read it
    c, errors = model_path("4-room flats in Bedok under 600k, I like jeans", towns=["BEDOK", "JURONG EAST"])
    assert c["towns"] == ["BEDOK"] and any("JURONG EAST" in e for e in errors)
    assert sv.named_towns("I like jeans") == [] and sv.named_towns("4rm in JE") == ["JURONG EAST"]


# ---------------------------------------------------------------------------
# 2. "What grants do/can we get?" with calculator figures is the calculator's question, not policy
# ---------------------------------------------------------------------------
CALCULATOR_GRANT_ASKS = [VARIED[f"E1-0{i}~casual"]["query"] for i in range(1, 9)] + \
    [EXTRA["E1-X01"]["query"], EXTRA["E1-X03"]["query"]]


@pytest.mark.parametrize("query", CALCULATOR_GRANT_ASKS, ids=[f"q{i}" for i in range(len(CALCULATOR_GRANT_ASKS))])
def test_grant_and_budget_question_with_figures_is_not_policy(query):
    assert sv.rule_based_extract(query).policy_question is None
    c, _ = model_path(query, policy_question=None)
    assert "policy_rag" not in sv.decide_routing(c) and "financial" in sv.decide_routing(c)


def test_grant_and_budget_question_does_not_route_to_policy_on_the_model_path(monkeypatch):
    query = VARIED["E1-01~casual"]["query"]
    out = run_extract(monkeypatch, query, towns=["JURONG EAST"], flat_types=["4 ROOM"], max_budget_sgd=650000)
    assert out["user_constraints"]["policy_question"] is None
    assert "policy_rag" not in sv.decide_routing(out["user_constraints"])


@pytest.mark.parametrize("query", [
    "What grants can I get?", "What grants can a first-timer couple get for a resale flat?",
    "What grants are there for singles?", "Am I eligible for the EHG?", CAT_C["C2-04"]["query"],
])
def test_real_grant_policy_questions_still_reach_policy(query):
    c = sv.rule_based_extract(query).model_dump()
    assert c["policy_question"] and "policy_rag" in sv.decide_routing(c)


def test_expected_agents_match_the_policy_reading_on_every_category_e_and_c_query():
    for items in (VARIED, EXTRA, CAT_C):
        for qid, item in items.items():
            policy = sv.rule_based_extract(item["query"]).policy_question is not None
            assert policy == ("policy_rag" in item["expected_agents"]), qid


# ---------------------------------------------------------------------------
# 3. A school name needs name casing (or words that can be a name); lower-case phrases about some school are not one
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    "Find 4-room flats in Tampines under $600k close to MRT with high floor",
    "Find a 4-room in Tampines under 600k near my kids school",
    "4-room in Bishan close to my son's primary school",
    "flats near to our old school",
    "resale flats in Hougang near to a secondary school",
    "Find 4-room flats in Bishan close to the nearest school",
])
def test_lower_case_phrase_is_not_a_school(query):
    c = sv.rule_based_extract(query)
    assert c.school_name is None and "school" not in c.amenity_radii_meters
    merged, _ = model_path(query, school_name=None)
    assert merged["school_name"] is None


@pytest.mark.parametrize("query, school", [
    ("Find 4-room flats in Bishan near Ai Tong School", "Ai Tong School"),
    ("Find 4-room resale flats in Pasir Ris under $600,000 within 1km of Meridian Primary School", "Meridian Primary School"),
    ("Find 4-room resale flats in Bishan under $800,000 within 1km of Anglo-Chinese School (Junior)",
     "Anglo-Chinese School (Junior)"),
    ("Find 4-room resale flats in Bishan under $800,000 within 1km of St. Hilda's Primary School", "St. Hilda's Primary School"),
    ("Find 4-room resale flats in Bishan under $800,000 within 1km of CHIJ St. Nicholas Girls' School",
     "CHIJ St. Nicholas Girls' School"),
    ("flats near to Kong Hwa School in Geylang under 800k", "Kong Hwa School"),
    ("show hdb blocks within 1–2km of catholic high school.", "catholic high school"),
])
def test_named_school_is_read(query, school):
    c = sv.rule_based_extract(query)
    assert c.school_name == school and "school" in c.amenity_radii_meters


def test_phrase_naming_no_school_still_asks_which_school():
    c = sv.rule_based_extract("Find 4-room flats in Bishan near the best schools").model_dump()
    assert c["school_name"] is None and c["clarifications"] == [ri.SCHOOL_CLARIFICATION]


def test_school_guard_lets_st_and_hyphenated_names_through():
    # Variant C's guard asks for a school name whenever the regex finds none
    from agents.school_queries import guard_school_request
    for name in ("St. Hilda's Primary School", "CHIJ St. Nicholas Girls' School", "Anglo-Chinese School (Primary)"):
        q = f"Find 4-room resale flats in Bishan under $800,000 within 1km of {name}"
        assert guard_school_request(q, lambda text: bool(sv.rule_based_extract(text).school_name)) is None, name


# ---------------------------------------------------------------------------
# 4. Commute destinations are places, limits keep the stricter value, workplace towns are not search towns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query, work, minutes", [
    ("4-room in Tampines under $600k, close to MRT, commute under 45 min to Raffles Place", ["Raffles Place"], 45),
    ("Find 4-room flats in Bishan, my commute to Raffles Place should be under 40 mins and I want to be close to a "
     "hawker centre", ["Raffles Place"], 40),
    ("I want to commute to work at Raffles Place within 30 minutes; find flats in Bishan", ["Raffles Place"], 30),
    ("Looking for a flat in Punggol near MRT. My commute to work is long. I want to be close to my parents.", [], None),
    ("I commute to Raffles Place daily. Find 4-room flats in Tampines with commute to work under 30 min",
     ["Raffles Place"], 30),
    ("4-room in Bishan, commute to one-north and to NUS under 25 min", ["one-north", "NUS"], 25),
    ("Find flats in Bishan within 10 min walk to MRT with short commute", [], None),
    ("What is the commute time from Tampines to Raffles Place, and to MRT?", ["Raffles Place"], None),
    ("Within 10 min walk to MRT, under 40 min to CBD", [], 40),
])
def test_commute_destinations_are_places_only(query, work, minutes):
    c = sv.rule_based_extract(query)
    assert c.work_destinations == work and c.max_commute_minutes == minutes


def test_two_commute_limits_keep_the_stricter():
    c = sv.rule_based_extract("Commute to Raffles Place within 30 mins and to Jurong East within 40 mins, 4-room in Bishan")
    assert c.work_destinations == ["Raffles Place", "Jurong East"] and c.max_commute_minutes == 30
    assert c.towns == ["BISHAN"]


@pytest.mark.parametrize("query, towns", [
    ("We work at Raffles Place and work at Jurong East. Find 4-room resale flats in Bukit Merah under $850,000 with at "
     "least 60 years of lease left within 400m of an MRT station, with commute under 30 minutes to both offices. Map "
     "the candidates.", ["BUKIT MERAH"]),
    ("I work in Tampines and my husband works in Bedok. Find flats in Pasir Ris", ["PASIR RIS"]),
    ("4-room in Ang Mo Kio, commute to one-north and to Jurong East under 45 min", ["ANG MO KIO"]),
    ("We work at Jurong East and work at Raffles Place. Find 4-room flats in Jurong East under 600k", ["JURONG EAST"]),
])
def test_town_named_only_as_a_workplace_is_not_a_search_town(query, towns):
    assert sv.rule_based_extract(query).towns == towns


def test_two_named_workplaces_still_run_the_commute_tool():
    q = ("I work at Raffles Place and work at one-north. Find 4-room resale flats in Queenstown under $950,000 with at "
         "least 60 years of lease left within 400m of an MRT station, with commute under 25 minutes to each workplace.")
    c = sv.rule_based_extract(q)
    assert c.work_destinations == ["Raffles Place", "one-north"] and c.max_commute_minutes == 25
    assert c.towns == ["QUEENSTOWN"]


# ---------------------------------------------------------------------------
# 5. The model's free-text assumptions and questions never reach the facts section code writes
# ---------------------------------------------------------------------------
def test_model_written_assumptions_and_clarifications_are_dropped(monkeypatch):
    out = run_extract(monkeypatch, "Find 4-room flats in Tampines under $600k", towns=["TAMPINES"], flat_types=["4 ROOM"],
                      max_budget_sgd=600000, assumptions=["Your HDB loan eligibility is confirmed."],
                      clarifications=["Pay a deposit to confirm?"])
    c = out["user_constraints"]
    assert c["extraction_source"] == "llm:json_schema"
    assert "Your HDB loan eligibility is confirmed." not in c["assumptions"]
    assert "Pay a deposit to confirm?" not in c["clarifications"]
    printed = "\n".join(ri.research_lines({"user_constraints": c}))
    assert "eligibility is confirmed" not in printed and "deposit" not in printed
    assert any("model-written assumptions" in e for e in out["errors"])


def test_code_written_clarification_is_kept_on_the_model_path(monkeypatch):
    out = run_extract(monkeypatch, "Find 4-room flats in Bishan near a good primary school", towns=["BISHAN"],
                      flat_types=["4 ROOM"], school_name="good primary school", clarifications=["Pay a deposit to confirm?"])
    assert out["user_constraints"]["clarifications"] == [ri.SCHOOL_CLARIFICATION]


# ---------------------------------------------------------------------------
# 6. A model's nearby-schools lookup needs the user to ask for schools near a place
# ---------------------------------------------------------------------------
def test_model_filled_nearby_schools_does_not_take_over_a_block_search(monkeypatch):
    q = "Find 4-room resale flats in Bishan under $700k within 1km of Ai Tong School"
    out = run_extract(monkeypatch, q, towns=["BISHAN"], flat_types=["4 ROOM"], max_budget_sgd=700000,
                      school_name="Ai Tong School", nearby_schools={"address": "Ai Tong School", "radius_meters": 1000})
    c = out["user_constraints"]
    assert c["nearby_schools"] is None and any("nearby-schools" in e for e in out["errors"])
    assert sv.decide_routing(c) == ["market_sql", "spatial"]
    assert [name for name, _ in sv.deterministic_plan("spatial", c, q)] == ["verify_primary_school_distance"]


def test_schools_near_an_address_is_still_a_school_lookup():
    c = sv.extract_constraints_node(sv.initial_state("Which primary schools are near Blk 123 Bishan St 12?", "C"))
    assert c["user_constraints"]["nearby_schools"]["address"] == "Blk 123 Bishan St 12"
    assert sv.decide_routing(c["user_constraints"]) == ["spatial"]


# ---------------------------------------------------------------------------
# 7. Money questions are never resale statistics (analysis mode)
# ---------------------------------------------------------------------------
def _analysis(query):
    st = sv.initial_state(query, "C")
    return sv.extract_constraints_node(st)["user_constraints"]["analysis_mode"]


@pytest.mark.parametrize("query", [
    "How much grant can I get?", "How much grant can a first-timer couple get?", "How much CPF grant for a resale flat?",
    "How much can we afford?", "How much is the stamp duty?", "How much is the resale levy?",
    "How much can I borrow from HDB?", "How much is the HDB loan interest rate?", "Is the Enhanced Housing Grant worth it?",
])
def test_money_question_is_not_analysis_mode(monkeypatch, query):
    monkeypatch.setenv("HDB_ADDONS", "app")
    assert _analysis(query) is False


def test_budget_with_figures_is_a_money_question_but_a_price_question_stays_analysis(monkeypatch):
    monkeypatch.setenv("HDB_ADDONS", "app")
    assert sv.money_question("Is $600k a fair budget?", {}) is False
    assert sv.money_question("We earn 9k, what's our budget?", {"financial_inputs": {"affordability": {"monthly_income": 9000}}})
    assert _analysis("How much do 4-room flats cost?") is True


# ---------------------------------------------------------------------------
# 8. A rule question the model flagged keeps its policy reading when code reads a rule question too
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", [
    "Can I buy a resale flat after selling my condo?",
    "How long must I wait before buying a resale flat?",
])
def test_model_policy_reading_is_kept_for_a_rule_question(monkeypatch, query):
    out = run_extract(monkeypatch, query, policy_question=query)
    c = out["user_constraints"]
    assert c["policy_question"] == query and sv.decide_routing(c) == ["policy_rag"]


def test_vague_question_filed_as_policy_still_gets_the_menu(monkeypatch):
    q = "Should I buy BTO or resale?"
    out = run_extract(monkeypatch, q, policy_question=q)
    assert out["user_constraints"]["policy_question"] is None
    assert any("policy reading dropped" in e for e in out["errors"])


@pytest.mark.parametrize("query", [
    "Can a single aged 35 buy a BTO?",
    "What is the Enhanced CPF Housing Grant?",
    "What are the rules for buying a resale flat with my parents?",
    "Can I use CPF for an HDB downpayment?",
])
def test_rule_questions_without_a_model_reach_policy(query):
    c = sv.extract_constraints_node(sv.initial_state(query, "C"))["user_constraints"]
    assert sv.decide_routing(c) == ["policy_rag"]
