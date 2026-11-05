"""Demo-readiness fixes for what only the live model showed (Member 4, 4 Nov; Qwen3-30B on the cluster, judge off:
data/local/qa/out/gpt_after_live.jsonl and gpt_after_live_grades.json). The model cannot run here, so each case is
replayed with a fake model double that writes the recorded opening or files the recorded reading. Every test uses the
tester's wording (data/local/qa/scenarios_gpt.json)."""
from __future__ import annotations

import os
from unittest.mock import patch

os.environ["LLM_PROFILE"] = "none"

from langchain_core.messages import AIMessage  # noqa: E402

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402


def conversation(*turns):
    """Each turn's final state with no model, every turn carrying the last (as the app does)."""
    states, prev = [], None
    for text in turns:
        prev = C.run(text, previous=prev)
        states.append(prev)
    return states


class Writer:
    """A stand-in model that always writes the same opening."""
    def __init__(self, text):
        self.text = text

    def invoke(self, msgs):
        return AIMessage(content=self.text)


def with_model_opening(state, text):
    """synthesize_node again on a no-model turn's state, the model writing `text` as its opening."""
    with patch.object(sv, "get_llm", return_value=Writer(text)):
        return sv.synthesize_node({**state, "draft_response": ""})


def opening(out):
    return (out.get("draft_response") or "").split("\n\n")[0]


class ReadingModel:
    """Structured output returns the given reading; the tool loops call nothing (code's plan runs); the writer writes
    `opening`."""
    def __init__(self, reading, opening_text=""):
        self.reading, self.opening_text = reading, opening_text

    def with_structured_output(self, schema, method=None, include_raw=False):
        reading = self.reading

        class Runner:
            def invoke(self, _messages):
                return {"parsed": schema(**reading), "raw": None}
        return Runner()

    def bind_tools(self, _tools, tool_choice=None):
        return self

    def invoke(self, _messages):
        return AIMessage(content=self.opening_text)


def live_turn(previous, text, reading, opening_text=""):
    """One turn with the live model's recorded reading (and opening), after no-model earlier turns."""
    model = ReadingModel(reading, opening_text)
    with patch.object(sv, "get_llm", lambda *a, **k: model):
        return C.graph.invoke(sv.initial_state(text, "C", None, previous))


def reading_of(previous, text, reading):
    with patch.object(sv, "get_llm", lambda *a, **k: ReadingModel(reading)):
        return sv.extract_constraints_node(sv.initial_state(text, "C", None, previous))


# ---------------------------------------------------------------------------
# A. Model opening checks
# ---------------------------------------------------------------------------
def test_a_ceiling_said_to_exclude_grants_that_it_includes_is_dropped():
    # HOUSE-02 t3: "The ceiling does not include grants" above a $704,472 ceiling that counts $105,000 of grants
    *_, third = conversation(
        "We are citizen first-timer buyers aged 30 and 32, no children, income $7,500, usable CPF $80k, cash $45k, no "
        "debts. Grants and HDB loan eligibility confirmed, no proximity grant. Buying 4-room resale with 70 years left; "
        "what's our budget?",
        "Yes, we both worked continuously for the last 12 months.",
        "Does that ceiling already include grants?")
    fin = third["financial_breakdown"]
    assert fin["calculate_maximum_affordability"]["grants_included"] and fin["calculate_cpf_housing_grants"]["total_grants"] > 0
    wrong = ("The ceiling does not include grants, as the grants have been calculated separately and are listed as part "
             "of the financial breakdown.")
    out = with_model_opening(third, wrong)
    assert "does not include grants" not in opening(out)
    assert any("does or does not include grants" in e for e in out["errors"])
    kept = with_model_opening(third, "Yes, the ceiling already includes your housing grants.")
    assert opening(kept).startswith("Yes, the ceiling already includes your housing grants.")


def test_a_ceiling_said_to_include_grants_when_none_were_counted_is_dropped():
    # the other way round: HOUSE-04's zero-grant budget ("Assume zero housing grants.")
    *_, second = conversation(
        "We are married citizens aged 37 and 35, second-time buyers, no children. Monthly income $10k, usable CPF $100k, "
        "cash $70k, no debts, loan eligibility confirmed. What can we afford with 70 years remaining?",
        "Assume zero housing grants.")
    assert second["financial_breakdown"]["calculate_maximum_affordability"]["status"] == "ok"
    out = with_model_opening(second, "This budget already includes your housing grants.")
    assert "already includes your housing grants" not in opening(out)
    kept = with_model_opening(second, "This budget does not include any housing grants.")
    assert opening(kept).startswith("This budget does not include any housing grants.")


def test_a_two_prs_are_not_told_they_may_get_a_housing_grant():
    # POLICY-03 t2: "They may still be eligible for the Proximity Housing Grant" for two PRs
    _, second = conversation("Can a citizen and PR couple get resale housing grants?", "What if both buyers are PRs?")
    assert second["user_constraints"]["pr_couple"]
    live = ("If both buyers are PRs, they are not eligible for the CPF Family Grant. They may still be eligible for the "
            "Proximity Housing Grant, which has no income ceiling and is available to both first-timers and second-timers.")
    out = with_model_opening(second, live)
    assert "may still be eligible" not in out["draft_response"]
    # code's rule leads, and the model's sentences do not follow it (5 Nov rerun: the live model opened with EIP)
    assert opening(out).startswith(sv.PR_COUPLE_OPENING)
    assert "Proximity Housing Grant" not in opening(out)


def test_a_mrt_nearness_is_dropped_when_the_facts_give_no_mrt_distance():
    # STATS-06 t1/t2: area statistics around a block; the only distances are to the user's block
    first, second = conversation("Show the price distribution for 4-room within 500m of Blk 510 Bedok North St 3.",
                                 "Use a 1km area instead.")
    for state, live in ((first, "Blk 523 is cheaper but further away, and Blk 524 has the same price as Blk 517 but "
                                "slightly further from the MRT station."),
                        (second, "Blk 515 offers a slightly higher price but similar proximity to the MRT station.")):
        out = with_model_opening(state, live)
        assert "MRT station" not in opening(out)
        assert any("a place the facts give no distance for" in e for e in out["errors"])


def test_a_mrt_nearness_stays_when_the_facts_list_mrt_distances():
    state = C.run("4-room in Tampines under $600k within 500m of an MRT station.")
    assert "mrt" in state["draft_response"].lower()
    out = with_model_opening(state, "The listed blocks are all close to an MRT station.")
    assert not any("a place the facts give no distance for" in e for e in out["errors"])


def test_a_policy_opening_about_subsidy_recovery_for_an_interest_rate_question_is_dropped():
    # POLICY-06 t2: the 3-6% Plus subsidy recovery called "the actual rate"
    _, second = conversation("What is the difference between an HDB loan and a bank loan?",
                             "Is the actual rate the same as the affordability assessment rate?")
    live = ("No, the actual rate for subsidy recovery upon resale is 3% to 6% on the higher of the resale price or "
            "market valuation, which differs from the affordability assessment rate.")
    out = with_model_opening(second, live)
    assert "subsidy recovery" not in opening(out)
    assert sv.sibling_terms(live, "Is the actual rate the same as the affordability assessment rate?") == {("rate", "recovery")}
    # a question about the clawback is not answered with the loan's interest either, and the same family stays
    assert sv.sibling_terms("The HDB loan interest rate is 2.6%.", "What is the subsidy clawback on a Plus flat?")
    assert not sv.sibling_terms("The clawback is 3% to 6% of the resale price.", "What is the subsidy clawback on a Plus flat?")


def test_a_block_choice_filler_is_dropped_when_no_block_is_listed():
    # DEMO-12 t2: the filler above a loan instalment
    _, second = conversation("Monthly repayment on a $400,000 loan at 3% for 25 years?", "What if I repay over 20 years instead?")
    out = with_model_opening(second, "The choice between price and convenience depends on the specific blocks available.")
    assert "price and convenience" not in out["draft_response"]
    assert out["draft_response"].startswith("**Monthly instalment: $2,218**")
    # a block answer keeps the trade-off the prompt's example teaches
    state = C.run("4-room in Tampines under $600k within 500m of an MRT station.")
    assert not sv.block_filler("The choice is between price and convenience.", state["draft_response"])


# ---------------------------------------------------------------------------
# B. Live reading drift and extra lookups
# ---------------------------------------------------------------------------
def test_b_a_generic_school_phrase_keeps_the_school_named_earlier():
    # SPATIAL-01 t2: the model filed "school distance" as the school; the answer could not tell which school
    first = C.run("Find 4-room Bishan below 950k within 1km of Ai Tong School.")
    out = reading_of(first, "Expand the school distance to 2km.",
                     dict(amenity_radii_meters={"school": 2000}, school_name="school distance"))
    c = out["user_constraints"]
    assert c["school_name"] == first["user_constraints"]["school_name"] and c["amenity_radii_meters"]["school"] == 2000
    state = live_turn(first, "Expand the school distance to 2km.",
                      dict(amenity_radii_meters={"school": 2000}, school_name="school distance"))
    assert "could not tell which school" not in state["draft_response"]
    assert any(r["tool"] == "verify_primary_school_distance" and r["arguments"]["bracket"] == "2km"
               for r in state["tool_call_log"])
    # removing the school limit, or naming another school, is still read as said
    dropped = reading_of(first, "Drop the school limit.", {})["user_constraints"]
    assert not dropped.get("school_name")
    other = reading_of(first, "Use Rosyth School instead.", dict(school_name="Rosyth School"))["user_constraints"]
    assert other["school_name"] == "Rosyth School"


def test_b_statistics_call_keeps_the_stated_period():
    # STATS-03 t2: the model's call left out "since 2020" and the table covered the last 12 months
    errors = []
    c = {"stats_since_year": 2020, "towns": ["TAMPINES"], "flat_types": ["4 ROOM"]}
    args = sv.statistics_args({"group_by": ["storey_band"], "towns": ["TAMPINES"], "flat_types": ["4 ROOM"],
                               "lookback_years": None}, c, errors, "market_sql")
    assert args["start_month"] == "2020-01" and "lookback_years" not in args
    assert any("2020-01" in e for e in errors)
    args = sv.statistics_args({"group_by": ["town"], "start_month": "2025-10"}, {"stats_lookback_years": 5}, [], "market_sql")
    assert args["lookback_years"] == 5 and "start_month" not in args
    # a call that already has the period is left as it is
    same = {"group_by": ["storey_band"], "start_month": "2020-01"}
    assert sv.statistics_args(same, c, [], "market_sql") == same
    first = C.run("How have 4-room prices changed by storey since 2020?")
    assert first["user_constraints"]["stats_since_year"] == 2020


def test_b_instalment_turn_drops_the_models_policy_reading():
    # DEMO-11 t1/t2: an EHG income tier opened a repayment answer and loan rules followed it
    t1 = live_turn(None, "Monthly repayment on a $500,000 loan at 2.6% for 25 years?",
                   dict(policy_question="Monthly repayment on a $500,000 loan at 2.6% for 25 years?"))
    assert not t1["user_constraints"].get("policy_question") and "policy_rag" not in t1["active_subagents"]
    assert t1["draft_response"].startswith("**Monthly instalment: $2,268**") and "**Policy:**" not in t1["draft_response"]
    t2 = live_turn(t1, "What if I repay over 20 years instead?", dict(policy_question="What if I repay over 20 years instead?"))
    assert not t2["user_constraints"].get("policy_question") and "**Policy:**" not in t2["draft_response"]
    assert "**Monthly instalment: $2,674**" in t2["draft_response"]


def test_b_a_corrected_stamp_duty_price_is_not_a_budget():
    # MONEY-07/08/09/12 t2: the corrected price read as a budget brought the grant and budget questionnaire
    for first, second, price, duty in (
            ("BSD only for a residential purchase of $180,000, valuation equals price.", "Change price and valuation to $360,000.", 360_000, "$5,400"),
            ("BSD only for a residential purchase of $360,000, valuation equals price.", "Change price and valuation to $500,000.", 500_000, "$9,600"),
            ("BSD only for a residential purchase of $650,000, valuation equals price.", "Change price and valuation to $700,000.", 700_000, "$15,600"),
            ("BSD only for a residential purchase of $3,000,000, valuation equals price.", "Change price and valuation to $3,200,000.", 3_200_000, "$131,600")):
        t1 = C.run(first)
        reading = dict(max_budget_sgd=price, asking_price_sgd=price,
                       financial_inputs={"upfront": {"purchase_price": price, "market_valuation": price}})
        out = reading_of(t1, second, reading)
        c = out["user_constraints"]
        assert not c.get("max_budget_sgd") and not c["search_stated"]
        assert any("model's budget dropped" in e for e in out["errors"])
        state = live_turn(t1, second, reading)
        text = state["draft_response"]
        assert f"Buyer's stamp duty: **{duty}**" in text
        assert "Housing grants" not in text and "To finish your grant and budget estimate" not in text


def test_b_a_stated_budget_with_a_price_is_still_a_budget():
    # the guard needs the duty-only or price-correction wording: a search budget the user states stays
    out = reading_of(None, "4-room in Tampines under $600k.",
                     dict(towns=["TAMPINES"], flat_types=["4 ROOM"], max_budget_sgd=600_000))
    assert out["user_constraints"]["max_budget_sgd"] == 600_000


def test_b_stamp_duty_on_the_ceiling_keeps_the_budget():
    # HOUSE-04 t3: the duty is worked out on the ceiling, and the ceiling stays in the answer
    *_, third = conversation(
        "We are married citizens aged 37 and 35, second-time buyers, no children. Monthly income $10k, usable CPF $100k, "
        "cash $70k, no debts, loan eligibility confirmed. What can we afford with 70 years remaining?",
        "Assume zero housing grants.", "Does stamp duty fit within that number?")
    text = third["draft_response"]
    assert "Max purchase price $680,000" in text and "Buyer's stamp duty: **$15,000**" in text


def test_b_models_flat_type_and_category_are_kept_only_when_stated():
    # CLARIFY-03 t2: "4 ROOM" filled in silently; HOUSE-06 t2: "bto" refused the grants as a BTO case
    t1 = C.run("What grants can we get?")
    out = reading_of(t1, "We are first-timers.",
                     dict(financial_inputs={"grants": {"flat_category": "resale", "first_timer_status": True, "flat_type": "4 ROOM"}}))
    c = out["user_constraints"]
    assert sv.FOUR_ROOM_GRANT_NOTE in (c.get("assumptions") or [])
    h1 = C.run("We are married citizen first-timers aged 31 and 32, household income $8k, CPF $90k and cash $50k. Help with grants.")
    reading = dict(financial_inputs={"grants": {"flat_category": "bto", "eligibility_confirmed": False}})
    state = live_turn(h1, "HDB has not confirmed our grant eligibility.", reading)
    grants = state["user_constraints"]["financial_inputs"]["grants"]
    assert grants.get("flat_category") != "bto"
    assert "BTO and second-timer/mixed households require separate assessment" not in state["draft_response"]
    # stated in the user's words: kept
    assert sv.flat_choice_stated("flat_category", "bto", "We are applying for a BTO flat.")
    assert sv.flat_choice_stated("flat_type", "4 ROOM", "a four-room flat")
    assert sv.flat_choice_stated("flat_type", "4 ROOM", "3 to 5 room flats")
    assert not sv.flat_choice_stated("flat_type", "4 ROOM", "We are first-timers.")
    top = reading_of(None, "We are first-timers. What grants can we get?", dict(flat_types=["4 ROOM"]))
    assert not top["user_constraints"].get("flat_types")


def test_b_no_rule_line_under_an_answered_calculation_question():
    # HOUSE-03 t3: "I couldn't find an HDB rule ..." under the bank-loan answer
    *_, third = conversation(
        "Both of us are PRs aged 32 and 33, first-time buyers with no children. Income $9,000, usable CPF $120k and cash "
        "$60k. No debt; our HDB loan eligibility is supposedly confirmed. We want a 4-room resale with 70 years "
        "remaining. What's our maximum HDB-loan budget?",
        "Neither of us is a citizen; we are both PRs. Can you really use an HDB loan?",
        "What information would you need for a bank-loan calculation instead?")
    state = {**third, "policy_no_match": True, "rag_policy_passages": [],
             "user_constraints": {**third["user_constraints"], "policy_question": "Can you really use an HDB loan?"}}
    assert third["financial_breakdown"]
    assert sv.NO_POLICY_ANSWER not in sv.template_response(state)
    # a rule question of this message's own that the search found nothing for still says so
    rule = {**state, "messages": state["messages"][:-1] + [{"role": "user", "content": "Can PRs take an HDB loan?"}],
            "user_constraints": {**state["user_constraints"], "policy_question": "Can PRs take an HDB loan?"}}
    assert sv.NO_POLICY_ANSWER in sv.template_response(rule)


def test_b_coverage_check_adds_no_block_search_to_a_price_history_question(monkeypatch):
    # LIMIT-03 t2: the model fetched the trend; the backstop added a $5,000,000 town-wide block search above it
    calls = [{"name": "compute_town_price_timeseries", "id": "t1",
              "args": {"towns": ["BEDOK"], "flat_type": "4 ROOM", "lookback_years": 3}}]

    class ToolModel(ReadingModel):
        def __init__(self):
            super().__init__({})
            self.n = 0

        def invoke(self, _messages):
            self.n += 1
            return AIMessage(content="", tool_calls=calls if self.n == 1 else [])

    monkeypatch.setattr(sv, "get_llm", lambda *a, **k: ToolModel())
    c = {"towns": ["BEDOK"], "flat_types": ["4 ROOM"], "wants_price_trend": True, "search_stated": True}
    tools = [n for n in sv.tools_for_agent("market_sql")]
    q = "Then show historical 4-room prices in Bedok."
    _, log, errors, _ = sv.run_react("market_sql", tools, "sys", q, c)
    assert not any(r["tool"] == "query_resale_blocks_by_criteria" for r in log)
    assert not any("added query_resale_blocks_by_criteria" in e for e in errors)
    # with a stated limit the skipped block search is still added
    _, log, errors, _ = sv.run_react("market_sql", tools, "sys", q, {**c, "max_budget_sgd": 600_000})
    assert any(r["tool"] == "query_resale_blocks_by_criteria" and r["agent"] == "supervisor" for r in log)


# ---------------------------------------------------------------------------
# "Can we afford Blk 264 Bishan St 24?" for buyers aged 27 (user's screen, 4 Nov): 64.8 years of lease left
# ---------------------------------------------------------------------------
BISHAN_FACTS = ("We are both Singapore citizens. We are first-time buyers. We are aged 27 and 27. We have no children. "
                "Our household income is $7,000 a month. Our CPF OA balance is $30,000. We have $150,000 in cash savings. "
                "We have no monthly debt. We have both worked continuously for the past 12 months. We are not buying "
                "near our parents. Our grant eligibility is confirmed. Our HDB loan eligibility is confirmed.")


def test_a_short_lease_block_is_explained_once_and_clearly():
    _, st = conversation("Can we afford Blk 264 Bishan St 24?", BISHAN_FACTS)
    text = st["draft_response"]
    assert text.startswith("I can't give an affordability verdict for Blk 264 BISHAN ST 24")
    assert "is in line with the nearby median" not in text          # no verdict of the assumed price against itself
    assert text.count("stamp duty on this price") == 1 and "Upfront funding" not in text
    assert "more information needed" not in text and not st.get("clarification")
    assert "\n\n**Financials:**" in text and "\n\n**Can you afford" in text
    assert "The ceiling excludes" not in text                         # no ceiling was given


def test_the_price_check_takes_only_a_stated_asking_price():
    errors = []
    held = sv.search_args("assess_resale_price", {"address": "Blk 264 Bishan St 24", "asking_price": 840444.0},
                          {"asking_price_sgd": None}, errors, "market_sql")
    assert held["asking_price"] is None and errors
    kept = sv.search_args("assess_resale_price", {"address": "Blk 264 Bishan St 24", "asking_price": 800000.0},
                          {"asking_price_sgd": 800000.0}, [], "market_sql")
    assert kept["asking_price"] == 800000.0


def test_a_school_made_up_from_the_address_is_dropped():
    # STATS-06 live (4 Nov): the model read "BEDOK PRIMARY SCHOOL" from "Blk 510 Bedok North St 3" and a school search
    # put a 65-block list in front of the area statistics
    c = {"school_name": "BEDOK PRIMARY SCHOOL", "amenity_radii_meters": {"school": 1000}}
    sv._drop_unstated(c, "Show the price distribution for 4-room within 500m of Blk 510 Bedok North St 3.", [])
    assert c["school_name"] is None
    c = {"school_name": "Ai Tong School"}
    sv._drop_unstated(c, "4-room in Bishan within 1km of Ai Tong", [])
    assert c["school_name"] == "Ai Tong School"


def test_a_duty_only_turn_offers_only_the_duty_calculator():
    # MONEY-08 t2 live: the model called the affordability calculator on "Change price and valuation to $500,000."
    c = {"duty_only": True}
    assert not sv.TASK_TOOLS["calculate_maximum_affordability"](c)
    assert not sv.TASK_TOOLS["calculate_cpf_housing_grants"](c)
    assert sv.TASK_TOOLS["calculate_maximum_affordability"]({})


class GrantsOnlyModel(ReadingModel):
    """The live model of HOUSE-04 t3: its financial agent calls only the grant calculator, then stops."""
    def __init__(self):
        super().__init__({})
        self.called = False

    def bind_tools(self, _tools, tool_choice=None):
        outer = self

        class Bound:
            def invoke(self, messages):
                if not outer.called and any("calculate_cpf_housing_grants" == getattr(t, "name", t) for t in _tools or []) \
                        or not outer.called:
                    outer.called = True
                    return AIMessage(content="", tool_calls=[{"name": "calculate_cpf_housing_grants", "id": "g1", "args": {}}])
                return AIMessage(content="")
        return Bound()


def test_a_duty_question_gets_the_duty_when_the_model_skips_it():
    # HOUSE-04 t3 live (4 Nov): "Does stamp duty fit within that number?" got no stamp duty at all
    turns = ["We are married citizens aged 37 and 35, second-time buyers, no children. Monthly income $10k, usable CPF "
             "$100k, cash $70k, no debts, loan eligibility confirmed. What can we afford with 70 years remaining?",
             "Assume zero housing grants."]
    prev = conversation(*turns)[-1]
    with patch.object(sv, "get_llm", lambda *a, **k: GrantsOnlyModel()):
        st = C.graph.invoke(sv.initial_state("Does stamp duty fit within that number?", "C", None, prev))
    assert "stamp duty" in st["draft_response"].lower() and "$15,000" in st["draft_response"], st["draft_response"][:400]
