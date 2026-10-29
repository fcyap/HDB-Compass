"""
Calculator facts come from the user's words, not the model's reading of them (Member 4). The 27 Oct live run showed
the model turning a married citizen couple into SC_Single, inventing ages and an interest rate, passing zero grants
and filling an empty financial_inputs that routed every search to the finance agent. No model needed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["LLM_PROFILE"] = "none"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import supervisor as sv  # noqa: E402
from evaluation.ground_category_e import SPECS, financial_facts, question_text  # noqa: E402

ITEMS = {i["id"]: i for i in json.loads((ROOT / "evaluation" / "category_e_queries.json").read_text())}


def test_every_category_e_question_parses_to_its_ground_truth_facts():
    for spec in SPECS:
        want, got = financial_facts(spec), sv.stated_financial_inputs(question_text(spec))
        if want is None:
            assert got == {}, question_text(spec)        # search-only questions state no calculator facts
            continue
        for sec, facts in want.items():
            assert got[sec] == facts, (sec, question_text(spec))


def test_nothing_is_inferred():
    got = sv.stated_financial_inputs("We are a couple earning $9,000 a month with $80,000 in CPF OA.")
    assert "citizenship_pair" not in got["grants"]       # "couple" alone states no citizenship
    assert "youngest_buyer_age" not in got["grants"] and "proximity" not in got["grants"]
    aged = sv.stated_financial_inputs("We are a Singapore citizen couple aged 30 and 28 earning $9,000 a month.")
    assert "youngest_core_member_age" not in aged["grants"]  # children not ruled out
    assert aged["grants"]["citizenship_pair"] == "SC_SC"


def test_follow_up_answers_complete_the_facts():
    for qid, field, value in (("E1-07", "cash_savings", 25000.0), ("E1-08", "youngest_buyer_age", 30)):
        before = sv.stated_financial_inputs(ITEMS[qid]["query"])["affordability"]
        after = sv.stated_financial_inputs(ITEMS[qid]["query"] + " " + ITEMS[qid]["follow_up"]["answer"])["affordability"]
        assert field not in before and after[field] == value


def test_model_values_for_facts_are_dropped():
    model = {"grants": {"citizenship_pair": "SC_Single", "proximity": None, "flat_category": "resale"},
             "affordability": {"interest_rate": 2.6, "tenure_years": 25, "total_grants": 0.0, "youngest_buyer_age": 35},
             "upfront": {"purchase_price": 650000.0, "legal_and_valuation_fees": 10000.0}}
    stated = sv.stated_financial_inputs(ITEMS["E1-01"]["query"])
    out = sv.trusted_financial_inputs(model, stated)
    assert out["grants"]["citizenship_pair"] == "SC_SC" and out["grants"]["proximity"] == "none"
    assert not {"interest_rate", "tenure_years", "total_grants"} & set(out["affordability"])
    assert out["affordability"]["youngest_buyer_age"] == 35   # stated ("aged 36 and 35"), so kept from the text
    assert "upfront" not in out                              # the user named no purchase to cost
    assert sv.trusted_financial_inputs({"grants": None, "affordability": None, "upfront": None}, {}) is None


def test_empty_financial_inputs_do_not_route_to_finance():
    c = {"towns": ["TAMPINES"], "max_budget_sgd": 700000.0, "financial_inputs": {"grants": None, "affordability": None}}
    assert "financial" not in sv.decide_routing(c)
    c["financial_inputs"] = {"grants": {"monthly_income": 9000.0}}
    assert "financial" in sv.decide_routing(c)


def test_guard_overrides_model_arguments_and_passes_grants_forward():
    c = {"financial_inputs": sv.stated_financial_inputs(ITEMS["E1-01"]["query"])}
    calls = []

    def grants(**kw):
        calls.append(kw)
        return {"status": "ok", "total_grants": 90000.0}

    outputs = {}
    args = sv.guard_financial_args("calculate_maximum_affordability",
                                   {"monthly_income": 8500, "interest_rate": 2.6, "total_grants": 0.0, "cash_savings": 1.0},
                                   c, outputs, {"calculate_cpf_housing_grants": grants})
    assert len(calls) == 1 and calls[0]["citizenship_pair"] == "SC_SC"   # grants run first, from stated facts
    assert args["total_grants"] == 90000.0 and "interest_rate" not in args and args["cash_savings"] == 30000.0
    assert sv.guard_financial_args("query_resale_blocks_by_criteria", {"towns": ["X"]}, c, {}, {}) == {"towns": ["X"]}


def test_follow_up_answer_addresses_the_original_request():
    state = {"messages": [{"role": "user", "content": "Find flats and tell us our grants."},
                          {"role": "assistant", "content": "Please tell me your ages."},
                          {"role": "user", "content": "We are aged 32 and 30."}],
             "pending_clarification": {"needed": ["youngest_buyer_age"]}}
    req = sv.user_request(state)
    assert req.startswith("Find flats and tell us our grants.") and "We are aged 32 and 30." in req
    assert sv.user_request({"messages": [{"role": "user", "content": "Find flats."}]}) == "Find flats."


def test_unrequested_specialist_results_are_not_evidence():
    c = {"towns": ["BUKIT MERAH"], "flat_types": ["4 ROOM"], "amenity_radii_meters": {"mrt": 400, "hawker": 500}}
    outputs = {"filter_blocks_by_amenity_proximity": [{"block": "14"}], "verify_primary_school_distance": [],
               "compute_dual_transit_commute": []}
    kept = sv.requested_outputs("spatial", outputs, c, "")
    assert list(kept) == ["filter_blocks_by_amenity_proximity"]          # school "null" and blank commute dropped
    market = sv.requested_outputs("market_sql", {"query_resale_blocks_by_criteria": {}, "query_comparable_resale_transactions": {}}, c, "")
    assert list(market) == ["query_resale_blocks_by_criteria"]           # no comparable sales asked for
    assert sv.requested_outputs("financial", {"x": 1}, c, "") == {"x": 1}


def test_writer_is_told_to_list_every_match():
    blocks = [{"block": str(n), "street": "TOH GUAN RD"} for n in range(8)]
    _, rule = sv.synthesis_evidence({"variant": "C", "candidate_blocks_intersection": blocks})
    assert "List all 8 blocks" in rule
    _, rule = sv.synthesis_evidence({"variant": "C", "candidate_blocks_intersection": []})
    assert "List all" not in rule


def test_amounts_without_a_dollar_sign_are_read():
    q = ("we are a couple that's not married and one of us is a uni student while another is working full time for "
         "2 years. We are looking to apply for 4 room btos. We have a combined savings of around 150k cash and cpf of ard 40k sgd")
    assert sv.stated_financial_inputs(q)["affordability"] == {"cpf_oa_balance": 40000.0, "cash_savings": 150000.0}
    both = sv.stated_financial_inputs("we have $90,000 in CPF OA we can use, plus $40,000 in cash savings")["affordability"]
    assert both["cpf_oa_balance"] == 90000.0 and both["cash_savings"] == 40000.0


def test_zero_distance_limits_are_not_constraints():
    from unittest.mock import patch

    class Stub:
        def with_structured_output(self, schema, method=None, include_raw=False):
            return self

        def invoke(self, _msgs):
            from agents.schemas import ExtractedConstraints
            return {"parsed": ExtractedConstraints(amenity_radii_meters={"mrt": 0, "hawker": 0, "park": 400})}

    with patch.object(sv, "get_llm", return_value=Stub()):
        out = sv.extract_constraints_node({"messages": [{"role": "user", "content": "Can we apply for a BTO?"}]})
    assert out["user_constraints"]["amenity_radii_meters"] == {"park": 400}


def test_never_ask_the_user_for_the_grant_total():
    clar = sv.clarification_needed({"financial_breakdown": {
        "calculate_cpf_housing_grants": {"status": "needs_information", "missing_inputs": ["citizenship_pair"]},
        "calculate_maximum_affordability": {"status": "needs_information", "missing_inputs": ["total_grants", "cash_savings"]}}})
    assert "total_grants" not in clar["needed"] and {"citizenship_pair", "cash_savings"} <= set(clar["needed"])


def test_short_eligibility_phrasing_is_read():
    got = sv.stated_financial_inputs("We have $9,000 a month income and our grant and HDB loan eligibility is confirmed.")
    assert got["grants"]["eligibility_confirmed"] is True and got["affordability"]["loan_eligibility_confirmed"] is True


def test_model_reading_kept_when_the_words_support_it():
    q = ("hey, i'm single, a singapore citizen, 38, first timer, no kids. household income 5.5k a month (HFE assessed), "
         "110k in cpf oa we can use, plus 50k cash. no other loans, been working continuously for the last 12 months, "
         "and our grant and hdb loan eligibility is already confirmed. no proximity grant.")
    model = {"grants": {"citizenship_pair": "SC_Single", "youngest_buyer_age": 38, "youngest_core_member_age": 38,
                        "monthly_income": 5500.0, "eligibility_confirmed": True, "proximity": "none"},
             "affordability": {"oldest_buyer_age": 38, "cash_savings": 50000.0, "cpf_oa_balance": 110000.0,
                               "interest_rate": 2.6, "total_grants": 0.0}}
    out = sv.trusted_financial_inputs(model, sv.stated_financial_inputs(q), q)
    g, a = out["grants"], out["affordability"]
    assert g["citizenship_pair"] == "SC_Single" and g["youngest_buyer_age"] == 38 and g["youngest_core_member_age"] == 38
    assert a["youngest_buyer_age"] == 38 and a["average_applicant_age"] == 38   # spread to both calculators
    assert a["cash_savings"] == 50000.0 and a["monthly_income"] == 5500.0
    assert "interest_rate" not in a and "total_grants" not in a                  # not in the user's words


def test_invented_facts_are_still_dropped():
    q = "We are a married Singapore citizen couple earning $9,000 a month with $90,000 in CPF OA."
    model = {"grants": {"citizenship_pair": "SC_Single", "youngest_buyer_age": 35, "eligibility_confirmed": True,
                        "proximity": "none", "first_timer_status": True},
             "affordability": {"cash_savings": 40000.0, "monthly_debt_obligations": 0.0, "monthly_income": 9000.0}}
    out = sv.trusted_financial_inputs(model, sv.stated_financial_inputs(q), q)
    g, a = out["grants"], out["affordability"]
    assert g["citizenship_pair"] == "SC_SC"          # code's reading wins; SC_Single has no support anyway
    for k in ("youngest_buyer_age", "eligibility_confirmed", "proximity", "first_timer_status"):
        assert k not in g, k
    assert "cash_savings" not in a and "monthly_debt_obligations" not in a and a["monthly_income"] == 9000.0


def test_numbers_match_in_any_form():
    for text, value in (("150k cash", 150000), ("budget $0.65m", 650000), ("650 thousand", 650000),
                        ("$8,500 a month", 8500), ("8.5k/mth", 8500)):
        assert sv._number_stated(value, text), text
    assert not sv._number_stated(32, "we have 32k in cash", plain_only=True)   # an age is a plain number


def test_one_bad_field_does_not_discard_the_reading():
    data = {"towns": ["TOA PAYOH"], "max_budget_sgd": 750000, "comparable_resale": {"flat_type": ""},
            "financial_inputs": {"grants": {"monthly_income": 7800, "youngest_buyer_age": 5}}}
    c, dropped = sv.lenient_constraints(data)
    assert c.towns == ["TOA PAYOH"] and c.max_budget_sgd == 750000          # the good fields survive
    assert "comparable_resale" in dropped and "financial_inputs.grants.youngest_buyer_age" in dropped
    assert c.financial_inputs.grants.monthly_income == 7800                   # only the bad fact went


def test_reply_that_is_not_json_gets_one_repair():
    from langchain_core.messages import AIMessage

    class Model:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema, method=None, include_raw=False):
            outer = self

            class S:
                def invoke(self, _m):
                    outer.calls += 1
                    return {"raw": AIMessage(content="Sure! towns: Bishan"), "parsed": None, "parsing_error": ValueError("x")}
            return S()

        def invoke(self, _m):
            return AIMessage(content='```json\n{"towns": ["BISHAN"], "flat_types": ["4 ROOM"]}\n```')

    c, notes = sv.read_constraints_with_model(Model(), [])
    assert c.towns == ["BISHAN"] and c.flat_types == ["4 ROOM"]
    assert any("repaired" in n for n in notes)


def test_c_answer_facts_are_written_by_code():
    from unittest.mock import patch
    from langchain_core.messages import AIMessage
    from agents import arch_3_specialist_5agent as C
    item = ITEMS["E1-03"]
    state = C.graph.invoke(sv.initial_state(item["query"], "C"))           # no model: the template answer
    seen = {}

    class Writer:
        def invoke(self, msgs):
            seen["prompt"] = msgs[-1].content
            return AIMessage(content="Both Ang Mo Kio blocks fit. Blk 627 is cheaper at $600,000.")   # a figure: dropped

    with patch.object(sv, "get_llm", return_value=Writer()):
        out = sv.synthesize_node({**state, "draft_response": ""})
    facts = sv.template_response(state)
    assert out["draft_response"].startswith("Both Ang Mo Kio blocks fit.")
    assert "$600,000" not in out["draft_response"]                          # the sentence with a wrong price is dropped
    assert facts in out["draft_response"]                                  # the facts section is the code's, verbatim
    assert "will be shown to the user exactly as written" in seen["prompt"]
    with patch.object(sv, "get_llm", return_value=Writer()):
        b = sv.synthesize_node({**state, "variant": "B", "draft_response": ""})
    assert facts not in b["draft_response"]                                # B keeps the model as the writer


def test_opening_states_no_figures():
    facts = "- Blk 627 ANG MO KIO AVE 9 — median $655,000\n- Grants = **$85,000**"
    text, dropped = sv.keep_supported_sentences(
        "Blk 627 fits best. It costs about $655k. Blk 999 is also nice. Blk 627 Ang Mo Kio Ave 9 is near the MRT.", facts)
    assert text == "Blk 627 fits best. Blk 627 Ang Mo Kio Ave 9 is near the MRT." and dropped == 2


# --- regressions found by the 30 Oct adversarial review -------------------------------------------------------------

def test_a_comma_before_cash_or_cpf_is_not_part_of_the_amount():
    aff = lambda t: sv.stated_financial_inputs(t)["affordability"]  # noqa: E731
    assert aff("Income 9,000, cash 50k")["cash_savings"] == 50000
    assert aff("Our income is 8000, CPF OA 90k")["cpf_oa_balance"] == 90000
    assert aff("We're 32 & 30, cash 40k, cpf 90k")["cash_savings"] == 40000


def test_a_later_message_corrects_an_earlier_fact():
    st = sv.extract_constraints_node({"messages": [
        {"role": "user", "content": "SC couple, first timers, income 9k, cash 40k, cpf 100k, aged 32 and 30, no kids"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "Sorry, correction: cash 60k and cpf 150k"}]})
    a = st["user_constraints"]["financial_inputs"]["affordability"]
    assert a["cash_savings"] == 60000 and a["cpf_oa_balance"] == 150000 and a["youngest_buyer_age"] == 30


def test_pr_status_only_when_a_buyer_is_a_pr():
    pair = lambda t: (sv.stated_financial_inputs(t).get("grants") or {}).get("citizenship_pair")  # noqa: E731
    assert pair("Both are Singapore citizens, no PR") == "SC_SC"
    assert pair("I'm a single Singapore citizen; my parents are PR") == "SC_Single"
    assert pair("I'm Singaporean, my wife is Malaysian, not a PR") is None
    for t in ("one Singapore citizen and one permanent resident", "SC + PR couple", "i'm a singapore citizen and my husband is a PR"):
        assert pair(t) == "SC_PR", t


def test_ages_are_the_buyers_own():
    assert sv.buyer_ages("We are 30 minutes from work, aged 32 and 30") == [32, 30]
    assert sv.buyer_ages("Our kids are 10 and 8. I'm 40.") == [40]
    assert sv.buyer_ages("Our kids are 10 and 8. We're 40.") == []      # a couple with one age: asked, not guessed
    assert sv.buyer_ages("Our parents are aged 65 and 62, we are 30 and 28") == [30, 28]
    assert sv.buyer_ages("I'm 45 and my husband is 52") == [45, 52]


def test_negated_first_timer_is_not_a_first_timer():
    ft = lambda t: sv.stated_financial_inputs(t)["grants"]["first_timer_status"]  # noqa: E731
    assert ft("We're not first-timers, we sold our 3-room") is False
    assert ft("We are no longer first timers") is False
    assert ft("both 1st timers") is True


def test_search_patterns_do_not_misread_money_or_repayments():
    assert sv.rule_based_extract("We have 1.2m to spend near an MRT in Bishan").amenity_radii_meters == {}
    assert sv.rule_based_extract("We can afford $2,500 per month for the mortgage").household_income is None
    for t in ("My budget is 600k for a 4-room in Bishan", "650k budget, 4rm Bishan", "Budget: $650,000", "budget around 650k"):
        assert sv.rule_based_extract(t).max_budget_sgd in (600000, 650000), t


def test_grounding_needs_the_number_beside_its_own_fact():
    q7, q8 = ITEMS["E1-07"]["query"], ITEMS["E1-08"]["query"]
    assert not sv.fact_is_grounded("cash_savings", 80000, q7)          # the $80,000 is CPF, not cash
    assert sv.fact_is_grounded("cpf_oa_balance", 80000, q7)
    assert not sv.fact_is_grounded("youngest_buyer_age", 65, q8)       # 65 is the lease, not an age


def test_questions_requests_and_negations_do_not_state_a_fact():
    for field, text in (("eligibility_confirmed", "Can you confirm if we are eligible for the grants?"),
                        ("loan_eligibility_confirmed", "Please confirm whether we are eligible for an HDB loan."),
                        ("first_timer_status", "This is not our first home; we sold our flat in 2020."),
                        ("continuous_employment_12_months", "I started a new job last year after 6 months unemployed.")):
        assert not sv.fact_is_grounded(field, True, text), text
    assert sv.fact_is_grounded("eligibility_confirmed", True, "Our grant and HDB loan eligibility is confirmed.")


def test_the_models_own_reading_is_kept_before_code_fills_it():
    from unittest.mock import patch
    from agents.schemas import ExtractedConstraints

    class Stub:
        def with_structured_output(self, schema, method=None, include_raw=False):
            return self

        def invoke(self, _msgs):
            return {"parsed": ExtractedConstraints(towns=["TOA PAYOH"])}       # the model reads only the town

    q = "Find 4-room resale flats in Toa Payoh under $750,000"
    with patch.object(sv, "get_llm", return_value=Stub()):
        out = sv.extract_constraints_node({"messages": [{"role": "user", "content": q}]})
    assert out["model_constraints"]["towns"] == ["TOA PAYOH"] and out["model_constraints"]["max_budget_sgd"] is None
    assert out["user_constraints"]["max_budget_sgd"] == 750000                # code filled the gap afterwards
    assert sv.extract_constraints_node({"messages": [{"role": "user", "content": q}]})["model_constraints"] is None


def test_opening_figures_are_dropped_in_every_form():
    facts = "- Blk 627 ANG MO KIO AVE 9 — median $655,000 — mrt 379m\n- Grants = **$85,000**\n- Max purchase price $784,366"
    for bad in ("Blk 627 is the cheaper one at 600k.", "Your grants come to 95,000 dollars.", "Six blocks match.",
                "It is 250 metres from the MRT.", "You could stretch to half a million.", "Prices rose 8 percent.",
                "Blk 627 is about $655k.", "Block #999 is nice.", "Also see 999 Toa Payoh Lorong 1."):
        assert sv.keep_supported_sentences(bad, facts)[1] == 1, bad
    for good in ("Blk 627 is the best fit for a 4-room.", "It is close to the MRT and within your budget.",
                 "Blk 627 Ang Mo Kio Ave 9 balances price and lease."):
        assert sv.keep_supported_sentences(good, facts)[1] == 0, good


# --- second round: near-variants and regressions found when the fixes were re-checked ---------------------------------

def test_amounts_are_paired_with_the_word_next_to_them():
    aff = lambda t: sv.stated_financial_inputs(t).get("affordability") or {}  # noqa: E731
    for text, want in (("cash 40k cpf 100k", {"cash_savings": 40000, "cpf_oa_balance": 100000}),
                       ("40k cash and 100k cpf", {"cash_savings": 40000, "cpf_oa_balance": 100000}),
                       ("Income 9000 cash 50k", {"monthly_income": 9000, "cash_savings": 50000}),
                       ("income 8000 cpf 90k", {"monthly_income": 8000, "cpf_oa_balance": 90000})):
        got = aff(text)
        assert all(got.get(k) == v for k, v in want.items()), (text, got)
    assert sv.rule_based_extract("Our income is 8000, CPF OA 90k").cpf_oa_balance == 90000   # the monolith's path too


def test_sc_pr_couples_in_everyday_wording_but_not_quotas_or_what_ifs():
    pair = lambda t: (sv.stated_financial_inputs(t).get("grants") or {}).get("citizenship_pair")  # noqa: E731
    for t in ("We're an SC-PR couple, both first timers", "We are a Singapore citizen and PR couple",
              "One of us is a citizen and the other is a PR", "my husband is a Malaysian PR, I'm a Singapore citizen"):
        assert pair(t) == "SC_PR", t
    for t in ("We're a Singapore citizen couple. What is the SC/PR quota under EIP?",
              "Both are Singapore citizens. What if my husband is a PR?"):
        assert pair(t) == "SC_SC", t


def test_more_ways_of_stating_the_buyers_ages():
    assert sv.buyer_ages("My husband and I are 32 and 30") == [32, 30]
    assert sv.buyer_ages("I'm 45 and my husband's 52") == [45, 52]
    assert sv.buyer_ages("My mum is 65 and I'm 35") == [35]


def test_one_partner_correction_does_not_collapse_the_couple():
    st = sv.extract_constraints_node({"messages": [
        {"role": "user", "content": "We are a Singapore citizen couple aged 32 and 30, first-timers, no kids. income 9k"},
        {"role": "assistant", "content": "..."}, {"role": "user", "content": "Sorry, my husband is 34, not 32."}]})
    a = st["user_constraints"]["financial_inputs"]["affordability"]
    assert "youngest_buyer_age" not in a and "oldest_buyer_age" not in a    # asked again, never collapsed to 34/34
    t = "We are a Singapore citizen couple aged 32 and 30. Sorry, my husband is 34, not 32."
    assert sv.fact_is_grounded("oldest_buyer_age", 34, t)


def test_income_kept_unless_the_sentence_is_about_paying():
    assert sv.rule_based_extract("We make $9,000 a month, no other loans.").household_income == 9000
    assert sv.rule_based_extract("We take home $9,000 a month. Budget is $650k.").household_income == 9000
    assert sv.rule_based_extract("We can spend $2,500 per month on the flat").household_income is None


def test_plain_statements_with_please_or_never_still_count():
    assert sv.fact_is_grounded("eligibility_confirmed", True,
                               "HDB has confirmed we are eligible for the grants and the HDB loan, so please work out our grants.")
    assert sv.fact_is_grounded("first_timer_status", True, "We have never owned a flat before.")
    assert not sv.fact_is_grounded("first_timer_status", True, "We aren't first-time buyers")
    assert not sv.fact_is_grounded("continuous_employment_12_months", True, "I lost my job last year")


def test_opening_leaves_even_the_users_limits_to_the_facts():
    facts = "- Blk 627 ANG MO KIO AVE 9 — median $655,000 — mrt 379m"
    user = "Find 4-room flats in Jurong East within 400m of an MRT station under 650k."
    assert sv.keep_supported_sentences("All are within 400m of an MRT station and under your 650k budget.", facts, user)[1] == 1
    assert sv.keep_supported_sentences("All are close to an MRT station and within your budget.", facts, user)[1] == 0


# --- third round: facts only from plain statements about the buyers; a contradiction is asked about ---------------------

def _facts(text):
    return {k: v for sec in sv.stated_financial_inputs(text).values() for k, v in sec.items()}


def test_questions_doubts_plans_and_the_past_state_nothing():
    assert sv.statements("We're an SC couple, 32 and 30, how much grant can we get?") == "We're an SC couple, 32 and 30."
    for text, field in (("How much grant can first timers get? We're 30 and 28, income 8k.", "first_timer_status"),
                        ("Do we need to have no debts?", "monthly_debt_obligations"),
                        ("Do we pay no ABSD if we buy for $600k? We own a condo.", "additional_buyer_stamp_duty"),
                        ("We are buying for $550,000 but not sure if no PHG applies", "proximity"),
                        ("My bank loan was not approved, HDB loan eligibility to be confirmed", "loan_eligibility_confirmed"),
                        ("We don't want a 30 year loan tenure", "tenure_years"),
                        ("Income ceiling is $14,000 a month right?", "monthly_income"),
                        ("Can a single Singapore citizen buy a 4 room resale? I'm 30 and engaged.", "citizenship_pair")):
        assert field not in _facts(text), text
    for field, value, text in (("eligibility_confirmed", True, "We're eligible I hope, waiting for HDB to confirm."),
                               ("eligibility_confirmed", True, "Help me confirm our eligibility for grants"),
                               ("continuous_employment_12_months", True, "I was working 12 months ago but now I'm freelancing"),
                               ("total_grants", 80000, "We're hoping for 80k grants"),
                               ("interest_rate", 25, "Downpayment 25% in cash, CPF 100k")):
        assert not sv.fact_is_grounded(field, value, text), text


def test_payments_shares_years_and_other_peoples_money_are_not_income_or_savings():
    for text in ("Car loan $800/month. cash 40k", "I give my parents $500 a month allowance. single SC 36",
                 "Ideally no more than $2,500 per month.", "Annual income 120k", "We earn $120,000 a year",
                 "I earn 5k, wife earns 4k", "Salary 6k each so 12k total", "monthly income 7k (me) + 3k (wife)"):
        assert "monthly_income" not in _facts(text), text
    assert _facts("HFE assessed: $9,000/month, HDB loan eligibility confirmed.")["monthly_income"] == 9000
    assert _facts("Together we bring home $9,000 a month and have no loans.")["monthly_income"] == 9000
    assert "cash_savings" not in _facts("My parents have 200k in savings")
    assert "cash_savings" not in _facts("I have 20k cash, my wife has 30k cash")
    assert not sv.fact_is_grounded("monthly_income", 2500, "We can afford $2,500 per month for the mortgage")
    assert not sv.fact_is_grounded("cash_savings", 120000, "our combined cpf oa is about 120k, savings minimal")
    assert sv.fact_is_grounded("monthly_income", 8500, "Monthly HFE-assessed income is $8,500; CPF OA available is $120,000.")
    assert sv.fact_is_grounded("monthly_debt_obligations", 600, "We owe 20k on a car loan, paying 600 per month")
    assert not sv.fact_is_grounded("monthly_debt_obligations", 20000, "We owe 20k on a car loan, paying 600 per month")
    assert "monthly_debt_obligations" not in _facts("We have no debts except a study loan of $300 a month.")


def test_a_couple_needs_both_ages_and_parents_or_addresses_are_not_buyers():
    own = lambda t: sorted(sv.buyer_ages(sv.statements(t, own=True)))  # noqa: E731
    assert own("I'm 35 and my husband is 2 years older") == []
    assert own("Me 33, wife 31, both citizens") == [31, 33]
    assert own("I'm 52 years old and my wife is 50 years old") == [50, 52]
    assert own("My husband's CPF is 60,000. Budget 600k.") == []
    assert own("he is 24 months into his new job. I'm 31.") == [31]
    for value, text in ((68, "Single Singapore citizen. My mother, 68, will live with me."),
                        (70, "Single, 36. Buying with my 70 yo dad"), (45, "Looking in Tampines St 45, we're a SC couple"),
                        (90, "Floor area at least 90 sqm, SC couple")):
        assert not sv.fact_is_grounded("youngest_buyer_age", value, text), text


def test_citizenship_only_when_the_buyers_state_it_without_contradiction():
    for text, want in (("Explain the grant difference between a citizen and a PR. We are both Singaporeans.", "SC_SC"),
                       ("Our friends are a citizen-PR couple. We are both Singapore citizens.", "SC_SC"),
                       ("My wife is getting PR next year. I'm a Singapore citizen.", None),
                       ("I'm SC and my husband holds PR", "SC_PR"), ("Singaporean and PR couple", "SC_PR"),
                       ("We're a couple, one Singapore citizen, my wife is Malaysian", None), ("We're PRs, not citizens.", None),
                       ("I'm a PR and single", None), ("Supposing my wife is a PR, how much grant", None),
                       ("Both are Singapore citizens, no PR", "SC_SC")):
        assert sv.citizenship_stated(text) == want, text
    assert not sv.fact_is_grounded("citizenship_pair", "SC_SC", "I'm a Singapore citizen and my wife is a foreigner")


def test_first_timer_only_when_nobody_owned_a_home():
    for text, want in (("Neither of us are first timers", False), ("We don't count as first-timers", False),
                       ("I'm a first timer but my husband owned a flat before", None),
                       ("First time applying for BTO, but my husband owned a flat with his ex-wife.", None),
                       ("we were first timers when we bought our BTO in 2015.", None), ("We sold our first flat in 2019", None),
                       ("We have never owned a flat before.", True), ("both 1st timers, no kids", True)):
        assert sv.first_timer_stated(text) == want, text


def test_a_later_message_code_cannot_read_in_full_removes_the_old_fact():
    def across(turns):
        unsure = set()
        r = sv.stated_across_turns(turns, unsure)
        return {k: v for sec in r.values() for k, v in sec.items()}, unsure
    r, _ = across(["cash 40k cpf 100k", "cash 60k"])
    assert (r["cash_savings"], r["cpf_oa_balance"]) == (60000, 100000)
    r, _ = across(["SC couple, 32 and 30, income 9k, cash 40k", "sorry cash is actually 60k"])
    assert r["cash_savings"] == 60000
    r, _ = across(["SC couple, income 9k, 4rm in Tampines under 600k", "also our car loan is $800 a month"])
    assert r["monthly_income"] == 9000
    r, unsure = across(["We are 32 and 30, SC couple, income 9k", "sorry, I'm 33 not 32"])
    assert "oldest_buyer_age" not in r and "oldest_buyer_age" in unsure
    r, _ = across(["We're 30 and 28, no kids, SC couple", "Oh we just had a baby last month, forgot to mention"])
    assert "youngest_core_member_age" not in r
    r, unsure = across(["cash 40k", "Oops, that 40k is in CPF, not cash"])
    assert "cash_savings" not in r and "cash_savings" in unsure
    # the model's reading of a removed fact is not used either
    assert "oldest_buyer_age" not in (sv.trusted_financial_inputs(
        {"affordability": {"oldest_buyer_age": 32, "youngest_buyer_age": 30}}, {}, "We are 32 and 30. sorry, I'm 33 not 32",
        {"oldest_buyer_age", "youngest_buyer_age"}) or {}).get("affordability", {})


def test_search_reading_skips_flat_age_walking_time_and_room_counts():
    rb = sv.rule_based_extract
    assert rb("at least 60 years old flat is fine, budget 450k").min_remaining_lease_years is None
    assert rb("Within 10 min walk to MRT, under 40 min to CBD").max_commute_minutes == 40
    assert rb("At least 2 rooms, budget under 500k, within 1km of Nan Hua Primary").flat_types == []
    assert rb("We're PRs, not citizens. Income 9k").citizenship_scheme is None


def test_opening_drops_abbreviated_and_relaxation_figures():
    facts = "- Blk 228 JURONG EAST ST 21 — median $532,500, 60 yrs lease — mrt 394m\n- Blk 288B JURONG EAST ST 21 — median $598,888"
    text, dropped = sv.keep_supported_sentences("Blk 228 at abt. $600k is fine. Blk 288B is the closest to the MRT.", facts)
    assert text == "Blk 288B is the closest to the MRT." and dropped == 1
    assert sv.keep_supported_sentences("Extending the MRT distance unlocks more blocks.", facts)[1] == 0
    assert sv.keep_supported_sentences("Extending the MRT distance to 550m unlocks 6 blocks.", facts)[1] == 1