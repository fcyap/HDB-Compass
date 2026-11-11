"""Demo-readiness fixes to the money answers (Member 4, 4 Nov): the stamp duty is costed only on a price the user gave
or code worked out, two PRs get a bank-loan budget, household facts in the tester's own words are read, a stamp-duty
price correction replaces the old price, instalment follow-ups change one figure, a short lease is used and explained,
and "which of those is cheapest?" / "stamp duty on that block" answer from the list just shown. Each test uses the
tester's wording (data/local/qa/gpt_issues.json, gpt_records.json). The live-model guard uses a fake model double."""
from __future__ import annotations

import os

os.environ["LLM_PROFILE"] = "none"
os.environ.setdefault("HDB_ADDONS", "app")

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from agents import arch_3_specialist_5agent as C  # noqa: E402
from agents import supervisor as sv  # noqa: E402
from agents.tool_registry import tools_for_agent  # noqa: E402


def app(query, previous=None):
    return C.graph.invoke(sv.initial_state(query=query, variant="C", previous=previous, fast_path=True))


def conversation(*turns):
    """Each turn's final state, every turn carrying the last (as the app does)."""
    states, prev = [], None
    for text in turns:
        prev = app(text, prev)
        states.append(prev)
    return states


def tools_run(state):
    return [r["tool"] for r in state.get("tool_call_log") or []]


def question(state):
    return (state.get("clarification") or {}).get("question") or ""


# ---------------------------------------------------------------------------
# LIVE-MONEY-01: the upfront calculator never costs a price the model made up
# ---------------------------------------------------------------------------
class ToolModel:
    """The tool loop makes the given calls once, then stops."""

    def __init__(self, calls):
        self.calls, self.n = calls, 0

    def bind_tools(self, _tools, tool_choice=None):
        return self

    def invoke(self, _messages):
        self.n += 1
        return AIMessage(content="", tool_calls=self.calls if self.n == 1 else [])


UPFRONT = "calculate_bsd_and_upfront_outlay"
AFFORD = "calculate_maximum_affordability"
MODEL_UPFRONT_CALL = [{"name": UPFRONT, "id": "u1", "args": {
    "purchase_price": 500000, "market_valuation": 500000, "loan_amount": 375000, "loan_type": "hdb"}}]
HOUSE_04 = ("We are married citizens aged 37 and 35, second-time buyers, no children. Monthly income $10k, usable CPF "
            "$100k, cash $70k, no debts, loan eligibility confirmed. What can we afford with 70 years remaining?")


def test_the_model_cannot_cost_a_price_nobody_gave(monkeypatch):
    # HOUSE-04 t3 live: "Does stamp duty fit within that number?" was answered with $9,600 BSD on a made-up $500,000
    monkeypatch.setattr(sv, "get_llm", lambda: ToolModel(MODEL_UPFRONT_CALL))
    c = {"requested_tools": [UPFRONT]}
    outputs, log, errors, _ = sv.run_react("financial", tools_for_agent("financial"), "sys",
                                           "Does stamp duty fit within that number?", c)
    assert UPFRONT not in outputs and not [r for r in log if r["tool"] == UPFRONT]
    assert any("no purchase price stated" in e for e in errors)


def test_the_model_s_price_is_replaced_by_the_worked_out_ceiling(monkeypatch):
    st = app(HOUSE_04)
    ceiling = st["financial_breakdown"][AFFORD]["max_purchase_price"]
    c = {**st["user_constraints"], "requested_tools": [UPFRONT]}
    monkeypatch.setattr(sv, "get_llm", lambda: ToolModel(MODEL_UPFRONT_CALL))
    outputs, log, errors, _ = sv.run_react("financial", tools_for_agent("financial"), "sys",
                                           "Does stamp duty fit within that number?", c)
    call = next(r for r in log if r["tool"] == UPFRONT)
    assert call["arguments"]["purchase_price"] == ceiling
    assert call["arguments"].get("market_valuation") is None and call["arguments"].get("loan_amount") is None


def test_a_stated_price_wins_over_the_model_s():
    c = {"financial_inputs": {"upfront": {"purchase_price": 650000.0}}}
    args = sv.guard_financial_args(UPFRONT, {"purchase_price": 900000, "market_valuation": 900000}, c, {}, {})
    assert args["purchase_price"] == 650000.0 and "market_valuation" not in args
    assert sv.guard_financial_args(UPFRONT, {"purchase_price": 900000}, {"max_budget_sgd": 600000.0}, {}, {})[
        "purchase_price"] == 600000.0


def test_stamp_duty_on_that_number_is_the_ceiling_s_duty_and_zero_grants_skips_the_grant_calculator():
    t1, t2, t3 = conversation(HOUSE_04, "Assume zero housing grants.", "Does stamp duty fit within that number?")
    assert t1["financial_breakdown"][AFFORD]["max_purchase_price"] == 680000
    assert "calculate_cpf_housing_grants" not in tools_run(t2) and not t2.get("clarification")
    assert t3["financial_breakdown"][UPFRONT]["buyer_stamp_duty"] == 15000
    assert "**$15,000** (on your maximum purchase price of $680,000" in t3["draft_response"]


# ---------------------------------------------------------------------------
# SEM-20: two PRs can only take a bank loan
# ---------------------------------------------------------------------------
HOUSE_03 = ("Both of us are PRs aged 32 and 33, first-time buyers with no children. Income $9,000, usable CPF $120k and "
            "cash $60k. No debt; our HDB loan eligibility is supposedly confirmed. We want a 4-room resale with 70 years "
            "remaining. What's our maximum HDB-loan budget?")


def test_a_pr_couple_is_worked_out_with_a_bank_loan():
    states = conversation(HOUSE_03, "Neither of us is a citizen; we are both PRs. Can you really use an HDB loan?",
                          "What information would you need for a bank-loan calculation instead?")
    for st in states:
        c = st["user_constraints"]
        assert c["financial_inputs"]["affordability"]["loan_type"] == "bank"
        assert any(a.startswith("An HDB loan needs at least one Singapore Citizen buyer, so two PRs can only take a bank loan")
                   for a in c["assumptions"])
        assert "in-principle approval" in question(st) and "interest rate the bank quoted" in question(st)
        assert "HDB loan eligibility" not in question(st)


def test_a_pr_hypothetical_gets_no_bank_loan_note():
    c = app("What if both buyers are PRs?")["user_constraints"]
    assert not any("bank loan" in a for a in c.get("assumptions") or [])


def test_bank_loan_with_a_hyphen_is_a_bank_loan():
    assert sv.stated_financial_inputs("We earn $9,000. What would a bank-loan calculation need?")[
        "affordability"]["loan_type"] == "bank"


# ---------------------------------------------------------------------------
# SEM-19: household facts in the tester's words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, pair", [
    ("We are married citizens aged 37 and 35, second-time buyers, no children.", "SC_SC"),
    ("Citizen first-timers aged 27 and 29, income $7k, CPF $60k, cash $35k, no debts.", "SC_SC"),
    ("We are citizen first-timer buyers aged 30 and 32, no children, income $7,500.", "SC_SC"),
    ("We are married citizen first-timers aged 31 and 32, household income $8k.", "SC_SC"),
    ("What is the difference between citizen buyers and PR buyers?", None),
])
def test_citizenship_in_the_tester_s_words(text, pair):
    assert sv.citizenship_stated(text) == pair


def test_second_time_buyers_are_not_first_timers():
    assert sv.first_timer_stated("We are married citizens aged 37 and 35, second-time buyers, no children.") is False


def test_income_written_as_monthly_is_read():
    fin = sv.stated_financial_inputs("We are a married citizen couple, aged 31 and 33. Income $8,000 monthly, usable CPF "
                                     "OA $100k, cash $50k, no debts.")
    assert fin["affordability"]["monthly_income"] == 8000
    assert "monthly_income" not in sv.stated_financial_inputs("We have been saving for 6 months.").get("affordability", {})


@pytest.mark.parametrize("text, expected", [
    ("Both worked 12 months, grants confirmed, loan eligibility not confirmed.",
     {"eligibility_confirmed": True, "loan_eligibility_confirmed": False}),
    ("Grants and HDB loan eligibility confirmed, no proximity grant.",
     {"eligibility_confirmed": True, "loan_eligibility_confirmed": True}),
    ("HDB has not confirmed our grant eligibility.", {"eligibility_confirmed": False}),
    ("Our proximity grant eligibility is confirmed.", {"proximity_eligibility_confirmed": True}),
])
def test_confirmations_are_read_clause_by_clause(text, expected):
    assert sv.confirmations(text) == expected


@pytest.mark.parametrize("text, worked", [
    ("Both worked 12 months, grants confirmed.", True),
    ("Yes, we both worked continuously for the last 12 months.", True),
    ("We have not worked continuously for the past twelve months.", False),
])
def test_continuous_work_in_the_tester_s_words(text, worked):
    assert sv.stated_financial_inputs(text)["grants"]["continuous_employment_12_months"] is worked


@pytest.mark.parametrize("text, grants", [
    ("Assume zero housing grants.", 0.0), ("We get no grants.", 0.0), ("What if we buy with no grants?", None)])
def test_zero_grants_is_read_only_when_stated(text, grants):
    assert sv.stated_financial_inputs(text).get("affordability", {}).get("total_grants") == grants


def test_house_01_grants_after_the_employment_answer():
    t1, t2 = conversation(
        "We are a married citizen couple, both first-timers, aged 31 and 33, no children. Income $8,000 monthly, usable "
        "CPF OA $100k, cash $50k, no debts. Grant and HDB loan eligibility confirmed. We are not buying near our parents. "
        "What grants and max budget for a 4-room resale with 70 years left?",
        "We have not worked continuously for the past twelve months.")
    # 12 months' work is assumed and said so instead of asked (live on Claude, 11 Nov); the stated "No" then drops the EHG
    assert not t1.get("clarification") and t1["financial_breakdown"]["calculate_cpf_housing_grants"]["total_grants"] > 80000
    assert t2["financial_breakdown"]["calculate_cpf_housing_grants"]["total_grants"] == 80000
    assert not t2.get("clarification")


def test_house_02_grants_once_employment_is_confirmed():
    t1, t2 = conversation(
        "We are citizen first-timer buyers aged 30 and 32, no children, income $7,500, usable CPF $80k, cash $45k, no "
        "debts. Grants and HDB loan eligibility confirmed, no proximity grant. Buying 4-room resale with 70 years left; "
        "what's our budget?", "Yes, we both worked continuously for the last 12 months.")
    assert "citizenship" not in question(t1) and "grant eligibility" not in question(t1)
    assert t2["financial_breakdown"]["calculate_cpf_housing_grants"]["total_grants"] == 105000
    assert t2["financial_breakdown"][AFFORD]["status"] == "ok"


def test_house_06_unconfirmed_grants_drop_the_grant_questions():
    t1, t2 = conversation("We are married citizen first-timers aged 31 and 32, household income $8k, CPF $90k and cash "
                          "$50k. Help with grants.", "HDB has not confirmed our grant eligibility.")
    assert "citizenship" not in question(t1)
    asked = question(t2)
    assert "youngest person" not in asked and "employed continuously" not in asked and "parents" not in asked
    # a grants-only conversation ("Help with grants.") is not asked for the budget's facts (live on Claude, 11 Nov)
    assert "debt" not in asked and "Housing grants: not worked out — your grant eligibility is not confirmed" in t2["draft_response"]


def test_house_07_grants_with_loan_eligibility_not_confirmed():
    st = app("We are married citizens, first-timers, ages 32 and 30. No children, income $9k, usable CPF $90k, cash $40k, "
             "no debt. Both worked 12 months, grants confirmed, loan eligibility not confirmed. Resale 4-room, lease 70 "
             "years, no proximity grant. Maximum budget?")
    assert st["financial_breakdown"]["calculate_cpf_housing_grants"]["total_grants"] == 85000
    assert not st.get("clarification")


# ---------------------------------------------------------------------------
# SEM-18, SEM-06: the stamp duty on a corrected price; no income needed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("first, second, duty", [
    ("$180,000", "$360,000", 5400), ("$360,000", "$500,000", 9600), ("$650,000", "$700,000", 15600),
    ("$1,000,000", "$1,200,000", 32600), ("$1,500,000", "$1,600,000", 49600), ("$3,000,000", "$3,200,000", 131600),
])
def test_a_price_and_valuation_correction_replaces_the_old_price(first, second, duty):
    _, st = conversation(f"BSD only for a residential purchase of {first}, valuation equals price.",
                         f"Change price and valuation to {second}.")
    assert st["financial_breakdown"][UPFRONT]["buyer_stamp_duty"] == duty
    assert f"Buyer's stamp duty: **${duty:,}**" in st["draft_response"]
    assert "Upfront funding: more information needed" not in st["draft_response"]


def test_price_is_now_corrects_a_duty_question_but_a_search_price_range_is_not_a_price():
    _, st = conversation("What is buyer's stamp duty on a $500,000 resale flat?", "The price is now $700,000.")
    assert st["financial_breakdown"][UPFRONT]["buyer_stamp_duty"] == 15600
    assert "upfront" not in (sv.stated_financial_inputs("4 room in Bedok, price range is $500k to $600k") or {})


@pytest.mark.parametrize("price, duty", [("$500,000", "$9,600"), ("$700,000", "$15,600")])
def test_the_duty_needs_no_income(price, duty):
    _, st = conversation(f"What is buyer's stamp duty on a {price} resale flat, valuation also {price}?",
                         "Do I have to give you my income just to calculate that duty?")
    assert st["draft_response"].startswith("No — BSD depends only on the price (or valuation, if higher).")
    assert f"**{duty}**" in st["draft_response"] and "Upfront funding" not in st["draft_response"]


# ---------------------------------------------------------------------------
# SEM-05, SEM-12: monthly instalments and their follow-ups
# ---------------------------------------------------------------------------
def monthly(state):
    return (state["user_constraints"].get("instalment") or {}).get("monthly")


@pytest.mark.parametrize("turns, figures", [
    (("Monthly repayment on a $500,000 loan at 2.6% for 25 years?", "What if I repay over 20 years instead?"), [2268, 2674]),
    (("Monthly repayment on a $400,000 loan at 3% for 25 years?", "What if I repay over 20 years instead?"), [1897, 2218]),
    (("Monthly instalment for a $300k loan, 2.6%, 20 years?", "Use 15 years instead."), [1604, 2015]),
    (("A $250k loan at 3.5% for 25 years, monthly repayment?", "Increase rate to 4.5%, leave the rest."), [1252, 1390]),
    (("$600,000 mortgage at 4 percent over 30 years: repayment per month?", "Same terms but borrow only $550,000."),
     [2864, 2626]),
    (("How much every month for a $420k mortgage at 3% over 20 years?", "If the loan is $400k instead?"), [2329, 2218]),
])
def test_an_instalment_follow_up_changes_one_figure(turns, figures):
    states = conversation(*turns)
    assert [monthly(st) for st in states] == figures
    for st in states:
        assert f"**Monthly instalment: ${monthly(st):,}**" in st["draft_response"]
        assert not st.get("clarification") and not st["user_constraints"].get("max_budget_sgd")


def test_an_instalment_asked_without_figures_asks_only_for_them():
    t1, t2, t3 = conversation("How much per month for my home loan?", "$450k at 2.6 percent, 25 years.",
                              "Thanks, what about 30 years at the same rate?")
    assert question(t1).startswith("To work out the monthly instalment") and not tools_run(t1)
    assert [monthly(t2), monthly(t3)] == [2042, 1802]


def test_instalment_readings():
    assert sv.instalment_question("Repayment on $100k at 0% interest over 10 years?")["monthly"] == 833
    assert sv.instalment_question("$600,000 mortgage at 4 percent over 30 years: repayment per month?")["loan"] == 600000
    prior = sv.instalment_question("Monthly repayment on a $500,000 loan at 2.6% for 25 years?")
    assert sv.instalment_question("We want a flat with 70 years lease instead.", prior) is None
    assert sv.instalment_question("Find 4-room in Tampines under $600k instead.", prior) is None
    assert sv.instalment_question("same loan at 3%", prior)["rate"] == 3
    assert sv.instalment_question("use $600k instead", prior)["loan"] == 600000
    assert sv.instalment_question("HDB loan or bank loan for a $500k flat?", prior) is None


def test_an_instalment_is_not_carried_into_a_flat_search():
    _, st = conversation("Monthly repayment on a $500,000 loan at 2.6% for 25 years?",
                         "Find 4-room resale in Tampines under $600k.")
    assert not st["user_constraints"].get("instalment") and "Monthly instalment" not in st["draft_response"]


# ---------------------------------------------------------------------------
# SEM-21: a stated short lease is used and its limit explained
# ---------------------------------------------------------------------------
def test_a_short_lease_is_used_and_explained_every_turn():
    states = conversation("Citizen first-timers aged 27 and 29, income $7k, CPF $60k, cash $35k, no debts. Grants and "
                          "loan eligibility confirmed. Buying 4-room resale with 40 years left. What grants and maximum loan?",
                          "We want you to use those exact ages and 40-year lease.")
    for st in states:
        c = st["user_constraints"]
        assert c["financial_inputs"]["grants"]["remaining_lease_years"] == 40
        assert any("HDB prorates the CPF use, loan and grants" in a and "27 + 40 = 67" in a for a in c["assumptions"])
        assert not st.get("clarification") and "68 years" not in st["draft_response"]


# ---------------------------------------------------------------------------
# SPAT-27, SPAT-28: "cheapest of those", then the stamp duty on that block
# ---------------------------------------------------------------------------
def test_cheapest_of_those_then_stamp_duty_on_that_block():
    t1, t2, t3 = conversation("Find 4-room resale in Bedok under $700k within 500m of MRT.",
                              "Which of those blocks is cheapest?",
                              "What is stamp duty on that block's typical 4-room price?")
    cheapest = min(t1["candidate_blocks_intersection"][:sv.LISTED_BLOCKS], key=lambda b: b["median_resale_price"])
    name = f"Blk {cheapest['block']} {cheapest['street']}"
    assert f"{name} is the cheapest" in t2["draft_response"]
    assert not tools_run(t2) and "Sales history" not in t2["draft_response"]
    assert sv.UNCHANGED_NOTE not in t2["draft_response"]
    assert t3["financial_breakdown"][UPFRONT]["buyer_stamp_duty"] > 0
    assert f"Worked out for {name} at its median resale price" in t3["draft_response"]
    assert "Sales history" not in t3["draft_response"]
