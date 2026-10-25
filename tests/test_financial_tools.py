"""Reference examples, policy boundaries, funding conservation and tool integration."""

import math

import anyio
import pytest
from pydantic import ValidationError

from agents.schemas import (
    CalculateBsdAndUpfrontOutlayOutput,
    CalculateCpfHousingGrantsInput,
    CalculateCpfHousingGrantsOutput,
    CalculateMaximumAffordabilityOutput,
)
from mcp_server.financial_calc_tools import (
    calculate_bsd_and_upfront_outlay as upfront,
)
from mcp_server.financial_calc_tools import (
    calculate_cpf_housing_grants as grants,
)
from mcp_server.financial_calc_tools import (
    calculate_maximum_affordability as affordability,
)


def grant_inputs(**changes):
    return dict(
        dict(
            flat_category="resale",
            citizenship_pair="SC_SC",
            first_timer_status=True,
            monthly_income=4400,
            flat_type="4 ROOM",
            youngest_buyer_age=30,
            youngest_core_member_age=30,
            remaining_lease_years=70,
            eligibility_confirmed=True,
            continuous_employment_12_months=True,
            proximity="none",
        ),
        **changes,
    )


def aff_inputs(**changes):
    return dict(
        dict(
            monthly_income=8000,
            cpf_oa_balance=100000,
            cash_savings=50000,
            loan_type="hdb",
            average_applicant_age=35,
            youngest_buyer_age=35,
            oldest_buyer_age=35,
            remaining_lease_years=65,
            monthly_debt_obligations=0,
            total_grants=80000,
            loan_eligibility_confirmed=True,
        ),
        **changes,
    )


def upfront_inputs(**changes):
    return dict(
        dict(
            purchase_price=600000,
            market_valuation=550000,
            loan_amount=400000,
            loan_type="bank",
            ltv_ratio=0.75,
            total_grants=80000,
            cpf_oa_balance=50000,
            cash_savings=90000,
            legal_and_valuation_fees=3000,
            additional_buyer_stamp_duty=0,
        ),
        **changes,
    )


def test_official_hdb_sc_pr_example():
    r = grants(**grant_inputs(citizenship_pair="SC_PR"))
    assert r["status"] == "ok"
    assert r["family_grant"] == 70000
    assert r["enhanced_housing_grant"] == 70000
    assert r["total_grants"] == 140000
    assert r["rule_version"] and r["sources"]
    CalculateCpfHousingGrantsOutput.model_validate(r)


@pytest.mark.parametrize(
    "income,expected",
    [
        (0, 120000),
        (1500, 120000),
        (1500.01, 110000),
        (4500, 70000),
        (4500.01, 65000),
        (8500, 10000),
        (8500.01, 5000),
        (9000, 5000),
        (9000.01, 0),
    ],
)
def test_ehg_tier_boundaries(income, expected):
    assert (
        grants(**grant_inputs(monthly_income=income))["enhanced_housing_grant"]
        == expected
    )


@pytest.mark.parametrize(
    "scheme,flat,income,family,ehg,phg",
    [
        ("SC_SC", "5 ROOM", 4400, 50000, 70000, 20000),
        ("SC_PR", "EXECUTIVE", 4400, 40000, 70000, 20000),
        ("SC_Single", "4 ROOM", 2250, 40000, 35000, 10000),
        ("SC_Single", "5 ROOM", 4500, 25000, 2500, 10000),
        ("SC_SC", "4 ROOM", 16000, 80000, 0, 20000),
        ("SC_SC", "4 ROOM", 16000.01, 0, 0, 20000),
        ("SC_Single", "4 ROOM", 8000, 40000, 0, 10000),
        ("SC_Single", "4 ROOM", 8000.01, 0, 0, 10000),
    ],
)
def test_grant_households_and_income_limits(scheme, flat, income, family, ehg, phg):
    r = grants(
        **grant_inputs(
            citizenship_pair=scheme,
            flat_type=flat,
            monthly_income=income,
            youngest_buyer_age=35,
            proximity="near",
            proximity_eligibility_confirmed=True,
        )
    )
    assert (
        r["family_grant"],
        r["enhanced_housing_grant"],
        r["proximity_housing_grant"],
    ) == (family, ehg, phg)


def test_grant_unknowns_and_unsupported_are_not_zero_grants():
    r = grants(monthly_income=8000)
    assert r["status"] == "needs_information" and r["total_grants"] is None
    assert "citizenship_pair" in r["missing_inputs"]
    for change in (
        {"first_timer_status": False},
        {"flat_category": "bto"},
        {"remaining_lease_years": 60},
        {"eligibility_confirmed": False},
    ):
        r = grants(**grant_inputs(**change))
        assert r["status"] == "unsupported" and r["total_grants"] is None
    assert (
        grants(**grant_inputs(continuous_employment_12_months=False))[
            "enhanced_housing_grant"
        ]
        == 0
    )


def test_hdb_interest_floor_annuity_and_purchase_funding():
    r = affordability(**aff_inputs())
    # Independent float annuity with official 3% assessment floor; 25 years.
    rate = 0.03 / 12
    ceiling = math.floor(2400 * (1 - (1 + rate) ** -300) / rate * 100) / 100
    assert r["assessment_interest_rate"] == 3
    assert r["max_loan_amount"] == pytest.approx(ceiling, abs=0.01)
    assert r["max_purchase_price"] == pytest.approx(ceiling + 230000, abs=0.01)
    assert r["ltv_ratio"] == 0.75 and r["grants_included"] is True
    assert r["purchase_costs_included"] is False
    actual = 0.026 / 12
    assert r["monthly_instalment"] == pytest.approx(
        ceiling * actual / (1 - (1 + actual) ** -300), abs=0.01
    )
    CalculateMaximumAffordabilityOutput.model_validate(r)


def test_bank_debts_and_reduced_ltv():
    r = affordability(
        **aff_inputs(loan_type="bank", interest_rate=2, monthly_debt_obligations=3000)
    )
    assert r["assessment_interest_rate"] == 4
    assert r["binding_constraint"] == "TDSR"
    assert (
        r["max_loan_amount"]
        < affordability(**aff_inputs(loan_type="bank", interest_rate=2))[
            "max_loan_amount"
        ]
    )
    r = affordability(**aff_inputs(loan_type="bank", interest_rate=4, tenure_years=30))
    assert r["ltv_ratio"] == 0.55
    assert r["max_purchase_price"] <= 50000 / 0.10


def test_age_tenure_missing_and_zero_funds():
    r = affordability(
        **aff_inputs(
            average_applicant_age=50, youngest_buyer_age=50, oldest_buyer_age=50
        )
    )
    assert r["tenure_months"] == 180
    r = affordability(**aff_inputs(cpf_oa_balance=0, cash_savings=0, total_grants=0))
    assert r["max_purchase_price"] == 0 and r["max_loan_amount"] == 0
    assert affordability(monthly_income=8000)["status"] == "needs_information"
    assert (
        affordability(**aff_inputs(monthly_debt_obligations=1))["status"]
        == "unsupported"
    )
    assert (
        affordability(**aff_inputs(youngest_buyer_age=25, remaining_lease_years=60))[
            "status"
        ]
        == "unsupported"
    )


@pytest.mark.parametrize(
    "price,expected",
    [
        (180000, 1800),
        (360000, 5400),
        (1000000, 24600),
        (1500000, 44600),
        (3000000, 119600),
        (3500000, 149600),
        (180099, 1801),
        (1, 1),
    ],
)
def test_bsd_independent_reference_cases(price, expected):
    r = upfront(
        **upfront_inputs(purchase_price=price, market_valuation=price, loan_amount=0)
    )
    assert r["buyer_stamp_duty"] == expected


def test_valuation_cash_funding_conservation():
    r = upfront(**upfront_inputs())
    # BSD on 600k: 1800 + 3600 + 7200. 200k purchase equity + 12600 duty + 3000 fees.
    assert r["buyer_stamp_duty"] == 12600
    assert r["cash_over_valuation"] == 50000
    assert r["min_cash_downpayment"] == 27500
    assert r["total_upfront"] == 215600
    assert r["cpf_oa_usable"] == 50000
    assert r["cash_required"] == 85600
    assert r["cash_shortfall"] == 0
    assert (
        r["cash_required"] + r["cpf_oa_usable"] + r["grants_used"] == r["total_upfront"]
    )
    CalculateBsdAndUpfrontOutlayOutput.model_validate(r)


def test_grants_never_pay_duty_fees_or_cov():
    r = upfront(**upfront_inputs(total_grants=500000, cpf_oa_balance=0, cash_savings=0))
    assert r["grants_used"] == 122500
    assert r["unused_grants"] == 377500
    assert r["cash_required"] == 93100  # COV + mandatory cash + duty + fees.
    assert r["cash_shortfall"] == 93100
    r = upfront(**upfront_inputs(market_valuation=700000))
    assert r["buyer_stamp_duty"] == 15600  # Higher valuation determines BSD.


def test_unknown_valuation_fees_and_over_ltv_loan():
    r = upfront(purchase_price=600000)
    assert r["status"] == "needs_information"
    assert r["cash_required"] is None and "market_valuation" in r["missing_inputs"]
    assert upfront(**upfront_inputs(loan_amount=500000))["status"] == "unsupported"


@pytest.mark.parametrize(
    "kwargs", [{"monthly_income": float("nan")}, {"monthly_income": -1}, {"unknown": 1}]
)
def test_invalid_grant_inputs_rejected(kwargs):
    with pytest.raises(ValidationError):
        CalculateCpfHousingGrantsInput(**kwargs)


def test_real_registry_and_agent_sequence(monkeypatch):
    from agents.llm import load_config
    from agents.supervisor import (
        _affordable_price,
        run_financial_sequence,
        template_response,
    )
    from agents.tool_registry import build_runners

    for name in (
        "calculate_cpf_housing_grants",
        "calculate_maximum_affordability",
        "calculate_bsd_and_upfront_outlay",
    ):
        monkeypatch.setitem(load_config()["tools"]["use_mock"], name, "real")
    runners = build_runners(
        [
            "calculate_cpf_housing_grants",
            "calculate_maximum_affordability",
            "calculate_bsd_and_upfront_outlay",
        ]
    )
    aff = aff_inputs()
    aff.pop("total_grants")
    up = upfront_inputs()
    up.pop("total_grants")
    output = run_financial_sequence(
        runners,
        {
            "financial_inputs": {
                "grants": grant_inputs(),
                "affordability": aff,
                "upfront": up,
            }
        },
        "resale",
    )
    assert all(value["status"] == "ok" for value in output.values())
    assert (
        runners["calculate_maximum_affordability"].log[-1].arguments["total_grants"]
        == 150000
    )
    assert (
        runners["calculate_bsd_and_upfront_outlay"].log[-1].arguments["total_grants"]
        == 150000
    )
    state = {"financial_breakdown": output}
    assert (
        _affordable_price(state)
        == output["calculate_maximum_affordability"]["max_purchase_price"]
    )
    assert "Financials" in template_response(state)
    output = run_financial_sequence(
        runners,
        {"household_income": 8000, "citizenship_scheme": "SC_SC_FirstTimer"},
        "couple",
    )
    assert output["calculate_cpf_housing_grants"]["status"] == "needs_information"
    assert (
        "citizenship_pair" in output["calculate_cpf_housing_grants"]["missing_inputs"]
    )
    assert "calculate_bsd_and_upfront_outlay" not in output
    assert "more information needed" in template_response({"financial_breakdown": output})


def test_financial_mcp_protocol():
    from fastmcp import Client

    from mcp_server.server import mcp

    async def check():
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert (
                tools["calculate_cpf_housing_grants"].inputSchema[
                    "additionalProperties"
                ]
                is False
            )
            for name, args, model in (
                (
                    "calculate_cpf_housing_grants",
                    grant_inputs(),
                    CalculateCpfHousingGrantsOutput,
                ),
                (
                    "calculate_maximum_affordability",
                    aff_inputs(),
                    CalculateMaximumAffordabilityOutput,
                ),
                (
                    "calculate_bsd_and_upfront_outlay",
                    upfront_inputs(),
                    CalculateBsdAndUpfrontOutlayOutput,
                ),
            ):
                response = await client.call_tool(name, args)
                assert not response.is_error
                assert model.model_validate(response.structured_content).status == "ok"
            response = await client.call_tool(
                "calculate_cpf_housing_grants", {"monthly_income": 8000}
            )
            assert response.structured_content["status"] == "needs_information"

    anyio.run(check)


def test_core_occupier_age_and_phg_with_parents():
    assert (
        grants(**grant_inputs(youngest_core_member_age=20))["status"] == "unsupported"
    )
    r = grants(**grant_inputs(proximity="with", proximity_eligibility_confirmed=True))
    assert r["proximity_housing_grant"] == 30000
    r = grants(**grant_inputs(proximity="near", proximity_eligibility_confirmed=False))
    assert r["status"] == "unsupported" and r["total_grants"] is None


def test_bank_purchase_without_loan_and_zero_interest_scenario():
    r = affordability(**aff_inputs(loan_type="bank", interest_rate=0, cash_savings=0))
    assert r["max_purchase_price"] == 180000
    assert r["max_loan_amount"] == 0
    r = affordability(**aff_inputs(interest_rate=0))
    assert r["assessment_interest_rate"] == 3
    assert r["monthly_instalment"] == pytest.approx(
        r["max_loan_amount"] / 300, abs=0.01
    )


def test_remaining_cpf_pays_bsd_and_absd_stays_cash_funded():
    r = upfront(
        **upfront_inputs(cpf_oa_balance=500000, additional_buyer_stamp_duty=20000)
    )
    assert r["cpf_oa_usable"] == 55100  # 42,500 eligible equity plus 12,600 BSD.
    assert r["cash_required"] == 100500  # COV, minimum cash, fees and ABSD.


def test_real_financial_specialist_fallback_and_price_cap(monkeypatch):
    from agents import supervisor as sv
    from agents.schemas import ExtractedConstraints
    from agents.tool_registry import tools_for_agent

    monkeypatch.setattr(sv, "get_llm", lambda: None)
    for name in tools_for_agent("financial"):
        monkeypatch.setitem(sv.load_config()["tools"]["use_mock"], name, "real")
    c = ExtractedConstraints(
        household_income=8000,
        financial_inputs={
            "grants": grant_inputs(),
            "affordability": aff_inputs(),
            "upfront": upfront_inputs(),
        },
    ).model_dump()
    outputs, log, errors, _ = sv.run_react(
        "financial", tools_for_agent("financial"), "", "resale", c
    )
    assert (
        not errors
        and len(log) == 3
        and all(value["status"] == "ok" for value in outputs.values())
    )
    state = {
        "financial_breakdown": outputs,
        "user_constraints": {"max_budget_sgd": 1000000},
        "sql_market_results": {
            "query_resale_blocks_by_criteria": {
                "blocks": [
                    {"block": "1", "street": "A", "median_resale_price": 100000},
                    {"block": "2", "street": "B", "median_resale_price": 950000},
                ]
            }
        },
    }
    assert len(sv.intersect_node(state)["candidate_blocks_intersection"]) == 1
    outputs["calculate_maximum_affordability"]["max_purchase_price"] = 0
    assert sv.intersect_node(state)["candidate_blocks_intersection"] == []
    assert sv.decide_routing(
        {"financial_inputs": {"affordability": {"monthly_income": 8000}}}
    ) == ["financial"]
