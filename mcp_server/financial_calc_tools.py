"""Deterministic, conditional resale estimates using a reviewed local rule snapshot."""

from __future__ import annotations

import json
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from functools import partial
from pathlib import Path

from agents.schemas import (
    CalculateBsdAndUpfrontOutlayInput,
    CalculateBsdAndUpfrontOutlayOutput,
    CalculateCpfHousingGrantsInput,
    CalculateCpfHousingGrantsOutput,
    CalculateMaximumAffordabilityInput,
    CalculateMaximumAffordabilityOutput,
)

RULES = json.loads(
    (Path(__file__).resolve().parents[1] / "config/financial_rules.json").read_text()
)
CENT = Decimal("0.01")


def D(value):
    return Decimal(str(value))


def money(value, rounding=ROUND_HALF_UP):
    return float(D(value).quantize(CENT, rounding=rounding))


def result(model, source_keys, **values):
    return model(
        rule_version=RULES["version"],
        verified_on=RULES["verified_on"],
        sources=[RULES["sources"][key] for key in source_keys],
        **values,
    ).model_dump()


def missing(data, names):
    return [name for name in names if getattr(data, name) is None]


def calculate_cpf_housing_grants(
    flat_category: str = "resale",
    citizenship_pair: str | None = None,
    first_timer_status: bool | None = None,
    monthly_income: float | None = None,
    near_parents_4km: bool = False,
    flat_type: str | None = None,
    youngest_buyer_age: int | None = None,
    youngest_core_member_age: int | None = None,
    remaining_lease_years: float | None = None,
    eligibility_confirmed: bool | None = None,
    continuous_employment_12_months: bool | None = None,
    proximity: str | None = None,
    proximity_eligibility_confirmed: bool | None = None,
) -> dict:
    """Estimate grants for standard first-timer resale households. Never infer eligibility confirmations. Supply HFE-assessed income, buyer and youngest core member ages and explicit proximity choice; unsupported lease/scheme cases require HDB assessment."""
    x = CalculateCpfHousingGrantsInput(**locals())
    sources = [
        "family_grant",
        "single_grant",
        "ehg",
        "ehg_eligibility",
        "ehg_single",
        "phg",
        "phg_single",
    ]
    finish = partial(result, CalculateCpfHousingGrantsOutput, sources)
    if x.flat_category != "resale" or x.first_timer_status is False:
        return finish(
            status="unsupported",
            notes=[
                "Only standard first-timer resale purchases are supported; BTO and second-timer/mixed households require separate assessment."
            ],
        )
    needed = missing(
        x,
        [
            "citizenship_pair",
            "first_timer_status",
            "monthly_income",
            "flat_type",
            "youngest_buyer_age",
            "youngest_core_member_age",
            "remaining_lease_years",
            "eligibility_confirmed",
            "continuous_employment_12_months",
        ],
    )
    proximity = x.proximity or ("near" if x.near_parents_4km else None)
    if proximity is None:
        needed.append("proximity")
    if proximity in ("near", "with") and x.proximity_eligibility_confirmed is None:
        needed.append("proximity_eligibility_confirmed")
    if needed:
        return finish(status="needs_information", missing_inputs=needed)
    if x.near_parents_4km and proximity != "near":
        return finish(status="unsupported", notes=["Conflicting proximity inputs."])
    if not x.eligibility_confirmed:
        return finish(
            status="unsupported",
            notes=[
                "Grant prerequisites have not been confirmed; no grant amount can be established."
            ],
        )
    single = x.citizenship_pair == "SC_Single"
    if single and (x.youngest_buyer_age < 35 or x.flat_type == "EXECUTIVE"):
        return finish(
            status="unsupported",
            notes=[
                "This calculator supports single citizens aged 35+ buying up to a 5-room resale flat."
            ],
        )
    if x.youngest_core_member_age > x.youngest_buyer_age:
        return finish(
            status="unsupported",
            notes=["Youngest core member age cannot exceed youngest buyer age."],
        )
    if (
        x.remaining_lease_years <= 20
        or x.youngest_core_member_age + x.remaining_lease_years < 95
    ):
        return finish(
            status="unsupported",
            notes=[
                "Lease coverage to age 95 is required by this model; prorated grants are not implemented."
            ],
        )
    ceiling = RULES[
        "grant_income_ceiling_single" if single else "grant_income_ceiling_family"
    ]
    family = (
        0
        if x.monthly_income > ceiling
        else RULES["family_grants"][x.citizenship_pair][
            int(x.flat_type in ("5 ROOM", "EXECUTIVE"))
        ]
    )
    ehg = 0
    assessed_income = D(x.monthly_income) * (2 if single else 1)
    if family and x.continuous_employment_12_months:
        for bound, amount in zip(
            RULES["ehg_family_income_upper_bounds"], RULES["ehg_family_amounts"]
        ):
            if assessed_income <= D(bound):
                ehg = amount / (2 if single else 1)
                break
    phg = RULES["phg_family"][proximity] / (2 if single else 1)
    if proximity != "none" and not x.proximity_eligibility_confirmed:
        return finish(
            status="unsupported", notes=["PHG prerequisites have not been confirmed."]
        )
    notes = [
        "Conditional estimate for the confirmed standard scheme; final eligibility and amounts require an HFE letter.",
        "Employment confirmation includes continuous work for the HFE assessment period and employment at application.",
    ]
    return finish(
        enhanced_housing_grant=ehg,
        family_grant=family,
        proximity_housing_grant=phg,
        total_grants=family + ehg + phg,
        eligibility_notes=notes,
        notes=notes,
    )


def calculate_maximum_affordability(
    monthly_income: float | None = None,
    cpf_oa_balance: float | None = None,
    cash_savings: float | None = None,
    loan_type: str = "hdb",
    interest_rate: float | None = None,
    tenure_years: int = 25,
    average_applicant_age: float | None = None,
    youngest_buyer_age: int | None = None,
    oldest_buyer_age: int | None = None,
    remaining_lease_years: float | None = None,
    monthly_debt_obligations: float | None = None,
    total_grants: float | None = None,
    loan_eligibility_confirmed: bool | None = None,
) -> dict:
    """Calculate a purchase-only ceiling at valuation equal to price, excluding taxes and fees. Grants are included once. Requires confirmed usable CPF and loan eligibility; bank rate must include applicable thereafter rates. Not loan approval."""
    x = CalculateMaximumAffordabilityInput(**locals())
    finish = partial(
        result,
        CalculateMaximumAffordabilityOutput,
        ["hdb_loan", "bank_loan", "bank_floor", "cash_downpayment", "cpf"],
    )
    needed = missing(
        x,
        [
            "monthly_income",
            "cpf_oa_balance",
            "cash_savings",
            "average_applicant_age",
            "youngest_buyer_age",
            "oldest_buyer_age",
            "remaining_lease_years",
            "monthly_debt_obligations",
            "total_grants",
            "loan_eligibility_confirmed",
        ],
    )
    if x.loan_type == "bank" and x.interest_rate is None:
        needed.append("interest_rate")
    if needed:
        return finish(status="needs_information", missing_inputs=needed)
    if not x.loan_eligibility_confirmed or x.oldest_buyer_age >= 55:
        return finish(
            status="unsupported",
            notes=[
                "Requires confirmed loan eligibility, no outstanding housing loans and all buyers below 55 with usable CPF limits confirmed."
            ],
        )
    if not x.youngest_buyer_age <= x.average_applicant_age <= x.oldest_buyer_age:
        return finish(
            status="unsupported",
            notes=[
                "Average applicant age must lie between youngest and oldest buyer ages."
            ],
        )
    if x.youngest_buyer_age + x.remaining_lease_years < 95:
        return finish(
            status="unsupported",
            notes=["Short-lease CPF and LTV proration is not supported."],
        )
    if x.loan_type == "hdb" and x.monthly_debt_obligations > 0:
        return finish(
            status="unsupported",
            notes=[
                "HDB credit assessment with existing debts requires the actual HFE-approved loan."
            ],
        )
    hdb = x.loan_type == "hdb"
    term = (
        min(
            D(x.tenure_years),
            D(25),
            65 - D(x.average_applicant_age),
            D(x.remaining_lease_years) - 20,
        )
        if hdb
        else D(x.tenure_years)
    )
    months = int(term * 12)
    if months < 1:
        return finish(
            status="unsupported", notes=["No supported positive loan tenure remains."]
        )
    ltv = D(RULES["standard_ltv"])
    if not hdb and (term > 25 or D(x.average_applicant_age) + term > 65):
        ltv = D(RULES["reduced_bank_ltv"])
    cash_ratio = D(0) if hdb else D("0.05" if ltv == D("0.75") else "0.10")
    actual_rate = D(
        x.interest_rate if x.interest_rate is not None else RULES["hdb_interest_rate"]
    )
    assessment = max(
        actual_rate,
        D(RULES["hdb_assessment_floor" if hdb else "bank_assessment_floor"]),
    )
    msr = D(x.monthly_income) * D(RULES["msr"])
    debt_cap = max(
        D(0), D(x.monthly_income) * D(RULES["tdsr"]) - D(x.monthly_debt_obligations)
    )
    payment = msr if hdb else min(msr, debt_cap)
    rate = assessment / 1200
    annuity = (1 - (1 + rate) ** -months) / rate
    borrowing = (payment * annuity).quantize(CENT, rounding=ROUND_DOWN)
    funds = D(x.cpf_oa_balance) + D(x.cash_savings) + D(x.total_grants)
    # Both constraints must hold: loan <= LTV * price and <= borrowing ceiling.
    bounds = {
        "LTV": funds / (1 - ltv),
        "MSR" if hdb or msr <= debt_cap else "TDSR": funds + borrowing,
    }
    if cash_ratio:
        bounds["DOWNPAYMENT"] = D(x.cash_savings) / cash_ratio
    binding = min(bounds, key=bounds.get)
    price = bounds[binding].quantize(CENT, rounding=ROUND_DOWN)
    actual_loan = min(borrowing, price * ltv).quantize(CENT, rounding=ROUND_DOWN)
    if not hdb and price < funds:
        # A purchase without borrowing does not incur a bank-loan cash minimum.
        price = funds.quantize(CENT, rounding=ROUND_DOWN)
        actual_loan = D(0)
    # A floor at cents can leave a sub-cent funding gap; reduce the price one cent.
    if price > funds + actual_loan:
        price -= CENT
        actual_loan = min(borrowing, price * ltv).quantize(CENT, rounding=ROUND_DOWN)
    actual_monthly_rate = actual_rate / 1200
    instalment = (
        actual_loan / months
        if not actual_monthly_rate
        else actual_loan
        * actual_monthly_rate
        / (1 - (1 + actual_monthly_rate) ** -months)
    )
    return finish(
        max_loan_amount=money(actual_loan),
        borrowing_ceiling=money(borrowing),
        max_purchase_price=money(price),
        monthly_instalment=money(instalment),
        msr_cap_monthly=money(msr),
        ltv_ratio=float(ltv),
        binding_constraint=binding,
        assessment_interest_rate=float(assessment),
        interest_rate=float(actual_rate),
        tenure_months=months,
        grants_included=True,
        purchase_costs_included=False,
        notes=[
            "Purchase-only ceiling: valuation equals price; excludes BSD, ABSD, fees and cash over valuation. Reserve those costs separately and run the upfront tool before recommending a purchase.",
            "CPF input must already reflect retention, age and withdrawal limits. Grants are included once. Approval and any stricter lender limits remain subject to HFE/bank assessment.",
        ],
    )


def buyer_stamp_duty(value: Decimal) -> Decimal:
    remaining, duty = value, D(0)
    for width, rate in RULES["bsd_tiers"]:
        portion = remaining if width is None else min(remaining, D(width))
        duty += portion * D(rate)
        remaining -= portion
        if not remaining:
            break
    return max(D(1), duty.to_integral_value(rounding=ROUND_DOWN))


def calculate_bsd_and_upfront_outlay(
    purchase_price: float,
    loan_type: str = "hdb",
    total_grants: float | None = None,
    market_valuation: float | None = None,
    loan_amount: float | None = None,
    cpf_oa_balance: float | None = None,
    cash_savings: float | None = None,
    legal_and_valuation_fees: float | None = None,
    additional_buyer_stamp_duty: float | None = None,
    ltv_ratio: float | None = None,
) -> dict:
    """Calculate residential BSD and completion funding from actual valuation and financing. Supply separately assessed ABSD, quoted fees, usable CPF and confirmed grants (explicit zero if none). Grants fund purchase only; COV is cash-only. Excludes renovation/insurance/commissions and payment timing."""
    x = CalculateBsdAndUpfrontOutlayInput(**locals())
    finish = partial(
        result,
        CalculateBsdAndUpfrontOutlayOutput,
        ["bsd", "cash_downpayment", "cpf", "family_grant"],
    )
    needed = missing(
        x,
        [
            "market_valuation",
            "loan_amount",
            "cpf_oa_balance",
            "cash_savings",
            "total_grants",
            "legal_and_valuation_fees",
            "additional_buyer_stamp_duty",
            "ltv_ratio",
        ],
    )
    if needed:
        return finish(status="needs_information", missing_inputs=needed)
    price, value, loan = D(x.purchase_price), D(x.market_valuation), D(x.loan_amount)
    basis = min(price, value)
    ltv = D(x.ltv_ratio)
    if (x.loan_type == "hdb" and ltv != D("0.75")) or loan > (basis * ltv).quantize(
        CENT, rounding=ROUND_DOWN
    ):
        return finish(
            status="unsupported",
            notes=[
                "Loan exceeds the supplied LTV limit or uses an unsupported HDB LTV. Short-lease and additional-housing-loan cases are excluded."
            ],
        )
    duty = buyer_stamp_duty(max(price, value))
    cov = max(D(0), price - value)
    mandatory_cash = (
        basis
        * (
            D(0)
            if x.loan_type == "hdb"
            else D("0.05")
            if ltv == D("0.75")
            else D("0.10")
        )
        if loan
        else D(0)
    )
    purchase_eligible = max(D(0), basis - loan - mandatory_cash)
    grants = min(D(x.total_grants), purchase_eligible)
    cpf_purchase = min(D(x.cpf_oa_balance), purchase_eligible - grants)
    # Fees and ABSD conservatively cash-funded; remaining CPF can pay BSD.
    cpf_bsd = min(D(x.cpf_oa_balance) - cpf_purchase, duty)
    cpf_used = cpf_purchase + cpf_bsd
    costs = duty + D(x.legal_and_valuation_fees) + D(x.additional_buyer_stamp_duty)
    total = price - loan + costs
    cash = total - grants - cpf_used
    return finish(
        buyer_stamp_duty=money(duty),
        min_cash_downpayment=money(mandatory_cash),
        cpf_oa_usable=money(cpf_used),
        legal_and_valuation_fees=x.legal_and_valuation_fees,
        total_upfront=money(total),
        cash_required=money(cash),
        cash_shortfall=money(max(D(0), cash - D(x.cash_savings))),
        cash_over_valuation=money(cov),
        grants_used=money(grants),
        unused_grants=money(D(x.total_grants) - grants),
        additional_buyer_stamp_duty=x.additional_buyer_stamp_duty,
        notes=[
            "Completion funding estimate, not a payment schedule. Grants cannot pay BSD, fees or cash over valuation. Fees and ABSD are conservatively cash-funded; CPF eligibility must be confirmed separately.",
            "Excludes insurance, commissions, renovation, option/deposit timing and CPF reimbursement timing. total_upfront includes grants/CPF/cash, excludes the loan.",
        ],
    )
