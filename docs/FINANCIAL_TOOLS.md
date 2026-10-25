# Financial calculators

The three real tools in `mcp_server/financial_calc_tools.py` are available through
the agent registry and the unified FastMCP server. They need no API key or market
database. Installation and server startup remain in the [README](../README.md#setup).

## Rules and results

`config/financial_rules.json` is a reviewed policy snapshot, version
`sg-hdb-resale-2026-10-25.v1`. Responses include its version, verification date
and relevant official HDB, CPF, MAS, MoneySense and IRAS sources. Known effective
dates are recorded separately from the verification date. This version does not
support historical-policy calculations or automatically fetch policy updates.

Inputs reject unknown fields, negative amounts and nonfinite numbers. A result
has `status`, `missing_inputs`, `notes` and nullable financial amounts. `ok` means
the supported conditional estimate was calculated, not that HDB or a lender has
approved it. `needs_information` lists missing facts; `unsupported` explains cases
outside the implemented model. Unknown grants, savings and debts never become
zero. Explicit zero is valid where the schema permits it.

The model supports standard first-timer resale purchases by SC/SC and SC/PR
families, or a single citizen aged 35+ under the Single Singapore Citizen scheme.
Joint singles, mixed first/second-timer households, extended-family schemes,
non-resident spouses, special grant schemes, BTO and lease proration are excluded.
It assumes an ordinary resale flat without additional classification restrictions.
The caller must verify the scheme before setting an eligibility confirmation.

`eligibility_confirmed` for grants attests to the standard household/scheme,
citizenship, first-timer subsidy history, property-ownership and disposal rules,
and other non-income prerequisites in the linked HDB grant documentation. It
does not replace the income, age, lease and employment inputs. Employment means
the required continuous 12-month assessment period and employment at application.
PHG has a separate confirmation for eligible relatives, residence/citizenship,
distance or co-residence, no prior PHG, and all other PHG conditions. The agent
must obtain these facts; saying “couple” does not confirm them.

## Grant calculation

`calculate_cpf_housing_grants` calculates EHG, CPF Housing Grant and PHG from the
reviewed tables. Income is the HFE-assessed monthly average, not an unadjusted
salary quotation. Full EHG requires the remaining lease to cover the youngest
core buyer or core occupier to age 95; provide `youngest_core_member_age`
separately from `youngest_buyer_age`. This release declines shorter-lease cases.

- EHG uses discrete income bands, with maximum family/single grants of
  $120,000/$60,000. Income just above a boundary enters the next band.
- CPF Housing Grant depends on citizenship, household type and flat size.
  This snapshot uses the current $16,000 family/$8,000 single income ceilings.
- PHG is $20,000 near or $30,000 with eligible parents/children for families;
  singles receive half. `proximity="none"` explicitly requests no PHG.
  PHG can still apply above the income ceiling for the other grants.

## Affordability calculation

`calculate_maximum_affordability` accepts assessed income, usable CPF, cash,
confirmed grants, debt obligations, borrower ages, lease and loan assumptions.
`loan_eligibility_confirmed` attests to eligibility for the requested loan and
no outstanding housing loans. All buyers must be below 55 in this release.
CPF input must already account for retention, withdrawal and eligibility limits;
the calculator does not determine CPF eligibility from a raw account balance.

HDB borrowing uses a 30% MSR, 75% LTV and the higher of the supplied rate or 3%
assessment floor. The default repayment rate in this snapshot is 2.6%. Tenure
is the shortest of the request, 25 years, 65 minus average applicant age and
remaining lease minus 20 years. Months are rounded down. HDB cases with other
debts require an actual HFE assessment and return `unsupported` here.

Bank borrowing uses 30% MSR, 55% TDSR after other debts, and an assessment rate
of at least 4%. Supply the applicable highest thereafter rate if higher. Use
the lender-assessed income-weighted age in `average_applicant_age`. For a first
housing loan, LTV is 75%, reduced to 55% for tenure over 25 years or ending
beyond age 65; the corresponding minimum cash is 5%/10%. Maximum tenure is
30 years. Lenders can impose stricter conditions than these ceilings.

The amortisation formula uses Decimal arithmetic. The purchase ceiling respects
the income-based borrowing ceiling, equity/LTV constraint and bank minimum cash.
`borrowing_ceiling` is the income-based limit; `max_loan_amount` is the modelled
loan at `max_purchase_price`. Repayment uses the actual input/default rate,
whereas qualification uses the assessment rate. Amounts that impose limits are
rounded down to cents; displayed repayments are rounded to the nearest cent.

**This is a purchase-only ceiling**, assuming valuation equals price. Grants are
already included (`grants_included=true`). Taxes, fees and cash over valuation
are excluded (`purchase_costs_included=false`). Reserve funds for those costs;
an asking price below this ceiling does not establish that completion is funded.

## BSD and upfront funding

`calculate_bsd_and_upfront_outlay` needs the candidate purchase price, actual
market valuation, actual approved/modelled loan, applicable LTV, usable CPF,
cash, grants, quoted fees and separately assessed ABSD. Supply explicit zero
for ABSD only when independently established; this tool does not assess ABSD.
Do not substitute the income-based borrowing ceiling for the candidate loan.

BSD uses the higher of price and valuation, the residential progressive tiers
in force since 15 February 2023, and rounding down to whole dollars (minimum $1).
The loan is checked against LTV on the lower of price and valuation. Cash over
valuation and bank minimum cash cannot be funded by grants or CPF. Grants fund
eligible purchase equity only; excess grants are reported as unused. Remaining
CPF can fund purchase equity and BSD, subject to the caller's confirmed limits.
Quoted fees and ABSD are conservatively cash-funded.

`total_upfront` is purchase price minus loan plus BSD, ABSD and quoted fees,
before grants and CPF. `cash_required + cpf_oa_usable + grants_used` reconciles
to it. `cash_shortfall` compares required cash with supplied cash savings.
This is completion funding, not an option/deposit or disbursement schedule.
Insurance, commissions, renovation and reimbursement timing are excluded.

## Agent integration and validation

The LLM can call the three tools individually. Without an LLM, the supervisor
uses `user_constraints.financial_inputs.grants`, `.affordability` and `.upfront`
for explicit structured inputs, passing a successful grant result forward when
no confirmed grant amount was supplied. Shared known ages, lease, loan type and
funds are carried forward. Valuation, candidate loan, fees and ABSD are never
invented. Without a positive candidate price, no upfront calculation is attempted.
Incomplete ordinary chat requests return missing-input lists; the deterministic
extractor does not attempt a full financial interview. An LLM or caller must
provide the remaining structured facts. No durable financial interview state
has been added to the web application.

The supervisor does not add grants a second time. It applies the lower of a
stated budget and a known affordability ceiling to candidate prices, including
a legitimate zero ceiling. Missing estimates do not produce a numeric cap.
Legacy mock fixtures remain available for architecture benchmarks; their
illustrative financial formulas are not the real calculators.

Run `python -m pytest tests/test_financial_tools.py -q` for independent official
examples, income boundaries, amortisation checks, BSD reference values, funding
conservation, missing inputs, and registry/agent/in-memory MCP integration.
These tests are not a Category D benchmark accuracy claim. That requires the
agreed evaluation dataset and Member 5's scoring under fixed rule/data versions.

For policy changes: review the linked primary documentation, update the rule
snapshot and verification date, assign a new version, add independent cases for
changed boundaries and rerun the suite before release. Do not silently overwrite
the rules while retaining their old version. Live policy freshness/fallback
search and expanded scenarios remain future work.
