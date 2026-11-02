"""Category D: financial questions (evaluation/build_category_d.py). Tools owned by Member 1, scored by Member 5.

Builds evaluation/category_d_queries.json in the benchmark format (id, category, sub_use_case, query, expected_agents,
expected_mcp_calls, ground_truth_facts). Sub-use cases: D1 CPF housing grants, D2 maximum affordability, D3 BSD and
upfront outlay. Every question is one self-contained household description in the plain-statement wording the app's
extractor reads (see tests/test_area_affordability.py HOUSEHOLD), so no model is needed to read it.

The ground truth is worked out here by calling the real calculators (mcp_server/financial_calc_tools.py) through the
Supervisor's own hand-off (agents.supervisor.run_financial_sequence: a successful grant total feeds affordability, an
affordability LTV feeds the upfront calculator). Facts are never invented: build() checks that the deterministic
extractor recovers exactly the facts each question states, and a fact a question leaves out is a fact the correct
answer must ask for. Some questions are designed to end in `needs_information` or `unsupported`; for those the correct
behaviour is to say so and give no figure.

    .venv/bin/python -m evaluation.build_category_d            # write evaluation/category_d_queries.json
    .venv/bin/python -m evaluation.build_category_d --check    # compare the file with the current rules, write nothing

Amounts depend on the rule snapshot recorded in each item (config/financial_rules.json); rebuild after a rules update.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import supervisor as sv  # noqa: E402
from agents.tool_registry import ToolRunner  # noqa: E402

OUT = PROJECT_ROOT / "evaluation" / "category_d_queries.json"
TOOLS = ("calculate_cpf_housing_grants", "calculate_maximum_affordability", "calculate_bsd_and_upfront_outlay")
SECTION = {"calculate_cpf_housing_grants": "grants", "calculate_maximum_affordability": "affordability",
           "calculate_bsd_and_upfront_outlay": "upfront"}

# Households: ages are (older, younger); the youngest is the youngest buyer and, with no children, the youngest core member.
PEOPLE = {
    "sc_sc": dict(pair="SC_SC", ages=(32, 30), income=9000, cpf=80000, cash=40000),
    "sc_pr": dict(pair="SC_PR", ages=(34, 33), income=7800, cpf=70000, cash=30000),
    "single": dict(pair="SC_Single", ages=(38, 38), income=5500, cpf=90000, cash=30000),
    "high_income": dict(pair="SC_SC", ages=(35, 33), income=17000, cpf=150000, cash=60000),
}

# sub, people, flat type, remaining lease, then overrides: prox (none/near), first (False = second-timer), debt, loan
# (hdb/bank), rate, tenure, cash (None = left out), price/valuation/loan_amt/fees (upfront facts), ask (the final question).
SPECS: List[Dict[str, Any]] = [
    dict(id="D1-01", hh="sc_sc", ft="4 ROOM", lease=70, prox="near",
         ask="How much housing grant can we get for a 4-room resale flat?",
         why="Report the EHG, Family Grant and Proximity Grant from the calculator and their total."),
    dict(id="D1-02", hh="sc_pr", ft="5 ROOM", lease=75, prox="none",
         ask="What grants do we qualify for on a 5-room resale flat?",
         why="Report the SC/PR grants from the calculator (the SC/PR Family Grant is lower than SC/SC) and the total."),
    dict(id="D1-03", hh="single", ft="4 ROOM", lease=70, prox="near",
         ask="What housing grants can I get for a 4-room resale flat?",
         why="Report the Single Singapore Citizen grants (half the family amounts, including the half-rate Proximity Grant)."),
    dict(id="D1-04", hh="high_income", ft="4 ROOM", lease=75, prox="near",
         ask="What grants can we get for a 4-room resale flat?",
         why="Income is above the Family Grant and EHG ceilings: report that only the Proximity Grant applies."),
    dict(id="D1-05", hh="sc_sc", ft="4 ROOM", lease=70, prox="none", first=False,
         ask="What grants can we get for a 4-room resale flat?",
         why="Second-timer households are outside the model: say it is unsupported and needs an HDB assessment; give no grant figure."),
    dict(id="D1-06", hh="sc_sc", ft="4 ROOM", lease=50, prox="none",
         ask="What grants can we get for a 4-room resale flat?",
         why="50 years of lease does not reach the youngest buyer's age 95: say the case is unsupported; give no grant figure."),
    dict(id="D2-01", hh="sc_sc", ft="4 ROOM", lease=70, prox="none",
         ask="How much can we afford to spend on a 4-room resale flat?",
         why="Report the maximum purchase price (HDB loan, MSR binding) with the grants included once."),
    dict(id="D2-02", hh="sc_sc", ft="4 ROOM", lease=70, prox="none", cpf=20000, cash=5000,
         ask="How much can we afford to spend on a 4-room resale flat?",
         why="Savings are low, so the loan-to-value limit binds: report the lower ceiling and its binding constraint."),
    dict(id="D2-03", hh="sc_pr", ft="4 ROOM", lease=80, prox="none", loan="bank", rate=4.5, tenure=25,
         ask="How much can we afford to spend on a 4-room resale flat?",
         why="Bank loan at the stated rate and tenure: report the ceiling and that the minimum cash downpayment is what binds."),
    dict(id="D2-04", hh="sc_sc", ft="4 ROOM", lease=70, prox="none", debt=800,
         ask="How much can we afford to spend on a 4-room resale flat?",
         why="An HDB loan with other debts needs an actual HFE assessment: say it is unsupported; give no ceiling."),
    dict(id="D2-05", hh="sc_sc", ft="4 ROOM", lease=70, prox="none", cash=None,
         ask="How much can we afford to spend on a 4-room resale flat?",
         why="Cash savings are not stated: ask for them instead of giving a maximum price."),
    dict(id="D3-01", hh="sc_sc", ft="4 ROOM", lease=70, prox="none", price=600000, valuation=600000, loan_amt=450000, fees=3500,
         ask="How much stamp duty and cash do we need upfront?",
         why="Report the BSD, the cash required and that there is no shortfall."),
    dict(id="D3-02", hh="sc_sc", ft="5 ROOM", lease=70, prox="none", price=1200000, valuation=1200000, loan_amt=900000, fees=3500,
         ask="How much stamp duty and cash do we need upfront?",
         why="Report the higher BSD tiers and the cash shortfall against the cash savings stated."),
    dict(id="D3-03", hh="sc_sc", ft="4 ROOM", lease=70, prox="none", price=600000, valuation=600000, loan_amt=480000, fees=3500,
         ask="How much stamp duty and cash do we need upfront?",
         why="The loan exceeds the 75% loan-to-value limit: say the scenario is unsupported; give no upfront figure."),
    dict(id="D3-04", hh="sc_sc", ft="4 ROOM", lease=70, prox="none", price=600000, loan_amt=450000, fees=3500,
         ask="How much stamp duty and cash do we need upfront?",
         why="The market valuation is not stated: ask for it; any stamp duty quoted is only on the price, not a final upfront figure."),
]


def _money(v: float) -> str:
    return f"${v:,.0f}"


def household(spec: Dict[str, Any]) -> Dict[str, Any]:
    h = dict(PEOPLE[spec["hh"]])
    for k in ("cpf", "cash"):
        if k in spec:
            h[k] = spec[k]
    return h


def facts_of(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The structured facts the question states, in the supervisor's financial_inputs shape."""
    h = household(spec)
    old, young = h["ages"]
    ages = (young,) if h["pair"] == "SC_Single" else (old, young)
    grants = dict(flat_category="resale", citizenship_pair=h["pair"], first_timer_status=spec.get("first", True),
                  monthly_income=float(h["income"]), flat_type=spec["ft"], remaining_lease_years=float(spec["lease"]),
                  eligibility_confirmed=True, continuous_employment_12_months=True, proximity=spec["prox"],
                  youngest_buyer_age=young, youngest_core_member_age=young)
    if spec["prox"] == "near":
        grants["proximity_eligibility_confirmed"] = True
    aff = dict(monthly_income=float(h["income"]), cpf_oa_balance=float(h["cpf"]), loan_type=spec.get("loan", "hdb"),
               remaining_lease_years=float(spec["lease"]), monthly_debt_obligations=float(spec.get("debt", 0)),
               loan_eligibility_confirmed=True, youngest_buyer_age=min(ages), oldest_buyer_age=max(ages),
               average_applicant_age=sum(ages) / len(ages))
    if spec.get("cash", 1) is not None:
        aff["cash_savings"] = float(h["cash"])
    if "rate" in spec:
        aff["interest_rate"] = float(spec["rate"])
    if "tenure" in spec:
        aff["tenure_years"] = int(spec["tenure"])
    out = {"grants": grants, "affordability": aff}
    if "price" in spec:
        up = dict(purchase_price=float(spec["price"]), loan_amount=float(spec["loan_amt"]),
                  legal_and_valuation_fees=float(spec["fees"]), additional_buyer_stamp_duty=0.0,
                  cpf_oa_balance=float(h["cpf"]), cash_savings=float(h["cash"]), loan_type=aff["loan_type"])
        if "valuation" in spec:
            up["market_valuation"] = float(spec["valuation"])
        out["upfront"] = up
    return out


def question_text(spec: Dict[str, Any]) -> str:
    h = household(spec)
    single = h["pair"] == "SC_Single"
    we, our, have = ("I", "my", "have") if single else ("We", "our", "have")
    if single:
        who = f"I am a single Singapore citizen aged {h['ages'][1]}. I am a first-time buyer. I have no children."
    elif h["pair"] == "SC_PR":
        who = (f"We are a married couple, one Singapore citizen and one permanent resident. We are first-time buyers. "
               f"We are aged {h['ages'][0]} and {h['ages'][1]}. We have no children.")
    else:
        who = (f"We are both Singapore citizens. {'We are first-time buyers.' if spec.get('first', True) else 'We are not first-time buyers.'} "
               f"We are aged {h['ages'][0]} and {h['ages'][1]}. We have no children.")
    income = f"{'My' if single else 'Our household'} income is {_money(h['income'])} a month."
    money = f"{our.capitalize()} CPF OA balance is {_money(h['cpf'])}."
    if spec.get("cash", 1) is not None:
        money += f" {we} {have} {_money(h['cash'])} in cash savings."
    debt = (f"{we} {have} a monthly debt payment of {_money(spec['debt'])}." if spec.get("debt")
            else f"{we} {have} no monthly debt.")
    work = f"{we} {'have' if single else 'have both'} worked continuously for the past 12 months."
    prox = (f"{we} {'am' if single else 'are'} buying near {our} parents." if spec["prox"] == "near"
            else f"{we} {'am' if single else 'are'} not buying near {our} parents.")
    loan = "bank" if spec.get("loan") == "bank" else "HDB"
    conf = f"{our.capitalize()} grant eligibility is confirmed."
    if spec["prox"] == "near":
        conf += f" {our.capitalize()} proximity grant eligibility is confirmed."
    conf += f" {our.capitalize()} {loan} loan eligibility is confirmed."
    extra = ""
    if "rate" in spec:
        extra += f" Use an interest rate of {spec['rate']}%."
    if "tenure" in spec:
        extra += f" Use a tenure of {spec['tenure']} years."
    extra += f" Use {spec['lease']} years of remaining lease for the calculation."
    if "price" in spec:
        flat = spec["ft"].split()[0] + "-room"
        buy = f"{we} {'am' if single else 'are'} buying a {flat} resale flat for {_money(spec['price'])}"
        if "valuation" in spec:
            buy += f", valued at {_money(spec['valuation'])}"
        buy += f", with a loan of {_money(spec['loan_amt'])} and fees of {_money(spec['fees'])}. There is no ABSD."
        extra = " " + buy + extra
    return f"{who} {income} {money} {debt} {work} {prox} {conf}{extra} {spec['ask']}"


def calculators() -> Dict[str, ToolRunner]:
    return {name: ToolRunner(name) for name in TOOLS}


def run_tools(spec: Dict[str, Any], facts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    constraints = {"financial_inputs": facts, "flat_types": [spec["ft"]]}
    return sv.run_financial_sequence(calculators(), constraints, question_text(spec))


KEEP = {
    "calculate_cpf_housing_grants": ("status", "missing_inputs", "enhanced_housing_grant", "family_grant",
                                     "proximity_housing_grant", "total_grants", "notes"),
    "calculate_maximum_affordability": ("status", "missing_inputs", "max_purchase_price", "max_loan_amount", "monthly_instalment",
                                        "binding_constraint", "ltv_ratio", "notes"),
    "calculate_bsd_and_upfront_outlay": ("status", "missing_inputs", "buyer_stamp_duty", "total_upfront", "cash_required",
                                         "cash_shortfall", "notes"),
}


def check_extraction(facts: Dict[str, Dict[str, Any]], text: str) -> List[str]:
    """The deterministic extractor must read every stated fact of the question back, with the same value."""
    read = sv.stated_financial_inputs(text)
    problems = []
    for sec, want in facts.items():
        got = read.get(sec) or {}
        for k, v in want.items():
            if k in ("flat_category", "flat_type"):
                continue
            if got.get(k) != v:
                problems.append(f"{sec}.{k}: wrote {v!r}, extractor read {got.get(k)!r}")
    return problems


def ground(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    text = question_text(spec)
    facts = facts_of(spec)
    problems = check_extraction(facts, text)
    outputs = run_tools(spec, facts)
    truth: Dict[str, Any] = {}
    for name, out in outputs.items():
        truth[name] = {k: out.get(k) for k in KEEP[name]}
    versions = {o.get("rule_version") for o in outputs.values()}
    statuses = {SECTION[n]: o.get("status") for n, o in truth.items()}
    sub = spec["id"][:2]
    main = {"D1": "calculate_cpf_housing_grants", "D2": "calculate_maximum_affordability",
            "D3": "calculate_bsd_and_upfront_outlay"}[sub]
    if main not in outputs:
        problems.append(f"{main} did not run")
    calls = [{"agent": "financial", "tool": name, "required_params": dict(facts[SECTION[name]])} for name in outputs]
    verdict = statuses.get(SECTION[main])
    item = {
        "id": spec["id"], "category": "D", "sub_use_case": sub, "query": text, "initial_map_state": None,
        "financial_inputs": facts, "expected_agents": ["financial"], "expected_mcp_calls": calls,
        "expected_behaviour": spec["why"],
        "ground_truth_facts": {"rule_version": sorted(v for v in versions if v)[0] if versions else None,
                               "target_tool": main, "expected_status": verdict, "statuses": statuses,
                               "give_figure": verdict == "ok", **truth},
    }
    return item, problems


def build() -> List[Dict[str, Any]]:
    items = []
    for spec in SPECS:
        item, problems = ground(spec)
        if problems:
            raise SystemExit(f"{spec['id']}: " + "; ".join(problems))
        items.append(item)
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    built = build()
    if args.check:
        same = json.loads(OUT.read_text(encoding="utf-8")) == json.loads(json.dumps(built))
        print("category_d_queries.json matches the calculators" if same else "category_d_queries.json is out of date")
        sys.exit(0 if same else 1)
    OUT.write_text(json.dumps(built, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(built)} questions to {OUT}")
