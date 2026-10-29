"""
Agent 1 — Supervisor & Constraint Negotiator, plus the specialist runner
(agents/supervisor.py). Owned by Member 4. Proposal §4.3 (Agent 1), §2 Category E.

Everything here is a plain function over HDBCompassState so the three
architecture variants can compose the same pieces differently:

  extract_constraints_node  NL -> UserConstraints   (LLM structured output, regex fallback)
  route_node                constraints -> active_subagents (deterministic, explainable)
  dispatch                  fan-out: one Send per active sub-agent (parallel)
  make_specialist_node      ReAct loop over ONE agent's tools, or a deterministic plan
  intersect_node            SQL blocks ∩ spatial blocks on block_key, budget-capped
  should_relax / relax_node zero-result sensitivity sweep (use case E2)
  synthesize_node           draft_response + map_ui_payload (use case E1)
  audit_node / guardrail    hook points for Member 5 (default: pass-through)
  finalize_node             appends the assistant message

Deterministic mode (LLM_PROFILE=none) exercises every path without a model, so
the graph is unit-testable and the "expected_mcp_calls" for Category E golden
queries can be generated from the same plans.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm import active_profile, get_llm, load_config
from agents.schemas import (
    TOOL_OWNER,
    AuditorVerdict,
    ExtractedConstraints,
    RelaxationSuggestion,
    RelaxUnsatisfiedConstraintsOutput,
    block_key,
)
from agents.state import HDBCompassState
from agents.tool_registry import ToolRunner, build_runners, tools_for_agent

SPECIALISTS = ["market_sql", "spatial", "financial", "policy_rag"]
SLOT_FOR_AGENT = {
    "market_sql": "sql_market_results",
    "spatial": "spatial_block_candidates",
    "financial": "financial_breakdown",
    "policy_rag": "rag_policy_passages",
}

HDB_TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", "BUKIT TIMAH",
    "CENTRAL AREA", "CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", "SENGKANG",
    "SERANGOON", "TAMPINES", "TOA PAYOH", "WOODLANDS", "YISHUN",
]
TOWN_ALIASES = {"TIONG BAHRU": "BUKIT MERAH", "KALLANG": "KALLANG/WHAMPOA", "WHAMPOA": "KALLANG/WHAMPOA",
                "REDHILL": "BUKIT MERAH", "TELOK BLANGAH": "BUKIT MERAH", "COMMONWEALTH": "QUEENSTOWN"}
# short forms people actually type, matched as whole upper-case words so "JE" never matches inside another word
TOWN_SHORT = {"AMK": "ANG MO KIO", "TPY": "TOA PAYOH", "JE": "JURONG EAST", "JW": "JURONG WEST", "CCK": "CHOA CHU KANG",
              "BP": "BUKIT PANJANG", "SK": "SENGKANG", "PG": "PUNGGOL"}
NUMBER_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5"}
AMENITY_WORDS = {"mrt": "mrt", "train": "mrt", "hawker": "hawker", "bus stop": "bus_stop", "bus": "bus_stop",
                 "supermarket": "supermarket", "park": "park", "school": "school", "clinic": "clinic"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sys_msg(text: str) -> SystemMessage:
    """System message with the active profile's suffix (e.g. Qwen3 '/no_think')."""
    try:
        suffix = active_profile().get("system_suffix") or ""
    except Exception:  # noqa: BLE001
        suffix = ""
    return SystemMessage(content=text + suffix)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> str:
    """Remove Qwen3 reasoning blocks if a server returns them inside the content."""
    return _THINK_RE.sub("", text or "").strip()


def last_user_query(state: HDBCompassState) -> str:
    for m in reversed(state.get("messages", [])):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        if role in ("user", "human"):
            return m.get("content") if isinstance(m, dict) else m.content
    return ""


def _money(s: str) -> Optional[float]:
    m = re.search(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\b", s, re.I)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    return v * (1000 if unit in ("k", "thousand") else 1_000_000 if unit in ("m", "million", "mil") else 1)


def rule_based_extract(query: str) -> ExtractedConstraints:
    """Regex extractor: the no-LLM path and the safety net when structured output fails."""
    q = query
    qu = q.upper()
    towns = [t for t in HDB_TOWNS if t in qu]
    for alias, town in TOWN_ALIASES.items():
        if alias in qu and town not in towns:
            towns.append(town)
    for short, town in TOWN_SHORT.items():
        if re.search(rf"\b{short}\b", q) and town not in towns:   # case-sensitive: "AMK", not "amk" inside a word
            towns.append(town)
    for word, digit in NUMBER_WORDS.items():
        q = re.sub(rf"\b{word}(?=[- ]?(?:room|rm)\b)", digit, q, flags=re.I)
    flat_types = [f"{n} ROOM" for n in re.findall(r"\b([2-5])\s*[- ]?\s*(?:room|rm)s?\b", q, re.I)]
    if re.search(r"\bexecutive\b", q, re.I):
        flat_types.append("EXECUTIVE")
    flat_types = list(dict.fromkeys(flat_types))

    budget = None
    # "under 650k", "<650k", "budget up to $0.65m", "no more than 650 thousand"; a bare "400m" is metres, not money
    for m in re.finditer(r"(?:under|below|less than|max(?:imum)?|up to|budget(?: of)?(?: up to)?|no more than|at most|<|≤)"
                         r"\s*(\$)?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\b", q, re.I):
        unit = (m.group(3) or "").lower()
        if unit == "m" and not m.group(1) and float(m.group(2).replace(",", "")) >= 10:
            continue                                                     # metres
        value = _money(f"{m.group(2)}{unit}")
        if value and value >= 50_000:                                    # an HDB budget, not a distance or an age
            budget = value
            break

    lease = None
    # "at least 60 yrs lease left", "65+ years of lease", "lease 65+ yrs", "remaining lease of 65 years or more"
    for pat in (r"(?:at least|minimum|min\.?|>=|≥)\s*(\d{2})\s*(?:years?|yrs?)",
                r"(\d{2})\s*\+\s*(?:years?|yrs?)",
                r"\blease\s*(?:of\s*)?(?:at least\s*|>=\s*|≥\s*)?(\d{2})\s*\+?\s*(?:years?|yrs?)",
                r"(\d{2})\s*(?:years?|yrs?)\s*(?:of\s*)?(?:remaining\s*)?lease"):
        m = re.search(pat, q, re.I)
        if m:
            lease = int(m.group(1))
            break

    radii: Dict[str, int] = {}
    for m in re.finditer(r"(?:within\s*)?(\d+(?:\.\d+)?)\s*(km|m)\b\s*(?:of|from|to)\s+(?:an?\s+|the\s+)?([a-z /-]+?)(?=[,.;?!]|\s+(?:with|within|that|which|under|over|have|has)\b|\s+and\s+(?:\d|have|has|at\s+least|within|no more than|max|≤|<)|$)", q, re.I):
        dist = float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1)
        targets = m.group(3).lower()
        for word, key in AMENITY_WORDS.items():
            if word in targets:
                radii[key] = int(dist)
    # "MRT within 400m"; never across "and", so "mrt and within 500m of a hawker centre" is not read as the MRT limit
    for m in re.finditer(r"\b(mrt|hawker|supermarket|bus stop|park)(?:(?!\band\b)[a-z ])*?within\s*(\d+)\s*m\b", q, re.I):
        radii[AMENITY_WORDS[m.group(1).lower()]] = int(m.group(2))
    # "≤400m MRT", "no more than 400 metres from an MRT station", "max 500m to a hawker centre"
    for m in re.finditer(r"(?:≤|<=|<|within|max(?:imum)?|no more than|at most)\s*(\d+(?:\.\d+)?)\s*(km|m|metres?|meters?)\b"
                         r"\s*(?:of|from|to)?\s*(?:an?\s+|the\s+)?(mrt|train|hawker|supermarket|bus stop|park|clinic)", q, re.I):
        dist = float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1)
        radii.setdefault(AMENITY_WORDS[m.group(3).lower()], int(dist))

    school = None
    m = re.search(r"(?:of|near|to)\s+((?:[A-Z][a-zA-Z'’]+\s){1,4}(?:Primary School|School|Primary|High))", q)
    if m:
        school = m.group(1).strip()
        radii.setdefault("school", 1000)

    income = None
    m = re.search(r"(?:earn(?:ing)?|income(?: of)?|salary(?: of)?)\s*(?:is\s*|:\s*)?\$?\s?(\d[\d,]*(?:\.\d+)?\s*k?)", q, re.I)
    if m:
        income = _money(m.group(1))
    if income is None:
        m = re.search(r"\$\s?(\d[\d,]*(?:\.\d+)?\s*k?)\s*(?:/|per|a)\s*month", q, re.I)
        if m:
            income = _money(m.group(1))

    cpf = None
    m = re.search(r"\$?\s?(\d[\d,]*(?:\.\d+)?\s*k?)\s*(?:in\s*)?(?:cpf|oa)", q, re.I)
    if m:
        cpf = _money(m.group(1))

    scheme = None
    if re.search(r"\bsingle\b", q, re.I):
        scheme = "SC_Single_35"
    elif re.search(r"\bPR\b|permanent resident", q, re.I):
        scheme = "SC_PR_FirstTimer"
    elif re.search(r"citizen|couple|first[- ]timer|family", q, re.I):
        scheme = "SC_SC_FirstTimer"

    work = [w.strip() for w in re.findall(r"work(?:s)?\s+(?:at|in)\s+([A-Za-z0-9 \-']+?)(?=[,.;]|\s+and\b|$)", q, re.I)]
    max_commute = None
    m = re.search(r"(?:under|within|less than|max)\s*(\d{2})\s*min", q, re.I)
    if m:
        max_commute = int(m.group(1))

    policy = None
    # "plus" only in the Plus-classification sense: "plus $30,000 in cash" is not a policy question.
    if re.search(r"\b(mop|minimum occupation|prime|plus (?:flats?|models?|bto|resale|classification)|standard flat|clawback|subsidy recovery|eip|ethnic|quota|wait[- ]out|eligib|scheme|rental restriction|exempt"
                 r"|not married|unmarried|fianc[eé]e?s?|apply(?:ing)? for (?:an? )?(?:[2-5][- ]room )?btos?|can (?:we|i) (?:buy|apply))\b", q, re.I):
        policy = q

    return ExtractedConstraints(
        towns=towns, flat_types=flat_types, max_budget_sgd=budget, min_remaining_lease_years=lease,
        amenity_radii_meters=radii, household_income=income, citizenship_scheme=scheme, cpf_oa_balance=cpf,
        school_name=school, work_destinations=work, max_commute_minutes=max_commute, policy_question=policy,
        wants_price_trend=bool(re.search(r"trend|appreciat|per sqm|psm|over the last|median price", q, re.I)),
        wants_bto_comparison=bool(re.search(r"\bbto\b.*(?:vs|versus|against|compar)|compar.*\bbto\b", q, re.I)),
    )


# ---------------------------------------------------------------------------
# Node: extract_constraints
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM = (
    "You extract structured housing-search constraints from a Singapore HDB buyer's message. "
    "Fill only what the user stated; leave the rest null/empty. Towns are HDB town names in upper case "
    "(e.g. BISHAN, TAMPINES, BUKIT MERAH for Tiong Bahru). Flat types like '4 ROOM'. Money in SGD as numbers. "
    "amenity_radii_meters keys: mrt, hawker, bus_stop, supermarket, park, school. "
    "policy_question: copy the regulatory question if the user asks about rules, MOP, grants eligibility, EIP, "
    "Prime/Plus/Standard, wait-out periods."
    " Financial inputs contain only explicit user facts: never infer citizenship from 'couple', "
    "first-timer status, CPF usability, no debts, zero ABSD, fees, grant/loan eligibility or confirmations. "
    "Use financial_inputs.grants/affordability/upfront for supplied calculator facts. "
    "For comparable transactions, use comparable_resale with explicit verified latitude, longitude and flat type; "
    "leave comparable_resale null unless the user gives coordinates. "
    "Never invent coordinates or substitute a town centre for a project."
    # how real users write (the varied-phrasing test set): short forms the model should read the same way
    " Users often write casually. Read 'AMK' as ANG MO KIO, 'TPY' as TOA PAYOH, 'JE' as JURONG EAST; "
    "'4rm', '4-rm', 'four-room' as '4 ROOM'; '650k', '$0.65m', '650 thousand' as 650000; '<650k', 'under 650k', "
    "'budget up to $0.65m' as max_budget_sgd; 'lease 65+ yrs', 'at least 65 years left', 'remaining lease of 65 "
    "years or more' as min_remaining_lease_years 65; '<=400m MRT', 'no more than 400 metres from an MRT station' as "
    "amenity_radii_meters.mrt 400. Leave amenity_radii_meters empty unless a distance is stated; never use 0. "
    "Money facts: 'SC couple' or 'both Singapore citizens' is citizenship_pair SC_SC, 'SC + PR' or 'one citizen, "
    "one PR' is SC_PR, a single citizen is SC_Single; '1st timers' or 'first-time buyers' is first_timer_status "
    "true; 'no PHG' or 'no proximity grant' is proximity 'none'; 'cash 40k' is cash_savings 40000; 'we're 32 & 30' "
    "is youngest_buyer_age 30 and oldest_buyer_age 32; 'employed 12 mths continuously' is "
    "continuous_employment_12_months true; 'calc w 65 yrs lease' is remaining_lease_years 65."
)


FIN_SECTIONS = {"grants": "calculate_cpf_housing_grants", "affordability": "calculate_maximum_affordability"}


def _section_fields(section: str) -> set:
    from agents.schemas import TOOL_INPUT_MODEL
    return set(TOOL_INPUT_MODEL[FIN_SECTIONS[section]].model_fields)


def merge_constraints(prior: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """The previous turn's constraints, updated with anything this message states. financial_inputs merges per field."""
    out = dict(prior or {})
    for k, v in (new or {}).items():
        if k == "financial_inputs":
            fin = {sec: dict(vals or {}) for sec, vals in (out.get("financial_inputs") or {}).items()}
            for sec, vals in (v or {}).items():
                fin.setdefault(sec, {}).update({f: x for f, x in (vals or {}).items() if x is not None})
            out["financial_inputs"] = fin
        elif v not in (None, [], {}, False):
            out[k] = v
    return out


# "aged 32 and 30", "we're 32 & 30", "ages: 32 and 30", "I am 38"
_AGES_RE = re.compile(r"\b(?:aged?|ages?:?|are|am|we're|i'm|we are)\s+(\d{2})(?:\s*(?:and|&|,|/)\s*(\d{2}))?\b(?!\s*(?:years?|yrs?)\s+(?:of\s+)?(?:remaining\s+)?lease)(?!\s*k\b|\s*%)", re.I)
_CASH_RE = re.compile(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:sgd\s*)?(?:in\s+)?(?:cash|savings)\b"
                      r"|\b(?:cash|savings)\b[^\d\n,;.]{0,20}?\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?", re.I)
# "$90,000 in CPF OA", "40k cpf", "cpf of ard 40k"; a comma or full stop in between ends the match
_CPF_RE = re.compile(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:sgd\s*)?(?:in\s+)?(?:(?:my|our)\s+)?(?:cpf|oa)\b"
                     r"|\b(?:cpf|oa)\b[^\d\n,;.]{0,20}?\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?", re.I)


def _re_amount(m: re.Match) -> float:
    amount, k = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return float(amount.replace(",", "")) * (1000 if k else 1)


def parse_clarification_answer(text: str, needed: List[str], conversation: str = "") -> Dict[str, Dict[str, Any]]:
    """No-model reading of a reply to the follow-up question, for the facts the calculators most often lack (cash,
    ages). Only stated facts are recorded; the youngest core member equals the youngest buyer only when the
    conversation says there are no children. The model path reads any fact through structured extraction."""
    found: Dict[str, Any] = {}
    m = _CASH_RE.search(text or "")
    if m and "cash_savings" in needed:
        amount, k = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        found["cash_savings"] = float(amount.replace(",", "")) * (1000 if k else 1)
    m = _AGES_RE.search(text or "")
    if m and any("age" in n for n in needed):
        ages = [int(a) for a in m.groups() if a]
        young, old = min(ages), max(ages)
        found.update(youngest_buyer_age=young, oldest_buyer_age=old, average_applicant_age=sum(ages) / len(ages))
        if re.search(r"\bno (?:children|kids)\b", f"{conversation} {text}", re.I):
            found["youngest_core_member_age"] = young
    if re.search(r"\bno (?:other )?debts?\b", text or "", re.I) and "monthly_debt_obligations" in needed:
        found["monthly_debt_obligations"] = 0.0
    return {sec: {f: v for f, v in found.items() if f in _section_fields(sec)} for sec in FIN_SECTIONS
            if any(f in _section_fields(sec) for f in found)}


# Calculator facts the user must state. Code reads them from the user's words (stated_financial_inputs); the
# model's value for any of them is never used, because in the live run it turned "married Singapore citizen
# couple" into SC_Single, invented ages and an interest rate, and passed zero grants. The model may still set
# the remaining fields (flat category and type, loan type, the purchase price to cost).
FACT_FIELDS = {
    "citizenship_pair", "first_timer_status", "monthly_income", "youngest_buyer_age", "youngest_core_member_age",
    "oldest_buyer_age", "average_applicant_age", "remaining_lease_years", "eligibility_confirmed",
    "loan_eligibility_confirmed", "continuous_employment_12_months", "proximity", "proximity_eligibility_confirmed",
    "cpf_oa_balance", "cash_savings", "monthly_debt_obligations", "total_grants", "interest_rate", "tenure_years",
    "legal_and_valuation_fees", "additional_buyer_stamp_duty", "ltv_ratio",
}
FIN_TOOL_SECTION = {"calculate_cpf_housing_grants": "grants", "calculate_maximum_affordability": "affordability",
                    "calculate_bsd_and_upfront_outlay": "upfront"}
_AMOUNT = r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?"


def _amount(m: re.Match) -> float:
    return float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)


def stated_financial_inputs(text: str) -> Dict[str, Dict[str, Any]]:
    """The calculator facts the user's own words state, in the financial_inputs shape. Nothing is inferred:
    "couple" alone gives no citizenship, and the youngest core member is the youngest buyer only when the
    user says there are no children."""
    t = text or ""
    f: Dict[str, Any] = {}
    citizen = r"(?:citizens?|singaporeans?|\bSC\b)"
    if re.search(rf"{citizen}.{{0,40}}\b(?:permanent resident|PR)\b|\b(?:permanent resident|PR)\b.{{0,40}}{citizen}", t, re.I):
        f["citizenship_pair"] = "SC_PR"
    elif re.search(rf"\bsingle\s+(?:singapore(?:an)?\s+)?{citizen}|\bi'?m single, (?:a )?singapore citizen", t, re.I):
        f["citizenship_pair"] = "SC_Single"
    elif re.search(r"\bsingapore(?:an)?\s+citizen\s+couple\b|\bboth\s+(?:are\s+)?singapore(?:an)?\s+citizens\b"
                   r"|\bSC couple\b|\bboth\s+(?:are\s+)?singaporeans\b", t, re.I):
        f["citizenship_pair"] = "SC_SC"
    if re.search(r"\bsecond[- ]timers?\b|\b2nd[- ]timers?\b", t, re.I):
        f["first_timer_status"] = False
    elif re.search(r"\bfirst[- ]?timers?\b|\b1st[- ]?timers?\b|\bfirst[- ]time (?:home )?buyers?\b", t, re.I):
        f["first_timer_status"] = True
    m = _AGES_RE.search(t)
    if m:
        ages = [int(a) for a in m.groups() if a]
        f.update(youngest_buyer_age=min(ages), oldest_buyer_age=max(ages), average_applicant_age=sum(ages) / len(ages))
        if re.search(r"\bno (?:children|kids|child)\b|\bwithout (?:children|kids)\b", t, re.I):
            f["youngest_core_member_age"] = min(ages)
    rb = rule_based_extract(t)
    if rb.household_income is not None:
        f["monthly_income"] = rb.household_income
    m = _CPF_RE.search(t)
    if m:
        f["cpf_oa_balance"] = _re_amount(m)
    elif rb.cpf_oa_balance is not None:
        f["cpf_oa_balance"] = rb.cpf_oa_balance
    m = _CASH_RE.search(t)
    if m:
        f["cash_savings"] = _re_amount(m)
    if re.search(r"\bno (?:other )?(?:debts?|loans?|monthly (?:debt|loan) (?:payments|obligations))\b", t, re.I):
        f["monthly_debt_obligations"] = 0.0
    if re.search(r"(?:continuous(?:ly)?\s+(?:employ\w*|working)|(?:employ\w*|working)\s+(?:\w+\s+){0,2}continuously)"
                 r".{0,40}\b(?:12|twelve)\s*(?:months?|mths?)\b"
                 r"|(?:employ\w*|working)\s+(?:12|twelve)\s*(?:months?|mths?)\s+continuously", t, re.I):
        f["continuous_employment_12_months"] = True
    if re.search(r"eligibility for (?:the )?grants?.{0,40}\bconfirmed\b|\bgrants?\b.{0,30}\beligibility\b.{0,25}\bconfirmed\b", t, re.I):
        f["eligibility_confirmed"] = True
    if re.search(r"eligibility for .{0,30}\b(?:HDB|bank) loan (?:is |has been )?confirmed"
                 r"|\b(?:HDB|bank) loan\b.{0,15}\beligibility\b.{0,25}\bconfirmed\b"
                 r"|\b(?:HDB|bank) loan(?: eligibility)? (?:is |has been )?confirmed", t, re.I):
        f["loan_eligibility_confirmed"] = True
    if re.search(r"\bbank loan\b", t, re.I):
        f["loan_type"] = "bank"
    elif re.search(r"\bHDB loan\b", t, re.I):
        f["loan_type"] = "hdb"
    if re.search(r"\bno proximity (?:housing )?grant\b|\bno PHG\b|\bproximity (?:housing )?grant (?:does not|doesn't) apply\b", t, re.I):
        f["proximity"] = "none"
    if re.search(r"\bno (?:housing )?grants\b", t, re.I):
        f["total_grants"] = 0.0
    # the lease the user wants the calculation to assume: "use 65 yrs remaining lease for the calc", "calc w 65 yrs
    # lease", "using a 65-year remaining lease", "assume 65 years of remaining lease"
    m = re.search(r"(\d{2})\s*(?:years?|yrs?) of remaining lease for the .{0,30}calc"
                  r"|\b(?:assume|use|using)\s+(?:a\s+)?(\d{2})[- ]?(?:years?|yrs?)\b(?:\s+of)?(?:\s+remaining)?\s+lease"
                  r"|\bcalc\w*\s+(?:w|with|using)\s+(\d{2})\s*(?:years?|yrs?)\b(?:\s+of)?(?:\s+remaining)?\s+lease", t, re.I)
    if m:
        f["remaining_lease_years"] = float(next(g for g in m.groups() if g))
    m = re.search(r"interest rate of (\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%\s*(?:p\.a\.|per annum|interest)", t, re.I)
    if m:
        f["interest_rate"] = float(m.group(1) or m.group(2))
    m = re.search(r"tenure of (\d{1,2}) years|(\d{1,2})[- ]year (?:loan )?tenure", t, re.I)
    if m:
        f["tenure_years"] = int(m.group(1) or m.group(2))
    if re.search(r"\bresale\b", t, re.I):
        f["flat_category"] = "resale"
    elif re.search(r"\bbto\b", t, re.I):
        f["flat_category"] = "bto"
    if len(rb.flat_types) == 1:
        f["flat_type"] = rb.flat_types[0]
    upfront: Dict[str, Any] = {}
    m = re.search(r"(?:buy(?:ing)?|purchas(?:e|ing))\b[^$.]{0,40}\bfor " + _AMOUNT, t, re.I)
    if m:
        upfront["purchase_price"] = _amount(m)
    for field, pattern in (("market_valuation", r"valued at " + _AMOUNT), ("loan_amount", r"loan of " + _AMOUNT),
                           ("legal_and_valuation_fees", r"fees of " + _AMOUNT)):
        m = re.search(pattern, t, re.I)
        if m:
            upfront[field] = _amount(m)
    if re.search(r"\bno ABSD\b|\bno additional buyer'?s stamp duty\b", t, re.I):
        upfront["additional_buyer_stamp_duty"] = 0.0
    from agents.schemas import TOOL_INPUT_MODEL
    out: Dict[str, Dict[str, Any]] = {}
    for tool, sec in FIN_TOOL_SECTION.items():
        fields = set(TOOL_INPUT_MODEL[tool].model_fields)
        vals = {k: v for k, v in {**f, **upfront}.items() if k in fields}
        # a section exists only when the user stated one of its facts, not just a flat type or loan type
        if any(k in FACT_FIELDS or k in upfront for k in vals):
            out[sec] = vals
    if "upfront" in out and "purchase_price" not in out["upfront"]:
        del out["upfront"]  # the upfront calculator needs a price; costing the budget is the agent's call
    return out


_NUM_RE = re.compile(r"(?<![\w.])\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\b", re.I)
_COUPLE = r"\b(?:we|we're|our|us|couple|wife|husband|spouse|partner|fianc[eé]e?|both)\b"
_CITIZEN = r"(?:citizens?|singaporeans?|\bSC\b)"
_NO_KIDS = r"\bno (?:children|kids|child)\b|\bwithout (?:children|kids)\b|\bchildless\b"


def _stated_numbers(text: str) -> List[Tuple[float, bool]]:
    """Every amount in the text as (value, had_unit): "150k" -> 150000, "$0.65m" -> 650000, "650 thousand", "32"."""
    out = []
    for m in _NUM_RE.finditer(text or ""):
        unit = (m.group(2) or "").lower()
        v = float(m.group(1).replace(",", ""))
        if unit == "m" and v >= 10:          # "400m" is metres
            out.append((v, False))
            continue
        mult = 1000 if unit in ("k", "thousand") else 1_000_000 if unit in ("m", "million", "mil") else 1
        out.append((v * mult, bool(unit)))
    return out


def _number_stated(value: Any, text: str, plain_only: bool = False) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(n - v) <= max(0.5, 0.005 * abs(v)) for n, unit in _stated_numbers(text) if not (plain_only and unit))


def fact_is_grounded(field: str, value: Any, text: str) -> bool:
    """True when the user's words support this calculator fact as read by the model. Numbers must appear in the text
    in some form; yes/no and choice facts need a supporting phrase. Derived values are never taken from the model."""
    t = text or ""
    has = lambda pat: re.search(pat, t, re.I) is not None  # noqa: E731
    if field in ("average_applicant_age", "ltv_ratio"):
        return False                                   # worked out by code / the calculators
    if field in ("youngest_buyer_age", "oldest_buyer_age"):
        return _number_stated(value, t, plain_only=True)
    if field == "youngest_core_member_age":
        return _number_stated(value, t, plain_only=True) and has(_NO_KIDS)
    if field == "monthly_debt_obligations":
        if value == 0:
            return has(r"\bno (?:other )?(?:debts?|loans?|outstanding loans?|car loans?|liabilities)\b|\bdebt[- ]free\b|\bzero debts?\b")
        return _number_stated(value, t)
    if field == "total_grants":
        return (value == 0 and has(r"\bno (?:housing )?grants\b")) or \
            (_number_stated(value, t) and has(r"\bgrants?\b"))
    if field == "additional_buyer_stamp_duty":
        return (value == 0 and has(r"\bno ABSD\b|\bno additional buyer'?s stamp duty\b")) or _number_stated(value, t)
    if field == "interest_rate":
        return _number_stated(value, t) and has(r"%|\binterest\b")
    if field == "tenure_years":
        return _number_stated(value, t, plain_only=True) and has(r"\btenure\b|[- ]year loan\b")
    if field in ("monthly_income", "cpf_oa_balance", "cash_savings", "remaining_lease_years", "legal_and_valuation_fees"):
        return _number_stated(value, t)
    if field == "citizenship_pair":
        pr = has(r"\bpermanent residents?\b|\bPR\b")
        if value == "SC_PR":
            return pr and has(_CITIZEN)
        if value == "SC_SC":
            return not pr and has(_CITIZEN) and has(_COUPLE)
        if value == "SC_Single":
            return not pr and has(_CITIZEN) and has(r"\bsingle\b") and not has(r"\b(?:wife|husband|spouse|fianc[eé]e?|couple)\b")
        return False
    if field == "first_timer_status":
        if value is True:
            return has(r"\bfirst[- ]?tim|\b1st[- ]?tim|\bfirst (?:home|flat)\b|\bnever owned\b")
        return has(r"\bsecond[- ]?tim|\b2nd[- ]?tim|\bowned (?:a flat|an? hdb|property) before\b")
    if field == "eligibility_confirmed":
        return value is True and has(r"\beligib") and has(r"\bconfirm")
    if field == "loan_eligibility_confirmed":
        return value is True and has(r"\bloan\b") and has(r"\beligib") and has(r"\bconfirm")
    if field == "continuous_employment_12_months":
        return value is True and has(r"\b(?:employ\w*|working|work|job)\b") and \
            has(r"\b(?:12|twelve)\s*(?:months?|mths?)\b|\b(?:a|one|1|past|last) year\b")
    if field == "proximity":
        if value == "none":
            return has(r"\bno proximity\b|\bno PHG\b|\bproximity (?:housing )?grant (?:does not|doesn't|won't) apply"
                       r"|\bnot (?:living )?near (?:my|our) parents\b")
        return has(r"\bparents?\b|\bchildren\b") and has(r"\bnear\b|\bwith\b|\blive\b")
    if field == "proximity_eligibility_confirmed":
        return value is True and has(r"\bproximity\b") and has(r"\bconfirm")
    return False


def trusted_financial_inputs(model: Optional[Dict[str, Any]], stated: Dict[str, Dict[str, Any]],
                             text: str = "") -> Optional[Dict[str, Any]]:
    """The calculator facts to use: the model's reading kept only where the user's words support it
    (fact_is_grounded), then the code's own reading on top (stated_financial_inputs; it wins where both read a
    fact). A grounded fact goes to every calculator that takes it; the average age is worked out from the ages.
    A section exists only when the user stated one of its facts; the upfront section only when code found a price.
    With no text (or no model) this is the code's reading alone."""
    from agents.schemas import TOOL_INPUT_MODEL
    flat: Dict[str, Any] = {}
    for sec in ("grants", "affordability"):
        for k, v in ((model or {}).get(sec) or {}).items():
            if v is None or k in flat:
                continue
            if k not in FACT_FIELDS or fact_is_grounded(k, v, text):
                flat[k] = v
    for sec in ("grants", "affordability"):
        flat.update(stated.get(sec) or {})
    ages = [flat[k] for k in ("youngest_buyer_age", "oldest_buyer_age") if isinstance(flat.get(k), (int, float))]
    if ages:
        flat["average_applicant_age"] = sum(ages) / len(ages)
    out: Dict[str, Dict[str, Any]] = {}
    for sec, tool in (("grants", "calculate_cpf_housing_grants"), ("affordability", "calculate_maximum_affordability")):
        fields = set(TOOL_INPUT_MODEL[tool].model_fields)
        vals = {k: v for k, v in flat.items() if k in fields}
        if any(k in FACT_FIELDS for k in vals):
            out[sec] = vals
    if "upfront" in stated:
        extra = {k: v for k, v in ((model or {}).get("upfront") or {}).items()
                 if v is not None and (k not in FACT_FIELDS or fact_is_grounded(k, v, text))}
        out["upfront"] = {**extra, **stated["upfront"]}
    return out or None


def has_financial_facts(c: Dict[str, Any]) -> bool:
    return any(vals for vals in (c.get("financial_inputs") or {}).values())


def guard_financial_args(name: str, args: Dict[str, Any], c: Dict[str, Any], outputs: Dict[str, Any],
                         runners: Dict[str, Any]) -> Dict[str, Any]:
    """A calculator call as the model wrote it, corrected at the tool boundary: stated facts override the model's
    arguments, facts the user never stated are left out (the calculator then reports them missing), and a confirmed
    grant total is passed forward, running the grant calculator first if the model has not."""
    sec = FIN_TOOL_SECTION.get(name)
    fin = c.get("financial_inputs") or {}
    if sec is None or not has_financial_facts(c):
        return args
    stated = fin.get(sec) or {}
    out = {k: v for k, v in args.items() if k not in FACT_FIELDS}
    out.update(stated)
    grant = "calculate_cpf_housing_grants"
    if sec != "grants" and "total_grants" not in out:
        if grant not in outputs and grant in runners and fin.get("grants"):
            outputs[grant] = runners[grant](**guard_financial_args(grant, {}, c, outputs, runners))
        if (outputs.get(grant) or {}).get("status") == "ok":
            out["total_grants"] = outputs[grant]["total_grants"]
    if sec == "upfront":
        for k in ("cpf_oa_balance", "cash_savings", "loan_type"):
            if k in (fin.get("affordability") or {}):
                out.setdefault(k, fin["affordability"][k])
        aff = outputs.get("calculate_maximum_affordability") or {}
        if aff.get("status") == "ok" and aff.get("ltv_ratio") is not None \
                and out.get("loan_type", "hdb") == (fin.get("affordability") or {}).get("loan_type", "hdb"):
            out.setdefault("ltv_ratio", aff["ltv_ratio"])
    return out


def _json_object(text: str) -> Dict[str, Any]:
    """The JSON object in a model reply, tolerating a <think> block, code fences or words around it."""
    m = re.search(r"\{.*\}", strip_think(text), re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def lenient_constraints(data: Dict[str, Any]) -> Tuple[ExtractedConstraints, List[str]]:
    """Validate the model's constraints field by field. A field that fails validation is dropped and named, rather
    than the whole reading being thrown away: a blank comparable-sales flat type used to discard the towns, budget
    and money facts read correctly beside it. Inside financial_inputs only the bad fact is dropped."""
    data = dict(data or {})
    dropped: List[str] = []
    for _ in range(12):
        try:
            return ExtractedConstraints(**data), dropped
        except ValidationError as e:
            changed = False
            for err in e.errors():
                loc = [x for x in (err.get("loc") or ()) if not isinstance(x, int)]
                if not loc or loc[0] not in data:
                    continue
                top = loc[0]
                section = data.get(top).get(loc[1]) if top == "financial_inputs" and len(loc) >= 3 \
                    and isinstance(data.get(top), dict) else None
                if isinstance(section, dict) and loc[2] in section:
                    data[top] = {**data[top], loc[1]: {k: v for k, v in section.items() if k != loc[2]}}
                    dropped.append(".".join(str(x) for x in loc[:3]))
                else:
                    data.pop(top)
                    dropped.append(str(top))
                changed = True
            if not changed:
                break
    return ExtractedConstraints(), dropped + ["(everything)"]


def read_constraints_with_model(llm: Any, messages: List[Any]) -> Tuple[Optional[ExtractedConstraints], List[str]]:
    """Structured constraint reading that degrades gracefully: invalid fields are dropped one by one
    (lenient_constraints), and a reply that is not JSON at all gets one repair turn showing the model its own output
    and the error. Returns (constraints or None, notes for the errors list)."""
    notes: List[str] = []
    for method in ("json_schema", "json_mode"):
        try:
            res = llm.with_structured_output(ExtractedConstraints, method=method, include_raw=True).invoke(messages)
        except Exception as e:  # noqa: BLE001
            notes.append(f"extract_constraints[{method}] failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        if res.get("parsed") is not None:
            return res["parsed"], notes
        raw = res.get("raw")
        text = raw.content if isinstance(getattr(raw, "content", None), str) else str(raw or "")
        try:
            data = _json_object(text)
        except Exception as e:  # noqa: BLE001
            try:
                fix = llm.invoke(messages + [AIMessage(content=text[:4000]), HumanMessage(content=(
                    f"That reply was not a valid JSON object ({type(e).__name__}: {str(e)[:120]}). "
                    "Reply with only the corrected JSON object for the same schema."))])
                data = _json_object(fix.content if isinstance(fix.content, str) else str(fix.content))
                notes.append(f"extract_constraints[{method}]: repaired a reply that was not JSON")
            except Exception as e2:  # noqa: BLE001
                notes.append(f"extract_constraints[{method}] failed after one repair: {type(e2).__name__}: {str(e2)[:120]}")
                continue
        extracted, dropped = lenient_constraints(data)
        if dropped:
            notes.append(f"extract_constraints[{method}]: dropped invalid fields {', '.join(dropped)}")
        return extracted, notes
    return None, notes


def extract_constraints_node(state: HDBCompassState) -> Dict[str, Any]:
    query = last_user_query(state)
    prior = state.get("prior_constraints")
    pending = state.get("pending_clarification")
    llm = get_llm()
    extracted: Optional[ExtractedConstraints] = None
    errors: List[str] = []
    system = EXTRACT_SYSTEM
    if pending:
        system += (f" This message answers the assistant's earlier question about: {', '.join(pending['needed'])}. "
                   "Record each stated answer under financial_inputs.grants and financial_inputs.affordability using exactly "
                   "those field names; facts already known: "
                   f"{json.dumps((prior or {}).get('financial_inputs') or {}, default=str)}. Leave everything else empty.")
    if llm is not None:
        messages = [sys_msg(system), HumanMessage(content=query)]
        # HDB_EXTRACTION_REASONING=1 switches it on for one run without editing the config (A/B testing)
        if os.getenv("HDB_EXTRACTION_REASONING") == "1" or load_config().get("architecture", {}).get("extraction_reasoning", False):
            # Guided JSON output leaves the model no room to think, so an optional first pass lists, in plain
            # words, every constraint and money fact the user stated; the JSON pass then reads from that list too.
            try:
                think = llm.invoke([sys_msg("List, as short bullet points, every housing-search constraint and every money "
                                            "or household fact this buyer states, quoting their words. Add nothing they did "
                                            "not say."), HumanMessage(content=query)])
                notes_text = strip_think(think.content if isinstance(think.content, str) else str(think.content))
                messages.append(HumanMessage(content=f"Facts the buyer stated (from a first read):\n{notes_text[:3000]}"))
            except Exception as e:  # noqa: BLE001
                errors.append(f"extract_constraints[reasoning] failed: {type(e).__name__}")
        extracted, notes = read_constraints_with_model(llm, messages)
        errors += notes
        if extracted is not None:
            # merge: regex fills anything the model left empty (cheap robustness)
            rb = rule_based_extract(query).model_dump()
            merged = extracted.model_dump()
            for k, v in rb.items():
                if merged.get(k) in (None, [], {}, False) and v not in (None, [], {}, False):
                    merged[k] = v
            extracted = ExtractedConstraints(**merged)
    if extracted is None:
        extracted = rule_based_extract(query)
    constraints = extracted.model_dump()
    if extracted.financial_inputs is not None:
        constraints["financial_inputs"] = extracted.financial_inputs.model_dump(exclude_unset=True)
    # calculator facts come from the user's own words in every turn so far, never from the model's reading of them
    users = " ".join(m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user")
    constraints["financial_inputs"] = trusted_financial_inputs(constraints.get("financial_inputs"),
                                                               stated_financial_inputs(users or query), users or query)
    # a distance limit of 0 m is not a constraint: the model filled every amenity with 0 for an eligibility question,
    # which routed it to the map agent and produced "relax from 0 m to 150 m" suggestions
    constraints["amenity_radii_meters"] = {k: v for k, v in (constraints.get("amenity_radii_meters") or {}).items()
                                           if v is not None and v > 0}
    if pending:
        conversation = " ".join(m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict))
        answers = parse_clarification_answer(query, pending["needed"], conversation)
        if answers:
            constraints = merge_constraints(constraints, {"financial_inputs": answers})
    if prior:
        constraints = merge_constraints(prior, constraints)
    return {"user_constraints": constraints, "errors": errors}


# ---------------------------------------------------------------------------
# Node: route  +  fan-out dispatch
# ---------------------------------------------------------------------------
def decide_routing(c: Dict[str, Any]) -> List[str]:
    active: List[str] = []
    if c.get("comparable_resale") or (c.get("towns") and (c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("flat_types")
                           or c.get("wants_price_trend") or c.get("wants_bto_comparison"))):
        active.append("market_sql")
    if c.get("amenity_radii_meters") or c.get("school_name") or len(c.get("work_destinations") or []) >= 2:
        active.append("spatial")
    if has_financial_facts(c) or c.get("household_income") or c.get("cpf_oa_balance") or (c.get("citizenship_scheme") and c.get("max_budget_sgd")):
        active.append("financial")
    if c.get("policy_question"):
        active.append("policy_rag")
    if not active:
        active.append("market_sql" if c.get("towns") else "policy_rag")
    return active


def route_node(state: HDBCompassState) -> Dict[str, Any]:
    return {"active_subagents": decide_routing(state.get("user_constraints", {}))}


def dispatch(state: HDBCompassState) -> List[Send]:
    """Parallel fan-out: LangGraph runs one Send per active specialist concurrently."""
    return [Send(f"{a}_agent", state) for a in state.get("active_subagents", [])]


# ---------------------------------------------------------------------------
# Specialist runner: ReAct loop over one agent's tools, or a deterministic plan
# ---------------------------------------------------------------------------
AGENT_SYSTEM = {
    "market_sql": "You are the Market & Valuation SQL agent for Singapore HDB resale flats. Use your tools to fetch "
                  "block-level prices, trends and BTO comparisons that satisfy the user's constraints. Call tools with "
                  "exact towns/flat types from the constraints. For comparable sales use verified coordinates and "
                  "explicit matching filters. Disclose small samples, truncated examples and missing geocodes; "
                  "do not widen filters silently or present comparables as an official valuation. "
                  "When done, reply with a one-paragraph summary.",
    "spatial": "You are the Geospatial & Commute agent. Use your tools to find HDB blocks meeting amenity-distance, "
               "school-boundary and commute constraints. Pass towns exactly as given. Reply with a short summary when done.",
    "financial": "You are the Financial & Grant Calculator agent. Never do arithmetic yourself: call the calculators "
                 "for grants, affordability (MSR/LTV) and upfront outlay (BSD). Never invent eligibility confirmations "
                 "or infer citizenship/first-timer status from legacy citizenship_scheme. Pass grant results forward once; "
                 "real affordability already includes grants. Supply actual valuation, loan and quoted fees for upfront "
                 "costs. Unknown funds/debts/grants are not zero. Report missing inputs and unsupported scenarios, "
                 "including cost exclusions and cash shortfalls. Reply with a short summary when done.",
    "policy_rag": "You are the Regulatory Policy RAG agent. Call retrieve_hdb_policy_knowledge with the user's regulatory "
                  "question, then answer ONLY from the returned passages with inline [source] citations.",
}


def _scheme_to_pair(scheme: Optional[str]) -> Tuple[str, bool]:
    s = (scheme or "SC_SC_FirstTimer").upper()
    pair = "SC_Single" if "SINGLE" in s else "SC_PR" if "PR" in s else "SC_SC"
    return pair, "SECOND" not in s


def deterministic_plan(agent: str, c: Dict[str, Any], query: str,
                       spatial_blocks: Optional[List[str]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """What a competent specialist would call for these constraints. Used when no LLM is
    configured, as the fallback when the LLM makes no tool call, and to derive
    expected_mcp_calls for Category E golden queries."""
    plan: List[Tuple[str, Dict[str, Any]]] = []
    towns = c.get("towns") or []
    if agent == "market_sql":
        if c.get("comparable_resale"):
            plan.append(("query_comparable_resale_transactions", c["comparable_resale"]))
        if towns:
            plan.append(("query_resale_blocks_by_criteria", dict(
                towns=towns, flat_types=c.get("flat_types") or ["4 ROOM"],
                max_price=c.get("max_budget_sgd") or 5_000_000, min_lease_years=c.get("min_remaining_lease_years") or 0)))
        if c.get("wants_price_trend") and towns:
            plan.append(("compute_town_price_timeseries", dict(towns=towns, flat_type=(c.get("flat_types") or ["4 ROOM"])[0], lookback_years=3)))
        if c.get("wants_bto_comparison"):
            m = re.search(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\s+(?:BTO|launch)", query)
            plan.append(("compare_bto_launch_vs_resale", dict(bto_project_name=m.group(1) if m else (towns[0] if towns else "Recent BTO"), radius_meters=500)))
    elif agent == "spatial":
        radii = {k: v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"}
        if radii:
            plan.append(("filter_blocks_by_amenity_proximity", dict(towns=towns, amenity_constraints=radii)))
        if c.get("school_name"):
            r = (c.get("amenity_radii_meters") or {}).get("school", 1000)
            plan.append(("verify_primary_school_distance", dict(school_name=c["school_name"], bracket="1km" if r <= 1000 else "2km")))
        work = c.get("work_destinations") or []
        if len(work) >= 2:
            plan.append(("compute_dual_transit_commute", dict(candidate_blocks=spatial_blocks or [], work_dest_a=work[0],
                                                             work_dest_b=work[1], max_minutes=c.get("max_commute_minutes") or 40)))
    elif agent == "financial":
        income = c.get("household_income")
        pair, first_timer = _scheme_to_pair(c.get("citizenship_scheme"))
        near_parents = bool(re.search(r"near|within\s*4\s*km of (?:my|our|their)?\s*parents", query, re.I))
        if income:
            plan.append(("calculate_cpf_housing_grants", dict(flat_category="bto" if re.search(r"\bbto\b", query, re.I) and not re.search(r"resale", query, re.I) else "resale",
                                                              citizenship_pair=pair, first_timer_status=first_timer,
                                                              monthly_income=income, near_parents_4km=near_parents)))
            plan.append(("calculate_maximum_affordability", dict(monthly_income=income, cpf_oa_balance=c.get("cpf_oa_balance") or 0.0,
                                                                 cash_savings=c.get("cash_savings") or 0.0, loan_type="hdb")))
        if c.get("max_budget_sgd") or income:
            plan.append(("calculate_bsd_and_upfront_outlay", dict(purchase_price=c.get("max_budget_sgd") or 0.0, loan_type="hdb", total_grants=0.0)))
    elif agent == "policy_rag":
        plan.append(("retrieve_hdb_policy_knowledge", dict(query=c.get("policy_question") or query, policy_domain=None)))
    return plan


def run_react(label: str, tool_names: List[str], system_prompt: str, query: str, c: Dict[str, Any],
              max_iter: Optional[int] = None, plan_agents: Optional[List[str]] = None
              ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], str]:
    """Generic ReAct loop. Returns (outputs_by_tool, tool_call_log, errors, final_text).
    label: logged as the calling agent. plan_agents: which deterministic plans to use as fallback."""
    cfg = load_config()["architecture"]
    max_iter = max_iter or cfg["subagent_max_tool_iterations"]
    runners: Dict[str, ToolRunner] = build_runners(tool_names)
    for r in runners.values():
        r.agent = label
    outputs: Dict[str, Any] = {}
    errors: List[str] = []
    final_text = ""
    llm = get_llm()
    used_llm_tools = False

    if llm is not None:
        try:
            bound = llm.bind_tools([r.as_langchain_tool() for r in runners.values()])
            msgs: List[Any] = [sys_msg(system_prompt),
                               HumanMessage(content=f"User query: {query}\n\nExtracted constraints (JSON): {json.dumps(c)}")]
            for _ in range(max_iter):
                ai: AIMessage = bound.invoke(msgs)
                msgs.append(ai)
                if not getattr(ai, "tool_calls", None):
                    final_text = strip_think(ai.content if isinstance(ai.content, str) else str(ai.content))
                    break
                for tc in ai.tool_calls:
                    name, args = tc["name"], tc.get("args") or {}
                    if name not in runners:
                        errors.append(f"{label}: model called unknown tool {name}")
                        msgs.append(ToolMessage(content=json.dumps({"error": f"unknown tool {name}"}), tool_call_id=tc["id"]))
                        continue
                    result = runners[name](**guard_financial_args(name, args, c, outputs, runners))
                    used_llm_tools = True
                    outputs[name] = result
                    msgs.append(ToolMessage(content=json.dumps(result, default=str)[:12000], tool_call_id=tc["id"]))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: LLM loop failed: {type(e).__name__}: {str(e)[:160]}")

    if not used_llm_tools:
        if llm is not None:
            errors.append(f"{label}: no valid tool call from model; deterministic plan used")
        for agent in (plan_agents or [label]):
            if agent == "financial" and all(
                load_config()["tools"]["use_mock"].get(name) == "real"
                for name in tools_for_agent("financial")
            ):
                outputs.update(run_financial_sequence(runners, c, query))
                continue
            for name, args in deterministic_plan(agent, c, query):
                if name in runners:
                    outputs[name] = runners[name](**args)

    log = [rec.model_dump() for r in runners.values() for rec in r.log]
    return outputs, log, errors, final_text


def run_financial_sequence(runners, constraints, query):
    """Pass confirmed results forward without replacing unknown financial facts with zero."""
    details = constraints.get("financial_inputs") or {}
    grant_args = {k: v for k, v in (details.get("grants") or {}).items() if v is not None}
    aff_args = {k: v for k, v in (details.get("affordability") or {}).items() if v is not None}
    upfront_args = {k: v for k, v in (details.get("upfront") or {}).items() if v is not None}
    for key in ("youngest_buyer_age", "remaining_lease_years"):
        if key in grant_args:
            aff_args.setdefault(key, grant_args[key])
    for key in ("loan_type", "cpf_oa_balance", "cash_savings"):
        if key in aff_args:
            upfront_args.setdefault(key, aff_args[key])
    income = constraints.get("household_income")
    if income is not None:
        grant_args.setdefault("monthly_income", income)
        aff_args.setdefault("monthly_income", income)
    flat_types = constraints.get("flat_types") or []
    if len(flat_types) == 1:
        grant_args.setdefault("flat_type", flat_types[0])
    # The legacy citizenship_scheme extractor infers citizenship/first-timer status;
    # only explicitly supplied financial facts may be used by real calculators.
    if re.search(r"\bbto\b", query, re.I) and not re.search(r"resale", query, re.I):
        grant_args["flat_category"] = "bto"
    for key in ("cpf_oa_balance", "cash_savings"):
        if constraints.get(key) is not None:
            aff_args.setdefault(key, constraints[key])
            upfront_args.setdefault(key, constraints[key])
    outputs = {}
    grant_name = "calculate_cpf_housing_grants"
    aff_name = "calculate_maximum_affordability"
    upfront_name = "calculate_bsd_and_upfront_outlay"
    if grant_name in runners:
        grants = outputs[grant_name] = runners[grant_name](**grant_args)
        if grants.get("status") == "ok":
            for target in (aff_args, upfront_args):
                if target.get("total_grants") is None:
                    target["total_grants"] = grants["total_grants"]
    if aff_name in runners:
        aff = outputs[aff_name] = runners[aff_name](**aff_args)
        if aff.get("status") == "ok" and upfront_args.get("loan_type", "hdb") == aff_args.get("loan_type", "hdb"):
            upfront_args.setdefault("ltv_ratio", aff["ltv_ratio"])
    if upfront_name in runners:
        if "purchase_price" not in upfront_args and constraints.get("max_budget_sgd") is not None:
            upfront_args["purchase_price"] = constraints["max_budget_sgd"]
        if upfront_args.get("purchase_price", 0) > 0:
            outputs[upfront_name] = runners[upfront_name](**upfront_args)
    return outputs


def _normalise_school_blocks(items: List[str], towns: List[str]) -> List[Dict[str, Any]]:
    out = []
    for s in items:
        m = re.match(r"Blk\s+(\S+)\s+(.+?)\s*(?:\((\d+)m\))?$", s, re.I)
        if not m:
            continue
        out.append({"block": m.group(1), "street": m.group(2).upper(), "town": towns[0] if len(towns) == 1 else "",
                    "distances": {"school_dist_m": float(m.group(3))} if m.group(3) else {}})
    return out


def slot_outputs(outputs: Dict[str, Any], towns: List[str]) -> Dict[str, Any]:
    """Map raw tool outputs into the HDBCompassState slots, by tool owner."""
    update: Dict[str, Any] = {}
    sql, fin = {}, {}
    spatial: List[Dict[str, Any]] = []
    for name, out in outputs.items():
        owner = TOOL_OWNER[name]
        if isinstance(out, dict) and "error" in out:
            continue
        if owner == "market_sql":
            sql[name] = out
        elif owner == "financial":
            fin[name] = out
        elif owner == "policy_rag":
            update["rag_policy_passages"] = out.get("passages", []) if isinstance(out, dict) else out
        elif name == "filter_blocks_by_amenity_proximity":
            spatial = out if not spatial else [b for b in spatial if block_key(b["block"], b["street"]) in
                                               {block_key(x["block"], x["street"]) for x in out}]
        elif name == "verify_primary_school_distance":
            sb = _normalise_school_blocks(out, towns)
            spatial = sb if not spatial else [b for b in spatial if block_key(b["block"], b["street"]) in
                                              {block_key(x["block"], x["street"]) for x in sb}]
        elif name == "compute_dual_transit_commute" and isinstance(out, list) and out:
            keep = {str(b.get("block", "")).upper() for b in out if isinstance(b, dict)}
            spatial = [b for b in spatial if str(b.get("block", "")).upper() in keep] if spatial else out
    if sql:
        update["sql_market_results"] = sql
    if fin:
        update["financial_breakdown"] = fin
    if any(TOOL_OWNER[n] == "spatial" for n in outputs):
        update["spatial_block_candidates"] = spatial
    return update


def requested_outputs(agent: str, outputs: Dict[str, Any], c: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Evidence gate for Variant C's market and spatial specialists: a tool's result becomes evidence only when the
    extracted constraints call for that tool (deterministic_plan). In the 27 Oct rerun the spatial agent ran the school
    check with school_name null and the commute check with blank workplaces, and their empty results filtered every
    block out (E1-04); the market agent pulled comparable sales at coordinates it made up. The calls stay in the
    tool-call log, so tool-selection metrics still count them."""
    if agent not in ("market_sql", "spatial"):
        return outputs
    asked = {name for name, _ in deterministic_plan(agent, c, query)}
    return {name: out for name, out in outputs.items() if name in asked or name == "sync_interactive_map_ui"}


def make_specialist_node(agent: str) -> Callable[[HDBCompassState], Dict[str, Any]]:
    def node(state: HDBCompassState) -> Dict[str, Any]:
        c = state.get("user_constraints", {})
        outputs, log, errors, final = run_react(agent, tools_for_agent(agent), AGENT_SYSTEM[agent], last_user_query(state), c)
        update = slot_outputs(requested_outputs(agent, outputs, c, last_user_query(state)), c.get("towns") or [])
        update.update({"tool_call_log": log, "errors": errors,
                       "subagent_notes": [{"agent": agent, "text": (final or "")[:4000]}] if final else []})
        return update
    node.__name__ = f"{agent}_agent"
    return node


# ---------------------------------------------------------------------------
# Node: intersect_candidates
# ---------------------------------------------------------------------------
def _sql_blocks(state: HDBCompassState) -> Optional[List[Dict[str, Any]]]:
    sql = state.get("sql_market_results") or {}
    q = sql.get("query_resale_blocks_by_criteria")
    return q.get("blocks") if isinstance(q, dict) else None


def _comparable_lines(sql: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    comparables = sql.get("query_comparable_resale_transactions")
    if comparables:
        summary = comparables.get("summary")
        if summary:
            prices = summary["price"]
            criteria = comparables["criteria"]
            metadata = comparables["metadata"]
            lines.append(f"- Comparison: {criteria['flat_type']}, within {criteria['radius_meters']} m straight-line, {metadata['window_start_month']}–{metadata['window_end_month']}.")
            lines.append(f"- Comparable resale sales: {summary['transaction_count']} transactions across {summary['block_count']} blocks; median ${prices['median']:,.0f}, middle 50% ${prices['lower_quartile']:,.0f}–${prices['upper_quartile']:,.0f}.")
            if comparables.get("status") == "insufficient_sample":
                lines.append("- The comparable sample is below the requested minimum; treat it as limited evidence.")
            if comparables.get("metadata", {}).get("sample_truncated"):
                lines.append("- Displayed transaction examples are limited; statistics use the full matched sample.")
        elif comparables.get("status") == "no_coverage":
            lines.append("- The requested comparable-sales period predates the database's source coverage.")
        elif "error" in comparables:
            lines.append("- Comparable resale analysis is unavailable; check the supplied criteria and database setup.")
        else:
            lines.append("- No geocoded resale transactions match the comparable-sales criteria; filters have not been widened.")
        if comparables.get("metadata", {}).get("transactions_without_coordinates", 0):
            lines.append("- Some otherwise matching transactions have no coordinates; their proximity could not be assessed.")
        if comparables.get("metadata", {}).get("history_truncated"):
            lines.append("- The requested window extends before source history; earlier transactions are unavailable.")
    return lines


def _affordable_price(state: HDBCompassState) -> Optional[float]:
    fin = state.get("financial_breakdown") or {}
    aff = fin.get("calculate_maximum_affordability") or {}
    grants = (fin.get("calculate_cpf_housing_grants") or {}).get("total_grants", 0.0)
    if aff.get("status", "ok") == "ok" and aff.get("max_purchase_price") is not None:
        return float(aff["max_purchase_price"]) + (0.0 if aff.get("grants_included") else float(grants or 0.0))
    return None


def compute_intersection(sql_blocks: Optional[List[Dict[str, Any]]], spatial_blocks: Optional[List[Dict[str, Any]]],
                         price_cap: Optional[float]) -> List[Dict[str, Any]]:
    if sql_blocks is not None and spatial_blocks is not None:
        by_key = {block_key(b["block"], b["street"]): b for b in spatial_blocks}
        merged = []
        for b in sql_blocks:
            k = block_key(b["block"], b["street"])
            if k in by_key:
                merged.append({**b, "distances": by_key[k].get("distances", {})})
        result = merged
    else:
        result = list(sql_blocks if sql_blocks is not None else (spatial_blocks or []))
    if price_cap is not None:
        result = [b for b in result if b.get("median_resale_price") is None or b["median_resale_price"] <= price_cap]
    return result


def intersect_node(state: HDBCompassState) -> Dict[str, Any]:
    cap = _affordable_price(state)
    budget = state.get("user_constraints", {}).get("max_budget_sgd")
    if budget is not None:
        aff = (state.get("financial_breakdown") or {}).get("calculate_maximum_affordability") or {}
        if not aff.get("rule_version"):
            cap = None  # Preserve the user-budget semantics of legacy mock benchmarks.
        cap = min(cap, budget) if cap is not None else budget
    return {"candidate_blocks_intersection": compute_intersection(_sql_blocks(state), state.get("spatial_block_candidates"), cap)}


# ---------------------------------------------------------------------------
# Node: relax_constraints (use case E2)
# ---------------------------------------------------------------------------
def should_relax(state: HDBCompassState) -> str:
    c = state.get("user_constraints", {})
    searched = _sql_blocks(state) is not None or state.get("spatial_block_candidates") is not None
    tunable = c.get("max_budget_sgd") or c.get("amenity_radii_meters") or c.get("min_remaining_lease_years")
    if searched and tunable and not state.get("candidate_blocks_intersection") and not state.get("relaxation"):
        return "relax_constraints"
    return "synthesize"


def _candidates_for(c: Dict[str, Any], query: str, runners: Dict[str, ToolRunner]) -> List[Dict[str, Any]]:
    outputs: Dict[str, Any] = {}
    for agent in ("market_sql", "spatial"):
        for name, args in deterministic_plan(agent, c, query):
            if name in runners and name in ("query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity",
                                            "verify_primary_school_distance"):
                outputs[name] = runners[name](**args)
    slots = slot_outputs(outputs, c.get("towns") or [])
    sql = (slots.get("sql_market_results") or {}).get("query_resale_blocks_by_criteria", {}).get("blocks") \
        if slots.get("sql_market_results") else None
    return compute_intersection(sql, slots.get("spatial_block_candidates"), None)


def relax_node(state: HDBCompassState) -> Dict[str, Any]:
    """Sensitivity sweep: relax ONE constraint at a time and count unlocked blocks."""
    steps = load_config()["architecture"]["relaxation_steps"]
    c = dict(state.get("user_constraints", {}))
    query = last_user_query(state)
    runners = build_runners(["query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity", "verify_primary_school_distance"])
    for r in runners.values():
        r.agent = "supervisor"
    suggestions: List[RelaxationSuggestion] = []

    if c.get("max_budget_sgd"):
        relaxed = round(c["max_budget_sgd"] * (1 + steps["max_budget_sgd_pct"] / 100), -3)
        n = len(_candidates_for({**c, "max_budget_sgd": relaxed}, query, runners))
        suggestions.append(RelaxationSuggestion(constraint="max_budget_sgd", original=c["max_budget_sgd"], relaxed=relaxed, unlocked_blocks=n))
    for k, v in (c.get("amenity_radii_meters") or {}).items():
        if k == "school":
            continue
        relaxed_r = {**c["amenity_radii_meters"], k: v + steps["amenity_radius_meters"]}
        n = len(_candidates_for({**c, "amenity_radii_meters": relaxed_r}, query, runners))
        suggestions.append(RelaxationSuggestion(constraint=f"amenity_radii_meters.{k}", original=v, relaxed=v + steps["amenity_radius_meters"], unlocked_blocks=n))
    if c.get("min_remaining_lease_years"):
        relaxed_l = max(0, c["min_remaining_lease_years"] - steps["min_remaining_lease_years"])
        n = len(_candidates_for({**c, "min_remaining_lease_years": relaxed_l}, query, runners))
        suggestions.append(RelaxationSuggestion(constraint="min_remaining_lease_years", original=c["min_remaining_lease_years"], relaxed=relaxed_l, unlocked_blocks=n))

    best = max(suggestions, key=lambda s: s.unlocked_blocks) if suggestions else None
    if best is not None and best.unlocked_blocks == 0:
        best = None
    out = RelaxUnsatisfiedConstraintsOutput(original_constraints=c, suggestions=suggestions, best_single_change=best)
    return {"relaxation": out.model_dump(),
            "tool_call_log": [rec.model_dump() for r in runners.values() for rec in r.log]}


# ---------------------------------------------------------------------------
# Node: synthesize (use case E1: response + map sync)
# ---------------------------------------------------------------------------
def build_map_payload(blocks: List[Dict[str, Any]], c: Dict[str, Any], log_sink: List[Dict[str, Any]]) -> Dict[str, Any]:
    pts = [(b["lat"], b["lng"]) for b in blocks if b.get("lat") and b.get("lng")]
    if not pts:
        return {}
    center = [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
    radii = {k: int(v) for k, v in (c.get("amenity_radii_meters") or {}).items()} or {"search": 1000}
    runner = ToolRunner("sync_interactive_map_ui")
    runner.agent = "supervisor"
    sync = runner(candidate_blocks=[block_key(b["block"], b["street"]) for b in blocks], buffer_radii=radii, center_latlon=center)
    log_sink.extend(rec.model_dump() for rec in runner.log)
    markers = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [b["lng"], b["lat"]]},
         "properties": {"name": f"Blk {b['block']} {b['street']}", "category": "hdb_candidate",
                        "median_resale_price": b.get("median_resale_price"),
                        "remaining_lease_years": b.get("median_remaining_lease_years"),
                        **{k: v for k, v in (b.get("distances") or {}).items()}}}
        for b in blocks if b.get("lat") and b.get("lng")]}
    return {**(sync if isinstance(sync, dict) else {}), "markers": markers, "candidate_blocks": [block_key(b["block"], b["street"]) for b in blocks]}


HFE_DISCLAIMER = ("\n\n_Figures are indicative. Grant and loan eligibility is confirmed only by an HDB Flat Eligibility (HFE) "
                  "letter. This is not financial advice._")


def template_response(state: HDBCompassState) -> str:
    lines: List[str] = []
    blocks = state.get("candidate_blocks_intersection") or []
    searched = (
        blocks or _sql_blocks(state) is not None
        or state.get("spatial_block_candidates") is not None
    )
    if searched:
        if blocks:
            lines.append(f"**{len(blocks)} candidate block(s)** match all stated constraints:")
            for b in blocks[:8]:
                price = f" — median ${b['median_resale_price']:,.0f}" if b.get("median_resale_price") else ""
                lease = f", {b['median_remaining_lease_years']:.0f} yrs lease" if b.get("median_remaining_lease_years") else ""
                dist = ", ".join(f"{k.replace('_dist_m','')} {v:.0f}m" for k, v in (b.get("distances") or {}).items())
                lines.append(f"- Blk {b['block']} {b['street']} ({b.get('town','')}){price}{lease}{(' — ' + dist) if dist else ''}")
        else:
            lines.append("**No block satisfies every constraint at once.**")
            rel = state.get("relaxation") or {}
            if rel.get("best_single_change"):
                bsc = rel["best_single_change"]
                lines.append(f"Relaxing **{bsc['constraint']}** from {bsc['original']} to {bsc['relaxed']} unlocks {bsc['unlocked_blocks']} block(s).")
            for s in rel.get("suggestions", []):
                lines.append(f"- {s['constraint']}: {s['original']} → {s['relaxed']} unlocks {s['unlocked_blocks']}")
    sql = state.get("sql_market_results") or {}
    lines.extend(_comparable_lines(sql))
    if "compute_town_price_timeseries" in sql:
        trend = sql["compute_town_price_timeseries"]
        for t in trend.get("summary", []):
            period = t.get("latest_quarter") or "latest"
            change = t.get("yoy_appreciation_pct")
            if change is None:
                reason = (
                    "prior-year quarter is incomplete"
                    if t.get("yoy_unavailable_reason") == "incomplete_prior_year_quarter"
                    else "no comparable prior-year quarter"
                )
                yoy = f"YoY unavailable ({reason})"
            else:
                yoy = f"YoY {change:+.1f}%"
            lines.append(f"- {t['town']}: {period} median ${t['latest_median_price']:,.0f}, ${t['median_price_per_sqm']:,.0f}/sqm, {yoy}")
        metadata = trend.get("metadata") or {}
        for town in metadata.get("towns_without_data", []):
            lines.append(f"- {town}: no transactions for the requested flat type in the trend window.")
        for town in metadata.get("towns_without_complete_quarters", []):
            lines.append(f"- {town}: only incomplete quarters are available; no complete-quarter summary or YoY comparison.")
    unbuilt = unbuilt_tools_used(state)
    if "compare_bto_launch_vs_resale" in sql and "compare_bto_launch_vs_resale" in unbuilt:
        lines.append("- BTO launch prices can't be compared in this version yet.")
    elif "compare_bto_launch_vs_resale" in sql:
        b = sql["compare_bto_launch_vs_resale"]
        lines.append(f"- BTO {b['bto_project_name']}: ${b['bto_indicative_price_range']['min']:,.0f}–${b['bto_indicative_price_range']['max']:,.0f} vs nearby resale median ${b['resale_median_nearby']:,.0f} ({b['price_gap_pct']:+.1f}%)")
    fin = state.get("financial_breakdown") or {}
    if fin:
        lines.append("**Financials:**")
        labels = {"calculate_cpf_housing_grants": "Housing grants",
                  "calculate_maximum_affordability": "Affordability",
                  "calculate_bsd_and_upfront_outlay": "Upfront funding"}
        for name, value in fin.items():
            label = labels.get(name, "Financial estimate")
            if "error" in value:
                lines.append(f"- {label}: calculation unavailable.")
                continue
            if value.get("status", "ok") != "ok":
                detail = ", ".join(field.replace("_", " ") for field in value.get("missing_inputs", [])) or "; ".join(value.get("notes", []))
                status = "more information needed" if value.get("status") == "needs_information" else "scenario unavailable"
                lines.append(f"- {label}: {status} — {detail}")
        fin = {name: value for name, value in fin.items() if value.get("status", "ok") == "ok" and "error" not in value}
        g = fin.get("calculate_cpf_housing_grants")
        if g:
            lines.append(f"- Grants: EHG ${g['enhanced_housing_grant']:,.0f} + Family ${g['family_grant']:,.0f} + PHG ${g['proximity_housing_grant']:,.0f} = **${g['total_grants']:,.0f}**")
        a = fin.get("calculate_maximum_affordability")
        if a:
            lines.append(f"- Max purchase price ${a['max_purchase_price']:,.0f} (loan ${a['max_loan_amount']:,.0f}, instalment ${a['monthly_instalment']:,.0f}/mth, binding: {a['binding_constraint']})")
            if not a.get("purchase_costs_included", True):
                lines.append("- This purchase-only ceiling excludes taxes, fees and cash over valuation; check upfront funding before choosing a budget.")
        u = fin.get("calculate_bsd_and_upfront_outlay")
        if u:
            lines.append(f"- Upfront: BSD ${u['buyer_stamp_duty']:,.0f}, cash required ${u['cash_required']:,.0f}, total ${u['total_upfront']:,.0f}")
            if (u.get("cash_shortfall") or 0) > 0:
                lines.append(f"- Cash shortfall: ${u['cash_shortfall']:,.0f}.")
    passages = state.get("rag_policy_passages") or []
    if passages and "retrieve_hdb_policy_knowledge" in unbuilt:
        lines.append("**Policy:**")
        lines.append("- I can't check HDB policy or eligibility rules yet in this version. Please confirm with HDB, "
                     "for example with an HDB Flat Eligibility (HFE) letter.")
    elif passages:
        lines.append("**Policy:**")
        for p in passages[:3]:
            lines.append(f"- {p['text']} [{p['source_url']}]")
    if not lines:
        lines.append("I could not find data for that request.")
    if state.get("financial_breakdown") or passages:
        lines.append(HFE_DISCLAIMER)
    return "\n".join(lines)


SYNTH_SYSTEM = (
    "You are HDB-Compass, a Singapore public-housing decision assistant. Write the final answer for the user from the "
    "JSON evidence only. Every number, price, block, distance and grant amount MUST appear verbatim in the evidence; "
    "never invent or round differently. Cite policy passages by their source_url. Present trade-offs objectively "
    "(price vs lease vs distance). No speculation about future prices or profits. Use short markdown sections."
    " Financial results with status needs_information or unsupported do not establish any numeric estimate; "
    "ask for the missing facts or explain the limitation. Never add grants again when grants_included is true. "
    "A purchase-only affordability ceiling excludes taxes and fees; disclose that and any reported cash shortfall."
    " If not_available_yet is present, tell the user plainly what cannot be checked yet, as it says, and do not guess."
)


GROUNDED_RULE = (
    "Recommend ONLY the blocks listed in matching_blocks, with the prices, leases and distances given there. "
    "Do not name any other block number. If matching_blocks is empty, say clearly that no block meets every "
    "constraint, then present the relaxation suggestions with their numbers.")
RECONCILE_RULE = (
    "The data agent's summary and the raw tool results are below; no code has joined them. Recommend only blocks "
    "the evidence shows satisfy every stated constraint: price, lease and every distance limit. A block missing from "
    "spatial_block_candidates has not been shown to meet the distance limits.")


def _compact_spatial(rows: Any, cap: int = 300) -> Any:
    if not isinstance(rows, list):
        return rows
    return [{k: b.get(k) for k in ("block", "street", "distances")} if isinstance(b, dict) else b for b in rows[:cap]]


# Tools whose real version is not built yet answer from mcp_server/mock_tools.py with fixed sample data. That data
# must not reach a user as if it answered the question (an unmarried couple asking about BTO eligibility got the
# same two sample passages every time). While a tool is mocked, the answer says what cannot be checked yet instead.
# Switch off with architecture.disclose_unbuilt_tools: false.
UNBUILT_NOTES = {
    "retrieve_hdb_policy_knowledge": (
        "HDB policy and eligibility rules cannot be checked in this version yet (the policy search is still being "
        "built). Say so plainly, state no rule, and suggest confirming with HDB, for example with an HDB Flat "
        "Eligibility (HFE) letter."),
    "compare_bto_launch_vs_resale": (
        "BTO launch prices cannot be compared in this version yet. Say so plainly and give no BTO price."),
}


def unbuilt_tools_used(state: HDBCompassState) -> List[str]:
    """Mocked tools (see UNBUILT_NOTES) that this answer called."""
    cfg = load_config()
    if not cfg.get("architecture", {}).get("disclose_unbuilt_tools", True):
        return []
    mocked = {t for t, v in cfg.get("tools", {}).get("use_mock", {}).items() if v == "mock" and t in UNBUILT_NOTES}
    called = {t.get("tool") for t in state.get("tool_call_log") or []}
    return sorted(mocked & called)


def synthesis_evidence(state: HDBCompassState) -> Tuple[Dict[str, Any], str]:
    """What the final writing step may see, by architecture.
    C (and the default): the supervisor's join is the ONLY source of blocks. Raw price-query rows are withheld, so the
      writer cannot re-introduce a block the intersection excluded (the failure seen in the first Category E run).
    B: no join in code by design. The writer gets the data agent's own reconciliation plus the raw price AND distance
      results, and must do the join itself."""
    unbuilt = unbuilt_tools_used(state)
    base = {"user_constraints": state.get("user_constraints"),
            "financial_breakdown": state.get("financial_breakdown"),
            "rag_policy_passages": [] if "retrieve_hdb_policy_knowledge" in unbuilt else state.get("rag_policy_passages")}
    if unbuilt:
        base["not_available_yet"] = [UNBUILT_NOTES[t] for t in unbuilt]
    sql = {k: v for k, v in (state.get("sql_market_results") or {}).items() if k not in unbuilt}
    if state.get("variant") == "B":
        notes = [n.get("text", "") for n in state.get("subagent_notes") or [] if n.get("agent") == "data"]
        return ({**base, "data_agent_summary": notes[-1] if notes else "", "sql_market_results": sql,
                 "spatial_block_candidates": _compact_spatial(state.get("spatial_block_candidates"))}, RECONCILE_RULE)
    sql_rows = (sql.get("query_resale_blocks_by_criteria") or {}).get("blocks")
    spatial = state.get("spatial_block_candidates")
    matches = state.get("candidate_blocks_intersection") or []
    counts = {"price_and_lease_matches": len(sql_rows) if sql_rows is not None else None,
              "distance_matches": len(spatial) if spatial is not None else None,
              "blocks_meeting_every_constraint": len(matches)}
    market_context = {k: v for k, v in sql.items() if k != "query_resale_blocks_by_criteria"}
    searched = ({"market_sql", "spatial"} & set(state.get("active_subagents") or [])) or sql_rows is not None \
        or spatial is not None or matches or (state.get("user_constraints") or {}).get("towns")
    if not searched:
        # no flat search was asked (e.g. a pure eligibility question): no "no block meets every constraint" answer
        return ({**base, "market_context": market_context},
                "Answer the user's question from the evidence; no flat search was requested.")
    rule = GROUNDED_RULE
    if 0 < len(matches) <= 50:
        # in the 27 Oct rerun the writer listed 4 of 8 and 5 of 7 matches although the join was exact
        rule += f" List all {len(matches)} blocks in matching_blocks; do not shorten the list."
    return ({**base, "matching_blocks": matches[:50], "match_counts": counts, "market_context": market_context,
             "relaxation": state.get("relaxation")}, rule)


FACT_WORDS = {
    "cash_savings": "your cash savings",
    "cpf_oa_balance": "the CPF Ordinary Account amount you can use",
    "monthly_income": "your HFE-assessed monthly household income",
    "monthly_debt_obligations": "your monthly debt repayments (or that you have none)",
    "remaining_lease_years": "the remaining lease you want the calculation to assume",
    "youngest_buyer_age": "the age of each buyer", "oldest_buyer_age": "the age of each buyer",
    "average_applicant_age": "the age of each buyer",
    "youngest_core_member_age": "the age of the youngest person who will live in the flat",
    "citizenship_pair": "each buyer's citizenship", "first_timer_status": "whether you are first-timers",
    "flat_type": "the flat type", "eligibility_confirmed": "whether your grant eligibility is confirmed",
    "continuous_employment_12_months": "whether you have been employed continuously for the past 12 months",
    "proximity": "whether you are buying near or with your parents or children",
    "proximity_eligibility_confirmed": "whether your proximity grant eligibility is confirmed",
    "loan_eligibility_confirmed": "whether your HDB loan eligibility is confirmed",
}


def clarification_needed(state: HDBCompassState) -> Optional[Dict[str, Any]]:
    """Facts the grant and affordability calculators reported missing, as one plain question. Upfront costs are
    left out: they need one chosen flat's valuation, loan and quoted fees, which a search question cannot supply."""
    fin = state.get("financial_breakdown") or {}
    needed: List[str] = []
    for name in ("calculate_cpf_housing_grants", "calculate_maximum_affordability"):
        v = fin.get(name)
        if isinstance(v, dict) and v.get("status") == "needs_information":
            needed += v.get("missing_inputs") or []
    if "calculate_cpf_housing_grants" in fin:
        needed = [n for n in needed if n != "total_grants"]   # the grant calculator supplies it once its facts are known
    needed = list(dict.fromkeys(needed))
    if not needed:
        return None
    phrases = list(dict.fromkeys(FACT_WORDS.get(n, n.replace("_", " ")) for n in needed))
    listed = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return {"needed": needed, "question": f"To finish your grant and budget estimate, please tell me {listed}."}


def user_request(state: HDBCompassState) -> str:
    """What the answer must address. A reply to the follow-up question ("We are aged 32 and 30.") completes the
    earlier request, so the answer still covers everything first asked for (in the 27 Oct rerun it listed the
    blocks but dropped the grants and budget it now had)."""
    users = [m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
    if state.get("pending_clarification") and len(users) > 1:
        return (" ".join(users[:-1]) + f"\n(Reply to your follow-up question: {users[-1]})")
    return last_user_query(state)


_FIGURE_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s*(?:k|m)?\b|\b(?:Blk|Block)\s*\d{1,4}[A-Z]?\b", re.I)


def keep_supported_sentences(opening: str, facts: str) -> Tuple[str, int]:
    """The model's opening with every sentence dropped that states a price or block number not in the code-written
    facts, so it cannot contradict them. Returns (text, sentences dropped)."""
    known_money = {round(n) for n, _ in _stated_numbers(facts)}
    known_blocks = {b.upper() for b in re.findall(r"\b(?:Blk|Block)\s*(\d{1,4}[A-Z]?)\b", facts, re.I)}
    kept, dropped = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", opening.strip()):
        bad = False
        for fig in _FIGURE_RE.findall(sentence):
            blk = re.match(r"(?:Blk|Block)\s*(\d{1,4}[A-Z]?)", fig, re.I)
            if blk:
                bad |= blk.group(1).upper() not in known_blocks
            else:
                # rounding such as "$784k" for $784,366 is fine (0.2%); a different price is not
                bad |= not any(abs(n - v) <= max(1, 0.002 * v) for n, _ in _stated_numbers(fig) for v in known_money)
        if bad:
            dropped += 1
        else:
            kept.append(sentence)
    return " ".join(kept), dropped


def writes_facts_in_code(state: HDBCompassState) -> bool:
    """Variant C (and the app): the facts section of the answer (blocks, prices, leases, distances, relaxations,
    grants, budget) is written by code, the same text the no-model template produces, and the model writes only
    the opening and the trade-offs, so it cannot drop or garble a number. Variant B keeps the model as the writer
    by design (no join in code). Switch: architecture.code_written_facts."""
    return state.get("variant") != "B" and bool(load_config().get("architecture", {}).get("code_written_facts", True))


def synthesize_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    blocks = state.get("candidate_blocks_intersection") or []
    log: List[Dict[str, Any]] = []
    payload = build_map_payload(blocks, c, log) if blocks else {}
    evidence, rule = synthesis_evidence(state)
    clar = clarification_needed(state)
    if clar:
        evidence["clarification_needed"] = clar
        rule += (" Some financial facts are missing: state no maximum budget or affordability figure, and end with "
                 "clarification_needed.question.")
    verdict = state.get("auditor_verdict") or {}
    text = ""
    errors: List[str] = []
    llm = get_llm()
    facts = template_response(state) if writes_facts_in_code(state) else ""
    if llm is not None:
        try:
            if facts:
                rule = ("The facts section below was written by code from the tool results and will be shown to the user "
                        "exactly as written, straight after your text:\n---\n" + facts + "\n---\nWrite only a short opening "
                        "(one or two sentences that answer the question directly) and, when there are several options, the "
                        "trade-offs between them (price vs lease vs distance). Do not repeat or change any block, price, grant, "
                        "budget or relaxation figure, and do not contradict the facts.")
            msgs = [sys_msg(SYNTH_SYSTEM),
                    HumanMessage(content=f"User query: {user_request(state)}\n\n{rule}\n\nEvidence JSON:\n{json.dumps(evidence, default=str)[:24000]}")]
            if verdict and not verdict.get("passed", True) and verdict.get("critique"):
                msgs.append(HumanMessage(content=f"Your previous draft was rejected by the auditor: {verdict['critique']}. Rewrite it, fixing every issue."))
            text = llm.invoke(msgs).content
            if isinstance(text, list):
                text = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in text)
            text = strip_think(text)
            if facts:
                opening, dropped = keep_supported_sentences(text, facts)
                if dropped:
                    errors.append(f"synthesize: dropped {dropped} opening sentence(s) with figures not in the facts")
                text = (opening.strip() + "\n\n" if opening.strip() else "") + facts
        except Exception as e:  # noqa: BLE001
            errors.append(f"synthesize: LLM failed, template used: {type(e).__name__}")
    if not text:
        text = facts or template_response(state)
    elif state.get("financial_breakdown") or state.get("rag_policy_passages"):
        if "HFE" not in text:
            text += HFE_DISCLAIMER
    if clar and clar["question"] not in text:
        text = text.rstrip() + "\n\n" + clar["question"]
    return {"draft_response": text, "map_ui_payload": payload, "tool_call_log": log, "errors": errors,
            "clarification": clar}


# ---------------------------------------------------------------------------
# Member 5 hook points: input guardrail and runtime auditor
# ---------------------------------------------------------------------------
def _load_member5_hooks() -> Tuple[Optional[Callable], Optional[Callable]]:
    try:
        from agents import guardrails

    except Exception:  # noqa: BLE001
        return None, None
    return getattr(guardrails, "input_guardrail_node", None), getattr(guardrails, "runtime_auditor_node", None)


_INPUT_GUARDRAIL, _AUDITOR = _load_member5_hooks()


def set_hooks(input_guardrail: Optional[Callable] = None, auditor: Optional[Callable] = None) -> None:
    """Tests and Member 5 can inject nodes without import-time coupling."""
    global _INPUT_GUARDRAIL, _AUDITOR
    if input_guardrail is not None:
        _INPUT_GUARDRAIL = input_guardrail
    if auditor is not None:
        _AUDITOR = auditor


def input_guardrail_node(state: HDBCompassState) -> Dict[str, Any]:
    if _INPUT_GUARDRAIL is not None:
        return _INPUT_GUARDRAIL(state)
    return {}  # pass-through until Member 5 delivers


def guardrail_gate(state: HDBCompassState) -> str:
    return "finalize" if state.get("draft_response") else "extract_constraints"


def audit_node(state: HDBCompassState) -> Dict[str, Any]:
    if _AUDITOR is not None:
        return _AUDITOR(state)
    prev = state.get("auditor_verdict") or {}
    return {"auditor_verdict": AuditorVerdict(passed=True, critique="auditor not installed (pass-through)",
                                              retry_count=int(prev.get("retry_count", 0))).model_dump()}


def audit_gate(state: HDBCompassState) -> str:
    v = state.get("auditor_verdict") or {}
    max_retries = load_config()["architecture"]["auditor_max_retries"]
    # Convention (docs/INTERFACES.md §3): the auditor sets retry_count = number of rejections so far.
    # With auditor_max_retries = 2 the draft is rewritten after rejections 1 and 2, and the 3rd rejection ends the loop.
    if not v.get("passed", True) and int(v.get("retry_count", 0)) <= max_retries:
        return "synthesize"
    return "finalize"


def finalize_node(state: HDBCompassState) -> Dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": state.get("draft_response", "")}]}


def initial_state(query: str, variant: str, map_state: Optional[Dict[str, Any]] = None,
                  previous: Optional[Dict[str, Any]] = None) -> HDBCompassState:
    """previous: the final state of the last turn in the same conversation. Its messages, constraints and any
    open follow-up question carry forward, so a reply such as "we have $25,000 in cash" completes the earlier request."""
    history = list((previous or {}).get("messages") or [])
    return {"messages": history + [{"role": "user", "content": query}], "variant": variant, "active_subagents": [],
            "candidate_blocks_intersection": [], "map_ui_payload": map_state or {}, "draft_response": "",
            "auditor_verdict": {}, "tool_call_log": [], "errors": [], "subagent_notes": [],
            "prior_constraints": (previous or {}).get("user_constraints"),
            "pending_clarification": (previous or {}).get("clarification"), "clarification": None}


def conversation_text(state: HDBCompassState) -> str:
    """The latest message, preceded by the earlier turns when there are any (for agents without structured memory)."""
    msgs = [m for m in state.get("messages", []) if isinstance(m, dict)]
    if len(msgs) <= 1:
        return last_user_query(state)
    return "\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in msgs)
