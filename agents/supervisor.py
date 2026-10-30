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


# ---------------------------------------------------------------------------
# What the user states. Code reads a calculator fact only from a plain statement: not from a question ("Can we get
# $80,000 in grants?"), a hypothetical ("supposing my wife is a PR"), a doubt or plan ("not sure if no PHG applies",
# "waiting for HDB to confirm"), the past ("our income was 12k") or a request ("please confirm we are eligible").
# Money, ages and citizenship are read only from what is said about the buyers, never about parents, siblings or
# friends. When code is unsure it leaves the fact out and the assistant asks for it.
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
# clauses: "Income 9,000, cash 50k" -> two; "aged 32, 30" stays one; "I'm 45 and my husband's 52" -> two
_CLAUSE_SPLIT = re.compile(r",\s+(?=\D)|\s+(?:but|though|although|however|whereas)\s+|\s+-\s+|\s+and\s+(?=(?:my|our|his|her)\b)",
                           re.I)
_ASKING = re.compile(r"^(?:(?:and|so|also|oh|hi|hey|ok|okay|sorry|btw|then|um|hmm)\b[\s,]*)*(?:(?:what|how|when|where|why|"
                     r"which|who|can|could|would|will|do|does|did|is|are|am|should|shall|any|isn't|aren't|don't|doesn't)\b"
                     r"|(?:please\s+)?(?:confirm|check|verify|tell me|let me know|explain|find out|advise|compare)\b)", re.I)
# where the asking part starts inside a clause: "As an SC couple aged 32 and 30 with 9k income how much can we borrow?"
_ASKING_INSIDE = re.compile(r"\b(?:so\s+)?(?:how|what|which|why|when|where|who)\b|\b(?:can|could|would|will|do|does|did|is|are|am|"
                            r"should|shall)\s+(?:we|i|you|it|they|there|my|our|he|she)\b", re.I)
# a hedge makes the whole clause unsure: "We're eligible I hope", "we may be first-timers", "approval pending"
_HEDGE = re.compile(r"\b(?:unsure|not sure|not certain|uncertain|wonder\w*|i hope|hopefully|i think|i guess|i believe|"
                    r"probably|maybe|perhaps|possibly|might|may|could be|pending|to be confirmed|tbc|waiting)\b", re.I)
# a plan or condition: the clause counts only up to it ("We're an SC couple aged 32 and 30 and we want to know if ...",
# "We earn 9k a month which should be enough", "We'll confirm eligibility next week")
_PLAN = re.compile(r"\b(?:if|whether|unless|suppos\w*|in case|once|will|going to|gonna|need to|needs to|have to|has to|"
                   r"help me|plan(?:s|ned|ning)? to|intend\w* to|hop(?:e|es|ed|ing) (?:to|for)|want(?:s|ed)?|"
                   r"would like|like to|should|expect\w*|soon|so that)\b"
                   r"|\bassuming\s+(?:that\s+)?(?:my|our|we|i|he|she|they)\b|'ll\b|'d like\b", re.I)
# the past: not what is true now ("our income was 12k", "we sold our flat 2 years ago"); kept only for ownership history
_PAST = re.compile(r"\b(?:used to|was|were|previously|formerly|former|last time|last year|ago)\b", re.I)
_OTHER_PEOPLE = re.compile(r"\b(?:parents?|parents'|mum|mom|mother|dad|father|in-laws?|grand\w*|sisters?|brothers?|siblings?|"
                           r"cousins?|relatives?|friends?|colleagues?|neighbou?rs?|boss|someone|they|them|their|kids?|"
                           r"child(?:ren)?|sons?|daughters?|baby|ex-\w+)\b", re.I)
_OWN_START = re.compile(r"^(?:i|i'm|im|i am|we|we're|we are|my|our|me|us|both|income|salary|cash|cpf|oa|savings|husband|"
                        r"wife|hubby|partner|spouse|single|sc|household|combined|together|hfe|no|aged?|ages)\b", re.I)


def _sentences_and_clauses(text: str) -> List[Tuple[str, List[str]]]:
    out = []
    for sentence in _SENTENCE_SPLIT.split(re.sub(r"[ \t]{2,}", " ", text or "")):
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", sentence).strip()
        if s:
            out.append((s, [c.strip() for c in _CLAUSE_SPLIT.split(s) if c and c.strip()]))
    return out


def _other_person(clause: str) -> Optional[re.Match]:
    """The first mention of someone other than the buyers ("no kids" is about the buyers)."""
    for m in _OTHER_PEOPLE.finditer(clause):
        if not re.search(r"\b(?:no|without|zero|not have any|don't have any|do not have any)\s+$", clause[:m.start()], re.I):
            return m
    return None


def statements(text: str, own: bool = False, keep: Tuple[str, ...] = ()) -> str:
    """The plain statements in the text, clause by clause. In a question only the part before the asking words counts
    ("We're an SC couple, 32 and 30, how much grant can we get?" keeps "We're an SC couple, 32 and 30"); a question
    shaped like a statement ("Income ceiling is $14,000 right?") counts for nothing. A hedged clause ("we're eligible I
    hope") counts for nothing, a clause with a plan or condition counts up to it, and a clause in the past counts only
    with keep=("past",) (ownership history); keep=("plan",) keeps plans ("my parents will live with us"). own=True
    also drops what is said about other people ("my mum is 65", "my parents have 200k") up to the next clause about the
    buyers."""
    kept_sentences = []
    for sentence, clauses in _sentences_and_clauses(text):
        question = sentence.rstrip().endswith("?")
        keep_parts: List[str] = []
        asked = skipping = False
        for c in clauses:
            if _ASKING.match(c):
                asked = True
                if question:
                    break
                continue
            inside = _ASKING_INSIDE.search(c) if question else None
            if inside:
                asked, c = True, c[:inside.start()]
            if _HEDGE.search(c) or ("past" not in keep and _PAST.search(c)):
                if inside:
                    break
                continue
            plan = _PLAN.search(c) if "plan" not in keep else None
            if plan:
                c = c[:plan.start()]
            c = c.strip(" ,")
            other = _other_person(c) if own else None
            if other:
                # keep what comes before the phrase about the other person: "buying within 4km of parents" keeps
                # "buying within 4km"; "my mother, 68" and "my parents have 200k" keep nothing
                cuts = [x.start() for x in re.finditer(r"\b(?:my|our|his|her|their|the|a|an|with|of|for|to|and|from)\b",
                                                       c[:other.start()], re.I)]
                c, skipping = c[:cuts[-1] if cuts else other.start()].strip(" ,"), True
            elif own and skipping and not _OWN_START.match(c):
                c = ""
            else:
                skipping = False
            if c:
                keep_parts.append(c.rstrip(".!?;"))
            if inside:
                break
        if question and not asked:
            keep_parts = []
        if keep_parts:
            kept_sentences.append(", ".join(keep_parts))
    return ". ".join(kept_sentences) + ("." if kept_sentences else "")


def _clauses_of(text: str) -> List[str]:
    return [c for _, clauses in _sentences_and_clauses(text) for c in clauses]


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
    # "4-room", "4rm"; not a room count ("at least 2 rooms", "3 bedrooms")
    flat_types = [f"{m.group(1)} ROOM" for m in re.finditer(r"\b([2-5])\s*[- ]?\s*(?:room|rm)s?\b", q, re.I)
                  if not re.search(r"(?:at least|min(?:imum)?|more than|over|minimum of)\s*$", q[:m.start()], re.I)]
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
    if budget is None:
        # "my budget is 600k", "budget: $650,000", "budget around 650k", "650k budget"
        m = re.search(r"\bbudget\b\s*(?:is|of|:|=|around|about|approx\.?|approximately|roughly|~|\s)*\s*(?:S?\$|SGD\s?)?\s?"
                      r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\b"
                      r"|(?:S?\$|SGD\s?)?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\s+budget\b", q, re.I)
        if m:
            num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            value = _money(f"{num}{unit or ''}")
            if value and value >= 50_000:
                budget = value

    lease = None
    # "at least 60 yrs lease left", "65+ years of lease", "lease 65+ yrs", "remaining lease of 65 years or more"
    # not a flat's age ("at least 60 years old flat") or a working life ("30 years left before retirement")
    for pat in (r"(?:at least|minimum|min\.?|>=|≥)\s*(\d{2})\s*(?:years?|yrs?)\b(?!\s*(?:old|of age|left before|to retire|"
                r"before retire|till retire|until retire))",
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
        if dist < 20 or q[max(0, m.start() - 2):m.start() + 1].strip().startswith("$"):
            continue                                        # "1.2m to spend near an MRT" is money, not metres
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
        if dist >= 20:
            radii.setdefault(AMENITY_WORDS[m.group(3).lower()], int(dist))

    school = None
    m = re.search(r"(?:of|near|to)\s+((?:[A-Z][a-zA-Z'’]+\s){1,4}(?:Primary School|School|Primary|High))", q)
    if m:
        school = m.group(1).strip()
        radii.setdefault("school", 1000)

    own = statements(q, own=True)
    income = stated_income(own, is_couple(q))
    cpf = bound_amounts(own, is_couple(q)).get("cpf_oa_balance")   # paired with its own word (never the income just before it)

    # the legacy scheme label, from the same stated reading as the calculator facts (nothing inferred from "couple")
    pair, first = citizenship_stated(q), first_timer_stated(q)
    scheme = {"SC_Single": "SC_Single_35", "SC_PR": "SC_PR_FirstTimer" if first else None,
              "SC_SC": {True: "SC_SC_FirstTimer", False: "SC_SC_SecondTimer"}.get(first)}.get(pair or "")

    work = [w.strip() for w in re.findall(r"work(?:s)?\s+(?:at|in)\s+([A-Za-z0-9 \-']+?)(?=[,.;]|\s+and\b|$)", q, re.I)]
    max_commute = None
    for m in re.finditer(r"(?:under|within|less than|max)\s*(\d{2})\s*min(?:s|utes)?\b(?!\s*(?:walk|walking|on foot|by foot))", q, re.I):
        max_commute = int(m.group(1))                 # a commute, not "within 10 min walk to MRT"
        break

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


# An amount: "9,000", "8500", "8.5". Thousands separators only between digit groups, so "Income 9,000, cash 50k" never
# reads "9,000," as the start of the cash amount.
_NUMBER = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
# what may follow a number that is an age: not "50,000", "65 years of lease" ("32 years old" is fine), "24 months",
# "20 km", "90 sqm", "8k", "3-room"
_AGE_END = (r"(?![,.]\d)(?!\s*(?:years?|yrs?)\b(?!\s+old))(?!\s*(?:k\b|%|mins?\b|minutes|km\b|m\b|months?\b|mths?\b|"
            r"sq\s?m|sqm|storeys?|floors?|rooms?\b|-room|rm\b|\+|blocks?\b))")
# "aged 32 and 30", "we're 32 & 30", "ages: 32 and 30", "I am 38", "im 45", "we are both 30"
_AGES_RE = re.compile(r"\b(?:aged?|ages?:?|we're|we are|i'm|im|i am)\s+(?:both\s+)?(\d{2})(?:\s*(?:and|&|,|/|n\b)\s*(\d{2}))?\b"
                      + _AGE_END, re.I)
_NOT_BUYER = re.compile(r"\b(?:parents?|mum|mom|mother|dad|father|in-laws?|grand\w*|kids?|child(?:ren)?|sons?|daughters?|"
                        r"brothers?|sisters?|siblings?|friends?|baby)\b", re.I)
_PARTNER = r"(?:husband|wife|spouse|partner|fianc[eé]e?|hubby|girlfriend|boyfriend|gf|bf|missus)"
# "my husband is 34", "husband's 52", "wife 31", "hubby: 40"; never "husband 50,000 in CPF"
_PARTNER_AGE = re.compile(rf"\b(?:{_PARTNER}|he|she)(?:\s+is|'s|\s+aged|\s*[:,(-]?)\s*(?:aged\s+)?(\d{{2}})\b" + _AGE_END, re.I)
# "me 33", "myself: 38", "I (34)"
_SELF_AGE = re.compile(r"\b(?:me|myself)\s*[:,-]?\s*(\d{2})\b" + _AGE_END + r"|\bI\s*\(\s*(\d{2})\s*\)", re.I)
# "SC couple, 32 and 30", "Both SC, 45 and 47", "wife and I 27/28"
_PAIR_AGES = re.compile(rf"\b(?:couple|SCs?|singaporeans?|citizens?|both|married|{_PARTNER}\s+and\s+i)\b[^.\d$]{{0,15}}?"
                        rf"(\d{{2}})\s*(?:and|&|/|,|n\b)\s*(\d{{2}})\b" + _AGE_END, re.I)
# "My husband and I are 32 and 30", "both 29 years old": an age only when the phrase is about the buyers
_AGES_LOOSE = re.compile(r"\b(?:are|am|is|both)\s+(?:both\s+)?(?:aged\s+)?(\d{2})(?:\s*(?:and|&|,|/)\s*(\d{2}))?\b" + _AGE_END,
                         re.I)
_BUYER_WORDS = re.compile(rf"\b(?:i|we|us|both|{_PARTNER})\b", re.I)
_MONEY_WORDS = re.compile(r"\b(?:cpf|oa|cash|income|salary|savings|budget|loans?|price|lease|grants?|debts?|tenure|rate)\b", re.I)


def _phrase_before(text: str, start: int) -> str:
    """The words before a position in the same phrase (stops at punctuation and at "and")."""
    return re.split(r"[.,;!?\n]|\band\b", text[max(0, start - 50):start], flags=re.I)[-1]


def is_couple(text: str) -> bool:
    """The buyers are two people: a partner, "couple", "married", or "we"/"both" (unless the buyer says they are
    single)."""
    t = text or ""
    if re.search(rf"\b(?:{_PARTNER}|couple|engaged)\b|(?<!not )(?<!un)\bmarried\b", t, re.I):
        return True
    if re.search(r"\bsingle\b", t, re.I):
        return False
    return bool(re.search(r"\b(?:we|we're|we are|us|both)\b", t, re.I))


def buyer_ages(text: str, couple: Optional[bool] = None) -> List[int]:
    """The buyers' own ages. Skips ages that belong to someone else ("our parents are aged 65 and 62", "my mum is
    65") and numbers that are not ages ("50,000 in CPF", "24 months"); reads "my husband and I are 32 and 30", "me 33,
    wife 31" and a partner's age stated separately ("I'm 45 and my husband's 52"). A couple needs both ages: with one
    ("I'm 35 and my husband is 2 years older") nothing is returned, so the assistant asks instead of using one age
    for both buyers. couple=None works it out from the text."""
    t = text or ""
    ages: List[int] = []
    both = False
    for m in _AGES_RE.finditer(t):
        if _NOT_BUYER.search(_phrase_before(t, m.start())):
            continue
        ages, both = [int(a) for a in m.groups() if a], "both" in m.group(0).lower()
        break
    if not ages:
        for m in _PAIR_AGES.finditer(t):
            if not _NOT_BUYER.search(_phrase_before(t, m.start())) and not _MONEY_WORDS.search(m.group(0)):
                ages = [int(m.group(1)), int(m.group(2))]
                break
    if not ages:
        for m in _AGES_LOOSE.finditer(t):
            before = re.split(r"[.,;!?\n]", t[max(0, m.start() - 50):m.start()])[-1]
            said_both = m.group(0).lower().startswith("both")
            if (_BUYER_WORDS.search(before) or said_both) and not _NOT_BUYER.search(before) and not _MONEY_WORDS.search(before):
                ages, both = [int(a) for a in m.groups() if a], "both" in m.group(0).lower()
                break
    for pattern in (_PARTNER_AGE, _SELF_AGE):
        for m in pattern.finditer(t):
            age = int(next(g for g in m.groups() if g))
            if age not in ages and not _NOT_BUYER.search(_phrase_before(t, m.start())):
                ages.append(age)
    ages = [a for a in ages if 18 <= a <= 99]
    if both and len(ages) == 1:
        ages = ages * 2                                    # "we are both 30"
    if len(ages) > 2:
        return []                                          # more ages than buyers: not sure which are theirs
    if len(ages) == 1 and (is_couple(t) if couple is None else couple):
        return []
    return ages


# Money facts: an amount is the cash, CPF or income only when that word sits right next to it ("cash 40k", "40k
# cash", "income: $8,500", "we take home $9,000 a month"). Income is never what the buyer pays ("afford $2,500 a
# month", "car loan $800/month", "$500 a month allowance for my parents") or a yearly figure.
_AMOUNT_TOKEN = re.compile(r"(?:S?\$|\bSGD\s?)?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k|thousand)?\b"
                           r"(?!\s*(?:%|m\b|km\b|years?|yrs?|mins?\b|minutes|months?|mths?|rooms?|rm\b|-room))", re.I)
_FACT_WORDS = {"cash_savings": r"\bcash(?:\s+savings)?\b|(?<!cpf )(?<!oa )\bsavings\b",
               "cpf_oa_balance": r"\bcpf(?:\s+oa)?\b|\boa\b|\bordinary account\b",
               "monthly_income": r"\b(?:household\s+|monthly\s+|combined\s+)?income\b|\bsalar(?:y|ies)\b|\bearn(?:s|ing|ings)?\b"
                                 r"|\bmakes?\b|\bmaking\b|\bgets\b|\bdraws?\b|\btakes?[- ]?home\b|\bbrings? home\b"
                                 r"|\bhousehold(?=\s+(?:now|is|of|:|~|\$|\d))"}
# what may sit between a fact word and its amount: "cash 40k", "CPF OA of $90,000", "cpf of ard 40k", "income: $8,500",
# "savings are $30,000", "CPF OA available is $90,000", "cash is actually 60k"; and between an amount and its word:
# "40k cash", "$90,000 in CPF"
_WORD_THEN_AMOUNT = re.compile(r"^[\s:=]*(?:(?:of|is|are|ard|around|about|approx\.?|roughly|balance|available|assessed|hfe|"
                               r"we can use|i can use|totals?|comes? to|actually|now|abt|~|got|only|just|exactly|currently|savings|"
                               r"in total)[\s:=]*)*$", re.I)
_AMOUNT_THEN_WORD = re.compile(r"^\s*(?:sgd\s*)?(?:in\s+)?(?:(?:my|our|the)\s+)?$", re.I)
_PAYMENT_BEFORE = re.compile(r"\b(?:afford\w*|spend\w*|pay|paying|pays|repay\w*|instal\w*|rent\w*|mortgage|loans?|debts?|owe\w*|"
                             r"allowance|give|giving|budget\w*|commit\w*|limit|max(?:imum)?|keep|no more than|most|handle|"
                             r"ideally|car|costs?|fees?|expenses?|ceiling|cap)\b", re.I)
_PAYMENT_AFTER = re.compile(r"^\s*(?:\S+\s+){0,6}?(?:for|on|towards|to)\s+(?:the\s+|our\s+|a\s+|my\s+)?(?:flat|housing|house|"
                            r"home|mortgage|loan|rent|instal\w*|repayments?|car|parents?)\b"
                            r"|^\s*(?:\S+\s+){0,2}?(?:allowance|rent|repayments?|instal\w*|mortgage|car loan|loan)\b", re.I)
_ANNUAL = re.compile(r"\b(?:a|per|each|every)\s+year\b|\bannual\w*|\byearly\b|\bper annum\b|\bp\.?a\.?(?!\w)|/\s*(?:yr|year)\b",
                     re.I)
_INCOME_CUE = re.compile(r"\b(?:income|salar(?:y|ies)|earn\w*|make|making|take[- ]?home|bring home|hfe|assessed|household|"
                         r"combined|together)\b", re.I)
_PER_MONTH = re.compile(r"^\s*(?:/\s*|per\s+|a\s+|each\s+)(?:month|mth)\b", re.I)
# one person's share: "70k each", "7k (me) + 3k (wife)", "4k and 5k respectively"; not "9k (combined)"
_SPLIT_AMOUNT = re.compile(r"\s*(?:each|per (?:person|pax|head)|apiece|respectively|\+|\(\s*(?:me|mine|my|his|hers?|him|wife|"
                           r"husband|hubby|partner|spouse|she|he)\b)", re.I)
_SHARE_BEFORE = re.compile(r"\beach\b|\bboth of us\b|\bwe both\b|\bboth\s+(?:earn|have|make|get|draw)", re.I)
_PARTNER_WORD = re.compile(rf"\b(?:{_PARTNER}(?:'s)?|she|he|her|his|hers)\b", re.I)
_SELF_WORD = re.compile(r"\b(?:i|i'm|my|me|mine)\b", re.I)
_HOUSEHOLD_WORD = re.compile(r"\b(?:we|our|us|household|combined|together|total|both|joint|family)\b", re.I)
_COMBINED = re.compile(r"\b(?:cash|savings|cpf|oa|income)\b\s*(?:and|plus|&|\+)\s*(?:my\s+|our\s+)?(?:cash|savings|cpf|oa|income)\b",
                       re.I)
_CORRECTION = re.compile(r"\b(?:i mean|sorry|actually|correction|no wait|oops|or rather)\b", re.I)


def _money_claims(text: str, couple: Optional[bool] = None) -> List[Tuple[float, set, bool, int]]:
    """(value, the facts it can be, split, position) for every amount next to a cash, CPF or income word; split when
    it is one person's share ("70k each", "we each earn 5k", "my wife earns 3k", "7k (me) + 3k (wife)"; in a couple,
    "I earn 6k") or a combined total ("cash and CPF together 150k"). A "$X a month" with an income word in its clause
    and nothing about paying counts as income ("HFE assessed: $9,000/month")."""
    t = text or ""
    couple = is_couple(t) if couple is None else couple
    words = sorted((m.start(), m.end(), field) for field, pat in _FACT_WORDS.items() for m in re.finditer(pat, t, re.I))
    raw = []
    for m in _AMOUNT_TOKEN.finditer(t):
        value = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        if value < 100:
            continue                                                   # an age or a count, not money
        before = {f for ws, we, f in words if we <= m.start() and m.start() - we <= 30 and _WORD_THEN_AMOUNT.match(t[we:m.start()])}
        after = {f for ws, we, f in words if ws >= m.end() and ws - m.end() <= 15 and _AMOUNT_THEN_WORD.match(t[m.end():ws])}
        clause = re.split(r"[.;!?\n]|,\s+(?=\D)", t[max(0, m.start() - 200):m.start()])[-1]
        rest = re.split(r"[.;!?\n]|,\s+(?=\D)", t[m.end():m.end() + 200])[0]          # the rest of the amount's clause
        paying = bool(_PAYMENT_BEFORE.search(clause) or _PAYMENT_AFTER.match(rest) or _ANNUAL.search(clause)
                      or _ANNUAL.search(t[m.end():m.end() + 20]))
        if paying:
            before.discard("monthly_income")
            after.discard("monthly_income")
        if not before and not after and not paying and _PER_MONTH.match(rest) and _INCOME_CUE.search(clause + " " + rest):
            before = {"monthly_income"}
        if before or after:
            own_share = not _HOUSEHOLD_WORD.search(clause) and (
                _PARTNER_WORD.search(clause) or (couple and _SELF_WORD.search(clause)))
            combined = any(not re.search(r"\d\s*(?:k|thousand)?\s*(?:sgd\s*)?(?:in\s+)?$", clause[:c.start()], re.I)
                           for c in _COMBINED.finditer(clause))          # "cash and CPF 150k", not "150k cash and cpf of 40k"
            split = bool(_SPLIT_AMOUNT.match(rest) or _SHARE_BEFORE.search(clause) or combined or own_share)
            raw.append((m.start(), value, before, after, split))
    # an amount with a fact word on both sides ("cash 40k cpf 100k") goes with the way the message pairs them: word
    # first if its first amount follows its word, amount first if it precedes it ("40k cash and 100k cpf")
    first = next((r for r in raw if not (r[2] and r[3])), None)
    word_first = first is None or bool(first[2])
    return [(value, (before if word_first else after) if before and after else (before or after), split, pos)
            for pos, value, before, after, split in raw]


def _bind_amounts(text: str, couple: Optional[bool] = None) -> Tuple[Dict[str, float], set]:
    t = text or ""
    claims = _money_claims(t, couple)
    ambiguous = {f for _, fields, split, _ in claims if split for f in fields}
    out: Dict[str, float] = {}
    pending = [(value, set(fields)) for value, fields, split, _ in claims if not split]
    changed = True
    while changed:
        changed = False
        for value, fields in pending:
            open_fields = fields - set(out)
            if len(open_fields) == 1:
                out[open_fields.pop()] = value
                fields.clear()
                changed = True
    for field in out:
        values = {v for v, fields, split, _ in claims if not split and fields == {field}} | {out[field]}
        if len(values) > 1:
            ambiguous.add(field)                                       # "I earn 6k, my wife earns 4k": not one figure
    # another amount in the same sentence that belongs to the partner ("my wife 4k", "hers 3k") or corrects one
    # ("7,000 per month, I mean 7,500") makes that sentence's figures unsure
    claimed = {pos for _, _, _, pos in claims}
    for sentence in re.finditer(r"[^.;!?\n]+", t):
        here = [(fields, pos) for _, fields, _, pos in claims if sentence.start() <= pos < sentence.end()]
        if not here:
            continue
        for m in _AMOUNT_TOKEN.finditer(t, sentence.start(), sentence.end()):
            if m.start() in claimed or float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1) < 100:
                continue
            lead = t[max(sentence.start(), m.start() - 25):m.start()]
            if _PARTNER_WORD.search(lead) or _CORRECTION.search(t[sentence.start():m.start()]) or \
                    re.search(r"\brespectively\b", sentence.group(0), re.I):
                ambiguous |= {f for fields, pos in here for f in fields}
    return {f: v for f, v in out.items() if f not in ambiguous}, ambiguous


def bound_amounts(text: str, couple: Optional[bool] = None) -> Dict[str, float]:
    """Cash, CPF and income amounts, each paired with the fact word right next to it ("cash 40k cpf 100k",
    "40k cash and 100k cpf", "Income 9,000, cash 50k"). An amount that two facts could claim goes to the one that
    has no other candidate. A fact stated with two different amounts, as one person's share or as a combined total,
    is left out, as is an amount that stays ambiguous, so the assistant asks for it instead of guessing."""
    return _bind_amounts(text, couple)[0]


def stated_income(own_text: str, couple: Optional[bool] = None) -> Optional[float]:
    return bound_amounts(own_text, couple).get("monthly_income")


def _re_amount(m: re.Match) -> float:
    amount, k = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return float(amount.replace(",", "")) * (1000 if k else 1)


# Citizenship. A buyer who is a permanent resident: "one Singapore citizen and one permanent resident", "SC + PR couple",
# "SC-PR couple", "Singaporean and PR couple", "one of us is a citizen and the other a PR", "my husband is a (Malaysian)
# PR", "my wife has PR", "me PR". Not "no PR", "not a PR", "a former PR", "getting PR next year", "the SC/PR quota",
# "what if my husband is a PR?" or "our friends are a citizen-PR couple".
_PR = r"(?:S?PRs?|permanent residents?)"
_CITIZEN_WORD = r"(?:singapore(?:an)?\s+)?(?:citizens?|SC|singaporeans?)"
_NATIONALITY = (r"(?:malaysian|chinese|indian|indonesian|filipin[oa]|vietnamese|thai|burmese|korean|japanese|british|american|"
                r"australian|taiwanese|bangladeshi|sri lankan|pakistani|european|french|german)")
_NOT_A_PR_YET = r"(?!\s+(?:quotas?|ratios?|limits?|households?|holders?|status|scheme))"
_PR_BUYER = re.compile(
    rf"\bone\s+{_CITIZEN_WORD}\b.{{0,20}}\bone\s+{_PR}\b|\bone\s+{_PR}\b.{{0,20}}\bone\s+{_CITIZEN_WORD}\b"
    rf"|\bone of us is (?:an?\s+)?{_CITIZEN_WORD}\b.{{0,30}}\bother\b.{{0,12}}\b{_PR}\b"
    rf"|\bone of us is (?:an?\s+)?{_PR}\b.{{0,30}}\bother\b.{{0,12}}\b{_CITIZEN_WORD}\b"
    rf"|\b(?:SC|citizens?|singaporeans?)\s*(?:\+|and|&|/|-)\s*(?:an?\s+)?S?PR\b{_NOT_A_PR_YET}"
    rf"|\bS?PR\s*(?:\+|and|&|/|-)\s*(?:an?\s+)?(?:SC|citizens?|singaporeans?)\b{_NOT_A_PR_YET}"
    rf"|\b(?:i'?m|i am|me|(?:my\s+)?{_PARTNER}(?:\s+is|'s)?)\s*[:,-]?\s*(?:an?\s+)?(?:{_NATIONALITY}\s+)?{_PR}\b{_NOT_A_PR_YET}"
    rf"|\b(?:i|{_PARTNER})\s+(?:has|have|holds?)\s+(?:an?\s+)?(?:S?PR|PR status|permanent residency)\b",
    re.I)
_FOREIGN = re.compile(
    rf"\bforeign(?:er|ers)?\b(?!\s+(?:S?PRs?|permanent))|\bnon[- ]?(?:citizens?|singaporeans?)\b"
    rf"|\bnot\s+(?:yet\s+)?(?:an?\s+)?(?:singapore(?:an)?\s+)?(?:citizens?|singaporeans?|SCs?)\b"
    rf"|\b(?:i'?m|i am|we(?:'re| are)|me|{_PARTNER}(?:\s+is|'s)?)\s+(?:both\s+)?(?:an?\s+)?{_NATIONALITY}s?\b(?!\s+(?:S?PRs?|permanent))"
    rf"|\b(?:we(?:'re| are)|both)\s+(?:both\s+)?(?:are\s+)?(?:PRs|permanent residents)\b", re.I)
_SC_COUPLE = re.compile(
    r"\bsingapore(?:an)?\s+citizens?\s+couple\b|\bsingaporean\s+couple\b|\bSC\s+couple\b|\bcitizen\s+couple\b"
    r"|\bcouple\s+of\s+(?:SCs|singapore(?:an)?\s+citizens|singaporeans)\b"
    r"|\bboth\s+(?:are\s+)?(?:singapore(?:an)?\s+)?(?:citizens|SCs?|singaporeans)\b"
    r"|\bwe(?:'re| are)\s+(?:both\s+)?(?:singapore(?:an)?\s+)?(?:citizens|singaporeans|SCs)\b|\bSC\s*(?:\+|&|and|/)\s*SC\b"
    rf"|\b{_PARTNER}\s+and\s+i\s+are\s+(?:both\s+)?(?:singapore(?:an)?\s+)?(?:citizens|singaporeans|SCs)\b"
    r"|\bboth of us are\s+(?:singapore(?:an)?\s+)?(?:citizens|singaporeans|SCs)\b|\bboth\s+singaporean\b"
    r"|\b(?:2|two)\s+(?:SCs|singaporeans|singapore citizens)\b", re.I)
# buying with someone who is not a spouse: not the couple schemes
_NOT_A_COUPLE = re.compile(r"\bjoint singles?\b|\btwo singles\b|\bwith (?:my|a) (?:friend|sister|brother|sibling|cousin|colleague)\b"
                           r"|\b(?:me|i) and my (?:friend|sister|brother|sibling|cousin|colleague)\b", re.I)
# talk about citizenship in general, not the buyers' own: "explain the difference between a citizen and a PR"
_GENERAL_TALK = re.compile(r"\b(?:explain|difference|differ\w*|between|compare\w*|versus|vs\.?|quotas?|ratios?|rules?|polic\w*|"
                           r"buyers|applicants|households|couples|people|eip)\b", re.I)


def pr_buyer(text: str) -> bool:
    """True when the user states that a buyer is a permanent resident, in a plain statement (not a question such as
    "what if my husband is a PR?" and not negated: "my wife is not a PR")."""
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text or ""):
        m = _PR_BUYER.search(sentence)
        if not m or sentence.strip().endswith("?") or re.search(r"\bwhat if\b|\bif (?:my|our|i|we)\b", sentence, re.I):
            continue
        if re.search(r"(?:\bnot|n't|\bno longer|\bneither)\b[^.;,]{0,20}$", sentence[:m.end() - len(m.group(0).split()[-1])], re.I):
            continue
        return True
    return False


def _citizen_view(text: str) -> str:
    return ". ".join(c for c in _clauses_of(statements(text, own=True)) if not _GENERAL_TALK.search(c))


def citizenship_stated(text: str) -> Optional[str]:
    """The citizenship pair the buyers state about themselves, or None when it is not stated plainly or the message
    contradicts itself (a citizen couple that also mentions a PR, a foreign partner, "one citizen")."""
    v = _citizen_view(text)
    if not v or _NOT_A_COUPLE.search(text or ""):
        return None
    foreign = _FOREIGN.search(v)
    citizen = re.search(rf"\b(?:citizens?|singaporeans?|SC)\b", v, re.I)
    if pr_buyer(v):
        return "SC_PR" if citizen and not foreign else None
    if foreign or any(not re.search(r"\b(?:no|not|not an?|neither|non|nor)\s*-?\s*$", v[max(0, m.start() - 12):m.start()], re.I)
                      for m in re.finditer(rf"\b{_PR}\b", v, re.I)):
        return None                                    # "no PR" is fine; any other PR mention is not a citizen couple
    if re.search(r"\bsingle\b", v, re.I) and citizen and \
            not re.search(rf"\b(?:{_PARTNER}|couple|engaged)\b|(?<!not )(?<!un)\bmarried\b", text or "", re.I):
        return "SC_Single"
    if _SC_COUPLE.search(v) and not re.search(r"\bone\s+(?:singapore(?:an)?\s+)?(?:citizen|SC)\b", v, re.I):
        return "SC_SC"
    return None


_FIRST_TIME = r"(?:first[- ]?timers?|1st[- ]?timers?|first[- ]time (?:home )?(?:buyers?|applicants?))"
_NOT_FIRST = re.compile(rf"\b(?:second|2nd)[- ]?timers?\b|\b(?:not|no|no longer|aren't|isn't|don't|doesn't|never been|"
                        rf"neither of us(?: is| are)?)\s+(?:\w+\s+){{0,3}}?{_FIRST_TIME}", re.I)
# a buyer who owns or owned a home ("we sold our flat 2 years ago", "my wife owns a condo", "had a 3-room flat")
_OWNED = re.compile(r"(?<!never )(?<!not )(?<!n't )\b(?:sold|owned|owns|own|bought|selling|inherited|had|has|have)\s+"
                    r"(?:(?:a|an|the|our|my|his|her|their)\s+)?(?:[\w-]+\s+){0,2}?(?:flats?|condos?|hdbs?|propert(?:y|ies)|"
                    r"homes?|houses?|btos?|units?|ecs?|executive condo\w*)\b", re.I)
_FIRST_CUE = re.compile(rf"\b{_FIRST_TIME}\b|\bnever owned\b|\bfirst (?:home|flat)\b", re.I)


def first_timer_stated(text: str) -> Optional[bool]:
    """True when the buyers plainly say they are first-timers and nothing says one of them owned a home; False when
    they say they are not; None otherwise (a question, a doubt, the past, or a first-timer who sold a flat)."""
    s, history = statements(text), statements(text, keep=("past",))
    if _NOT_FIRST.search(history):
        return False
    for c in _clauses_of(s):
        m = _FIRST_CUE.search(c)
        if m and not (_NEGATED_BEFORE.search(c[:m.start()]) and not m.group(0).lower().startswith("never")):
            history = re.sub(r"\b(?:have\s+|has\s+)?never\s+(?:owned|had|bought)\b", " ", history, flags=re.I)
            return None if _OWNED.search(history) else True
    return None


_KID_WORDS = re.compile(r"\b(?:kids?|child(?:ren)?|sons?|daughters?|baby|babies|toddlers?|infants?)\b", re.I)


def no_children(text: str) -> bool:
    """The buyers say they have no children, and nothing they say later mentions one ("we just had a baby")."""
    s = statements(text)
    return bool(re.search(_NO_KIDS, s, re.I)) and not _KID_WORDS.search(re.sub(_NO_KIDS, " ", s, flags=re.I))


_DEBT_NONE = re.compile(r"\bno (?:other |outstanding |existing |monthly )?(?:debts?|loans?|liabilities|car loans?|financial "
                        r"commitments|(?:debt|loan) (?:payments|obligations))\b|\bdebt[- ]free\b|\bzero debts?\b"
                        r"|\b(?:don'?t|do not) have any (?:other )?(?:debts?|loans?)\b", re.I)
_DEBT_SOME = re.compile(r"\b(?:car|study|student|personal|renovation|reno|education)\s+loans?\b|\bcredit[- ]cards?\b|\bowe\w*\b"
                        r"|\bdebts?\b|\bloans?\b", re.I)


def no_debts_stated(statement_text: str) -> bool:
    """The buyers say they have no debts, without an exception ("no debts except a study loan") and without a debt
    mentioned elsewhere ("car loan of $600 a month"). A housing loan ("HDB loan eligibility", "a loan of $500,000") is
    not a debt."""
    found = False
    for c in _clauses_of(statement_text):
        none = [(m.start(), m.end()) for m in _DEBT_NONE.finditer(c)]
        if none:
            if re.search(r"\b(?:except|other than|apart from|besides|only|aside from|save for)\b", c, re.I):
                return False
            found = True
        for m in _DEBT_SOME.finditer(c):
            if any(a <= m.start() < b for a, b in none):
                continue
            before, after = c[:m.start()], c[m.end():]
            if re.search(r"\b(?:no|not|without|zero|any|nil|never)\b[^,.;]{0,20}$", before, re.I) \
                    or re.search(r"\b(?:hdb|bank|housing|home|mortgage|max(?:imum)?)\s*$", before, re.I) \
                    or re.search(r"\bgrants?\s+(?:and|&|\+)\s*$", before, re.I) \
                    or re.match(r"\s*(?:eligib\w*|tenure|amount|type|approval|limit|quantum|calc\w*|estimate|ceiling)", after, re.I) \
                    or (re.match(r"\s*of\s+\$", after) and not re.match(r"\s*of\s+\$\s?[\d,.]+\s*k?\s*(?:a|per|/)\s*(?:month|mth)", after, re.I)):
                continue
            return False
    return found


def _stated_match(pattern: str, statement_text: str) -> Optional[re.Match]:
    """The first match in a clause of the statements that is not negated just before it."""
    for c in _clauses_of(statement_text):
        m = re.search(pattern, c, re.I)
        if m and not _NEGATED_BEFORE.search(c[:m.start()]) and not re.search(r"\bnot\b|n't\b|\bnever\b|\byet\b", m.group(0), re.I):
            return m
    return None


def parse_clarification_answer(text: str, needed: List[str], conversation: str = "") -> Dict[str, Dict[str, Any]]:
    """No-model reading of a reply to the follow-up question, for the facts the calculators most often lack (cash,
    ages). Only stated facts are recorded; the youngest core member equals the youngest buyer only when the
    conversation says there are no children. The model path reads any fact through structured extraction."""
    found: Dict[str, Any] = {}
    own = statements(text or "", own=True)
    cash = bound_amounts(own, is_couple(f"{conversation} {text}")).get("cash_savings")
    if cash is not None and "cash_savings" in needed:
        found["cash_savings"] = cash
    ages = buyer_ages(own, couple=is_couple(f"{conversation} {text}"))
    if ages and any("age" in n for n in needed):
        young, old = min(ages), max(ages)
        found.update(youngest_buyer_age=young, oldest_buyer_age=old, average_applicant_age=sum(ages) / len(ages))
        if no_children(f"{conversation}. {text}"):
            found["youngest_core_member_age"] = young
    if no_debts_stated(statements(text or "")) and "monthly_debt_obligations" in needed:
        found["monthly_debt_obligations"] = 0.0
    return {sec: {f: v for f, v in found.items() if f in _section_fields(sec)} for sec in FIN_SECTIONS
            if any(f in _section_fields(sec) for f in found)}


# Calculator facts the user must state. Code reads them from the user's own words (stated_financial_inputs); the
# model's value for one of them is used only when the user's words support it (fact_is_grounded), because in the live
# run it turned "married Singapore citizen couple" into SC_Single, invented ages and an interest rate, and passed zero
# grants. The model may still set the remaining fields (flat category and type, loan type, the purchase price to cost).
FACT_FIELDS = {
    "citizenship_pair", "first_timer_status", "monthly_income", "youngest_buyer_age", "youngest_core_member_age",
    "oldest_buyer_age", "average_applicant_age", "remaining_lease_years", "eligibility_confirmed",
    "loan_eligibility_confirmed", "continuous_employment_12_months", "proximity", "proximity_eligibility_confirmed",
    "near_parents_4km", "cpf_oa_balance", "cash_savings", "monthly_debt_obligations", "total_grants", "interest_rate",
    "tenure_years", "legal_and_valuation_fees", "additional_buyer_stamp_duty", "ltv_ratio",
}
AGE_FIELDS = {"youngest_buyer_age", "oldest_buyer_age", "average_applicant_age", "youngest_core_member_age"}
MONEY_FIELDS = ("cash_savings", "cpf_oa_balance", "monthly_income")
FIN_TOOL_SECTION = {"calculate_cpf_housing_grants": "grants", "calculate_maximum_affordability": "affordability",
                    "calculate_bsd_and_upfront_outlay": "upfront"}
_AMOUNT = r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?"
# "buying a 4-room for $600k"; not a budget ("looking to buy in Tampines for $600k or less", "for $600k max")
_PURCHASE_PRICE = (r"(?:buy(?:ing)?|purchas(?:e|ing))\b[^$.]{0,40}\bfor " + _AMOUNT
                   + r"(?!\d|[,.]\d|\s*k\b)(?!\s*(?:(?:or|and)\s+(?:less|lower|below|under|cheaper)|max\w*|at most|tops)\b)")
# the lease the user wants the calculation to assume: "use 65 yrs remaining lease for the calc", "calc w 65 yrs lease",
# "using a 65-year remaining lease", "assume 65 years of remaining lease"
_CALC_LEASE = (r"(\d{2})\s*(?:years?|yrs?) of remaining lease for the .{0,30}calc"
               r"|\b(?:assume|use|using)\s+(?:a\s+)?(\d{2})[- ]?(?:years?|yrs?)\b(?:\s+of)?(?:\s+remaining)?\s+lease"
               r"|\bcalc\w*\s+(?:w|with|using)\s+(\d{2})\s*(?:years?|yrs?)\b(?:\s+of)?(?:\s+remaining)?\s+lease")
_CONTINUOUS_WORK = (r"(?:continuous(?:ly)?\s+(?:employ\w*|working)|(?:employ\w*|working)\s+(?:\w+\s+){0,2}continuously)"
                    r".{0,40}\b(?:12|twelve)\s*(?:months?|mths?)\b"
                    r"|(?:employ\w*|working)\s+(?:12|twelve)\s*(?:months?|mths?)\s+continuously")


def _amount(m: re.Match) -> float:
    return float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)


def stated_financial_inputs(text: str, context: str = "") -> Dict[str, Dict[str, Any]]:
    """The calculator facts the user's own words state, in the financial_inputs shape. Nothing is inferred:
    "couple" alone gives no citizenship, and the youngest core member is the youngest buyer only when the
    user says there are no children. Facts come only from plain statements (statements()); money, ages and
    citizenship only from what is said about the buyers. context: the earlier messages, which say whether the
    buyers are a couple (a later "sorry, I'm 33" names one of two ages)."""
    t = text or ""
    s, own = statements(t), statements(t, own=True)
    f: Dict[str, Any] = {}
    pair = citizenship_stated(t)
    if pair:
        f["citizenship_pair"] = pair
    first = first_timer_stated(t)
    if first is not None:
        f["first_timer_status"] = first
    ages = buyer_ages(own, couple=is_couple(f"{context} {t}"))
    if ages:
        f.update(youngest_buyer_age=min(ages), oldest_buyer_age=max(ages), average_applicant_age=sum(ages) / len(ages))
        if no_children(f"{context}. {t}"):
            f["youngest_core_member_age"] = min(ages)
    f.update(bound_amounts(own, is_couple(f"{context} {t}")))
    if no_debts_stated(s):
        f["monthly_debt_obligations"] = 0.0
    if _stated_match(_CONTINUOUS_WORK, s) and not _JOB_LOSS.search(s):
        f["continuous_employment_12_months"] = True
    if _stated_match(r"eligibility for (?:the )?grants?.{0,40}\bconfirmed\b|\bgrants?\b.{0,30}\beligibility\b.{0,25}\bconfirmed\b", s):
        f["eligibility_confirmed"] = True
    if _stated_match(r"eligibility for .{0,30}\b(?:HDB|bank) loan (?:is |has been )?confirmed"
                     r"|\b(?:HDB|bank) loan\b.{0,15}\beligibility\b.{0,25}\bconfirmed\b"
                     r"|\b(?:HDB|bank) loan(?: eligibility)? (?:is |has been )?confirmed", s):
        f["loan_eligibility_confirmed"] = True
    if re.search(r"\bbank loan\b", t, re.I):
        f["loan_type"] = "bank"
    elif re.search(r"\bHDB loan\b", t, re.I):
        f["loan_type"] = "hdb"
    if re.search(r"\bno proximity (?:housing )?grant\b|\bno PHG\b|\bproximity (?:housing )?grant (?:does not|doesn't) apply\b", s, re.I):
        f["proximity"] = "none"
    if re.search(r"\bno (?:housing )?grants\b", s, re.I):
        f["total_grants"] = 0.0
    m = re.search(_CALC_LEASE, s, re.I)
    if m:
        f["remaining_lease_years"] = float(next(g for g in m.groups() if g))
    m = _stated_match(r"interest rate of (\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%\s*(?:p\.a\.|per annum|interest)", s)
    if m:
        f["interest_rate"] = float(m.group(1) or m.group(2))
    m = _stated_match(r"tenure of (\d{1,2}) years|(\d{1,2})[- ]year (?:loan )?tenure", s)
    if m:
        f["tenure_years"] = int(m.group(1) or m.group(2))
    if re.search(r"\bresale\b", t, re.I):
        f["flat_category"] = "resale"
    elif re.search(r"\bbto\b", t, re.I):
        f["flat_category"] = "bto"
    flat_types = rule_based_extract(t).flat_types
    if len(flat_types) == 1:
        f["flat_type"] = flat_types[0]
    upfront: Dict[str, Any] = {}
    m = re.search(_PURCHASE_PRICE, s, re.I)
    if m:
        upfront["purchase_price"] = _amount(m)
    for field, pattern in (("market_valuation", r"valued at " + _AMOUNT), ("loan_amount", r"loan of " + _AMOUNT),
                           ("legal_and_valuation_fees", r"fees of " + _AMOUNT)):
        m = re.search(pattern, s, re.I)
        if m:
            upfront[field] = _amount(m)
    if re.search(r"\bno ABSD\b|\bno additional buyer'?s stamp duty\b", s, re.I):
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


_GROUNDING_WORDS = {
    "cash": r"\bcash\b|\bsavings?\b|\bsaved\b", "cpf": r"\bcpf\b|\boa\b|ordinary account",
    "income": r"\bincome\b|\bearn\w*|\bsalar(?:y|ies)\b|\btake[- ]?home\b|\bbring home\b", "lease": r"\blease\b",
    "debt": r"debt|loan|repay|instal|commitment|mortgage|\brent\b|afford|\bspend|\bpay\b|paying|allowance|\bowe",
    "grant": r"\bgrants?\b", "fee": r"\bfees?\b", "absd": r"ABSD|stamp duty",
    "rate": r"\binterest\b|\brate\b|p\.a\.|per annum", "tenure": r"\btenure\b|[- ]year loan\b",
}


def _matches_of(value: Any, text: str) -> List[re.Match]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return []
    return [m for m in _NUM_RE.finditer(text or "")
            if any(abs(n - v) <= max(0.5, 0.005 * abs(v)) for n, _ in _stated_numbers(m.group(0)))]


def _number_near(value: Any, text: str, group: str, window: int = 35) -> bool:
    """The number appears in the text and the nearest fact word to it (within the window) is this fact's
    (_GROUNDING_WORDS[group]): in "$80,000 in CPF OA, no cash savings" the 80,000 is CPF's, never cash's; in
    "remaining lease: 65 years" the 65 is the lease's."""
    t = text or ""
    for m in _matches_of(value, t):
        nearest: Dict[str, int] = {}
        for g, pat in _GROUNDING_WORDS.items():
            for k in re.finditer(pat, t, re.I):
                d = max(k.start() - m.end(), m.start() - k.end(), 0)
                if d <= window and d < nearest.get(g, window + 1):
                    nearest[g] = d
        if group in nearest and nearest[group] == min(nearest.values()):
            return True
    return False


def _monthly_figure(value: Any, text: str) -> bool:
    """The number is a monthly amount: "$800 a month", "600/mth", "monthly payment of $500"."""
    return any(re.match(r"\s*(?:/\s*|per\s+|a\s+)(?:month|mth)\b|\s*monthly\b", text[m.end():m.end() + 20], re.I)
               or re.search(r"\bmonthly\b[^.;,]{0,20}$", text[max(0, m.start() - 30):m.start()], re.I)
               for m in _matches_of(value, text))


def _percent_figure(value: Any, text: str) -> bool:
    return any(re.match(r"\s*%", text[m.end():m.end() + 3]) for m in _matches_of(value, text))


_LEASE_FIG = re.compile(r"(\d{2})\s*(?:-|\s)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:remaining\s+)?lease"
                        r"|\blease\s*(?:of|:|remaining|left|is)?\s*(?:at least\s*)?(\d{2})\s*\+?\s*(?:years?|yrs?)"
                        r"|(\d{2})\s*(?:years?|yrs?)\s*(?:of\s+lease\s+)?(?:left|remaining)", re.I)


# A fact is stated only in a plain statement: not a question ("Can you confirm if we are eligible?"), not a request
# about it ("please check whether ..."), and not negated just before the fact words ("we haven't confirmed", "this isn't
# our first home"). A "please" elsewhere in the sentence ("..., so please work out our grants") is fine.
_QUESTION_OR_REQUEST = re.compile(r"\b(?:can|could|would|will) you\b|\b(?:please\s+)?(?:confirm|check|tell me|let me know)\s+"
                                  r"(?:if|whether)\b|\bwhether\b|\bwhat if\b", re.I)
_NEGATED_BEFORE = re.compile(r"(?:\bnot|n't|\bno longer|\bnever|\bneither|\bnor|\byet to|\bcannot)\b[^.;,]{0,25}$", re.I)


def _stated_in_one_sentence(text: str, *cues: str) -> bool:
    """Every cue appears, un-negated, in one plain statement."""
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text or ""):
        if sentence.strip().endswith("?") or _QUESTION_OR_REQUEST.search(sentence):
            continue
        found = True
        for cue in cues:
            m = re.search(cue, sentence, re.I)
            if not m or (_NEGATED_BEFORE.search(sentence[:m.start()]) and not m.group(0).lower().startswith("never")):
                found = False
                break
        if found:
            return True
    return False


_JOB_LOSS = re.compile(r"\b(?:lost|quit|left|between|changed)\s+(?:my |our |his |her |a |the )?jobs?\b|\bunemploy\w*|"
                       r"\blaid off\b|\bretrench\w*|\bwithout a job\b|\bquit\b|\bresign\w*", re.I)
_NOT_CONTINUOUS = re.compile(r"\b(?:less than|under a year|fewer than|stopped|freelanc\w*|started|part[- ]time|gap|between jobs|"
                             r"break|sabbatical)\b|\btook\b[^.;]{0,20}\boff\b", re.I)
# the buyers near or with their parents (the PHG cases): "we want to buy near my parents", "my parents live in Tampines,
# we want to buy near them", "my parents will live with us", "parents moving in with us"; not "our parents live nearby
# our office", "we don't live near our parents" or "my parents live with my brother"
_PARENTS_NEAR = (r"\b(?:we|i|us|me)\b[^.;]{0,40}\b(?:near|nearby|close to|within 4\s?km of)\s+(?:my|our|his|her)?\s*(?:parents|them)\b"
                 r"|\bparents?\b[^.;]{0,40}\b(?:near|nearby|close to|within 4\s?km of)\s+(?:us|me|our (?:home|flat|place))\b"
                 r"|\bparents?\b[^.;]{0,40}\bsame (?:town|estate|block|street)\b")
_PARENTS_WITH = (r"\bparents?\b[^.;]{0,40}\b(?:live|living|stay|staying|move in|moving in)\s+with\s+(?:us|me)\b"
                 r"|\b(?:live|living|stay|staying)\s+with\s+(?:my|our)\s+parents\b")
_SOMEONE_ELSE = re.compile(r"\b(?:brothers?|sisters?|siblings?|relatives?|far|away)\b", re.I)


def _parents_clause(pattern: str, text: str) -> bool:
    """A sentence (plans kept: "will live with us") that states the pattern un-negated and about the buyers."""
    for sentence, _ in _sentences_and_clauses(statements(text, keep=("plan",))):
        m = re.search(pattern, sentence, re.I)
        if m and not _SOMEONE_ELSE.search(sentence) and not _NEGATED_BEFORE.search(sentence[:m.start()]) \
                and not re.search(r"\bnot\b|n't|\bnever\b", m.group(0), re.I):
            return True
    return False


def fact_is_grounded(field: str, value: Any, text: str) -> bool:
    """True when the user's words support this calculator fact as read by the model, by the same evidence code
    accepts: a money amount must be paired with its own fact word; an age must be one of the buyers' ages; yes/no
    and choice facts must be stated plainly (not asked, hoped for, planned, in the past or negated). Derived values
    are never taken from the model."""
    t = text or ""
    s, own = statements(t), statements(t, own=True)
    stated = lambda *cues: _stated_in_one_sentence(s, *cues)  # noqa: E731
    if field in ("average_applicant_age", "ltv_ratio"):
        return False                                   # worked out by code / the calculators
    corrected = {int(x) for x in re.findall(r"\bnot\s+(\d{2})\b", t, re.I)}      # "sorry, wife is 31 not 30"
    if field in ("youngest_buyer_age", "oldest_buyer_age"):
        return _isint(value) and int(value) not in corrected and int(value) in (_ages_anywhere(own) or _bare_ages(own))
    if field == "youngest_core_member_age":
        return _isint(value) and int(value) not in corrected and int(value) in (_ages_anywhere(own) or _bare_ages(own)) \
            and no_children(t)
    if field in MONEY_FIELDS:
        bound, ambiguous = _bind_amounts(own, is_couple(t))
        if field in ambiguous:
            return False
        if field in bound:
            return abs(bound[field] - float(value)) <= max(0.5, 0.005 * abs(float(value)))
        return any(fields == {field} and abs(v - float(value)) <= max(0.5, 0.005 * abs(float(value)))
                   for v, fields, split, _ in _money_claims(own, is_couple(t)) if not split)
    if field == "monthly_debt_obligations":
        if value == 0:
            return no_debts_stated(s)
        # an existing debt's monthly payment, not what the buyer can afford, rent or gives as an allowance
        return _number_near(value, s, "debt") and _monthly_figure(value, s) and any(
            _DEBT_SOME.search(c) and not re.search(r"\b(?:afford\w*|rent\w*|allowance|budget\w*|spend\w*|can|willing)\b", c, re.I)
            for c, _ in _sentences_and_clauses(s) if _matches_of(value, c))
    if field == "total_grants":
        return (value == 0 and bool(re.search(r"\bno (?:housing )?grants\b", s, re.I))) or _number_near(value, s, "grant")
    if field == "additional_buyer_stamp_duty":
        return (value == 0 and bool(re.search(r"\bno ABSD\b|\bno additional buyer'?s stamp duty\b", s, re.I))) \
            or _number_near(value, s, "absd")
    if field == "interest_rate":
        return _number_near(value, s, "rate", window=15) and _percent_figure(value, s)
    if field == "tenure_years":
        return bool(_stated_match(r"tenure of (\d{1,2}) years|(\d{1,2})[- ]year (?:loan )?tenure", s)) and \
            _number_near(value, s, "tenure", window=20)
    if field == "remaining_lease_years":
        return _isint(value) and any(int(next(g for g in m.groups() if g)) == int(float(value))
                                     for pat in (_LEASE_FIG, re.compile(_CALC_LEASE, re.I)) for m in pat.finditer(s))
    if field == "legal_and_valuation_fees":
        return _number_near(value, s, "fee")
    if field == "citizenship_pair":
        return value == citizenship_stated(t)
    if field == "first_timer_status":
        return value == first_timer_stated(t)
    if field == "eligibility_confirmed":
        return value is True and stated(r"\beligib", r"\bconfirmed\b")
    if field == "loan_eligibility_confirmed":
        return value is True and stated(r"\bloan\b", r"\beligib|\bapprov", r"\bconfirmed\b|\bapproved\b")
    if field == "continuous_employment_12_months":
        return value is True and not _JOB_LOSS.search(t) and not _NOT_CONTINUOUS.search(s) and \
            stated(r"\b(?:employ\w*|working|work|job)\b", r"\b(?:12|twelve)\s*(?:months?|mths?)\b|\b(?:a|one|1|past) year\b")
    if field in ("proximity", "near_parents_4km"):
        none = bool(re.search(r"\bno proximity\b|\bno PHG\b|\bproximity (?:housing )?grant (?:does not|doesn't|won't) apply"
                              r"|\bnot (?:living )?near (?:my|our) parents\b", s, re.I))
        if value in ("none", False):
            return none
        if value == "with":
            return _parents_clause(_PARENTS_WITH, t)
        return value in ("near", True) and (_parents_clause(_PARENTS_NEAR, t) or _parents_clause(_PARENTS_WITH, t))
    if field == "proximity_eligibility_confirmed":
        return value is True and stated(r"\bproximity\b", r"\bconfirmed\b")
    return False


# A two-digit number that can only be an age: not after a block, street, exit or floor ("Blk 45", "Tampines St 45"),
# not before an area or a floor count ("90 sqm"), and not next to someone else's word ("my mother, 68", "68 year old
# mother", "70 yo dad")
_BARE_AGE = re.compile(r"(?<![\d$.,/#])(\d{2})(?!\d|[,.]\d|st\b|nd\b|rd\b|th\b|\s*-?\s*(?:years?|yrs?)\b(?!\s+old)|\s*(?:k\b|%|m\b|"
                       r"km\b|mins?\b|minutes|months?|mths?|sq\s?m|sqm|square|storeys?|stor(?:e)?ys?|stories|floors?|blocks?\b|"
                       r"percent|per cent|-?\s*rooms?\b|rm\b|\+|or more|or above|or longer))", re.I)
_OTHER_WORD = (r"\b(?:parents?|mum|mom|mother|dad|father|in-laws?|grand\w*|kids?|child(?:ren)?|sons?|daughters?|brothers?|"
               r"sisters?|baby)\b")
_PLACE_BEFORE = re.compile(r"\b(?:blk|block|st|street|ave|avenue|rd|road|exit|floor|level|lvl|no|unit|bus|service|stop|ring|"
                           r"dr|drive|cres|crescent|lorong|lor|jalan|jln|area|page|room|rooms|sector|district|postal)\.?\s*$", re.I)


def _bare_ages(text: str) -> List[int]:
    """Two-digit numbers that can only be an age ("i'm single, a singapore citizen, 38, first timer"): not a lease,
    duration, distance, percentage, amount, address or room count, and not someone else's. Used only to accept the
    model's reading of an age when buyer_ages finds none."""
    out = []
    t = text or ""
    for m in _BARE_AGE.finditer(t):
        before = re.split(r"[.,;!?\n]", t[max(0, m.start() - 40):m.start()])[-1]
        if re.search(r"\b(?:lease|remaining|tenure|loan)\b[^.;,]{0,25}$", before, re.I) or _PLACE_BEFORE.search(before) \
                or re.search(r"\b(?:flats?|blocks?|buildings?|units?|propert(?:y|ies)|estates?|houses?|condos?|it)\b[^.;,]{0,20}$",
                             before, re.I):
            continue                                            # "lease of 65 years", "Blk 45", "a flat less than 30 years old"
        someone_else = re.search(_OTHER_WORD + r"[\s,:']*(?:s\s+)?(?:is\s+|aged\s+|who is\s+)?$", t[max(0, m.start() - 30):m.start()], re.I) \
            or re.match(r"\s*-?\s*(?:years?|yrs?|yo|y/o)?(?:[- ]old)?\s*(?:\w+\s+)?" + _OTHER_WORD, t[m.end():m.end() + 30], re.I)
        if 18 <= int(m.group(1)) <= 99 and not _NOT_BUYER.search(before) and not someone_else:
            out.append(int(m.group(1)))
    return out


def _ages_anywhere(text: str) -> List[int]:
    """Buyer ages stated in any sentence, so a model reading a later correction ("my husband is 34, not 32") can be
    grounded even though the first sentence gave both ages."""
    out: List[int] = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text or ""):
        out += [a for a in buyer_ages(sentence, couple=False) if a not in out]
    return out


def _isint(v: Any) -> bool:
    try:
        return float(v) == int(float(v))
    except (TypeError, ValueError):
        return False


def stated_across_turns(turns: List[str], unsure: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """The code's reading of every user message so far. The messages are read together (a fact can depend on another
    message: ages in a reply, "no kids" in the first question), then each later message is read on its own and its
    facts win, so "sorry, cash is 60k" corrects the earlier figure. A later message that changes a fact code cannot
    read in full ("sorry, I'm 33 not 32" names one of two ages; "that 40k is CPF, not cash"; "we just had a baby";
    "my wife lost her job") removes the old value, so the assistant asks again instead of keeping a stale fact. The
    removed fields are added to `unsure`, so the model's reading of them is not used either."""
    out = {sec: dict(vals) for sec, vals in stated_financial_inputs("\n".join(turns)).items()}
    drop: set = set()
    for i, turn in enumerate(turns[1:], 1):
        reading = stated_financial_inputs(turn, context="\n".join(turns[:i]))
        new = {k for vals in reading.values() for k in vals}
        own = statements(turn, own=True)
        if (buyer_ages(own, couple=False) or re.search(r"\b(?:not|isn't|wasn't)\s+\d{2}\b(?!\s*[k%m,])", own, re.I)) \
                and "youngest_buyer_age" not in new:
            drop |= AGE_FIELDS
        if _KID_WORDS.search(re.sub(_NO_KIDS, " ", statements(turn), flags=re.I)):
            drop.add("youngest_core_member_age")
        for field in MONEY_FIELDS:
            if field not in new and re.search(_FACT_WORDS[field], turn, re.I) and \
                    any(float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1) >= 100 for m in _AMOUNT_TOKEN.finditer(turn)):
                drop.add(field)
        if _JOB_LOSS.search(turn):
            drop |= {"continuous_employment_12_months", "monthly_income"}
        if "monthly_debt_obligations" not in new and not no_debts_stated(statements(turn)) and re.search(
                r"\b(?:car|study|student|personal|renovation|reno|education)\s+loans?\b|\bcredit[- ]cards?\b|\bowe\w*\b|\bdebts?\b",
                statements(turn), re.I):
            drop.add("monthly_debt_obligations")                         # "oh we just took a car loan, 800 a month"
        if any(_PARTNER_WORD.search(turn[max(0, m.start() - 25):m.start()]) for m in _AMOUNT_TOKEN.finditer(turn)
               if float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1) >= 100) and "monthly_income" not in new:
            drop.add("monthly_income")                                   # "plus my husband's 4k"
        drop -= new
        for sec, vals in reading.items():
            out.setdefault(sec, {}).update(vals)
    for vals in out.values():
        for k in drop:
            vals.pop(k, None)
    g = out.get("grants") or {}
    if "youngest_core_member_age" in g and "youngest_buyer_age" in g:
        g["youngest_core_member_age"] = g["youngest_buyer_age"]       # follow a corrected age
    if unsure is not None:
        unsure |= drop
    return {sec: vals for sec, vals in out.items() if vals}


def trusted_financial_inputs(model: Optional[Dict[str, Any]], stated: Dict[str, Dict[str, Any]],
                             text: str = "", unsure: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """The calculator facts to use: the model's reading kept only where the user's words support it
    (fact_is_grounded), then the code's own reading on top (stated_financial_inputs; it wins where both read a
    fact). A grounded fact goes to every calculator that takes it; the average age is worked out from the ages, and a
    couple's ages are taken from the model only when both are grounded. Fields in `unsure` (changed later in the
    conversation in a way code could not read) are left out, so the assistant asks. A section exists only when the
    user stated one of its facts; the upfront section only when code found a price. With no text (or no model) this
    is the code's reading alone."""
    from agents.schemas import TOOL_INPUT_MODEL
    unsure = unsure or set()
    flat: Dict[str, Any] = {}
    for sec in ("grants", "affordability"):
        for k, v in ((model or {}).get(sec) or {}).items():
            if v is None or k in flat or k in unsure:
                continue
            if k not in FACT_FIELDS or fact_is_grounded(k, v, text):
                flat[k] = v
    if is_couple(text) and not {"youngest_buyer_age", "oldest_buyer_age"} <= set(flat):
        for k in AGE_FIELDS:
            flat.pop(k, None)                          # one grounded age of two is not the couple's ages
    stated_flat = {k: v for sec in ("grants", "affordability") for k, v in (stated.get(sec) or {}).items() if k not in unsure}
    if "youngest_buyer_age" in stated_flat:
        for k in AGE_FIELDS:
            flat.pop(k, None)                          # the code's ages replace the model's as a set
    flat.update(stated_flat)
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
    model_reading: Optional[Dict[str, Any]] = None
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
            model_reading = extracted.model_dump()
            if extracted.financial_inputs is not None:
                model_reading["financial_inputs"] = extracted.financial_inputs.model_dump(exclude_unset=True)
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
    turns = [m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
    users = "\n".join(turns)
    unsure: set = set()
    stated = stated_across_turns(turns or [query], unsure)
    constraints["financial_inputs"] = trusted_financial_inputs(constraints.get("financial_inputs"), stated,
                                                               users or query, unsure)
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
    # a fact a later message changed in a way code could not read is asked again, not carried over from the last turn
    for vals in (constraints.get("financial_inputs") or {}).values():
        for k in unsure:
            vals.pop(k, None)
    # read by code from every user message so far, so an ask in the first turn still counts after a follow-up reply
    constraints["requested_tools"] = requested_tools(users or query)
    comparables = "query_comparable_resale_transactions"
    if comparables not in constraints["requested_tools"] and coordinates_stated(constraints.get("comparable_resale"), users or query):
        constraints["requested_tools"].append(comparables)
    return {"user_constraints": constraints, "errors": errors, "model_constraints": model_reading}


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
                 "for grants, affordability (MSR/LTV) and, when the user asks for upfront costs, upfront outlay (BSD). "
                 "Never invent eligibility confirmations "
                 "or infer citizenship/first-timer status from legacy citizenship_scheme. Pass grant results forward once; "
                 "real affordability already includes grants. Supply actual valuation, loan and quoted fees for upfront "
                 "costs. Unknown funds/debts/grants are not zero. Report missing inputs and unsupported scenarios, "
                 "including cost exclusions and cash shortfalls. Reply with a short summary when done.",
    "policy_rag": "You are the Regulatory Policy RAG agent. Call retrieve_hdb_policy_knowledge with the user's regulatory "
                  "question, then answer ONLY from the returned passages with inline [source] citations.",
}


# Tools that run only when the user asks for them in so many words. The plans used to cost the budget for stamp duty
# whenever a budget or income was given, and to pull comparable sales whenever comparable_resale was filled, so a
# search-and-budget question got an upfront-costs section nobody asked for (and lost Tool F1 precision: no Category E
# question expects either call). "Cash savings" or "resale transaction" alone is not an ask. The patterns are read one
# sentence at a time (requested_tools), so ".{0,60}" never reaches into the next sentence.
_ASKS_UPFRONT = re.compile(
    r"\bup[- ]?front\b|(?<!monthly )\boutlays?\b|\bdown[- ]?payments?\b|\bstamp dut(?:y|ies)\b|\b(?:BSD|ABSD)\b"
    # "how much cash do we need", "how much would we need in cash and CPF", "how much do I need to pay in cash"
    r"|\bhow much\s+(?:cash|CPF)\b(?!\s+savings).{0,40}?\b(?:need|needed|required|pay|put down|set aside)\b"
    r"|\bhow much\b.{0,40}?\b(?:need|pay|put down|fork out|set aside|come up with)\b.{0,30}?\b(?:cash|CPF)\b(?!\s+savings)"
    # "cash needed", "the total cash and CPF needed", "the initial costs", "the costs of buying"
    r"|\b(?:cash|CPF)(?:\s+(?:and|or|&|\+|plus)\s+(?:cash|CPF))?\s+(?:[\w-]+\s+){0,2}?(?:needed|required|outlay|up[- ]?front)\b"
    r"|\b(?:initial|one[- ]?(?:time|off)|closing|completion|buying|purchase)\s+(?:[\w-]+\s+)?(?:costs?|fees|expenses|outlay)\b"
    r"|\bcosts?\s+(?:of|for|involved in)\s+(?:the\s+)?(?:buying|purchas\w*|completing|completion)\b"
    r"|" + _PURCHASE_PRICE, re.I)
_NO_BUYER_WORDS = r"(?:(?!\b(?:I|I'm|I'll|I'd|we|we're|we'll|we'd|you|you're|to)\b).)"   # words with no buyer as the subject
_ASKS_COMPARABLES = re.compile(
    r"\bcomparables?\b|\bcomps\b|\bsimilar\s+(?:[\w-]+\s+){0,2}?(?:flats?|units?|homes?|ones)\b"
    r"|\brecent(?:ly)?\s+(?:[\w-]+\s+){0,2}?(?:transactions?|sales?|sold|transacted)\b"
    r"|\btransactions?\s+(?:[\w-]+\s+){0,3}?(?:recently|lately)\b|\btransactions?\s+(?:around|near|nearby|close to|within)\b"
    r"|\bcompar\w*\s+(?:[\w-]+\s+){0,2}?(?:sales|transactions)\b"
    r"|\b(?:nearby|neighbou?ring|surrounding)\s+(?:[\w-]+\s+){0,2}?(?:sales|transactions|resale prices?)\b"
    # sales or prices at a place given by a distance or coordinates: "resale prices within 500m of latitude 1.33"
    r"|\b(?:sales|prices?)\s+(?:within\s+\d+\s*k?m\b|(?:around|near|close to|at)\s+(?:lat\w*|-?\d{1,3}\.\d{3,}))"
    # what flats sold for: "how much did 4-room flats near ... sell for", "what are flats around ... going for",
    # "what prices did nearby flats fetch"; not "which flats should we go for" or "flats that go for under $600k"
    r"|\b(?:flats?|units?|homes?|ones|they|blocks?|HDBs?|apartments?)\b" + _NO_BUYER_WORDS + r"{0,60}?"
    r"\b(?:(?:sell|sold|selling|go|going|gone|went|transact(?:ed)?)\s+for\b(?!\s+(?:under|below|less|at most|no more|up to)\b|\s*\$)"
    r"|fetch(?:ed|es|ing)?\b)"
    r"|\bhow much\s+(?:did|do|does|have|has|were|was|are|is)\b" + _NO_BUYER_WORDS + r"{0,60}?"
    r"\b(?:sell|sold|selling|go|going|gone|went|transact(?:ed)?)\s+for\b"
    r"|\bwhat\s+(?:did\s+)?(?:people|buyers|others)\s+(?:pay|paid)\b", re.I)
OPTIONAL_TOOLS = {"calculate_bsd_and_upfront_outlay": _ASKS_UPFRONT,
                  "query_comparable_resale_transactions": _ASKS_COMPARABLES}
# What turns a match into a non-ask: the user turning it down just before ("we don't care about stamp duty"), a
# question about the rules ("what are the ABSD rules for PR couples": the policy search answers that, not the
# calculator), and a "how much ... need ... CPF" about the loan or income ("how much would I pay monthly from CPF"),
# which affordability covers.
_TURNED_DOWN = re.compile(r"\b(?:(?:don'?t|do not|doesn'?t|does not)(?!\s+forget\b)|no need|not (?:interested|asking|"
                          r"worried|bothered|concerned)|never ?mind|skip|ignore|leave out|(?<!n't )(?<!not )forget)\b"
                          r"\W+(?:[\w'-]+\W+){0,3}$", re.I)   # "don't forget the stamp duty" is still an ask
_DUTY_RULES = re.compile(r"\b(?:rules?|polic(?:y|ies)|regulations?|regime|remissions?|exemptions?|appl(?:y|ies)|works?)\b",
                         re.I)
_LOAN_WORDS = re.compile(r"\b(?:month\w*|instal\w*|loan|mortgage|earn|income|salary)\b", re.I)


def _is_ask(name: str, sentence: str, m: re.Match) -> bool:
    if _TURNED_DOWN.search(sentence[:m.start()].rsplit(",", 1)[-1]):
        return False
    if name != "calculate_bsd_and_upfront_outlay":
        return True
    said = m.group(0)
    if re.fullmatch(r"stamp dut(?:y|ies)|A?BSD", said, re.I):
        # rule words within three words either side: "the ABSD rules", "rules for stamp duty", "does BSD apply"
        around = " ".join(sentence[:m.start()].split()[-3:] + sentence[m.end():].split()[:3])
        return not _DUTY_RULES.search(around)
    return not (said.lower().startswith("how much") and _LOAN_WORDS.search(said))


def requested_tools(text: str) -> List[str]:
    """The optional tools (OPTIONAL_TOOLS) the user's words ask for: upfront costs (upfront, stamp duty or BSD, cash
    needed, a down payment, the initial costs, or a price they are buying for) and comparable sales (comparables,
    recent or nearby transactions, similar flats, what flats sold for). Read sentence by sentence; a match the user
    turns down ("we don't care about stamp duty") or that asks about the rules ("the ABSD rules") is not an ask."""
    sentences = re.split(r"(?<=[.?!;])\s+|\n+", text or "")   # "1.3329" has no space after its point, so stays whole
    return [name for name, asks in OPTIONAL_TOOLS.items()
            if any(_is_ask(name, s, m) for s in sentences for m in asks.finditer(s))]


def coordinates_stated(where: Optional[Dict[str, Any]], text: str) -> bool:
    """Whether comparable_resale's latitude and longitude are both numbers the user typed. Comparable sales are the
    only tool that takes coordinates, so a user who gives them is asking for those sales whatever words they use. The
    model reads them; code checks they are the user's own numbers."""
    if not where or where.get("latitude") is None or where.get("longitude") is None:
        return False
    typed = [float(x) for x in re.findall(r"-?\d{1,3}\.\d{2,}", text or "")]
    return all(any(abs(x - float(where[k])) < 1e-4 for x in typed) for k in ("latitude", "longitude"))


def asked_for(name: str, c: Dict[str, Any], query: str = "") -> bool:
    """Whether a tool may run for this question. Only the optional tools need an ask: in the user's messages as the
    extraction step recorded it (c["requested_tools"]), or in the query itself where no extraction ran. A purchase
    price code read from the user's words (financial_inputs.upfront) is an ask for upfront costs."""
    if name not in OPTIONAL_TOOLS:
        return True
    if name == "calculate_bsd_and_upfront_outlay" and (c.get("financial_inputs") or {}).get("upfront"):
        return True
    asked = c.get("requested_tools")
    return name in (asked if asked is not None else requested_tools(query))


CODE_ONLY_KEYS = ("requested_tools",)


def constraints_for_model(c: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The constraints as a model is shown them. requested_tools is code's reading for code and is never shown: it
    names the optional tools, so it would steer which tools A's and B's models pick (their unasked calls must still
    count against Tool F1), and live prompts stay as they were before it existed."""
    return None if c is None else {k: v for k, v in c.items() if k not in CODE_ONLY_KEYS}


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
        if c.get("comparable_resale") and asked_for("query_comparable_resale_transactions", c, query):
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
        price = ((c.get("financial_inputs") or {}).get("upfront") or {}).get("purchase_price") or c.get("max_budget_sgd")
        if (price or income) and asked_for("calculate_bsd_and_upfront_outlay", c, query):
            plan.append(("calculate_bsd_and_upfront_outlay", dict(purchase_price=price or 0.0, loan_type="hdb", total_grants=0.0)))
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
                               HumanMessage(content=f"User query: {query}\n\nExtracted constraints (JSON): {json.dumps(constraints_for_model(c))}")]
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
                    guarded = guard_financial_args(name, args, c, outputs, runners)   # may run the grant calculator (code)
                    runners[name].source = "model"
                    try:
                        result = runners[name](**guarded)
                    finally:
                        runners[name].source = "code"
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
    # only the trusted financial_inputs reach the calculators: the top-level household_income, cash and CPF fields
    # hold the model's ungrounded reading
    flat_types = constraints.get("flat_types") or []
    if len(flat_types) == 1:
        grant_args.setdefault("flat_type", flat_types[0])
    # The legacy citizenship_scheme extractor infers citizenship/first-timer status;
    # only explicitly supplied financial facts may be used by real calculators.
    if re.search(r"\bbto\b", query, re.I) and not re.search(r"resale", query, re.I):
        grant_args["flat_category"] = "bto"
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
    if upfront_name in runners and asked_for(upfront_name, constraints, query):   # never cost the budget unasked
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
    block out (E1-04); the market agent pulled comparable sales at coordinates it made up. The financial agent's
    upfront costs count only when the user asked for them (asked_for). The calls stay in the tool-call log, so
    tool-selection metrics still count them."""
    if agent == "financial":
        return {name: out for name, out in outputs.items() if asked_for(name, c, query)}
    if agent not in ("market_sql", "spatial"):
        return outputs
    asked = {name for name, _ in deterministic_plan(agent, c, query)}
    return {name: out for name, out in outputs.items() if name in asked or name == "sync_interactive_map_ui"}


def make_specialist_node(agent: str) -> Callable[[HDBCompassState], Dict[str, Any]]:
    def node(state: HDBCompassState) -> Dict[str, Any]:
        c = state.get("user_constraints", {})
        query = last_user_query(state)
        # the optional tools (upfront costs, comparable sales) are offered to the model only when the user asked for
        # them; Variants A and B keep the full list, so their models' unasked calls still show in Tool F1
        tools = [name for name in tools_for_agent(agent) if asked_for(name, c, query)]
        outputs, log, errors, final = run_react(agent, tools, AGENT_SYSTEM[agent], query, c)
        update = slot_outputs(requested_outputs(agent, outputs, c, query), c.get("towns") or [])
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


LISTED_BLOCKS = 20   # matching blocks written out in full; the rest are counted (all are on the map)
AMENITY_LABELS = {"mrt": "an MRT station", "hawker": "a hawker centre", "bus_stop": "a bus stop",
                  "supermarket": "a supermarket", "park": "a park", "school": "the school", "clinic": "a clinic"}


def plain_relaxation(constraint: str, original: Any, relaxed: Any) -> Tuple[str, str, str]:
    """A relaxation in words a buyer reads: ("distance to an MRT station", "400 m", "550 m"), not
    ("amenity_radii_meters.mrt", 400, 550)."""
    if constraint == "max_budget_sgd":
        return "budget", f"${float(original):,.0f}", f"${float(relaxed):,.0f}"
    if constraint == "min_remaining_lease_years":
        return "minimum remaining lease", f"{float(original):g} years", f"{float(relaxed):g} years"
    if constraint.startswith("amenity_radii_meters."):
        what = AMENITY_LABELS.get(constraint.split(".", 1)[1], constraint.split(".", 1)[1].replace("_", " "))
        return f"distance to {what}", f"{float(original):g} m", f"{float(relaxed):g} m"
    return constraint.replace("_", " "), str(original), str(relaxed)


# Member 3's policy search returns raw markdown chunks: a "[Doc > Section > Sub]" breadcrumb line, bullets, table rows
# and rules, '#' headings, and some chunks that are only a header ("[...]\n---"). The facts section shows each passage
# as its section title, the few lines that best answer the question, and the source link. Every line shown is a line
# of the passage with its list marker and bold marks removed; a table row reads "A: B" (with the passage's own column
# names) and a line under a bare lead-in carries that lead-in. Nothing else is added and nothing is reworded.
POLICY_PASSAGES_SHOWN = 3
POLICY_LINES_SHOWN = 4      # at most; at least 2 when the passage has them
POLICY_LINE_CHARS = 240     # a longer line keeps its whole sentences that fit; a sentence is never cut…
POLICY_LINE_MAX = 400       # …unless it alone is longer than this (no real passage line is)
_STOP_WORDS = frozenset((
    "a about after all also am an and any are as at be been before but by can could did do does for from get got had has "
    "have how i if in into is it its me more must my no not now of on or our out so than that the their them then there "
    "these they this those to under up us was we were what when where whether which who why will with would you your "
    "yours exact exactly key differences difference compare comparing vs versus tell know please").split())
_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
_TABLE_RULE = re.compile(r":?-{3,}:?")
_LIST_MARKER = re.compile(r"^(?:[-*•]|\d{1,2}[.)])\s+")


def _content_words(text: str) -> set:
    """Lower-case words and numbers that carry meaning, with simple endings folded so "aged" meets "age", "flats"
    meets "flat" and "renting" meets "rent"."""
    words = [w for w in re.findall(r"[a-z]+|\d+", (text or "").lower())
             if w not in _STOP_WORDS and (len(w) > 2 or w.isdigit())]
    folded = set()
    for w in words:
        if len(w) > 4 and not w.endswith("ss"):
            w = re.sub(r"(?:ing|ed|es|s)$", "", w)
        folded.add(w.rstrip("e") if len(w) > 2 and not w.isdigit() else w)
    return folded


def _pick_lines(lines: List[str], question: str, pool: List[str]) -> List[int]:
    """Indexes of the 2-4 lines that best answer the question, in the passage's order. A question word counts more
    the fewer lines of the pool (every line that could be shown in this answer) contain it, so "income" outweighs
    "flat", and a line's score is divided by the root of its length so a short exact line ("BTO Income Ceiling: ...
    $7,000/month") is not beaten by a long one that mentions everything. Each next line is the one adding the most
    question words not yet covered, so "minimum age and income ceiling" gets a line for each; once nothing new is
    added, the pick stops (topped up to 2 lines by plain score). Ties keep the passage's order."""
    words = _content_words(question)
    pool_words = [_content_words(line) for line in pool]
    weight = {w: 1.0 / max(1, sum(w in pw for pw in pool_words)) for w in words}
    line_words = [_content_words(line) & words for line in lines]
    size = [max(1.0, len(_content_words(line))) ** 0.5 for line in lines]

    def score(i: int, covered: set) -> float:
        return sum(weight[w] for w in line_words[i] - covered) / size[i]

    chosen: List[int] = []
    covered: set = set()
    while len(chosen) < POLICY_LINES_SHOWN:
        rest = [i for i in range(len(lines)) if i not in chosen]
        best = max(rest, key=lambda i: (score(i, covered), -i), default=None)
        if best is None or score(best, covered) <= 0:
            break
        chosen.append(best)
        covered |= line_words[best]
    by_score = sorted((i for i in range(len(lines)) if i not in chosen), key=lambda i: (-score(i, set()), i))
    chosen += by_score[:max(0, 2 - len(chosen))]
    return sorted(chosen)


def _policy_title(p: Dict[str, Any]) -> Tuple[str, str]:
    """(title, parent): the last breadcrumb part and the part above it (when that is not the document title),
    without their numbering. "... > 5. Deep Dive: Standard Flats Regulations > 5.1 Baseline Parameters" gives
    ("Baseline Parameters", "Deep Dive: Standard Flats Regulations")."""
    crumb = p.get("section") or ""
    first = (p.get("text") or "").split("\n", 1)[0].strip()
    if not crumb and first.startswith("[") and first.endswith("]"):
        crumb = first[1:-1]
    parts = [_SECTION_NUMBER.sub("", s.strip()) for s in crumb.split(" > ") if s.strip()]
    return (parts[-1] if parts else ""), (parts[-2] if len(parts) > 2 else "")


def _table_row(cells: List[str], header: Optional[List[str]]) -> str:
    """A table row as text: "Up to $1,500: $120,000", or with more columns
    "Minimum Occupation Period (MOP) — Standard Flats: 5 years; Plus Flats: 10 years"."""
    if len(cells) == 2:
        return f"{cells[0]}: {cells[1]}"
    if header and len(header) == len(cells):
        return f"{cells[0]} — " + "; ".join(f"{h}: {v}" for h, v in zip(header[1:], cells[1:]) if v)
    return f"{cells[0]}: " + "; ".join(v for v in cells[1:] if v)


def _policy_lines(text: str) -> List[str]:
    """The readable lines of a passage: breadcrumb, rules, separator rows and headings dropped; list markers and bold
    marks removed; table rows read as text. A bare lead-in ("Grant Amounts:") is not a line of its own but goes in
    front of the lines indented under it ("Grant Amounts — 5-room Resale Flat: $25,000")."""
    raw = (text or "").split("\n")
    if raw and raw[0].strip().startswith("[") and raw[0].strip().endswith("]"):
        raw = raw[1:]
    rows = [r.strip() for r in raw]
    indent = [len(r) - len(r.lstrip()) for r in raw]
    cells_of = [[c.strip().replace("**", "") for c in r.strip("|").split("|")] if r.startswith("|") else None for r in rows]
    is_rule = [bool(c) and all(_TABLE_RULE.fullmatch(x) for x in c) for c in cells_of]
    out: List[str] = []
    header: Optional[List[str]] = None
    lead: Optional[Tuple[int, str]] = None   # (indent, words) of the bare lead-in the next lines sit under
    for i, row in enumerate(rows):
        cells = cells_of[i]
        if cells is not None:
            if is_rule[i]:
                continue
            if i + 1 < len(rows) and is_rule[i + 1]:
                header = cells              # the row above the rule names the columns
                continue
            line = _table_row(cells, header)
        else:
            header = None
            line = _LIST_MARKER.sub("", row.replace("**", "")).replace("\\times", "×").strip()
            if not line or line.startswith(("#", "$$")) or not re.search(r"[A-Za-z0-9]", line):
                continue                    # headings, LaTeX formula lines and separators
            if lead and indent[i] <= lead[0]:
                lead = None
            if line.endswith(":") and len(line.split()) <= 8:
                lead = (indent[i], line.rstrip(":").strip())
                continue
            if lead:
                line = f"{lead[1]} — {line}"
        if line not in out:
            out.append(line)
    return out


def _trim_line(line: str, question: str = "", limit: int = POLICY_LINE_CHARS) -> str:
    """A long line cut to whole sentences (a table row's cells, which end in ";", count as sentences), never inside
    one, so a figure keeps its unit and a rule keeps its condition ("must wait 15 months from ...", "ALWAYS PERMITTED,
    even if the quota ... is completely full"). The first sentence is always kept and the ones after it while the line
    fits in `limit`; a later sentence holding a question word the kept ones lack is kept too while the line fits in
    POLICY_LINE_MAX ("Wait-Out After Selling Private Property: ..." for "How long must I wait"). Sentences left out
    between or after the kept ones show as "…". Only one sentence longer than POLICY_LINE_MAX on its own is cut inside:
    at a comma or dash, else at a word that is not a number."""
    if len(line) <= limit:
        return line
    parts = _split_opening(line)
    if len(parts[0]) > POLICY_LINE_MAX:
        return _cut_sentence(parts[0])
    keep = [0]
    while len(keep) < len(parts) and len(_join_kept(parts, keep + [len(keep)])) <= limit:
        keep.append(len(keep))
    words = _content_words(question)
    covered = set().union(*(_content_words(parts[i]) for i in keep)) & words
    for i in range(keep[-1] + 1, len(parts)):
        adds = (_content_words(parts[i]) & words) - covered
        if adds and len(_join_kept(parts, keep + [i])) <= POLICY_LINE_MAX:
            keep.append(i)
            covered |= adds
    return _join_kept(parts, keep)


def _join_kept(parts: List[str], keep: List[int]) -> str:
    text = parts[keep[0]]
    for prev, i in zip(keep, keep[1:]):
        text += (" " if i == prev + 1 else " … ") + parts[i]
    if keep[-1] < len(parts) - 1 and not text.endswith((".", "!", "?")):
        text = text.rstrip(";,: ") + " …"
    return text


def _cut_sentence(sentence: str, limit: int = POLICY_LINE_MAX) -> str:
    head = sentence[:limit - 1]
    at = max(head.rfind(", "), head.rfind(" — "), head.rfind(" ("))
    if at > limit // 2:
        return head[:at].rstrip() + "…"
    words = head.rsplit(" ", 1)[0].split(" ")
    while len(words) > 1 and re.search(r"\d", words[-1]):
        words.pop()                     # "must wait 15" would lose "months"
    return " ".join(words).rstrip(",;:—- ") + "…"


def shown_policy_passages(passages: Any, question: str = "") -> List[Dict[str, Any]]:
    """The policy passages as the answer shows them: at most three, in the search's order, chunks with no text under
    their header skipped and chunks of the same section merged; for each, the 2-4 lines sharing the most words with
    the question (in the passage's own order) and the source link."""
    sections: Dict[str, Dict[str, Any]] = {}
    for p in passages if isinstance(passages, list) else []:
        if not isinstance(p, dict):
            continue
        lines = _policy_lines(p.get("text") or "")
        if not lines:
            continue
        title, parent = _policy_title(p)
        key = p.get("section") or title or lines[0]
        if key in sections:
            sections[key]["lines"] += [line for line in lines if line not in sections[key]["lines"]]
        elif len(sections) < POLICY_PASSAGES_SHOWN:
            sections[key] = {"title": title, "parent": parent, "lines": lines, "source_url": p.get("source_url") or ""}
    pool = [line for s in sections.values() for line in s["lines"]]
    for s in sections.values():
        s["lines"] = [_trim_line(s["lines"][i], question) for i in _pick_lines(s["lines"], question, pool)]
    return list(sections.values())


def policy_fact_lines(passages: Any, question: str = "") -> List[str]:
    out: List[str] = []
    for s in shown_policy_passages(passages, question):
        indent = "  " if s["title"] else ""
        if s["title"]:
            out.append(f"- **{s['title']}**" + (f" ({s['parent']})" if s["parent"] else ""))
        out += [f"{indent}- {line}" for line in s["lines"]]
        if s["source_url"]:
            out.append(f"{indent}- Source: {s['source_url']}")
    return out or ["- The policy search found no passage that answers this. Please check with HDB."]


def template_response(state: HDBCompassState) -> str:
    lines: List[str] = []
    blocks = state.get("candidate_blocks_intersection") or []
    searched = (
        blocks or _sql_blocks(state) is not None
        or state.get("spatial_block_candidates") is not None
    )
    # a search tool that could not be loaded checked nothing, so the blocks meet only the constraints that were checked
    partial = any(t in BLOCK_FILTERS for t in unavailable_tools(state))
    if searched:
        if blocks:
            lines.append(f"**{len(blocks)} candidate block(s)** match "
                         f"{'the constraints that could be checked' if partial else 'all stated constraints'}:")
            for b in blocks[:LISTED_BLOCKS]:
                price = f" — median ${b['median_resale_price']:,.0f}" if b.get("median_resale_price") else ""
                lease = f", {b['median_remaining_lease_years']:.0f} yrs lease" if b.get("median_remaining_lease_years") else ""
                dist = ", ".join(f"{k.replace('_dist_m','')} {v:.0f}m" for k, v in (b.get("distances") or {}).items())
                town = f" ({b['town']})" if b.get("town") else ""
                lines.append(f"- Blk {b['block']} {b['street']}{town}{price}{lease}{(' — ' + dist) if dist else ''}")
            if len(blocks) > LISTED_BLOCKS:
                lines.append(f"- …and {len(blocks) - LISTED_BLOCKS} more, all shown on the map.")
        else:
            lines.append("**No block satisfies every constraint that could be checked.**" if partial
                         else "**No block satisfies every constraint at once.**")
            rel = state.get("relaxation") or {}
            if rel.get("best_single_change"):
                bsc = rel["best_single_change"]
                label, orig, new = plain_relaxation(bsc["constraint"], bsc["original"], bsc["relaxed"])
                lines.append(f"Relaxing **{label}** from {orig} to {new} unlocks {bsc['unlocked_blocks']} block(s).")
            for s in rel.get("suggestions", []):
                label, orig, new = plain_relaxation(s["constraint"], s["original"], s["relaxed"])
                lines.append(f"- {label}: {orig} → {new} unlocks {s['unlocked_blocks']}")
    lines.extend(not_checked_lines(state))   # straight after the blocks, so what was not checked is read with them
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
        lines.extend(policy_fact_lines(passages, user_request(state)))
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


# What a tool checks, for the answer to say when its module could not be loaded (tool_registry: status
# "unavailable"). Its result is dropped like any error (slot_outputs), so without this note a broken price search left
# spatial-only blocks listed as matching "all stated constraints" and a broken calculator's section simply vanished.
NOT_CHECKED = {
    "query_resale_blocks_by_criteria": "Prices, remaining leases and flat types",
    "compute_town_price_timeseries": "Town price trends",
    "query_comparable_resale_transactions": "Comparable resale sales",
    "compare_bto_launch_vs_resale": "BTO launch prices",
    "filter_blocks_by_amenity_proximity": "Distances to amenities",
    "verify_primary_school_distance": "Distances to the primary school",
    "compute_dual_transit_commute": "Commute times",
    "calculate_cpf_housing_grants": "Housing grants",
    "calculate_maximum_affordability": "The maximum budget",
    "calculate_bsd_and_upfront_outlay": "Upfront costs",
    "retrieve_hdb_policy_knowledge": "HDB policy rules",
}
BLOCK_FILTERS = ("query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity",
                 "verify_primary_school_distance", "compute_dual_transit_commute")


def unavailable_tools(state: HDBCompassState) -> List[str]:
    """Tools this answer called whose module could not be loaded (see NOT_CHECKED)."""
    return sorted({t.get("tool") for t in state.get("tool_call_log") or []
                   if str(t.get("error") or "").startswith("unavailable:") and t.get("tool") in NOT_CHECKED})


def not_checked_lines(state: HDBCompassState) -> List[str]:
    return [f"- {NOT_CHECKED[t]} could not be checked: that tool is unavailable right now." for t in unavailable_tools(state)]


def synthesis_evidence(state: HDBCompassState) -> Tuple[Dict[str, Any], str]:
    """What the final writing step may see, by architecture.
    C (and the default): the supervisor's join is the ONLY source of blocks. Raw price-query rows are withheld, so the
      writer cannot re-introduce a block the intersection excluded (the failure seen in the first Category E run).
    B: no join in code by design. The writer gets the data agent's own reconciliation plus the raw price AND distance
      results, and must do the join itself."""
    unbuilt = unbuilt_tools_used(state)
    base = {"user_constraints": constraints_for_model(state.get("user_constraints")),
            "financial_breakdown": state.get("financial_breakdown"),
            "rag_policy_passages": [] if "retrieve_hdb_policy_knowledge" in unbuilt else state.get("rag_policy_passages")}
    if unbuilt:
        base["not_available_yet"] = [UNBUILT_NOTES[t] for t in unbuilt]
    if unavailable_tools(state):   # only then, so every other prompt stays as it was
        base["could_not_check"] = [line[2:] for line in not_checked_lines(state)]
    sql ={k: v for k, v in (state.get("sql_market_results") or {}).items() if k not in unbuilt}
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
    if any(t in BLOCK_FILTERS for t in unavailable_tools(state)):   # a search that could not run checked nothing
        counts["blocks_meeting_the_constraints_that_could_be_checked"] = counts.pop("blocks_meeting_every_constraint")
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


# The opening carries no figures: the code-written facts section right after it states every block's price, lease and
# distance, the grants and the budget, and the prompt tells the model not to repeat or change them. Checking each figure
# against the right fact proved endless (a grant component taken as the total, a loan amount as the budget, an unlock
# count paired with the wrong relaxation, "over $800k", "the farthest block is 107m"), so an opening sentence that states
# any figure is dropped. Allowed: block names the facts list ("Blk 288B", "Blk 627 Ang Mo Kio Ave 9") and flat types, and
# in an answer about policy alone, figures the policy lines shown give ("10-year MOP"; see keep_supported_sentences).
_BLOCK_LIST = re.compile(r"\b(?:Blk|Block|Blocks|Blks)\.?\s*(?:No\.?\s*|#\s*)?(\d{1,4}[A-Z]?)\b((?:\s*(?:,|and|&|or)\s*(?:(?:Blk|Block)\.?\s*)?"
                         r"\d{1,4}[A-Z]?\b)*)", re.I)
_ADDRESS = re.compile(r"\b(\d{1,4}[A-Z]?)\s+(?:[A-Z][A-Za-z]+\s+){1,4}(?i:Ave|Avenue|St|Street|Rd|Road|Dr|Drive|Cres|Crescent|"
                      r"Ring|Way|Lane|Close|Central|Lorong|Lor)\b")
_NOT_A_FIGURE = re.compile(r"\b[2-5]\s*-?\s*(?:room|rm)s?\b|\b(?:Ave|Avenue|St|Street|Rd|Road|Ring|Sector|Lorong|Lor)\s+\d+[A-Z]?\b",
                           re.I)
_SPELLED_FIGURE = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                             r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
                             r"thousand|million|billion|dozen|percent|per cent|half a)\b", re.I)


def _split_opening(text: str) -> List[str]:
    """Sentences, also at ";", but not after "approx.", "abt.", "est.", "Blk.", "No." and similar abbreviations."""
    parts = re.split(r"(?<!approx\.)(?<!\babt\.)(?<!\best\.)(?<!Blk\.)(?<!No\.)(?<!\bSt\.)(?<!Ave\.)(?<!\bRd\.)(?<!\bvs\.)"
                     r"(?<!e\.g\.)(?<!i\.e\.)(?<=[.!?;])\s+", (text or "").strip(), flags=re.I)
    return [p for p in parts if p.strip()]


# A policy answer may state a figure the policy lines shown to the user give, written the same way: "Plus flats have a
# 10-year MOP" is what the passage says. A figure is a number with its unit: an amount as written ("$120,000" is not
# "$120k"), "6%", "10-year" / "10 years", "15 months", or a bare number ("aged 35"). A range ("6% to 9%", "between 6%
# and 9%", "6% up to 9%") must appear as that range, and its ends are not figures on their own: otherwise "3%" alone, or
# "between 3% and 9%", would pass by taking the ends of the passage's "3% to 6%" and "6% to 9%". Figures in a worked
# example ("Example: If a Plus flat ... is sold for $900,000") are not facts about the user's case, so they are not
# allowed.
_POLICY_FIGURE = (r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?(?:k\b)?"
                  r"|\d+(?:,\d{3})*(?:\.\d+)?(?:\s?%|\s?-?\s?(?:years?|yrs?|months?|weeks?|days?|km)\b)?")
_POLICY_FIGURE_SPAN = re.compile(rf"\bbetween\s+({_POLICY_FIGURE})\s+and\s+({_POLICY_FIGURE})"
                                 rf"|({_POLICY_FIGURE})(?:\s*(?:to|up\s+to|through|-|–|—)\s*({_POLICY_FIGURE}))?", re.I)


def _figure_key(text: str) -> Tuple[str, float]:
    """("$", 120000.0) for "$120,000"; ("$k", 120.0) for "$120k"; ("year", 10.0) for "10-year" and "10 years";
    ("", 35.0) for "35"."""
    s = text.lower().replace(",", "")
    value = float(re.search(r"\d+(?:\.\d+)?", s).group())
    if s.startswith("$"):
        return ("$k" if s.endswith("k") else "$"), value
    for unit in ("%", "year", "yr", "month", "week", "day", "km"):
        if unit in s:
            return ("year" if unit == "yr" else unit), value
    return "", value


def _figures(text: str) -> List[Any]:
    """The figures in a text, each a key or, for a range, a pair of keys sharing a unit ("3 to 6%" is 3% to 6%)."""
    found: List[Any] = []
    for m in _POLICY_FIGURE_SPAN.finditer(_NOT_A_FIGURE.sub(" ", (text or "").replace("**", ""))):
        low, high = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        a = _figure_key(low)
        if not high:
            found.append(a)
            continue
        b = _figure_key(high)
        unit = a[0] or b[0]
        found.append(((unit, a[1]), (unit, b[1])))
    return found


def policy_figures(passages: Any, question: str = "") -> set:
    """Every figure in the policy lines the answer shows (titles and lines, but not worked examples). A range is one
    figure: its ends are not allowed on their own."""
    shown = shown_policy_passages(passages, question)
    lines = [s["title"] for s in shown] + [line for s in shown for line in s["lines"]]
    return set(_figures("\n".join(line for line in lines if not re.match(r"(?:worked )?example\b", line, re.I))))


def keep_supported_sentences(opening: str, facts: str, user_text: str = "",
                             policy_passages: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, int]:
    """The model's opening with every sentence dropped that states a figure (an amount, distance, lease, percentage,
    count or time, in digits or words) or names a block the facts do not list. The facts section that follows states
    every figure. The one exception is a policy answer (policy_passages given): a sentence that names no block may
    state figures that the policy lines shown to the user give, in digits as they give them. user_text is the user's
    request; it picks which policy lines are shown, and the user's own limits are not repeated either.
    Returns (text, sentences dropped)."""
    listed = {m.group(1).upper() for m in re.finditer(r"\bBlk\s+(\d{1,4}[A-Z]?)\b", facts or "")}
    allowed = policy_figures(policy_passages, user_text) if policy_passages else set()
    kept, dropped = [], 0
    for sentence in _split_opening(opening):
        named = [m.group(1).upper() for m in _BLOCK_LIST.finditer(sentence)]
        named += [b.upper() for m in _BLOCK_LIST.finditer(sentence) for b in re.findall(r"(\d{1,4}[A-Z]?)\b", m.group(2) or "")]
        named += [m.group(1).upper() for m in _ADDRESS.finditer(sentence)]
        rest = _NOT_A_FIGURE.sub(" ", _ADDRESS.sub(" ", _BLOCK_LIST.sub(" ", sentence)))
        states_figure = bool(re.search(r"\d", rest) or _SPELLED_FIGURE.search(rest))
        policy_figure = (bool(allowed) and not named and not _SPELLED_FIGURE.search(rest)
                         and all(f in allowed for f in _figures(rest)))
        if any(b not in listed for b in named) or (states_figure and not policy_figure):
            dropped += 1
        else:
            kept.append(sentence)
    return " ".join(kept), dropped


def code_written_facts_on() -> bool:
    """The architecture.code_written_facts switch. HDB_CODE_WRITTEN_FACTS=0 or 1 overrides it for one run without
    editing the config (the live ablation), like HDB_EXTRACTION_REASONING. Any other value is refused rather than
    guessed at, so an ablation never runs with the switch silently the wrong way round."""
    env = os.getenv("HDB_CODE_WRITTEN_FACTS", "").strip()
    if env:
        if env not in ("0", "1"):
            raise ValueError(f"HDB_CODE_WRITTEN_FACTS must be 0 or 1, not {env!r}")
        return env == "1"
    return bool(load_config().get("architecture", {}).get("code_written_facts", True))


def writes_facts_in_code(state: HDBCompassState) -> bool:
    """Variant C (and the app): the facts section of the answer (blocks, prices, leases, distances, relaxations,
    grants, budget) is written by code, the same text the no-model template produces, and the model writes only
    the opening and the trade-offs, so it cannot drop or garble a number. Variant B keeps the model as the writer
    by design (no join in code). Switch: architecture.code_written_facts (code_written_facts_on)."""
    return state.get("variant") != "B" and code_written_facts_on()


def policy_answer_passages(state: HDBCompassState) -> Optional[List[Dict[str, Any]]]:
    """The real policy passages an opening may take figures from, or None. Only an answer about policy alone: when the
    facts also carry grants, budgets, prices or blocks, a passage figure (the $80,000 Family Grant tier) could be read
    as the user's own result, so those openings stay figure-free. Never while the policy search is mocked."""
    passages = state.get("rag_policy_passages")
    if not passages or "retrieve_hdb_policy_knowledge" in unbuilt_tools_used(state):
        return None
    if any(state.get(k) for k in ("financial_breakdown", "sql_market_results", "spatial_block_candidates",
                                  "candidate_blocks_intersection", "relaxation")):
        return None
    return passages


def synthesize_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    blocks = state.get("candidate_blocks_intersection") or []
    log: List[Dict[str, Any]] = []
    payload = build_map_payload(blocks, c, log) if blocks else {}
    evidence, rule = synthesis_evidence(state)
    unchecked = (" Some tools could not run: say plainly what could_not_check lists, and never say the blocks meet "
                 "every constraint or give a figure those tools would have produced.") if evidence.get("could_not_check") else ""
    rule += unchecked
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
    policy = policy_answer_passages(state) if facts else None
    if llm is not None:
        try:
            if facts:
                numbers = ("write no numbers except block names exactly as the facts give them (any other sentence with a "
                           "number is removed)")
                if policy:
                    numbers = ("you may state a policy figure (a period, age, percentage or amount) only exactly as the "
                               "Policy lines above write it, digits and unit included (\"10-year MOP\", \"3% to 6%\"), a range "
                               "always whole and never one end of it or ends of two ranges joined; write no other numbers "
                               "(any other sentence with a number is removed)")
                rule = ("The facts section below was written by code from the tool results and will be shown to the user "
                        "exactly as written, straight after your text:\n---\n" + facts + "\n---\nWrite only a short opening "
                        "(one or two sentences that answer the question directly) and, when there are several options, the "
                        "trade-offs between them (price vs lease vs distance). Do not repeat or change any price, grant, "
                        "budget, distance, lease, count or relaxation figure: " + numbers + ", and do not contradict the facts."
                        + unchecked)
            msgs = [sys_msg(SYNTH_SYSTEM),
                    HumanMessage(content=f"User query: {user_request(state)}\n\n{rule}\n\nEvidence JSON:\n{json.dumps(evidence, default=str)[:24000]}")]
            if verdict and not verdict.get("passed", True) and verdict.get("critique"):
                msgs.append(HumanMessage(content=f"Your previous draft was rejected by the auditor: {verdict['critique']}. Rewrite it, fixing every issue."))
            text = llm.invoke(msgs).content
            if isinstance(text, list):
                text = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in text)
            text = strip_think(text)
            if facts:
                opening, dropped = keep_supported_sentences(text, facts, user_request(state), policy)
                if dropped:
                    errors.append(f"synthesize: dropped {dropped} opening sentence(s) that stated figures")
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
HOOK_NAMES = ("input_guardrail_node", "runtime_auditor_node")


def _load_member5_hooks(module: str = "agents.guardrails") -> Tuple[Optional[Callable], Optional[Callable], List[str]]:
    """Member 5's guardrail and auditor, and what went wrong loading them. A module that does not exist yet is the
    expected state until Member 5 delivers: both stay pass-throughs, quietly. A module that exists but fails to import
    or lacks a node is a bug: one loud warning, and the problem is kept in HOOK_LOAD_PROBLEMS, which the guardrail and
    auditor pass-throughs write into each run's errors and the evaluation records in its run settings."""
    import importlib
    import importlib.util
    try:
        if importlib.util.find_spec(module) is None:
            return None, None, []
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001 — a broken hook module must not stop the app; it is reported instead
        problems = [f"{module} failed to import ({type(e).__name__}: {str(e)[:200]})"]
        hooks: List[Optional[Callable]] = [None, None]
    else:
        hooks = [getattr(mod, name, None) for name in HOOK_NAMES]
        problems = [f"{module} has no callable {name}" for name, fn in zip(HOOK_NAMES, hooks) if not callable(fn)]
        hooks = [fn if callable(fn) else None for fn in hooks]
    if problems:
        print(f"\n{'!' * 78}\nWARNING: Member 5 hooks: {'; '.join(problems)}.\nThe missing guardrail or auditor "
              f"runs as a pass-through; every answer records this in its errors.\n{'!' * 78}\n", file=sys.stderr, flush=True)
    return hooks[0], hooks[1], problems


_INPUT_GUARDRAIL, _AUDITOR, HOOK_LOAD_PROBLEMS = _load_member5_hooks()


def hook_problem_note(node: str) -> List[str]:
    """The errors entry a pass-through adds when its Member 5 hook exists but could not be loaded. The note quotes no
    exception text (that is in HOOK_LOAD_PROBLEMS), so it can never be mistaken for a model error by the evaluation."""
    if not HOOK_LOAD_PROBLEMS:
        return []
    return [f"{node}: Member 5 hook could not be loaded ({len(HOOK_LOAD_PROBLEMS)} problem(s), see the run settings); "
            "pass-through used"]


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
    note = hook_problem_note("input_guardrail")
    return {"errors": note} if note else {}  # pass-through until Member 5 delivers


def guardrail_gate(state: HDBCompassState) -> str:
    return "finalize" if state.get("draft_response") else "extract_constraints"


def audit_node(state: HDBCompassState) -> Dict[str, Any]:
    if _AUDITOR is not None:
        return _AUDITOR(state)
    prev = state.get("auditor_verdict") or {}
    critique = "auditor failed to load (pass-through)" if HOOK_LOAD_PROBLEMS else "auditor not installed (pass-through)"
    update: Dict[str, Any] = {"auditor_verdict": AuditorVerdict(passed=True, critique=critique,
                                                                retry_count=int(prev.get("retry_count", 0))).model_dump()}
    if not prev:   # once per answer, not on every rewrite round
        update["errors"] = hook_problem_note("runtime_auditor")
    return update


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
