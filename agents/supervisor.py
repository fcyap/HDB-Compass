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
from agents import addons
from agents import scope
from agents import research_intents as ri
from agents.schemas import (
    TOOL_OWNER,
    LLMExtractedConstraints,
    AuditorVerdict,
    ExtractedConstraints,
    RelaxationSuggestion,
    RelaxUnsatisfiedConstraintsOutput,
    block_key,
)
from agents.state import HDBCompassState
from agents.school_queries import guard_school_request, present_school_result, school_location_request
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
                    r"probably|maybe|perhaps|possibly|might|may|could be|pending|to be confirmed|tbc|waiting|worried|afraid|"
                    r"heard|says|told (?:us|me)|according to|not true)\b", re.I)
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
    nearby = school_location_request(query)
    if nearby is not None:
        return ExtractedConstraints(nearby_schools=nearby)
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
    m = re.search(r"(?:of|near|to)\s+((?:[A-Z][a-zA-Z'’]+\s){1,4}(?:Primary School|School|Primary|High))", q, re.I)
    if not m:
        m = re.match(r"\s*([A-Za-z'’ -]+?\b(?:Primary School|School|Primary|High))\s*(?=[12]\s*(?:[-–]\s*2\s*)?km\b|within\b|[.?!]|$)", q, re.I)
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in {"primary school", "school", "moe primary school"}:
            school = candidate
            radius = 2000 if re.search(r"\b(?:1\s*[-–]\s*)?2\s*km\b|\b2000\s*m\b", q, re.I) else 1000
            radii.setdefault("school", radius)

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
    if re.search(r"\b(mop|minimum occupation|prime|plus (?:flats?|models?|bto|resale|classification)|standard flat|clawback|subsidy recovery|eip|ethnic\w*|quota\w*|wait[- ]out|eligible|eligibility (?:criteria|requirements?|rules?|conditions?|to )|scheme\w*|rental restriction|exempt\w*"
                 r"|not married|unmarried|fianc[eé]e?s?|apply(?:ing)? for (?:an? )?(?:[2-5][- ]room )?(?:flexi )?btos?|can (?:we|i) (?:buy|apply)"
                 # rules asked about in other words: "grant rules", "2-room Flexi", "how long must I wait", "can I rent out",
                 # "after selling my private property"
                 r"|(?:grant|housing|hdb|resale|bto) rules?"
                 r"|how long (?:must|do|should|will|would) (?:i|we) (?:have to )?wait (?:before|until|to) (?:i |we )?(?:can )?(?:buy|apply|purchase)"
                 r"|(?:sold|selling|disposed of|disposal of) (?:my |our |the |a )?private (?:residential )?propert(?:y|ies)"
                 # who may buy, which grants exist, income limits, how often one may apply (the stems above need endings)
                 r"|(?:can|may|could) (?:an? )?(?:pr|prs|permanent residents?|foreigners?|singles?|singaporeans?|divorcees?|widow(?:er)?s?)\s+(?:buy|apply|own|purchase|get)"
                 r"|(?:pr|prs|permanent residents?|foreigners?|singles?|divorcees?|widow(?:er)?s?)\s+(?:allowed|able|permitted|eligible)\s+to\s+(?:buy|apply|own|purchase)"
                 r"|(?:what|which) (?:housing |cpf )?grants? (?:are|is|exist|can|could|do|does|would|will|am|might)|grants? (?:are )?(?:available|offered)|grants? for (?:first|second|singles?|young|large|married|couples?|families|seniors?|resale|bto)|income (?:ceiling|cap|limit)s?"
                 r"|how many times (?:can|may) (?:i|we) apply|second[- ]?timers?"
                 r"|(?:can|may) (?:i|we) (?:rent out|sublet)|rent out (?:my|our|the) (?:whole )?(?:flat|room|bedroom|unit)|buyback)\b", q, re.I):
        policy = q

    base = dict(
        towns=towns, flat_types=flat_types, max_budget_sgd=budget, min_remaining_lease_years=lease,
        amenity_radii_meters=radii, household_income=income, citizenship_scheme=scheme, cpf_oa_balance=cpf,
        school_name=school, work_destinations=work, max_commute_minutes=max_commute, policy_question=policy,
        wants_price_trend=bool(re.search(r"trend|appreciat|per sqm|psm|over the (?:last|past)|median price|price changes?|prices? (?:have |has )?(?:changed|moved|risen|rise|fallen|fall)", q, re.I)),
        wants_bto_comparison=bool(re.search(r"\bbto\b.*(?:vs|versus|against|compar)|compar.*\bbto\b", q, re.I)),
    )
    base.update(ri.extract_research_fields(q, base))
    return ExtractedConstraints(**base)


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
    " School-distance and Primary 1 boundary searches belong to the spatial agent; "
    "they are not HDB housing-policy questions. Do not invent a school name."
    " addresses: copy every HDB block address, postal code or named place (MRT station, hawker centre, school) "
    "exactly as written. asking_price_sgd: only a specific flat's asking/offer price to check. "
    "wants_block_history: past sales of a named block. wants_price_assessment: is a price fair/reasonable. "
    "wants_option_comparison: comparing two or more named blocks. wants_nearby_amenities: what is near a place. "
    "wants_market_snapshot: how the market or prices stand now. wants_town_ranking: which town is best value or "
    "cheapest. trend_years: N in 'over the last N years'. bto_project_name: a named BTO launch."
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
    "continuous_employment_12_months true; 'calc w 65 yrs lease' is remaining_lease_years 65. Respond with JSON only."
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
# "a $600k flat", "a 4-room resale at..." is not here: only a price written next to the flat, and only read when the user is
# asking for upfront costs (stated_financial_inputs), since "find a $600k flat in Tampines" is a budget
_FLAT_PRICES = (r"\b(?:an?|the)\s+" + _AMOUNT + r"(?!\d)\s+(?:[\w-]+\s+){0,2}?(?:flat|unit|home|bto)\b",
                r"\b(?:flat|unit|home)s?\s+(?:costing|priced at|worth|that costs?|valued at)\s+" + _AMOUNT + r"(?!\d)")
# the lease the user wants the calculation to assume: "use 65 yrs remaining lease for the calc", "calc w 65 yrs lease",
# "using a 65-year remaining lease", "assume 65 years of remaining lease"
_CALC_LEASE = (r"(\d{2})\s*(?:years?|yrs?) of remaining lease for the .{0,30}calc"
               r"|\b(?:assume|use|using)\s+(?:an?\s+)?(\d{2})[- ]?(?:years?|yrs?)\b(?:\s+of)?(?:\s+remaining)?\s+lease"
               r"|\b(?:work out|calculate|calc\w*|sums?|grants?|loan)\b[^.;,]{0,40}\bon\s+an?\s+(\d{2})[- ]?(?:years?|yrs?)\s+"
               r"(?:of\s+)?remaining\s+lease"
               r"|\bcalc\w*\s+(?:w|with|using)\s+(\d{2})\s*(?:years?|yrs?)\b(?:\s+of)?(?:\s+remaining)?\s+lease")
# the buyers say no proximity grant applies: "no PHG", "no proximity grant (for us)", "we don't qualify for the proximity grant"
_NO_PROXIMITY = (r"\bno proximity (?:housing )?grant\b|\bno PHG\b|\bproximity (?:housing )?grant (?:does not|doesn't|won't) apply\b"
                 r"|\b(?:we|i)\s*(?:don't|do not|won't|will not|'re not|are not|'m not|am not|aren't)\s+(?:qualify|eligible)\s+for\s+"
                 r"(?:the\s+|a\s+|any\s+)?(?:proximity (?:housing )?grant|PHG)\b")
_WORKED = r"(?:employ\w*|working|(?:have|has|'ve)\s+(?:both\s+|each\s+|all\s+)?worked)"   # "have worked", not "worked in 2019"
_CONTINUOUS_WORK = (rf"(?:continuous(?:ly)?\s+(?:employ\w*|working)|{_WORKED}\s+(?:\w+\s+){{0,2}}continuously)"
                    r".{0,40}\b(?:12|twelve)\s*(?:months?|mths?)\b"
                    rf"|{_WORKED}\s+(?:12|twelve)\s*(?:months?|mths?)\s+continuously")


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
    # "eligibility for the grants is confirmed", "HDB has confirmed our eligibility for the grants"
    if _stated_match(r"eligibility for (?:the )?grants?.{0,40}\bconfirmed\b|\bgrants?\b.{0,30}\beligibility\b.{0,25}\bconfirmed\b"
                     r"|\bconfirmed (?:our|my) eligibility for (?:the )?grants?", s):
        f["eligibility_confirmed"] = True
    if _stated_match(r"eligibility for .{0,30}\b(?:HDB|bank) loan (?:is |has been )?confirmed"
                     r"|\b(?:HDB|bank) loan\b.{0,15}\beligibility\b.{0,25}\bconfirmed\b"
                     r"|\b(?:HDB|bank) loan(?: eligibility)? (?:is |has been )?confirmed"
                     r"|\bconfirmed (?:our|my) eligibility for (?:the )?(?:grants? and (?:an? |the )?)?(?:HDB|bank) loan", s):
        f["loan_eligibility_confirmed"] = True
    if re.search(r"\bbank loan\b", t, re.I):
        f["loan_type"] = "bank"
    elif re.search(r"\bHDB loan\b", t, re.I):
        f["loan_type"] = "hdb"
    if re.search(_NO_PROXIMITY, s, re.I):
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
    elif re.search(_ASKS_UPFRONT, t):
        for pattern in _FLAT_PRICES:   # read from the whole message: the price is in the question itself
            m = re.search(pattern, t, re.I)
            if m:
                upfront["purchase_price"] = _amount(m)
                break
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
_NEGATED_BEFORE = re.compile(r"(?:\bnot|n't|\bno longer|\bnever|\bneither|\bnor|\byet to|\bcannot|\bnobody|\bno one)\b[^.;,]{0,25}$", re.I)


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
                                     for pat in (_LEASE_FIG, re.compile(_CALC_LEASE, re.I)) for m in pat.finditer(own))
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
        none = bool(re.search(_NO_PROXIMITY + r"|\bno proximity\b|\bnot (?:living )?near (?:my|our) parents\b", s, re.I))
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


def read_constraints_with_model(llm: Any, messages: List[Any],
                                used: Optional[List[str]] = None) -> Tuple[Optional[ExtractedConstraints], List[str]]:
    """Structured constraint reading that degrades gracefully: invalid fields are dropped one by one
    (lenient_constraints), and a reply that is not JSON at all gets one repair turn showing the model its own output
    and the error. Returns (constraints or None, notes for the errors list)."""
    notes: List[str] = []
    # LLMExtractedConstraints types amenity_radii_meters as fixed fields: strict JSON-schema providers (Groq, OpenAI)
    # reject the open-ended dict of ExtractedConstraints. function_calling is the last resort for providers without
    # JSON modes. json_mode needs the schema in the prompt itself.
    for method in ("json_schema", "json_mode", "function_calling"):
        call = messages
        if method == "json_mode" and messages:
            call = [sys_msg(f"{messages[0].content} Reply with one JSON object for this schema: "
                            f"{json.dumps(LLMExtractedConstraints.model_json_schema())}")] + list(messages[1:])
        try:
            res = llm.with_structured_output(LLMExtractedConstraints, method=method, include_raw=True).invoke(call)
        except Exception as e:  # noqa: BLE001
            notes.append(f"extract_constraints[{method}] failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        parsed = res.get("parsed") if isinstance(res, dict) else None
        if parsed is not None:
            if used is not None:
                used.append(method)   # recorded as extraction_source "llm:<method>"
            return (parsed.to_constraints() if hasattr(parsed, "to_constraints") else parsed), notes
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
        if isinstance(data.get("amenity_radii_meters"), dict):   # the typed schema sends null for unstated radii
            data["amenity_radii_meters"] = {k: v for k, v in data["amenity_radii_meters"].items() if v}
        extracted, dropped = lenient_constraints(data)
        if dropped:
            notes.append(f"extract_constraints[{method}]: dropped invalid fields {', '.join(dropped)}")
        if used is not None:
            used.append(method)
        return extracted, notes
    return None, notes


def extract_constraints_node(state: HDBCompassState) -> Dict[str, Any]:
    query = last_user_query(state)
    pending = state.get("pending_clarification")
    if school_location_request(query) is not None:
        # a self-contained address-centred school lookup: read by code, earlier turns not carried in
        c = rule_based_extract(query).model_dump()
        c["requested_tools"] = requested_tools(query)
        c["extraction_source"] = "regex"
        return {"user_constraints": c, "errors": []}
    if state.get("fast_path") and not pending:
        # App mode: when code alone pins down every tool a self-contained question needs, a model round trip adds
        # latency but no information. Read from this message only; a reply to an open question never takes this path.
        fast = _finalize_constraints(rule_based_extract(query).model_dump(), query, state, [], from_model=False,
                                     with_history=False)
        if ri.answerable_without_model(fast, query):
            fast["extraction_source"] = FAST_PATH_SOURCE
            return {"user_constraints": fast, "errors": []}
    prior = state.get("prior_constraints")
    llm = get_llm()
    extracted: Optional[ExtractedConstraints] = None
    model_reading: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    source = "regex"
    system = EXTRACT_SYSTEM
    if pending and pending.get("kind", "financial") == "financial":
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
        used: List[str] = []
        extracted, notes = read_constraints_with_model(llm, messages, used)
        errors += notes
        if extracted is not None:
            source = f"llm:{used[-1]}" if used else "llm"
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
        if llm is not None:
            errors.append("extract_constraints: model extraction failed; rule-based fallback used")
        extracted = rule_based_extract(query)
    constraints = extracted.model_dump()
    if extracted.financial_inputs is not None:
        constraints["financial_inputs"] = extracted.financial_inputs.model_dump(exclude_unset=True)
    constraints = _finalize_constraints(constraints, query, state, errors, from_model=source != "regex")
    constraints["extraction_source"] = source
    return {"user_constraints": constraints, "errors": errors, "model_constraints": model_reading}


def _finalize_constraints(constraints: Dict[str, Any], query: str, state: HDBCompassState, errors: List[str],
                          from_model: bool, with_history: bool = True) -> Dict[str, Any]:
    """Everything code does to a reading, whichever path produced it (model, regex fallback or the app's fast path),
    so the paths cannot drift apart. with_history=False reads this message alone (the fast path's self-contained
    questions); otherwise earlier turns and any open follow-up question are carried in."""
    prior = state.get("prior_constraints") if with_history else None
    pending = state.get("pending_clarification") if with_history else None
    # calculator facts come from the user's own words in every turn so far, never from the model's reading of them
    turns = [m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
    if not with_history or not turns:
        turns = [query]
    users = "\n".join(turns)
    unsure: set = set()
    stated = stated_across_turns(turns, unsure)
    constraints["financial_inputs"] = trusted_financial_inputs(constraints.get("financial_inputs"), stated, users, unsure)
    # a distance limit of 0 m is not a constraint: the model filled every amenity with 0 for an eligibility question,
    # which routed it to the map agent and produced "relax from 0 m to 150 m" suggestions
    constraints["amenity_radii_meters"] = {k: v for k, v in (constraints.get("amenity_radii_meters") or {}).items()
                                           if v is not None and v > 0}
    _verify_extraction(constraints, query, errors, from_model=from_model)
    if from_model and constraints.get("policy_question") and not scope.mentions_policy_topic(query):
        # the model filed a vague question ("Should I buy BTO or resale?") as a policy question; the policy search
        # would return loosely related rules, so the user gets the menu instead (decide_routing)
        constraints["policy_question"] = None
        errors.append("extract_constraints: model's policy reading dropped: the message names no HDB rule")
    if pending and pending.get("kind", "financial") == "financial":
        conversation = " ".join(m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict))
        answers = parse_clarification_answer(query, pending["needed"], conversation)
        if answers:
            constraints = merge_constraints(constraints, {"financial_inputs": answers})
    # the place an earlier turn named is a candidate even for a self-contained question ("what's near there?")
    _apply_map_pin(constraints, query, state, prior=state.get("prior_constraints"), pending=pending)
    if prior:
        # A new question does not inherit the last one's one-off asks (block history, a snapshot, a clarifying
        # question); a reply to an open question keeps them, since it completes that request.
        carried = {k: v for k, v in prior.items()
                   if k not in ri.ALWAYS_FRESH_KEYS and (pending or k not in ri.TURN_ONLY_KEYS)}
        constraints = merge_constraints(carried, constraints)
    # a fact a later message changed in a way code could not read is asked again, not carried over from the last turn
    for vals in (constraints.get("financial_inputs") or {}).values():
        for k in unsure:
            vals.pop(k, None)
    # a data question no specific intent recognised goes to the market agent's analysis tools instead of the menu
    constraints["analysis_mode"] = False
    if (with_history and not pending and not constraints.get("clarifications") and not constraints.get("place_options")
            and not constraints.get("addresses") and not constraints.get("map_pin") and scope.mentions_data(query)
            and addons.offered_in(state.get("variant"), bool(state.get("fast_path")))
            # nothing specific was recognised; a town on its own (which only started a block search) does not count
            and not decide_routing({**constraints, "towns": []})):
        constraints["analysis_mode"] = True
    # read by code from every user message so far, so an ask in the first turn still counts after a follow-up reply
    constraints["requested_tools"] = requested_tools(users)
    comparables = "query_comparable_resale_transactions"
    if comparables not in constraints["requested_tools"] and coordinates_stated(constraints.get("comparable_resale"), users):
        constraints["requested_tools"].append(comparables)
    return constraints


FAST_PATH_SOURCE = "regex:fast_path"


def _apply_map_pin(c: Dict[str, Any], query: str, state: HDBCompassState, prior: Optional[Dict[str, Any]] = None,
                   pending: Optional[Dict[str, Any]] = None) -> None:
    """Which place the message is about: the user's map pin, a place it names, or one from earlier turns
    (research_intents.resolve_place asks when the pin and an earlier place are both plausible)."""
    map_state = state.get("map_ui_payload") or {}
    pin = map_state.get("center")
    pin = (float(pin[0]), float(pin[1])) if isinstance(pin, (list, tuple)) and len(pin) == 2 else None
    earlier = list((prior or {}).get("addresses") or [])
    ri.resolve_place(c, query, pin, earlier, pending, radius_meters=map_state.get("radius_meters"))


def _verify_extraction(c: Dict[str, Any], query: str, errors: List[str], from_model: bool = True) -> None:
    """Drop extracted facts the user did not actually supply."""
    comp = c.get("comparable_resale")
    if from_model and comp and not all(re.search(re.escape(f"{float(comp[k]):.3f}")[:-1], query) for k in ("latitude", "longitude")):
        c["comparable_resale"] = None
        errors.append("extract_constraints: comparable coordinates were not in the message; dropped")
    if from_model and c.get("towns"):
        qu = query.upper()
        stated = {t for t in HDB_TOWNS if t in qu} | {town for alias, town in TOWN_ALIASES.items() if alias in qu}
        invented = [t for t in c["towns"] if t not in stated]
        if invented:
            c["towns"] = [t for t in c["towns"] if t in stated]
            errors.append(f"extract_constraints: towns not named in the message were dropped: {invented}")
    if ri.generic_school_phrase(c.get("school_name")):
        c["school_name"] = None
        (c.get("amenity_radii_meters") or {}).pop("school", None)
        c["clarifications"] = list(dict.fromkeys((c.get("clarifications") or []) + [ri.SCHOOL_CLARIFICATION]))
    # Distance phrases ("200m of a hawker centre") are constraints, not places.
    c["addresses"] = [a for a in c.get("addresses") or [] if a.strip() and not re.match(r"\s*\d+(?:\.\d+)?\s*k?m\b", a, re.I)]


# ---------------------------------------------------------------------------
# Node: route  +  fan-out dispatch
# ---------------------------------------------------------------------------
# first-timer status, citizenship and flat type alone are not something the calculators can work from
_WEAK_FINANCIAL_FACTS = frozenset({"first_timer_status", "citizenship_pair", "flat_type"})


def calculator_facts(c: Dict[str, Any]) -> bool:
    """Whether the user gave the calculators anything to work with beyond who they are ("I'm a first-time buyer")."""
    return any(v for vals in (c.get("financial_inputs") or {}).values() if vals for k, v in vals.items()
               if k not in _WEAK_FINANCIAL_FACTS)


def decide_routing(c: Dict[str, Any], pending: Optional[Dict[str, Any]] = None) -> List[str]:
    """pending: the open follow-up question, if any. While the user is answering a financial question, the weak facts
    they give (citizenship, first-timer status) still go to the calculators, which ask for the rest."""
    if c.get("place_options"):
        return []   # the user must first say which place they mean (research_intents.resolve_place)
    if c.get("nearby_schools"):
        return ["spatial"]
    active: List[str] = []
    if c.get("comparable_resale") or (c.get("towns") and (c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("flat_types")
                           or c.get("wants_price_trend") or c.get("wants_bto_comparison"))):
        active.append("market_sql")
    if c.get("amenity_radii_meters") or c.get("school_name") or len(c.get("work_destinations") or []) >= 2:
        active.append("spatial")
    if ri.needs_market_agent(c) and "market_sql" not in active:
        active.append("market_sql")
    if ri.needs_spatial_agent(c) and "spatial" not in active:
        active.append("spatial")
    financial_facts = has_financial_facts(c) if (pending or {}).get("kind") == "financial" else calculator_facts(c)
    if financial_facts or c.get("household_income") or c.get("cpf_oa_balance") or (c.get("citizenship_scheme") and c.get("max_budget_sgd")):
        active.append("financial")
    if c.get("policy_question"):
        active.append("policy_rag")
    if not active and (c.get("addresses") or c.get("map_pin")):
        pass  # a place with no recognised question: route_node asks instead of guessing a search or policy query
    elif not active and c.get("towns"):
        active.append("market_sql")
    elif not active and c.get("analysis_mode"):
        active.append("market_sql")
    # Nothing recognised and no towns: no specialist runs. route_node answers with a menu of what the app can do, rather
    # than searching the policy text for a message that is not a policy question.
    return active


def route_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    active = decide_routing(c, state.get("pending_clarification"))
    if not active and not c.get("clarifications") and (c.get("addresses") or c.get("map_pin")):
        return {"active_subagents": active, "user_constraints": {**c, "clarifications": [ri.place_clarification(c)]}}
    if not active and not c.get("clarifications"):
        menu = scope.menu_for(last_user_query(state))
        return {"active_subagents": active,
                "user_constraints": {**c, "clarifications": [menu["question"]], "menu_options": menu["options"]}}
    return {"active_subagents": active}


def dispatch(state: HDBCompassState) -> List[Send]:
    """Parallel fan-out: LangGraph runs one Send per active specialist concurrently.
    With no specialist (only a clarifying question to ask) go straight to synthesis."""
    active = state.get("active_subagents", [])
    return [Send(f"{a}_agent", state) for a in active] if active else [Send("synthesize", state)]


# ---------------------------------------------------------------------------
# Specialist runner: ReAct loop over one agent's tools, or a deterministic plan
# ---------------------------------------------------------------------------
AGENT_SYSTEM = {
    "market_sql": "You are the Market & Valuation SQL agent for Singapore HDB resale flats. Use your tools to fetch "
                  "block-level prices, trends and BTO comparisons that satisfy the user's constraints. Call tools with "
                  "exact towns/flat types from the constraints. For town trends or 'over the last N years' use "
                  "compute_town_price_timeseries with lookback_years=N and its period_change_pct. For comparable sales use "
                  "verified coordinates (from a tool result or the user's pin, never estimated) and explicit matching "
                  "filters. Disclose small samples, truncated examples and missing geocodes; "
                  "do not widen filters silently or present comparables as an official valuation. "
                  "When done, reply with a one-paragraph summary.",
    "spatial": "You are the Geospatial & Commute agent. Use your tools to find HDB blocks meeting amenity-distance, "
               "school-boundary and commute constraints. Pass towns exactly as given. Commute times are town-level "
               "estimates, not routed journeys; say so. Reply with a short summary when done.",
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
CORE_AGENT_SYSTEM = dict(AGENT_SYSTEM)
# each add-on tool's "when to use" line goes into its owner's prompt (agents/addons), so a model offered it knows it
AGENT_SYSTEM = {agent: text + addons.prompt_section(agent) for agent, text in AGENT_SYSTEM.items()}


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


# A Variant C specialist is offered only the tools its task needs, read from the stated constraints the same way its
# deterministic plan reads them: a price trend only when one is asked for, the BTO comparison only when asked, the school
# check only for a named school, the commute only for two workplaces. The map is drawn by code (build_map_payload), so
# the map-sync tool is never offered. In the 31 Oct live run C's models called these unasked in 6-10 of 15 questions
# each (Tool F1 0.81 vs 0.94 for A and B). Variants A and B keep their full lists.
TASK_TOOLS = {"compute_town_price_timeseries": lambda c: bool(c.get("wants_price_trend")),
              # a named launch (research_intents.known_bto_project) is an ask too, not only "BTO vs resale" wording
              "compare_bto_launch_vs_resale": lambda c: bool(c.get("wants_bto_comparison") or c.get("bto_project_name")),
              "verify_primary_school_distance": lambda c: bool(c.get("school_name")),
              "compute_dual_transit_commute": lambda c: len(c.get("work_destinations") or []) >= 2,
              "sync_interactive_map_ui": lambda c: False}
TASK_TOOLS.update(addons.task_tools())   # add-on tools: offered only when their intent was read


def needed_for_task(name: str, c: Dict[str, Any], query: str = "") -> bool:
    """Whether a Variant C specialist is offered this tool: an optional tool only when asked for (asked_for), a
    task tool only when the constraints call for it (TASK_TOOLS), every other tool always."""
    if c.get("analysis_mode") and TOOL_OWNER.get(name) == "market_sql":
        return name in ri.ANALYSIS_TOOLS            # the analysis tools only; the block search and its kin are not offered
    return asked_for(name, c, query) and TASK_TOOLS.get(name, lambda _: True)(c)


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
        if towns and ri.is_block_search(c, query):
            plan.append(("query_resale_blocks_by_criteria", dict(
                towns=towns, flat_types=c.get("flat_types") or ["4 ROOM"],
                max_price=c.get("max_budget_sgd") or 5_000_000, min_lease_years=c.get("min_remaining_lease_years") or 0)))
        if c.get("wants_price_trend") and towns:
            plan.append(("compute_town_price_timeseries", dict(towns=towns, flat_type=(c.get("flat_types") or ["4 ROOM"])[0],
                                                               lookback_years=c.get("trend_years") or 3)))
        plan.extend(addons.plan(agent, c, query))
        if c.get("analysis_mode") and addons.active():
            plan.extend(ri.analysis_fallback(c, query))
        if c.get("wants_bto_comparison") and not c.get("bto_project_name"):
            m = re.search(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\s+(?:BTO|launch)", query)
            plan.append(("compare_bto_launch_vs_resale", dict(bto_project_name=m.group(1) if m else (towns[0] if towns else "Recent BTO"), radius_meters=500)))
    elif agent == "spatial":
        if c.get("nearby_schools"):
            return addons.plan(agent, c, query)   # schools near an address: that lookup alone (schools pack)
        radii = {k: v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"}
        if radii:
            plan.append(("filter_blocks_by_amenity_proximity", dict(towns=towns, amenity_constraints=radii)))
        if c.get("school_name"):
            r = (c.get("amenity_radii_meters") or {}).get("school", 1000)
            bracket = "1-2km" if re.search(r"\b1\s*[-–]\s*2\s*km\b", query, re.I) else ("1km" if r <= 1000 else "2km")
            plan.append(("verify_primary_school_distance", dict(school_name=c["school_name"], bracket=bracket)))
        plan.extend(step for step in addons.plan(agent, c, query) if step[0] != "find_primary_schools_near_address")
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


REPEATABLE_TOOLS = ("get_block_transaction_history", "get_nearby_amenities")
SPATIAL_BLOCK_TOOLS = ("filter_blocks_by_amenity_proximity", "verify_primary_school_distance", "compute_dual_transit_commute")


def _store(outputs: Dict[str, Any], name: str, result: Any) -> None:
    """Repeatable research tools keep every call (one per address); others keep the latest."""
    if name in REPEATABLE_TOOLS and name in outputs:
        previous = outputs[name] if isinstance(outputs[name], list) else [outputs[name]]
        outputs[name] = previous + [result]
    else:
        outputs[name] = result


def _commute_candidates(outputs: Dict[str, Any]) -> List[str]:
    """Commute only the blocks the other spatial filters kept, instead of an arbitrary index slice."""
    blocks = slot_outputs({k: v for k, v in outputs.items() if k in SPATIAL_BLOCK_TOOLS[:2]}, []).get("spatial_block_candidates") or []
    return [f"Blk {b['block']} {b['street']}" for b in blocks]


def run_react(label: str, tool_names: List[str], system_prompt: str, query: str, c: Dict[str, Any],
              max_iter: Optional[int] = None, plan_agents: Optional[List[str]] = None,
              known_points: Optional[List[Tuple[float, float]]] = None,
              use_model: bool = True, stop_when_covered: bool = False,
              ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], str]:
    """Generic ReAct loop. Returns (outputs_by_tool, tool_call_log, errors, final_text).
    label: logged as the calling agent. plan_agents: which deterministic plans to use as fallback.
    known_points: verified coordinates (e.g. the user's map pin) that model tool calls may use.
    use_model=False runs the deterministic plan directly. stop_when_covered ends the loop once
    every planned tool has run, skipping the model's closing summary (app mode)."""
    cfg = load_config()["architecture"]
    max_iter = max_iter or cfg["subagent_max_tool_iterations"]
    runners: Dict[str, ToolRunner] = build_runners(tool_names)
    for r in runners.values():
        r.agent = label
    outputs: Dict[str, Any] = {}
    errors: List[str] = []
    final_text = ""
    if c.get("nearby_schools") and "find_primary_schools_near_address" in runners:
        runner = runners["find_primary_schools_near_address"]
        outputs[runner.name] = runner(**c["nearby_schools"])
        return outputs, [r.model_dump() for r in runner.log], errors, final_text
    llm = get_llm() if use_model else None
    used_llm_tools = False
    extra_log: List[Dict[str, Any]] = []
    planned = {name for agent in (plan_agents or [label]) for name, _ in deterministic_plan(agent, c, query)}

    if llm is not None:
        try:
            tools = [r.as_langchain_tool() for r in runners.values()]
            bound = llm.bind_tools(tools)
            # an agent with one tool (the policy agent) exists to call it: its first turn must be that call, with the
            # model writing the arguments. In the 31 Oct live run B's and C's policy agents answered from memory
            # instead in 15 of 15 questions, and code had to run the search.
            first = bound
            if len(tools) == 1:
                try:
                    first = llm.bind_tools(tools, tool_choice=tools[0].name)
                except TypeError:              # a test double without tool_choice
                    first = bound
            msgs: List[Any] = [sys_msg(system_prompt),
                               HumanMessage(content=f"User query: {query}\n\nExtracted constraints (JSON): {json.dumps(constraints_for_model(c))}")]
            for turn in range(max_iter):
                ai: AIMessage = (first if turn == 0 else bound).invoke(msgs)
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
                    known = list(known_points or []) + list(ri.coordinates_in(outputs)) + list(ri.coordinates_in(c.get("comparable_resale") or {}))
                    rejection = ri.unverified_coordinates(name, args, known)
                    if rejection:
                        errors.append(f"{label}: rejected unverified coordinates for {name}")
                        msgs.append(ToolMessage(content=json.dumps({"error": rejection}), tool_call_id=tc["id"]))
                        continue
                    if name == "compute_dual_transit_commute" and not args.get("candidate_blocks"):
                        args = {**args, "candidate_blocks": _commute_candidates(outputs)}
                    guarded = guard_financial_args(name, args, c, outputs, runners)   # may run the grant calculator (code)
                    runners[name].source = "model"
                    try:
                        result = runners[name](**guarded)
                    finally:
                        runners[name].source = "code"
                    used_llm_tools = True
                    _store(outputs, name, result)
                    msgs.append(ToolMessage(content=ri.compact_tool_result(result), tool_call_id=tc["id"]))
                if stop_when_covered and planned and planned <= set(outputs):
                    break  # the evidence is in; the writer, not this agent, phrases the answer
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
                    if name == "compute_dual_transit_commute" and not args.get("candidate_blocks"):
                        args = {**args, "candidate_blocks": _commute_candidates(outputs)}
                    _store(outputs, name, runners[name](**args))
    else:
        # Coverage backstop: research evidence the question needs but the model did not fetch.
        # Logged as the supervisor so Tool Selection F1 still measures only the model's own calls.
        for agent in (plan_agents or [label]):
            for name, args in addons.plan(agent, c, query):
                if name in runners and name not in outputs:
                    backstop = ToolRunner(name)
                    backstop.agent = "supervisor"
                    _store(outputs, name, backstop(**args))
                    extra_log.extend(rec.model_dump() for rec in backstop.log)
                    errors.append(f"{label}: added {name} the question needs (coverage check)")

    log = [rec.model_dump() for r in runners.values() for rec in r.log] + extra_log
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
    # A stamp-duty / upfront-cost question that states no grant or budget fact runs only the upfront calculator: the
    # grant and affordability calculators would only answer with a long list of facts nobody asked about.
    upfront_only = (upfront_name in runners and asked_for(upfront_name, constraints, query)
                    and upfront_args.get("purchase_price") and not grant_args and not aff_args)
    if grant_name in runners and not upfront_only:
        grants = outputs[grant_name] = runners[grant_name](**grant_args)
        if grants.get("status") == "ok":
            for target in (aff_args, upfront_args):
                if target.get("total_grants") is None:
                    target["total_grants"] = grants["total_grants"]
    if aff_name in runners and not upfront_only:
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


# Lowest raw BM25 score at which the best policy passage is shown. Every Category C question scores 15.99 or more
# (tests/test_scope_routing.py checks it); questions with no policy in them score 0 to 4.8 ("How do I buy a condo?"
# 4.6, "best durian in Singapore" 4.8), so 5.0 sits between the two.
POLICY_SCORE_FLOOR = 5.0
NO_POLICY_ANSWER = ("I couldn't find an HDB rule in my knowledge base that answers this. Please check hdb.gov.sg or ask "
                    "HDB for an HFE letter.")


def policy_below_floor(out: Any) -> bool:
    score = out.get("best_keyword_score") if isinstance(out, dict) else None
    return score is not None and score < POLICY_SCORE_FLOOR


def slot_outputs(outputs: Dict[str, Any], towns: List[str]) -> Dict[str, Any]:
    """Map raw tool outputs into the HDBCompassState slots, by tool owner."""
    update: Dict[str, Any] = {}
    sql, fin = {}, {}
    spatial: List[Dict[str, Any]] = []
    for name, out in outputs.items():
        owner = TOOL_OWNER[name]
        if name == "find_primary_schools_near_address":
            update["nearby_school_results"] = out
            continue
        if name in ri.LOCATION_TOOLS:
            update.setdefault("location_results", {})[name] = out
            continue
        if isinstance(out, dict) and "error" in out:
            continue
        if owner == "market_sql":
            sql[name] = out
        elif owner == "financial":
            fin[name] = out
        elif owner == "policy_rag":
            update["rag_policy_passages"] = out.get("passages", []) if isinstance(out, dict) else out
            if policy_below_floor(out):   # nothing in the knowledge base really matches: show no passages at all
                update["rag_policy_passages"] = []
                update["policy_no_match"] = True
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
    if any(n in SPATIAL_BLOCK_TOOLS for n in outputs):
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
    if c.get("analysis_mode") and agent == "market_sql":
        asked |= set(ri.ANALYSIS_TOOLS)     # whatever the model chose from the analysis tools is the evidence
    return {name: out for name, out in outputs.items() if name in asked or name == "sync_interactive_map_ui"}


def make_specialist_node(agent: str) -> Callable[[HDBCompassState], Dict[str, Any]]:
    def node(state: HDBCompassState) -> Dict[str, Any]:
        c = state.get("user_constraints", {})
        query = last_user_query(state)
        pin = (state.get("map_ui_payload") or {}).get("center")
        known = [tuple(pin)] if isinstance(pin, (list, tuple)) and len(pin) == 2 else []
        # only the tools this task needs (needed_for_task); Variants A and B keep the full list, so their models'
        # unasked calls still show in Tool F1
        visible = addons.visible_tools(tools_for_agent(agent), state.get("variant"), bool(state.get("fast_path")))
        tools = [name for name in visible if needed_for_task(name, c, query)]
        system = AGENT_SYSTEM[agent] if addons.offered_in(state.get("variant"), bool(state.get("fast_path"))) \
            else CORE_AGENT_SYSTEM[agent]
        analysis = bool(c.get("analysis_mode")) and agent == "market_sql" and addons.offered_in(state.get("variant"), bool(state.get("fast_path")))
        if analysis:
            system = system + "\n\n" + addons.analysis_section(agent)
        outputs, log, errors, final = run_react(agent, tools, system, query, c,
                                                max_iter=ri.ANALYSIS_MAX_ITERATIONS if analysis else None,
                                                known_points=known, use_model=c.get("extraction_source") != FAST_PATH_SOURCE,
                                                stop_when_covered=bool(state.get("fast_path")))
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
                         price_cap: Optional[float], prices_checkable: bool = True) -> List[Dict[str, Any]]:
    """Blocks meeting every limit. Under a price cap a block with no price evidence is dropped (it has not been shown
    to be within budget), unless prices could not be checked at all (the price search is unavailable): then the
    blocks stay, and the answer says the price limit was not checked (not_checked_lines)."""
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
        result = [b for b in result if (b.get("median_resale_price") is None and not prices_checkable)
                  or (b.get("median_resale_price") is not None and b["median_resale_price"] <= price_cap)]
    return result


def _price_spatial_blocks(state: HDBCompassState, log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Join resale prices onto spatial-only results (school, commute) when price or lease limits were stated."""
    c = state.get("user_constraints", {})
    spatial = state.get("spatial_block_candidates")
    if _sql_blocks(state) is not None or not spatial or not (
            c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("flat_types")):
        return None
    runner = ToolRunner("query_resale_blocks_by_criteria")
    runner.agent = "supervisor"
    out = runner(towns=c.get("towns") or [], flat_types=c.get("flat_types") or [],
                 max_price=c.get("max_budget_sgd") or 5_000_000, min_lease_years=c.get("min_remaining_lease_years") or 0,
                 block_keys=[block_key(b["block"], b["street"]) for b in spatial])
    log.extend(rec.model_dump() for rec in runner.log)
    return out if isinstance(out, dict) and "blocks" in out else None


def intersect_node(state: HDBCompassState) -> Dict[str, Any]:
    cap = _affordable_price(state)
    budget = state.get("user_constraints", {}).get("max_budget_sgd")
    if budget is not None:
        aff = (state.get("financial_breakdown") or {}).get("calculate_maximum_affordability") or {}
        if not aff.get("rule_version"):
            cap = None  # Preserve the user-budget semantics of legacy mock benchmarks.
        cap = min(cap, budget) if cap is not None else budget
    log: List[Dict[str, Any]] = []
    update: Dict[str, Any] = {}
    priced = _price_spatial_blocks(state, log)
    if priced is not None:
        update["sql_market_results"] = {**(state.get("sql_market_results") or {}), "query_resale_blocks_by_criteria": priced}
        state = {**state, **update}
    checkable = "query_resale_blocks_by_criteria" not in unavailable_tools({**state, "tool_call_log": (state.get("tool_call_log") or []) + log})
    update["candidate_blocks_intersection"] = compute_intersection(_sql_blocks(state), state.get("spatial_block_candidates"),
                                                                   cap, prices_checkable=checkable)
    update["tool_call_log"] = log
    return update


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
    if state.get("nearby_school_results") is not None:
        return present_school_result(state["nearby_school_results"])[0]
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
                kind = f" {b['flat_type']}" if b.get("flat_type") and len({x.get("flat_type") for x in blocks}) > 1 else ""
                town = f" ({b['town']})" if b.get("town") else ""
                lines.append(f"- Blk {b['block']} {b['street']}{town}{kind}{price}{lease}{(' — ' + dist) if dist else ''}")
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
    lines.extend(ri.research_lines(state))
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
            span = (f", {t['period_change_pct']:+.1f}% since {t['period_start_quarter']} ({t['period_cagr_pct']:+.1f}%/yr)"
                    if t.get("period_change_pct") is not None else "")
            lines.append(f"- {t['town']}: {period} median ${t['latest_median_price']:,.0f}, ${t['median_price_per_sqm']:,.0f}/sqm, {yoy}{span}")
        metadata = trend.get("metadata") or {}
        for town in metadata.get("towns_without_data", []):
            lines.append(f"- {town}: no transactions for the requested flat type in the trend window.")
        for town in metadata.get("towns_without_complete_quarters", []):
            lines.append(f"- {town}: only incomplete quarters are available; no complete-quarter summary or YoY comparison.")
    unbuilt = unbuilt_tools_used(state)
    bto_result = sql.get("compare_bto_launch_vs_resale")
    if bto_result is not None and "compare_bto_launch_vs_resale" in unbuilt:
        lines.append("- BTO launch prices can't be compared in this version yet.")
    elif isinstance(bto_result, dict) and bto_result.get("status", "ok") == "ok" and "error" not in bto_result:
        b = sql["compare_bto_launch_vs_resale"]
        if b.get("classification"):
            lines.append(f"**BTO {b['bto_project_name']}** ({b['town'].title()}, {b['classification']}, {b['launch_month']} launch, "
                         f"{b['flat_type']}, ~{b['waiting_time_months']} months wait): ${b['bto_indicative_price_range']['min']:,.0f}–"
                         f"${b['bto_indicative_price_range']['max']:,.0f} before grants [{b['source_url']}]")
            if b.get("resale_median_nearby"):
                lines.append(f"- {b['resale_basis']}: ${b['resale_median_nearby']:,.0f} ({b['resale_sample_size']} sales); "
                             f"BTO midpoint is {b['price_gap_pct']:+.1f}% vs that median.")
            for note in b.get("notes", [])[1:]:
                lines.append(f"- {note}")
        else:
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
                if name == "calculate_bsd_and_upfront_outlay" and value.get("buyer_stamp_duty") is not None:
                    # stamp duty needs only the price, so it is shown even when the full funding estimate cannot run
                    lines.append(f"- Buyer's stamp duty: **${value['buyer_stamp_duty']:,.0f}** (on the purchase price; "
                                 "charged on the valuation instead if that is higher)")
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
    elif state.get("policy_no_match"):
        lines.append(NO_POLICY_ANSWER)
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
RESEARCH_RULE = (
    "No block search was run, so do not discuss matching blocks or relaxation. Answer the user's actual question "
    "directly from market_context, location_results, financial_breakdown and rag_policy_passages: lead with the "
    "answer, then the supporting figures with their time window. Use tables for comparisons. State every entry in "
    "user_constraints.assumptions, and ask every question in user_constraints.clarifications instead of guessing. "
    "If a tool returned ambiguous or not_found, ask the user which place they mean, listing the candidates.")
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
NOT_CHECKED.update(addons.not_checked())   # add-on tools say what they could not check, too
BLOCK_FILTERS = ("query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity",
                 "verify_primary_school_distance", "compute_dual_transit_commute")


def unavailable_tools(state: HDBCompassState) -> List[str]:
    """Tools this answer called whose module could not be loaded (see NOT_CHECKED)."""
    return sorted({t.get("tool") for t in state.get("tool_call_log") or []
                   if str(t.get("error") or "").startswith("unavailable:") and t.get("tool") in NOT_CHECKED})


RESEARCH_TOOLS = frozenset(ri.MARKET_RESEARCH_TOOLS + ri.LOCATION_TOOLS + ri.AREA_TOOLS)


def has_research_evidence(state: HDBCompassState) -> bool:
    """Whether the research tools (research_intents) produced anything for this answer."""
    sql = state.get("sql_market_results") or {}
    return bool(state.get("location_results")) or any(name in sql for name in RESEARCH_TOOLS)


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
            "rag_policy_passages": [] if "retrieve_hdb_policy_knowledge" in unbuilt else state.get("rag_policy_passages"),
            "location_results": state.get("location_results")}
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
    if sql_rows is None and spatial is None and has_research_evidence(state):
        # a research question (market snapshot, block history, nearby amenities, BTO...): no block search ran
        return ({**base, "market_context": market_context}, RESEARCH_RULE)
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


def says_something(opening: str) -> bool:
    """False for an opening left with only lead-ins whose list was dropped ("The options are:", "**Trade-offs:**")."""
    return any(not s.strip().strip("*").rstrip().endswith(":") for s in _split_opening(opening))


# When nothing of the model's opening survives (in the 31 Oct live runs Qwen3 wrote figures in all 90 C openings, and
# in 26 of them every sentence was dropped), or there is no model, code writes the opening from the same state the
# facts come from: one sentence that answers the question, and, when several blocks are listed, which of them is the
# cheapest (by median price), has the most lease left and is nearest the MRT (or the amenity the user limited). A role
# is named only when one block holds it at the precision the facts show (three blocks all showing "73 yrs lease" have
# no longest lease), and only among the blocks the facts list. The opening names blocks and nothing else, so it passes
# the same figure check as the model's (code_opening drops any sentence that would not), and its trade-off passes the
# same comparison check (comparisons_hold). In a block-search answer code's answer sentence goes first even when some
# of the model's opening survives: in the 31 Oct runs what survived was often only a note on taxes or a comparison the
# facts contradict, with no answer at all (see check_model_opening).
# Only C (and the app) gets it: its facts are code-written. B and A keep their own writer and plain template fallback.
def _sole_best(blocks: List[Dict[str, Any]], value: Callable[[Dict[str, Any]], Any], lowest: bool) -> Optional[Dict[str, Any]]:
    """The one block with the lowest (or highest) value as the facts write it, or None when a block has no value or
    two blocks share the best one."""
    shown = [value(b) for b in blocks]
    if len(blocks) < 2 or any(v is None for v in shown):
        return None
    best = min(shown) if lowest else max(shown)
    return blocks[shown.index(best)] if shown.count(best) == 1 else None


def _join_words(parts: List[str], last: str = "and") -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" {last} " + parts[-1]


def trade_off_sentence(blocks: List[Dict[str, Any]], c: Dict[str, Any], more_than_listed: bool = False) -> str:
    """"Blk 228 is the cheapest, while Blk 288B has the most lease left and is nearest an MRT station" ("cheaper",
    "more", "nearer" between two blocks); cheapest is by median price, as the facts list it. Each role is held by
    exactly one listed block; a block holding several is named once. "" when no role is clear."""
    def shown(v: Any) -> Optional[float]:
        return None if v is None else float(f"{float(v):.0f}")     # the facts show prices, leases and distances whole

    two = len(blocks) == 2
    limits = list((c or {}).get("amenity_radii_meters") or {})
    amenity = next((k for k in ["mrt"] + limits if blocks and all(f"{k}_dist_m" in (b.get("distances") or {})
                                                                  for b in blocks)), None)
    # "cheaper", not "the lower median price": the evaluation reads "lower ... price" as a relaxation offered
    roles = [(_sole_best(blocks, lambda b: shown(b.get("median_resale_price")), True),
              "is", "cheaper" if two else "the cheapest"),
             (_sole_best(blocks, lambda b: shown(b.get("median_remaining_lease_years")), False),
              "has", f"{'more' if two else 'the most'} lease left")]
    if amenity:
        place = AMENITY_LABELS.get(amenity, "the " + amenity.replace("_", " "))
        roles.append((_sole_best(blocks, lambda b: shown((b.get("distances") or {}).get(f"{amenity}_dist_m")), True),
                      "is", f"{'nearer' if two else 'nearest'} {place}"))
    numbers = [str(b["block"]).upper() for b in blocks]
    by_block: Dict[int, List[Tuple[str, str]]] = {}
    for block, verb, what in roles:
        if block is not None:
            by_block.setdefault(blocks.index(block), []).append((verb, what))
    if not by_block:
        return ""
    clauses = []
    for i, held in by_block.items():
        b = blocks[i]
        # the street too when two listed blocks share a number (Blk 101 in two towns)
        name = f"Blk {b['block']}" + (f" {b['street']}" if numbers.count(numbers[i]) > 1 and b.get("street") else "")
        # "is the cheapest and nearest an MRT station": a verb is said once for the roles that share it
        phrases = [f"{verb} {_join_words([w for v, w in held if v == verb])}"
                   for verb in dict.fromkeys(v for v, _ in held)]
        both = len(phrases) == 2 and " and " in phrases[0]
        clauses.append(f"{name} {phrases[0]}, and {phrases[1]}" if both else f"{name} {_join_words(phrases)}")
    sentence = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + ", while " + clauses[-1]
    return ("Of the listed blocks, " if more_than_listed else "") + sentence + "."


# What each calculator works out, in the words of an opening ("your housing grants follow the list"), and whether
# the words take a plural verb.
ESTIMATE_WORDS = {"calculate_cpf_housing_grants": ("housing grants", True),
                  "calculate_maximum_affordability": ("maximum purchase price", False),
                  "calculate_bsd_and_upfront_outlay": ("upfront costs", True)}
ESTIMATE_SHORT = {"calculate_cpf_housing_grants": "grant", "calculate_maximum_affordability": "budget",
                  "calculate_bsd_and_upfront_outlay": "upfront cost"}


def stated_limits(c: Dict[str, Any]) -> bool:
    """Whether the user set any limit the blocks were checked against (budget, lease, distance, school, commute)."""
    return bool(c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("amenity_radii_meters")
                or c.get("school_name") or c.get("max_commute_minutes") or len(c.get("work_destinations") or []) >= 2)


def unlimited_search_phrase(c: Dict[str, Any]) -> str:
    """What a block list is when the user set no limit: "Blocks with recent 4-room sales in Tampines, cheapest first"."""
    flat = " or ".join(t.lower().replace(" room", "-room") for t in c.get("flat_types") or [])
    towns = _join_words([t.title() for t in c.get("towns") or []], "and") if c.get("towns") else ""
    return "Blocks with recent " + (f"{flat} " if flat else "") + "sales" + (f" in {towns}" if towns else "") + ", cheapest first"


def code_opening_parts(state: HDBCompassState, facts: str) -> Tuple[List[str], str, List[str]]:
    """Code's opening in two parts: the sentences that answer the question, and the trade-off sentence between the
    listed blocks ("" when there is none). Built from the same state as the facts (template_response), so it never
    says what they do not: an estimate is said to follow only when its calculator worked it out, and a summary only
    when the tool found one. Returns (answer sentences, trade-off, errors); a sentence that would not pass
    keep_supported_sentences, or a trade-off that comparisons_hold does not bear out, is left out and reported."""
    c = state.get("user_constraints") or {}
    blocks = state.get("candidate_blocks_intersection") or []
    searched = blocks or _sql_blocks(state) is not None or state.get("spatial_block_candidates") is not None
    partial = any(t in BLOCK_FILTERS for t in unavailable_tools(state))
    limits = "the limits that could be checked" if partial else "every limit you set"
    fin = state.get("financial_breakdown") or {}
    fin_ok = [n for n, v in fin.items() if isinstance(v, dict) and "error" not in v and v.get("status", "ok") == "ok"]
    done = [ESTIMATE_WORDS.get(n, ("estimate", False)) for n in fin_ok]
    clar = clarification_needed(state)
    unbuilt = unbuilt_tools_used(state)
    policy_shown = ("retrieve_hdb_policy_knowledge" not in unbuilt
                    and bool(shown_policy_passages(state.get("rag_policy_passages") or [], user_request(state))))
    sentences: List[str] = []
    trade = ""
    if searched and blocks:
        # what follows the list: only what was worked out ("your housing grants", not "your grant and budget
        # estimate" when the budget still needs the cash savings), with the verb its words take
        # (two or more worked out read as one estimate: "your grant, budget and upfront cost estimate follows")
        short = [ESTIMATE_SHORT.get(n, "financial") for n in fin_ok]
        estimate = ((f"your {_join_words(short)} estimate", False) if len(done) > 1
                    else (f"your {done[0][0]}", done[0][1]) if done else None)
        after = ([estimate] if estimate else []) + ([("the relevant HDB rules", True)] if policy_shown else [])
        plural = len(after) > 1 or (after and after[0][1])
        follows = (f", and {_join_words([w for w, _ in after])} follow{'' if plural else 's'} the list" if after else "")
        if clar:
            follows += (f"; {'the rest of y' if done else 'y'}our grant and budget estimate needs a few more facts, "
                        "asked for at the end")
        elif len(fin_ok) < len(fin):
            follows += f"; {'part of y' if done else 'y'}our estimate could not be worked out, as said below"
        if c and not stated_limits(c):   # (a state with no constraints at all is test data, not a real run)
            # nothing was asked of the blocks, so "they meet every limit you set" would promise a check that never happened
            sentences.append(f"{unlimited_search_phrase(c)}{follows}.")
        elif len(blocks) == 1:
            sentences.append(f"Only Blk {blocks[0]['block']} meets {limits}{follows}.")
        elif len(blocks) == 2:
            sentences.append(f"Both blocks below meet {limits}{follows}.")
        else:
            sentences.append(f"All the blocks below meet {limits}{follows}.")
        listed = blocks[:LISTED_BLOCKS]
        if len(listed) > 1:
            trade = trade_off_sentence(listed, c, more_than_listed=len(blocks) > LISTED_BLOCKS)
    elif searched:
        sentences.append("No block meets all the limits that could be checked." if partial
                         else "No block meets every limit you set at once." if stated_limits(c) or not c
                         else "I found no blocks with recent sales for that search.")
        rel = state.get("relaxation") or {}
        best = rel.get("best_single_change") or {}
        top = [s for s in rel.get("suggestions") or [] if best and s.get("unlocked_blocks") == best.get("unlocked_blocks")]
        if top:
            names = list(dict.fromkeys(f"the {plain_relaxation(s['constraint'], s['original'], s['relaxed'])[0]}"
                                       for s in top))
            sentences.append(f"Relaxing {'either ' if len(names) == 2 else ''}{_join_words(names, 'or')} unlocks the "
                             "most blocks; each option is below.")
        elif rel.get("suggestions"):
            sentences.append("Relaxing any one limit on its own unlocks no block either, as shown below.")
    else:
        # no flat search: policy first (the question was about the rules), then what the calculators could do
        if policy_shown:
            sentences.append("The HDB policy passages that bear on your question are below.")
        elif state.get("rag_policy_passages") and "retrieve_hdb_policy_knowledge" not in unbuilt:
            sentences.append("The policy search found no HDB passage that answers this directly.")
        if done:
            words = [w for w, _ in done]
            sentences.append(f"Here {'is' if len(done) == 1 and not done[0][1] else 'are'} your {_join_words(words)}, "
                             "worked out from the facts you gave.")
        if clar:
            sentences.append(f"{'The rest of y' if done else 'Y'}our grant and budget estimate needs a few more facts, "
                             "asked for at the end.")
        elif fin and not done:
            sentences.append("Your estimate could not be worked out in full; what is missing is said below.")
        sql = state.get("sql_market_results") or {}
        stats = sql.get("query_resale_statistics")
        if isinstance(stats, dict) and not sentences and stats.get("status") != "ok":
            # a working breakdown needs no lead-in: its title and headline (the direct answer) open the facts
            sentences.append("That breakdown could not be worked out; the reason is below.")
        # a lead promises a summary only when the tool found one; otherwise it points to the reason the facts give
        if not sentences and "compute_town_price_timeseries" in sql:
            trend = sql["compute_town_price_timeseries"] or {}
            sentences.append("The latest town price trends for your search are below." if trend.get("summary")
                             else "No recent town price trend could be found for your search; the reason is below.")
        elif not sentences and sql.get("query_comparable_resale_transactions"):
            sentences.append("Recent comparable resale sales for your search are summarised below."
                             if sql["query_comparable_resale_transactions"].get("summary")
                             else "No comparable resale sales could be summarised for your search; the reason is below.")
    kept, errors = [], []
    for s in filter(None, sentences):
        if keep_supported_sentences(s, facts)[1]:
            errors.append(f"synthesize: code opening sentence left out, it failed the figure check: {s[:80]}")
        else:
            kept.append(s)
    if trade:
        failed = ("figure" if keep_supported_sentences(trade, facts)[1]
                  else "" if comparisons_hold(trade, blocks[:LISTED_BLOCKS]) else "comparison")
        if failed:
            errors.append(f"synthesize: code opening sentence left out, it failed the {failed} check: {trade[:80]}")
            trade = ""
    return kept, trade, errors


def code_opening(state: HDBCompassState, facts: str) -> Tuple[str, List[str]]:
    """The opening code writes when there is no model, or none of the model's opening survives: a direct answer, and
    the trade-off between the listed blocks (code_opening_parts). Returns (opening, errors)."""
    answer, trade, errors = code_opening_parts(state, facts)
    return " ".join(answer + ([trade] if trade else [])), errors


# The model's comparisons between listed blocks are checked too: a sentence passes the figure check without a number,
# but "Blk 201E is marginally closer to the MRT" was false in the 31 Oct live runs (355 m against 346 m), as were three
# more such claims, and each was kept while code's own trade-off had the roles right. A comparison is about price
# (cheaper, more expensive), lease (more lease left, shorter lease) or distance (nearer, farther from a named place).
# Each is checked against the listed blocks at the precision the facts show; a sentence holding one that the facts do
# not bear out, or that cannot be checked (no block named, a block list, "not cheaper", "cheaper" among three blocks
# with no "than"), is dropped.
_COMPARISONS = [
    ("price", True, r"cheap(?:er|est)|(?:less|least) expensive|(?:more|most) affordable|low(?:er|est)[- ]priced"
                    r"|low(?:er|est) (?:median )?(?:resale )?price"),
    ("price", False, r"(?:more|most) expensive|pric(?:ier|iest)|costl(?:ier|iest)|high(?:er|est)[- ]priced"
                     r"|high(?:er|est) (?:median )?(?:resale )?price"),
    ("lease", True, r"(?:long(?:er|est)|more|most) (?:of (?:its|the) )?(?:remaining )?lease|new(?:er|est)"),
    ("lease", False, r"(?:short(?:er|est)|less|least) (?:of (?:its|the) )?(?:remaining )?lease|old(?:er|est)"),
    ("distance", True, r"clos(?:er|est)|near(?:er|est)"),
    ("distance", False, r"farth(?:er|est)|furth(?:er|est)(?= (?:from|away|to)\b)|more distant"),
]
_COMPARISON = re.compile("|".join(f"(?P<c{i}>\\b(?:{p})\\b)" for i, (_, _, p) in enumerate(_COMPARISONS)), re.I)
# the place a distance is to, by the words a sentence uses for it (the keys of AMENITY_LABELS)
_PLACES = {"mrt": r"\bMRT\b|\bstations?\b|\btrains?\b", "hawker": r"\bhawker|\bfood cent(?:re|er)",
           "bus_stop": r"\bbus\b", "supermarket": r"\bsupermarket|\bgrocer", "park": r"\bparks?\b",
           "school": r"\bschools?\b", "clinic": r"\b(?:poly)?clinics?\b"}
_CLAUSE = re.compile(r"[,;:()]|\s(?:while|whereas|but|though|although|yet)\s|\s[-–—]\s", re.I)


def _named_blocks(text: str, listed: List[Dict[str, Any]]) -> List[Tuple[int, Optional[int]]]:
    """(position, index into listed) for each block the text names, in order; the index is None when the number is not
    one listed block (not listed, or two listed blocks share it and the text gives neither street)."""
    found: Dict[str, int] = {}
    for m in list(_BLOCK_LIST.finditer(text)) + list(_ADDRESS.finditer(text)):
        found.setdefault(m.group(1).upper(), m.start())
        for extra in re.finditer(r"(\d{1,4}[A-Z]?)\b", m.group(2) or "") if m.re is _BLOCK_LIST else []:
            found.setdefault(extra.group(1).upper(), m.start(2) + extra.start())
    out = []
    for number, at in sorted(found.items(), key=lambda kv: kv[1]):
        same = [i for i, b in enumerate(listed) if str(b.get("block", "")).upper() == number]
        if len(same) > 1:
            same = [i for i in same if str(listed[i].get("street") or "").upper() in text.upper()]
        out.append((at, same[0] if len(same) == 1 else None))
    return out


def comparisons_hold(sentence: str, listed: List[Dict[str, Any]]) -> bool:
    """Whether every price, lease and distance comparison the sentence makes between the listed blocks is true of
    them as the facts show them (whole dollars, years and metres). True for a sentence that compares nothing."""
    if not _COMPARISON.search(sentence):
        return True
    if any((m.group(2) or "").strip() for m in _BLOCK_LIST.finditer(sentence)):
        return False      # "Blk 228, 266 and 268 are cheaper": a list split over clauses cannot be read block by block
    def shown(b: Dict[str, Any], what: str, place: str = "") -> Optional[float]:
        v = (b.get("median_resale_price") if what == "price" else b.get("median_remaining_lease_years") if what == "lease"
             else (b.get("distances") or {}).get(f"{place}_dist_m"))
        return None if v is None else float(f"{float(v):.0f}")

    limited = {k[:-len("_dist_m")] for b in listed for k in (b.get("distances") or {}) if k.endswith("_dist_m")}
    subject: Optional[int] = None
    for clause in _CLAUSE.split(sentence):
        names = _named_blocks(clause, listed)
        than = re.search(r"\bthan\b", clause, re.I)
        own = [i for at, i in names if not than or at < than.start()]
        refs = [i for at, i in names if than and at > than.start()]
        if own:
            subject = own[0] if len(own) == 1 else None
        found = [m for m in _COMPARISON.finditer(clause)]
        if not found:
            continue
        if subject is None or None in refs or re.search(r"\bnot\b|n't\b|\bno\b", clause, re.I):
            return False
        for m in found:
            what, better, _ = _COMPARISONS[int(m.lastgroup[1:])]
            most = bool(re.search(r"est\b|^(?:most|least)\b", m.group(0), re.I))
            others = refs or ([i for i in range(len(listed)) if i != subject] if most or len(listed) == 2 else [])
            # a place named by its own words when it has none above ("the community club")
            places = ([k for k, p in _PLACES.items() if re.search(p, clause, re.I)]
                      + [k for k in limited if k not in _PLACES and k.replace("_", " ") in clause.lower()]
                      if what == "distance" else [""])
            if not others or not places:
                return False
            for place in places:
                mine, theirs = shown(listed[subject], what, place), [shown(listed[i], what, place) for i in others]
                if mine is None or None in theirs:
                    return False
                lower_is_better = what != "lease"
                wins = all((mine < t) if lower_is_better == better else (mine > t) for t in theirs)
                if not wins:
                    return False
    return True


# Other sentences of the model's opening an answer leaves out. Notes the facts already carry (taxes, fees, the HFE
# letter): "Ensure you account for additional costs like taxes and fees." was a whole kept opening in the 31 Oct runs,
# with no answer in it; they are kept only in an answer about policy without a block search, where the HFE letter may
# be what was asked. In a block-search answer, where code's own answer goes first, also: restating whether the blocks
# meet the limits ("No block satisfies all the constraints at once."), and a sentence that leans on the one before it
# ("This increases the budget requirement ..."): code's answer now stands before it, so "this" would point at what
# code wrote, and what the model meant (often a sentence dropped for its figures) is not shown.
_RESTATED_ANSWER = re.compile(r"^\W*(?:no|none|both|all|each|every|only|these|the (?:following|listed|above))\b[^.]*?"
                              r"\b(?:meet|meets|satisf\w*|match|matches|fit|fits|fulfil\w*)\b", re.I)
_NOTE = re.compile(r"\b(?:tax(?:es)?|fees|cash over valuation|HFE|conditional estimate|indicative|financial advice"
                   r"|upfront funding)\b", re.I)
_LEANS_BACK = re.compile(r"^\W*(?:this|that|these|those|it|they|its|their|such|however|but|also|otherwise|instead)\b", re.I)


def check_model_opening(text: str, state: HDBCompassState, facts: str,
                        policy: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[str], Dict[str, int]]:
    """The sentences of the model's opening that may be shown, and how many were dropped for each reason. Every
    answer: keep_supported_sentences (figures, unlisted blocks) and bare lead-ins whose list is gone ("The options
    are:"), and _NOTE unless the answer is about policy without a block search. A block-search answer also:
    comparisons_hold, _RESTATED_ANSWER and _LEANS_BACK."""
    blocks = state.get("candidate_blocks_intersection") or []
    searched = bool(blocks) or _sql_blocks(state) is not None or state.get("spatial_block_candidates") is not None
    notes_are_facts = searched or not state.get("rag_policy_passages")
    kept: List[str] = []
    dropped = {"figures": 0, "comparisons": 0, "notes": 0, "leans_back": 0, "restated": 0, "lead_ins": 0}
    for sentence in _split_opening(text):
        reason = ""
        if keep_supported_sentences(sentence, facts, user_request(state), policy)[1]:
            reason = "figures"
        elif not says_something(sentence):
            reason = "lead_ins"
        elif searched and _LEANS_BACK.match(sentence):
            reason = "leans_back"
        elif searched and not comparisons_hold(sentence, blocks[:LISTED_BLOCKS]):
            reason = "comparisons"
        elif notes_are_facts and _NOTE.search(sentence):
            reason = "notes"
        elif searched and _RESTATED_ANSWER.match(sentence):
            reason = "restated"
        if reason:
            dropped[reason] += 1
        else:
            kept.append(sentence)
    return kept, dropped


# A short figure-free example of the opening for the synthesis prompt: it shows the model how to compare in words (the
# 31 Oct live openings were nearly all figures, and all of those sentences were deleted). The example says nothing about
# this answer's blocks or limits: the block numbers are placeholders no answer lists, so an example copied as it is
# gets dropped too. What is true of this answer comes with it, as code's own opening (see synthesize_node). It starts at
# the trade-off: in a block-search answer code's own sentence on whether the blocks meet the limits goes first.
OPENING_EXAMPLE = ('"Blk 101 is cheaper, while Blk 202 has more lease left and is nearer the MRT station, so the choice '
                   'is between price and convenience."')


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


def place_clarification(state: HDBCompassState) -> Optional[Dict[str, Any]]:
    """A question about the place or school (research_intents) as this turn's clarification, so the next message is
    read as its answer (initial_state carries it in as pending_clarification). kind "location": pin or earlier place."""
    c = state.get("user_constraints") or {}
    questions = c.get("clarifications") or []
    if not questions:
        return None
    if c.get("place_options"):
        return {"kind": "location", "needed": [], "question": questions[0], "options": c["place_options"]}
    if c.get("menu_options"):
        return {"kind": "menu", "needed": [], "question": questions[0], "options": c["menu_options"]}
    kind = "school" if questions[0] == ri.SCHOOL_CLARIFICATION else "place"
    return {"kind": kind, "needed": [], "question": questions[0]}


def synthesize_node(state: HDBCompassState) -> Dict[str, Any]:
    if state.get("nearby_school_results") is not None:
        text, payload = present_school_result(state["nearby_school_results"])
        return {"draft_response": text, "map_ui_payload": payload}
    c = state.get("user_constraints", {})
    if c.get("analysis_mode") and not (state.get("sql_market_results") or state.get("location_results")):
        # a data question the model could not turn into a valid call and code has no overview for: the menu, as before
        menu = scope.menu_for(last_user_query(state))
        return {"draft_response": menu["question"], "map_ui_payload": {}, "tool_call_log": [], "errors": [],
                "clarification": {"kind": "menu", "needed": [], "question": menu["question"], "options": menu["options"]},
                "suggestions": []}
    blocks = state.get("candidate_blocks_intersection") or []
    log: List[Dict[str, Any]] = []
    payload = build_map_payload(blocks, c, log) if blocks else ri.research_map_payload(state)
    evidence, rule = synthesis_evidence(state)
    unchecked = (" Some tools could not run: say plainly what could_not_check lists, and never say the blocks meet "
                 "every constraint or give a figure those tools would have produced.") if evidence.get("could_not_check") else ""
    rule += unchecked
    clar = clarification_needed(state) or place_clarification(state)
    if clar and clar.get("kind", "financial") == "financial":
        evidence["clarification_needed"] = clar
        rule += (" Some financial facts are missing: state no maximum budget or affordability figure, and end with "
                 "clarification_needed.question.")
    verdict = state.get("auditor_verdict") or {}
    text = ""
    errors: List[str] = []
    # app fast path, or a turn that only asks the user something: code writes it all
    only_asking = bool(c.get("clarifications")) and not state.get("active_subagents")
    llm = get_llm() if c.get("extraction_source") != FAST_PATH_SOURCE and not only_asking else None
    facts = template_response(state) if writes_facts_in_code(state) else ""
    policy = policy_answer_passages(state) if facts else None
    # code's own opening (code_opening_parts). In a block-search answer its answer sentence always goes first and the
    # model's checked sentences follow; otherwise it is used when none of the model's survives, or there is no model.
    answer, trade, lead_errors = code_opening_parts(state, facts) if facts else ([], "", [])
    lead = " ".join(answer + ([trade] if trade else []))
    blocks_listed = blocks[:LISTED_BLOCKS]
    code_answers = bool(answer) and (bool(blocks) or _sql_blocks(state) is not None
                                     or state.get("spatial_block_candidates") is not None)
    if llm is not None:
        try:
            if facts:
                numbers = ("write no numbers at all except block names exactly as the facts give them: every sentence "
                           "with any other number, in digits or in words, is deleted before the user sees it")
                if policy:
                    numbers = ("you may state a policy figure (a period, age, percentage or amount) only exactly as the "
                               "Policy lines above write it, digits and unit included (\"10-year MOP\", \"3% to 6%\"), a range "
                               "always whole and never one end of it or ends of two ranges joined; write no other numbers "
                               "(every sentence with any other number is deleted before the user sees it)")
                if code_answers:
                    # code says whether the blocks meet the limits; the model adds the trade-off, which is checked
                    task = ("Your text goes after code's own first sentence, shown to the user exactly as written: \""
                            + " ".join(answer) + "\" Do not say again whether the blocks meet the limits. Write only one or "
                            "two sentences: " + ("the trade-offs between the listed blocks in words (cheaper, more lease "
                                                 "left, nearer the MRT station), naming the block each one is about. Every "
                                                 "such comparison is checked against the facts, and a sentence with a "
                                                 "wrong one is deleted" if len(blocks_listed) > 1 else
                                                 "what the user should weigh in the options below, in words"))
                else:
                    task = ("Write only a short opening (one or two sentences that answer the question directly) and, when "
                            "there are several options, the trade-offs between them (price vs lease vs distance) in words: "
                            "cheaper, more lease left, nearer the MRT station")
                reading = trade if code_answers else lead
                rule = ("The facts section below was written by code from the tool results and will be shown to the user "
                        "exactly as written, straight after your text:\n---\n" + facts + "\n---\n" + task + ". Do not "
                        "repeat or change any price, grant, budget, distance, lease, count or relaxation figure: " + numbers
                        + ", and do not contradict the facts. Write no headings, lists or notes on taxes, fees or the HFE "
                        "letter: the facts already carry them, and such sentences are deleted too."
                        + (f" An opening in this spirit, about other blocks: {OPENING_EXAMPLE}"
                           if not policy and (len(blocks_listed) > 1 or not code_answers) else "")
                        + (f" What code reads from these facts, which you may reword or add to but must not contradict: "
                           f"\"{reading}\"" if reading else "")
                        + unchecked)
            msgs = [sys_msg(SYNTH_SYSTEM),
                    HumanMessage(content=f"User query: {user_request(state)}\n\n{rule}\n\nEvidence JSON:\n{ri.compact_tool_result(evidence, 24000)}")]
            if verdict and not verdict.get("passed", True) and verdict.get("critique"):
                msgs.append(HumanMessage(content=f"Your previous draft was rejected by the auditor: {verdict['critique']}. Rewrite it, fixing every issue."))
            text = llm.invoke(msgs).content
            if isinstance(text, list):
                text = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in text)
            text = strip_think(text)
            if facts:
                kept, dropped = check_model_opening(text, state, facts, policy)
                if code_answers:
                    # code's answer, then code's trade-off unless the model made a (checked) comparison of its own,
                    # then the model's sentences
                    compares = any(_COMPARISON.search(k) for k in kept)
                    opening = " ".join(answer + ([trade] if trade and not compares else []) + kept)
                else:
                    # the model's surviving sentences are preferred; code's opening only when none of them survives
                    opening = " ".join(kept) or lead
                code_only = not kept and bool(lead)
                why = {"figures": "stated figures", "comparisons": "compared blocks in a way the facts do not bear out",
                       "notes": "repeated a note the facts carry", "leans_back": "leaned on the sentence before"}
                drops = [f"synthesize: dropped {dropped[r]} opening sentence(s) that {w}" for r, w in why.items() if dropped[r]]
                if drops and code_only:
                    drops[0] += "; code opening used"
                errors += drops
                if code_answers or code_only:
                    errors += lead_errors
                text = (opening.strip() + "\n\n" if opening.strip() else "") + facts
        except Exception as e:  # noqa: BLE001
            errors.append(f"synthesize: LLM failed, template used: {type(e).__name__}")
            text = ""      # a failure after the model replied must not let its unchecked text through
    if not text:
        # no model, or it failed: the code-written facts, led by code's opening (variant C and the app)
        text = (lead + "\n\n" + facts if lead else facts) or template_response(state)
        errors += lead_errors
    elif state.get("financial_breakdown") or state.get("rag_policy_passages"):
        if "HFE" not in text:
            text += HFE_DISCLAIMER
    if clar and clar["question"] not in text:
        text = text.rstrip() + "\n\n" + clar["question"]
    return {"draft_response": text, "map_ui_payload": payload, "tool_call_log": log, "errors": errors,
            "clarification": clar, "suggestions": ri.drilldown_suggestions(state) if not clar else []}


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
        update = _INPUT_GUARDRAIL(state)
        if (update or {}).get("draft_response"):
            return update
        update = update or {}
    else:
        note = hook_problem_note("input_guardrail")
        update = {"errors": note} if note else {}  # pass-through until Member 5 delivers
        # stand-in for Member 5's check: greetings, off-topic asks, forecasts and prompt-injection get a fixed reply
        # (agents/scope.py). An open question or earlier turns mean a short message is an answer, not a stray ask.
        user_turns = [m for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
        mid = bool(state.get("pending_clarification")) or len(user_turns) > 1 or bool(state.get("map_ui_payload"))
        stray = scope.scope_reply(last_user_query(state), mid)
        if stray:
            return {**update, "draft_response": stray}
    if state.get("menu_followup"):    # the user picked a menu option: ask its follow-up question, nothing to search yet
        ask = state["menu_followup"]
        return {**update, "draft_response": ask["question"], "clarification": ask}
    query = last_user_query(state)
    unsupported = scope.unsupported_reply(query, len(rule_based_extract(query).work_destinations))
    if unsupported:
        return {**update, "draft_response": unsupported}
    # school questions the lookup cannot answer are stopped here, after Member 5's own check
    reply = guard_school_request(last_user_query(state), lambda q: bool(rule_based_extract(q).school_name))
    return {**update, "draft_response": reply} if reply else update


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
                  previous: Optional[Dict[str, Any]] = None, fast_path: bool = False) -> HDBCompassState:
    """previous: the final state of the last turn in the same conversation. Its messages, constraints and any
    open follow-up question carry forward, so a reply such as "we have $25,000 in cash" completes the earlier request.
    fast_path=True (the app) answers fully recognised self-contained questions without model calls; experiments leave
    it off so every variant is measured on its model's own choices."""
    history = list((previous or {}).get("messages") or [])
    # a reply to a menu (or to the question one of its options asked) is completed by code: an option's own follow-up
    # question is answered at once, and a bare launch name becomes the full question about it (agents/scope.py)
    menu_reply = scope.resolve_menu_reply((previous or {}).get("clarification"), query) if previous else None
    followup = (menu_reply or {}).get("ask")
    query = (menu_reply or {}).get("query") or query
    return {"menu_followup": followup, "messages": history + [{"role": "user", "content": query}], "variant": variant, "active_subagents": [],
            "candidate_blocks_intersection": [], "map_ui_payload": map_state or {}, "draft_response": "",
            "auditor_verdict": {}, "tool_call_log": [], "errors": [], "subagent_notes": [],
            "prior_constraints": (previous or {}).get("user_constraints"),
            "pending_clarification": (previous or {}).get("clarification"), "clarification": None,
            "fast_path": fast_path}


def conversation_text(state: HDBCompassState) -> str:
    """The latest message, preceded by the earlier turns when there are any (for agents without structured memory)."""
    msgs = [m for m in state.get("messages", []) if isinstance(m, dict)]
    if len(msgs) <= 1:
        return last_user_query(state)
    return "\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in msgs)
