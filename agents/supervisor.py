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
from datetime import date as _date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm import active_profile, get_llm, load_config
from agents import addons
from agents import sanitizer, scope
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
from agents.school_queries import (SCHOOL_NAME_QUESTION, guard_school_request, present_school_result,
                                   school_location_request)
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
              "BP": "BUKIT PANJANG", "SK": "SENGKANG", "PG": "PUNGGOL", "TAMP": "TAMPINES"}
# short forms that are not everyday words, so lower case is read too: "4rm tamp under 650k can?" (SPAT-51, LIVE-WS-01)
_SHORT_ANY_CASE = {"AMK", "TPY", "CCK", "TAMP"}
NO_DATA = "I could not find data for that request."   # the facts section when no tool found anything
FLAT_TYPES = ("1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION")   # as in the resale data
NUMBER_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5"}
AMENITY_WORDS = {"mrt": "mrt", "lrt": "mrt", "train": "mrt", "hawker": "hawker", "bus stop": "bus_stop", "bus": "bus_stop",
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


def named_towns(text: str, any_case_short: bool = False) -> List[str]:
    """The HDB towns a text names: full names, place aliases ("Tiong Bahru") and short forms ("AMK", "JE"). The regex
    reading and the check on a model's reading both use this one map, so code never drops a town the prompt taught the
    model to read. Short forms are whole words in upper case ("JE", not "je" in "jeans"); any_case_short also accepts
    "amk" written in lower case, for checking a town the model already read. "Bt" is Bukit ("Bt Batok", "Bt Merah")."""
    up = re.sub(r"\bBT\.?\s+", "BUKIT ", text.upper())
    towns = [t for t in HDB_TOWNS if t in up]
    for alias, town in TOWN_ALIASES.items():
        if alias in up and town not in towns:
            towns.append(town)
    for short, town in TOWN_SHORT.items():
        if re.search(rf"\b{short}\b", text, re.I if any_case_short or short in _SHORT_ANY_CASE else 0) and town not in towns:
            towns.append(town)
    return towns


def misspelt_towns(text: str) -> List[Tuple[str, str]]:
    """Words that are a near spelling of one HDB town ("Tampenis" -> TAMPINES, "Sengkand" -> SENGKANG), as
    (the user's word, the town). Only a word of 5+ letters that is not itself an English word in the message's other
    readings, and only a single close match, so "Jurong" (East or West) is not guessed."""
    import difflib
    names = {t.replace("/", " "): t for t in HDB_TOWNS}
    found: List[Tuple[str, str]] = []
    words = re.findall(r"[A-Za-z]+", text)
    for n in (2, 1):
        for i in range(len(words) - n + 1):
            w = " ".join(words[i:i + n])
            if len(w.replace(" ", "")) < 5 or w.upper() in names:
                continue
            # the same-first-two-letters rule ("Tampenis", not "question") holds for one word only: two words that start
            # like a town ("centres are" -> CENTRAL AREA) are a phrase, not a misspelling (SEM-29, SPAT-36)
            close = [c for c in difflib.get_close_matches(w.upper(), list(names), n=2, cutoff=0.75)
                     if difflib.SequenceMatcher(None, w.upper(), c).ratio() >= 0.84
                     or (n == 1 and c[:2] == w.upper()[:2] and abs(len(c) - len(w)) <= 1)]
            if len(close) == 1 and names[close[0]] not in [t for _, t in found]:
                found.append((w, names[close[0]]))
    return found


def stated_towns(text: str, any_case_short: bool = False) -> Tuple[List[str], List[str]]:
    """The towns a message states, and notes for the user: the towns it names, else a region's towns ("in the east"),
    else a near spelling of one town ("Tampenis"). The regex reading and the check on a model's reading use this one
    reading, so a region or a misspelt town the model read right is kept."""
    towns = named_towns(text, any_case_short=any_case_short)
    if towns:
        return towns, []
    region = ri.region_towns(text)
    if region:
        # the same note the grouped-statistics reading gives (research_intents), so it is shown once
        return list(region[1]), [f"'The {region[0]}' was taken as {', '.join(t.title() for t in region[1])} (URA planning regions)."]
    guess = misspelt_towns(text)
    if guess:
        return [t for _, t in guess], [f"Read “{w}” as {t.title()}." for w, t in guess]
    return [], []


# A school's name: up to four words before School/Primary/High ("Ai Tong School", "St. Hilda's Primary School",
# "CHIJ St. Nicholas Girls' School", "Anglo-Chinese School (Junior)"). A bare "High" is a whole word in name casing
# ("Catholic High") and not a floor ("MRT high-floor", "high up", "pref high flr").
_SCHOOL_WORD = r"(?:St\.|[A-Za-z][A-Za-z'’\-]*)"
_SCHOOL_END = (r"(?:Primary Schools?|Schools?|Primary|(?-i:High))\b(?:\s*\((?:Primary|Junior|Secondary)\))?"
               r"(?![\s\-]+(?:floors?|flrs?|fl|storeys?|stor(?:e)?y|rise|level|up)\b)")
_SCHOOL_AFTER = re.compile(rf"\b(?:of|near|to)\s+(?:to\s+)?((?:{_SCHOOL_WORD}\s+){{1,4}}{_SCHOOL_END})", re.I)
_SCHOOL_AT_START = re.compile(rf"\s*((?:{_SCHOOL_WORD}\s+){{1,4}}{_SCHOOL_END})\s*"
                              r"(?=[12]\s*(?:[-–]\s*2\s*)?km\b|within\b|[.?!]|$)", re.I)
# A word written in lower case that never starts or makes up a school's name: "near my kids school", "close to MRT
# with high floor", "to our old school", "the nearest school". A capitalised word is the user's own name casing.
_NOT_A_NAME_WORD = re.compile(r"^(?:a|an|the|my|our|your|his|her|their|its|this|that|to|of|near|with|by|at|in|on|for|"
                              r"from|and|or|nearest|closest|nearby|any|some|old|new|current|future|same|kids?|kids'|"
                              r"kid's|sons?|son's|daughters?|daughter's|child|child's|children|children's|mrt|lrt|"
                              r"station|flats?|resale|room|hdb|bto|is|are|be|go|goes|going|attend\w*|which|where|"
                              r"secondary)$", re.I)


def school_named(text: str) -> Optional[str]:
    """The school the message names, or None. A word in lower case counts only when it can be part of a name
    ("catholic high school"), so a lower-case phrase about some school is never read as one. A phrase that names no
    school at all ("near good schools") is returned as it is: _verify_extraction then asks which school is meant."""
    candidates = [m.group(1) for m in _SCHOOL_AFTER.finditer(text)]
    start = _SCHOOL_AT_START.match(text)
    if start:
        candidates.append(start.group(1))
    for candidate in candidates:
        candidate = candidate.strip()
        if ri.generic_school_phrase(candidate) and candidate.lower() not in {"primary school", "school"}:
            return candidate
        described = re.match(r"(?:a|an)\s", candidate, re.I)
        # "the Henry Park Primary School", "our Rosyth School": the name follows the determiner
        candidate = re.sub(r"^(?:(?:the|a|an|our|my|your|his|her|their)\s+)+", "", candidate, flags=re.I)
        name_words = re.sub(_SCHOOL_END + r"\s*$", "", candidate, flags=re.I).split()
        if candidate.lower() in {"primary school", "school", "moe primary school"} or candidate.lower().endswith("for primary"):
            continue
        if (not name_words or any(w[0].islower() and _NOT_A_NAME_WORD.match(w) for w in name_words)
                or any(w.upper() in {"MRT", "LRT"} for w in name_words)):    # "near MRT High floor" names a station
            continue
        if described and not names_a_school(candidate):   # "a Catholic School": a kind of school; C's guard asks which
            continue
        return candidate
    return None


# A workplace or commute destination after "to". In the clause that says "commute" the place may be typed in lower
# case ("commute to raffles place under 45 min"); elsewhere only capitalised words ("Raffles Place", "NUS") or one-north
# count. So "my commute to work is long", "close to my parents", "move to Sengkang", "open to Bishan" and "to MRT" are not
# workplaces, and a time limit after the name ("to Jurong East under 45 min") is not part of it.
_PLACE_WORD = r"(?:one-north|[A-Z][A-Za-z0-9'’\-]*)"
_DESTINATION = re.compile(r"(?<![A-Za-z])(?<!close )(?<!closer )(?<!near )(?<!nearer )(?<!next )(?<!walk )(?<!walking )"
                          r"(?<!distance )(?<!move )(?<!moving )(?<!moved )(?<!relocate )(?<!relocating )(?<!open )"
                          r"(?<!opened )(?<!back )(?<!upgrade )(?<!upgrading )(?<!downgrade )(?<!downgrading )to\s+"
                          # read ahead without consuming, so "to one-north and to jurong east" gives both
                          r"(?=((?:one-north|[A-Za-z][A-Za-z0-9'’\-]*)(?:\s+(?:one-north|[A-Za-z][A-Za-z0-9'’\-]*)){0,4}))",
                          re.I)
_IN_COMMUTE_CLAUSE = re.compile(r"\b(?:commut\w*|workplaces?)\b[^,.;!?]*$", re.I)
_NOT_A_WORKPLACE = {"MRT", "LRT", "HDB", "BTO", "WORK", "OFFICE", "HOME", "SCHOOL", "STATION"}
_STATION_WORDS = {"MRT", "LRT", "STATION", "INTERCHANGE"}
# lower-case words that start no place name: "commute to my office", "commute to and from work", "to both offices"
_NOT_A_PLACE_START = re.compile(r"^(?:my|our|your|his|her|their|a|an|and|both|each|either|all|get|reach|go|be|"
                                r"there|here|it|them|work|office|home|school|mrt|lrt|station)$", re.I)
# where a workplace stated after "work at" ends: a limit, a time or the next clause ("Raffles Place within 30 minutes")
_WORKPLACE_END = re.compile(r"\s+(?:under|within|less|below|max\w*|no|at|by|daily|every|each|should|must|is|are|was|and|"
                            r"with|where|but|so|which|that|for|to|near|on|in|from|or|while|plus|\d)\b.*$", re.I)


def workplaces(q: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Workplaces and commute destinations the message names, and where they are in it (so a town named only as a
    workplace is not also a search town)."""
    found: List[str] = []
    spans: List[Tuple[int, int]] = []

    def add(name: str, start: int) -> None:
        name = name.strip(" -'’")
        if name and name.upper() not in _NOT_A_WORKPLACE and name.lower() not in {w.lower() for w in found}:
            found.append(name)
        if name:
            spans.append((start, start + len(name)))

    for m in re.finditer(r"\bworks?\s+(?:at|in)\s+([A-Za-z0-9 \-'’]+?)(?=[,.;?!]|\s+and\b|$)", q, re.I):
        add(_WORKPLACE_END.sub("", m.group(1)), m.start(1))
    # "commutes", "Change the workplaces to ..." and a bare limit ("below 30 minutes to Raffles Place") start the
    # commute clause too (SEM-27, SEM-28); "to Raffles Place and one-north" names two workplaces
    if len(found) < 2 and (re.search(r"\b(?:commut\w*|workplaces?)\b", q, re.I) or commute_limits(q)):
        for m in _DESTINATION.finditer(q):
            commute_clause = bool(_IN_COMMUTE_CLAUSE.search(q[:m.start()]))
            start = m.start(1)
            words = m.group(1).split()
            if words[0].lower() == "the":                    # "commute to the CBD"
                start, words = start + len(words[0]) + 1, words[1:]
            second = None
            if commute_clause:
                head = _WORKPLACE_END.sub("", " ".join(words))
                words = head.split()
                tail = re.match(r"\s+(?:and|&)\s+(?:to\s+)?(?:the\s+)?((?:one-north|[A-Za-z][A-Za-z0-9'’\-]*)"
                                r"(?:\s+(?:one-north|[A-Za-z][A-Za-z0-9'’\-]*)){0,4})", q[start + len(head):])
                if tail and words:                           # "to Raffles Place and one-north": both
                    name2 = _WORKPLACE_END.sub("", tail.group(1)).strip()
                    if name2 and not _NOT_A_PLACE_START.match(name2.split()[0]):
                        second = (name2, start + len(head) + tail.start(1))
            else:                                            # outside the commute clause, name casing only
                cased = [bool(re.fullmatch(_PLACE_WORD, w)) for w in words] + [False]
                words = words[:cased.index(False)]
            if not words or _NOT_A_PLACE_START.match(words[0]) and words[0][0].islower():
                continue
            if words[-1].upper() in _STATION_WORDS:
                if not commute_clause or len(words) == 1:    # "10 min walk to Bishan MRT" is a station
                    continue
                words = words[:-1]                           # "commute to Raffles Place MRT": the place
            add(" ".join(words), start)
            if second:
                add(*second)
    return found, spans


def without_workplaces(q: str) -> str:
    """The message with its workplace names blanked out when it names two or more, so a town named only as a
    workplace ("We work at Raffles Place and work at Jurong East. Find flats in Bukit Merah", "Change the workplaces to
    Changi Business Park and Tampines.") is not a search town; one also named elsewhere ("flats in Jurong East") stays."""
    work, spans = workplaces(q)
    if len(work) < 2:
        return q
    return "".join(" " if any(a <= i < b for a, b in spans) else ch for i, ch in enumerate(q))


# A minutes limit ("under 45 mins") is the commute limit unless it is about an amenity or family: "MRT within 10 mins",
# "near clinic within 10 min", "within 10 mins of my parents", "within 15 mins by bus to MRT", "10 min walk".
# "below 30 minutes to Raffles Place" is one too (SEM-28: it was never checked).
_MINUTES_LIMIT = re.compile(r"(?:under|within|less than|below|no more than|at most|up to|max|<)\s*(\d{2})\s*min(?:s|utes)?\b", re.I)
_AMENITY_PLACE = (r"(?:mrt|lrt|trains?|stations?|bus(?: stops?| interchange)?|hawker\w*|food (?:centre|court)s?|"
                  r"supermarkets?|markets?|malls?|parks?|clinics?|polyclinics?|hospitals?|schools?|parents?|in-laws|"
                  r"family|mum|mom|dad|grand\w+)")
_AMENITY_AFTER = re.compile(r"\s*(?:walk|walking|on foot|by foot|(?:by (?:bus|train|mrt|car|taxi)\s+)?(?:away\s+)?"
                            rf"(?:from|of|to)\s+(?:(?:an?|the|my|our|his|her|their)\s+)?(?:nearest\s+)?{_AMENITY_PLACE}\b)", re.I)
_AMENITY_BEFORE = re.compile(rf"\b{_AMENITY_PLACE}\b", re.I)
_COMMUTE_WORDS = re.compile(r"\b(?:commut\w*|travel\w*|journey|work|office|workplaces?|cbd)\b"
                            r"|\bto\s+(?:one-north|(?-i:[A-Z][a-z]))", re.I)


def commute_limits(q: str) -> List[int]:
    """The commute limits in minutes the message states. A limit tied to an amenity or to family, by the words right
    after it or by the words before it in its clause with no commute word nearby, is not one."""
    out: List[int] = []
    matches = list(_MINUTES_LIMIT.finditer(q))
    for i, m in enumerate(matches):
        clause_start = max(q.rfind(p, 0, m.start()) for p in ",.;!?")
        before = q[max(clause_start + 1, matches[i - 1].end() if i else 0):m.start()]
        nxt = matches[i + 1].start() if i + 1 < len(matches) else len(q)
        after = re.split(r"[,.;!?]|\band\b", q[m.end():nxt], maxsplit=1)[0]
        if _AMENITY_AFTER.match(q, m.end()):
            continue
        if _AMENITY_BEFORE.search(before) and not _COMMUTE_WORDS.search(before) and not _COMMUTE_WORDS.search(after):
            continue
        out.append(int(m.group(1)))
    return out


# The regex reading of a policy question: an HDB rule named outright (MOP, EIP, Prime/Plus) or asked about in other
# words (who may buy, which grants exist, how long to wait, renting out). It also keeps a model's policy reading that
# scope's topic words miss. "plus" only in the Plus-classification sense: "plus $30,000 in cash" is not a policy question.
_POLICY_WORDING = re.compile(
    r"\b(mop|minimum occupation|prime|plus (?:flats?|models?|bto|resale|classification)|standard flat|clawback|subsidy recovery|eip|ethnic\w*|quota\w*|wait[- ]out|eligible|eligibility (?:criteria|requirements?|rules?|conditions?|to )|scheme\w*|rental restriction|exempt\w*"
    r"|not married|unmarried|fianc[eé]e?s?|apply(?:ing)? for (?:an? )?(?:[2-5][- ]room )?(?:flexi )?(?:bto|btr)s?|can (?:we|i) (?:buy|apply)"
    # rules asked about in other words: "grant rules", "2-room Flexi", "how long must I wait", "can I rent out",
    # "after selling my private property"
    r"|(?:grant|housing|hdb|resale|bto|btr) rules?"
    r"|how long (?:must|do|should|will|would) (?:i|we) (?:have to )?wait (?:before|until|to) (?:i |we )?(?:can )?(?:buy|apply|purchase)"
    r"|(?:sold|selling|disposed of|disposal of) (?:my |our |the |a )?private (?:residential )?propert(?:y|ies)"
    # who may buy, which grants exist, income limits, how often one may apply (the stems above need endings)
    r"|(?:can|may|could) (?:an? )?(?:single citizens?|singles?|singapore citizens?|citizens?|singaporeans?|pr|prs|permanent residents?|foreigners?|divorcees?|widow(?:er)?s?)\s+(?:(?:aged?|above|over)\s+\d{1,2}(?:\s+and\s+above)?\s+)?(?:still\s+)?(?:buy|apply|own|purchase|get)"
    r"|(?:single citizens?|singles?|singapore citizens?|citizens?|singaporeans?|pr|prs|permanent residents?|foreigners?|divorcees?|widow(?:er)?s?)\s+(?:(?:aged?|above|over)\s+\d{1,2}\s+)?(?:allowed|able|permitted|eligible)\s+to\s+(?:buy|apply|own|purchase)"
    r"|can\s+(?:aged?\s+\d{1,2}\s+and\s+above|someone\s+aged?\s+\d{1,2})\s+(?:still\s+)?(?:buy|apply|own|purchase)"
    r"|(?:what|which) (?:housing |cpf )?grants? (?:are|is|exist|does|(?:can|could|do|would|will|am|might)(?! (?:we|i)\b))|grants? (?:are )?(?:available|offered)|grants? for (?:first|second|singles?|young|large|married|couples?|families|seniors?|resale|bto)|income (?:ceiling|cap|limit)s?"
    r"|how many times (?:can|may) (?:i|we) apply|second[- ]?timers?"
    # a named grant explained, the rules for a purchase, CPF use: "What is the Enhanced CPF Housing Grant?",
    # "What are the rules for buying a resale flat with my parents?", "Can I use CPF for an HDB downpayment?"
    r"|what (?:is|are) (?:the |an? )?(?:ehg|phg|enhanced (?:cpf )?housing grant|proximity (?:housing )?grant|family grant|step-up grant|half-housing grant)"
    r"|rules (?:for|on|about) (?:buying|selling|owning|renting|subletting|applying)"
    r"|(?:can|may) (?:i|we) use (?:my |our )?cpf (?:for|to)"
    r"|(?:can|may) (?:i|we) (?:rent out|sublet)|rent out (?:my|our|the) (?:whole )?(?:flat|room|bedroom|unit)|buyback"
    # loan, letter and purchase-step rules asked in other words (QA sweep, 3 Nov: these got the menu)
    r"|ltv|loan[- ]to[- ]value|hfe letter|(?:apply|applying|get|getting|need|needs|have)\s+(?:for\s+)?(?:an?\s+|the\s+|my\s+)?hfe\b(?!-)|flat eligibility letter|option to purchase|otp|cash over valuation|cov"
    r"|(?:do|does|will|would|must|pay|paying|charged?)\b[^.?]{0,30}\b(?:absd|additional buyer'?s? stamp duty)|resale levy|lease buyback|age (?:of )?95|cover (?:me|us|them) (?:up )?to (?:age )?95"
    r"|who (?:can|may|is allowed to|are allowed to)\s+(?:buy|apply|get|qualif\w*|rent|own)|who qualif\w*"
    r"|\b(?:do|does|can|could|will|would|am|are|is)\b[^.?]{0,25}\bqualif\w* for (?:an? |the )?"
    r"(?:ehg|phg|proximity|enhanced|family grant|step-up|half-housing|[\w-]+ scheme|bto|hdb loan)[^.]*\?"
    r"|how long (?:must|do|should|will|would|have) (?:i|we) (?:have to )?(?:stay|live|wait|keep)"
    r"|sell (?:it|my flat|our flat|the flat|my bto|our bto) (?:before|after|within)|minimum age"
    r"|(?:pr|prs|permanent residents?|singles?|foreigners?) (?:couples? )?(?:buy|apply|own|purchase|get)\b"
    r"|(?:max(?:imum)?|how much) (?:hdb )?loan (?:can|may) (?:i|we) (?:get|take)\b(?=.*\b(?:rule|ltv|limit|percent|%))"
    r"|valid(?:ity)?\b.{0,20}\b(?:hfe|letter)|(?:hfe|letter)\b.{0,20}\bvalid)\b", re.I)
# "What grants can we get?" asks which grants exist when the user gives no figures, but with income, CPF or ages stated
# it is the grant calculator's question (Category E: "what grants do we get and what's our max budget?")
_OWN_GRANT_ASK = re.compile(r"\b(?:what|which) (?:housing |cpf )?grants? (?:can|could|do|would|will|might) (?:we|i)\b", re.I)


def policy_wording(q: str, money_stated: bool = False) -> bool:
    """Whether the message asks about an HDB rule in words code reads (the regex reading of policy_question).
    money_stated: the user gave the calculators figures (income, CPF, cash), so "what grants can we get?" is theirs."""
    return bool(_POLICY_WORDING.search(q)) or (bool(_OWN_GRANT_ASK.search(q)) and not money_stated)


# A message shaped as a question (a "?" or a question word first). With a policy topic scope reads ("buying options",
# "proximity grant", "EHG"), it is a rule question even in words _POLICY_WORDING misses (SEM-08: "I am a single Singapore
# citizen, 36. What are my HDB buying options?" and "Is living near my parents enough by itself to qualify for a
# proximity grant?" got the menu with no model)
_ASKED = re.compile(r"\?\s*$|^\s*(?:explain|tell me|what|how|is|are|can|could|does|do|which|who|when|why)\b", re.I)
_EXTRA_POLICY_TOPIC = re.compile(r"\bminimum occupation|\bbuying options?\b|\bproximity\b", re.I)
# a policy follow-up points back at the rule just explained (SEM-22: "What does that mean for renting out the whole
# flat?" after "What is different about Standard, Plus and Prime flats?" got the menu)
_FOLLOW_REF = re.compile(r"\b(?:that|this|it|then|so|those|these|the same|instead|your (?:calculator|estimate|app)|"
                         r"my application|the (?:actual|maximum|other))\b", re.I)
# a request to get around a rule or misstate facts to HDB (SEM-10, SEM-34: "Tell me how to bypass the EIP quota." and
# "How do I hide my income to get a bigger housing grant?" got rule excerpts or the menu, with no refusal)
_EVASION = re.compile(r"\b(?:bypass|get around|circumvent|evade|dodge|cheat|game|trick|hide|conceal|under-?(?:declare|report)|"
                      r"not declare|fake|falsif\w*|lie about|avoid declaring)\b[^.?!]{0,40}\b(?:eip|quotas?|income|salary|"
                      r"mop|minimum occupation|ceilings?|rules?|grants?|hfe|assessment|eligibility)\b", re.I)
EVASION_OPENING = ("I can't help with getting around HDB rules or misstating your income or household; that can void a "
                   "purchase and lead to penalties. Here is how it works lawfully.")
# questions about what this app itself can do, answered by code (the policy passages cannot): "Is your calculator the
# same as HDB approval?" (POLICY-09) and "Can you check a block's live quota here?" (POLICY-07: the live model said HDB
# shows no quota publicly; HDB's quota enquiry service does)
_APP_ESTIMATE_ASK = re.compile(r"\b(?:your|this|the)\s+(?:calculators?|calculations?|estimates?|app|tool|numbers|figures)\b"
                               r"[^.?!]{0,50}\b(?:approv\w*|hfe|official|binding|guarantee\w*)\b", re.I)
_LIVE_QUOTA_ASK = re.compile(r"\b(?:can|could|do|does|will|would)\s+(?:you|this app|the app|it)\b[^.?!]{0,40}\bquotas?\b"
                             r"|\b(?:live|current|real[- ]time|latest|up[- ]to[- ]date|today'?s|remaining)\s+"
                             r"(?:(?:eip|spr|ethnic(?:ity)?|racial|block'?s?|neighbou?rhood)\s+)*quotas?\b"
                             r"|\bquotas?\b[^.?!]{0,30}\b(?:right now|currently|live|real[- ]time)\b", re.I)
APP_ESTIMATE_OPENING = "No, these are estimates; only HDB's HFE letter confirms eligibility, grants and loan."
LIVE_QUOTA_OPENING = ("I can't check a block's current EIP/SPR quota here; HDB's quota enquiry service on hdb.gov.sg "
                      "shows it — check before the OTP.")
APP_ESTIMATE_POLICY_QUERY = "HDB Flat Eligibility HFE letter housing grant loan eligibility"
LIVE_QUOTA_POLICY_QUERY = "EIP SPR quota block neighbourhood"


def asks_policy(q: str) -> bool:
    """A question about a policy topic in words _POLICY_WORDING misses, that is not a data, search or money question."""
    return bool(_ASKED.search(q) and (scope.mentions_policy_topic(q) or _EXTRA_POLICY_TOPIC.search(q))
                and not scope.mentions_data(q) and not _MONEY_AMOUNT_ASK.search(q)
                # "Monthly repayment on a $500,000 loan at 2.6% for 25 years?" is code's instalment
                and instalment_question(q) is None)


# Terms that name one of several sibling schemes. A policy line or opening sentence about a sibling the question does
# not name answers a different question: "Can I rent out the whole of a Prime flat once its MOP is over?" opened with
# the Standard-flat rental rule (SEM-07), and an EIP question was answered with the SPR quota's 5%/8% (LIVE-POL-01/02).
# "PR" alone is not an SPR-quota term: "a citizen and PR couple" asks about grants, not the quota.
_SCHEME_FAMILIES = {
    "class": {"standard": r"\bstandard\b", "plus": r"\bplus\b(?!\s*(?:S?\$|\d))", "prime": r"\bprime\b"},
    "quota": {"eip": r"\beip\b|\bethnic|\bracial|\brace\b", "spr": r"\bsprs?\b|\bpermanent residents?\b"},
    "grant": {"ehg": r"\behg\b|\benhanced\b", "family": r"\bfamily grant", "phg": r"\bphg\b|\bproximity\b",
              "singles": r"\bsingles grant"},
}


def scheme_terms(text: str) -> set:
    """{(family, term)} named in a text: ("class", "prime") for "a Prime flat"."""
    return {(f, k) for f, terms in _SCHEME_FAMILIES.items() for k, p in terms.items() if re.search(p, text or "", re.I)}


def sibling_terms(text: str, question: str) -> set:
    """Terms the text names from a family the question names, that the question itself does not name ("Standard"
    in a line, when the question asks about Prime)."""
    asked = scheme_terms(question)
    families = {f for f, _ in asked}
    return {(f, k) for f, k in scheme_terms(text) if f in families and (f, k) not in asked}


# "Tampines Blk 884": a town, then a block number with no street after it. The block lookups read "Blk 884 Tampines".
_STREET_WORD = (r"(?:st|street|ave|avenue|rd|road|dr|drive|cres|crescent|ctrl|central|lane|ln|way|walk|ring|close|cl|"
                r"hts|heights|link|view|vale|place|pl|green|grove|loop|terrace|track|rise|hill|park|garden|gardens)")
_TOWN_THEN_BLOCK = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(HDB_TOWNS, key=len, reverse=True))
                              + r")\s+(?:blk|block)\.?\s*(\d{1,4}[a-z]?)\b(?!\s+(?:[a-z0-9]+\s+){0,2}" + _STREET_WORD + r"\b)",
                              re.I)


def town_first_block(text: str) -> str:
    """'Resale price for Tampines Blk 884?' -> 'Resale price for Blk 884 Tampines?'. A block number with a street after
    it ('Tampines Blk 884 Tampines St 83') is already an address and is left alone."""
    return _TOWN_THEN_BLOCK.sub(lambda m: f"Blk {m.group(2)} {m.group(1)}", text)


_MONEY_NUM = r"(?:S?\$|SGD\s?)?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\b"
# a price a search is held to, in words the "under 650k" reading misses: "between 500k and 600k" (the upper end),
# "600k max", "for 600k", "make it 800k", "800k instead", "raise the budget to 800k", or a bare "$600k" / "600k" in a
# flat search (a town or flat type named) that is not a price being checked, an income or savings
_NOT_A_BUDGET = re.compile(r"\b(?:fair|worth|asking|offer\w*|sold|selling|sell|income|salary|earn\w*|savings?|cash|cpf|"
                           r"loan|grants?|stamp|duty|paid|bought|valuation|cov|profit|debt|too much|too high|overpriced|"
                           r"reasonable|expensive|good (?:price|deal))\b", re.I)


def price_range(q: str) -> Optional[Tuple[float, float]]:
    """"between $600k and $650k", "from 500k to 600k", "$500k-$600k", "500 to 600k": the (lower, upper) price."""
    m = re.search(r"(?:\bbetween\s+|\bfrom\s+)?(?:S?\$\s?)?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand)?\s*(?:and|to|-|–)\s*"
                  r"(?:S?\$\s?)?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)\b", q, re.I)
    if not m or (not m.group(2) and not re.search(r"\bbetween\b|\bfrom\b|\$", m.group(0), re.I)
                 and float(m.group(1).replace(",", "")) < 1000):
        return None
    hi = _money(f"{m.group(3)}{m.group(4)}")
    lo = _money(f"{m.group(1)}{m.group(2) or m.group(4)}")
    if lo and hi and 50_000 <= lo < hi <= 5_000_000 and not re.search(r"\b(?:years?|yrs?|room|rm|sqm|metres?|m\b)", m.group(0)[:-1], re.I):
        return lo, hi
    return None


def _budget_phrase(q: str) -> Optional[float]:
    rng = price_range(q)
    if rng:
        return rng[1]
    pats = (r"\bbetween\s+" + _MONEY_NUM + r"\s*(?:and|to|-|–)\s*" + _MONEY_NUM,
            _MONEY_NUM + r"\s*(?:max(?:imum)?|tops|or less|and below|or below|cap)\b",
            r"\b(?:make it|change (?:it|the budget|budget) to|(?:increase|raise|lower|reduce|up|drop|bump)\s+(?:the\s+|my\s+)?"
            r"(?:budget|price|limit|max)?\s*to|budget to|for)\s+" + _MONEY_NUM,
            _MONEY_NUM + r"\s+instead\b")
    for i, pat in enumerate(pats):
        m = re.search(pat, q, re.I)
        if not m:
            continue
        num, unit = (m.group(3), m.group(4)) if i == 0 else (m.group(1), m.group(2))
        if not unit and not re.search(r"[$,]", m.group(0)) and float(num.replace(",", "")) < 50_000:
            continue
        if (unit or "").lower() == "m" and not re.search(r"\$", m.group(0)) and float(num.replace(",", "")) >= 10:
            continue                                                     # metres
        value = _money(f"{num}{unit or ''}")
        if value and value >= 50_000:
            return value
    if (named_towns(q) or re.search(r"\b[2-5]\s*[- ]?\s*(?:room|rm)\b|\bexecutive\b", q, re.I)) and not _NOT_A_BUDGET.search(q):
        m = re.search(r"(?:S?\$|SGD\s?)\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\b|\b(\d[\d,]*(?:\.\d+)?)\s*(k|thousand)\b", q, re.I)
        if m:
            num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            value = _money(f"{num}{unit or ''}")
            if value and 50_000 <= value <= 3_000_000:
                return value
    return None


_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def spelled_numbers(q: str) -> str:
    """Spelled-out amounts and lease years as digits, so the readings below see them: "under a million" is "under $1m",
    "half a million" "$500k", "at least seventy years lease" "at least 70 years lease" (SEM-02, SPAT-03, SPAT-15). Tens
    only before "years", so "one" in "under one million" or "the fifty" elsewhere is never a number by itself."""
    q = re.sub(r"\bhalf a million\b", "$500k", q, flags=re.I)
    q = re.sub(r"\b(?:a|one)\s+million\b", "$1m", q, flags=re.I)
    return re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\b"
                  r"(?=[- ]?(?:years?|yrs?)\b)",
                  lambda m: str(_TENS[m.group(1).lower()] + (_UNITS[m.group(2).lower()] if m.group(2) else 0)), q, flags=re.I)


def rule_based_extract(query: str) -> ExtractedConstraints:
    """Regex extractor: the no-LLM path and the safety net when structured output fails."""
    nearby = school_location_request(query)
    if nearby is not None:
        return ExtractedConstraints(nearby_schools=nearby)
    q = town_first_block(query)
    towns, notes = named_towns(q), []
    for word, digit in NUMBER_WORDS.items():
        q = re.sub(rf"\b{word}(?=[- ]?(?:room|rm)\b)", digit, q, flags=re.I)
    q = spelled_numbers(q)
    # "4-room", "4rm"; not a room count ("at least 2 rooms", "3 bedrooms")
    flat_types = [f"{m.group(1)} ROOM" for m in re.finditer(r"\b([2-5])\s*[- ]?\s*(?:room|rm)s?\b", q, re.I)
                  if not re.search(r"(?:at least|min(?:imum)?|more than|over|minimum of)\s*$", q[:m.start()], re.I)]
    # "3 or 4 room", "4/5-room", "3-4 room", "3 to 5 room": every type in the range
    for m in re.finditer(r"\b([2-5])\s*(?:,|or|and|/|-|–|to)\s*([2-5])\s*[- ]?\s*(?:room|rm)s?\b", q, re.I):
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        flat_types = [f"{n} ROOM" for n in range(lo, hi + 1)] + flat_types
    if re.search(r"\bexecutive\b", q, re.I):
        flat_types.append("EXECUTIVE")
    flat_types = list(dict.fromkeys(flat_types))

    budget = None
    # "under 650k", "<650k", "budget up to $0.65m", "no more than 650 thousand"; a bare "400m" is metres, not money.
    # Also "<= 550000", "ceiling $500,000", "cap of 600k", "raise the top end to 700k" (SEM-02, SPAT-56).
    for m in re.finditer(r"(?:under|below|less than|max(?:imum)?|up to|budget(?: of)?(?: up to)?|no more than|at most|"
                         r"\b(?:ceiling|cap)(?:\s+(?:of|at))?|\b(?:top|upper)\s+(?:end|limit)(?:\s+(?:of|to|at))?|<=|<|≤)"
                         r"\s*(\$)?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|mil)?\b", q, re.I):
        unit = (m.group(3) or "").lower()
        if unit == "m" and not m.group(1) and float(m.group(2).replace(",", "")) >= 10:
            continue                                                     # metres
        value = _money(f"{m.group(2)}{unit}")
        if value and value >= 50_000:                                    # an HDB budget, not a distance or an age
            budget = value
            break
    if budget is None:
        budget = _budget_phrase(q)
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
    # not a flat's age ("at least 60 years old flat") or a working life ("30 years left before retirement").
    # Also "at least 65-year lease" and "70 years remaining" / "70 years left" (SEM-03, SPAT-15), but not a
    # "60-year-old flat" or "30 years left before retirement".
    for pat in (r"(?:at least|minimum|min\.?|>=|≥)\s*(\d{2})\s*-?\s*(?:years?|yrs?)\b(?![\s-]*(?:old|of age|left before|to retire|"
                r"before retire|till retire|until retire))",
                r"(\d{2})\s*\+\s*(?:years?|yrs?)",
                r"\blease\s*(?:of\s*)?(?:at least\s*|>=\s*|≥\s*)?(\d{2})\s*\+?\s*(?:years?|yrs?)",
                r"(\d{2})\s*-?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:remaining\s*)?lease",
                r"(\d{2})\s*-?\s*(?:years?|yrs?)\s+(?:of\s+lease\s+)?(?:remaining|left)\b(?!\s+(?:before|to|till|until|on)\b)"):
        m = re.search(pat, q, re.I)
        if m:
            lease = int(m.group(1))
            break
    if lease is None and re.search(r"\blease\b", q, re.I):
        # "remaining lease more than 70 years", "lease > 80 years", "lease above 65 yrs", "over 70 years of lease"
        m = re.search(r"(?:more than|over|above|>|≥|>=|no less than|not less than)\s*(\d{2})\s*(?:years?|yrs?)\b", q, re.I)
        if m:
            lease = int(m.group(1))
    if lease is None:
        # "built after 2015", "completed after 2010", "flats from 2012 onwards": the lease left on a flat whose 99-year
        # lease started that year (stated as an assumption: HDB leases start at completion, roughly)
        m = re.search(r"\b(?:built|completed|lease (?:started|commenced)|from)\s+(?:after|since|in or after)?\s*((?:19|20)\d\d)\b"
                      r"(?:\s+onwards)?", q, re.I)
        if m and re.search(r"\b(?:built|completed|lease|onwards|after)\b", m.group(0), re.I):
            year = int(m.group(1))
            lease = max(0, 99 - (_date.today().year - year))
            notes.append(f"Read “{m.group(0).strip()}” as at least {lease} years of lease left (a 99-year lease from {year}).")

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
    for m in re.finditer(r"\b(mrt|lrt|hawker|supermarket|bus stop|park)(?:(?!\band\b)[a-z ])*?within\s*(\d+)\s*m\b", q, re.I):
        radii[AMENITY_WORDS[m.group(1).lower()]] = int(m.group(2))
    # "MRT 300m", "near mrt 400m", "LRT < 300 m", "hawker 0.5km", "the MRT limit 900m", "mrt less than 800 metres"
    # (SEM-16: "Make the MRT limit 900m." repeated the 500 m search)
    for m in re.finditer(r"\b(mrt|lrt|hawker|supermarket)(?:\s+(?:stations?|centres?|centers?))?(?:\s+(?:limit|distance|radius))?\s*"
                         r"(?:<=|<|≤|:|=|to|of|at|less than|under|below|within|max(?:imum)?|up to|no more than|at most)?\s*"
                         r"(\d+(?:\.\d+)?)\s*(km|m|metres?|meters?)\b",
                         q, re.I):
        dist = float(m.group(2)) * (1000 if m.group(3).lower() == "km" else 1)
        if dist >= 20:
            radii.setdefault(AMENITY_WORDS[m.group(1).lower()], int(dist))
    # "within 300m of LRT 4-room in Punggol": the amenity right after the distance, whatever follows it
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(km|m|metres?|meters?)\s*(?:of|from|to)\s+(?:an?\s+|the\s+)?(mrt|lrt|train|hawker|supermarket)\b", q, re.I):
        dist = float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1)
        if dist >= 20:
            radii.setdefault(AMENITY_WORDS[m.group(3).lower()], int(dist))
    # "5 min walk to MRT", "10-minute walk from a hawker centre": about 80 m a minute (stated as an assumption)
    for m in re.finditer(r"\b(\d{1,2})[- ]?(?:mins?|minutes?)\s*(?:walk|walking)\s*(?:distance\s*)?(?:to|from|of)?\s*(?:an?\s+|the\s+)?"
                         r"(mrt|lrt|train|hawker|supermarket)", q, re.I):
        key = AMENITY_WORDS[m.group(2).lower()]
        if key not in radii:
            radii[key] = int(m.group(1)) * 80
            notes.append(f"Read a {m.group(1)}-minute walk as {radii[key]} m (about 80 m a minute).")
    # "≤400m MRT", "no more than 400 metres from an MRT station", "max 500m to a hawker centre"
    for m in re.finditer(r"(?:≤|<=|<|within|max(?:imum)?|no more than|at most)\s*(\d+(?:\.\d+)?)\s*(km|m|metres?|meters?)\b"
                         r"\s*(?:of|from|to)?\s*(?:an?\s+|the\s+)?(mrt|train|hawker|supermarket|bus stop|park|clinic)", q, re.I):
        dist = float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1)
        if dist >= 20:
            radii.setdefault(AMENITY_WORDS[m.group(3).lower()], int(dist))
    stated_rail = "mrt" in radii   # a rail distance the user gave (not research_intents' vague "near an MRT" default)

    # a name in the user's own casing, or lower-case words that can be a name (school_named)
    school = school_named(q)
    if school:
        radius = 2000 if re.search(r"\b(?:1\s*[-–]\s*)?2\s*km\b|\b2000\s*m\b", q, re.I) else 1000
        radii.setdefault("school", radius)

    own = statements(q, own=True)
    income = stated_income(own, is_couple(q))
    amounts = bound_amounts(own, is_couple(q))
    cpf = amounts.get("cpf_oa_balance")   # paired with its own word (never the income just before it)

    # the legacy scheme label, from the same stated reading as the calculator facts (nothing inferred from "couple")
    pair, first = citizenship_stated(q), first_timer_stated(q)
    scheme = {"SC_Single": "SC_Single_35", "SC_PR": "SC_PR_FirstTimer" if first else None,
              "SC_SC": {True: "SC_SC_FirstTimer", False: "SC_SC_SecondTimer"}.get(first)}.get(pair or "")

    work, _ = workplaces(q)
    rest = without_workplaces(q)
    if rest != q:
        towns = [t for t in towns if t in named_towns(rest)]

    # two different limits ("30 mins to Raffles Place and 40 mins to Jurong East") keep the stricter one: one limit
    # holds for every workplace, and a looser limit than the user stated is never applied. An amenity's minutes ("MRT
    # within 10 mins") are not a commute limit (commute_limits).
    commute_mins = commute_limits(q)
    max_commute = min(commute_mins) if commute_mins else None

    policy = q if policy_wording(q, money_stated=bool(income or amounts)) else None

    base = dict(
        towns=towns, flat_types=flat_types, max_budget_sgd=budget, min_remaining_lease_years=lease,
        amenity_radii_meters=radii, household_income=income, citizenship_scheme=scheme, cpf_oa_balance=cpf,
        school_name=school, work_destinations=work, max_commute_minutes=max_commute, policy_question=policy,
        wants_price_trend=bool(re.search(r"trend|appreciat|per sqm|psm|over the (?:last|past)|median price|price changes?|prices? (?:have |has )?(?:changed|moved|risen|rise|fallen|fall)"
                                         r"|\b(?:gone|went|going) (?:up|down)\b|\b(?:increased|decreased|dropped|risen|climbed)\b", q, re.I)),
        wants_bto_comparison=bool(re.search(r"\bbto\b.*(?:vs|versus|against|compar|cheaper|pricier|more expensive)"
                                            r"|compar.*\bbto\b|\bresale\b.*\b(?:cheaper|pricier|more expensive) than\b.*\bbto\b", q, re.I)),
    )
    base.update(ri.extract_research_fields(q, base))
    rail = base.get("amenity_radii_meters") or {}
    if stated_rail and "mrt" in rail:
        # "mrt" measures to the nearest MRT or LRT station. "4-room in Punggol below 750k; LRT must be within 300m."
        # means LRT stations only, and "Use MRT stations only instead." MRT stations only (the spatial tool's "lrt" and
        # "mrt_only" keys): read as "mrt", LRT-only blocks passed the MRT check (SEM-01, SPAT-38). Renamed after the
        # research reading, so a grouped question still drops the distance it groups by.
        if re.search(r"\blrt\b", q, re.I) and not re.search(r"\bmrt\b|\btrains?\b", q, re.I):
            rail["lrt"] = rail.pop("mrt")
        elif re.search(r"\bmrt(?:\s+stations?)?\s+only\b|\bonly\s+(?:the\s+)?mrt\b|\b(?:no|not|exclud\w*|without)\s+(?:the\s+)?lrt\b",
                       q, re.I):
            rail["mrt_only"] = rail.pop("mrt")
    if not base.get("towns"):
        # a region ("in the east") or a misspelt town ("Tampenis") when the research reading (which groups a region's
        # towns itself for statistics) left the towns empty: a flat search there. Not a town named only as a
        # workplace ("Change the workplaces to Changi Business Park and Tampines.": SPAT-33)
        found, more = stated_towns(rest)
        if found:
            base["towns"], notes = found, notes + more
    if notes:
        base["assumptions"] = list(dict.fromkeys(list(base.get("assumptions") or []) + notes))
    if not base.get("policy_question") and asks_policy(q) and not (income or amounts or base.get("household_income")) \
            and not (base.get("towns") or base.get("flat_types") or base.get("max_budget_sgd") or base.get("addresses")):
        base["policy_question"] = query   # SEM-08, see asks_policy (a message with the household's figures is the calculators')
    if not base.get("bto_project_name"):
        # "Compare BTO project Imaginary Garden QZX with nearby resale.": a launch named after the words, which
        # known_bto_project cannot find when it is not in the launch data; the comparison then says so (SPAT-60).
        # A town is not a project name ("BTO launch Tampines" compares the town's launches, as before).
        m = re.search(r"\bBTO (?:project|launch)\s+(?:called\s+|named\s+)?([A-Z][\w']*(?:\s+[A-Z][\w']*)*)", query)
        if m and m.group(1).upper() not in HDB_TOWNS:
            base["bto_project_name"] = m.group(1)
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
    " work_destinations: list of workplace or commute destinations (e.g. ['Raffles Place', 'one-north', 'Jurong East']). "
    "max_commute_minutes: maximum travel/commute time in minutes (e.g. 30, 35). "
    "addresses: copy every HDB block address, postal code or named place (MRT station, hawker centre, school) "
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
# month", "car loan $800/month", "$500 a month allowance for my parents") or a yearly figure. "6 months" is a period,
# but "Income $8,000 monthly" is an amount (SEM-19: it asked for the income again).
_AMOUNT_TOKEN = re.compile(r"(?:S?\$|\bSGD\s?)?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k|thousand)?\b"
                           r"(?!\s*(?:%|m\b|km\b|years?|yrs?|mins?\b|minutes|months?\b|mths?\b|rooms?|rm\b|-room))", re.I)
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
    r"|\b(?:2|two)\s+(?:SCs|singaporeans|singapore citizens)\b"
    # "We are married citizens aged 37 and 35", "Citizen first-timers aged 27 and 29", "We are citizen first-timer
    # buyers" (SEM-19: each asked for the citizenship again)
    r"|\b(?:we(?:'re| are)\s+(?:a\s+)?)?married\s+(?:singapore(?:an)?\s+)?citizens\b"
    r"|^\W*(?:we(?:'re| are)\s+)?(?:singapore(?:an)?\s+)?citizen\s+(?:first[- ]?timers|first[- ]?timer buyers|buyers|first[- ]time buyers)\b"
    r"|\bwe(?:'re| are)\s+(?:married\s+)?(?:singapore(?:an)?\s+)?citizen\s+(?:first[- ]?timers?|first[- ]?timer buyers|buyers)\b", re.I)
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
    # "first-time buyers" and "citizen first-timer buyers" are the buyers' own status, not talk about buyers in general
    return ". ".join(c for c in _clauses_of(statements(text, own=True))
                     if not _GENERAL_TALK.search(re.sub(r"\b(?:citizen\s+)?first[- ]tim(?:e|er)s? (?:home )?(?:buyers|applicants)\b",
                                                        "", c, flags=re.I)))


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
# "second-timers", "second-time buyers" (SEM-19: HOUSE-04 asked whether they were first-timers)
_NOT_FIRST = re.compile(rf"\b(?:second|2nd)[- ]?tim(?:ers?|e (?:home )?(?:buyers?|applicants?))\b|\b(?:not|no|no longer|aren't|isn't|don't|doesn't|never been|"
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
    if "interest_rate" in needed:
        # "The interest rate is 3%.", "3%", "rate 2.6% p.a." in reply to the question that asked for it
        m = re.search(r"(?:^|\b(?:rate|interest)\b[^%\d]{0,15})(\d{1,2}(?:\.\d+)?)\s*%", (text or "").strip(), re.I)
        if m and 0 < float(m.group(1)) < 15:
            found["interest_rate"] = float(m.group(1))
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
# "Change price and valuation to $360,000.", "set the price to $1.2m", "price is now $500k", "the valuation becomes $700k"
_PRICE_FIX = re.compile(r"\b(?:change|set|make|update|use|try)\s+(?:the\s+)?(?:purchase\s+)?(?:price|it)\b(?:\s+(?:and|&)\s+"
                        r"(?:the\s+)?valuation)?\s+(?:to|at|as)\s+(?:S?\$\s?)(\d[\d,]*(?:\.\d+)?)\s*(k|m|million)?\b"
                        r"|\b(?:price|valuation)(?:\s+and\s+(?:the\s+)?valuation)?\s+(?:is\s+now|now|becomes|is)\s+(?:S?\$\s?)"
                        r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|million)?\b", re.I)
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
                    rf"|{_WORKED}\s+(?:12|twelve)\s*(?:months?|mths?)\s+continuously"
                    # "Both worked 12 months", "we both worked continuously for the last 12 months" (SEM-19)
                    r"|\b(?:we\s+)?(?:both|each|all)\s+(?:have\s+)?worked\s+(?:continuously\s+)?(?:for\s+)?(?:the\s+)?"
                    r"(?:last|past)?\s*(?:12|twelve)\s*(?:months?|mths?)\b"
                    r"|\bworked\s+continuously\s+(?:for\s+)?(?:the\s+)?(?:last|past)\s+(?:12|twelve)\s*(?:months?|mths?)\b")
# the buyers say they have NOT worked the 12 months (the household form's "No": "We have not worked continuously for the
# past 12 months"; "I haven't been employed continuously for 12 months"; "not continuously employed for 12 months")
_NOT_CONTINUOUS_WORK = (r"(?:\bnot|n't|\bnever)\s+(?:\w+\s+){0,3}(?:continuous(?:ly)?|consistently)\s+(?:\w+\s+){0,2}"
                        r"(?:employ\w*|working|worked|work)\b.{0,40}\b(?:12|twelve)\s*(?:months?|mths?)\b"
                        r"|(?:\bnot|n't|\bnever)\s+(?:\w+\s+){0,2}(?:employ\w*|working|worked)\s+(?:\w+\s+){0,2}continuously\b"
                        r".{0,40}\b(?:12|twelve)\s*(?:months?|mths?)\b")


# Grant, loan and proximity-grant eligibility, each read per sentence as confirmed (True) or not confirmed (False). The
# household form sends "Our grant eligibility is (not) confirmed.", "Our HDB loan eligibility is (not) confirmed." and
# "Our proximity grant eligibility is (not) confirmed."; a "not confirmed" answer must be read too, or the app asks
# again forever, and the proximity sentence is not the general grant eligibility.
_GRANT_ELIGIBILITY = (r"eligibility for (?:the )?(?:housing )?grants?.{0,40}\bconfirmed\b"
                      r"|(?<!proximity )(?<!proximity housing )\bgrants?\b.{0,30}\beligibility\b.{0,25}\bconfirmed\b"
                      r"|\bconfirmed (?:our|my) eligibility for (?:the )?grants?"
                      r"|\bconfirmed (?:that )?(?:we are|we're|i am|i'm|we|i) (?:are |am )?eligible for (?:the )?(?:housing )?grants?")
# "grants confirmed", "HDB has not confirmed our grant eligibility" (SEM-19)
_GRANT_ELIGIBILITY += (r"|^\W*(?:(?:our|my|the)\s+)?(?:housing\s+)?grants?\s+(?:are\s+|is\s+|have been\s+|has been\s+)?"
                       r"(?:already\s+)?confirmed\b"
                       r"|\b(?:HDB|they)\s+(?:has|have)(?:\s+not|n't)?\s+(?:yet\s+)?confirmed\s+(?:our|my)\s+(?:housing\s+)?"
                       r"grant\s+eligibility")
_LOAN_ELIGIBILITY = (r"eligibility for .{0,30}\b(?:HDB |bank )?loan (?:is |has been |has not been |is not |isn't )?(?:yet )?confirmed"
                     r"|\b(?:HDB |bank )?loan\b.{0,15}\beligibility\b.{0,25}\bconfirmed\b"
                     r"|\b(?:HDB|bank) loan(?: eligibility)? (?:is |has been |is not |has not been )?confirmed"
                     r"|\bconfirmed (?:our|my) eligibility for (?:the )?(?:grants? and (?:an? |the )?)?(?:HDB |bank )?loan"
                     r"|\bconfirmed (?:that )?(?:we are|we're|i am|i'm) eligible for (?:the )?(?:grants? and (?:an? |the )?)?(?:HDB |bank )?loan")
# a bank's in-principle approval (IPA) is a bank loan's eligibility: "we have a bank IPA", "got an in-principle approval"
_IPA = r"\b(?:in[- ]principle approval|IPA)\b"
_BANK_APPROVAL = re.compile(_IPA + r"|\bbank\b[^.]{0,30}\b(?:approv\w*|confirm\w*)\b|\b(?:approv\w*|confirm\w*)\b[^.]{0,30}\bbank\b", re.I)
_PROXIMITY_ELIGIBILITY = r"\bproximity (?:housing )?grant\b.{0,40}\bconfirmed\b|\bconfirmed\b.{0,40}\bproximity\b"
_NOT_CONFIRMED = re.compile(r"\bnot (?:yet )?(?:been )?confirmed\b|n't (?:been )?confirmed\b|\bunconfirmed\b|"
                            r"\bnot sure\b|\bpending\b|\b(?:nobody|no one|never)\b[^.]{0,30}\bconfirmed\b", re.I)


def confirmations(statement_text: str) -> Dict[str, bool]:
    """Read clause by clause: "grants confirmed, loan eligibility not confirmed" confirms the grants only, and in
    "Grants and HDB loan eligibility confirmed, no proximity grant" the "no proximity grant" clause is not about the
    grant eligibility (SEM-19: read per sentence, both were missed)."""
    out: Dict[str, bool] = {}
    for sent in re.split(r"(?<=[.!?;])\s+|\n+", statement_text):
        for cl in re.split(r",\s+|\s+but\s+|\s+while\s+", sent):
            no = bool(_NOT_CONFIRMED.search(cl))
            if re.search(_PROXIMITY_ELIGIBILITY, cl, re.I):
                out["proximity_eligibility_confirmed"] = not no
                # "our proximity grant eligibility is confirmed" says nothing about the general grant eligibility
                if not re.search(r"\bgrants? and\b|\band (?:the |our |my )?(?:housing )?grants?\b", cl, re.I):
                    continue
            if re.search(_GRANT_ELIGIBILITY, cl, re.I) and not (
                    re.search(r"\bproximity\b|\bPHG\b", cl, re.I) and re.search(r"\bproximity\b", cl.lower().split("grant")[0])):
                out["eligibility_confirmed"] = not no
            if re.search(_LOAN_ELIGIBILITY, cl, re.I):
                out["loan_eligibility_confirmed"] = not no
            elif (m := re.search(_IPA, cl, re.I)) and re.search(r"\b(?:got|have|has|received|obtained|with)\b", cl[:m.start()], re.I):
                # "we have a bank IPA"; not "we haven't got an IPA yet"
                out["loan_eligibility_confirmed"] = not (no or re.search(r"(?:\bnot|n't|\bno|\bnever|\bwithout|\byet to)\b",
                                                                        cl[:m.start()], re.I))
    return out


def said_not_continuously_employed(statement_text: str) -> bool:
    """The buyers state they have not worked the 12 months (_stated_match skips negated matches, so this looks itself)."""
    return any(re.search(_NOT_CONTINUOUS_WORK, c, re.I) for c in _clauses_of(statement_text))


def _amount(m: re.Match) -> float:
    return float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)


# "Our monthly debt repayments are $500", and the age of the youngest person who will live in the flat
_DEBT_PAYMENT = r"\bmonthly (?:debt|loan) (?:repayments?|payments?|obligations?)\s+(?:are|is|of|total)\s+" + _AMOUNT
_YOUNGEST_OCCUPANT = (r"\byoungest (?:person|occupant|member|resident)\b[^.;]{0,40}\b(?:flat|home|household)\b[^.;]{0,15}"
                     r"\bis (\d{1,2})(?:\s*(?:years?|yrs?)(?:\s+old)?)?(?!\d)")


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
    elif (m := _stated_match(_DEBT_PAYMENT, s)):
        f["monthly_debt_obligations"] = _amount(m)
    if (m := _stated_match(_YOUNGEST_OCCUPANT, statements(t, keep=("plan",)))):   # "who will live in the flat"
        f["youngest_core_member_age"] = float(m.group(1))
    if said_not_continuously_employed(s):
        f["continuous_employment_12_months"] = False      # a stated "no" is an answer: asking again would loop
    elif _stated_match(_CONTINUOUS_WORK, s) and not _JOB_LOSS.search(s):
        f["continuous_employment_12_months"] = True
    # "eligibility for the grants is confirmed", "HDB has confirmed our eligibility for the grants"
    f.update(confirmations(s))
    if re.search(r"\bbank[- ]loan\b", t, re.I):          # "a bank-loan calculation" (SEM-20)
        f["loan_type"] = "bank"
    elif re.search(r"\bHDB loan\b", t, re.I):
        f["loan_type"] = "hdb"
    if re.search(_NO_PROXIMITY + r"|\bnot (?:buying |living |staying )?near (?:my|our) parents\b", s, re.I):
        f["proximity"] = "none"
    elif _parents_clause(_PARENTS_WITH, t):
        f["proximity"] = "with"
    elif _parents_clause(_PARENTS_NEAR, t):
        f["proximity"] = "near"
    # "no grants", "Assume zero housing grants." (HOUSE-04: it ran the grant calculator and asked its facts again)
    if re.search(r"\b(?:no|zero|nil)\s+(?:housing\s+|cpf\s+)?grants\b", s, re.I) or re.search(
            r"\b(?:assume|assuming|use|using|with)\s+(?:zero|no|nil|\$?0)\s+(?:housing\s+|cpf\s+)?grants?\b", s, re.I):
        f["total_grants"] = 0.0
    m = re.search(_CALC_LEASE, s, re.I)
    if m:
        f["remaining_lease_years"] = float(next(g for g in m.groups() if g))
    m = _stated_match(r"interest rate (?:of|is|at|:)?\s*(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%\s*(?:p\.a\.|per annum|interest)", s) \
        or _stated_match(r"\b(?:bank |loan )?rate (?:is|of|at)\s*(\d+(?:\.\d+)?)\s*%()", s)
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
        else:
            # "stamp duty on $650k?", "BSD for a 0.65m flat", "stamp duty of 1.2m": the price the duty is asked on
            m = re.search(r"\b(?:on|for|of)\s+(?:an?\s+|the\s+|my\s+|our\s+)?(?:S?\$\s?)?(\d[\d,]*(?:\.\d+)?)\s*"
                          r"(k|m|mil|million|thousand)?\b(?!\s*(?:sqm|years?|yrs?|metres?|meters?|mins?))", t, re.I)
            value = _money(f"{m.group(1)}{m.group(2) or ''}") if m else None
            if value and value >= 50_000:
                upfront["purchase_price"] = value
    # a correction of the price the duty or costs were asked on: "Change price and valuation to $360,000.", "The price
    # is now $1.2M" (SEM-18: the old price and its duty were kept). Read only in a conversation about those costs, so
    # "a price of $600k" in a flat search stays a budget; "price range" is never a price.
    fix = _PRICE_FIX.search(t) if re.search(_ASKS_UPFRONT, f"{context}\n{t}") else None
    if fix:
        num, unit = (fix.group(1), fix.group(2)) if fix.group(1) else (fix.group(3), fix.group(4))
        price = _money(f"{num}{unit or ''}")
        if price and price >= 50_000:
            upfront["purchase_price"] = price
            if re.search(r"\bvaluation\b", fix.group(0), re.I):
                upfront["market_valuation"] = price
    for field, pattern in (("market_valuation", r"valued at " + _AMOUNT), ("loan_amount", r"loan of " + _AMOUNT),
                           ("legal_and_valuation_fees", r"fees of " + _AMOUNT)):
        m = re.search(pattern, s, re.I)
        if m:
            upfront[field] = _amount(m)
    if "purchase_price" in upfront and "market_valuation" not in upfront:
        # "valuation also $500,000", "valuation equals price"
        m = re.search(r"\bvaluation\s+(?:is\s+|of\s+|also\s+|=\s*)*(?:S?\$\s?)(\d[\d,]*(?:\.\d+)?)\s*(k|m|million)?\b", t, re.I)
        if m:
            upfront["market_valuation"] = _money(f"{m.group(1)}{m.group(2) or ''}")
        elif re.search(r"\bvaluation\s+(?:equals|is the same as|same as|=|matches)\s*(?:the\s+)?(?:purchase\s+)?price\b", t, re.I):
            upfront["market_valuation"] = upfront["purchase_price"]
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
                 r"|\b(?:live|living|stay|staying|buy|buying)\s+with\s+(?:my|our)\s+parents\b")
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
        m = _stated_match(_YOUNGEST_OCCUPANT, statements(t, keep=("plan",)))
        if m and _isint(value) and int(value) == int(m.group(1)):
            return True
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
    if field in ("eligibility_confirmed", "loan_eligibility_confirmed", "proximity_eligibility_confirmed"):
        # only what the user said of that eligibility, per sentence: a proximity-grant confirmation is not the general
        # grant eligibility, and "not confirmed" is an answer (False)
        return confirmations(s).get(field) is value
    if field == "continuous_employment_12_months":
        if value is False:
            return said_not_continuously_employed(s)
        return value is True and not _JOB_LOSS.search(t) and not _NOT_CONTINUOUS.search(s) and \
            not said_not_continuously_employed(s) and \
            stated(r"\b(?:employ\w*|working|work|worked|job)\b", r"\b(?:12|twelve)\s*(?:months?|mths?)\b|\b(?:a|one|1|past) year\b")
    if field in ("proximity", "near_parents_4km"):
        none = bool(re.search(_NO_PROXIMITY + r"|\bno proximity\b|\bnot (?:buying |living |staying )?near (?:my|our) parents\b", s, re.I))
        if value in ("none", False):
            return none
        if value == "with":
            return _parents_clause(_PARENTS_WITH, t)
        return value in ("near", True) and (_parents_clause(_PARENTS_NEAR, t) or _parents_clause(_PARENTS_WITH, t))
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
    if "youngest_core_member_age" in g and "youngest_buyer_age" in g and \
            not g["youngest_core_member_age"] <= g["youngest_buyer_age"] - 15:      # a child's age is its own
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


# the figures of a flat the model may not choose for the upfront calculator (LIVE-MONEY-01)
UPFRONT_FLAT_FIGURES = ("purchase_price", "market_valuation", "loan_amount")
NO_PRICE_ERROR = ("purchase price not stated; ask the user for the flat's purchase price (and its valuation) before "
                  "working out the stamp duty or upfront costs")


def upfront_price(c: Dict[str, Any], outputs: Dict[str, Any], runners: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """The price the upfront calculator may cost: one code read from the user's words, else their stated budget, else
    the ceiling the affordability calculator worked out (run first, by code, if the model has not). Never a figure the
    model chose: live (HOUSE-04 t3) it costed "Does stamp duty fit within that number?" at a price nobody gave and
    answered $9,600 BSD. None when there is no such price."""
    fin = c.get("financial_inputs") or {}
    price = (fin.get("upfront") or {}).get("purchase_price")
    if price:
        return float(price)
    if c.get("max_budget_sgd"):
        return float(c["max_budget_sgd"])
    aff_name = "calculate_maximum_affordability"
    if aff_name not in outputs and runners and aff_name in runners and fin.get("affordability"):
        outputs[aff_name] = runners[aff_name](**guard_financial_args(aff_name, {}, c, outputs, runners))
    aff = outputs.get(aff_name) or {}
    if aff.get("status") == "ok" and aff.get("max_purchase_price"):
        return float(aff["max_purchase_price"])
    return None


def guard_financial_args(name: str, args: Dict[str, Any], c: Dict[str, Any], outputs: Dict[str, Any],
                         runners: Dict[str, Any]) -> Dict[str, Any]:
    """A calculator call as the model wrote it, corrected at the tool boundary: stated facts override the model's
    arguments, facts the user never stated are left out (the calculator then reports them missing), and a confirmed
    grant total is passed forward, running the grant calculator first if the model has not. The upfront calculator's
    price, valuation and loan are never the model's own (upfront_price; run_react refuses the call when no price is
    known)."""
    sec = FIN_TOOL_SECTION.get(name)
    fin = c.get("financial_inputs") or {}
    if sec is None:
        return args
    if sec == "upfront":
        args = {k: v for k, v in args.items() if k not in UPFRONT_FLAT_FIGURES}
        price = upfront_price(c, outputs, runners)
        if price is not None:
            args["purchase_price"] = price
    if not has_financial_facts(c):
        # nothing stated: the model's household figures are its own inventions (live, 3 Nov: SC_SC, $12,000 income,
        # $80k CPF for "How much grant can we get?"), so they are left out and the calculator asks for them
        return {k: v for k, v in args.items() if k not in FACT_FIELDS}
    stated = fin.get(sec) or {}
    out = {k: v for k, v in args.items() if k not in FACT_FIELDS}
    out.update(stated)
    grant = "calculate_cpf_housing_grants"
    if sec != "grants" and "total_grants" not in out:
        if grant not in outputs and grant in runners and fin.get("grants"):
            outputs[grant] = runners[grant](**guard_financial_args(grant, {}, c, outputs, runners))
        if (outputs.get(grant) or {}).get("status") == "ok":
            out["total_grants"] = outputs[grant]["total_grants"]
        elif c.get("pr_couple") or ((outputs.get(grant) or {}).get("status") == "unsupported" and c.get("budget_asked")):
            out["total_grants"] = 0.0
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


EXTRACTION_MAX_TOKENS = int(os.getenv("HDB_EXTRACTION_MAX_TOKENS", "1536"))


def capped(llm: Any, max_tokens: int) -> Any:
    """The same client with a lower output limit (a copy; the shared client keeps its own)."""
    try:
        if llm is not None and (getattr(llm, "max_tokens", None) or 0) > max_tokens:
            return llm.model_copy(update={"max_tokens": max_tokens})
    except Exception:  # noqa: BLE001 — a client that cannot be copied keeps its own limit
        pass
    return llm


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
    # the constraint JSON is a few hundred tokens: a reading that runs on to the 4,096-token limit is a runaway
    # (live, 3 Nov: 30 of 166 turns, about 30 s each, before the json_mode retry succeeded), so it is cut off early
    llm = capped(get_llm(), EXTRACTION_MAX_TOKENS)
    extracted: Optional[ExtractedConstraints] = None
    model_reading: Optional[Dict[str, Any]] = None
    model_policy = False
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
            model_policy = bool(extracted.policy_question)
            # merge: regex fills anything the model left empty (cheap robustness)
            rb = rule_based_extract(query).model_dump()
            merged = extracted.model_dump()
            for k, v in rb.items():
                if merged.get(k) in (None, [], {}, False) and v not in (None, [], {}, False):
                    merged[k] = v
            # Assumptions and questions are printed as they stand in the facts section code writes, so they are
            # code's own (the research reading of this message), never the model's free text.
            for k in MODEL_FREE_TEXT_FIELDS:
                if merged.get(k) != rb.get(k):
                    errors.append(f"extract_constraints: model-written {k} replaced by code's")
                merged[k] = rb.get(k) or []
            extracted = ExtractedConstraints(**merged)
    if extracted is None:
        if llm is not None:
            errors.append("extract_constraints: model extraction failed; rule-based fallback used")
        extracted = rule_based_extract(query)
    constraints = extracted.model_dump()
    if extracted.financial_inputs is not None:
        constraints["financial_inputs"] = extracted.financial_inputs.model_dump(exclude_unset=True)
    constraints = _finalize_constraints(constraints, query, state, errors, from_model=source != "regex",
                                        model_policy=model_policy)
    constraints["extraction_source"] = source
    return {"user_constraints": constraints, "errors": errors, "model_constraints": model_reading}


# A question about the buyer's own money or eligibility: the calculators or the policy agent answer it, never resale
# price statistics ("How much grant can I get?", "How much is the stamp duty?", "How much can we afford?").
_MONEY_ASK = re.compile(r"\b(?:grants?|ehg|phg|cpf|loans?|borrow\w*|mortgage|interest rates?|stamp dut(?:y|ies)|bsd|absd|"
                        r"down ?payments?|resale levy|levy|ltv|msr|tdsr|hfe|eligib\w*)\b"
                        r"|\b(?:can|could) (?:we|i) afford\b|\bhow much (?:can|could|should) (?:we|i) (?:afford|pay|spend)\b", re.I)


# a question about the buyer's own amounts ("how much can I borrow", "what's our max budget", "what grants can we get",
# "stamp duty on $650k"): the calculators answer it, and ask for the household facts they need, instead of the menu
_MONEY_AMOUNT_ASK = re.compile(
    r"\bhow much\b.{0,30}\b(?:borrow|loan|grants?|afford|budget|stamp duty|bsd|cash|cpf|downpayment|down payment|"
    r"instalments?|installments?|monthly)\b|\bwhat(?:'s| is| are)?\s+(?:our|my)\s+(?:max(?:imum)?\s+)?(?:budget|loan|grants?)\b"
    r"|\b(?:max(?:imum)?|how big a?)\s+(?:hdb\s+|bank\s+|housing\s+)?(?:loan|budget)\b"
    r"|\b(?:stamp duty|bsd)\b.{0,25}\b(?:on|for|of)\b.{0,15}\d|\bcan (?:we|i) afford\b|\bhow much can (?:we|i) (?:get|borrow)\b", re.I)


_DUTY_WORDS = re.compile(r"\b(?:stamp )?dut(?:y|ies)\b|\bbsd\b", re.I)
_FUNDING_WORDS = re.compile(r"\bcash\b|\bup[- ]?front\b|\bdown ?payment\b|\boutlay\b|\bcosts?\b|\bcpf\b", re.I)


def _stamp_duty_only(state: Dict[str, Any]) -> Optional[float]:
    """The buyer's stamp duty when the message asks only for the duty (not the cash needed), and the calculator gave it.
    "that duty" counts, and so does a message that corrects the price of a duty-only question asked earlier
    ("Change price and valuation to $360,000.") or asks what that duty needs ("Do I have to give you my income just to
    calculate that duty?"): SEM-18, SEM-06 listed the whole funding checklist instead."""
    users = [m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
    q = users[-1] if users else last_user_query(state)
    if _FUNDING_WORDS.search(q):
        return None
    if not _DUTY_WORDS.search(q):
        follows_duty = any(_DUTY_WORDS.search(u) and not _FUNDING_WORDS.search(u) for u in users[:-1])
        if not (follows_duty and (_PRICE_FIX.search(q) or (re.search(r"\bincome\b", q, re.I) and "?" in q))):
            return None
    up = (state.get("financial_breakdown") or {}).get("calculate_bsd_and_upfront_outlay") or {}
    return up.get("buyer_stamp_duty")


def duty_needs_income_question(state: Dict[str, Any]) -> bool:
    """"Do I have to give you my income just to calculate that duty?" after a stamp-duty answer (SEM-06)."""
    q = last_user_query(state)
    return bool(re.search(r"\bincome\b", q, re.I) and "?" in q and _stamp_duty_only(state) is not None)


def money_question(query: str, c: Dict[str, Any]) -> bool:
    """Whether the message asks about the buyer's money (grants, loan, stamp duty, CPF, what they can afford). A budget
    or "afford" also counts when the user gave calculator facts ("we earn 9k, what's our budget?")."""
    return bool(_MONEY_ASK.search(query)) or (has_financial_facts(c) and bool(re.search(r"\b(?:budget|afford\w*)\b", query, re.I)))


SEARCH_KEYS = ("towns", "flat_types", "max_budget_sgd", "min_remaining_lease_years", "amenity_radii_meters", "mrt_station",
               "school_name", "work_destinations", "max_commute_minutes", "min_price_sgd")
_NEW_SEARCH = re.compile(r"^\W*(?:(?:can you |could you |please |pls |now |ok(?:ay)?,? |also )*)(?:find|search|show|list|look(?:ing)? for|"
                         r"any|get|give)\b", re.I)
# "remove the MRT limit", "no lease limit", "drop the budget", "without the hawker centre", "any budget"
# and "Remove just the hawker condition." (SPAT-22)
_REMOVE = re.compile(r"\b(?:remove|drop|without|no|forget|ignore|skip|don'?t need|do not need|don'?t care about|any|clear)\b"
                     r"(?:\s+just)?(?:\s+(?:the|my|a|an|that|limit on|filter on|need for))?\s+"
                     r"(mrt|lrt|train|hawker|supermarket|school|commute|travel|budget|price|lease|distance)", re.I)
_REMOVES = {"mrt": ("amenity_radii_meters", "mrt"), "lrt": ("amenity_radii_meters", "lrt"),
            "train": ("amenity_radii_meters", "mrt"), "hawker": ("amenity_radii_meters", "hawker"),
            "supermarket": ("amenity_radii_meters", "supermarket"), "school": ("school_name", None),
            "commute": ("max_commute_minutes", None), "travel": ("max_commute_minutes", None),
            "budget": ("max_budget_sgd", None), "price": ("max_budget_sgd", None),
            "lease": ("min_remaining_lease_years", None), "distance": ("amenity_radii_meters", None)}
# a distance on its own changes the one distance limit: "within 300m", "try 400m", "Tighten it to 300m." (SPAT-62),
# "Only within 1km, please." and "Expand the school distance to 2km." (SEM-09: both got the menu or "Which school?")
_BARE_DISTANCE = re.compile(r"^\W*(?:(?:what|how) about|try|make it|change (?:it )?to|"
                            r"(?:tighten|loosen|widen|relax|reduce|cut|shrink|lower|raise|increase|extend|bring|set|narrow|expand|"
                            r"enlarge|change|bump)"
                            r"(?:\s+(?:it|that|this|the\s+(?:\w+\s+)?(?:limit|distance|radius|search)))?(?:\s+(?:down|up|back))?\s+to|"
                            r"ok(?:ay)?|and|then|or|only|just)?\s*"
                            r"(?:within|under|less than|below|<|up to)?\s*(\d+(?:\.\d+)?)\s*(km|m|metres?|meters?)\b"
                            r"(?:\s+(?:instead|then|only|radius|away))?(?:\W*(?:please|pls|thanks|thank you))?\W*$", re.I)
_SHORT_FOLLOWUP = re.compile(r"^\W*(?:(?:and|so|ok(?:ay)?|then)\W+)?(?:what|how) about\b|^\W*and (?:for|if|what|with)\b"
                             r"|^\W*(?:what|how) if\b|^\W*same (?:for|with)\b|^\W*(?:and|also) (?:the )?\w+\s*(?:flats?)?\s*\?*\s*$", re.I)
_REFINE_WORDS = re.compile(r"\b(?:cheap\w*|expensive|nicer|better|bigger|smaller|larger|closer|nearer|further|more|less|"
                           r"relax\w*|loosen|tighten|increase|decrease|raise|lower|change|instead|only|without|add|remove|"
                           r"drop|within|under|over|below|above|budget|lease|mrt|hawker|room|rm|newer|older)\b", re.I)
# a budget or lease on its own changes that limit: "Actually only $850k." (SEM-02), "Actually 60 years will do." (SEM-16).
# The whole message must be the amount, so a figure inside another sentence is read by the usual readings only.
_LEAD = r"^\W*(?:(?:actually|ok(?:ay)?|no|sorry|then|maybe|hmm|so|make it|change it to|how about|what about|try)\W+)*"
_BARE_BUDGET = re.compile(_LEAD + r"(?:only|just|max(?:imum)?|up to|under|below|at most)?\s*"
                          r"(S?\$\s?\d[\d,]*(?:\.\d+)?\s*(?:k|m|thousand|million)?|\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|million)\b)"
                          r"(?:\s+(?:only|max|instead|then|tops|budget|is fine|will do))?\W*$", re.I)
_BARE_LEASE = re.compile(_LEAD + r"(?:at least|min(?:imum)?)?\s*(\d{2})\s*-?\s*(?:years?|yrs?)\b(?:\s+(?:of\s+)?(?:lease|left|remaining))?"
                         r"(?:\s+(?:will do|is fine|is ok(?:ay)?|instead|then|only|enough))?\W*$", re.I)
UNCHANGED_NOTE = ("I couldn't tell what to change, so this is the same search as before. You can say, for example, "
                  "\"within 300m\", \"under $700k\", \"5-room instead\" or \"drop the MRT limit\".")
# "If nothing fits, show which one change would help.": a request about the results, not a change to the search
_CONDITIONAL_ASK = re.compile(r"\bif (?:nothing|none|no (?:blocks?|flats?|results?|matches?))\b", re.I)
# "I don't know the street or postal code.", "I have not picked a school yet.", "I have no address yet."
_DONT_KNOW = re.compile(r"\b(?:don'?t|do not|didn'?t|did not) (?:know|have)\b|\bno (?:idea|address)\b|\bnot sure\b|"
                        r"\bhave(?:n'?t| not) (?:picked|chosen|decided|got)\b", re.I)
NO_SCHOOL_PICKED = ("No problem. The data doesn't rank schools, so I can't pick a good one for you. Name one primary "
                    "school (for example \"Ai Tong School\") and I'll find blocks within 1 km or 1-2 km of it, or tell me "
                    "a town and budget to search without a school limit.")
# The question asked when a commute limit cannot be checked because a workplace was not found on the map (SEM-30)
_WORKPLACE_ASKS = ("Where is “", "Without an address I can't time the commute")

# Questions about the last answer itself, answered by code instead of running the same search again: whether the
# blocks are live listings (SEM-14: "Are these live flats being sold now?" listed the same 37 blocks again), how the
# distances were measured (SPATIAL-06/t2: "Is that actual walking-route distance or a straight line?") and whether a
# kind of place can be checked at all (LIMIT-02/t2: "Can you actually measure swimming-pool proximity?" repeated the
# list and said it met every limit).
_ABOUT_LIVE = re.compile(r"\b(?:live|current|actual|real)\s+(?:listings?|flats?|units?)\b|\bbeing sold\b|"
                         r"\b(?:still|currently|now)\s+(?:available|for sale|on (?:the )?market|listed)\b|\bfor sale (?:now|today)\b|"
                         r"\bare (?:these|they|those) (?:available|listings?)\b", re.I)
_ABOUT_METHOD = re.compile(r"\bwalking[- ](?:route|distance|path)\b.*\bstraight\b|\bstraight[- ]line\b|\bas the crow flies\b|"
                           r"\bhow (?:is|are|was|were) (?:the |those |these )?distances? (?:measured|calculated|worked out)\b", re.I)
_ABOUT_COVERAGE = re.compile(r"\b(?:can|could|do|does) (?:you|it|the app)\s+(?:actually\s+|really\s+)?(?:measure|check|filter|track|know)\b",
                             re.I)
ABOUT_LIVE = ("These are not live listings. Each block is summarised from **past resale transactions** (HDB resale records, "
              "last 12 months: median price and lease of flats that already sold). The app has no feed of flats for sale "
              "now; check current listings on the HDB Flat Portal or a property portal.")
ABOUT_METHOD = ("Those distances are **straight-line** distances between indexed points (a block or place and the nearest "
                "indexed MRT/LRT exit, hawker centre, supermarket or school), not routed walking distances. Real walks are "
                "usually longer; the walking minutes shown assume 80 m a minute along that straight line.")


def _plural(kind: str) -> str:
    kind = kind.lower()
    return kind if kind.endswith("s") else kind[:-1] + "ies" if kind.endswith("y") else \
        kind + "es" if kind.endswith(("ch", "sh")) else kind + "s"


def about_last_answer(query: str, prior: Optional[Dict[str, Any]]) -> Optional[str]:
    """Code's answer to a question about the last search or place answer, or None. Only after a turn that searched or
    looked up a place, and only when the message states nothing new to search: a new search in the same message wins
    ("Are there live listings for 4-room in Bedok?" is a search)."""
    if not prior or not (prior.get("towns") or prior.get("islandwide") or prior.get("addresses") or prior.get("map_pin")) \
            or len(query.split()) > 16:
        return None
    fresh = rule_based_extract(query).model_dump()
    if any(fresh.get(k) for k in SEARCH_KEYS + ("addresses",)):
        return None
    if _ABOUT_LIVE.search(query):
        return ABOUT_LIVE
    if _ABOUT_METHOD.search(query):
        return ABOUT_METHOD
    if _ABOUT_COVERAGE.search(query):
        kinds = [m.group(1) for m in _UNMAPPED_PLACES.finditer(query.replace("-", " "))]
        if kinds:
            kind = kinds[0].lower()
            stem = re.sub(r"(?:ies|s)$", "", kind)   # "swimming pool" / "librar" as the earlier note writes it
            # the earlier list was searched with that limit left out (_note_unchecked_places said so)
            listed = any(stem in a.lower() for a in prior.get("assumptions") or [] if a.startswith("I can't measure"))
            return (f"No. I can't measure distances to {_plural(kind)}: there is no location data for them"
                    + (", so that limit was not applied to the list." if listed else ".")
                    + " I can check distances to MRT/LRT stations, hawker centres, supermarkets and primary schools.")
    return None


def refine_search(carried: Dict[str, Any], fresh: Dict[str, Any], query: str) -> Dict[str, Any]:
    """The earlier search, changed the way a follow-up message asks (no open question): a complete new search starts
    afresh; "also within 500m of a hawker centre" adds a limit; "within 300m" alone changes the one distance limit;
    "remove the MRT limit" removes it; "cheaper" lowers the budget by 10%; a second workplace joins the first.
    Code's own reading of the message is used for these words, so the live model's reading changes nothing here."""
    out = dict(carried)
    out.pop("commute_unlimited", None)
    if not carried.get("towns") and not carried.get("addresses"):
        return out
    if fresh.get("towns") and fresh.get("flat_types") and (
            _NEW_SEARCH.search(query) or fresh.get("max_budget_sgd") or fresh.get("min_remaining_lease_years")):
        for k in SEARCH_KEYS:          # "Find 3-room flats in Clementi": not the last search's budget and MRT limit
            out.pop(k, None)
        return out
    # "Include Pasir Ris too." adds a town; "Switch to Tampines" replaces them (SEM-15: it dropped Tampines)
    if fresh.get("towns") and out.get("towns") and re.search(r"\b(?:include|add|also|too|as well|plus)\b", query, re.I) \
            and not re.search(r"\b(?:instead|switch|change|only|rather than|replace|not)\b", query, re.I):
        fresh["towns"] = list(dict.fromkeys(list(out["towns"]) + list(fresh["towns"])))   # "Include Pasir Ris too"
    bare = _BARE_BUDGET.search(query)
    if bare and not fresh.get("max_budget_sgd") and not _NOT_A_BUDGET.search(query):
        value = _money(bare.group(1))
        if value and 50_000 <= value <= 5_000_000:
            fresh["max_budget_sgd"] = value                                               # "Actually only $850k."
    lease = _BARE_LEASE.search(query)
    if lease and out.get("min_remaining_lease_years") and not fresh.get("min_remaining_lease_years"):
        fresh["min_remaining_lease_years"] = int(lease.group(1))                          # "Actually 60 years will do."
    radii = dict(out.get("amenity_radii_meters") or {})
    new_radii = dict(fresh.get("amenity_radii_meters") or {})
    # "Use MRT stations only instead." keeps the distance and changes which stations count (SPAT-38)
    only = re.search(r"\b(mrt|lrt)(?:\s+stations?)?\s+only\b|\bonly\s+(?:the\s+)?(mrt|lrt)\b", query, re.I)
    rail = next((k for k in ("mrt", "lrt", "mrt_only") if k in radii), None)
    if only and rail and not new_radii:                                                   # "Use MRT stations only instead."
        radii["mrt_only" if (only.group(1) or only.group(2)).lower() == "mrt" else "lrt"] = radii.pop(rail)
    m = _BARE_DISTANCE.search(query)
    if m and not new_radii:
        dist = int(float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1))
        if len(radii) == 1:
            new_radii = {next(iter(radii)): dist}          # "within 300m" after "within 500m of an MRT"
        elif len(radii) > 1:
            names = _join_words([{"mrt": "the MRT", "hawker": "the hawker centre", "supermarket": "the supermarket",
                                  "school": "the school"}.get(k, k) for k in radii], last="or")
            fresh["clarifications"] = list(fresh.get("clarifications") or []) + [
                f"Which limit should be {dist:,} m: {names}? Say, for example, \"MRT within {dist:,}m\"."]
    for rm in _REMOVE.finditer(query):
        if re.search(r"\d", query[rm.start():rm.end() + 12]):
            continue                                      # "no more than 400m of an MRT" is a limit, not a removal
        key, sub = _REMOVES[rm.group(1).lower()]
        if key == "amenity_radii_meters":
            if sub in ("mrt", "lrt"):
                for rail in ("mrt", "lrt", "mrt_only"):   # any rail limit, whichever kind it was read as
                    radii.pop(rail, None)
            elif sub:
                radii.pop(sub, None)
            else:
                radii = {k: v for k, v in radii.items() if k == "school"}
        else:
            out[key] = None
            if key == "school_name":
                radii.pop("school", None)
    if new_radii and not re.search(r"\b(?:instead|only)\b", query, re.I):
        radii.update(new_radii)                           # "also within 500m of a hawker centre" keeps the MRT limit
    elif new_radii:
        radii = new_radii
    out["amenity_radii_meters"] = radii
    if not new_radii:
        fresh["amenity_radii_meters"] = radii             # the merge keeps the fresh (empty) value otherwise
    else:
        fresh["amenity_radii_meters"] = radii
    if re.search(r"^\W*(?:any(?:thing)?\s+)?cheaper\b|\bcheaper (?:ones?|options?|flats?|blocks?)\b|\blower (?:the )?(?:budget|price)\b",
                 query, re.I) and not fresh.get("max_budget_sgd") and out.get("max_budget_sgd"):
        out["max_budget_sgd"] = round(out["max_budget_sgd"] * 0.9, -3)
        fresh["assumptions"] = list(fresh.get("assumptions") or []) + [
            f"Lowered the budget by 10%, to ${out['max_budget_sgd']:,.0f}."]
    relax = re.search(r"\b(?:relax|loosen|increase|raise|bump up|higher)\b(?:\s+(?:the|my|our))?\s+(budget|price|lease|distance|mrt|hawker)", query, re.I)
    if relax:                                             # one relaxation step, the same steps the sweep tries
        steps = load_config()["architecture"]["relaxation_steps"]
        what = relax.group(1).lower()
        if what in ("budget", "price") and out.get("max_budget_sgd") and not fresh.get("max_budget_sgd"):
            out["max_budget_sgd"] = round(out["max_budget_sgd"] * (1 + steps["max_budget_sgd_pct"] / 100), -3)
            note = f"Raised the budget to ${out['max_budget_sgd']:,.0f}."
        elif what == "lease" and out.get("min_remaining_lease_years"):
            out["min_remaining_lease_years"] = max(0, out["min_remaining_lease_years"] - steps["min_remaining_lease_years"])
            note = f"Lowered the minimum lease to {out['min_remaining_lease_years']} years."
        elif radii:
            keys = [k for k in radii if k != "school" and (what == "distance" or k == what)]
            for k in keys:
                radii[k] += steps["amenity_radius_meters"]
            fresh["amenity_radii_meters"] = out["amenity_radii_meters"] = radii
            note = f"Widened the distance limit{'s' if len(keys) > 1 else ''} by {steps['amenity_radius_meters']} m." if keys else ""
        else:
            note = ""
        if note:
            fresh["assumptions"] = list(fresh.get("assumptions") or []) + [note]
    work = list(fresh.get("work_destinations") or [])
    if len(work) == 1 and len(out.get("work_destinations") or []) == 1 and work != out["work_destinations"] \
            and not re.search(r"\binstead\b", query, re.I):
        if out.get("max_commute_minutes") or fresh.get("max_commute_minutes"):
            fresh["work_destinations"] = list(out["work_destinations"]) + work   # "my wife works at Changi Business Park"
        else:
            # no commute limit was stated: the two-workplace check needs one, and a default would cut blocks unasked
            fresh["work_destinations"] = list(out["work_destinations"]) + work
            out["max_commute_minutes"] = None
            fresh["assumptions"] = list(fresh.get("assumptions") or []) + [
                "Tell me the longest commute you'd accept (for example \"under 40 minutes\") and I'll check the travel "
                "time to both workplaces; no commute limit has been applied yet."]
            fresh["commute_unlimited"] = True
    return out


def _pin_radius_followup(c: Dict[str, Any], query: str, prior: Dict[str, Any], fresh: Dict[str, Any]) -> None:
    """"within 1km" after "What's near here?" on a map pin: the same pin and question, with that radius. The same for
    a school lookup around an address and for a nearby answer about one place."""
    m = _BARE_DISTANCE.search(query)
    dist = int(float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1)) if m else None
    if dist and prior.get("nearby_schools") and not (fresh.get("addresses") or fresh.get("towns") or fresh.get("nearby_schools")):
        # "Only within 1km, please." after "Which primary schools are within 2km of Blk 510 Bedok North St 3?": the
        # same address's school lookup with that radius; it got the menu, as the message names no place (SPAT-11)
        c["nearby_schools"] = {**prior["nearby_schools"], "radius_meters": float(dist)}
        c["clarifications"] = []
        return
    radii = {k: v for k, v in (fresh.get("amenity_radii_meters") or {}).items() if k != "school"}
    if prior.get("wants_nearby_amenities") and len(prior.get("addresses") or []) == 1 and (radii or dist) and not (
            fresh.get("flat_types") or fresh.get("max_budget_sgd") or fresh.get("addresses") or fresh.get("map_pin")
            or set(fresh.get("towns") or []) - set(prior.get("towns") or [])):
        # "Which hawker centres are within 700m?" after "Where is Blk 288B Jurong East St 21?": what is near that block
        # within 700 m, not a block search for blocks within 700 m of a hawker centre (SPAT-36: it searched the Central
        # Area, read from "centres are", and opened with "No block meets every limit")
        c.update(addresses=list(prior["addresses"]), wants_nearby_amenities=True, towns=[], amenity_radii_meters={},
                 area_radius_meters=max(radii.values()) if radii else dist)
        c["assumptions"] = [a for a in c.get("assumptions") or []   # a town guess, now unused ("Read “x” as Bedok.")
                            if not (a.startswith("Read “") and any(a.endswith(f" as {t.title()}.") for t in HDB_TOWNS))]
        return
    if not dist or not prior.get("map_pin") or fresh.get("map_pin") or fresh.get("addresses") or fresh.get("towns"):
        return
    c["map_pin"] = prior["map_pin"]
    c["pin_radius_meters"] = dist
    for k in ("wants_nearby_amenities", "wants_area_statistics", "wants_price_trend"):
        if prior.get(k):
            c[k] = True
    if not any(c.get(k) for k in ("wants_nearby_amenities", "wants_area_statistics", "wants_price_trend")):
        c["wants_nearby_amenities"] = True
    c["clarifications"] = [q for q in c.get("clarifications") or [] if not q.startswith(PLACE_QUESTION)]


_INSTALMENT_ASK = re.compile(r"\b(?:monthly\s+)?(?:instal+ments?|repayments?|mortgage payments?|monthly payments?|"
                             r"how much (?:do|would|will) (?:i|we) (?:pay|repay) (?:each|a|per|every) month)\b"
                             # "How much per month for my home loan?", "How much every month for a $420k mortgage" (SEM-12)
                             r"|\bhow much\b[^.?!]{0,30}?\b(?:per|a|each|every) month\b", re.I)
_INSTALMENT_RATE = r"(\d{1,2}(?:\.\d+)?)\s*(?:%|per ?cent\b)"                      # "2.6%", "4 percent", "3 per cent"
# the loan tenure, not a lease or an age: "over 20 years", "25-year"; never "70 years lease", "40 years left"
_INSTALMENT_YEARS = r"\b(\d{1,2})[- ]?(?:years?|yrs?)\b(?!\s*(?:of\s+)?(?:lease|left|remaining|old)\b)"
# an amount: "$600,000 mortgage" is $600,000 (the unit must end the word, or the "m" of "mortgage" read it as millions)
_INSTALMENT_AMOUNT = (r"(?:S?\$\s?)(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?\b(?![\d%])"
                      r"|\b(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)\b")
_INSTALMENT_LOAN = (r"(?:S?\$\s?)?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?\s*(?:hdb |bank |housing |home )?(?:loan|mortgage)\b"
                    r"|(?:loan|borrow\w*|mortgage)\s+(?:of|amount of|is)?\s*(?:S?\$\s?)(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?\b")
# a follow-up that changes one figure of the last instalment: "What if I repay over 20 years instead?", "Use 15 years
# instead.", "Increase rate to 4.5%, leave the rest.", "Same terms but borrow only $550,000.", "If the loan is $400k
# instead?", "what about 30 years at the same rate?" (SEM-05, SEM-12)
_INSTALMENT_FOLLOWUP = re.compile(r"\b(?:instead|same|the rest|what if|what about|how about|and over|if the loan|use|increase|"
                                  r"decrease|raise|lower|change|borrow\w*|repay\w*|tenure|rate|loan)\b", re.I)


def _instalment_figure(m: Optional[re.Match]) -> Optional[float]:
    if not m:
        return None
    num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return _money(f"{num}{unit or ''}")


def _instalment(loan: float, rate: float, years: int) -> Dict[str, Any]:
    """Equal monthly payments; a 0% loan is the loan spread evenly over the months."""
    r, n = rate / 100 / 12, years * 12
    monthly = loan / n if r == 0 else loan * r / (1 - (1 + r) ** -n)
    return {"loan": loan, "rate": rate, "years": years, "monthly": round(monthly), "interest": round(monthly * n - loan)}


def instalment_question(q: str, prior: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """"monthly instalment for a $500k loan at 2.6% over 25 years": worked out by code with the standard loan formula
    (equal monthly payments). No calculator tool answers it; it was the menu before (QA sweep, 3 Nov). Returns {} when
    the question is asked but a figure is missing, None when it is not an instalment question. prior: the last turn's
    instalment, which a follow-up naming one new figure changes ("Use 15 years instead."); the other figures stay."""
    ask = asked_part(q)
    asked = bool(_INSTALMENT_ASK.search(ask)) and not re.search(r"\bdebt\b", ask, re.I)
    m_loan = re.search(_INSTALMENT_LOAN, q, re.I)
    loan = _instalment_figure(m_loan)
    if not loan or loan < 10_000:
        # "$100k at 0% interest": the first amount that can be a loan
        loan = next((v for v in map(_instalment_figure, re.finditer(_INSTALMENT_AMOUNT, q, re.I)) if v and v >= 10_000), None)
    m_rate = re.search(_INSTALMENT_RATE, q, re.I)
    m_years = re.search(_INSTALMENT_YEARS, q, re.I)
    rate = float(m_rate.group(1)) if m_rate else None
    years = int(m_years.group(1)) if m_years else None
    if prior and prior.get("monthly") and not (asked and loan and rate is not None and years) \
            and (loan or rate is not None or years) and _INSTALMENT_FOLLOWUP.search(q):
        rb = rule_based_extract(q)
        # not a search ("Find 4-room in Tampines under $600k instead"), a flat's price ("HDB loan or bank loan for a $500k
        # flat?") or a budget question with household amounts ("We have $100k CPF, what's our max loan?")
        if not (rb.towns or rb.flat_types or rb.addresses) and not any(re.search(p, q, re.I) for p in _FACT_WORDS.values()) \
                and not re.search(r"\b(?:flats?|units?|resale|bto|condos?|house|homes?)\b(?!\s+loans?)", q, re.I) \
                and not re.search(r"\b(?:budget|afford\w*|max(?:imum)?|ceiling|grants?)\b", ask, re.I):
            loan, rate, years = loan or prior["loan"], prior["rate"] if rate is None else rate, years or prior["years"]
            if loan >= 10_000 and 0 <= rate < 20 and 1 <= years <= 35:
                return _instalment(loan, rate, years)
    # "our monthly debt repayments are $500" states a fact; it does not ask for an instalment
    if not asked or not re.search(r"\bloan\b|\bmortgage\b|\bborrow|\brepay|\binstal", q, re.I):
        return None
    if not (loan and rate is not None and years) or loan < 10_000 or not (0 <= rate < 20) or not (1 <= years <= 35):
        return {}
    return _instalment(loan, rate, years)


_BUDGET_ASK = re.compile(r"\b(?:budget|afford\w*|borrow\w*|how much (?:hdb |bank )?loan|max(?:imum)? (?:hdb |bank )?loan|"
                         r"max(?:imum)? (?:purchase )?price|ceiling)\b", re.I)


def asked_part(text: str) -> str:
    """The question in a message, not the facts before it ("Our HDB loan eligibility is confirmed. What grants…?")."""
    parts = [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    asks = [s for s in parts if "?" in s or re.match(r"\s*(?:what|how|can|could|tell|show|find|work out|is|are|do|does)\b", s, re.I)]
    return " ".join(asks) or (parts[-1] if parts else "")


def _search_unchanged(prior: Dict[str, Any], merged: Dict[str, Any], fresh: Dict[str, Any], query: str) -> bool:
    """A short follow-up to a search that changed nothing code can read ("something nicer?"), so the same search
    would run again without saying why."""
    if not prior.get("towns") or prior.get("addresses") or len(query.split()) > 12 or not _REFINE_WORDS.search(query):
        return False
    if any(fresh.get(k) for k in SEARCH_KEYS + ("addresses", "policy_question", "map_pin", "nearby_schools")):
        return False
    if any(v for k, v in fresh.items() if k.startswith("wants_")) or any(stated_financial_inputs(query).values()):
        return False
    if _REMOVE.search(query) or re.search(r"\bcheaper\b|\b(?:relax|loosen|increase|raise)\b", query, re.I):
        return False
    return all((merged.get(k) or None) == (prior.get(k) or None) for k in SEARCH_KEYS)


# the distance limits the spatial index has places for; any other ("within 200m of a park") would pass every block
CHECKABLE_DISTANCES = {"mrt", "lrt", "mrt_only", "mrt_lrt", "hawker", "supermarket", "school"}
_UNMAPPED_PLACES = re.compile(r"\b(parks?|bus stops?|mosques?|churche?s?|temples?|(?:poly)?clinics?|malls?|shopping (?:centre|center|mall)s?|"
                              r"librar(?:y|ies)|community (?:club|centre|center)s?|hospitals?|markets?|gyms?|swimming pools?|playgrounds?)\b", re.I)


def _note_unchecked_places(c: Dict[str, Any], query: str) -> None:
    """A distance to a kind of place the app has no locations for is not applied, and the answer says so, rather
    than listing every block as meeting it (QA sweep, 3 Nov: "within 200m of a park" passed 208 blocks)."""
    radii = c.get("amenity_radii_meters") or {}
    dropped = sorted(k for k in radii if k not in CHECKABLE_DISTANCES)
    if dropped:
        c["amenity_radii_meters"] = {k: v for k, v in radii.items() if k in CHECKABLE_DISTANCES}
    named = [m.group(1).lower() for m in _UNMAPPED_PLACES.finditer(query)
             if re.search(r"\b(?:near|within|close to|next to|walking distance|nearby|\d+\s*k?m)\b", query, re.I)]
    kinds = list(dict.fromkeys([k.replace("_", " ") for k in dropped] + named))
    if kinds and (c.get("towns") or c.get("flat_types") or dropped):
        c["assumptions"] = list(c.get("assumptions") or []) + [
            f"I can't measure distances to these places: {_join_words(kinds)} (there is no location data for them), so that limit was "
            "not applied. Distances can be checked to MRT/LRT stations, hawker centres, supermarkets and schools."]


def workplace_known(name: str) -> bool:
    """Whether the commute tool can place this workplace: a known employment hub or a place the location search finds.
    The tool falls back to Raffles Place for a name it cannot place, so "Unknown Workplace QZX" was timed as Raffles
    Place and the answer promised every limit (SPAT-64, SPAT-65). Interim check until Member 5's tool reports it."""
    try:
        from mcp_server import spatial_onemap_tools as S
        from mcp_server.location_tools import resolve_place
    except Exception:  # noqa: BLE001
        return True
    n = " ".join(str(name).upper().replace("-", " ").split()); nh = " ".join(str(name).upper().split())
    if any(h in n or h in nh for h in S._EMPLOYMENT_HUBS):
        return True
    try:
        return bool((resolve_place(str(name), limit=1) or {}).get("candidates"))
    except Exception:  # noqa: BLE001
        return True


def policy_followup(query: str, prior: Dict[str, Any], c: Dict[str, Any], fresh: Dict[str, Any],
                    blocks_listed: bool = False) -> bool:
    """Whether a message continues the rule question the last turn asked: a question naming no place of its own
    (town, address, pin, BTO launch) that points back at it ("that", "then", "your calculator", "my application") or
    that nothing else would answer. Not a money question ("so how much can we borrow?"): the calculators answer it.
    c: the constraints with the earlier turns merged in; fresh: this message's own; blocks_listed: an earlier answer
    listed blocks."""
    root = str(prior.get("policy_question") or "")
    if not root or root in str(c.get("policy_question") or "") or not _ASKED.search(query):
        return False
    if any(fresh.get(k) or c.get(k) for k in ("towns", "addresses", "map_pin", "bto_project_name")):
        return False
    if _MONEY_AMOUNT_ASK.search(query) or calculator_facts(fresh) or instalment_question(query) is not None:
        return False
    if (blocks_listed and (_LISTED_REF.search(query) or _LIST_WORDS.search(query))) or about_last_answer(query, prior):
        return False   # "which of those blocks is cheapest?": a question about the blocks listed earlier
    return bool(_FOLLOW_REF.search(query)) or not decide_routing({**c, "analysis_mode": False})


def _finalize_constraints(constraints: Dict[str, Any], query: str, state: HDBCompassState, errors: List[str],
                          from_model: bool, with_history: bool = True, model_policy: bool = True) -> Dict[str, Any]:
    """Everything code does to a reading, whichever path produced it (model, regex fallback or the app's fast path),
    so the paths cannot drift apart. with_history=False reads this message alone (the fast path's self-contained
    questions); otherwise earlier turns and any open follow-up question are carried in. model_policy: the model itself
    filled policy_question (not the regex merge that fills what the model left empty)."""
    prior = state.get("prior_constraints") if with_history else None
    pending = state.get("pending_clarification") if with_history else None
    which_limit = re.match(r"Which limit should be ([\d,]+) m", str((pending or {}).get("question") or ""))
    if which_limit:
        # the reply to "Which limit should be 1,000 m: the MRT or the hawker centre?" is read as a follow-up to the
        # search: "the MRT" sets that limit; anything else ("remove the MRT limit") is read as usual
        pending = None
        named = [k for k in mentioned_amenities(query) if k in ((prior or {}).get("amenity_radii_meters") or {})]
        if len(named) == 1 and not (constraints.get("amenity_radii_meters") or {}) and not _REMOVE.search(query):
            constraints["amenity_radii_meters"] = {named[0]: int(which_limit.group(1).replace(",", ""))}
    if (pending or {}).get("kind") == "workplace":
        # the reply to "Where is “QZX”?" is a follow-up to the search ("drop the commute limit" removes the limit, as
        # after any search), and a place it gives ("Raffles Place", "It's at 048616") is that workplace, timed with the
        # one already found
        pending = None
        said = rule_based_extract(query)
        place = next(iter(list(said.work_destinations or []) + list(said.addresses or [])), None)
        if place is None and len(query.split()) <= 6 and not _DONT_KNOW.search(query) and not _REMOVE.search(query):
            words = re.sub(r"^\W*(?:it'?s|it is|that'?s|the address is)?\s*(?:at|in|near|on)?\s+", " ", f" {query}", flags=re.I)
            words = words.strip(" .!?")
            if re.search(r"\b\d{6}\b", words) or [w for w in re.findall(r"\b[A-Z][\w-]*", words)
                                                   if w.lower() not in ("ok", "okay", "thanks", "yes", "no", "sure", "i")]:
                place = words
        if place and (prior or {}).get("unresolved_workplaces"):
            constraints["work_destinations"] = list((prior or {}).get("work_destinations") or []) + [place]
            constraints["addresses"] = []
    answered_money = False
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
    # the model's schema has one rail key ("mrt", nearest MRT or LRT); code's reading says which kind was meant, so
    # "LRT must be within 300m" stays LRT-only on the model path too (SEM-01)
    rb_radii = rule_based_extract(query).amenity_radii_meters or {}
    radii = constraints["amenity_radii_meters"]
    for kind in ("lrt", "mrt_only"):
        if kind in rb_radii and "mrt" in radii and kind not in radii and "mrt" not in rb_radii:
            radii[kind] = radii.pop("mrt")
    _verify_extraction(constraints, query, errors, from_model=from_model)
    _note_unchecked_places(constraints, query)
    strong_facts = any(v is not None and k not in _WEAK_FINANCIAL_FACTS
                       for vals in stated_financial_inputs(query).values() for k, v in (vals or {}).items())
    if from_model and constraints.get("policy_question") and not policy_wording(query, money_stated=calculator_facts(constraints)) \
            and (_MONEY_AMOUNT_ASK.search(query) or strong_facts):
        # "How much grant can we get?" / a reply with household facts: the calculators' question, not a rule search
        # (live, 3 Nov: the model filed "grant_eligibility" and the answer added "I couldn't find an HDB rule").
        # Who the buyer is (citizenship, first-timer) is not such a fact: "I am a single Singapore citizen, 36. What
        # are my HDB buying options?" is a rule question (SEM-08, DEMO-16: it got the menu)
        constraints["policy_question"] = None
        errors.append("extract_constraints: model's policy reading dropped: a money question, not a rule question")
    if from_model and constraints.get("policy_question") and not scope.mentions_policy_topic(query) \
            and not (model_policy and policy_wording(query, money_stated=calculator_facts(constraints))):
        # the model filed a vague question ("Should I buy BTO or resale?") as a policy question; the policy search
        # would return loosely related rules, so the user gets the menu instead (decide_routing). A rule question in
        # words code reads itself ("Can I buy a resale flat after selling my condo?") keeps the model's own reading;
        # code never adds a policy route the model did not choose ("Can I buy a 4-room in Tampines under $600k?").
        constraints["policy_question"] = None
        errors.append("extract_constraints: model's policy reading dropped: the message names no HDB rule")
    if pending and pending.get("kind", "financial") == "financial":
        conversation = " ".join(m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict))
        answers = parse_clarification_answer(query, pending["needed"], conversation)
        answered_money = bool(answers)
        if answers:
            constraints = merge_constraints(constraints, {"financial_inputs": answers})
    # the place an earlier turn named is a candidate even for a self-contained question ("what's near there?")
    _apply_map_pin(constraints, query, state, prior=state.get("prior_constraints"), pending=pending)
    # Keep the current policy question separate from a previous flat-search turn. A question such as
    # "can citizens aged 35 and above buy a 4-room BTO?" may mention a flat type, but it is not a request to
    # search resale blocks. If old towns, budgets or amenity limits are carried into it, the market/spatial agents
    # can produce thousands of unrelated candidate flats alongside the policy answer.
    fresh_constraints = dict(constraints)
    if prior:
        # A new question does not inherit the last one's one-off asks (block history, a snapshot, a clarifying
        # question); a reply to an open question keeps them, since it completes that request.
        carried = {k: v for k, v in prior.items()
                   if k not in ri.ALWAYS_FRESH_KEYS and (pending or k not in ri.TURN_ONLY_KEYS)}
        if not pending:
            carried = refine_search(carried, constraints, query)
        constraints = merge_constraints(carried, constraints)
        if not pending and prior.get("policy_question") and _SHORT_FOLLOWUP.search(query) \
                and (len(str(constraints.get("policy_question") or "").split()) <= 3 or not policy_wording(query)) \
                and not any(fresh_constraints.get(k) for k in SEARCH_KEYS):
            # "what about Plus?" after "Can I rent out my whole Prime flat after the MOP?": the same rule question.
            # Also when the follow-up's own reading is only its topic word ("What if both buyers are PRs?" after "Can a
            # citizen and PR couple get resale housing grants?" is read as a question about PRs: asks_policy)
            constraints["policy_question"] = f"{prior['policy_question']} {query}"
        if not pending and policy_followup(query, prior, constraints, fresh_constraints, bool(state.get("prior_blocks"))):
            # "What does that mean for renting out the whole flat?" after "What is different about Standard, Plus and
            # Prime flats?", "If I turn 35 next year, does that automatically approve my application?" after a singles
            # scheme question: the earlier rule question with this one (SEM-22: the menu, or unrelated excerpts).
            # Set after the checks above that drop a model's policy reading, so they cannot undo it.
            constraints["policy_question"] = f"{str(prior['policy_question'])[:300]} {query}"
    constraints["listed_pick"] = None   # this message's own answer from the last list only (listed_block_reference)
    if pending and pending.get("kind") == "place":
        answer_place_question(constraints, query)
    elif prior and not pending:
        narrow_block_question(constraints, prior, query)
        listed_block_reference(constraints, query, state.get("prior_blocks") or [], fresh_constraints, prior,
                               state.get("prior_block_facts") or [])
        _pin_radius_followup(constraints, query, prior, fresh_constraints)
    # "the same search as before" only when it is: checked after the follow-up readings above, which may have read the
    # message as a question about a listed block or a new radius for the last place, and never for a conditional ask
    # ("If nothing fits, show which one change would help." got the note above 5 matching blocks: SEM-17)
    if prior and not pending and not constraints.get("clarifications") and not constraints.get("listed_pick") \
            and not constraints.get("addresses") and not constraints.get("map_pin") and not constraints.get("nearby_schools") \
            and not _CONDITIONAL_ASK.search(query) and _search_unchanged(prior, constraints, fresh_constraints, query):
        constraints["assumptions"] = list(constraints.get("assumptions") or []) + [UNCHANGED_NOTE]
    if pending and pending.get("kind") == "school" and _DONT_KNOW.search(query) and not constraints.get("school_name"):
        # "I have not picked a school yet." in reply to "Which primary school?": the place lookup ran on "a good primary
        # school" and said no place matches (SEM-30, SPAT-40); say what can be done instead
        constraints["clarifications"] = [NO_SCHOOL_PICKED]
        constraints["addresses"] = []
        constraints["wants_nearby_amenities"] = False

    money_talk = (pending and pending.get("kind", "financial") == "financial") or (
        prior and not pending and not prior.get("policy_question") and has_financial_facts(prior))
    if money_talk and constraints.get("policy_question") == query and not policy_wording(query):
        # a reply to the calculators' question, or a question about their answer, keeps to them: "Neither of us is a
        # citizen; we are both PRs. Can you really use an HDB loan?" and "Does that ceiling already include grants?"
        # are not also rule searches on their topic words (asks_policy), which found no rule and said so
        constraints["policy_question"] = (prior or {}).get("policy_question") if pending else None
    if _EVASION.search(query) and not constraints.get("policy_question"):
        constraints["policy_question"] = query   # answered as a policy question, led by a refusal (EVASION_OPENING)
    if _LIVE_QUOTA_ASK.search(query) and _ASKED.search(query):
        # the quota rules, not the MCPS ballot quota the words "live quota" retrieved (POLICY-07)
        constraints["policy_question"] = LIVE_QUOTA_POLICY_QUERY
    elif _APP_ESTIMATE_ASK.search(query) and _ASKED.search(query):
        constraints["policy_question"] = APP_ESTIMATE_POLICY_QUERY   # what the HFE letter confirms (POLICY-09)
    if constraints.get("policy_question") and not scope.mentions_data(query):
        for key in (
            "towns", "flat_types", "max_budget_sgd", "min_remaining_lease_years", "amenity_radii_meters",
            "mrt_station", "school_name", "work_destinations", "max_commute_minutes", "addresses", "map_pin",
            "wants_price_trend", "wants_bto_comparison", "comparable_resale", "wants_block_history",
            "wants_market_snapshot", "wants_town_ranking", "wants_option_comparison", "wants_price_assessment",
            "wants_nearby_amenities", "wants_area_statistics", "wants_group_stats", "analysis_mode",
            "wants_flat_affordability", "wants_area_affordability", "flat_purchase",
        ):
            constraints[key] = fresh_constraints.get(key)
    # a fact a later message changed in a way code could not read is asked again, not carried over from the last turn
    for vals in (constraints.get("financial_inputs") or {}).values():
        for k in unsure:
            vals.pop(k, None)
    if constraints.get("wants_flat_affordability"):
        _add_flat_purchase(constraints)
        if not constraints.get("flat_purchase"):
            _assume_flat_basics(constraints)   # the block's own facts are unknown: the same 4-room default as any estimate
    elif constraints.get("wants_area_affordability"):
        _add_area_purchase(constraints)
    elif constraints.get("household_income") or any((constraints.get("financial_inputs") or {}).get(sec)
                                                    for sec in ("grants", "affordability")):
        _assume_flat_basics(constraints)   # "how much can we afford?": no flat chosen, so no lease or type to ask for
    # the calculators run when this message is about money (asks it, states a fact, or answers the open money
    # question), not on every later search because an earlier turn gave an income: that repeated the whole household
    # question under each new search
    if re.search(r"\b(?:both|we are|we're|my (?:wife|husband|spouse) and i are)\b[^.]{0,25}\b(?:PRs?|permanent residents?)\b",
                 users, re.I) and not re.search(r"\b(?:one|either)\b[^.]{0,20}\bcitizens?\b|\bSC\b", users, re.I):
        # two Singapore PRs get no CPF housing grants; the budget is still worked out (the grant calculator has no PR
        # couple option, so it asked for citizenship again and again: QA sweep, 3 Nov)
        constraints["pr_couple"] = True
        fin_in = constraints.setdefault("financial_inputs", {}) or {}
        constraints["financial_inputs"] = fin_in
        fin_in.pop("grants", None)
        aff = fin_in.get("affordability") or {}
        bank_note = None
        if any(k in FACT_FIELDS for k in aff if k != "total_grants"):
            # two PRs cannot take an HDB loan, so their budget is a bank loan's: the bank's rate is asked for, and its
            # in-principle approval rather than an "HDB loan eligibility" they cannot have (SEM-20: all three turns of
            # HOUSE-03 asked whether their HDB loan eligibility was confirmed). Only with the household's own facts, not
            # for "What if both buyers are PRs?"
            aff["loan_type"] = "bank"
            if not _BANK_APPROVAL.search(users):
                aff.pop("loan_eligibility_confirmed", None)
            bank_note = ("An HDB loan needs at least one Singapore Citizen buyer, so two PRs can only take a bank loan; "
                         "the budget is worked out for a bank loan.")
        fin_in.setdefault("affordability", aff)["total_grants"] = 0.0
        constraints["assumptions"] = list(constraints.get("assumptions") or []) + [
            "Two Singapore permanent residents get no CPF housing grants, so the budget below uses $0 in grants."] + (
            [bank_note] if bank_note else [])
    grants = (constraints.get("financial_inputs") or {}).get("grants") or {}
    if grants.get("proximity_eligibility_confirmed") is False and grants.get("proximity") in ("near", "with"):
        # the calculator refuses every grant while a PHG is unconfirmed; the other grants do not depend on it
        grants["proximity"] = "none"
        grants.pop("proximity_eligibility_confirmed", None)
        constraints["assumptions"] = list(constraints.get("assumptions") or []) + [
            "The Proximity Housing Grant is left out because your eligibility for it is not confirmed."]
    inst = instalment_question(query, (state.get("prior_constraints") or {}).get("instalment"))
    if inst is None and str((pending or {}).get("question") or "").startswith("To work out the monthly instalment"):
        inst = instalment_question(f"monthly instalment on a loan of {query}")   # "$500k at 2.6% over 25 years"
    # this message's own instalment only: the last one was shown again under a later flat search (X-STALE)
    constraints["instalment"] = inst if inst and "monthly" in inst else None
    if inst is not None and "monthly" not in inst:
        constraints["clarifications"] = [q for q in constraints.get("clarifications") or [] if q != WHICH_TOWN] + [
            "To work out the monthly instalment, tell me the loan amount, the interest rate and the loan tenure "
            "(for example \"$500k at 2.6% over 25 years\")."]
    rng = price_range(query)
    if rng and constraints.get("max_budget_sgd") == rng[1]:
        # the search tools take only an upper price: the lower end is applied to the matches by code (intersect_node)
        constraints["min_price_sgd"] = rng[0]
        constraints["assumptions"] = list(constraints.get("assumptions") or []) + [
            f"Showing blocks whose median price is between ${rng[0]:,.0f} and ${rng[1]:,.0f}."]
    constraints["money_intent"] = bool(_MONEY_AMOUNT_ASK.search(query)) and inst is None   # an instalment is code's
    own = _OWN_GRANT_ASK.search(query)
    if own and own.group(0).lower().endswith(" we") and not calculator_facts(constraints) and inst is None:
        # "What grants can we get?" with no household facts: the calculators ask for them (CLARIFY-03: the policy
        # search alone found no rule, and the answer said so twice instead of asking). "What grants can I get?" stays
        # the policy question it is routed as (tests/test_scope_routing.py)
        constraints["money_intent"] = True
    if prior and not pending and prior.get("bto_project_name") and not constraints.get("bto_project_name") \
            and re.search(r"\b(?:that|this|the same)\s+(?:bto\s+)?(?:project|launch|bto)\b", query, re.I):
        # "Do you really have prices for that project?": the launch the last turn named (SPAT-61)
        constraints["bto_project_name"] = prior["bto_project_name"]
    m = re.search(r"\b(?:can|could) (?:we|i) afford\b[^?.]{0,30}?(?:S?\$\s?)(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?"
                  r"|\b(?:can|could) (?:we|i) afford\b[^?.]{0,30}?\b(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)\b", query, re.I)
    constraints["asked_price"] = None   # this message's own question only
    constraints["budget_asked"] = bool(_BUDGET_ASK.search(asked_part(query)))
    if m:
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        price = _money(f"{num}{unit or ''}")
        constraints["asked_price"] = price if price and price >= 50_000 else None
    # this message itself describes a flat search (not only limits carried from earlier turns)
    constraints["search_stated"] = bool(any(fresh_constraints.get(k) for k in SEARCH_KEYS))
    # an island-wide search stays island-wide in a follow-up ("Within 700m of MRT as well") until a town is named
    constraints["islandwide"] = bool(re.search(r"\b(?:anywhere|island[- ]?wide|all (?:of )?(?:the )?towns|any town|whole (?:of )?singapore|all over singapore)\b",
                                               query, re.I) or (prior or {}).get("islandwide")) and not constraints.get("towns")
    constraints["money_asked"] = bool(
        not prior or answered_money or money_question(query, constraints)
        # a reply to the open money question that code could not read ("We are both PRs.") is still about money
        or (pending and pending.get("kind", "financial") == "financial"
            and not any(fresh_constraints.get(k) for k in SEARCH_KEYS + ("addresses", "policy_question")))
        # "what about a 5 room?" after a grant estimate with no search: the estimate again, for that flat
        or (prior and not prior.get("towns") and not prior.get("addresses") and has_financial_facts(prior)
            and not fresh_constraints.get("towns") and not fresh_constraints.get("addresses"))
        or constraints.get("wants_flat_affordability") or constraints.get("wants_area_affordability")
        or any(stated_financial_inputs(query).values()))
    if inst is not None:
        # a loan instalment question, or a follow-up to one: worked out by code, not a flat search, a budget or the
        # calculators' household questions (SEM-12: "How much per month for my home loan?" asked for the grant facts,
        # "$450k at 2.6 percent, 25 years." got the menu, "If the loan is $400k instead?" asked "Which town?")
        constraints["clarifications"] = [q for q in constraints.get("clarifications") or [] if q != WHICH_TOWN]
        constraints["menu_options"] = None
        for k in ("max_budget_sgd", "trend_years"):
            constraints[k] = None
        constraints["money_intent"] = False
        constraints["search_stated"] = False
        if not re.search(r"\b(?:budget|afford\w*|max(?:imum)?|ceiling|grants?)\b", asked_part(query), re.I):
            constraints["money_asked"] = False
    # a question about the last answer is answered by code from what it showed (about_last_answer); the carried search
    # stays in the constraints, so the next follow-up ("within 300m") still refines it
    constraints["about_results"] = about_last_answer(query, prior) if inst is None else None
    if constraints["about_results"]:
        constraints["assumptions"] = [a for a in constraints.get("assumptions") or [] if a != UNCHANGED_NOTE]
        constraints["clarifications"] = []
        constraints["menu_options"] = None
    # a data question no specific intent recognised goes to the market agent's analysis tools instead of the menu
    constraints["analysis_mode"] = False
    if (with_history and not pending and inst is None and not constraints.get("clarifications") and not constraints.get("place_options")
            and not constraints.get("about_results")
            # a named BTO launch is the launch comparison's question, not a general data question (SPAT-60: "Compare
            # BTO project Imaginary Garden QZX with nearby resale." ran no tool and got the menu). A comparison with no
            # launch named ("Compare BTO vs resale in Sembawang") keeps the resale statistics it was answered with.
            and not constraints.get("bto_project_name")
            and not constraints.get("addresses") and not constraints.get("map_pin") and scope.mentions_data(query)
            and not money_question(query, constraints)
            and addons.offered_in(state.get("variant"), bool(state.get("fast_path")))
            # nothing specific was recognised; a town on its own (which only started a block search) does not count, but
            # a town with a budget, a lease or a distance is a block search ("cheap 4-room flats in Bedok under 500k")
            and not (constraints.get("towns") and (constraints.get("max_budget_sgd") or constraints.get("min_remaining_lease_years")
                                                   or constraints.get("amenity_radii_meters") or constraints.get("school_name")))
            and not decide_routing({**constraints, "towns": []})):
        constraints["analysis_mode"] = True
    # a workplace nobody can locate is not timed (the tool would time Raffles Place instead); the answer says so
    work = constraints.get("work_destinations") or []
    unknown = [w for w in work if not workplace_known(w)]
    if not unknown and prior and not fresh_constraints.get("work_destinations"):
        unknown = list(prior.get("unresolved_workplaces") or [])   # still not located: say so again
    constraints["unresolved_workplaces"] = unknown
    if unknown:
        constraints["work_destinations"] = [w for w in work if w not in unknown]
    found = len(constraints.get("work_destinations") or [])
    if unknown and constraints.get("max_commute_minutes") and found < 2 <= found + len(unknown) \
            and not constraints.get("clarifications") and not constraints.get("about_results"):
        # the commute limit cannot be checked until the workplace is found: ask for it, as a question the next message
        # answers, rather than only noting it under the list (SEM-30, SPAT-40: "I meant a building called QZX; I have no
        # address yet." got the same list with no question). Only when finding it gives the two workplaces the check
        # needs: with one ("I work at Zorgon Tower ... commute under 30 mins") its address would still not be timed
        names = _join_words([f"“{w}”" for w in unknown])
        if prior and prior.get("unresolved_workplaces") and _DONT_KNOW.search(query):
            ask = (f"Without an address I can't time the commute to {names}. Give its postal code or the nearest MRT "
                   "station, or say \"drop the commute limit\" to search without it.")
        else:
            ask = (f"Where is {names}? I couldn't find it on the map, so the commute limit is not applied yet. Give its "
                   "address or postal code and I'll time both commutes.")
        constraints["clarifications"] = [ask]
    # read by code from every user message so far, so an ask in the first turn still counts after a follow-up reply
    constraints["requested_tools"] = requested_tools(users)
    comparables = "query_comparable_resale_transactions"
    if comparables not in constraints["requested_tools"] and coordinates_stated(constraints.get("comparable_resale"), users):
        constraints["requested_tools"].append(comparables)
    if FOUR_ROOM_GRANT_NOTE in (constraints.get("assumptions") or []) and "financial" not in decide_routing(constraints, pending):
        # no grant is worked out, so the note on how grants were worked out is not said (DEMO-16: it stood above the
        # menu after "I am a single Singapore citizen, 36. What are my HDB buying options?")
        constraints["assumptions"] = [a for a in constraints["assumptions"] if a != FOUR_ROOM_GRANT_NOTE]
    return constraints


# A reply to "What would you like to know about <place>?" (research_intents.place_clarification), read as the option it
# picks. The question offers past sales, resale prices around the place, the price trend, a fair-price check and
# amenities; "resale price" on its own is the block's own past sales.
_PLACE_REPLY = (
    # "Where is Blk 288B Jurong East St 21?" locates the block: what is near it, not "What would you like to know?" (SEM-30)
    ("wants_nearby_amenities", r"^\W*where(?:'s|\s+is)\b"),
    ("wants_price_assessment", r"\bfair\b|\breasonable\b|\bover\s*priced\b|\basking\b|\bworth\b|\$\s?\d"),
    ("wants_area_statistics", r"\b(?:around|nearby|near\s+(?:it|there|the\s+block)|within|radius)\b.*\bpric|"
                              r"\bpric\w*\b.*\b(?:around|nearby|within|radius)\b|\barea\b"),
    ("wants_price_trend", r"\btrend\b|\bover\s+(?:the\s+)?(?:years|time)\b|\bgone\s+up\b|\bappreciat"),
    ("wants_nearby_amenities", r"\bamenit|\bnear\s+(?:it|there)\b|\bnearby\b|\bwhat'?s\s+(?:near|around)\b|\bmrt\b|\bschools?\b|"
                               r"\bhawker|\bsupermarket"),
    ("wants_block_history", r"\bpast\s+sales\b|\bsales\b|\bsold\b|\bhistory\b|\btransactions?\b|\bresale\s+pric|\bprices?\b"),
)
_TOWN_FIRST_BLOCK = re.compile(r"^\s*(.+?)\s+(?:blk|block)\.?\s*(\d{1,4}[a-z]?)\s*$", re.I)
# a block number with no street, town or postal code: "Blk 123", "Blk 123 in Singapore"
_BARE_BLOCK = re.compile(r"(?:blk|block)\.?\s*(\d{1,4}[a-z]?)(?:\s+in\s+singapore)?", re.I)


# A question about one block whose answer a short follow-up can narrow ("4 room" after "Resale price for Blk 884?").
_BLOCK_INTENTS = ("wants_block_history", "wants_price_assessment")


def narrow_block_question(c: Dict[str, Any], prior: Dict[str, Any], query: str) -> None:
    """'4 room' straight after a question about one block narrows that block's answer to 4-room flats, instead of
    starting a town-wide search with the town the last turn named. Only a short message that states a flat type and
    nothing else new (no other town, place, budget, lease, distance or question) does this; '4 room in Bedok under
    600k' is a new search, as before. The block and its question are turn-only keys, so they are carried here."""
    asked = [k for k in _BLOCK_INTENTS if prior.get(k)]
    if not (asked and prior.get("addresses")) or len(query.split()) > 6:
        return
    said = rule_based_extract(query).model_dump()
    new_facts = (said.get("addresses") or said.get("max_budget_sgd") or said.get("min_remaining_lease_years")
                 or said.get("amenity_radii_meters") or said.get("school_name") or said.get("work_destinations")
                 or said.get("household_income") or said.get("policy_question")
                 or any(v for k, v in said.items() if k.startswith("wants_") and v)
                 or set(said.get("towns") or []) - set(prior.get("towns") or []))
    if not said.get("flat_types") or new_facts:
        return
    c["addresses"] = list(prior["addresses"])
    for k in asked:
        c[k] = True
    if prior.get("asking_price_sgd"):
        c["asking_price_sgd"] = prior["asking_price_sgd"]
    c["flat_types"] = said["flat_types"]
    c["towns"] = []   # the block is the place; a town would start a town-wide block search


PLACE_QUESTION = "What would you like to know about"   # research_intents.place_clarification


_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4,
             "top": 0, "last": -1}
_LISTED_REF = re.compile(r"\b(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|top|last)\s+(?:one|block|flat|option|result|listing)\b"
                         r"|\b(?:that|this|the same)\s+(?:block|flat|one)\b|\b(?:blk|block)\s+(\d{1,4}[a-z]?)\b(?!\s+\w)", re.I)


_SUPERLATIVE = re.compile(r"\b(cheapest|lowest[- ]priced|least expensive|most expensive|priciest|most lease|longest lease|"
                          r"newest|youngest|nearest|closest)\b", re.I)
_LIST_WORDS = re.compile(r"\b(?:those|these|them|the list|listed|the blocks|the results|which (?:one|block|flat)|of all)\b", re.I)


def _superlative_block(query: str, facts: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], str]]:
    """"Which of those blocks is cheapest?": the block the earlier list ranks first on that measure, and the measure."""
    m = _SUPERLATIVE.search(query)
    if not m or not _LIST_WORDS.search(query) or not facts:
        return None
    word = m.group(1).lower()
    key, best, label = {"cheapest": ("price", min, "the cheapest"), "lowest-priced": ("price", min, "the cheapest"),
                        "lowest priced": ("price", min, "the cheapest"), "least expensive": ("price", min, "the cheapest"),
                        "most expensive": ("price", max, "the most expensive"), "priciest": ("price", max, "the most expensive"),
                        "most lease": ("lease", max, "the one with the most lease left"),
                        "longest lease": ("lease", max, "the one with the most lease left"),
                        "newest": ("lease", max, "the newest (most lease left)"), "youngest": ("lease", max, "the newest (most lease left)"),
                        "nearest": ("dist", min, "the nearest"), "closest": ("dist", min, "the nearest")}[word]
    known = [f for f in facts if isinstance(f.get(key), (int, float))]
    if not known:
        return None
    pick = best(known, key=lambda f: f[key])
    value = {"price": lambda v: f"median ${v:,.0f}", "lease": lambda v: f"{v:.0f} years of lease left",
             "dist": lambda v: f"{v:.0f} m away"}[key](pick[key])
    return pick, f"Of the {len(facts)} blocks listed earlier, {pick['name']} is {label} ({value})."


def listed_block_reference(c: Dict[str, Any], query: str, listed: List[str], fresh: Dict[str, Any],
                           prior: Optional[Dict[str, Any]] = None, facts: Optional[List[Dict[str, Any]]] = None) -> None:
    """"tell me more about the first one", "past sales of that block", "how far is Blk 627 from MRT?" after a search:
    the block the previous answer listed, as the place this message asks about. The place question then reads what
    is asked (past sales, a price check, what is near it) or asks it. Not when the message names its own place."""
    sup = _superlative_block(query, facts or []) if not fresh.get("towns") and not fresh.get("addresses") else None
    if sup:
        # "Which of those blocks is cheapest?" is answered from the list just shown, not with five years of that
        # block's sales (SPAT-27); the block is kept as the place a next question ("stamp duty on that block") means
        c["addresses"] = [sup[0]["name"]]
        for k in SEARCH_KEYS:
            c[k] = {} if k == "amenity_radii_meters" else [] if k in ("towns", "flat_types", "work_destinations") else None
        c["listed_pick"] = sup[1]
        c["assumptions"] = [a for a in c.get("assumptions") or [] if a != UNCHANGED_NOTE]   # it did answer the question
        return
    priced = {f["name"]: f.get("price") for f in facts or []}
    asks_cost = bool(_ASKS_UPFRONT.search(query)) and not (c.get("financial_inputs") or {}).get("upfront", {}).get("purchase_price")
    # the place resolver fills this message's place with the block the last turn named ("Used Blk 35 CHAI CHEE AVE,
    # which you mentioned earlier."): that is the same block, not a new place named here (SPAT-28)
    earlier = set((prior or {}).get("addresses") or [])
    named_here = [a for a in fresh.get("addresses") or [] if a not in earlier]
    if asks_cost and re.search(r"\b(?:that|it|this|the (?:first|second|third|last) one)\b", query, re.I) and not named_here:
        # "What is the stamp duty on that?" after a search: the block meant, priced at its median
        last = [a for a in (prior or {}).get("addresses") or [] if a]
        m = _LISTED_REF.search(query)
        name = (listed[_ORDINALS[m.group(1).lower()]] if m and m.group(1) and abs(_ORDINALS[m.group(1).lower()]) < len(listed)
                else last[0] if len(last) == 1 else listed[0] if len(listed) == 1 else None)
        price = priced.get(name) if name else None
        if name and price:
            fin = dict(c.get("financial_inputs") or {})
            if _stamp_duty_only({"messages": [{"role": "user", "content": query}]}) is None and \
                    not re.search(r"\bcash\b|\bup[- ]?front\b|\bdown ?payment\b|\bcpf\b|\bcosts?\b", query, re.I):
                fin = {k: v for k, v in fin.items() if k == "upfront"}   # the duty alone: no grant or loan questions
            if re.search(r"\bstamp dut|\bbsd\b", query, re.I) and not re.search(r"\bcash\b|\bup[- ]?front\b|\bcpf\b", query, re.I):
                fin = {k: v for k, v in fin.items() if k == "upfront"}
            fin["upfront"] = {**(fin.get("upfront") or {}), "purchase_price": float(price)}
            c["financial_inputs"] = fin
            for k in SEARCH_KEYS:      # a question about one block's cost, not the search again
                c[k] = {} if k == "amenity_radii_meters" else [] if k in ("towns", "flat_types", "work_destinations") else None
            c["assumptions"] = [a for a in c.get("assumptions") or [] if not a.startswith(f"Used {name}")] + [
                f"Worked out for {name} at its median resale price of ${price:,.0f}."]
            c["money_intent"] = True
            c["addresses"] = []                    # the cost of the block, not its sales history again (SPAT-28)
            for k, _ in _PLACE_REPLY:
                c[k] = False
        elif listed:
            c["clarifications"] = list(c.get("clarifications") or []) + [
                "Which block do you mean? Say, for example, \"the first one\" or its block number, or give the price."]
        return
    bare = [re.fullmatch(r"\s*(?:blk|block)?\.?\s*(\d{1,4}[a-z]?)\s*", str(a), re.I) for a in fresh.get("addresses") or []]
    if bare and all(bare) and listed:
        # the model copied "blk 627" from "is $650k fair for blk 627?": the block just listed with that number
        query = f"{query} blk {bare[0].group(1)}"
        fresh = {**fresh, "addresses": []}
        c["addresses"] = []
    if fresh.get("addresses") or fresh.get("map_pin") or fresh.get("towns"):
        return
    last = [a for a in (prior or {}).get("addresses") or [] if a]
    if len(last) == 1 and re.search(r"\b(?:it|its|there|that (?:block|flat|one|place)|the same (?:block|flat|place))\b", query, re.I) \
            and not re.search(r"\b(?:first|second|third|1st|2nd|3rd|last|top)\s+(?:one|block|flat)\b", query, re.I):
        listed = last                     # "how far is it from MRT" after a question about one block: that block
    if not listed:
        return
    m = _LISTED_REF.search(query) or (re.search(r"\b(?:it|its|there)\b", query, re.I) if listed == last else None)
    if not m:
        return
    if m.re is not _LISTED_REF:
        block = listed[0]
    elif m.group(1):
        i = _ORDINALS[m.group(1).lower()]
        block = listed[i] if -len(listed) <= i < len(listed) else None
    elif m.group(2):
        block = next((b for b in listed if b.split()[1].upper() == m.group(2).upper()), None)
    else:
        block = listed[0] if len(listed) == 1 else None
    if block is None:
        if not m.group(2):
            c["clarifications"] = list(c.get("clarifications") or []) + [
                "Which block do you mean? Say, for example, \"the first one\" or its block number."]
        return
    c["addresses"] = [block]
    for k in SEARCH_KEYS:
        c[k] = None if k not in ("amenity_radii_meters", "towns", "flat_types", "work_destinations") else (
            {} if k == "amenity_radii_meters" else [])
    answer_place_question(c, query, default=False)
    if not any(c.get(k) for k, _ in _PLACE_REPLY) and re.search(r"\b(?:how far|distance|near|mrt|hawker|amenit)", query, re.I):
        c["wants_nearby_amenities"] = True
    elif not any(c.get(k) for k, _ in _PLACE_REPLY) and re.search(r"\b(?:more|details?|about|tell me)\b", query, re.I):
        c["wants_block_history"] = True


def answer_place_question(c: Dict[str, Any], query: str, default: bool = True) -> None:
    """The user answered the open question about a place: read the reply as the option it picks, so the next turn
    answers it instead of asking the same question again. "Tampines blk 884" is the block "Blk 884 Tampines", the
    order the block lookups read. A reply no option matches gets the place's past sales (a block) or what is near
    it (any other place) rather than the question again."""
    c["addresses"] = [_TOWN_FIRST_BLOCK.sub(lambda m: f"Blk {m.group(2)} {m.group(1)}", a) for a in c.get("addresses") or []]
    if not c["addresses"] and not c.get("map_pin"):
        return
    bare = [m for m in (_BARE_BLOCK.fullmatch(a.strip()) for a in c["addresses"]) if m]
    named_town = bool(bare) and len(c.get("towns") or []) == 1
    if named_town:
        # the reply to "Which town or street is it on?" names the town: that town's block
        c["addresses"] = [f"Blk {bare[0].group(1).upper()} {c['towns'][0].title()}"]
        c["towns"] = []
    elif bare:
        # "Where is Blk 123 in Singapore?": many towns have a Blk 123, so ask where it is rather than what to look up,
        # and do not show the past sales of a namesake block after "I don't know the street or postal code." (SEM-30,
        # SPAT-58: it answered with an internal "Ask for the block number and street" message)
        block = f"Blk {bare[0].group(1).upper()}"
        c["clarifications"] = [
            f"Without a street or postal code I can't tell which {block} you mean: many towns have one. Which town is it "
            "in? You can also drop a pin on the map." if _DONT_KNOW.search(query) else
            f"Which {block} do you mean? Many towns have one. Which town or street is it on, or its postal code? You "
            "can also drop a pin on the map."]
        for k, _ in _PLACE_REPLY:
            c[k] = False
        return
    if any(c.get(k) for k, _ in _PLACE_REPLY):
        c["clarifications"] = [q for q in c.get("clarifications") or [] if not q.startswith(PLACE_QUESTION)]
        return
    picked = next((k for k, pat in _PLACE_REPLY if re.search(pat, query, re.I)), None)
    if picked is None and not default:
        return   # the message names a place and asks nothing about it: the question stays
    c["clarifications"] = [q for q in c.get("clarifications") or [] if not q.startswith(PLACE_QUESTION)]
    if picked is None:
        is_block = any(re.match(r"(?:blk\s+)?\d", a, re.I) for a in c["addresses"])
        picked = "wants_block_history" if is_block else "wants_nearby_amenities"
        c["assumptions"] = list(c.get("assumptions") or []) + ([] if named_town else [
            "Your reply did not match one of the options I offered, so this shows "
            + ("the block's past sales." if is_block else "what is near it.")])
    c[picked] = True


def _add_flat_purchase(c: Dict[str, Any]) -> None:
    """'Can we afford Blk X?': the block's price and remaining lease, read from the sales data, go to the calculators
    as facts about that flat (what the user stated wins), and the source of the price is disclosed."""
    facts = ri.flat_purchase_facts(c)
    if not facts:
        return
    c["flat_purchase"] = facts
    fin = c.get("financial_inputs") or {}
    c["financial_inputs"] = fin
    fin.setdefault("upfront", {}).setdefault("purchase_price", facts["purchase_price"])
    fin.setdefault("grants", {}).setdefault("flat_type", facts["flat_type"])
    lease = facts.get("remaining_lease_years")
    if lease and lease > 20:
        for sec in ("grants", "affordability"):
            fin.setdefault(sec, {}).setdefault("remaining_lease_years", lease)
    note = f"Price for {facts['block']} taken as ${facts['purchase_price']:,.0f}: {facts['price_source']}."
    if lease:
        note += f" Remaining lease about {lease:g} years (from the block's latest sale)."
    c.setdefault("assumptions", []).append(note)


def _add_area_purchase(c: Dict[str, Any]) -> None:
    """'Can I afford a resale near Novena MRT?': no one flat is named, so the calculators work out the household's
    ceiling for a flat whose lease lasts until the youngest buyer turns 95 (the case they model; an older flat's CPF use,
    loan and grants are prorated), and that ceiling then filters the area's blocks (intersect_node). Both are disclosed."""
    fin = c.get("financial_inputs") or {}
    c["financial_inputs"] = fin
    if not c.get("flat_types"):
        c["flat_types"] = ["4 ROOM"]   # one flat type, so the prices compared with the ceiling are like for like
        c.setdefault("assumptions", []).append("Prices and grants are for 4-room flats; name a flat type to change this.")
    fin.setdefault("grants", {})["flat_type"] = c["flat_types"][0]
    _assume_flat_basics(c)


FOUR_ROOM_GRANT_NOTE = ("Grants were worked out for a 4-room flat (a 5-room or larger flat gets a smaller Family Grant); "
                        "name a flat type to change this.")


def _assume_flat_basics(c: Dict[str, Any]) -> None:
    """A budget question with no flat chosen: rather than asking which flat type and what lease to assume (a flat the
    user has not picked has neither), the grants use their stated flat type or a 4-room one, and the lease is taken to
    last until the youngest buyer turns 95 (the case the calculators model). Each choice is disclosed; a stated
    flat type or lease wins."""
    fin = c.get("financial_inputs") or {}
    c["financial_inputs"] = fin
    grants = fin.setdefault("grants", {})
    if not grants.get("flat_type"):
        grants["flat_type"] = (c.get("flat_types") or ["4 ROOM"])[0]
        if not c.get("flat_types"):
            c.setdefault("assumptions", []).append(FOUR_ROOM_GRANT_NOTE)
    ages = [v.get("youngest_buyer_age") for v in fin.values() if isinstance(v, dict) and v.get("youngest_buyer_age")]
    stated = c.get("min_remaining_lease_years")
    if stated and not c.get("wants_area_affordability") and \
            not any((fin.get(sec) or {}).get("remaining_lease_years") for sec in ("grants", "affordability")):
        # "Buying 4-room resale with 40 years left": that lease, not the age-95 one (SEM-21: it assumed 68 years)
        for sec in ("grants", "affordability"):
            if fin.get(sec) is not None:
                fin[sec]["remaining_lease_years"] = float(stated)
    if any((fin.get(sec) or {}).get("remaining_lease_years") for sec in ("grants", "affordability")) \
            and not c.get("wants_area_affordability"):
        # the user stated a lease. One that ends before the youngest buyer turns 95 is a case the calculators do not
        # model; said every turn, and no fact is asked for that could not change it (clarification_needed)
        lease = min(float((fin.get(sec) or {}).get("remaining_lease_years") or 999) for sec in ("grants", "affordability"))
        if ages and min(ages) + lease < 95:
            c["short_lease"] = {"lease": lease, "age": min(ages)}
            c.setdefault("assumptions", []).append(
                f"A flat with {lease:g} years of lease left runs out before the youngest buyer turns 95 ({min(ages):g} + "
                f"{lease:g} = {min(ages) + lease:g}). HDB prorates the CPF use, loan and grants for such a flat; this estimate "
                "doesn't model it, so it gives no grant amount or ceiling for this flat.")
        return
    if ages:
        lease = max(95 - int(min(ages)), 20)
        for sec in ("grants", "affordability"):
            fin.setdefault(sec, {})["remaining_lease_years"] = lease
        c.setdefault("assumptions", []).append(
            f"The ceiling assumes a flat with at least {lease} years of lease left (until the youngest buyer turns 95). "
            "For a flat with less, HDB reduces the CPF you can use, the loan and the grants, so your ceiling there is lower.")


FAST_PATH_SOURCE = "regex:fast_path"
# free-text fields of the model's schema that the answer prints verbatim: code writes them, the model's are dropped
MODEL_FREE_TEXT_FIELDS = ("assumptions", "clarifications")


def _apply_map_pin(c: Dict[str, Any], query: str, state: HDBCompassState, prior: Optional[Dict[str, Any]] = None,
                   pending: Optional[Dict[str, Any]] = None) -> None:
    """Which place the message is about: the user's map pin, a place it names, or one from earlier turns
    (research_intents.resolve_place asks when the pin and an earlier place are both plausible)."""
    map_state = state.get("map_ui_payload") or {}
    pin = map_state.get("center")
    pin = (float(pin[0]), float(pin[1])) if isinstance(pin, (list, tuple)) and len(pin) == 2 else None
    earlier = list((prior or {}).get("addresses") or [])
    ri.resolve_place(c, query, pin, earlier, pending, radius_meters=map_state.get("radius_meters"))


# Words that mention an amenity beyond AMENITY_WORDS: a station or LRT is a rail stop, a food centre is a hawker centre
_AMENITY_MENTION = {"mrt": r"\b(?:mrt|lrt|train|stations?)\b", "hawker": r"\b(?:hawkers?|food\s+cent(?:re|er)s?|kopitiam)\b",
                    "supermarket": r"\b(?:supermarkets?|grocer(?:y|ies)|ntuc|fairprice|giant|sheng\s+siong|cold\s+storage)\b",
                    "bus_stop": r"\bbus(?:\s+stops?|\s+interchange)?\b", "park": r"\bparks?\b(?!\s+primary)",
                    "school": r"\bschools?\b|\bprimary\b", "clinic": r"\b(?:clinics?|polyclinics?|gp)\b"}


def mentioned_amenities(text: str) -> set:
    """The amenity keys a message mentions in any words, plus any the regex reading gives a distance to."""
    found = {k for k, pat in _AMENITY_MENTION.items() if re.search(pat, text, re.I)}
    return found | set((rule_based_extract(text).amenity_radii_meters or {}).keys())


def _station_key(name: Any) -> str:
    """'Novena MRT Station', 'NOVENA' and 'novena mrt' are the same station."""
    return " ".join(re.sub(r"\b(?:mrt|lrt|station)\b|[^a-z0-9]+", " ", str(name or "").lower()).split())


def _amount_in(value: Any, text: str) -> bool:
    """Whether a money amount appears in the text in any usual form ("$650,000", "650k", "0.65m", "650 thousand")."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    # a bare number counts only as money with a $ sign or thousands commas: "520884" is a postal code, not $520,884
    if re.search(rf"(?:S?\$|SGD)\s?(?:{v:,.0f}|{v:.0f})(?![\d,])|(?<![\d,]){v:,.0f}(?![\d,])", text, re.I) and v >= 1000:
        return True
    t = re.sub(r"[$,]|\bS\$|\bSGD\b", "", text, flags=re.I).lower()
    forms = {f"{v / 1000:g}k", f"{v / 1000:g} k", f"{v / 1000:g} thousand", f"{v / 1e6:g}m", f"{v / 1e6:g} m",
             f"{v / 1e6:g} million", f"{v / 1e6:g} mil"}
    return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", t) for f in forms)


def _drop_unstated(c: Dict[str, Any], query: str, errors: List[str]) -> None:
    """The model's reading keeps a limit, place or fact only when the user's words contain it. A runaway reading
    (live, 3 Nov: 9 turns where every field was filled) invented a lease, a commute limit, a school, an income and
    every intent, and the search then found nothing."""
    rb = rule_based_extract(query)
    dropped = []
    if c.get("min_remaining_lease_years") and not re.search(rf"\b{int(c['min_remaining_lease_years'])}\b", query) \
            and not rb.min_remaining_lease_years:
        dropped.append("min_remaining_lease_years"); c["min_remaining_lease_years"] = None
    if c.get("max_commute_minutes") and not re.search(rf"\b{int(c['max_commute_minutes'])}\b", query):
        dropped.append("max_commute_minutes"); c["max_commute_minutes"] = None
    for k in ("max_budget_sgd", "household_income", "cpf_oa_balance", "cash_savings"):
        if c.get(k) and not _amount_in(c[k], query) and getattr(rb, k, None) != c[k]:
            dropped.append(k); c[k] = None
    words = lambda s: [w for w in re.findall(r"[a-z]{4,}", str(s).lower()) if w not in ("school", "primary", "high", "secondary")]
    if c.get("school_name") and not any(w in query.lower() for w in words(c["school_name"])):
        dropped.append("school_name"); c["school_name"] = None
        (c.get("amenity_radii_meters") or {}).pop("school", None)
    if c.get("bto_project_name") and not any(w in query.lower() for w in words(c["bto_project_name"])):
        dropped.append("bto_project_name"); c["bto_project_name"] = None
    if c.get("work_destinations"):
        kept = [w for w in c["work_destinations"] if any(x in query.lower() for x in words(w) or [str(w).lower()])]
        if kept != list(c["work_destinations"]):
            dropped.append("work_destinations"); c["work_destinations"] = kept
    flags = [k for k, v in c.items() if k.startswith("wants_") and v]
    if len(flags) > 3:
        # a question asks for one or two things; a reading that wants everything is a runaway: code's intents instead
        for k in flags:
            c[k] = bool(getattr(rb, k, False))
        dropped.append(f"{len(flags)} intents")
    if dropped:
        errors.append(f"extract_constraints: model's reading not in the message was dropped: {dropped}")


def _verify_extraction(c: Dict[str, Any], query: str, errors: List[str], from_model: bool = True) -> None:
    """Drop extracted facts the user did not actually supply."""
    comp = c.get("comparable_resale")
    if from_model and comp and not all(re.search(re.escape(f"{float(comp[k]):.3f}")[:-1], query) for k in ("latitude", "longitude")):
        c["comparable_resale"] = None
        errors.append("extract_constraints: comparable coordinates were not in the message; dropped")
    if from_model:
        _drop_unstated(c, query, errors)
    if from_model and c.get("towns"):
        # the same names, aliases and short forms ("JE", "AMK", "TPY") the prompt teaches and the regex reads; a model
        # that splits "Kallang/Whampoa" into KALLANG and WHAMPOA gets the HDB town both names stand for. A region the
        # user named ("in the east", "the north-east region") states its towns (research_intents.REGION_TOWNS): the model
        # that read "in the east" as Bedok, Pasir Ris and Tampines read it right (Experiment 1, 2 Nov: A1-05, A2-04, A3-05)
        # not a town named only as a workplace (SPAT-33: the live model searched Tampines after "Change the workplaces
        # to Changi Business Park and Tampines.")
        where = without_workplaces(query)
        stated = set(named_towns(where, any_case_short=True)) | set(stated_towns(where, any_case_short=True)[0])
        region = ri.region_towns(query)
        if region:
            stated |= set(region[1])
        kept, invented = [], []
        for t in c["towns"]:
            same = [t] if t in stated else [s for s in named_towns(str(t), any_case_short=True) if s in stated]
            kept.extend(s for s in same if s not in kept)
            if not same:
                invented.append(t)
        c["towns"] = kept or [t for t in stated_towns(where)[0]]   # "CLEMENI", "THE EAST": code's own reading
        if invented:
            errors.append(f"extract_constraints: towns not named in the message were dropped: {invented}")
    if from_model and c.get("wants_group_stats") and c.get("amenity_radii_meters"):
        # "Resale prices within 300m of an MRT" groups sales by distance; it is not a block search. Code's own reading
        # leaves these distances out for a grouped question (research_intents.extract_research_fields), so the model's
        # reading does too: on 2 Nov it kept MRT 300 m and C listed 2,657 blocks above the statistics (A1-04).
        grouped = {k: v for k, v in c["amenity_radii_meters"].items() if k in ("mrt", "hawker", "supermarket", "school")}
        if grouped:
            c["amenity_radii_meters"] = {k: v for k, v in c["amenity_radii_meters"].items() if k not in grouped}
            errors.append(f"extract_constraints: distances of a grouped statistics question are its groups, not limits: {sorted(grouped)}")
    if from_model and c.get("amenity_radii_meters"):
        # "within 1km of Meridian Primary School" is one limit: a model that also sets MRT, hawker, supermarket, bus
        # stop and park limits of 1,000 m cuts blocks the user never asked to cut. A limit is kept only for an amenity
        # the user's words mention.
        mentioned = mentioned_amenities(query)
        extra = sorted(k for k in c["amenity_radii_meters"] if k not in mentioned)
        if extra:
            c["amenity_radii_meters"] = {k: v for k, v in c["amenity_radii_meters"].items() if k in mentioned}
            errors.append(f"extract_constraints: distance limits for amenities not in the message were dropped: {extra}")
    if c.get("addresses"):
        # "Tampines Blk 884" (the model's reading or the user's words) is the block the lookups read as "Blk 884 Tampines"
        c["addresses"] = [town_first_block(a) for a in c["addresses"]]
    if from_model and c.get("flat_types"):
        # the live model read "Resale price for Tampines Blk 884?" as flat type "BLK 884", which every search refuses
        kinds = [t for t in c["flat_types"] if str(t).upper() in FLAT_TYPES]
        if len(kinds) < len(c["flat_types"]):
            errors.append(f"extract_constraints: flat types that are not HDB flat types were dropped: "
                          f"{[t for t in c['flat_types'] if str(t).upper() not in FLAT_TYPES]}")
        c["flat_types"] = [str(t).upper() for t in kinds]
    if from_model and c.get("mrt_station") is not None:
        # "within 400m of an MRT station" in Ang Mo Kio names no station: a model that fills in the town's station
        # (or "not specified") would measure every block to that one station. Kept only when code reads the same
        # named station in the user's words ("near Novena MRT").
        said = ri.named_mrt_station(query)   # the station index's own spelling
        if str(c["mrt_station"]).strip() and (not said or _station_key(said) != _station_key(c["mrt_station"])):
            errors.append(f"extract_constraints: MRT station not named in the message was dropped: {c['mrt_station']!r}")
        c["mrt_station"] = said
    if from_model and c.get("nearby_schools") and school_location_request(query) is None:
        # a school lookup around an address replaces the whole search (decide_routing), so it needs the same words
        # from the user that the code path reads ("schools near Blk 123 Bishan St 12"), not "flats near a school"
        c["nearby_schools"] = None
        errors.append("extract_constraints: model's nearby-schools lookup dropped: the message asks for no schools near a place")
    if ri.generic_school_phrase(c.get("school_name")):
        c["school_name"] = None
        (c.get("amenity_radii_meters") or {}).pop("school", None)
        c["clarifications"] = list(dict.fromkeys((c.get("clarifications") or []) + [ri.SCHOOL_CLARIFICATION]))
    # Distance phrases ("200m of a hawker centre") are constraints, not places; nor is a kind of school ("Show flats near
    # a good primary school": the place lookup then said no place matches "a good primary school", SEM-30)
    c["addresses"] = [a for a in c.get("addresses") or [] if a.strip() and not re.match(r"\s*\d+(?:\.\d+)?\s*k?m\b", a, re.I)
                      and not ri.generic_school_phrase(a)]


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
    if c.get("about_results"):
        return []   # "Are these live flats being sold now?": answered from the last answer, nothing is searched again
    if c.get("place_options"):
        return []   # the user must first say which place they mean (research_intents.resolve_place)
    if c.get("nearby_schools"):
        return ["spatial"]
    active: List[str] = []
    if c.get("comparable_resale") or ((c.get("towns") or c.get("islandwide")) and (c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("flat_types")
                           or c.get("wants_price_trend") or c.get("wants_bto_comparison"))):
        active.append("market_sql")
    # a commute check times the blocks of a search: two workplaces with no town or block would time the whole island
    # (QA sweep, 3 Nov: a long message mentioning where a couple works listed 1,832 blocks)
    if c.get("amenity_radii_meters") or c.get("school_name") or (
            len(c.get("work_destinations") or []) >= 2 and not c.get("commute_unlimited") and (
                c.get("towns") or c.get("addresses") or c.get("flat_types") or c.get("max_budget_sgd")
                or c.get("min_remaining_lease_years") or not c.get("policy_question"))):
        active.append("spatial")
    if ri.needs_market_agent(c) and "market_sql" not in active:
        active.append("market_sql")
    if ri.needs_spatial_agent(c) and "spatial" not in active:
        active.append("spatial")
    financial_facts = has_financial_facts(c) if (pending or {}).get("kind") == "financial" else calculator_facts(c)
    if c.get("money_asked", True) and (
            financial_facts or c.get("household_income") or c.get("cpf_oa_balance")
            or (c.get("citizenship_scheme") and c.get("max_budget_sgd"))
            or c.get("wants_area_affordability") or c.get("wants_flat_affordability") or c.get("money_intent")):
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


WHICH_TOWN = ("Which town or area should I search? For example \"Tampines\", \"Ang Mo Kio\" or \"the east\" "
              "(I'll keep your other limits).")


def _distance_label(key: str, value: float) -> str:
    """'mrt_dist_m' 379 -> 'mrt 379m' (the form the auditor reads); 'commute_a_mins' 41 -> 'commute to workplace 1:
    41 min', not 'commute_a_mins 41m', which read as metres."""
    m = re.match(r"commute_([a-z])_mins?$", key)
    if m:
        return f"commute to workplace {ord(m.group(1)) - 96}: {value:.0f} min"
    # "mrt_only_dist_m" is shown as "mrt 250m", the form the auditor and the answer already use
    return f"{key.replace('_dist_m', '').replace('mrt_only', 'mrt')} {value:.0f}m"


def unplaced_distance(c: Dict[str, Any]) -> bool:
    """A distance limit with no place to search around: no town, area, address, pin or named station."""
    return bool(c.get("amenity_radii_meters")) and not (
        c.get("towns") or c.get("islandwide") or c.get("addresses") or c.get("map_pin") or c.get("mrt_station"))


def route_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    if c.get("about_results"):
        return {"active_subagents": []}   # a question about the last answer, answered by code (template_response)
    active = decide_routing(c, state.get("pending_clarification"))
    if "spatial" in active and unplaced_distance(c) \
            and "spatial" not in decide_routing({**c, "amenity_radii_meters": {}}, state.get("pending_clarification")):
        # a distance limit filters the blocks of a place: with no town yet ("4rm tamp under 650k can?" then "near mrt
        # 700m also") the map agent ran over all 2,161 blocks island-wide; ask "Which town?" below instead (SEM-33,
        # SPAT-52). decide_routing itself still names the map agent for the reading ("Flats near an MRT").
        active = [a for a in active if a != "spatial"]
    if not active and not c.get("clarifications") and not c.get("listed_pick") and (c.get("addresses") or c.get("map_pin")):
        # "Resale price for Blk 884 Tampines?" names the place and already says what it wants: answer, don't ask
        picked = {**c}
        answer_place_question(picked, last_user_query(state), default=False)
        active = decide_routing(picked, state.get("pending_clarification"))
        if active or picked.get("clarifications"):
            # the reading asked its own, more specific question ("Which Blk 123 do you mean?"): keep it
            return {"active_subagents": active, "user_constraints": picked}
        return {"active_subagents": active, "user_constraints": {**c, "clarifications": [ri.place_clarification(c)]}}
    if not active and not c.get("clarifications") and not c.get("towns") and not c.get("policy_question") \
            and not c.get("islandwide") and not c.get("instalment") \
            and c.get("search_stated", True) and not (c.get("money_asked") and has_financial_facts(c)) and (
            c.get("flat_types") or c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("amenity_radii_meters")):
        # "I want a 4-room flat under $600k": a search with no place yet, so ask where rather than show the menu
        return {"active_subagents": active, "user_constraints": {**c, "clarifications": [WHICH_TOWN]}}
    if not active and (c.get("instalment") or c.get("listed_pick")):
        return {"active_subagents": active}   # worked out by code; the answer shows it (facts section)
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


def addon_prompt(agents: List[str], variant: Optional[str], fast_path: bool = False) -> str:
    """The add-on tools' prompt lines for an agent holding these owners' tools, when this run is offered the add-ons
    ("" otherwise). C's specialists get their own owner's lines (AGENT_SYSTEM); A's monolith and B's data agent hold
    several owners' tools, so they get all of those owners' lines. Under addons.offer_in: all each variant's model is
    then told the same about the same tools (Experiment 1 compares architectures, not prompts about tools)."""
    if not addons.offered_in(variant, fast_path):
        return ""
    return "".join(addons.prompt_section(agent) for agent in agents)


def analysis_prompt(c: Dict[str, Any], variant: Optional[str], fast_path: bool = False) -> str:
    """The data guide C's market agent gets in analysis mode (a data question no specific intent covers), for any
    variant whose extraction set analysis_mode: C's market agent and B's data agent. Variant A has no extraction
    step, so it never gets one (see arch_1_monolith)."""
    if not (c.get("analysis_mode") and addons.offered_in(variant, fast_path)):
        return ""
    return "\n\n" + addons.analysis_section("market_sql")


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


def search_args(name: str, args: Dict[str, Any], c: Dict[str, Any], errors: List[str], label: str) -> Dict[str, Any]:
    """A search call as the model wrote it, held to the limits code read and checked: the model chooses the call, but
    not a different budget, lease, town, flat type or distance (live, 3 Nov: a monthly income of $9,000 passed as the
    price cap; the MRT limit left out of "also within 500m of a hawker"; a 1-2 km ring for "within 2km")."""
    out = dict(args)
    if name == "query_resale_blocks_by_criteria" and not out.get("block_keys"):
        want = {"max_price": c.get("max_budget_sgd") or NO_PRICE_LIMIT, "min_lease_years": c.get("min_remaining_lease_years") or 0}
        if c.get("towns"):
            want["towns"] = list(c["towns"])
        if c.get("flat_types"):
            want["flat_types"] = list(c["flat_types"])
    elif name == "filter_blocks_by_amenity_proximity":
        radii = {k: v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"}
        want = {"amenity_constraints": radii} if radii else {}
        if c.get("towns"):
            want["towns"] = list(c["towns"])
    elif name == "verify_primary_school_distance" and c.get("school_name"):
        r = (c.get("amenity_radii_meters") or {}).get("school") or 1000
        want = {"school_name": c["school_name"], "bracket": "2km" if r >= 2000 else "1km"}
    else:
        return out
    changed = sorted(k for k, v in want.items() if out.get(k) != v)
    if changed:
        errors.append(f"{label}: {name} arguments held to the stated limits: {', '.join(changed)}")
        out.update(want)
    return out


def statistics_args(args: Dict[str, Any], c: Dict[str, Any], errors: List[str], label: str) -> Dict[str, Any]:
    """'Resale prices within 300m of an MRT' compares sales within 300 m with the rest: one band at the distance the
    user named, the bands code's own plan passes. A model that grouped by its default bands (0-200, 200-500, ...) and
    capped the distance answered a different question (A1-04, 2 Nov), so code puts the user's distance back."""
    d = c.get("stats_distance_m")
    if not (d and "distance_band" in (args.get("group_by") or []) and not c.get("stats_bands")):
        return args
    if list(args.get("bands") or []) == [d] and not args.get("max_distance_m"):
        return args
    errors.append(f"{label}: statistics grouped at the {d} m the user named, not the model's bands")
    return {**{k: v for k, v in args.items() if k != "max_distance_m"}, "bands": [d]}


NO_PRICE_LIMIT = 5_000_000   # a block search with no stated budget: above every resale price on record


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
        if (towns or c.get("islandwide")) and ri.is_block_search(c, query):
            plan.append(("query_resale_blocks_by_criteria", dict(
                towns=towns, flat_types=c.get("flat_types") or ["4 ROOM"],
                max_price=c.get("max_budget_sgd") or NO_PRICE_LIMIT, min_lease_years=c.get("min_remaining_lease_years") or 0)))
        if c.get("wants_price_trend") and towns:
            plan.append(("compute_town_price_timeseries", dict(towns=towns, flat_type=(c.get("flat_types") or ["4 ROOM"])[0],
                                                               lookback_years=c.get("trend_years") or 3)))
        plan.extend(addons.plan(agent, c, query))
        if c.get("analysis_mode") and addons.active():
            plan.extend(ri.analysis_fallback(c, query))
        if c.get("wants_bto_comparison") and not c.get("bto_project_name"):
            m = re.search(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\s+(?:BTO|launch)", query)
            if m and re.fullmatch(r"(?:Is|Are|Was|Which|What|How|Does|Do|Compare|Should|Can|The|A|An|Any|New|Upcoming|Recent|Latest)"
                                  r"(?: [A-Z][A-Za-z]+)*", m.group(1)):
                m = None                  # "Is BTO cheaper than resale in Sembawang?": no project named
            plan.append(("compare_bto_launch_vs_resale", dict(bto_project_name=m.group(1) if m else (towns[0] if towns else "Recent BTO"), radius_meters=500)))
    elif agent == "spatial":
        if c.get("nearby_schools"):
            return addons.plan(agent, c, query)   # schools near an address: that lookup alone (schools pack)
        radii = {k: v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"}
        if radii:
            # the tool limits any rail kind to the named station ("within 300m of Sumang LRT" is read as "lrt")
            station = c.get("mrt_station") if set(radii) & {"mrt", "lrt", "mrt_only"} else None
            plan.append(("filter_blocks_by_amenity_proximity",
                         dict(towns=towns, amenity_constraints=radii, **({"mrt_station": station} if station else {}))))
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


def _dedupe_stations(result: Any) -> Any:
    """One line per MRT/LRT station in a nearby-amenities result: the index has a point per exit, so "Bishan Interchange
    Mrt Station" was listed twice, 6 m and 135 m away (SEM-32, SPAT-09). The nearest point is kept and the count within
    the radius drops by the points removed. Stop-gap until the tool dedupes itself (mcp_server/location_tools.py,
    amenities_near); other kinds share names across real places ("NTUC FairPrice"), so only stations are merged."""
    if not isinstance(result, dict) or result.get("status") != "ok" or not isinstance(result.get("amenities"), list):
        return result
    seen, kept, dropped = set(), [], {}
    for a in sorted(result["amenities"], key=lambda a: a.get("distance_meters") or 0):
        key = (a.get("category"), _station_key(a.get("name")))
        if a.get("category") == "mrt" and key in seen:
            dropped[a["category"]] = dropped.get(a["category"], 0) + 1
            continue
        seen.add(key)
        kept.append(a)
    if not dropped:
        return result
    counts = {k: max(0, v - dropped.get(k, 0)) for k, v in (result.get("counts") or {}).items()}
    return {**result, "amenities": kept, "counts": counts}


def _store(outputs: Dict[str, Any], name: str, result: Any) -> None:
    """Repeatable research tools keep every call (one per address); others keep the latest."""
    if name == "get_nearby_amenities":
        result = _dedupe_stations(result)
    if name in REPEATABLE_TOOLS and name in outputs:
        previous = outputs[name] if isinstance(outputs[name], list) else [outputs[name]]
        outputs[name] = previous + [result]
    else:
        outputs[name] = result


def _commute_candidates(outputs: Dict[str, Any]) -> List[str]:
    """Commute only the blocks the other spatial filters kept, instead of an arbitrary index slice."""
    blocks = slot_outputs({k: v for k, v in outputs.items() if k in SPATIAL_BLOCK_TOOLS[:2]}, []).get("spatial_block_candidates") or []
    return [f"Blk {b['block']} {b['street']}" for b in blocks]


_DECIMAL = re.compile(r"(?<![\d.])-?\d{1,3}\.\d{2,}(?![\d.])")


def typed_points(text: str) -> List[Tuple[float, float]]:
    """Coordinate pairs the user typed ("latitude 1.3521, longitude 103.9440", "1.3521, 103.944"): a Singapore
    latitude (1.1 to 1.5) followed by a Singapore longitude (103.5 to 104.1), read in the order they were written."""
    points: List[Tuple[float, float]] = []
    lat: Optional[float] = None
    for m in _DECIMAL.finditer(text or ""):
        x = float(m.group())
        if 1.1 <= x <= 1.5:
            lat = x
        elif 103.5 <= x <= 104.1 and lat is not None:
            points.append((lat, x))
            lat = None
    return points


def trusted_points(state: HDBCompassState) -> List[Tuple[float, float]]:
    """The coordinates a model's tool call may use without a tool having produced them (run_react's known_points): the
    user's map pin and any coordinates the user typed. Every variant passes the same, so a coordinate tool is refused
    or allowed the same way in A, B and C."""
    pin = (state.get("map_ui_payload") or {}).get("center")
    points = [tuple(pin)] if isinstance(pin, (list, tuple)) and len(pin) == 2 else []
    users = " ".join(m.get("content", "") for m in state.get("messages", []) if isinstance(m, dict) and m.get("role") == "user")
    return points + typed_points(users)


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
                    if name == "compute_dual_transit_commute":
                        # the blocks to time are the ones the other spatial filters kept: a model that passes a town
                        # ("QUEENSTOWN") or blocks it made up ("BLK 123") would time nothing, and the empty result
                        # would empty the whole search
                        kept = _commute_candidates(outputs)
                        given = args.get("candidate_blocks") or []
                        if not kept:
                            # no filter ran: no block list exists yet, and a town name ("QUEENSTOWN") is not a block
                            # (LIVE-SPATIAL-01: timed nothing, then claimed no match); [] times every block and the
                            # intersection keeps the searched towns' ones
                            if given:
                                errors.append(f"{label}: commute candidate_blocks the model wrote were dropped (no filter ran)")
                            args = {**args, "candidate_blocks": []}
                        elif not given or not set(given) & set(kept):
                            if given:
                                errors.append(f"{label}: commute blocks that no filter kept were replaced by the filtered blocks")
                            args = {**args, "candidate_blocks": kept}
                    if name == "query_resale_statistics":
                        args = statistics_args(args, c, errors, label)
                    if name == "query_resale_blocks_by_criteria":
                        # "4 room" in Tampines states no budget or lease: the model passed null for both and the tool
                        # refused the call; no stated limit is the same no-limit code's own plan uses
                        args = {**args, "max_price": NO_PRICE_LIMIT if args.get("max_price") is None else args["max_price"],
                                "min_lease_years": 0 if args.get("min_lease_years") is None else args["min_lease_years"]}
                    args = search_args(name, args, c, errors, label)
                    if FIN_TOOL_SECTION.get(name) == "upfront" and upfront_price(c, outputs, runners) is None:
                        # no price the user gave or code worked out: the model asks for it instead of costing its own
                        # (LIVE-MONEY-01)
                        errors.append(f"{label}: upfront calculator call refused: no purchase price stated")
                        msgs.append(ToolMessage(content=json.dumps({"error": NO_PRICE_ERROR}), tool_call_id=tc["id"]))
                        continue
                    guarded = guard_financial_args(name, args, c, outputs, runners)   # may run the grant calculator (code)
                    runners[name].source = "model"
                    try:
                        result = runners[name](**guarded)
                    finally:
                        runners[name].source = "code"
                    # a call the tool refused or that failed is not evidence: if every call fails, code's plan runs
                    used_llm_tools = used_llm_tools or bool(runners[name].log and runners[name].log[-1].ok)
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
            # and the block search, distance filter or school check the plan needs: the live model answered "I could
            # not find data" for a search the code path answers (LIVE-SEARCH-01)
            core = [(n, a) for n, a in deterministic_plan(agent, c, query)
                    if n in ("query_resale_blocks_by_criteria", "filter_blocks_by_amenity_proximity", "verify_primary_school_distance")]
            for name, args in core + addons.plan(agent, c, query):
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
    if aff_args.get("total_grants") == 0:
        upfront_args.setdefault("total_grants", 0.0)   # "No grants." holds for the upfront costs too
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
    # no grants at all ("Assume zero housing grants.", two PRs): the grant calculator would only ask its facts again
    # (HOUSE-04)
    grants_known = aff_args.get("total_grants") == 0 or constraints.get("pr_couple")
    if grant_name in runners and not upfront_only and not grants_known:
        grants = outputs[grant_name] = runners[grant_name](**grant_args)
        if grants.get("status") == "ok":
            for target in (aff_args, upfront_args):
                if target.get("total_grants") is None:
                    target["total_grants"] = grants["total_grants"]
        elif grant_args.get("eligibility_confirmed") is False:
            # "HDB has not confirmed our grant eligibility": no grant can be counted, so the budget uses none (SEM-19)
            for target in (aff_args, upfront_args):
                if target.get("total_grants") is None:
                    target["total_grants"] = 0.0
        elif grants.get("status") == "unsupported" and _BUDGET_ASK.search(asked_part(query)):
            # a case the grant rules here do not cover (second-timers, a short lease), when the budget was asked for:
            # it is still worked out, with no grants, and the answer says so (financial_lines)
            for target in (aff_args, upfront_args):
                if target.get("total_grants") is None:
                    target["total_grants"] = 0.0
    if aff_name in runners and not upfront_only:
        aff = outputs[aff_name] = runners[aff_name](**aff_args)
        if aff.get("status") == "ok" and upfront_args.get("loan_type", "hdb") == aff_args.get("loan_type", "hdb"):
            upfront_args.setdefault("ltv_ratio", aff["ltv_ratio"])
    if upfront_name in runners and asked_for(upfront_name, constraints, query):   # never cost the budget unasked
        if "purchase_price" not in upfront_args:
            # the stated budget, else the ceiling just worked out ("Does stamp duty fit within that number?"); never a
            # price nobody gave (LIVE-MONEY-01)
            price = upfront_price(constraints, outputs)
            if price is not None:
                upfront_args["purchase_price"] = price
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
    # Whether a block filter has run yet. An empty list from one means no block met its limit, so later filters keep
    # the set empty: the commute check's [] once kept every block the amenity filter found, and its island-wide
    # results replaced the amenity filter's [] (the commute then ran on all blocks), dropping the MRT limit.
    filtered = False
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
            spatial = out if not filtered else [b for b in spatial if block_key(b["block"], b["street"]) in
                                                {block_key(x["block"], x["street"]) for x in out}]
            filtered = True
        elif name == "verify_primary_school_distance":
            sb = _normalise_school_blocks(out, towns)
            spatial = sb if not filtered else [b for b in spatial if block_key(b["block"], b["street"]) in
                                               {block_key(x["block"], x["street"]) for x in sb}]
            filtered = True
        elif name == "compute_dual_transit_commute" and isinstance(out, list):
            rows = [b for b in out if isinstance(b, dict)]
            # by block and street, so a block passes only if it met the limit itself, not a namesake on another street
            keep = {block_key(str(b.get("block", "")), str(b.get("street", ""))) for b in rows}
            spatial = [b for b in spatial if block_key(str(b.get("block", "")), str(b.get("street", ""))) in keep] \
                if filtered else rows
            filtered = True
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
        # only the tools this task needs (needed_for_task); Variants A and B keep the full list, so their models'
        # unasked calls still show in Tool F1
        visible = addons.visible_tools(tools_for_agent(agent), state.get("variant"), bool(state.get("fast_path")))
        tools = [name for name in visible if needed_for_task(name, c, query)]
        system = AGENT_SYSTEM[agent] if addons.offered_in(state.get("variant"), bool(state.get("fast_path"))) \
            else CORE_AGENT_SYSTEM[agent]
        guide = analysis_prompt(c, state.get("variant"), bool(state.get("fast_path"))) if agent == "market_sql" else ""
        outputs, log, errors, final = run_react(agent, tools, system + guide, query, c,
                                                max_iter=ri.ANALYSIS_MAX_ITERATIONS if guide else None,
                                                known_points=trusted_points(state), use_model=c.get("extraction_source") != FAST_PATH_SOURCE,
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


def _price_query_truncated(sql_out: Any) -> bool:
    """The price search stops at its result limit (100 blocks, cheapest first), so a block the distance filters kept
    can be missing from it only because it was cut off (B2-04: 121 matches, 122 Simei St 1 not among the first 100)."""
    return isinstance(sql_out, dict) and bool((sql_out.get("metadata") or {}).get("truncated"))


def _price_spatial_blocks(state: HDBCompassState, log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Join resale prices onto spatial-only results (school, commute) when price or lease limits were stated, and
    price the spatial blocks directly when the town-wide price search was cut off at its result limit."""
    c = state.get("user_constraints", {})
    spatial = state.get("spatial_block_candidates")
    truncated = _price_query_truncated((state.get("sql_market_results") or {}).get("query_resale_blocks_by_criteria"))
    if (_sql_blocks(state) is not None and not truncated) or not spatial or not (
            c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("flat_types")
            or c.get("wants_area_affordability") or re.search(r"\bresale\b", last_user_query(state), re.I)):
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
    floor = state.get("user_constraints", {}).get("min_price_sgd")
    sql_out = (state.get("sql_market_results") or {}).get("query_resale_blocks_by_criteria")
    if floor and _price_query_truncated(sql_out) and state.get("spatial_block_candidates") is None:
        # the price search returns the 100 cheapest: a range above them needs every block of the towns priced
        c = state.get("user_constraints", {})
        lister = ToolRunner("filter_blocks_by_amenity_proximity")
        lister.agent = "supervisor"
        everything = lister(towns=c.get("towns") or [], amenity_constraints={})
        log.extend(rec.model_dump() for rec in lister.log)
        if isinstance(everything, list) and everything:
            runner = ToolRunner("query_resale_blocks_by_criteria")
            runner.agent = "supervisor"
            out = runner(towns=c.get("towns") or [], flat_types=c.get("flat_types") or [],
                         max_price=c.get("max_budget_sgd") or NO_PRICE_LIMIT, min_lease_years=c.get("min_remaining_lease_years") or 0,
                         block_keys=list(dict.fromkeys(block_key(b["block"], b["street"]) for b in everything)))
            log.extend(rec.model_dump() for rec in runner.log)
            if isinstance(out, dict) and "blocks" in out:
                update["sql_market_results"] = {**(state.get("sql_market_results") or {}), "query_resale_blocks_by_criteria": out}
                state = {**state, **update}
    blocks = compute_intersection(_sql_blocks(state), state.get("spatial_block_candidates"), cap, prices_checkable=checkable)
    if floor:   # "between $600k and $650k": the lower end of the range
        blocks = [b for b in blocks if b.get("median_resale_price") is None or b["median_resale_price"] >= floor]
    update["candidate_blocks_intersection"] = blocks
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
            if name in runners and name in ("query_resale_blocks_by_criteria",) + SPATIAL_BLOCK_TOOLS:
                if name == "compute_dual_transit_commute":
                    # the commute limit still holds while another limit is relaxed, so an unlocked block must meet it too
                    blocks = _commute_candidates(outputs)
                    if not blocks and any(k in outputs for k in SPATIAL_BLOCK_TOOLS[:2]):
                        continue   # the other spatial filters kept no block: nothing to time
                    args = {**args, "candidate_blocks": blocks}
                outputs[name] = runners[name](**args)
    slots = slot_outputs(outputs, c.get("towns") or [])
    sql_out = (slots.get("sql_market_results") or {}).get("query_resale_blocks_by_criteria")
    spatial = slots.get("spatial_block_candidates")
    if _price_query_truncated(sql_out) and spatial and "query_resale_blocks_by_criteria" in runners:
        # a relaxed price search cut off at its limit would undercount the blocks a step unlocks
        repriced = runners["query_resale_blocks_by_criteria"](
            towns=c.get("towns") or [], flat_types=c.get("flat_types") or [],
            max_price=c.get("max_budget_sgd") or NO_PRICE_LIMIT, min_lease_years=c.get("min_remaining_lease_years") or 0,
            block_keys=[block_key(b["block"], b["street"]) for b in spatial])
        if isinstance(repriced, dict) and "blocks" in repriced:
            sql_out = repriced
    sql = sql_out.get("blocks") if isinstance(sql_out, dict) else None
    return compute_intersection(sql, spatial, None)


def relax_node(state: HDBCompassState) -> Dict[str, Any]:
    """Sensitivity sweep: relax ONE constraint at a time and count unlocked blocks."""
    steps = load_config()["architecture"]["relaxation_steps"]
    c = dict(state.get("user_constraints", {}))
    query = last_user_query(state)
    runners = build_runners(["query_resale_blocks_by_criteria", *SPATIAL_BLOCK_TOOLS])
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


LISTED_BLOCKS = 20   # matching blocks written out in full; the rest are counted (on the map when they have coordinates)
AMENITY_LABELS = {"lrt": "an LRT station", "mrt_only": "an MRT station", "mrt": "an MRT station", "hawker": "a hawker centre", "bus_stop": "a bus stop",
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
POLICY_PASSAGES_SHOWN = 4  # of the search's 5: with 3, the passage that answered was often the 4th (Experiment 1,
                           # 2 Nov: C1-02, C2-01); A and B's models read all 5, so C's facts should not show far fewer
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

# Topic signals used only when a question names more than one policy dimension. A section that matches only the
# generic flat-classification words ("Prime", "Plus", "flat") is not enough to justify showing it: for example,
# a question about MOP and eligibility should not also dump a clawback section merely because it mentions Prime.
_POLICY_TOPIC_PATTERNS = {
    "mop": re.compile(r"\b(?:mop|minimum occupation|physical occupancy|occupancy period)\b", re.I),
    "eligibility": re.compile(r"\b(?:eligibility|who can|age requirement|income ceiling|citizen|singles? scheme|eligible single buyers)\b", re.I),
    "clawback": re.compile(r"\b(?:clawback|subsidy recovery)\b", re.I),
    "rental": re.compile(r"\b(?:whole[- ]flat rental|rental restriction|rent(?:al|ed|ing)?|bedroom)\b", re.I),
    "classification": re.compile(r"\b(?:prime|plus|standard|flat classification|flat type)\b", re.I),
}


def _policy_topic_groups(question: str) -> set:
    """Return named policy dimensions in a question; generic words such as "rules" are ignored."""
    return {name for name, pattern in _POLICY_TOPIC_PATTERNS.items() if pattern.search(question or "")}


def _section_topic_groups(section: Dict[str, Any]) -> set:
    title = " ".join([section.get("title", ""), section.get("parent", "")])
    body = " ".join(section.get("lines", []))
    groups = set()
    for name, pattern in _POLICY_TOPIC_PATTERNS.items():
        # Bare "MOP" in a clawback explanation is supporting context, not an MOP section. Require the expanded
        # phrase in body text unless the breadcrumb/title itself names MOP.
        body_pattern = (re.compile(r"\b(?:minimum occupation|occupancy period)\b", re.I)
                        if name == "mop" else pattern)
        if pattern.search(title) or body_pattern.search(body):
            groups.add(name)
    return groups


def _line_topic_groups(line: str) -> set:
    """Topics stated by one displayed line; unlike section scoring, a bare MOP is meaningful here."""
    return {name for name, pattern in _POLICY_TOPIC_PATTERNS.items() if pattern.search(line or "")}


def _line_starts_with_topic(line: str, topic: str) -> bool:
    """Whether a line is headed by a topic (table rows such as "Whole-Flat Rental After MOP — ...")."""
    head = re.split(r"\s+—\s+|:", line or "", maxsplit=1)[0]
    return bool(_POLICY_TOPIC_PATTERNS[topic].search(head))


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
    # a picked line that introduces a list ("... the HFE letter consolidates three critical assessments into a single
    # application:") is shown with the items under it while there is room: SEM-13 showed one of the three
    for i in sorted(chosen):
        if not lines[i].rstrip().endswith(":"):
            continue
        for j in range(i + 1, len(lines)):
            if len(chosen) >= POLICY_LINES_SHOWN or lines[j].rstrip().endswith(":"):
                break
            if j not in chosen:
                chosen.append(j)
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
    last = len(parts) - 1
    if line.rstrip().endswith(":") and keep[-1] != last and len(_join_kept(parts, keep + [last])) <= POLICY_LINE_MAX:
        keep.append(last)   # a lead-in keeps the sentence that introduces the items shown under it (_pick_lines)
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


def _classes(text: str) -> set:
    """The flat classes a text names: {"prime"} for "a Prime flat"."""
    return {k for f, k in scheme_terms(text) if f == "class"}


def _asked_cells(line: str, asked: set) -> str:
    """A comparison-table row cut to the asked classes' cells: "Whole-Flat Rental After MOP — Standard Flats: Permitted;
    Plus Flats: Prohibited; Prime Flats: Prohibited" is "Whole-Flat Rental After MOP — Prime Flats: Prohibited" for a
    Prime question. Only a row whose every cell is headed by a class."""
    if " — " not in line:
        return line
    head, cells = line.split(" — ", 1)
    parts = [c.strip() for c in cells.split(";")]
    if not all(_classes(c.split(":")[0]) for c in parts):
        return line
    keep = [c for c in parts if _classes(c.split(":")[0]) & asked]
    return f"{head} — {'; '.join(keep)}" if keep and len(keep) < len(parts) else line


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
            sections[key]["score"] = max(sections[key].get("score", 0.0), float(p.get("rerank_score") or 0.0))
        else:
            sections[key] = {"title": title, "parent": parent, "lines": lines, "source_url": p.get("source_url") or "",
                             "score": float(p.get("rerank_score") or 0.0)}

    # Select sections by the retriever's relevance score, not merely by the order in which duplicate chunks happen to
    # arrive. This is generic section-level aggregation: any decisive passage can outrank several supporting chunks.
    ordered_sections = sorted(sections.values(), key=lambda s: s.get("score", 0.0), reverse=True)

    # For multi-part policy questions, keep sections that address one of the explicitly named dimensions. The
    # classification group is deliberately not sufficient on its own because most sections mention Prime/Plus.
    # Single-topic questions retain the existing top-three behavior and its fallback lines.
    topics = _policy_topic_groups(question)
    required_topics = topics - {"classification"}
    if len(required_topics) >= 2:
        relevant = [s for s in ordered_sections if _section_topic_groups(s) & required_topics]
        if relevant:
            sections = {str(i): s for i, s in enumerate(relevant[:POLICY_PASSAGES_SHOWN])}
    else:
        sections = {str(i): s for i, s in enumerate(ordered_sections[:POLICY_PASSAGES_SHOWN])}

    asked = _classes(question)
    if asked:
        # "Can I rent out the whole of a Prime flat once its MOP is over?": not the Standard-flat section, whose
        # 5-year whole-flat rental rule became the opening (SEM-07), nor a table row's Standard and Plus cells. A
        # section about another class stays when its lines name the asked class too: "Just like Prime flats, Plus flat
        # owners are never permitted to rent out the whole flat" (with its spare-bedroom rule) answers a Prime question
        # (C1-02, whose clause was lost without it)
        about = [s for s in sections.values()
                 if not _classes(s["title"] + " " + s["parent"]) - asked
                 or (_classes(s["title"] + " " + s["parent"]) | _classes(" ".join(s["lines"]))) & asked]
        if about:
            sections = {str(i): s for i, s in enumerate(about)}
        for s in sections.values():
            s["lines"] = [_asked_cells(line, asked) for line in s["lines"]]

    pool = [line for s in sections.values() for line in s["lines"]]
    for s in sections.values():
        picked = [_trim_line(s["lines"][i], question) for i in _pick_lines(s["lines"], question, pool)]
        if len(required_topics) >= 2:
            def relevant_line(line: str) -> bool:
                groups = _line_topic_groups(line)
                return bool(groups & required_topics) and not any(
                    _line_starts_with_topic(line, topic) for topic in groups - required_topics)

            topical = [line for line in picked if relevant_line(line)]
            if len(topical) < 2:
                # The general word-overlap scorer can spend its slots on "Prime/Plus" boilerplate. Supplement from
                # the full section so an explicitly requested eligibility or MOP rule is still shown.
                topical += [_trim_line(line, question) for line in s["lines"]
                            if relevant_line(line) and _trim_line(line, question) not in topical]
            if len(topical) < 2:
                topical += [line for line in picked if line not in topical]
            topical = topical[:POLICY_LINES_SHOWN]
            # Keep the section's best lines when topic extraction cannot classify a short table row; otherwise
            # an explicitly requested topic must not be accompanied by an unasked rental/clawback rule.
            s["lines"] = topical or picked[:1]
        else:
            s["lines"] = picked
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
    about = (state.get("user_constraints") or {}).get("about_results")
    if about:
        return about   # code's answer to a question about the last answer (about_last_answer); nothing else ran
    lines: List[str] = []
    blocks = state.get("candidate_blocks_intersection") or []
    searched = (
        blocks or _sql_blocks(state) is not None
        or state.get("spatial_block_candidates") is not None
    )
    # a search tool that could not be loaded checked nothing (nor did a commute limit with one workplace), so the
    # blocks meet only the constraints that were checked
    partial = partly_checked(state)
    if searched:
        if blocks:
            meta = ((state.get("sql_market_results") or {}).get("query_resale_blocks_by_criteria") or {}).get("metadata") or {}
            total = meta.get("total_matching_blocks")
            cut = bool(meta.get("truncated")) and isinstance(total, int) and total > len(blocks) \
                and state.get("spatial_block_candidates") is None
            if cut:
                # the price search stops at its limit (cheapest first): say so, rather than calling the limit the total
                lines.append(f"**The {len(blocks)} cheapest of {total} matching blocks** "
                             f"(narrow the search, e.g. a lower budget or a distance limit, to see the rest), "
                             f"meeting {'the constraints that could be checked' if partial else 'all stated constraints'}:")
            else:
                lines.append(f"**{len(blocks)} candidate block(s)** match "
                             f"{'the constraints that could be checked' if partial else 'all stated constraints'}:")
            for b in blocks[:LISTED_BLOCKS]:
                price = f" — median ${b['median_resale_price']:,.0f}" if b.get("median_resale_price") else ""
                lease = f", {b['median_remaining_lease_years']:.0f} yrs lease" if b.get("median_remaining_lease_years") else ""
                dist = ", ".join(_distance_label(k, v) for k, v in (b.get("distances") or {}).items())
                kind = f" {b['flat_type']}" if b.get("flat_type") and len({x.get("flat_type") for x in blocks}) > 1 else ""
                town = f" ({b['town']})" if b.get("town") else ""
                lines.append(f"- Blk {b['block']} {b['street']}{town}{kind}{price}{lease}{(' — ' + dist) if dist else ''}")
            if len(blocks) > LISTED_BLOCKS:
                # a block with no coordinates on file has no marker: "all shown on the map" was untrue (SEM-26, SPAT-01)
                mapped = sum(1 for b in blocks if b.get("lat") and b.get("lng"))
                if mapped == len(blocks):
                    lines.append(f"- …and {len(blocks) - LISTED_BLOCKS} more, all shown on the map.")
                else:
                    missing = len(blocks) - mapped
                    lines.append(f"- …and {len(blocks) - LISTED_BLOCKS} more; {mapped} of the {len(blocks)} blocks are on "
                                 f"the map ({missing} {'has' if missing == 1 else 'have'} no map location on file).")
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
    lines[:0] = area_affordability_lines(state)   # the answer to "can I afford…" comes before the block list
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
    pick = (state.get("user_constraints") or {}).get("listed_pick")
    if pick:
        lines.append(pick)                   # "Of the 20 blocks listed earlier, Blk 35 ... is the cheapest (median $446,500)."
    inst = (state.get("user_constraints") or {}).get("instalment")
    if inst:
        lines.append(f"**Monthly instalment: ${inst['monthly']:,.0f}** on a ${inst['loan']:,.0f} loan at {inst['rate']:g}% a year "
                     f"over {inst['years']} years (equal monthly payments; ${inst['interest']:,.0f} interest in total).")
    fin = state.get("financial_breakdown") or {}
    uc = state.get("user_constraints") or {}
    grants_unconfirmed = ((uc.get("financial_inputs") or {}).get("grants") or {}).get("eligibility_confirmed") is False
    duty_basis = _duty_price_basis(state)
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
                if name == "calculate_bsd_and_upfront_outlay" and _stamp_duty_only(state) is not None:
                    # "stamp duty on $650k?": the duty needs only the price; the funding facts were not asked for
                    lines.append(f"- Buyer's stamp duty: **${value['buyer_stamp_duty']:,.0f}** (on {duty_basis}; "
                                 "charged on the valuation instead if that is higher)")
                    continue
                if value.get("status") == "needs_information" and name != "calculate_bsd_and_upfront_outlay" \
                        and uc.get("short_lease"):
                    # the facts it lists could not give an estimate for this flat; none is asked for (SEM-21)
                    lines.append(f"- {label}: not worked out — the lease ends before the youngest buyer turns 95, a case "
                                 "this estimate does not model.")
                    continue
                if value.get("status") == "needs_information" and name == "calculate_cpf_housing_grants" and grants_unconfirmed:
                    lines.append(f"- {label}: not worked out — your grant eligibility is not confirmed yet (HDB confirms "
                                 "it in the HFE letter).")
                    continue
                detail = ", ".join(field.replace("_", " ") for field in value.get("missing_inputs", [])) or "; ".join(value.get("notes", []))
                status = "more information needed" if value.get("status") == "needs_information" else "scenario unavailable"
                lines.append(f"- {label}: {status} — {detail}")
                if name == "calculate_bsd_and_upfront_outlay" and value.get("buyer_stamp_duty") is not None:
                    # stamp duty needs only the price, so it is shown even when the full funding estimate cannot run
                    lines.append(f"- Buyer's stamp duty: **${value['buyer_stamp_duty']:,.0f}** (on {duty_basis}; "
                                 "charged on the valuation instead if that is higher)")
        fin = {name: value for name, value in fin.items() if value.get("status", "ok") == "ok" and "error" not in value}
        g = fin.get("calculate_cpf_housing_grants")
        if g:
            lines.append(f"- Grants: EHG ${g['enhanced_housing_grant']:,.0f} + Family ${g['family_grant']:,.0f} + PHG ${g['proximity_housing_grant']:,.0f} = **${g['total_grants']:,.0f}**")
        a = fin.get("calculate_maximum_affordability")
        if a:
            lines.append(f"- Max purchase price ${a['max_purchase_price']:,.0f} (loan ${a['max_loan_amount']:,.0f}, instalment ${a['monthly_instalment']:,.0f}/mth, binding: {a['binding_constraint']})")
            g_status = ((state.get("financial_breakdown") or {}).get("calculate_cpf_housing_grants") or {}).get("status")
            if not g and (g_status == "unsupported" or uc.get("pr_couple") or grants_unconfirmed):
                lines.append("- This budget assumes no housing grants, since none could be worked out for your case.")
            price = ((state.get("user_constraints") or {}).get("asked_price"))
            if price:
                verdict = "within" if price <= a["max_purchase_price"] else "above"
                lines.append(f"- A ${price:,.0f} flat is **{verdict}** this ceiling"
                             + (f" (${a['max_purchase_price'] - price:,.0f} to spare)." if verdict == "within"
                                else f" by ${price - a['max_purchase_price']:,.0f}."))
            if not a.get("purchase_costs_included", True):
                lines.append("- This purchase-only ceiling excludes taxes, fees and cash over valuation; check upfront funding before choosing a budget.")
        u = fin.get("calculate_bsd_and_upfront_outlay")
        if u:
            lines.append(f"- Upfront: BSD ${u['buyer_stamp_duty']:,.0f}, cash required ${u['cash_required']:,.0f}, total ${u['total_upfront']:,.0f}")
            if (u.get("cash_shortfall") or 0) > 0:
                lines.append(f"- Cash shortfall: ${u['cash_shortfall']:,.0f}.")
    lines.extend(flat_affordability_lines(state))
    passages = state.get("rag_policy_passages") or []
    if passages and "retrieve_hdb_policy_knowledge" in unbuilt:
        lines.append("**Policy:**")
        lines.append("- I can't check HDB policy or eligibility rules yet in this version. Please confirm with HDB, "
                     "for example with an HDB Flat Eligibility (HFE) letter.")
    elif passages:
        lines.append("**Policy:**")
        lines.extend(policy_fact_lines(passages, user_request(state)))
    elif state.get("policy_no_match") and not (state.get("financial_breakdown") and _OWN_GRANT_ASK.search(
            str((state.get("user_constraints") or {}).get("policy_question") or ""))):
        # "What grants can we get?" is answered by the calculators (which ask for the household's facts); a "no rule
        # found" line beside them reads as no answer (CLARIFY-03)
        lines.append(NO_POLICY_ANSWER)
    if not lines:
        lines.append(NO_DATA)
    if state.get("financial_breakdown") or passages:
        lines.append(HFE_DISCLAIMER)
    return "\n".join(lines)


def _duty_price_basis(state: HDBCompassState) -> str:
    """What the stamp duty was worked out on: the price the user gave, or else the budget or ceiling code costed in its
    place (run_financial_sequence, upfront_price), named so the figure is not read as the duty on a price they gave."""
    c = state.get("user_constraints") or {}
    if ((c.get("financial_inputs") or {}).get("upfront") or {}).get("purchase_price"):
        return "the purchase price"
    if c.get("max_budget_sgd"):
        return f"your budget of ${c['max_budget_sgd']:,.0f}"
    aff = (state.get("financial_breakdown") or {}).get("calculate_maximum_affordability") or {}
    if aff.get("status") == "ok" and aff.get("max_purchase_price"):
        return f"your maximum purchase price of ${aff['max_purchase_price']:,.0f}"
    return "the purchase price"


def area_affordability_lines(state: HDBCompassState) -> List[str]:
    """'Can I afford a resale near Novena MRT?': the household's ceiling and how many of the area's blocks have a
    median price within it (intersect_node already applied it), or what is needed to work the ceiling out."""
    c = state.get("user_constraints") or {}
    if not c.get("wants_area_affordability") or c.get("flat_purchase"):
        return []
    where = (f"near {c['mrt_station']} MRT" if c.get("mrt_station")
             else "in " + " / ".join(t.title() for t in c["towns"]) if c.get("towns") else "in this area")
    kind = (c.get("flat_types") or [None])[0]
    flat = f"a {kind.lower()} flat" if kind else "a flat"
    out = [f"**Can you afford {flat} {where}?**"]
    fin = state.get("financial_breakdown") or {}
    a, g = (fin.get(k) or {} for k in ("calculate_maximum_affordability", "calculate_cpf_housing_grants"))
    priced = [b["median_resale_price"] for b in (_sql_blocks(state) or []) if b.get("median_resale_price")]
    blocks = state.get("candidate_blocks_intersection") or []
    cap = _affordable_price(state)
    if cap is not None:
        out.append(f"- Your purchase ceiling, with grants, is about **${cap:,.0f}**.")
        if blocks:
            out.append(f"- **{len(blocks)} block(s)** {where} have a median price within it; they are listed below.")
        elif priced:
            out.append(f"- **None** of the blocks {where} has a median price within it: the cheapest median is "
                       f"${min(priced):,.0f}, ${min(priced) - cap:,.0f} above your ceiling.")
        else:
            out.append(f"- No recent sales {where} could be compared with it.")
        out.append("- Use **Afford?** on a block for a verdict on that flat, with its own lease and stamp duty.")
    elif any(r.get("status") == "unsupported" for r in (a, g)):
        out.append("- I can't work out a ceiling for these facts: the calculators don't yet model a lease that ends "
                   "before the youngest buyer turns 95. An HDB Flat Eligibility (HFE) letter gives the exact figure.")
    else:
        if priced:
            out.append(f"- Recent median prices {where} range from ${min(priced):,.0f} to ${max(priced):,.0f}.")
        out.append("- To say which of these you can afford, I need your household facts (the form below). I'll then "
                   "work out your price ceiling and keep only the blocks within it.")
    return out


def flat_affordability_lines(state: HDBCompassState) -> List[str]:
    """'Can you afford Blk X?': the verdict, from tool results only (the block's price and its source, the affordability
    ceiling, the instalment, the grants and the stamp duty). Nothing here is computed beyond one subtraction."""
    fp = (state.get("user_constraints") or {}).get("flat_purchase")
    fin = state.get("financial_breakdown") or {}
    if not fp or not fin:
        return []
    price = fp["purchase_price"]
    out = [f"**Can you afford {fp['block']}?**",
           f"- Price used: **${price:,.0f}** ({fp['price_source']})."]
    a, g, u = (fin.get(k) or {} for k in ("calculate_maximum_affordability", "calculate_cpf_housing_grants",
                                           "calculate_bsd_and_upfront_outlay"))
    if a.get("status", "ok") == "ok" and "error" not in a and a.get("max_purchase_price") is not None:
        ceiling, gap = a["max_purchase_price"], a["max_purchase_price"] - price
        if gap >= 0:
            out.append(f"- Your purchase ceiling is ${ceiling:,.0f}, so this flat is **within your ceiling by ${gap:,.0f}**.")
        else:
            out.append(f"- Your purchase ceiling is ${ceiling:,.0f}, so this flat is **above your ceiling by ${-gap:,.0f}**.")
        if a.get("monthly_instalment") is not None:
            out.append(f"- Monthly instalment at the ceiling: ${a['monthly_instalment']:,.0f}.")
    elif any(r.get("status") == "unsupported" for r in (a, g)):
        # the calculators model only a lease that lasts until the youngest buyer is 95; for a shorter one CPF use,
        # the loan (LTV) and grants are prorated, which they do not compute, so no ceiling is given rather than a wrong one
        lease = fp.get("remaining_lease_years")
        out.append(f"- I can't give a verdict for this flat: its remaining lease{f' (about {lease:g} years)' if lease else ''} "
                   "does not last until the youngest buyer turns 95, so HDB prorates the CPF you can use, the loan and the "
                   "grants, and I don't calculate that proration yet. Ask HDB for an HFE letter, or try a block with "
                   "more lease left.")
    elif state.get("clarification"):
        out.append("- A verdict needs your household facts (see my question below); no ceiling can be stated until then.")
    else:
        out.append("- No purchase ceiling could be worked out from the facts given, so there is no verdict yet.")
    if g.get("status", "ok") == "ok" and "error" not in g and g.get("total_grants") is not None:
        out.append(f"- Housing grants you would get: ${g['total_grants']:,.0f}.")
    if u.get("buyer_stamp_duty") is not None:
        out.append(f"- Buyer's stamp duty on this price: ${u['buyer_stamp_duty']:,.0f}.")
    out.append("- The ceiling excludes stamp duty, legal and valuation fees, and any cash over valuation.")
    return out


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


# Why a commute limit was not checked, by what code knows: the workplaces it read (never "you did not name two", which
# may contradict the user's words when the reading missed one), or no reason when two were read but no check ran.
UNCHECKED_COMMUTE = "- Your commute limit was not checked."
UNCHECKED_COMMUTE_READ = {0: ("- Your commute limit was not checked: the commute check works out travel times to two "
                              "workplaces, and I could not read a workplace in your message."),
                          1: ("- Your commute limit was not checked: the commute check works out travel times to two "
                              "workplaces, and I could read only one in your message.")}
UNCHECKED_SCHOOL = ("- Your school distance limit was not checked: I could not tell which school you meant. "
                    "Please give the school's full name.")


def unchecked_commute_line(state: HDBCompassState) -> str:
    """The line not_checked_lines adds for commute_unchecked: the reason only when fewer than two workplaces were read."""
    c = state.get("user_constraints") or {}
    if c.get("unresolved_workplaces"):
        asked = any(q.startswith(_WORKPLACE_ASKS) for q in c.get("clarifications") or [])
        return ("- Your commute limit was not checked: I could not find "
                + _join_words([f"“{w}”" for w in c["unresolved_workplaces"]])
                + (" on the map (see my question below)." if asked
                   else " on the map. Give an address or postal code and I'll time the commute."))
    read = len(c.get("work_destinations") or [])
    return UNCHECKED_COMMUTE_READ.get(read, UNCHECKED_COMMUTE)


def school_unchecked(state: HDBCompassState) -> bool:
    """Whether a block search ran for a message that asks for blocks near a school but no school name reached the
    constraints. C's guard asks "Which primary school?" when no school is named; when the guard read a name that
    extraction could not ("St. Hilda's Primary School" with no model), or in A and B, which have no guard, the search
    runs without the school check, and the answer must say so rather than claim every limit was met."""
    if (state.get("user_constraints") or {}).get("school_name"):
        return False
    searched = (state.get("candidate_blocks_intersection") or _sql_blocks(state) is not None
                or state.get("spatial_block_candidates") is not None)
    # the guard with "no school named" says whether the words ask for a distance to a school (school_queries)
    return bool(searched) and guard_school_request(last_user_query(state), lambda q: False) == SCHOOL_NAME_QUESTION


def commute_unchecked(state: HDBCompassState) -> bool:
    """Whether a block search ran with a commute limit that no commute check applied. The commute check needs two
    workplaces, so a search with one ("under 30 min to Raffles Place") runs without it; the answer must then not say
    the blocks meet every limit (a fact code is unsure of is said to be unchecked). A commute tool that could not load
    is already reported by not_checked_lines."""
    c = state.get("user_constraints") or {}
    searched = (state.get("candidate_blocks_intersection") or _sql_blocks(state) is not None
                or state.get("spatial_block_candidates") is not None)
    if not (searched and c.get("max_commute_minutes")) or "compute_dual_transit_commute" in unavailable_tools(state):
        return False
    ran = any(t.get("tool") == "compute_dual_transit_commute" and t.get("ok", True) and not t.get("error")
              for t in state.get("tool_call_log") or [])
    return not (ran and len(c.get("work_destinations") or []) >= 2)


def partly_checked(state: HDBCompassState) -> bool:
    """Whether the blocks were checked against only some of the limits the user set: a block filter could not load,
    or a commute or school limit was never checked (commute_unchecked, school_unchecked), or a distance to a place with
    no location data was not applied ("within 200m of a park": SEM-11, the opening said every limit was met)."""
    return (any(t in BLOCK_FILTERS for t in unavailable_tools(state)) or commute_unchecked(state)
            or school_unchecked(state)
            or any(str(a).startswith("I can't measure distances") for a in (state.get("user_constraints") or {}).get("assumptions") or []))


def not_checked_lines(state: HDBCompassState) -> List[str]:
    lines = [f"- {NOT_CHECKED[t]} could not be checked: that tool is unavailable right now." for t in unavailable_tools(state)]
    lines += [unchecked_commute_line(state)] if commute_unchecked(state) else []
    return lines + ([UNCHECKED_SCHOOL] if school_unchecked(state) else [])


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
    if (state.get("user_constraints") or {}).get("flat_purchase"):
        base["flat_purchase"] = state["user_constraints"]["flat_purchase"]
    if unbuilt:
        base["not_available_yet"] = [UNBUILT_NOTES[t] for t in unbuilt]
    if not_checked_lines(state):   # only then, so every other prompt stays as it was
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
    if partly_checked(state):   # a search that could not run, or a limit never checked: not every constraint was met
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
    "interest_rate": "the interest rate the bank quoted you",
}
# a bank loan's eligibility is the bank's in-principle approval, not HDB's (SEM-20: two PRs were asked whether their
# HDB loan eligibility was confirmed)
BANK_FACT_WORDS = {"loan_eligibility_confirmed": "whether a bank has given you an in-principle approval (IPA) for the loan"}


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
    c = state.get("user_constraints") or {}
    if c.get("short_lease"):
        return None   # the lease ends before the youngest buyer turns 95: no fact asked for could give an estimate (SEM-21)
    grants_unconfirmed = ((c.get("financial_inputs") or {}).get("grants") or {}).get("eligibility_confirmed") is False
    if c.get("pr_couple") or grants_unconfirmed:
        # no grant can be worked out (two PRs; grant eligibility not confirmed: "HDB has not confirmed our grant
        # eligibility", SEM-19), so the grant calculator's own facts are not asked for
        grant_only = set(((fin.get("calculate_cpf_housing_grants") or {}).get("missing_inputs") or [])) - \
            set(((fin.get("calculate_maximum_affordability") or {}).get("missing_inputs") or []))
        needed = [n for n in needed if n not in grant_only and (n != "citizenship_pair" or not c.get("pr_couple"))]
    needed = list(dict.fromkeys(needed))
    if not c.get("flat_purchase"):
        needed = [n for n in needed if n != "remaining_lease_years"]   # assumed from the buyers' ages (_assume_flat_basics)
    if not needed:
        return None
    bank = ((c.get("financial_inputs") or {}).get("affordability") or {}).get("loan_type") == "bank"
    words = {**FACT_WORDS, **(BANK_FACT_WORDS if bank else {})}
    phrases = list(dict.fromkeys(words.get(n, n.replace("_", " ")) for n in needed))
    listed = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return {"kind": "financial", "needed": needed, "question": f"To finish your grant and budget estimate, please tell me {listed}."}


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


def _plain(text: str) -> str:
    """A line as words only: no markdown, bullets or punctuation, lower case."""
    return " ".join(re.sub(r"[^a-z0-9$ ]+", " ", text.lower()).split())


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
    # a sentence that only repeats a line of the facts ("I could not find data for that request.") shows it twice
    fact_lines = {_plain(line) for line in (facts or "").splitlines() if _plain(line)}
    kept, dropped = [], 0
    for sentence in _split_opening(opening):
        if _plain(sentence) in fact_lines:
            continue   # not counted: nothing is lost, the facts say it
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


# Choosing tenants, buyers or neighbours by race, religion or nationality ("How do I make sure I don't get Indian or
# PRC tenants?"): refused here until the input guardrail covers it; the room-rental rules may still be shown below.
_GROUPS = (r"(?:indians?|chinese|malays?|eurasians?|prcs?|mainlanders?|foreigners?|bangladeshis?|filipinos?|indonesians?|"
           r"muslims?|christians?|hindus?|buddhists?|blacks?|whites?|races?|racial|religions?|religious|nationalit(?:y|ies)|ethnic\w*)")
_SCREENING = re.compile(rf"\b(?:no|don'?t|do not|avoid|exclude|reject|refuse|keep out|filter out|screen out|only)\b[^.?!]{{0,40}}\b{_GROUPS}\b"
                        r"[^.?!]{0,30}\b(?:tenants?|renters?|lodgers?|buyers?|neighbou?rs?|occupants?|flatmates?)\b"
                        rf"|\b(?:tenants?|renters?|lodgers?|flatmates?)\b[^.?!]{{0,30}}\b(?:by|based on|of (?:a |the )?(?:certain|specific))\s+{_GROUPS}",
                        re.I)
SCREENING_OPENING = ("I can't help choose or turn away tenants, buyers or neighbours by race, religion or nationality. "
                     "Treat applicants on their merits; HDB's rules for renting out rooms are below.")


def forced_opening(state: HDBCompassState) -> Tuple[Optional[str], bool]:
    """Code's own opening for questions the passages and tools cannot answer in their words, and whether the model's
    checked sentences may follow it: a request to get around a rule (refused, then the lawful rule: the model may
    explain it), what this app can confirm or check (its estimates, a live quota) and a BTO launch the launch data does
    not have. The model wrote "HDB does not publicly display a block's live quota" and confused the EIP and SPR quotas
    (LIVE-POL-02), and the no-model answers quoted unrelated passages or the menu (POLICY-07, POLICY-09, LIMIT-08)."""
    query = user_request(state)
    bto = (state.get("sql_market_results") or {}).get("compare_bto_launch_vs_resale")
    named = (state.get("user_constraints") or {}).get("bto_project_name")
    if isinstance(bto, dict) and bto.get("status") == "not_found" and named and bto.get("bto_project_name") == named:
        # only a launch the user named: with none named the comparison searched the town's name ("SEMBAWANG")
        lead = "No, I" if re.match(r"\W*(?:do|does|did|is|are|can|could|have|has)\b", query, re.I) else "I"
        return f"{lead} don't have prices for {bto['bto_project_name']}; it isn't in my BTO launch data.", False
    if not (state.get("user_constraints") or {}).get("policy_question"):
        return None, False
    if _SCREENING.search(query):
        return SCREENING_OPENING, False
    if _EVASION.search(query):
        return EVASION_OPENING, True
    if _LIVE_QUOTA_ASK.search(query) and _ASKED.search(query):
        return LIVE_QUOTA_OPENING, False
    if _APP_ESTIMATE_ASK.search(query) and _ASKED.search(query):
        return APP_ESTIMATE_OPENING, False
    return None, False


def code_opening_parts(state: HDBCompassState, facts: str) -> Tuple[List[str], str, List[str]]:
    """Code's opening in two parts: the sentences that answer the question, and the trade-off sentence between the
    listed blocks ("" when there is none). Built from the same state as the facts (template_response), so it never
    says what they do not: an estimate is said to follow only when its calculator worked it out, and a summary only
    when the tool found one. Returns (answer sentences, trade-off, errors); a sentence that would not pass
    keep_supported_sentences, or a trade-off that comparisons_hold does not bear out, is left out and reported."""
    c = state.get("user_constraints") or {}
    blocks = state.get("candidate_blocks_intersection") or []
    searched = blocks or _sql_blocks(state) is not None or state.get("spatial_block_candidates") is not None
    partial = partly_checked(state)
    limits = "the limits that could be checked" if partial else "every limit you set"
    # the limits the blocks were checked against: a commute limit no check applied is not one (commute_unchecked)
    checked = ({k: v for k, v in c.items() if k not in ("max_commute_minutes", "work_destinations")}
               if commute_unchecked(state) else c)
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
        if c and not stated_limits(checked):   # (a state with no constraints at all is test data, not a real run)
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
                         else "No block meets every limit you set at once." if stated_limits(checked) or not c
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
        forced = forced_opening(state)[0]
        if forced:
            sentences.append(forced)
        elif policy_shown:
            sentences.append(policy_fallback_opening(state))
        elif state.get("rag_policy_passages") and "retrieve_hdb_policy_knowledge" not in unbuilt:
            sentences.append("The policy search found no HDB passage that answers this directly.")
        if done:
            words = [w for w, _ in done]
            sentences.append(f"Here {'is' if len(done) == 1 and not done[0][1] else 'are'} your {_join_words(words)}, "
                             "worked out from the facts you gave.")
        if clar:
            sentences.append(f"{'The rest of y' if done else 'Y'}our grant and budget estimate needs a few more facts, "
                             "asked for at the end.")
        elif fin and not done and _stamp_duty_only(state) is not None:
            if duty_needs_income_question(state):
                sentences.append("No — BSD depends only on the price (or valuation, if higher).")
            sentences.append("Here is the buyer's stamp duty.")
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
        # Permit policy figures only for a policy-only turn. Block/market answers must retain the existing
        # figure-free code opening checks even if a policy passage was also retrieved.
        opening_policy = state.get("rag_policy_passages") if policy_shown and not searched else None
        if keep_supported_sentences(s, facts, user_request(state), opening_policy)[1]:
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
_PLACES = {"lrt": r"\bLRT\b", "mrt_only": r"\bMRT\b", "mrt": r"\bMRT\b|\bstations?\b|\btrains?\b", "hawker": r"\bhawker|\bfood cent(?:re|er)",
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
            if what == "distance":
                places = [p for p in places if p in limited] or places   # "an LRT station": the lrt distance
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
_GENERIC_POLICY_LEAD = re.compile(r"\bThe (?:HDB )?policy passages (?:that bear on|relevant to) your question are below|The relevant HDB rules follow\b", re.I)
_SOURCE_ONLY = re.compile(r"^\s*(?:\[+\s*(?:source|official source)\s*\]+\s*)?\(?https?://[^\s)]+\)?\s*$|^\s*\[[^\]]{0,40}\]\(https?://[^)]+\)\s*$", re.I)

_POLICY_OPENING_STOPWORDS = frozenset(
    "can could would should what which who how are is the a an we i me my our to buy apply own get for with "
    "and or of do does this that rules rule flat flats grant grants housing policy user question yes no".split()
)


def policy_opening_is_relevant(sentence: str, question: str) -> bool:
    """Require meaningful lexical overlap with the user's question without enumerating policy names."""
    q_words = {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", (question or "").lower())
               if w not in _POLICY_OPENING_STOPWORDS}
    s_words = set(re.findall(r"[a-z][a-z0-9-]{2,}", (sentence or "").lower()))
    overlap = q_words & s_words
    # A sentence that shares only generic words such as "grant" or "housing" can still answer a different scheme.
    # Require at least one non-generic query term; this is corpus/query agnostic and works for unseen policy names.
    return bool(overlap)


def policy_scope_conflict(sentence: str, question: str, shown: List[Dict[str, Any]]) -> bool:
    """Whether a policy opening sentence is about a different scheme from the one asked: it names a sibling of the
    question's scheme the question does not name (the SPR quota in an answer about the EIP: "The EIP quota applies to
    households where all buyers and occupiers are Singapore Permanent Residents (SPR)", LIVE-POL-01), or it states a
    figure that the shown passages give only in sections about none of the schemes it names (SPR's 5% and 8% presented
    as the ethnic quotas, LIVE-POL-02). In the 3 Nov live answers this dropped 5 sentences, all in turns the tester
    marked FAIL or REVIEW; "No, there is no legal way to bypass the EIP quota." stays."""
    if sibling_terms(sentence, question):
        return True
    terms = scheme_terms(sentence)
    if not terms:
        return False
    for fig in _figures(sentence):
        homes = [s for s in shown if fig in set(_figures("\n".join(s["lines"])))]
        if homes and not any(scheme_terms(" ".join([s["title"], s["parent"]] + s["lines"])) & terms for s in homes):
            return True
    return False


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
    dropped = {"figures": 0, "comparisons": 0, "notes": 0, "leans_back": 0, "restated": 0, "lead_ins": 0, "relevance": 0,
               "scope": 0}
    fact_lines = {_plain(line) for line in (facts or "").splitlines() if _plain(line)}
    nothing_found = (facts or "").strip() == NO_DATA
    shown = shown_policy_passages(policy, user_request(state)) if policy else []
    for sentence in _split_opening(text):
        reason = ""
        plain = _plain(sentence)
        # a sentence the model wrote twice is shown once ("Relaxing the budget unlocks the most blocks; ..." twice), and
        # outside a block search one that is the start of a facts line is that line said twice ("I couldn't find an HDB
        # rule in my knowledge base that answers this." above the same line with "Please check hdb.gov.sg ...":
        # CLARIFY-03; a block-search answer has its own checks for such sentences below)
        if nothing_found or plain in fact_lines or plain in {_plain(k) for k in kept} \
                or (not searched and len(plain) >= 20 and any(line.startswith(plain) for line in fact_lines)):
            # the facts already say it ("I could not find data for that request."), and with nothing found the model
            # has no evidence to add anything from: the line would show twice, or reworded ("I couldn't find any data")
            reason = "restated"
        elif keep_supported_sentences(sentence, facts, user_request(state), policy)[1]:
            reason = "figures"
        elif not says_something(sentence) or (policy and (_GENERIC_POLICY_LEAD.search(sentence) or _SOURCE_ONLY.match(sentence))):
            reason = "lead_ins"
        elif policy and not policy_opening_is_relevant(sentence, user_request(state)):
            reason = "relevance"
        elif policy and policy_scope_conflict(sentence, user_request(state), shown):
            reason = "scope"
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


def policy_fallback_opening(state: HDBCompassState) -> str:
    """Non-LLM safety net used only when no valid model opening can be produced.

    It does not classify or special-case queries: it quotes the first relevant grounded policy line and leaves the
    actual answer wording to the LLM whenever an LLM is available.
    """
    query = user_request(state)
    passages = state.get("rag_policy_passages") or []
    # the line that shares the most of the question's own words, not simply the first one: "MOP for a Standard flat"
    # had opened with the Plus 10-year line, an MOP question with a stamp-duty line (QA sweep, 3 Nov). A line with no
    # word of the question is not quoted at all.
    stop = {"what", "which", "when", "where", "does", "will", "would", "could", "should", "have", "after", "before", "with",
            "from", "that", "this", "your", "flat", "flats", "hdb", "rule", "rules", "there", "about", "much", "many", "long",
            "need", "they", "them", "their", "still", "able", "allowed", "the", "and", "for", "can", "how",
            # words nearly every passage shares, which would match any rule
            "housing", "grant", "grants", "cpf", "resale", "buy", "buying", "owner", "owners", "singapore", "years", "year",
            "must", "may", "are", "not", "any", "get", "you", "our", "who", "out"}
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if (len(w) > 2 or w.isdigit()) and w not in stop}
    best, best_score = None, 0
    heading = ""
    for i, line in enumerate(policy_fact_lines(passages, query)):
        clean = re.sub(r"^\s*-\s*", "", line).strip()
        if clean.startswith("**"):
            heading = clean
        if not clean or clean.startswith(("**", "Source:")) or len(clean) < 25 or re.match(r"^[\w ]{1,30}:\s*\$?[\d,]", clean) \
                or re.match(r"(?:consider|for example|example|e\.g\.|worked example|scenario|suppose|imagine)\b", clean, re.I):
            continue
        foreign = sibling_terms(f"{heading} {clean}", query)
        if foreign and not scheme_terms(f"{heading} {clean}") & scheme_terms(query):
            # a line about Standard flats for a Prime question, the SPR quota for an EIP one (SEM-07); not one naming
            # the asked scheme too ("Just like Prime flats, Plus flat owners are never permitted to rent out the whole
            # flat" answers a Plus question)
            continue
        parts = _split_opening(clean)
        if clean.endswith(":") and len(parts) > 1:
            clean = parts[0]   # a lead-in's first sentence: the opening is not a list's introduction
        have = set(re.findall(r"[a-z0-9]+", clean.lower()))
        # a word of four letters or more also meets, at half weight, the longer words it starts ("rent" meets
        # "Rental": "Can I rent out my whole Prime flat after the 10-year MOP?" had opened with the Prime MOP row, not
        # the Prime rental row); half, so "citizen" still prefers "Citizen + Permanent Resident" to "Both Citizens"
        score = (len(words & have) + 0.5 * len({w for w in words - have if len(w) >= 4 and any(h.startswith(w) for h in have)})
                 - i * 0.01 - (1 if foreign else 0))   # a line also about a sibling scheme only when no other answers
        if score > best_score:
            best, best_score = clean, score
    if best and best_score >= 2 - 0.5:          # at least two of the question's own words
        return f"The relevant HDB rule is: {best}"
    return "The retrieved HDB rules relevant to your question are summarized below."


def repair_policy_opening(llm: Any, query: str, facts: str) -> str:
    """Ask the same LLM for a replacement when its first policy draft had no usable opening sentence."""
    prompt = (
        f"User question: {query}\n\n"
        "Write exactly one plain-text opening sentence that directly answers the question using the grounded policy "
        "facts below. Start with Yes or No when appropriate. Do not write a heading, a generic introduction, a source "
        "link, a citation-only line, or a list. Reuse at least one substantive term from the question, and do not "
        "introduce a different scheme or topic. Do not answer an unasked policy topic.\n\n"
        f"Grounded policy facts:\n{facts}"
    )
    try:
        response = llm.invoke([SystemMessage(content=SYNTH_SYSTEM), HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            content = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        return strip_think(str(content or "")).strip()
    except Exception:
        return ""


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
    if questions[0].startswith(_WORKPLACE_ASKS) and c.get("unresolved_workplaces"):
        return {"kind": "workplace", "needed": [], "question": questions[0]}   # read as a search follow-up next turn
    # "I have not picked a school yet." is answered with NO_SCHOOL_PICKED, which still asks for a school
    kind = "school" if questions[0] in (ri.SCHOOL_CLARIFICATION, NO_SCHOOL_PICKED) else "place"
    return {"kind": kind, "needed": [], "question": questions[0]}


def _merge_chips(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Suggestion chips from several sources, in order, without repeating a question."""
    out, seen = [], set()
    for group in groups:
        for chip in group or []:
            key = (chip.get("text") or chip.get("label") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(chip)
    return out


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
    if unchecked and not unavailable_tools(state):   # every tool ran, but a stated limit was never checked
        unchecked = (" A limit was not checked: say plainly what could_not_check lists, and never say the blocks meet "
                     "every constraint.")
    rule += unchecked
    clar = clarification_needed(state) or place_clarification(state)
    if clar and clar.get("kind", "financial") == "financial":
        evidence["clarification_needed"] = clar
        rule += (" Some financial facts are missing: state no maximum budget or affordability figure, and end with "
                 "clarification_needed.question.")
    verdict = state.get("auditor_verdict") or {}
    text = ""
    model_text = ""   # what the model wrote and the checks kept: the only part of the answer the auditor judges
    errors: List[str] = []
    # app fast path, or a turn that only asks the user something: code writes it all
    only_asking = bool(c.get("clarifications")) and not state.get("active_subagents")
    facts = template_response(state) if writes_facts_in_code(state) else ""
    policy = policy_answer_passages(state) if facts else None
    # Fast path may skip model calls for fully recognized market/form flows, but policy answers still need the LLM
    # opening. The selected policy passages and the user's question are handed to that call below.
    # a question about the last answer ("Are these live flats?") is answered by code alone: no tool ran this turn, so a
    # model opening would have no evidence to stand on
    llm = get_llm() if not only_asking and not c.get("about_results") \
        and (c.get("extraction_source") != FAST_PATH_SOURCE or policy) else None
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
                elif policy:
                    task = ("Answer the user's specific policy or eligibility question directly in 1 to 2 clear, authoritative sentences. "
                            "State the direct answer upfront (e.g. explicitly state 'No' or 'Yes', or give the exact rule, "
                            "such as flat size restriction, BTO vs resale eligibility, MOP duration, or grant entitlement). "
                            "If the question names several policy topics, cover those topics in the opening sentence, "
                            "reuse at least one substantive term from the question, and do not answer an unasked topic. "
                            "Never output generic placeholders, boilerplate or lead-ins like 'The HDB policy passages that bear on your question are below'. "
                            "Write only your direct answer to the user's question.")
                else:
                    task = ("Write only a short opening (one or two sentences that answer the question directly) and, when "
                            "there are several options, the trade-offs between them (price vs lease vs distance) in words: "
                            "cheaper, more lease left, nearer the MRT station")
                reading = trade if code_answers else ("" if policy else lead)
                policy_context = ("\nRelevant retrieved policy passages selected for this question:\n---\n"
                                  + "\n".join(policy_fact_lines(policy, user_request(state)))
                                  + "\n---\n") if policy else ""
                rule = ("The facts section below was written by code from the tool results and will be shown to the user "
                        "exactly as written, straight after your text:\n---\n" + facts + "\n---\n" + task + ". Do not "
                        "repeat or change any price, grant, budget, distance, lease, count or relaxation figure: " + numbers
                        + ", and do not contradict the facts. Write no headings, lists or notes on taxes, fees or the HFE "
                        "letter: the facts already carry them, and such sentences are deleted too."
                        + (f" An opening in this spirit, about other blocks: {OPENING_EXAMPLE}"
                           if not policy and (len(blocks_listed) > 1 or not code_answers) else "")
                        + (f" What code reads from these facts, which you may reword or add to but must not contradict: "
                           f"\"{reading}\"" if reading else "")
                        + policy_context
                        + unchecked)
            msgs = [sys_msg(SYNTH_SYSTEM),
                    HumanMessage(content=f"User query: {user_request(state)}\n\n{rule}\n\nEvidence JSON:\n{ri.compact_tool_result(evidence, 24000)}")]
            if verdict and not verdict.get("passed", True) and verdict.get("critique"):
                msgs.append(HumanMessage(content=f"Your previous draft was rejected by the auditor: {verdict['critique']}. Rewrite it, fixing every issue."))
            text = llm.invoke(msgs).content
            if isinstance(text, list):
                text = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in text)
            text = strip_think(text)
            model_text = text   # with code_written_facts off the whole draft is the model's
            if facts:
                kept, dropped = check_model_opening(text, state, facts, policy)
                if policy and not kept:
                    # A model sometimes returns only a source link or a generic lead. Repair it with a second,
                    # answer-first instruction instead of deciding the opening from query-specific rules.
                    repaired = repair_policy_opening(llm, user_request(state), facts)
                    repaired_kept, repaired_dropped = check_model_opening(repaired, state, facts, policy)
                    if repaired_kept:
                        text = repaired
                        kept = repaired_kept
                        dropped += repaired_dropped
                model_text = " ".join(kept)
                if code_answers:
                    # code's answer, then code's trade-off unless the model made a (checked) comparison of its own,
                    # then the model's sentences (deduplicating any sentence that repeats code's answer)
                    compares = any(_COMPARISON.search(k) for k in kept)
                    # sentence by sentence: one item of code's answer can hold two ("No block meets ... Relaxing ...")
                    answer_plains = {_plain(sentence) for a in answer for sentence in _split_opening(a)}
                    unique_kept = [k for k in kept if _plain(k) not in answer_plains]
                    opening = " ".join(answer + ([trade] if trade and not compares else []) + unique_kept)
                else:
                    # the model's surviving sentences are preferred; code's opening only when none of them survives
                    opening = " ".join(kept) or lead
                    forced, with_model = forced_opening(state)
                    if forced:
                        # code's refusal or disclosure first; after a refusal the model's lawful explanation may follow
                        opening = " ".join([forced] + [k for k in kept if with_model and _plain(k) != _plain(forced)])
                code_only = not kept and bool(lead)
                why = {"figures": "stated figures", "comparisons": "compared blocks in a way the facts do not bear out",
                       "notes": "repeated a note the facts carry", "leans_back": "leaned on the sentence before"}
                why["relevance"] = "answered a different policy topic from the user's question"
                why["scope"] = "named a scheme the question does not ask about"
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
            model_text = ""
    if not text:
        # no model, or it failed: the code-written facts, led by code's opening (variant C and the app)
        text = (lead + "\n\n" + facts if lead else facts) or template_response(state)
        errors += lead_errors
    elif state.get("financial_breakdown") or state.get("rag_policy_passages"):
        if "HFE" not in text:
            text += HFE_DISCLAIMER
    if clar and clar["question"] not in text:
        text = text.rstrip() + "\n\n" + clar["question"]
    return {"draft_response": text, "model_opening": model_text, "map_ui_payload": payload, "tool_call_log": log,
            "errors": errors,
            # drill-down chips first, then radius chips (both are whole questions, not answers to a pending question)
            "clarification": clar, "suggestions": [] if clar else _merge_chips(ri.drilldown_suggestions(state), ri.comparison_chips(state), ri.price_check_chips(state), ri.radius_chips(c))}


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
        return {**update, **menu_pick_reply(state)}
    query = last_user_query(state)
    unsupported = scope.unsupported_reply(query, len(rule_based_extract(query).work_destinations))
    if unsupported:
        return {**update, "draft_response": unsupported}
    # school questions the lookup cannot answer are stopped here, after Member 5's own check
    # a school named earlier in the conversation counts: "Expand the school distance to 2km." after a search near Ai Tong
    # School is about that school, not a question to ask "Which primary school?" (SPAT-30)
    earlier = bool((state.get("prior_constraints") or {}).get("school_name"))
    reply = guard_school_request(last_user_query(state),
                                 lambda q: earlier or bool(rule_based_extract(q).school_name) or names_a_school(q))
    return {**update, "draft_response": reply} if reply else update


def menu_pick_reply(state: HDBCompassState) -> Dict[str, Any]:
    """The answer to a menu pick (initial_state resolved it into menu_followup): the option's own follow-up question,
    with nothing searched yet. Read by C's input_guardrail_node and by B, whose graph has no guardrail node, so a pick
    in B is answered too instead of showing the menu again."""
    ask = state.get("menu_followup")
    return {"draft_response": ask["question"], "clarification": ask} if ask else {}


# A school's name as people write it, for the school guard only: capitalised words ending in School, Primary or High,
# with "St." abbreviations, hyphens ("Anglo-Chinese"), initials ("CHIJ"), apostrophes and a campus in brackets. The
# extraction regex needs every word to be letters only, so the guard asked "Which primary school?" for "St. Hilda's
# Primary School" although the message named it. Case-sensitive: "a primary school" names no school. "High" is not a
# school when a floor follows it ("Tampines High floor flat").
_SCHOOL_NAME = re.compile(r"(?<![\w'’.-])((?:[A-Z][A-Za-z'’]*\.?(?:-[A-Z][A-Za-z'’]*)*\s+){1,5})"
                          r"(?:Primary School|School|Primary|High(?![\s-]*(?:[Ff]loors?|[Ss]tor(?:e)?ys?|[Ll]evels?|[Rr]ise)\b))"
                          r"\b(?:\s*\([A-Z][A-Za-z ]*\))?")
# Words that describe a kind of school or belong to the sentence, not to a name: a match made only of these (and of
# research_intents' generic school words: "SAP", "Elite", "Reputable"...) names no school, so the guard still asks.
# "Catholic High School" and "Methodist Girls' School" keep a name word ("high", "methodist").
_NOT_A_NAME = frozenset({"a", "an", "the", "any", "my", "our", "her", "his", "their", "your", "which", "what", "moe",
                         "primary", "secondary", "good", "best", "top", "popular", "nearby", "near", "nearest",
                         "closest", "find", "show", "list", "looking", "is", "are", "of", "to", "from", "for", "at",
                         "by", "with", "around", "within", "in", "and", "or", "hdb", "flat", "flats", "block", "blocks",
                         "school", "schools", "catholic", "christian", "gep", "ip", "girls", "boys", "girl", "boy",
                         "chinese", "english", "malay", "tamil", "mission", "international", "private", "government",
                         "aided", "autonomous", "independent", "integrated", "special", "neighbourhood", "neighborhood",
                         "co", "ed", "mixed", "single", "sex", "same", "new", "kid", "kids", "child", "children", "son",
                         "daughter"})


def names_a_school(text: str) -> bool:
    """Whether the message names a specific school (_SCHOOL_NAME): at least one word of the name is not a generic or
    descriptive word such as "The", "Primary", "MOE", "Catholic" or "SAP"."""
    for m in _SCHOOL_NAME.finditer(text or ""):
        words = [re.sub(r"['’]s?$", "", w.strip(".")).lower() for w in re.split(r"[\s-]+", m.group(1)) if w]
        name = [w for w in words if w and w not in _NOT_A_NAME]
        if name and not ri.generic_school_phrase(" ".join(name)):
            return True
    return False


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


def refused_by_guardrail(state: Optional[Dict[str, Any]]) -> bool:
    return any(str(e).startswith("input_guardrail:") for e in (state or {}).get("errors") or [])


def before_refused_turn(previous: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A turn the input guardrail refused carries nothing forward: the conversation continues from the turn before it
    (its messages, constraints and open question), so a refused "guaranteed profit on a BTO launch" does not make the
    next grants question read as a BTO purchase."""
    if not refused_by_guardrail(previous):
        return previous
    msgs = list(previous.get("messages") or [])
    users = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "user"]
    return {"messages": msgs[:users[-1]] if users else [], "user_constraints": previous.get("prior_constraints"),
            "clarification": previous.get("pending_clarification")}


def initial_state(query: str, variant: str, map_state: Optional[Dict[str, Any]] = None,
                  previous: Optional[Dict[str, Any]] = None, fast_path: bool = False) -> HDBCompassState:
    """previous: the final state of the last turn in the same conversation. Its messages, constraints and any
    open follow-up question carry forward, so a reply such as "we have $25,000 in cash" completes the earlier request.
    fast_path=True (the app) answers fully recognised self-contained questions without model calls; experiments leave
    it off so every variant is measured on its model's own choices."""
    query = sanitizer.redact(query)[0]   # every entry point (app, evaluation, command line) builds its state here
    previous = before_refused_turn(previous)
    history = list((previous or {}).get("messages") or [])
    # a reply to a menu (or to the question one of its options asked) is completed by code: an option's own follow-up
    # question is answered at once, and a bare launch name becomes the full question about it (agents/scope.py)
    menu_reply = scope.resolve_menu_reply((previous or {}).get("clarification"), query) if previous else None
    followup = (menu_reply or {}).get("ask")
    query = (menu_reply or {}).get("query") or query
    return {"menu_followup": followup, "messages": history + [{"role": "user", "content": query}], "variant": variant, "active_subagents": [],
            "candidate_blocks_intersection": [], "map_ui_payload": map_state or {}, "draft_response": "",
            "auditor_verdict": {}, "model_opening": "", "tool_call_log": [], "errors": [], "subagent_notes": [],
            "prior_constraints": (previous or {}).get("user_constraints"),
            "pending_clarification": (previous or {}).get("clarification"), "clarification": None,
            # the last list shown stays the reference through a turn that listed none ("tell me more about the first one")
            "prior_blocks": [f"Blk {b.get('block')} {b.get('street')}" for b in
                             ((previous or {}).get("candidate_blocks_intersection") or [])[:LISTED_BLOCKS]]
                            or list((previous or {}).get("prior_blocks") or []),
            "prior_block_facts": [{"name": f"Blk {b.get('block')} {b.get('street')}", "price": b.get("median_resale_price"),
                                   "lease": b.get("median_remaining_lease_years"),
                                   "dist": min([v for v in (b.get("distances") or {}).values() if isinstance(v, (int, float))] or [None],
                                               key=lambda v: float("inf") if v is None else v)}
                                  for b in ((previous or {}).get("candidate_blocks_intersection") or [])[:LISTED_BLOCKS]]
                                 or list((previous or {}).get("prior_block_facts") or []),
            "fast_path": fast_path}


def conversation_text(state: HDBCompassState) -> str:
    """The latest message, preceded by the earlier turns when there are any (for agents without structured memory)."""
    msgs = [m for m in state.get("messages", []) if isinstance(m, dict)]
    if len(msgs) <= 1:
        return last_user_query(state)
    return "\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in msgs)
