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
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm import active_profile, get_llm, load_config
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
from agents.school_queries import school_location_request, present_school_result
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
    m = re.search(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b", s, re.I)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    return v * (1000 if unit == "k" else 1_000_000 if unit == "m" else 1)


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
    flat_types = [f"{n} ROOM" for n in re.findall(r"\b([2-5])[- ]room", q, re.I)]
    if re.search(r"\bexecutive\b", q, re.I):
        flat_types.append("EXECUTIVE")
    flat_types = list(dict.fromkeys(flat_types))

    budget = None
    m = re.search(r"(?:under|below|less than|max(?:imum)?|up to|budget(?: of)?)\s*\$?\s?(\d[\d,]*(?:\.\d+)?\s*[km]?)", q, re.I)
    if m:
        budget = _money(m.group(1))

    lease = None
    m = re.search(r"(\d{2})\s*\+?\s*years?\s*(?:of\s*)?(?:remaining\s*)?lease|(?:at least|minimum|min)\s*(\d{2})\s*years", q, re.I)
    if m:
        lease = int(m.group(1) or m.group(2))

    radii: Dict[str, int] = {}
    for m in re.finditer(r"(?:within\s*)?(\d+(?:\.\d+)?)\s*(km|m)\b\s*(?:of|from|to)\s+(?:an?\s+|the\s+)?([a-z /-]+?)(?=[,.;?!]|\s+(?:with|within|that|which|under|over|have|has)\b|\s+and\s+(?:\d|have|has|at\s+least)|$)", q, re.I):
        dist = float(m.group(1)) * (1000 if m.group(2).lower() == "km" else 1)
        targets = m.group(3).lower()
        for word, key in AMENITY_WORDS.items():
            if word in targets:
                radii[key] = int(dist)
    for m in re.finditer(r"\b(mrt|hawker|supermarket|bus stop|park)[a-z ]*?within\s*(\d+)\s*m\b", q, re.I):
        radii[AMENITY_WORDS[m.group(1).lower()]] = int(m.group(2))

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

    income = None
    m = re.search(r"(?:earn(?:ing)?|income(?: of)?|salary(?: of)?)\s*(?:is\s*)?\$?\s?(\d[\d,]*(?:\.\d+)?\s*k?)", q, re.I)
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
    if re.search(r"\b(mop|minimum occupation|prime|plus (?:flats?|models?|bto|resale|classification)|standard flat|clawback|subsidy recovery|eip|ethnic|quota|wait[- ]out|eligib|scheme|rental restriction|exempt)\b", q, re.I):
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
    "For comparable transactions, use comparable_resale with explicit verified latitude, longitude and flat type. "
    "Never invent coordinates or substitute a town centre for a project."
    " School-distance and Primary 1 boundary searches belong to the spatial agent; "
    "they are not HDB housing-policy questions. Do not invent a school name."
    " addresses: copy every HDB block address, postal code or named place (MRT station, hawker centre, school) "
    "exactly as written. asking_price_sgd: only a specific flat's asking/offer price to check. "
    "wants_block_history: past sales of a named block. wants_price_assessment: is a price fair/reasonable. "
    "wants_option_comparison: comparing two or more named blocks. wants_nearby_amenities: what is near a place. "
    "wants_market_snapshot: how the market or prices stand now. wants_town_ranking: which town is best value or "
    "cheapest. trend_years: N in 'over the last N years'. bto_project_name: a named BTO launch. "
    "Leave amenity_radii_meters fields null unless a distance is stated. Respond with JSON only."
)


def extract_constraints_node(state: HDBCompassState) -> Dict[str, Any]:
    query = last_user_query(state)
    if school_location_request(query) is not None:
        return {"user_constraints": rule_based_extract(query).model_dump(), "errors": []}
    if state.get("fast_path"):
        # App mode: when the regex already pins down every tool the question needs, a model
        # round trip adds latency but no information.
        fast = rule_based_extract(query).model_dump()
        _verify_extraction(fast, query, [], from_model=False)
        _apply_map_pin(fast, query, state)
        if ri.answerable_without_model(fast, query):
            fast["extraction_source"] = FAST_PATH_SOURCE
            return {"user_constraints": fast, "errors": []}
    llm = get_llm()
    extracted: Optional[ExtractedConstraints] = None
    errors: List[str] = []
    source = "regex"
    if llm is not None:
        for method in ("json_schema", "json_mode", "function_calling"):
            try:
                structured = llm.with_structured_output(LLMExtractedConstraints, method=method)
                prompt = EXTRACT_SYSTEM
                if method == "json_mode":
                    # JSON mode needs the word "json" and the schema in the prompt itself.
                    prompt += " JSON schema: " + json.dumps(LLMExtractedConstraints.model_json_schema())
                result = structured.invoke([sys_msg(prompt), HumanMessage(content=query)])
                if result is None:
                    raise ValueError("model returned no structured output")
                extracted = result.to_constraints()
                source = f"llm:{method}"
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"extract_constraints[{method}] failed: {type(e).__name__}: {str(e)[:120]}")
        if extracted is not None:
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
    _verify_extraction(constraints, query, errors, from_model=source != "regex")
    _apply_map_pin(constraints, query, state)
    constraints["extraction_source"] = source
    return {"user_constraints": constraints, "errors": errors}


FAST_PATH_SOURCE = "regex:fast_path"


def _apply_map_pin(c: Dict[str, Any], query: str, state: HDBCompassState) -> None:
    map_state = state.get("map_ui_payload") or {}
    pin = map_state.get("center")
    if isinstance(pin, (list, tuple)) and len(pin) == 2:
        ri.apply_map_pin(c, query, pin, radius_meters=map_state.get("radius_meters"))


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
def decide_routing(c: Dict[str, Any]) -> List[str]:
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
    if c.get("financial_inputs") or c.get("household_income") or c.get("cpf_oa_balance") or (c.get("citizenship_scheme") and c.get("max_budget_sgd")):
        active.append("financial")
    if c.get("policy_question"):
        active.append("policy_rag")
    if not active and (c.get("addresses") or c.get("map_pin")):
        pass  # a place with no recognised question: route_node asks instead of guessing a search or policy query
    elif not active and c.get("towns"):
        active.append("market_sql")
    elif not active:
        active.append("policy_rag")
    if c.get("clarifications") and active == ["policy_rag"] and not c.get("policy_question"):
        active = []  # only a clarifying question to ask; nothing to retrieve
    return active


def route_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    active = decide_routing(c)
    if not active and not c.get("clarifications") and (c.get("addresses") or c.get("map_pin")):
        return {"active_subagents": active, "user_constraints": {**c, "clarifications": [ri.place_clarification(c)]}}
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
    "market_sql": "You are the Market & Valuation SQL agent for Singapore HDB resale flats. Pick tools by question: "
                  "blocks matching price/lease -> query_resale_blocks_by_criteria; town trends or 'over the last N "
                  "years' -> compute_town_price_timeseries (lookback_years=N; use period_change_pct); the market now -> "
                  "get_market_snapshot; which town is best value -> rank_towns_by_value; a named block's past sales -> "
                  "get_block_transaction_history; is an asking price fair -> assess_resale_price; two or more named "
                  "blocks -> compare_resale_options; a named BTO launch -> compare_bto_launch_vs_resale. Prices AROUND a "
                  "place (an area, not one block) -> find_resale_blocks_near (which blocks, each block's median), then "
                  "get_area_resale_statistics (pooled distribution) and, for change over time, get_area_price_trend; pass "
                  "the same area={address, radius_meters} to each, or block_keys from find_resale_blocks_near. "
                  "get_block_transaction_history also takes area or block_keys. Pass the user's "
                  "addresses verbatim; these tools resolve them. Never estimate coordinates: query_comparable_resale_"
                  "transactions only takes coordinates from a tool result or the user's pin. Call tools with exact "
                  "towns/flat types from the constraints. Disclose small samples, truncated examples and missing geocodes; "
                  "do not widen filters silently or present comparables as an official valuation. "
                  "When done, reply with a one-paragraph summary.",
    "spatial": "You are the Geospatial & Commute agent. Use your tools to find HDB blocks meeting amenity-distance, "
               "school-boundary and commute constraints, and to describe places: get_nearby_amenities for 'what is near X', "
               "resolve_location to turn a user's place into coordinates. Pass towns and addresses exactly as given; "
               "never estimate coordinates. Commute times are town-level estimates, not routed journeys; say so. "
               "Reply with a short summary when done.",
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
        if towns and ri.is_block_search(c, query):
            plan.append(("query_resale_blocks_by_criteria", dict(
                towns=towns, flat_types=c.get("flat_types") or ["4 ROOM"],
                max_price=c.get("max_budget_sgd") or 5_000_000, min_lease_years=c.get("min_remaining_lease_years") or 0)))
        if c.get("wants_price_trend") and towns:
            plan.append(("compute_town_price_timeseries", dict(towns=towns, flat_type=(c.get("flat_types") or ["4 ROOM"])[0],
                                                               lookback_years=c.get("trend_years") or 3)))
        plan.extend(ri.research_plan(agent, c, query))
        if c.get("wants_bto_comparison") and not c.get("bto_project_name"):
            m = re.search(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)\s+(?:BTO|launch)", query)
            plan.append(("compare_bto_launch_vs_resale", dict(bto_project_name=m.group(1) if m else (towns[0] if towns else "Recent BTO"), radius_meters=500)))
    elif agent == "spatial":
        if c.get("nearby_schools"):
            return [("find_primary_schools_near_address", c["nearby_schools"])]
        radii = {k: v for k, v in (c.get("amenity_radii_meters") or {}).items() if k != "school"}
        if radii:
            plan.append(("filter_blocks_by_amenity_proximity", dict(towns=towns, amenity_constraints=radii)))
        if c.get("school_name"):
            r = (c.get("amenity_radii_meters") or {}).get("school", 1000)
            bracket = "1-2km" if re.search(r"\b1\s*[-–]\s*2\s*km\b", query, re.I) else ("1km" if r <= 1000 else "2km")
            plan.append(("verify_primary_school_distance", dict(school_name=c["school_name"], bracket=bracket)))
        plan.extend(ri.research_plan(agent, c, query))
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
                    known = list(known_points or []) + list(ri.coordinates_in(outputs)) + list(ri.coordinates_in(c.get("comparable_resale") or {}))
                    rejection = ri.unverified_coordinates(name, args, known)
                    if rejection:
                        errors.append(f"{label}: rejected unverified coordinates for {name}")
                        msgs.append(ToolMessage(content=json.dumps({"error": rejection}), tool_call_id=tc["id"]))
                        continue
                    if name == "compute_dual_transit_commute" and not args.get("candidate_blocks"):
                        args = {**args, "candidate_blocks": _commute_candidates(outputs)}
                    result = runners[name](**args)
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
            for name, args in ri.research_plan(agent, c, query):
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


def make_specialist_node(agent: str) -> Callable[[HDBCompassState], Dict[str, Any]]:
    def node(state: HDBCompassState) -> Dict[str, Any]:
        c = state.get("user_constraints", {})
        pin = (state.get("map_ui_payload") or {}).get("center")
        known = [tuple(pin)] if isinstance(pin, (list, tuple)) and len(pin) == 2 else []
        outputs, log, errors, final = run_react(agent, tools_for_agent(agent), AGENT_SYSTEM[agent], last_user_query(state), c,
                                                known_points=known, use_model=c.get("extraction_source") != FAST_PATH_SOURCE,
                                                stop_when_covered=bool(state.get("fast_path")))
        update = slot_outputs(outputs, c.get("towns") or [])
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
        # A block with no price evidence has not been shown to be within budget.
        result = [b for b in result if b.get("median_resale_price") is not None and b["median_resale_price"] <= price_cap]
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
    update["candidate_blocks_intersection"] = compute_intersection(_sql_blocks(state), state.get("spatial_block_candidates"), cap)
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


def template_response(state: HDBCompassState) -> str:
    if state.get("nearby_school_results") is not None:
        return present_school_result(state["nearby_school_results"])[0]
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
                kind = f" {b['flat_type']}" if b.get("flat_type") and len({x.get("flat_type") for x in blocks}) > 1 else ""
                lines.append(f"- Blk {b['block']} {b['street']} ({b.get('town','')}){kind}{price}{lease}{(' — ' + dist) if dist else ''}")
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
    if "compare_bto_launch_vs_resale" in sql and sql["compare_bto_launch_vs_resale"].get("status", "ok") == "ok" \
            and "error" not in sql["compare_bto_launch_vs_resale"]:
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
    if passages:
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
KEY_FACTS_RULE = (
    "key_facts holds the verified figures computed in code. Copy every number, sign (+/-), ranking and time "
    "window from key_facts exactly; do not recompute, re-rank or describe a trend direction differently. "
    "Do not mention towns, budgets or preferences the user did not state.")
RECONCILE_RULE = (
    "The data agent's summary and the raw tool results are below; no code has joined them. Recommend only blocks "
    "the evidence shows satisfy every stated constraint: price, lease and every distance limit. A block missing from "
    "spatial_block_candidates has not been shown to meet the distance limits.")


def _compact_spatial(rows: Any, cap: int = 300) -> Any:
    if not isinstance(rows, list):
        return rows
    return [{k: b.get(k) for k in ("block", "street", "distances")} if isinstance(b, dict) else b for b in rows[:cap]]


def synthesis_evidence(state: HDBCompassState) -> Tuple[Dict[str, Any], str]:
    """What the final writing step may see, by architecture.
    C (and the default): the supervisor's join is the ONLY source of blocks. Raw price-query rows are withheld, so the
      writer cannot re-introduce a block the intersection excluded (the failure seen in the first Category E run).
    B: no join in code by design. The writer gets the data agent's own reconciliation plus the raw price AND distance
      results, and must do the join itself."""
    base = {"user_constraints": state.get("user_constraints"),
            "financial_breakdown": state.get("financial_breakdown"),
            "rag_policy_passages": state.get("rag_policy_passages"),
            "location_results": state.get("location_results")}
    sql = state.get("sql_market_results") or {}
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
    if sql_rows is None and spatial is None:
        return ({**base, "market_context": market_context}, RESEARCH_RULE)
    return ({**base, "matching_blocks": matches[:50], "match_counts": counts, "market_context": market_context,
             "relaxation": state.get("relaxation")}, GROUNDED_RULE)


def synthesize_node(state: HDBCompassState) -> Dict[str, Any]:
    if state.get("nearby_school_results") is not None:
        text, payload = present_school_result(state["nearby_school_results"])
        return {"draft_response": text, "map_ui_payload": payload}
    c = state.get("user_constraints", {})
    blocks = state.get("candidate_blocks_intersection") or []
    log: List[Dict[str, Any]] = []
    payload = build_map_payload(blocks, c, log) if blocks else ri.research_map_payload(state)
    evidence, rule = synthesis_evidence(state)
    if state.get("variant") != "B":  # B's writer must reconcile raw results itself, by design
        evidence = {"key_facts": template_response({**state, "map_ui_payload": payload}), **evidence}
        rule = f"{rule} {KEY_FACTS_RULE}"
    verdict = state.get("auditor_verdict") or {}
    text = ""
    errors: List[str] = []
    llm = get_llm() if c.get("extraction_source") != FAST_PATH_SOURCE else None  # the template is the answer
    if llm is not None:
        try:
            msgs = [sys_msg(SYNTH_SYSTEM),
                    HumanMessage(content=f"User query: {last_user_query(state)}\n\n{rule}\n\nEvidence JSON:\n{ri.compact_tool_result(evidence, 24000)}")]
            if verdict and not verdict.get("passed", True) and verdict.get("critique"):
                msgs.append(HumanMessage(content=f"Your previous draft was rejected by the auditor: {verdict['critique']}. Rewrite it, fixing every issue."))
            text = llm.invoke(msgs).content
            if isinstance(text, list):
                text = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in text)
            text = strip_think(text)
        except Exception as e:  # noqa: BLE001
            errors.append(f"synthesize: LLM failed, template used: {type(e).__name__}")
    if not text:
        text = template_response(state)
    elif state.get("financial_breakdown") or state.get("rag_policy_passages"):
        if "HFE" not in text:
            text += HFE_DISCLAIMER
    return {"draft_response": text, "map_ui_payload": payload, "tool_call_log": log, "errors": errors}


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
        update = _INPUT_GUARDRAIL(state)
        if update.get("draft_response"):
            return update
    else:
        update = {}
    query = last_user_query(state)
    nearby = school_location_request(query)
    if nearby is not None:
        if not 0 < nearby["radius_meters"] <= 10000:
            return {**update, "draft_response": "Please choose a school search radius greater than 0 and at most 10,000 metres."}
        if re.search(r"\bsecondary\b|\bjunior college\b|\bpreschools?\b", query, re.I):
            return {**update, "draft_response": "This school lookup currently covers primary schools only. Please resend the address if you want a primary-school search."}
        return update
    school_topic = re.search(r"\bschools?\b|\bmoe\b|\bprimary\s*1\b", query, re.I)
    distance_topic = re.search(r"\b[12]\s*(?:[-–]\s*2\s*)?km\b|\b[12]000\s*m\b|boundar|radius|distance", query, re.I)
    if school_topic and distance_topic and not rule_based_extract(query).school_name:
        return {**update, "draft_response": (
            "Which primary school would you like to check? Please include its name and the distance band, "
            "for example: ‘Show HDB blocks within 1km of Ai Tong School’ or "
            "‘Show HDB blocks within 1–2km of Catholic High School’. "
            "I can map approximate distance bands; official registration distances should be checked with MOE."
        )}
    return update


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
                  fast_path: bool = False) -> HDBCompassState:
    """fast_path=True (the app) answers fully recognised questions without model calls;
    experiments leave it off so every variant is measured on its model's own choices."""
    return {"messages": [{"role": "user", "content": query}], "variant": variant, "active_subagents": [],
            "candidate_blocks_intersection": [], "map_ui_payload": map_state or {}, "draft_response": "",
            "auditor_verdict": {}, "tool_call_log": [], "errors": [], "subagent_notes": [], "fast_path": fast_path}
