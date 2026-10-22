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
from agents.schemas import (
    AuditorVerdict,
    ExtractedConstraints,
    RelaxationSuggestion,
    RelaxUnsatisfiedConstraintsOutput,
    TOOL_OWNER,
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
    m = re.search(r"(?:of|near|to)\s+((?:[A-Z][a-zA-Z'’]+\s){1,4}(?:Primary School|School|Primary|High))", q)
    if m:
        school = m.group(1).strip()
        radii.setdefault("school", 1000)

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
    if re.search(r"\b(mop|minimum occupation|prime|plus|standard flat|clawback|subsidy recovery|eip|ethnic|quota|wait[- ]out|eligib|scheme|rental restriction|exempt)\b", q, re.I):
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
)


def extract_constraints_node(state: HDBCompassState) -> Dict[str, Any]:
    query = last_user_query(state)
    llm = get_llm()
    extracted: Optional[ExtractedConstraints] = None
    errors: List[str] = []
    if llm is not None:
        for method in ("json_schema", "json_mode"):
            try:
                structured = llm.with_structured_output(ExtractedConstraints, method=method)
                extracted = structured.invoke([sys_msg(EXTRACT_SYSTEM), HumanMessage(content=query)])
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
        extracted = rule_based_extract(query)
    return {"user_constraints": extracted.model_dump(), "errors": errors}


# ---------------------------------------------------------------------------
# Node: route  +  fan-out dispatch
# ---------------------------------------------------------------------------
def decide_routing(c: Dict[str, Any]) -> List[str]:
    active: List[str] = []
    if c.get("towns") and (c.get("max_budget_sgd") or c.get("min_remaining_lease_years") or c.get("flat_types")
                           or c.get("wants_price_trend") or c.get("wants_bto_comparison")):
        active.append("market_sql")
    if c.get("amenity_radii_meters") or c.get("school_name") or len(c.get("work_destinations") or []) >= 2:
        active.append("spatial")
    if c.get("household_income") or c.get("cpf_oa_balance") or (c.get("citizenship_scheme") and c.get("max_budget_sgd")):
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
                  "exact towns/flat types from the constraints. When done, reply with a one-paragraph summary.",
    "spatial": "You are the Geospatial & Commute agent. Use your tools to find HDB blocks meeting amenity-distance, "
               "school-boundary and commute constraints. Pass towns exactly as given. Reply with a short summary when done.",
    "financial": "You are the Financial & Grant Calculator agent. Never do arithmetic yourself: call the calculators "
                 "for grants, affordability (MSR/LTV) and upfront outlay (BSD). Reply with a short summary when done.",
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
                    result = runners[name](**args)
                    used_llm_tools = True
                    outputs[name] = result
                    msgs.append(ToolMessage(content=json.dumps(result, default=str)[:12000], tool_call_id=tc["id"]))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: LLM loop failed: {type(e).__name__}: {str(e)[:160]}")

    if not used_llm_tools:
        if llm is not None:
            errors.append(f"{label}: no valid tool call from model; deterministic plan used")
        for agent in (plan_agents or [label]):
            for name, args in deterministic_plan(agent, c, query):
                if name in runners:
                    outputs[name] = runners[name](**args)

    log = [rec.model_dump() for r in runners.values() for rec in r.log]
    return outputs, log, errors, final_text


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


def make_specialist_node(agent: str) -> Callable[[HDBCompassState], Dict[str, Any]]:
    def node(state: HDBCompassState) -> Dict[str, Any]:
        c = state.get("user_constraints", {})
        outputs, log, errors, final = run_react(agent, tools_for_agent(agent), AGENT_SYSTEM[agent], last_user_query(state), c)
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


def _affordable_price(state: HDBCompassState) -> Optional[float]:
    fin = state.get("financial_breakdown") or {}
    aff = fin.get("calculate_maximum_affordability") or {}
    grants = (fin.get("calculate_cpf_housing_grants") or {}).get("total_grants", 0.0)
    if aff.get("max_purchase_price"):
        return float(aff["max_purchase_price"]) + float(grants or 0.0)
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
    cap = _affordable_price(state) if not state.get("user_constraints", {}).get("max_budget_sgd") else None
    return {"candidate_blocks_intersection": compute_intersection(_sql_blocks(state), state.get("spatial_block_candidates"), cap)}


# ---------------------------------------------------------------------------
# Node: relax_constraints (use case E2)
# ---------------------------------------------------------------------------
def should_relax(state: HDBCompassState) -> str:
    c = state.get("user_constraints", {})
    searched = {"market_sql", "spatial"} & set(state.get("active_subagents", []))
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
    c = state.get("user_constraints", {})
    lines: List[str] = []
    blocks = state.get("candidate_blocks_intersection") or []
    searched = blocks or state.get("sql_market_results") or state.get("spatial_block_candidates") is not None
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
    if "compute_town_price_timeseries" in sql:
        for t in sql["compute_town_price_timeseries"].get("summary", []):
            lines.append(f"- {t['town']}: latest median ${t['latest_median_price']:,.0f}, ${t['median_price_per_sqm']:,.0f}/sqm, YoY {t['yoy_appreciation_pct']:+.1f}%")
    if "compare_bto_launch_vs_resale" in sql:
        b = sql["compare_bto_launch_vs_resale"]
        lines.append(f"- BTO {b['bto_project_name']}: ${b['bto_indicative_price_range']['min']:,.0f}–${b['bto_indicative_price_range']['max']:,.0f} vs nearby resale median ${b['resale_median_nearby']:,.0f} ({b['price_gap_pct']:+.1f}%)")
    fin = state.get("financial_breakdown") or {}
    if fin:
        lines.append("**Financials:**")
        g = fin.get("calculate_cpf_housing_grants")
        if g:
            lines.append(f"- Grants: EHG ${g['enhanced_housing_grant']:,.0f} + Family ${g['family_grant']:,.0f} + PHG ${g['proximity_housing_grant']:,.0f} = **${g['total_grants']:,.0f}**")
        a = fin.get("calculate_maximum_affordability")
        if a:
            lines.append(f"- Max purchase price ${a['max_purchase_price']:,.0f} (loan ${a['max_loan_amount']:,.0f}, instalment ${a['monthly_instalment']:,.0f}/mth, binding: {a['binding_constraint']})")
        u = fin.get("calculate_bsd_and_upfront_outlay")
        if u:
            lines.append(f"- Upfront: BSD ${u['buyer_stamp_duty']:,.0f}, cash required ${u['cash_required']:,.0f}, total ${u['total_upfront']:,.0f}")
    passages = state.get("rag_policy_passages") or []
    if passages:
        lines.append("**Policy:**")
        for p in passages[:3]:
            lines.append(f"- {p['text']} [{p['source_url']}]")
    if not lines:
        lines.append("I could not find data for that request.")
    if fin or passages:
        lines.append(HFE_DISCLAIMER)
    return "\n".join(lines)


SYNTH_SYSTEM = (
    "You are HDB-Compass, a Singapore public-housing decision assistant. Write the final answer for the user from the "
    "JSON evidence only. Every number, price, block, distance and grant amount MUST appear verbatim in the evidence; "
    "never invent or round differently. Cite policy passages by their source_url. Present trade-offs objectively "
    "(price vs lease vs distance). No speculation about future prices or profits. Use short markdown sections."
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


def synthesis_evidence(state: HDBCompassState) -> Tuple[Dict[str, Any], str]:
    """What the final writing step may see, by architecture.
    C (and the default): the supervisor's join is the ONLY source of blocks. Raw price-query rows are withheld, so the
      writer cannot re-introduce a block the intersection excluded (the failure seen in the first Category E run).
    B: no join in code by design. The writer gets the data agent's own reconciliation plus the raw price AND distance
      results, and must do the join itself."""
    base = {"user_constraints": state.get("user_constraints"),
            "financial_breakdown": state.get("financial_breakdown"),
            "rag_policy_passages": state.get("rag_policy_passages")}
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
    return ({**base, "matching_blocks": matches[:50], "match_counts": counts, "market_context": market_context,
             "relaxation": state.get("relaxation")}, GROUNDED_RULE)


def synthesize_node(state: HDBCompassState) -> Dict[str, Any]:
    c = state.get("user_constraints", {})
    blocks = state.get("candidate_blocks_intersection") or []
    log: List[Dict[str, Any]] = []
    payload = build_map_payload(blocks, c, log) if blocks else {}
    evidence, rule = synthesis_evidence(state)
    verdict = state.get("auditor_verdict") or {}
    text = ""
    errors: List[str] = []
    llm = get_llm()
    if llm is not None:
        try:
            msgs = [sys_msg(SYNTH_SYSTEM),
                    HumanMessage(content=f"User query: {last_user_query(state)}\n\n{rule}\n\nEvidence JSON:\n{json.dumps(evidence, default=str)[:24000]}")]
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
        from agents import guardrails  # Member 5's module (proposal §8: agents/guardrails.py)
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


def initial_state(query: str, variant: str, map_state: Optional[Dict[str, Any]] = None) -> HDBCompassState:
    return {"messages": [{"role": "user", "content": query}], "variant": variant, "active_subagents": [],
            "candidate_blocks_intersection": [], "map_ui_payload": map_state or {}, "draft_response": "",
            "auditor_verdict": {}, "tool_call_log": [], "errors": [], "subagent_notes": []}
