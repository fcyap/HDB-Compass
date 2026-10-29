"""
Experiment 1 metrics (evaluation/architecture_metrics.py). Owned by Member 4.
Pure functions over one benchmark item + one variant's final HDBCompassState.
Member 5's runner (run_architecture_eval.py) calls these per query and
aggregates; Member 4 presents the aggregated A vs B vs C numbers.

Proposal §6.3:
  Tool Selection F1       — predicted tool names vs expected_mcp_calls
  Parameter Exact-Match % — for matched tools, required_params ⊆ actual arguments
  Constraint Satisfaction — recommended blocks satisfying 100% of user constraints
  Routing accuracy        — active_subagents vs expected_agents (C only; A/B map to coarse labels)
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().upper()
    if isinstance(v, (list, tuple)):
        return sorted(_norm(x) for x in v)
    if isinstance(v, dict):
        return {str(k).lower(): _norm(x) for k, x in v.items()}
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return v


def tool_selection_prf(expected_calls: List[Dict[str, Any]], tool_call_log: List[Dict[str, Any]]) -> Dict[str, float]:
    exp: Set[str] = {c["tool"] for c in expected_calls if c["tool"] != MAP_TOOL}
    pred: Set[str] = {r["tool"] for r in tool_call_log
                      if r.get("ok", True) and r.get("agent") != "supervisor" and r["tool"] != MAP_TOOL}
    tp = len(exp & pred)
    p = tp / len(pred) if pred else (1.0 if not exp else 0.0)
    r = tp / len(exp) if exp else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "expected": sorted(exp), "predicted": sorted(pred)}


def parameter_exact_match(expected_calls: List[Dict[str, Any]], tool_call_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each expected call whose tool was invoked, every required_param must equal the actual argument."""
    by_tool: Dict[str, List[Dict[str, Any]]] = {}
    for r in tool_call_log:
        if r.get("agent") != "supervisor":
            by_tool.setdefault(r["tool"], []).append(r.get("arguments") or {})
    checked, matched, details = 0, 0, []
    for c in expected_calls:
        actuals = by_tool.get(c["tool"])
        if not actuals:
            continue
        checked += 1
        req = _norm(c.get("required_params") or {})
        ok = any(all(_norm(a.get(k)) == v for k, v in req.items()) for a in actuals)
        matched += int(ok)
        details.append({"tool": c["tool"], "match": ok})
    return {"checked": checked, "matched": matched, "rate": (matched / checked) if checked else None, "details": details}


def constraint_satisfaction_rate(user_constraints: Dict[str, Any], blocks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    blocks = list(blocks)
    if not blocks:
        return {"recommended": 0, "satisfying": 0, "rate": None}
    budget = user_constraints.get("max_budget_sgd")
    lease = user_constraints.get("min_remaining_lease_years")
    radii = user_constraints.get("amenity_radii_meters") or {}
    towns = {t.upper() for t in (user_constraints.get("towns") or [])}
    ok = 0
    for b in blocks:
        # Strict: a stated constraint must be VERIFIED by evidence the variant retrieved.
        # A block recommended without a price (or lease, or distance) cannot be shown to satisfy it.
        good = not b.get("_fabricated", False)
        if budget and (b.get("median_resale_price") is None or b["median_resale_price"] > budget):
            good = False
        if lease and (b.get("median_remaining_lease_years") is None or b["median_remaining_lease_years"] < lease):
            good = False
        if towns and b.get("town") and b["town"].upper() not in towns:
            good = False
        for k, limit in radii.items():
            if k == "school":
                continue
            d = (b.get("distances") or {}).get(f"{k}_dist_m")
            if d is None or d > limit:
                good = False
        ok += int(good)
    return {"recommended": len(blocks), "satisfying": ok, "rate": ok / len(blocks)}


import re as _re

_BLOCK_RE = _re.compile(r"\b(?:Blk|Block)\s*\**\s*[#:]?\s*\**\s*([0-9]{1,4}[A-Z]?)\b", _re.I)  # also "**Block**: 62"
_HEADING_RE = _re.compile(r"^\s*(#{1,6})\s+(.*)$")
_PSEUDO_HEADING_RE = _re.compile(r"^\s*\*\*([^*]+)\*\*:?\s*$")
# words that mark a section or line as NOT a recommendation (near misses, exclusions, alternatives)
_NOT_REC = _re.compile(r"\b(not|n't|except|exclud\w*|fail\w*|outside|exceed\w*|over budget|too far|beyond|"
                       r"near[- ]?miss\w*|partial\w*|most criteria|some criteria|alternative\w*|relax\w*|instead|"
                       r"other options|unsuitable|reject\w*|cannot|no longer|missing)\b", _re.I)
MAP_TOOL = "sync_interactive_map_ui"   # a UI side-effect: scored as map sync, not as tool selection


def _cells(line: str) -> List[str]:
    return [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]


def recommended_block_numbers(draft: str) -> List[str]:
    """Block numbers the answer PRESENTS AS MATCHES. Skips sections whose heading (or any parent heading)
    marks exclusions or near misses, skips negated lines, and reads the Block column of markdown tables."""
    stack: List[tuple] = []          # (level, excluded)
    out: List[str] = []
    table_col: Optional[int] = None
    for raw in (draft or "").splitlines():
        line = raw.rstrip()
        h = _HEADING_RE.match(line)
        ph = None if h else _PSEUDO_HEADING_RE.match(line)
        if h or ph:
            level, text = (len(h.group(1)), h.group(2)) if h else (7, ph.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, bool(_NOT_REC.search(text))))
            table_col = None
            excluded = any(e for _, e in stack)
            if not excluded and h:                      # "#### Block 201 - Tampines St 21" counts
                out += [m.upper() for m in _BLOCK_RE.findall(text)]
            continue
        excluded = any(e for _, e in stack)
        if line.lstrip().startswith("|"):
            cells = _cells(line)
            if table_col is None:
                for i, c in enumerate(cells):
                    if _re.fullmatch(r"(blk|block)( no\.?| number)?", c, _re.I):
                        table_col = i
                continue_row = table_col is not None and any(_re.fullmatch(r"(blk|block)( no\.?| number)?", c, _re.I) for c in cells)
                if continue_row:
                    continue
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue                                 # separator row
            if excluded or _NOT_REC.search(line):
                continue
            if table_col is not None and table_col < len(cells):
                m = _re.match(r"(?:blk|block)?\s*([0-9]{1,4}[A-Z]?)\b", cells[table_col], _re.I)
                if m:
                    out.append(m.group(1).upper())
                continue
        else:
            table_col = None
        if excluded or _NOT_REC.search(line):
            continue
        out += [m.upper() for m in _BLOCK_RE.findall(line)]
    return list(dict.fromkeys(out))


def _evidence(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    sql = ((state.get("sql_market_results") or {}).get("query_resale_blocks_by_criteria") or {}).get("blocks") or []
    for b in sql:
        evidence.setdefault(str(b.get("block", "")).upper(), {}).update(b)
    for b in state.get("spatial_block_candidates") or []:
        if isinstance(b, dict):
            e = evidence.setdefault(str(b.get("block", "")).upper(), {})
            e.update({k: v for k, v in b.items() if k != "distances"})
            e.setdefault("distances", {}).update(b.get("distances") or {})
    for b in state.get("candidate_blocks_intersection") or []:
        evidence.setdefault(str(b.get("block", "")).upper(), {}).update(b)
    return evidence


def recommended_blocks(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Blocks the variant recommended, read from its answer (all variants alike), each joined to the
    evidence that variant retrieved. A block in the answer but in no tool output is fabricated."""
    mentioned = recommended_block_numbers(state.get("draft_response") or "")
    evidence = _evidence(state)
    return [dict(evidence.get(m, {}), block=m, _fabricated=m not in evidence) for m in mentioned]


_BLOCK_TOKEN = _re.compile(r"^(?:blk|block)?[\s_-]*([0-9]{1,4}[A-Z]?)$", _re.I)


def _block_number(key: Any) -> Optional[str]:
    """'924 TAMPINES ST 91' -> '924'; 'Blk 9B' -> '9B'; 'BISHAN-01' / 'BLOCK_1' -> None (not a block number)."""
    text = str(key).strip()
    first = _re.split(r"\s+", text)[0]
    m = _BLOCK_TOKEN.match(first) or _BLOCK_TOKEN.match(text)
    return m.group(1).upper() if m else None


def map_blocks(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Blocks drawn as markers on the map. B and C: the supervisor's map payload. A (no supervisor):
    the candidate_blocks the model itself passed to the map tool in its last successful call.
    A marker whose id is not a block number in the retrieved evidence counts as fabricated."""
    keys = ((state.get("map_ui_payload") or {}).get("candidate_blocks")) or []
    # A (no supervisor) and B (no join in code, so its supervisor has no candidate list) draw the map through
    # the model's own map-tool call. In C the supervisor owns the map payload: an empty one means no map.
    if not keys and (state.get("variant") in ("A", "B") or "monolith" in (state.get("active_subagents") or [])):
        calls = [t for t in state.get("tool_call_log") or [] if t.get("tool") == MAP_TOOL and t.get("ok", True)]
        keys = (calls[-1].get("arguments") or {}).get("candidate_blocks") or [] if calls else []
    evidence = _evidence(state)
    out = []
    for k in keys:
        num = _block_number(k)
        fabricated = num is None or num not in evidence
        out.append(dict(evidence.get(num, {}) if num else {}, block=num or str(k), _fabricated=fabricated))
    return out


_RELAX_TEXT = _re.compile(r"\b(increas\w*|rais\w*|extend\w*|expand\w*|widen\w*|relax\w*|loosen\w*|lower\w*|reduc\w*|"
                          r"stretch\w*|consider\w*|adjust\w*)\b[^.\n]{0,60}\b(budget|price|radius|distance|metres|meters|"
                          r"lease|walk\w*|town|flat type)\b", _re.I)


def relaxation_offered(state: Dict[str, Any]) -> bool:
    """A concrete trade-off suggestion: C's structured relaxation, or the same advice in any answer's text."""
    if (state.get("relaxation") or {}).get("suggestions"):
        return True
    return bool(_RELAX_TEXT.search(state.get("draft_response") or ""))


def task_success(sub_use_case: str, csr: Dict[str, Any], fabricated: int, relaxed: bool) -> bool:
    """E1: at least one block recommended, every recommended block satisfies every constraint, none fabricated.
    E2 (no block can satisfy everything): nothing presented as a match, and a concrete trade-off offered."""
    if sub_use_case == "E1":
        return csr["recommended"] > 0 and csr["satisfying"] == csr["recommended"] and fabricated == 0
    if sub_use_case == "E2":
        return csr["recommended"] == 0 and relaxed
    return False


# ---------------------------------------------------------------------------
# Ground-truth scoring (grounded questions carry ground_truth_facts; see evaluation/ground_category_e.py)
# ---------------------------------------------------------------------------
_NUM_RE = _re.compile(r"(?<![\w.])\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s*k\b)?", _re.I)
_ASK_CUE = _re.compile(r"\?|\b(please|could you|can you|let (?:me|us) know|share|provide|tell (?:me|us)|confirm)\b", _re.I)
_BUDGET_CLAIM = _re.compile(r"\b(?:maximum|max\.?)\s+(?:purchase\s+price|budget|price|loan(?:\s+amount)?)\b[^.\n]{0,40}?\$\s?\d", _re.I)
# A missing calculator input counts as asked for when the answer names it in plain words.
_MISSING_WORDS = {
    "cash_savings": ("cash", "saving"), "cpf_oa_balance": ("cpf",), "monthly_income": ("income",),
    "monthly_debt_obligations": ("debt",), "remaining_lease_years": ("lease",),
    "youngest_buyer_age": ("age",), "oldest_buyer_age": ("age",), "average_applicant_age": ("age",),
    "youngest_core_member_age": ("age",), "loan_eligibility_confirmed": ("loan",), "eligibility_confirmed": ("eligib",),
    "continuous_employment_12_months": ("employ",), "total_grants": ("grant",),
}


def numbers_in(text: str) -> List[float]:
    out = []
    for whole, frac, k in _NUM_RE.findall(text or ""):
        v = float(whole.replace(",", "") + (frac or ""))
        out.append(v * 1000 if k else v)
    return out


def mentions_amount(text: str, value: Optional[float], tol: float = 1.0) -> bool:
    return value is not None and any(abs(n - float(value)) <= tol for n in numbers_in(text))


def asked_for_missing(draft: str, missing: List[str]) -> bool:
    """Every missing fact is named, and a sentence that names one actually asks for it. Boilerplate such as "please
    confirm with HDB" elsewhere in the answer does not count as asking."""
    # boilerplate that is not asking the user anything ("please confirm with HDB", "an HFE letter")
    draft = _re.sub(r"(?:please\s+)?confirm (?:it\s+)?with (?:hdb|the bank)|hdb flat eligibility \(hfe\) letter|hfe letter", " ",
                    draft or "", flags=_re.I)
    low = draft.lower()
    words = {m: _MISSING_WORDS.get(m, (m.replace("_", " "),)) for m in missing}
    named = all(any(w in low for w in ws) for ws in words.values())
    sentences = _re.split(r"(?<=[.!?])\s+|\n+", draft or "")
    asks = any(_ASK_CUE.search(s) and any(w in s.lower() for ws in words.values() for w in ws) for s in sentences)
    return named and asks


def ground_truth_scores(item: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compare an answer with the question's stored ground truth. None for questions without ground truth."""
    truth = item.get("ground_truth_facts")
    if not truth or "matching_block_keys" not in truth:
        return None
    draft = state.get("draft_response") or ""
    rec = set(recommended_block_numbers(draft))
    true = {str(k).split()[0].upper() for k in truth["matching_block_keys"]}
    tp = len(rec & true)
    out: Dict[str, Any] = {
        "blocks_exact": rec == true,
        "block_precision": tp / len(rec) if rec else (1.0 if not true else 0.0),
        "block_recall": tp / len(true) if true else (1.0 if not rec else 0.0),
    }
    sub = item.get("sub_use_case")
    fin = truth.get("financial") or {}
    if sub == "E1" and fin:
        if item.get("omitted_financial_facts"):
            out["asked_for_missing"] = asked_for_missing(draft, truth.get("missing_financial_inputs") or [])
            out["invented_budget"] = bool(_BUDGET_CLAIM.search(draft))
            out["success"] = out["blocks_exact"] and out["asked_for_missing"] and not out["invented_budget"]
        else:
            out["grants_correct"] = mentions_amount(draft, fin["calculate_cpf_housing_grants"].get("total_grants"))
            out["max_price_correct"] = mentions_amount(draft, fin["calculate_maximum_affordability"].get("max_purchase_price"))
            out["success"] = out["blocks_exact"] and out["grants_correct"] and out["max_price_correct"]
    elif sub == "E1":
        out["success"] = out["blocks_exact"]
    elif sub == "E2":
        best = truth.get("best_single_change") or {}
        out["best_change_mentioned"] = mentions_amount(draft, best.get("relaxed"), tol=0.5)
        out["success"] = not rec and out["best_change_mentioned"]
    return out


COARSE_OF = {"market_sql": "data", "spatial": "data", "financial": "data", "policy_rag": "policy"}


def routing_accuracy(expected_agents: List[str], active_subagents: List[str], variant: str) -> Dict[str, Any]:
    if variant == "A":
        return {"applicable": False}
    exp = set(expected_agents) if variant == "C" else {COARSE_OF[a] for a in expected_agents}
    act = set(active_subagents)
    return {"applicable": True, "exact": exp == act, "expected": sorted(exp), "actual": sorted(act)}


SEARCH_FIELDS = ("towns", "flat_types", "max_budget_sgd", "min_remaining_lease_years", "amenity_radii_meters")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(0.5, 0.005 * abs(float(b)))
    if isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    if isinstance(a, dict) and isinstance(b, dict):
        a = {k: v for k, v in a.items() if v}          # a 0 m distance is no constraint
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in b)
    return a == b


def model_reading_accuracy(item: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How well the MODEL alone read the question, before code filled gaps or overrode money facts
    (state["model_constraints"]). The search fields and the stated money facts are compared with the question's own
    stated values; a money fact the model gave that the question does not state is counted as invented. This is the
    measure that shows whether the model, not the code patterns, copes with a phrasing. None without a model."""
    reading = state.get("model_constraints")
    stated = item.get("stated_constraints")
    if not stated:
        return None
    if not reading:
        # the model was asked but produced no usable reading: it scores zero here, rather than the row dropping out
        # and leaving only the questions the model managed to read
        if any(str(e).startswith("extract_constraints[") for e in state.get("errors") or []):
            facts = sum(len(v or {}) for v in (item.get("financial_inputs") or {}).values())
            return {"search_correct": 0, "search_total": len(SEARCH_FIELDS), "facts_correct": 0, "facts_total": facts,
                    "facts_wrong": 0, "facts_invented": 0, "reading_failed": True}
        return None
    search_ok = sum(1 for f in SEARCH_FIELDS if _same(reading.get(f) or ([] if isinstance(stated.get(f), list) else
                                                      {} if isinstance(stated.get(f), dict) else None), stated.get(f)))
    facts_total = facts_ok = wrong = invented = 0
    want = item.get("financial_inputs") or {}
    got = reading.get("financial_inputs") or {}
    for sec in ("grants", "affordability"):
        w, g = want.get(sec) or {}, got.get(sec) or {}
        for k, v in w.items():
            facts_total += 1
            if k in g and g[k] is not None:
                facts_ok += _same(g[k], v)
                wrong += not _same(g[k], v)
        # only calculator facts count as invented (flat type, loan type and the like are not facts about the buyer)
        from agents.supervisor import FACT_FIELDS
        invented += sum(1 for k, v in g.items() if v is not None and k not in w and k in FACT_FIELDS)
    return {"search_correct": search_ok, "search_total": len(SEARCH_FIELDS), "facts_correct": facts_ok,
            "facts_total": facts_total, "facts_wrong": wrong, "facts_invented": invented}


def score_item(item: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """One row of the Experiment 1 results table."""
    variant = state.get("variant", "?")
    csr = constraint_satisfaction_rate(state.get("user_constraints", {}), recommended_blocks(state))
    fabricated = sum(1 for b in recommended_blocks(state) if b.get("_fabricated"))
    relaxed = relaxation_offered(state)
    row = {
        "id": item["id"], "variant": variant,
        "tool_selection": tool_selection_prf(item.get("expected_mcp_calls", []), state.get("tool_call_log", [])),
        # the same score over only the calls the model chose (ToolCallRecord.source); left out for logs recorded
        # before the source tag existed, which cannot say who chose a call
        **({"tool_selection_model_chosen": tool_selection_prf(
            item.get("expected_mcp_calls", []), [r for r in state.get("tool_call_log", []) if r.get("source") == "model"])}
           if any("source" in r for r in state.get("tool_call_log", [])) else {}),
        "extraction_model_only": model_reading_accuracy(item, state),
        "tool_calls_by_source": {src: sum(1 for r in state.get("tool_call_log", []) if (r.get("source") or "unknown") == src)
                                 for src in ("model", "code", "unknown")},
        "parameter_em": parameter_exact_match(item.get("expected_mcp_calls", []), state.get("tool_call_log", [])),
        "csr": csr,
        "fabricated_blocks": fabricated,
        "map_fabricated": sum(1 for b in map_blocks(state) if b.get("_fabricated")),
        "relaxation_offered": relaxed,
        "task_success_structural": task_success(item.get("sub_use_case", ""), csr, fabricated, relaxed),
        "ground_truth": ground_truth_scores(item, state),
        "map": constraint_satisfaction_rate(state.get("user_constraints", {}), map_blocks(state)),
        "routing": routing_accuracy(item.get("expected_agents", []), state.get("active_subagents", []), variant),
        "relaxation_attempted": bool((state.get("relaxation") or {}).get("suggestions")),
        "relaxation_unlocks": bool((state.get("relaxation") or {}).get("best_single_change")),
        "n_tool_calls": len(state.get("tool_call_log", [])),
        "n_errors": len(state.get("errors", [])),
    }
    gt = row["ground_truth"]
    # With ground truth, success means the answer is right; without it (stand-in questions), the structural rule.
    row["task_success"] = gt["success"] if gt else row["task_success_structural"]
    return row


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    out: Dict[str, Any] = {}
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v]
        out[v] = {
            "n": len(rs),
            "tool_selection_f1": mean(r["tool_selection"]["f1"] for r in rs),
            "parameter_em_rate": mean(r["parameter_em"]["rate"] for r in rs),
            "csr": mean(r["csr"]["rate"] for r in rs),
            "routing_exact_rate": mean((1.0 if r["routing"].get("exact") else 0.0) for r in rs if r["routing"].get("applicable")),
            "mean_tool_calls": mean(r["n_tool_calls"] for r in rs),
            "fabricated_blocks_total": sum(r.get("fabricated_blocks", 0) for r in rs),
            "map_precision": mean((r.get("map") or {}).get("rate") for r in rs),
            "task_success_rate": mean((1.0 if r.get("task_success") else 0.0) for r in rs),
            "blocks_exact_rate": mean((1.0 if (r.get("ground_truth") or {}).get("blocks_exact") else 0.0)
                                      for r in rs if r.get("ground_truth")),
        }
    return out
