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

_BLOCK_RE = _re.compile(r"\b(?:Blk|Block)\s*[#:]?\s*\**\s*([0-9]{1,4}[A-Z]?)\b", _re.I)
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


COARSE_OF = {"market_sql": "data", "spatial": "data", "financial": "data", "policy_rag": "policy"}


def routing_accuracy(expected_agents: List[str], active_subagents: List[str], variant: str) -> Dict[str, Any]:
    if variant == "A":
        return {"applicable": False}
    exp = set(expected_agents) if variant == "C" else {COARSE_OF[a] for a in expected_agents}
    act = set(active_subagents)
    return {"applicable": True, "exact": exp == act, "expected": sorted(exp), "actual": sorted(act)}


def score_item(item: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """One row of the Experiment 1 results table."""
    variant = state.get("variant", "?")
    csr = constraint_satisfaction_rate(state.get("user_constraints", {}), recommended_blocks(state))
    fabricated = sum(1 for b in recommended_blocks(state) if b.get("_fabricated"))
    relaxed = relaxation_offered(state)
    return {
        "id": item["id"], "variant": variant,
        "tool_selection": tool_selection_prf(item.get("expected_mcp_calls", []), state.get("tool_call_log", [])),
        "parameter_em": parameter_exact_match(item.get("expected_mcp_calls", []), state.get("tool_call_log", [])),
        "csr": csr,
        "fabricated_blocks": fabricated,
        "map_fabricated": sum(1 for b in map_blocks(state) if b.get("_fabricated")),
        "relaxation_offered": relaxed,
        "task_success": task_success(item.get("sub_use_case", ""), csr, fabricated, relaxed),
        "map": constraint_satisfaction_rate(state.get("user_constraints", {}), map_blocks(state)),
        "routing": routing_accuracy(item.get("expected_agents", []), state.get("active_subagents", []), variant),
        "relaxation_attempted": bool((state.get("relaxation") or {}).get("suggestions")),
        "relaxation_unlocks": bool((state.get("relaxation") or {}).get("best_single_change")),
        "n_tool_calls": len(state.get("tool_call_log", [])),
        "n_errors": len(state.get("errors", [])),
    }


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
        }
    return out
