"""Auditor catch rate by synthetic fault injection (proposal section 6.3, "Auditor Catch Rate", target above 90%).

    python -m evaluation.auditor_fault_injection                  # rules only: no model, no quota
    python -m evaluation.auditor_fault_injection --llm --limit 20 # also the LLM judge on figure-free false claims
    python -m evaluation.auditor_fault_injection --out evaluation/results/auditor_catch_rate.json

Part 1 (figures). For every non-adversarial benchmark question the real Variant C answer is produced without a model
(deterministic). Each figure kind found in that correct answer (price, distance, lease, block number) is corrupted
once, and the auditor's real `runtime_auditor_node` (the corrupted text placed in `model_opening`, its contract and rule layer) must reject it. Catch rate = rejected /
corrupted. The same run reports the false-rejection rate on the untouched answers: a checker that rejects everything
would score 100% on the first number, so the two are read together.

What is judged here is the whole code-written answer, which is what the proposal's metric describes. In the live graph
the auditor only sees the model's own sentences (`model_opening`, docs/INTERFACES.md section 3), which carry no figures
by construction, so this measures the checking capability, not how often the live app catches an error.

Part 2 (--llm). False statements without figures ("All of these blocks are in Bedok", "exempt from the MOP") are put in
the model's slot of each answer. The rule layer runs first; the LLM judge sees only what the rules pass. True sentences
are tried too, so wrongly rejected good text shows up. Uses the active LLM profile (one call per sentence).

Seeded (SEED = 42). Results go to evaluation/results/ (ignored by git).
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import guardrails as g  # noqa: E402

SEED = 42
EVAL = Path(__file__).resolve().parent
KINDS = ("price", "distance", "lease", "block")
# gross: obviously wrong figures; subtle: the kind of slip a model makes (a few percent, a few metres, a few years, the next
# block number). Each list is tried in turn until the new value is not in the evidence.
SEVERITY = {
    "gross": {"price": (1.25, 1.37, 0.71, 1.53), "distance": (173, 311, 457), "lease": (-13, 14, -21), "block": None},
    "subtle": {"price": (1.03, 0.96, 1.05, 0.93), "distance": (25, 41, 63), "lease": (-3, 4, -5), "block": "next"},
}

FALSE_CLAIMS = [
    "All of these blocks are in {town}.",
    "These flats are exempt from the Minimum Occupation Period, so you can sell right away.",
    "Every one of these blocks has a swimming pool next to it.",
    "None of these blocks meets your lease requirement.",
    "These blocks are all cheaper than any other flat in Singapore.",
]
TRUE_CLAIM = "These blocks all meet the limits you set."
OTHER_TOWNS = ["BEDOK", "WOODLANDS", "QUEENSTOWN", "PUNGGOL", "BUKIT MERAH"]


def load_questions() -> List[Dict[str, Any]]:
    items = json.loads((EVAL / "benchmark_hdb_90.json").read_text(encoding="utf-8"))
    return [i for i in items if i.get("category") != "F"]


def through_node(state: Dict[str, Any], text: str) -> Tuple[bool, Dict[str, Any]]:
    """(passed, node output) for `text` placed in the model's slot of `state`, judged by the real runtime_auditor_node: the
    contract (only model_opening is judged), the rule layer, and the LLM judge unless HDB_AUDITOR_LLM=off."""
    out = g.runtime_auditor_node({**state, "model_opening": text, "auditor_verdict": {}})
    return out["auditor_verdict"]["passed"], out


def answer_state(query: str) -> Dict[str, Any]:
    from agents.arch_3_specialist_5agent import run
    return run(query)


# ---------------------------------------------------------------------------
# corruption
# ---------------------------------------------------------------------------
def _pool(evidence: str) -> List[float]:
    return sorted(g._evidence_numbers(evidence))


def _absent(value: float, pool: List[float]) -> bool:
    return all(abs(value - p) > 1.5 for p in pool)


def _spans(text: str) -> Dict[str, List[Tuple[int, int, str]]]:
    out: Dict[str, List[Tuple[int, int, str]]] = {k: [] for k in KINDS}
    for m in g._MONEY.finditer(text):
        out["price"].append((m.start(), m.end(), m.group(0)))
    for m in g._METRES.finditer(text):
        out["distance"].append((m.start(), m.end(), m.group(0)))
    for m in g._YEARS.finditer(text):
        out["lease"].append((m.start(), m.end(), m.group(0)))
    for m in g._BLOCK.finditer(text):
        out["block"].append((m.start(), m.end(), m.group(0)))
    return out


def corrupt(text: str, kind: str, span: Tuple[int, int, str], pool: List[float], severity: str = "gross") -> Optional[str]:
    """`text` with one figure of `kind` changed to a value that is not in the evidence, or None if none can be found."""
    start, end, original = span
    end -= len(original) - len(original.rstrip())     # keep the whitespace that followed the figure
    original = original.rstrip()
    if kind == "price":
        m = g._MONEY.search(original)
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        value = g._to_float(num) * g._scale(unit)
        for factor in SEVERITY[severity]["price"]:
            new = round(value * factor, -3)
            if _absent(new, pool):
                return text[:start] + f"${int(new):,}" + text[end:]
    elif kind == "distance":
        value = g._to_float(g._METRES.search(original).group(1))
        for delta in SEVERITY[severity]["distance"]:
            new = value + delta if "k" not in original.lower() else (value * 1000 + delta)
            if _absent(new, pool):
                return text[:start] + f"{int(new)}m" + text[end:]
    elif kind == "lease":
        value = g._to_float(g._YEARS.search(original).group(1))
        for delta in SEVERITY[severity]["lease"]:
            new = value + delta
            if new > 0 and _absent(new, pool):
                return text[:start] + f"{int(new)} years" + text[end:]
    elif kind == "block":
        if SEVERITY[severity]["block"] == "next":        # a neighbouring number that the evidence does not hold
            ident = g._BLOCK.search(original).group(1)
            digits = re.match(r"\d+", ident).group(0)
            for step in (1, 2, 3, -1, 4):
                cand = str(int(digits) + step)
                if cand not in re.findall(r"(?<![\w.])\d+(?![\w.])", " ".join(map(str, pool))) and _absent(float(cand), pool):
                    return text[:start] + f"Blk {cand}" + text[end:]
            return None
        return text[:start] + "Blk 999" + text[end:]
    return None


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def measure_figures(items: List[Dict[str, Any]], states: Dict[str, Dict[str, Any]], severity: str = "gross") -> Dict[str, Any]:
    rng = random.Random(SEED)
    per_kind = {k: {"caught": 0, "total": 0, "missed": []} for k in KINDS}
    per_category: Dict[str, Dict[str, int]] = {}
    clean_total = clean_rejected = 0
    rejected_ids: List[str] = []
    for it in items:
        st = states[it["id"]]
        draft = st.get("draft_response") or ""
        evidence = g.evidence_text(st)
        spans = _spans(draft)
        if not any(spans.values()):
            continue
        clean_total += 1
        if not through_node(st, draft)[0]:
            clean_rejected += 1
            rejected_ids.append(it["id"])
        pool = _pool(evidence + " " + it["query"])
        cat = per_category.setdefault(it["category"], {"caught": 0, "total": 0})
        for kind in KINDS:
            if not spans[kind]:
                continue
            span = rng.choice(spans[kind])
            bad = corrupt(draft, kind, span, pool, severity)
            if bad is None:
                continue
            caught = not through_node(st, bad)[0]
            per_kind[kind]["total"] += 1
            per_kind[kind]["caught"] += caught
            cat["total"] += 1
            cat["caught"] += caught
            if not caught:
                per_kind[kind]["missed"].append({"id": it["id"], "original": span[2]})
    total = sum(v["total"] for v in per_kind.values())
    caught = sum(v["caught"] for v in per_kind.values())
    return {"severity": severity, "faults": total, "caught": caught, "catch_rate": round(caught / total, 4) if total else None,
            "per_kind": per_kind, "per_category": per_category,
            "clean_answers_with_figures": clean_total, "clean_rejected": clean_rejected,
            "false_rejection_rate": round(clean_rejected / clean_total, 4) if clean_total else None,
            "falsely_rejected_ids": rejected_ids}


def measure_misattribution(items: List[Dict[str, Any]], states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A real figure put where another figure belongs: one block's price or lease written for a different block. Every
    value is in the evidence, so a value-only check cannot see it; this reports how many it misses (expected: nearly all)."""
    total = caught = 0
    for it in items:
        st = states[it["id"]]
        draft = st.get("draft_response") or ""
        evidence = g.evidence_text(st)
        for kind in ("price", "lease", "distance"):
            spans = _spans(draft)[kind]
            texts = [sp[2].strip() for sp in spans]
            distinct = list(dict.fromkeys(texts))
            if len(distinct) < 2:
                continue
            first = next(sp for sp in spans if sp[2].strip() == distinct[0])
            bad = draft[:first[0]] + distinct[1] + draft[first[0] + len(first[2].rstrip()):]
            total += 1
            caught += not through_node(st, bad)[0]
    return {"faults": total, "caught": caught, "catch_rate": round(caught / total, 4) if total else None}


def measure_judge(items: List[Dict[str, Any]], states: Dict[str, Dict[str, Any]], limit: int) -> Dict[str, Any]:
    rng = random.Random(SEED + 1)
    rows: List[Dict[str, Any]] = []
    usable = [i for i in items if states[i["id"]].get("candidate_blocks_intersection")][:limit]
    for it in usable:
        st = states[it["id"]]
        data, user = g._judge_data(st), it["query"]
        towns_in_answer = (st.get("draft_response") or "").upper()
        town = next(t for t in OTHER_TOWNS if t not in towns_in_answer)
        claim = rng.choice(FALSE_CLAIMS).format(town=town.title())
        for label, text in (("false", claim), ("true", TRUE_CLAIM)):
            passed, out = through_node(st, text)
            if passed and out.get("errors"):      # the judge could not answer (quota, network): neither a catch nor a miss
                continue
            critique = out["auditor_verdict"]["critique"]
            layer = "judge" if critique.startswith("the data does not support") else "rules"
            rows.append({"id": it["id"], "label": label, "text": text, "layer": layer, "passed": passed})
    false = [r for r in rows if r["label"] == "false"]
    true = [r for r in rows if r["label"] == "true"]
    return {"false_claims": len(false), "false_caught": sum(not r["passed"] for r in false),
            "true_claims": len(true), "true_wrongly_rejected": sum(not r["passed"] for r in true),
            "caught_by_rules": sum(r["layer"] == "rules" and not r["passed"] for r in false),
            "caught_by_judge": sum(r["layer"] == "judge" and not r["passed"] for r in false),
            "misses": [r for r in false if r["passed"]], "wrong_rejections": [r for r in true if not r["passed"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llm", action="store_true", help="also test the LLM judge on figure-free false claims")
    ap.add_argument("--limit", type=int, default=20, help="questions used for the judge part (2 calls each)")
    ap.add_argument("--out", help="write the results as JSON")
    args = ap.parse_args(argv)

    os.environ.setdefault("HDB_ADDONS", "off")   # the core question sets run without the add-on packs (as the other runners)
    from agents.llm import active_profile
    configured = active_profile()["name"]       # the profile .env / the shell selects, read before it is overridden below
    os.environ["LLM_PROFILE"] = "none"          # the answers themselves are deterministic
    items = load_questions()
    states = {it["id"]: answer_state(it["query"]) for it in items}
    os.environ["LLM_PROFILE"] = configured      # back to the real profile for the judge part

    os.environ["HDB_AUDITOR_LLM"] = "off"         # the figure parts measure the node's rule layer; the judge part turns it back on
    result: Dict[str, Any] = {"seed": SEED, "questions": len(items), "figures": {}}
    for severity in SEVERITY:
        f = result["figures"][severity] = measure_figures(items, states, severity)
        print(f"{severity:6s} figure faults: {f['caught']}/{f['faults']} caught ({f['catch_rate']:.1%}); target above 90%")
        print("       " + "  ".join(f"{k} {v['caught']}/{v['total']}" for k, v in f["per_kind"].items()))
    f = result["figures"]["gross"]
    print(f"Clean answers wrongly rejected: {f['clean_rejected']}/{f['clean_answers_with_figures']} "
          f"({f['false_rejection_rate']:.1%}) {f['falsely_rejected_ids'][:6]}")
    result["misattributed"] = m = measure_misattribution(items, states)
    print(f"Misattributed real figures (a value-only check cannot see these): {m['caught']}/{m['faults']} caught"
          + (f" ({m['catch_rate']:.1%})" if m["faults"] else ""))
    os.environ.pop("HDB_AUDITOR_LLM", None)
    if args.llm:
        if configured == "none":
            print("--llm needs a model profile in .env (LLM_PROFILE is none)")
        else:
            j = measure_judge(items, states, args.limit)
            result["judge"] = j
            print(f"Figure-free false claims: {j['false_caught']}/{j['false_claims']} caught "
                  f"(rules {j['caught_by_rules']}, judge {j['caught_by_judge']}); "
                  f"true sentences wrongly rejected: {j['true_wrongly_rejected']}/{j['true_claims']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
