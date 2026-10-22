"""
Analysis of the preliminary Category E run (evaluation/analyze_category_e.py). Owned by Member 4.

Reads a results JSONL written by run_category_e_prelim.py and writes a markdown report with:
  1. per-variant Experiment 1 table (Tool F1, Param EM, routing, CSR, fabricated blocks, E1/E2 behaviour,
     latency P50/P95, tokens, errors)
  2. E1 vs E2 breakdown
  3. tool-selection error taxonomy (expected tools missed, unexpected tools called)
  4. per-query matrix
  5. trace excerpts chosen automatically for the qualitative section

    .venv/bin/python evaluation/analyze_category_e.py evaluation/results/prelim_vllm_Qwen_Qwen3-32B.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ITEMS = {i["id"]: i for i in json.loads((PROJECT_ROOT / "evaluation" / "category_e_queries.json").read_text())}


def load(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    latest: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        latest[(r["variant"], r["id"])] = r
    return list(latest.values())


def mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(xs, q) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def f(x, d=2) -> str:
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def variant_table(rows: List[Dict[str, Any]]) -> str:
    out = ["| Variant | Runs | Task success | Tool F1 | Param EM | Routing exact | CSR | Fabricated blocks (answer / map) | E1 map synced | Map precision | E2 trade-off offered | Latency P50 (s) | Latency P95 (s) | Mean tokens / query | Errors |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v]
        ok = [r for r in rs if r.get("status") == "ok"]
        sc = [r["scores"] for r in ok]
        routing = [1.0 if s["routing"].get("exact") else 0.0 for s in sc if s["routing"].get("applicable")]
        e1 = [r for r in ok if r["sub_use_case"] == "E1"]
        e2 = [r for r in ok if r["sub_use_case"] == "E2"]
        out.append("| " + " | ".join([
            v, str(len(rs)),
            f"{sum(1 for s in sc if s.get('task_success'))}/{len(rs)}",
            f(mean(s["tool_selection"]["f1"] for s in sc)),
            f(mean(s["parameter_em"]["rate"] for s in sc)),
            f(mean(routing)) if routing else "n/a",
            f(mean(s["csr"]["rate"] for s in sc)),
            f"{sum(s.get('fabricated_blocks', 0) for s in sc)} / {sum(s.get('map_fabricated', 0) for s in sc)}",
            f"{sum(1 for r in e1 if (r['scores'].get('map') or {}).get('recommended', 0) > 0)}/{len(e1)}",
            f(mean((s.get('map') or {}).get('rate') for s in sc)),
            f"{sum(1 for r in e2 if r['scores'].get('relaxation_offered'))}/{len(e2)}",
            f(pct([r["latency_s"] for r in ok], 0.5), 1),
            f(pct([r["latency_s"] for r in ok], 0.95), 1),
            f(mean((r.get("tokens") or {}).get("total_tokens") for r in ok), 0).split(".")[0],
            str(sum(1 for r in rs if r.get("status") != "ok") + sum(len(r["state_excerpt"].get("errors") or []) for r in ok)),
        ]) + " |")
    return "\n".join(out)


def subcase_table(rows: List[Dict[str, Any]]) -> str:
    out = ["| Sub-use case | Variant | Task success | Tool F1 | Param EM | CSR | Mean latency (s) |", "|---|---|---|---|---|---|---|"]
    for sub in ("E1", "E2"):
        for v in sorted({r["variant"] for r in rows}):
            ok = [r for r in rows if r["variant"] == v and r["sub_use_case"] == sub and r.get("status") == "ok"]
            sc = [r["scores"] for r in ok]
            out.append(f"| {sub} | {v} | {sum(1 for s in sc if s.get('task_success'))}/{len(sc)} | {f(mean(s['tool_selection']['f1'] for s in sc))} | {f(mean(s['parameter_em']['rate'] for s in sc))} | "
                       f"{f(mean(s['csr']['rate'] for s in sc))} | {f(mean(r['latency_s'] for r in ok), 1)} |")
    return "\n".join(out)


def taxonomy(rows: List[Dict[str, Any]]) -> str:
    out = ["| Variant | Expected tools never called (count over queries) | Unexpected tools called |", "|---|---|---|"]
    for v in sorted({r["variant"] for r in rows}):
        missed, extra = Counter(), Counter()
        for r in rows:
            if r["variant"] != v or r.get("status") != "ok":
                continue
            ts = r["scores"]["tool_selection"]
            for t in set(ts["expected"]) - set(ts["predicted"]):
                missed[t] += 1
            for t in set(ts["predicted"]) - set(ts["expected"]):
                extra[t] += 1
        fmt = lambda c: ", ".join(f"{k} ×{n}" for k, n in c.most_common()) or "none"
        out.append(f"| {v} | {fmt(missed)} | {fmt(extra)} |")
    return "\n".join(out)


def per_query(rows: List[Dict[str, Any]]) -> str:
    vs = sorted({r["variant"] for r in rows})
    by = {(r["variant"], r["id"]): r for r in rows}
    head = "| Query | " + " | ".join(f"{v}: ok / F1 / CSR / s" for v in vs) + " |"
    out = [head, "|---|" + "---|" * len(vs)]
    for qid in sorted(ITEMS):
        cells = []
        for v in vs:
            r = by.get((v, qid))
            if not r:
                cells.append("–")
            elif r.get("status") != "ok":
                cells.append("error")
            else:
                s = r["scores"]
                cells.append(f"{'✓' if s.get('task_success') else '✗'} / {s['tool_selection']['f1']:.2f} / {f(s['csr']['rate'])} / {r['latency_s']:.0f}")
        out.append(f"| {qid} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def trace(r: Dict[str, Any], draft_chars: int = 900) -> str:
    st = r["state_excerpt"]
    c = {k: v for k, v in (st.get("user_constraints") or {}).items() if v not in (None, [], {}, False)}
    lines = [f"**Variant {r['variant']} · {r['id']} · {r['latency_s']:.0f} s · {(r.get('tokens') or {}).get('total_tokens', '–')} tokens**", "",
             f"- Constraints extracted: `{json.dumps(c)}`",
             f"- Routing: {st.get('active_subagents')}",
             "- Tool calls: " + "; ".join(f"{t['agent']}→{t['tool']}" + ("" if t.get("ok", True) else " (failed)") for t in st.get("tool_call_log", [])),
             f"- Candidates after intersection: {st.get('n_candidates')}; fabricated blocks in the answer: {r['scores'].get('fabricated_blocks', 0)}; CSR: {f(r['scores']['csr']['rate'])}",
             f"- Errors: {st.get('errors') or 'none'}", "", "```text", (st.get("draft_response") or "")[:draft_chars], "```"]
    return "\n".join(lines)


def pick_traces(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {(r["variant"], r["id"]): r for r in rows if r.get("status") == "ok"}
    chosen: List[Dict[str, Any]] = []
    # 1. a query where A fabricates or violates constraints and C satisfies them
    for qid in sorted(ITEMS):
        a, c = by.get(("A", qid)), by.get(("C", qid))
        if a and c and (a["scores"].get("fabricated_blocks", 0) > 0 or (a["scores"]["csr"]["rate"] or 1) < 1) and c["scores"]["csr"]["rate"] == 1.0:
            chosen += [a, c]
            break
    # 2. a query where B never called a financial tool that was expected
    for qid in sorted(ITEMS):
        b = by.get(("B", qid))
        if b and "calculate_maximum_affordability" in b["scores"]["tool_selection"]["expected"] and \
                not any(t.startswith("calculate_") for t in b["scores"]["tool_selection"]["predicted"]):
            chosen.append(b)
            break
    # 3. an E2 query where C offered a relaxation and A did not
    for qid in sorted(q for q in ITEMS if q.startswith("E2")):
        a, c = by.get(("A", qid)), by.get(("C", qid))
        if c and c["scores"].get("relaxation_attempted") and a and not a["scores"].get("relaxation_attempted"):
            chosen += [c, a]
            break
    seen, uniq = set(), []
    for r in chosen:
        if (r["variant"], r["id"]) not in seen:
            seen.add((r["variant"], r["id"])); uniq.append(r)
    return uniq


def main() -> None:
    path = Path(sys.argv[1]).resolve()
    rows = load(path)
    model = next((r.get("model") for r in rows if r.get("model")), "?")
    profile = next((r.get("profile") for r in rows if r.get("profile")), "?")
    out_dir = PROJECT_ROOT / "evaluation" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + ".md")
    md = [f"# Experiment 1, preliminary: Category E ({len(ITEMS)} queries)", "",
          f"Model `{model}` via profile `{profile}`; {len(rows)} runs; source `{path.relative_to(PROJECT_ROOT)}`. "
          "SQL, financial and policy tools are deterministic mocks; spatial tools are real (Member 2's index). "
          "No runtime auditor yet (Member 5): Variant C runs with a pass-through auditor.", "",
          f"![Accuracy and latency by architecture]({path.stem}_accuracy_latency.svg)", "",
          "## Per variant", "", variant_table(rows), "",
          "CSR = share of the blocks an answer presents as matches that satisfy every stated constraint, judged on the evidence "
          "that variant retrieved (a block with no price, lease or distance evidence fails that constraint; a block in no tool "
          "output is fabricated). Map precision = the same check on the blocks drawn on the map. Tool F1 leaves out the map "
          "tool, which is scored by the map columns instead. Task success: an E1 answer recommends at least one block and every "
          "recommended block satisfies every constraint with none fabricated; an E2 answer presents nothing as a match and "
          "offers a concrete trade-off (C's structured relaxation, or the same advice in any variant's text).", "",
          "## E1 (full-stack synthesis) vs E2 (zero-result relaxation)", "", subcase_table(rows), "",
          "## Tool-selection errors", "", taxonomy(rows), "",
          "## Per query", "", per_query(rows), "",
          "## Trace excerpts for the qualitative section", ""]
    for r in pick_traces(rows):
        md += [trace(r), ""]
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
