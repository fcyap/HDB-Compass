"""
Analysis of a preliminary Experiment 1 run (evaluation/analyze_category_e.py). Owned by Member 4.

Reads a results JSONL written by run_category_e_prelim.py (Category E, Category C policy questions, or a file with
both) and writes a markdown report with:
  1. per-variant Experiment 1 table (Tool F1, Param EM, routing, CSR, fabricated blocks, E1/E2 behaviour,
     latency P50/P95, tokens, errors), and the same per category
  2. ground-truth results: Category E blocks and figures; Category C key clauses, with the clauses in Member 3's
     ground truth that are known to be wrong listed beside them
  3. a Wilson 95% interval on every task-success and fully-right rate, and a per-question paired table (A vs C,
     B vs C) with an exact McNemar test
  4. a cost proxy (tokens and latency; no paid API)
  5. sub-use-case breakdown, tool-selection error taxonomy, per-query matrix
  6. trace excerpts chosen automatically for the qualitative section

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

from evaluation import architecture_metrics as M  # noqa: E402

# the 15 grounded questions, their varied-phrasing rewrites ("E1-01~casual") and Member 3's Category C policy
# questions, so any of these runs can be analysed
ITEMS = {i["id"]: i for name in ("category_e_queries.json", "category_e_varied.json", "category_c_queries.json")
         if (PROJECT_ROOT / "evaluation" / name).exists()
         for i in json.loads((PROJECT_ROOT / "evaluation" / name).read_text())}
CATEGORY_NAMES = {"E": "Category E (multi-agent search)", "C": "Category C (policy questions)"}


def cat(r: Dict[str, Any]) -> str:
    return M.category_of(r.get("sub_use_case"), r["id"])


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


def model_chosen_f1(ok: List[Dict[str, Any]]) -> str:
    """Mean F1 over model-chosen calls, zeros included; n/a only when there is nothing to measure (no-model run, or
    rows recorded before calls carried a source). Same rule as run_category_e_prelim.summarise."""
    vals = [r["scores"]["tool_selection_model_chosen"]["f1"] for r in ok if "tool_selection_model_chosen" in r["scores"]
            and (r.get("settings") or {}).get("profile", r.get("profile")) != "none"]
    return f(mean(vals)) if vals else "n/a"


def model_reading_table(rows: List[Dict[str, Any]]) -> str:
    """How well the model alone read each question, before code filled gaps or overrode money facts."""
    out = ["| Variant | Questions | Search fields right | Stated money facts right | Money facts wrong | Money facts invented |",
           "|---|---|---|---|---|---|"]
    any_rows = False
    for v in sorted({r["variant"] for r in rows}):
        reads = [r["scores"]["extraction_model_only"] for r in rows
                 if r["variant"] == v and r.get("status") == "ok" and (r.get("scores") or {}).get("extraction_model_only")]
        if not reads:
            continue
        any_rows = True
        st, sc = sum(x["search_total"] for x in reads), sum(x["search_correct"] for x in reads)
        ft, fc = sum(x["facts_total"] for x in reads), sum(x["facts_correct"] for x in reads)
        out.append(f"| {v} | {len(reads)} | {sc}/{st} ({sc / st:.0%}) | " + (f"{fc}/{ft} ({fc / ft:.0%})" if ft else "–") +
                   f" | {sum(x['facts_wrong'] for x in reads)} | {sum(x['facts_invented'] for x in reads)} |")
    return "\n".join(out) if any_rows else "No model reading recorded (no-model run, Variant A, or rows from before 30 Oct)."


def variant_table(rows: List[Dict[str, Any]]) -> str:
    out = ["| Variant | Runs | Task success (Wilson 95% CI) | Tool F1 | Tool F1 (model-chosen calls) | Param EM | Routing exact | CSR | Fabricated blocks (answer / map) | E1 map synced | Map precision | E2 trade-off offered | Latency P50 (s) | Latency P95 (s) | Mean tokens / query | Answers with a code fallback | Errors |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v]
        ok = [r for r in rs if r.get("status") == "ok"]
        sc = [r["scores"] for r in ok]
        routing = [1.0 if s["routing"].get("exact") else 0.0 for s in sc if s["routing"].get("applicable")]
        e1 = [r for r in ok if r["sub_use_case"] == "E1"]
        e2 = [r for r in ok if r["sub_use_case"] == "E2"]
        out.append("| " + " | ".join([
            v, str(len(rs)),
            M.rate_with_ci(sum(1 for s in sc if s.get("task_success")), len(rs)),
            f(mean(s["tool_selection"]["f1"] for s in sc)),
            model_chosen_f1(ok),
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
            f"{sum(1 for r in ok if r.get('fallbacks'))}/{len(ok)}" if any("fallbacks" in r for r in ok) else "n/a",
            str(sum(1 for r in rs if r.get("status") != "ok") + sum(len(r["state_excerpt"].get("errors") or []) for r in ok)),
        ]) + " |")
    return "\n".join(out)


def ground_truth_table(rows: List[Dict[str, Any]]) -> str:
    """Answers checked against each question's stored ground truth (grounded questions only)."""
    out = ["| Variant | Answers fully right (Wilson 95% CI) | Exact blocks | E1 grant and max price right | "
           "E1 asked for the missing fact | E1 invented a budget | E2 named the unlocking change | "
           "Follow-up completed right (Wilson 95% CI) |", "|---|---|---|---|---|---|---|---|"]
    for v in sorted({r["variant"] for r in rows}):
        gts = [(r, r["scores"].get("ground_truth")) for r in rows if r["variant"] == v and r.get("status") == "ok"]
        gts = [(r, g) for r, g in gts if g and "blocks_exact" in g]       # block questions; policy ones are below
        if not gts:
            continue
        def frac(pred, pool, ci=False):
            pool = [g for r, g in gts if pool(r, g)]
            k = sum(1 for g in pool if pred(g))
            return M.rate_with_ci(k, len(pool)) if ci else (f"{k}/{len(pool)}" if pool else "–")
        e1_full = lambda r, g: r["sub_use_case"] == "E1" and "grants_correct" in g
        e1_miss = lambda r, g: r["sub_use_case"] == "E1" and "asked_for_missing" in g
        fus = [r["follow_up"] for r, g in gts if r.get("follow_up")]
        out.append("| " + " | ".join([
            v, frac(lambda g: g["success"], lambda r, g: True, ci=True), frac(lambda g: g["blocks_exact"], lambda r, g: True),
            frac(lambda g: g["grants_correct"] and g["max_price_correct"], e1_full),
            frac(lambda g: g["asked_for_missing"], e1_miss), frac(lambda g: g["invented_budget"], e1_miss),
            frac(lambda g: g["best_change_mentioned"], lambda r, g: r["sub_use_case"] == "E2"),
            M.rate_with_ci(sum(1 for f in fus if (f.get("scores") or {}).get("success")), len(fus)),
        ]) + " |")
    return "\n".join(out) if len(out) > 2 else "_No ground truth in these questions._"


def category_table(rows: List[Dict[str, Any]]) -> str:
    """The headline numbers per benchmark category, so a file with both E and C rows is not read as one pool."""
    out = ["| Category | Variant | Runs | Task success (Wilson 95% CI) | Tool F1 | Param EM | Routing exact |",
           "|---|---|---|---|---|---|---|"]
    for c in sorted({cat(r) for r in rows}):
        for v in sorted({r["variant"] for r in rows}):
            rs = [r for r in rows if r["variant"] == v and cat(r) == c]
            if not rs:
                continue
            sc = [r["scores"] for r in rs if r.get("status") == "ok"]
            routing = [1.0 if s["routing"].get("exact") else 0.0 for s in sc if s["routing"].get("applicable")]
            out.append(f"| {c} | {v} | {len(rs)} | {M.rate_with_ci(sum(1 for s in sc if s.get('task_success')), len(rs))} | "
                       f"{f(mean(s['tool_selection']['f1'] for s in sc))} | {f(mean(s['parameter_em']['rate'] for s in sc))} | "
                       + (f"{f(mean(routing))} |" if routing else "n/a |"))
    return "\n".join(out)


def policy_table(rows: List[Dict[str, Any]]) -> str:
    """Category C answers against Member 3's key clauses."""
    out = ["| Variant | Questions | Fully right: every key clause (Wilson 95% CI) | Mean clause coverage | Tool F1 | "
           "Param EM | Routing exact | Passes that rest on a known-wrong clause |", "|---|---|---|---|---|---|---|---|"]
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v and cat(r) == "C"]
        gts = [(r["scores"], r["scores"]["ground_truth"]) for r in rs if r.get("status") == "ok"
               and ((r.get("scores") or {}).get("ground_truth") or {}).get("kind") == "policy"]
        if not gts:
            continue
        routing = [1.0 if s["routing"].get("exact") else 0.0 for s, _ in gts if s["routing"].get("applicable")]
        out.append("| " + " | ".join([
            v, str(len(rs)), M.rate_with_ci(sum(1 for _, g in gts if g["success"]), len(rs)),
            f(mean(g["clause_coverage"] for _, g in gts)),
            f(mean(s["tool_selection"]["f1"] for s, _ in gts)), f(mean(s["parameter_em"]["rate"] for s, _ in gts)),
            f(mean(routing)) if routing else "n/a",
            str(sum(1 for _, g in gts if g["success"] and g.get("known_wrong_clauses"))),
        ]) + " |")
    return "\n".join(out) if len(out) > 2 else "_No Category C rows._"


def known_wrong_caveat(rows: List[Dict[str, Any]]) -> str:
    """The key clauses in Member 3's ground truth that are known to be wrong, for the questions in this run."""
    ids = sorted({r["id"] for r in rows if cat(r) == "C"})
    lines = []
    for qid in ids:
        for w in M.known_wrong_clauses(ITEMS.get(qid) or {}):
            passed = sorted(r["variant"] for r in rows if r["id"] == qid and r.get("status") == "ok"
                            and ((r.get("scores") or {}).get("ground_truth") or {}).get("success"))
            lines.append(f"> - **{qid}**, clause `{json.dumps(w['clause'], ensure_ascii=False)}`: {w['note']} "
                         f"Variants that pass this question: {', '.join(passed) or 'none'}.")
    if not lines:
        return ""
    return "\n".join(["> **Caveat: known errors in the Category C ground truth.** Member 3's "
                      "`evaluation/category_c_queries.json` (not edited here) expects the clauses below, which are wrong "
                      "or out of date. A pass on these questions means the answer repeated the knowledge base, not that "
                      "it stated current HDB policy. Check the figures on hdb.gov.sg before quoting them; they were not "
                      "re-checked online for this report.", ">"] + lines)


def paired_table(rows: List[Dict[str, Any]]) -> str:
    """A and B each against C on the same questions. Only the questions where the two differ say anything about
    which is better; the exact McNemar test asks how likely a split that uneven is if neither were."""
    right = {(r["variant"], r["id"]): bool(r["scores"].get("task_success")) for r in rows if r.get("status") == "ok"}
    others = [o for o in ("A", "B") if any(v == o for v, _ in right)]
    if not any(v == "C" for v, _ in right) or not others:
        return "_Needs Variant C and at least one of A or B._"
    out = ["| Category | Pair | Questions run by both | Both right | Only C right | Only the other right | Both wrong | "
           "Exact McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for c in sorted({cat(r) for r in rows}):
        ids = sorted({r["id"] for r in rows if cat(r) == c})
        for o in others:
            both = [(right[(o, q)], right[("C", q)]) for q in ids if (o, q) in right and ("C", q) in right]
            if not both:
                continue
            only_c = sum(1 for x, y in both if y and not x)
            only_o = sum(1 for x, y in both if x and not y)
            p = M.mcnemar_exact(only_c, only_o)
            out.append(f"| {c} | {o} vs C | {len(both)} | {sum(1 for x, y in both if x and y)} | {only_c} | {only_o} | "
                       f"{sum(1 for x, y in both if not x and not y)} | {'<0.001' if p < 0.001 else f'{p:.3f}'} |")
    mark = lambda q, v: "–" if (v, q) not in right else ("✓" if right[(v, q)] else "✗")

    def versus(q, o):
        if (o, q) not in right or ("C", q) not in right:
            return "–"
        x, y = right[(o, q)], right[("C", q)]
        return "same" if x == y else ("C only" if y else f"{o} only")
    out += ["", "| Query | " + " | ".join(others + ["C"]) + " | " + " | ".join(f"{o} vs C" for o in others) + " |",
            "|---|" + "---|" * (2 * len(others) + 1)]
    for q in sorted({r["id"] for r in rows}):
        out.append(f"| {q} | " + " | ".join(mark(q, v) for v in others + ["C"]) + " | " +
                   " | ".join(versus(q, o) for o in others) + " |")
    return "\n".join(out)


def cost_table(rows: List[Dict[str, Any]]) -> str:
    from evaluation.run_category_e_prelim import COST_PROXY_NOTE, cost_proxy
    out = ["| Variant | Questions | Total tokens | Median tokens / question | Total latency (s) | Median latency / question (s) |",
           "|---|---|---|---|---|---|"]
    for v in sorted({r["variant"] for r in rows}):
        cp = cost_proxy([r for r in rows if r["variant"] == v and r.get("status") == "ok"])
        out.append(f"| {v} | {cp['questions']} | {f(cp['total_tokens'], 0)} | {f(cp['median_tokens'], 0)} | "
                   f"{f(cp['total_latency_s'], 1)} | {f(cp['median_latency_s'])} |")
    return "\n".join(out) + f"\n\nCost is a {COST_PROXY_NOTE}."


def subcase_table(rows: List[Dict[str, Any]]) -> str:
    out = ["| Sub-use case | Variant | Task success (Wilson 95% CI) | Tool F1 | Param EM | CSR | Mean latency (s) |",
           "|---|---|---|---|---|---|---|"]
    for sub in sorted({r["sub_use_case"] for r in rows if r.get("sub_use_case")}):
        for v in sorted({r["variant"] for r in rows}):
            ok = [r for r in rows if r["variant"] == v and r["sub_use_case"] == sub and r.get("status") == "ok"]
            sc = [r["scores"] for r in ok]
            out.append(f"| {sub} | {v} | {M.rate_with_ci(sum(1 for s in sc if s.get('task_success')), len(sc))} | "
                       f"{f(mean(s['tool_selection']['f1'] for s in sc))} | {f(mean(s['parameter_em']['rate'] for s in sc))} | "
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
    for qid in sorted({r['id'] for r in rows}):
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
    for qid in sorted({r['id'] for r in rows}):
        a, c = by.get(("A", qid)), by.get(("C", qid))
        if a and c and (a["scores"].get("fabricated_blocks", 0) > 0 or (a["scores"]["csr"]["rate"] or 1) < 1) and c["scores"]["csr"]["rate"] == 1.0:
            chosen += [a, c]
            break
    # 2. a query where B never called a financial tool that was expected
    for qid in sorted({r['id'] for r in rows}):
        b = by.get(("B", qid))
        if b and "calculate_maximum_affordability" in b["scores"]["tool_selection"]["expected"] and \
                not any(t.startswith("calculate_") for t in b["scores"]["tool_selection"]["predicted"]):
            chosen.append(b)
            break
    # 3. an E2 query where C offered a relaxation and A did not
    for qid in sorted(q for q in {r["id"] for r in rows} if q.startswith("E2")):
        a, c = by.get(("A", qid)), by.get(("C", qid))
        if c and c["scores"].get("relaxation_attempted") and a and not a["scores"].get("relaxation_attempted"):
            chosen += [c, a]
            break
    seen, uniq = set(), []
    for r in chosen:
        if (r["variant"], r["id"]) not in seen:
            seen.add((r["variant"], r["id"])); uniq.append(r)
    return uniq


def run_header(rows: List[Dict[str, Any]]) -> str:
    """What produced these rows, from the settings each row recorded at run time (run_category_e_prelim.run_settings).
    Older rows have none; then the current config is described and labelled as such."""
    settings = [r["settings"] for r in rows if r.get("settings")]
    missing = sum(1 for r in rows if not r.get("settings"))
    if not settings:
        from agents.supervisor import load_config
        wiring = load_config().get("tools", {}).get("use_mock", {})
        mocked = sorted(t for t, v in wiring.items() if v == "mock")
        return (f"(Rows recorded no run settings; current config:) mocked tools: {', '.join(mocked) or 'none'}.")
    commits = sorted({s.get("commit") or "?" for s in settings})
    first = settings[0]
    parts = [f"Code: {', '.join(commits)}" + (" (with uncommitted changes)" if any(s.get("uncommitted_changes") for s in settings) else "") + ".",
             f"Mocked tools: {', '.join(first.get('mocked_tools') or []) or 'none'}.",
             f"Code-written facts in C: {'on' if first.get('code_written_facts') else 'off'}; extraction reasoning pass: "
             f"{'on' if first.get('extraction_reasoning') else 'off'}.",
             "Runtime auditor: " + ("installed" if first.get("runtime_auditor_installed") else "not installed (pass-through)") +
             "; input guardrail: " + ("installed" if first.get("input_guardrail_installed") else "not installed (pass-through)") + "."]
    datasets = sorted({s.get("dataset") for s in settings if s.get("dataset")})
    if len(datasets) > 1:
        parts.append(f"Datasets: {', '.join(datasets)}.")
    # a file that holds two benchmark categories was run on two datasets; that alone is not a mixed setting
    same = lambda s: {k: v for k, v in s.items() if k not in ("uncommitted_changes", "dataset", "dataset_sha")}
    if len({json.dumps(same(s), sort_keys=True) for s in settings}) > 1:
        parts.append("**Warning: these rows come from more than one code version or setting.**")
    if any(len({s.get("dataset_sha") for s in settings if s.get("dataset") == d}) > 1 for d in datasets):
        parts.append("**Warning: these rows come from two versions of the same dataset file.**")
    if missing:
        parts.append(f"**Warning: {missing} of {len(rows)} rows recorded no settings (run before 30 Oct); they may come "
                     "from other code.**")
    return " ".join(parts)


def main() -> None:
    path = Path(sys.argv[1]).resolve()
    rows = load(path)
    model = next((r.get("model") for r in rows if r.get("model")), "?")
    profile = next((r.get("profile") for r in rows if r.get("profile")), "?")
    out_dir = PROJECT_ROOT / "evaluation" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + ".md")
    cats = sorted({cat(r) for r in rows})
    title = " and ".join(f"Category {c}" for c in cats)
    md = [f"# Experiment 1, preliminary: {title} ({len({r['id'] for r in rows})} queries)", "",
          f"Model `{model}` via profile `{profile}`; {len(rows)} runs; source `{path.relative_to(PROJECT_ROOT)}`. "
          + run_header(rows), "",
          f"![Accuracy and latency by architecture]({path.stem}_accuracy_latency.svg)", "",
          "Every task-success and fully-right rate carries a Wilson 95% interval: with 15 questions per variant, one "
          "question moves a rate by about 7 points, so two variants whose intervals overlap widely are not shown apart "
          "by this run. The paired table below compares variants on the same questions.", "",
          "## Per variant", "", variant_table(rows), "",
          "CSR = share of the blocks an answer presents as matches that satisfy every stated constraint, judged on the evidence "
          "that variant retrieved (a block with no price, lease or distance evidence fails that constraint; a block in no tool "
          "output is fabricated). Map precision = the same check on the blocks drawn on the map. Tool F1 leaves out the map "
          "tool, which is scored by the map columns instead. Tool F1 (model-chosen calls) counts only the calls the model chose "
          "itself, not those made by a deterministic plan, the supervisor or a guard. Answers with a code fallback: answers "
          "where the harness stood in for a model step (no valid tool call, writer failed, constraint reading failed). "
          "Task success: an E1 answer recommends at least one block and every "
          "recommended block satisfies every constraint with none fabricated; an E2 answer presents nothing as a match and "
          "offers a concrete trade-off (C's structured relaxation, or the same advice in any variant's text). A Category C "
          "policy answer succeeds when it contains every key clause of the question.", "",
          "## Per category", "", category_table(rows), ""]
    if len(cats) > 1:
        md += ["The per-variant table above pools the categories; the questions differ in kind, so compare variants "
               "within a category.", ""]
    if "E" in cats:
        md += ["## Category E checked against the ground truth", "", ground_truth_table(rows), "",
               "Fully right: E1 names exactly the true blocks and the calculators' grant total and maximum price (within "
               "$1), or, where a fact was left out, asks for it without inventing a budget; E2 presents no match and names "
               "the number of the single change that unlocks blocks. Follow-up: after the user supplies the missing fact, "
               "the second answer names exactly the true blocks within the new affordability cap and the correct grant "
               "total and maximum price.", "",
               "## The model's own reading", "", model_reading_table(rows), "",
               "Before code fills gaps or overrides money facts: how many of the five search fields (towns, flat types, "
               "budget, lease, distances) and of the stated money facts the model read correctly by itself. Code's "
               "patterns were tuned on the varied-phrasing set, so on that set this table, not the fully-right rate, "
               "shows how the model copes.", ""]
    if "C" in cats:
        md += ["## Category C: policy questions against Member 3's key clauses", "", policy_table(rows), "",
               "Fully right: every key clause of the question appears in the answer, matched by Member 3's own rule "
               "(`evaluation/run_rag_ablation.py`, `is_clause_in_passage`, the rule behind its answer fact coverage): "
               "markdown marks are removed and case ignored; a clause given as a list of words needs every word; a "
               "clause given as a sentence needs the whole sentence or at least 75% of its words longer than two "
               "letters. The match is on wording, so a correct answer in other words can miss a clause. Mean clause "
               "coverage is Member 3's answer fact coverage. Param EM compares the policy tool's free-text `query` "
               "with the hand-written search phrase in `expected_mcp_calls`, which no variant reproduces word for "
               "word; read Tool F1 and routing for whether the policy tool was chosen.", "",
               known_wrong_caveat(rows), ""]
    md += ["## Paired comparison on the same questions", "", paired_table(rows), "",
           "## Cost proxy", "", cost_table(rows), "",
           "## By sub-use case", "", subcase_table(rows), "",
           "E1 full-stack synthesis, E2 zero-result relaxation; C1 flat classification rules, C2 scheme and age "
           "eligibility, C3 wait-out periods and ethnic quotas.", "",
          "## Tool-selection errors", "", taxonomy(rows), "",
          "## Per query", "", per_query(rows), "",
          "## Trace excerpts for the qualitative section", ""]
    for r in pick_traces(rows):
        md += [trace(r), ""]
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
