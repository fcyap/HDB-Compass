"""Experiment 1 on the whole 90-question benchmark, and its report (proposal sections 6.2 and 6.3).

    # on the cluster (starts the GPU job, opens the tunnel, runs both passes, cancels the job, writes the report):
    bash cluster/run_architecture_eval.sh arch90

    # run directly (resumable; the model must already be reachable: LLM_PROFILE=vllm with the tunnel open, ollama, or
    # none for a no-model check)
    python -m evaluation.run_architecture_eval --run --tag arch90
    python -m evaluation.run_architecture_eval --run --tag arch90 --langsmith     # traces go to LangSmith project hdb-compass-arch90

    # report from the saved rows (offline; also reads results made by cluster/run_experiment1.sh if given its tag)
    python -m evaluation.run_architecture_eval --report --tag arch90

This file does not run questions itself. `evaluation/run_category_e_prelim.py` (Member 4) does that and owns the scoring
(`architecture_metrics.py`); `--run` calls it twice on `evaluation/benchmark_hdb_90.json`: Variants A, B and C with the
auditor on, then Variant C again with the runtime auditor switched off (HDB_AUDITOR=off), so the auditor's effect is
measured on its own. Both runs share one model, as section 6.2 requires. `--report` reads the saved rows and writes
`evaluation/results/architecture_eval_<tag>.md` and `.json`:

  1. task success per variant and category (Wilson 95% interval), tool F1, parameter exact match, CSR, latency, tokens;
  2. Category F safety by variant (A and B have no guardrail, by design);
  3. what the auditor did in Variant C: answers rejected, rewrites, loops that ran out, the latency those cost;
  4. Variant C with and without the auditor, question by question (exact McNemar test).

Hosted models (gemini, groq, openai) are refused by the runner it calls: the study runs on one open model we run ourselves.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import architecture_metrics as M  # noqa: E402

RESULTS = PROJECT_ROOT / "evaluation" / "results"
BENCHMARK = PROJECT_ROOT / "evaluation" / "benchmark_hdb_90.json"
RUNNER = "evaluation.run_category_e_prelim"


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def runner_command(tag: str, variants: str, dataset: Path = BENCHMARK) -> List[str]:
    return [sys.executable, "-m", RUNNER, "--dataset", str(dataset), "--variants", variants, "--tag", tag]


def plan(tag: str, variants: str = "A,B,C", ablation: bool = True) -> List[Tuple[List[str], Dict[str, str]]]:
    """(command, extra environment) for each pass: the variants with the auditor on, then C with it off."""
    steps: List[Tuple[List[str], Dict[str, str]]] = [(runner_command(tag, variants), {})]
    if ablation and "C" in [v.strip() for v in variants.split(",")]:
        steps.append((runner_command(f"{tag}-noauditor", "C"), {"HDB_AUDITOR": "off"}))
    return steps


def run(tag: str, variants: str, ablation: bool, langsmith: bool) -> int:
    base = dict(os.environ)
    if langsmith:
        base.update(LANGSMITH_TRACING="true", LANGSMITH_PROJECT=f"hdb-compass-{tag}")
    for cmd, extra in plan(tag, variants, ablation):
        label = extra.get("HDB_AUDITOR") and "auditor off" or "auditor on"
        print(f"\n=== {' '.join(cmd[2:])}   [{label}]", flush=True)
        rc = subprocess.call(cmd, cwd=PROJECT_ROOT, env={**base, **extra})
        if rc != 0:
            print(f"!! the runner stopped with exit code {rc}; run the same command again to resume.")
            return rc
    return 0


# ---------------------------------------------------------------------------
# reading results
# ---------------------------------------------------------------------------
def load_rows(tag: str) -> Dict[str, List[Dict[str, Any]]]:
    """{"on": rows with the auditor on (A, B, C), "off": Variant C rows with it off}; only rows that succeeded."""
    from evaluation.run_category_e_prelim import load_done
    out: Dict[str, List[Dict[str, Any]]] = {"on": [], "off": []}
    for path in sorted(RESULTS.glob(f"prelim_*_{tag}.jsonl")):
        out["on"] += [r for r in load_done(path).values() if r.get("status") == "ok"]
    for path in sorted(RESULTS.glob(f"prelim_*_{tag}-noauditor.jsonl")):
        out["off"] += [r for r in load_done(path).values() if r.get("status") == "ok"]
    return out


def _median(xs: List[float]) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _verdict(row: Dict[str, Any]) -> Dict[str, Any]:
    return (row.get("state") or {}).get("auditor_verdict") or (row.get("state_excerpt") or {}).get("auditor_verdict") or {}


def auditor_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the runtime auditor did across Variant C's answers (rows must be Variant C with the auditor on)."""
    c = [r for r in rows if r["variant"] == "C"]
    retried = [r for r in c if int(_verdict(r).get("retry_count", 0)) > 0]
    exhausted = [r for r in c if _verdict(r) and not _verdict(r).get("passed", True)]
    clean = [r for r in c if r not in retried]
    return {
        "answers": len(c),
        "rejected_at_least_once": len(retried),
        "rejected_rate": round(len(retried) / len(c), 4) if c else None,
        "mean_rewrites": round(sum(int(_verdict(r).get("retry_count", 0)) for r in c) / len(c), 3) if c else None,
        "ran_out_of_retries": len(exhausted),
        "rejected_but_correct": sum(1 for r in retried if r["scores"].get("task_success")),   # a rewrite of a right answer
        "latency_p50_s_rejected": _median([r["latency_s"] for r in retried]),
        "latency_p50_s_not_rejected": _median([r["latency_s"] for r in clean]),
    }


def paired_ablation(on: List[Dict[str, Any]], off: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Variant C with and without the auditor on the same questions: who is right where, and what it costs."""
    a = {r["id"]: r for r in on if r["variant"] == "C"}
    b = {r["id"]: r for r in off if r["variant"] == "C"}
    ids = sorted(set(a) & set(b))
    only_on = sum(1 for i in ids if a[i]["scores"].get("task_success") and not b[i]["scores"].get("task_success"))
    only_off = sum(1 for i in ids if b[i]["scores"].get("task_success") and not a[i]["scores"].get("task_success"))
    lat = [a[i]["latency_s"] - b[i]["latency_s"] for i in ids if a[i].get("latency_s") is not None and b[i].get("latency_s") is not None]
    tok = [(a[i].get("tokens") or {}).get("total_tokens", 0) - (b[i].get("tokens") or {}).get("total_tokens", 0) for i in ids]
    return {
        "paired_questions": len(ids),
        "auditor_on_success": sum(1 for i in ids if a[i]["scores"].get("task_success")),
        "auditor_off_success": sum(1 for i in ids if b[i]["scores"].get("task_success")),
        "right_only_with_auditor": only_on, "right_only_without_auditor": only_off,
        "mcnemar_exact_p": M.mcnemar_exact(only_on, only_off) if ids else None,
        "median_extra_latency_s": _median(lat), "median_extra_tokens": _median(tok),
    }


def safety_by_variant(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for v in sorted({r["variant"] for r in rows}):
        f = [r for r in rows if r["variant"] == v and str(r["id"]).startswith("F")]
        out[v] = {"questions": len(f), "handled": sum(1 for r in f if r["scores"].get("task_success")),
                  "blocked_by_guardrail": sum(1 for r in f if (_verdict(r).get("critique") == "blocked_by_input_guardrail"
                                                                or any(str(e).startswith("input_guardrail:") for e in (r.get("state") or {}).get("errors") or [])))}
    return out


def build_report(tag: str) -> Dict[str, Any]:
    from evaluation.run_category_e_prelim import summarise
    rows = load_rows(tag)
    return {"tag": tag, "rows_auditor_on": len(rows["on"]), "rows_auditor_off": len(rows["off"]),
            "experiment1": summarise(rows["on"]) if rows["on"] else {},
            "safety_category_f": safety_by_variant(rows["on"]),
            "auditor": auditor_stats(rows["on"]),
            "ablation": paired_ablation(rows["on"], rows["off"]) if rows["off"] else None}


def _ci(a: Dict[str, Any]) -> str:
    lo, hi = (a.get("task_success_ci95") or (None, None))
    return f"{a['task_success']}/{a['n']}" + (f" ({lo:.0%}–{hi:.0%})" if lo is not None else "")


def to_markdown(rep: Dict[str, Any]) -> str:
    lines = [f"# Experiment 1: `{rep['tag']}`", "",
             f"Rows: {rep['rows_auditor_on']} with the auditor on, {rep['rows_auditor_off']} with it off.", ""]
    exp = rep["experiment1"]
    if exp:
        cats = sorted({c for a in exp.values() for c in a.get("by_category", {})})
        lines += ["## Task success by variant and category (Wilson 95% interval)", "",
                  "| Variant | All | " + " | ".join(cats) + " | Tool F1 | Param EM | CSR | P50 s | P95 s |",
                  "|---|---|" + "---|" * len(cats) + "---|---|---|---|---|"]
        for v, a in exp.items():
            cells = [_ci(a["by_category"][c]) if c in a["by_category"] else "-" for c in cats]
            f = lambda x: "-" if x is None else f"{x:.2f}"
            lines.append(f"| {v} | {_ci(a)} | " + " | ".join(cells) +
                         f" | {f(a.get('tool_selection_f1'))} | {f(a.get('parameter_em_rate'))} | {f(a.get('csr'))} | "
                         f"{f(a.get('latency_p50_s'))} | {f(a.get('latency_p95_s'))} |")
        lines.append("")
    saf = rep["safety_category_f"]
    if saf:
        lines += ["## Category F (adversarial and safety) by variant", "", "| Variant | Handled | Stopped by the input guardrail |", "|---|---|---|"]
        lines += [f"| {v} | {s['handled']}/{s['questions']} | {s['blocked_by_guardrail']} |" for v, s in saf.items()]
        lines.append("")
    a = rep["auditor"]
    if a["answers"]:
        lines += ["## What the runtime auditor did (Variant C, auditor on)", "",
                  f"- answers: {a['answers']}; rejected at least once: {a['rejected_at_least_once']} ({a['rejected_rate']:.1%}); "
                  f"mean rewrites per answer: {a['mean_rewrites']}",
                  f"- loops that ran out of retries (the model's sentences were then removed): {a['ran_out_of_retries']}",
                  f"- rejected although the final answer scored correct: {a['rejected_but_correct']}",
                  f"- median latency: {a['latency_p50_s_rejected']} s when rejected, {a['latency_p50_s_not_rejected']} s when not", ""]
    ab = rep["ablation"]
    if ab:
        lines += ["## Variant C with and without the auditor (same questions)", "",
                  f"- paired questions: {ab['paired_questions']}; correct with auditor: {ab['auditor_on_success']}, without: {ab['auditor_off_success']}",
                  f"- right only with the auditor: {ab['right_only_with_auditor']}; right only without it: {ab['right_only_without_auditor']} "
                  f"(exact McNemar p = {ab['mcnemar_exact_p']:.3f})",
                  f"- median extra latency with the auditor: {ab['median_extra_latency_s']} s; extra tokens: {ab['median_extra_tokens']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run the benchmark (resumable) through the Experiment 1 runner")
    mode.add_argument("--report", action="store_true", help="aggregate the saved rows into tables")
    ap.add_argument("--tag", default="arch90", help="suffix of the results files")
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--no-ablation", action="store_true", help="skip the Variant C run without the auditor")
    ap.add_argument("--langsmith", action="store_true", help="send traces to LangSmith project hdb-compass-<tag>")
    args = ap.parse_args(argv)
    if args.run:
        return run(args.tag, args.variants, not args.no_ablation, args.langsmith)
    rep = build_report(args.tag)
    RESULTS.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rep)
    (RESULTS / f"architecture_eval_{args.tag}.json").write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
    (RESULTS / f"architecture_eval_{args.tag}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nWrote evaluation/results/architecture_eval_{args.tag}.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
