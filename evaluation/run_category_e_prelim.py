"""
Preliminary Experiment 1 runner — Category E only (evaluation/run_category_e_prelim.py).
Owned by Member 4. Member 5's run_architecture_eval.py supersedes this for the
full 90-query benchmark; the per-item scoring is shared (architecture_metrics.py).

Runs Variants A/B/C over evaluation/category_e_queries.json on whatever LLM
profile is active (ollama laptop, vllm cluster, or none), checkpointing one
JSON line per (variant, query) so a dropped VPN, a Slurm wall-time limit or a
closed laptop lid loses at most one query. Re-running resumes.

    LLM_PROFILE=ollama .venv/bin/python evaluation/run_category_e_prelim.py            # all variants, all 15
    LLM_PROFILE=vllm   .venv/bin/python evaluation/run_category_e_prelim.py --variants C --ids E1-01,E2-01
    .venv/bin/python evaluation/run_category_e_prelim.py --summary-only                 # re-aggregate a results file

Output: evaluation/results/prelim_<profile>_<model>.jsonl  (+ .summary.json)
Each line: {run, variant, id, sub_use_case, profile, model, latency_s, scores, state_excerpt}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm import active_profile  # noqa: E402
from evaluation import architecture_metrics as M  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
QUERIES = PROJECT_ROOT / "evaluation" / "category_e_queries.json"
VARIED = PROJECT_ROOT / "evaluation" / "category_e_varied.json"   # the same questions in four user styles (vary_category_e.py)


def dataset_path(name: str) -> Path:
    if not name:
        return QUERIES
    if name == "varied":
        return VARIED
    return Path(name).resolve()


def load_variant(name: str):
    if name == "A":
        from agents import arch_1_monolith as mod
    elif name == "B":
        from agents import arch_2_coarse as mod
    elif name == "C":
        from agents import arch_3_specialist_5agent as mod
    else:
        raise ValueError(name)
    return mod


def state_excerpt(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enough to reconstruct a trace walkthrough without the full tool payloads."""
    blocks = state.get("candidate_blocks_intersection") or []
    return {
        "user_constraints": state.get("user_constraints"),
        "active_subagents": state.get("active_subagents"),
        "tool_call_log": [{k: r.get(k) for k in ("agent", "tool", "arguments", "ok", "error", "latency_ms")} for r in state.get("tool_call_log", [])],
        "n_candidates": len(blocks),
        "candidates": [{"block": b.get("block"), "street": b.get("street"), "town": b.get("town"),
                        "median_resale_price": b.get("median_resale_price"),
                        "median_remaining_lease_years": b.get("median_remaining_lease_years"),
                        "distances": b.get("distances")} for b in blocks[:10]],
        "relaxation": state.get("relaxation"),
        "auditor_verdict": state.get("auditor_verdict"),
        "errors": state.get("errors"),
        "map_markers": len(((state.get("map_ui_payload") or {}).get("markers") or {}).get("features") or []),
        "draft_response": (state.get("draft_response") or "")[:4000],
    }


RESCORE_KEYS = ["variant", "messages", "clarification", "user_constraints", "active_subagents", "tool_call_log", "sql_market_results",
                "spatial_block_candidates", "financial_breakdown", "rag_policy_passages",
                "candidate_blocks_intersection", "relaxation", "map_ui_payload", "draft_response",
                "auditor_verdict", "errors", "model_constraints"]


def rescore_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of the final state the metrics read, so scores can be recomputed without the model."""
    out = {k: state.get(k) for k in RESCORE_KEYS}
    if isinstance(out.get("spatial_block_candidates"), list):
        out["spatial_block_candidates"] = out["spatial_block_candidates"][:1000]
    return out


def results_path(profile: Dict[str, Any], tag: str) -> Path:
    model = (profile.get("model") or "none").replace("/", "_").replace(":", "-")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"prelim_{profile['name']}_{model}{('_' + tag) if tag else ''}.jsonl"


def load_done(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    done: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["variant"], row["id"])] = row
    return done


def _token_totals(usage: Dict[str, Any]) -> Dict[str, int]:
    """Sum LangChain usage_metadata across models: input/output/total tokens for the whole query."""
    tot = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for u in (usage or {}).values():
        for k in tot:
            tot[k] += int((u or {}).get(k, 0) or 0)
    return tot


# A model call that never got an answer: lost connection, timeout, server error or rate limit (the exception's type
# name ends up in the node's error message). The nodes then fall back to their no-model plans, which score like a
# perfect run, so such a row is not a model result and is rerun on resume.
UNREACHABLE = ("ConnectionError", "Timeout", "TimedOut", "InternalServerError", "ServiceUnavailable",
               "RateLimitError", "APIStatusError", "BadGateway", "GatewayTimeout",
               # the server answered but refused: wrong model name, bad key, request too long
               "NotFoundError", "AuthenticationError", "PermissionDenied", "BadRequestError", "UnprocessableEntity",
               "APIError")
# The harness standing in for the model by design (the model made no valid tool call, the writer failed, the
# constraint reading failed in both modes). The answer is then partly a no-model answer: kept, but counted.
FALLBACK_MARKS = ("deterministic plan used", "template used")   # both-modes reading failure is detected below


RUN_SETTINGS: Dict[str, Any] = {}   # filled once per run in main() (run_settings)
DATASET_NAME = "category_e_queries.json"
DATASET_SHA = ""


def same_settings(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> bool:
    """Rows from the same code and switches (the recorded time-free settings all match)."""
    return bool(a) and all(a.get(k) == v for k, v in b.items())


def lost_model(state: Dict[str, Any]) -> bool:
    """No usable answer from the model: an unanswered or refused call, or (whatever the error) an answer in which the
    model did nothing at all: no constraint reading, no tool call it chose, and the writer fell back to the template."""
    errors = [str(e) for e in state.get("errors") or []]
    json_mode_failed = any("extract_constraints[json_mode] failed" in e for e in errors)
    # a json_schema attempt that json_mode then recovered from (the reading exists) is not a lost model: some servers
    # refuse the schema but accept JSON mode
    recovered = bool(state.get("model_constraints"))
    counted = [e for e in errors if not (recovered and e.startswith("extract_constraints[json_schema]"))]
    if any(mark in e for e in counted for mark in UNREACHABLE):
        return True
    # whatever the error type: some model step failed outright, and the model contributed nothing at all
    # (no constraint reading, no tool call it chose). Variant A has only its tool loop, so this covers it too.
    model_failed = any("LLM loop failed" in e or "LLM failed" in e for e in errors) or json_mode_failed
    wrote = not any(("monolith: LLM loop failed" if state.get("variant") == "A" else "template used") in e for e in errors)
    contributed = (bool(state.get("model_constraints")) or wrote
                   or any(r.get("source") == "model" for r in state.get("tool_call_log") or []))
    return model_failed and not contributed


def model_fallbacks(state: Dict[str, Any]) -> List[str]:
    """Where code stood in for the model in this answer (empty when the model did every step)."""
    errors = [str(e) for e in state.get("errors") or []]
    out = [e for e in errors if any(mark in e for mark in FALLBACK_MARKS)]
    if any("extract_constraints[json_schema] failed" in e for e in errors) and \
            any("extract_constraints[json_mode] failed" in e for e in errors):
        out.append("extract_constraints: both modes failed; code-only reading used")
    return out


def run_settings(profile: Dict[str, Any]) -> Dict[str, Any]:
    """What produced a result row: the code version and the switches that change answers. Stored in every row so
    results from different code or settings are never mixed up (the analysis used to read the config at analysis
    time, not at run time)."""
    import os
    import subprocess
    from agents import supervisor as sv

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    cfg = sv.load_config()
    arch = cfg.get("architecture", {})
    # the exact code: commit plus a fingerprint of uncommitted edits and new Python files, so two different
    # uncommitted versions of the same commit are told apart
    new_files = sorted(f for f in git("ls-files", "--others", "--exclude-standard").splitlines()
                       if f.endswith((".py", ".json", ".yaml", ".yml")))
    edits = git("diff", "HEAD") + "".join(f"\n{f} {git('hash-object', f)}" for f in new_files)   # contents, not names
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(edits.strip()),
        "code_fingerprint": hashlib.sha1(edits.encode()).hexdigest()[:12] if edits.strip() else "clean",
        "dataset": DATASET_NAME,
        "dataset_sha": DATASET_SHA,      # the file's contents: two files with the same name are not the same dataset
        "profile": profile.get("name"), "model": profile.get("model"),
        "code_written_facts": bool(arch.get("code_written_facts", True)),
        "extraction_reasoning": os.getenv("HDB_EXTRACTION_REASONING") == "1" or bool(arch.get("extraction_reasoning", False)),
        "disclose_unbuilt_tools": bool(arch.get("disclose_unbuilt_tools", True)),
        "mocked_tools": sorted(t for t, v in cfg.get("tools", {}).get("use_mock", {}).items() if v == "mock"),
        "input_guardrail_installed": sv._INPUT_GUARDRAIL is not None,
        "runtime_auditor_installed": sv._AUDITOR is not None,
        "llm_timeout_s": os.getenv("LLM_TIMEOUT_S") or (cfg.get("llm", {}).get("profiles", {}).get(profile.get("name") or "", {}) or {}).get("timeout_s"),
    }


def run_one(mod, item: Dict[str, Any], variant: str, profile: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    from langchain_core.callbacks import get_usage_metadata_callback
    t0 = time.perf_counter()
    try:
        with get_usage_metadata_callback() as cb:
            state = mod.run(item["query"], item.get("initial_map_state"))
        latency = time.perf_counter() - t0
        row = {"run": run_id, "variant": variant, "id": item["id"], "sub_use_case": item.get("sub_use_case"),
               "profile": profile["name"], "model": profile.get("model"), "latency_s": round(latency, 2),
               "tokens": _token_totals(cb.usage_metadata),
               "status": "ok", "scores": M.score_item(item, state), "state_excerpt": state_excerpt(state),
               "fallbacks": model_fallbacks(state), "settings": RUN_SETTINGS, "state": rescore_state(state)}
        if item.get("follow_up"):
            row["follow_up"] = run_follow_up(mod, item, state)
        if lost_model(state) or (row.get("follow_up") or {}).get("status") == "model_unreachable":
            row["status"] = "model_unreachable"  # not a result: resume runs it again
        return row
    except Exception as e:  # noqa: BLE001 — a crash is a data point, not the end of the run
        return {"run": run_id, "variant": variant, "id": item["id"], "sub_use_case": item.get("sub_use_case"),
                "profile": profile["name"], "model": profile.get("model"), "latency_s": round(time.perf_counter() - t0, 2),
                "status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:],
                "settings": RUN_SETTINGS}


def follow_up_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": item["id"], "sub_use_case": "E1", "ground_truth_facts": item["follow_up"]["ground_truth_facts"]}


def run_follow_up(mod, item: Dict[str, Any], first_state: Dict[str, Any]) -> Dict[str, Any]:
    """Second turn: the user answers the follow-up question; scored against the complete-facts ground truth."""
    from langchain_core.callbacks import get_usage_metadata_callback
    t0 = time.perf_counter()
    try:
        with get_usage_metadata_callback() as cb:
            state = mod.run(item["follow_up"]["answer"], previous=first_state)
        return {"status": "model_unreachable" if lost_model(state) else "ok", "fallbacks": model_fallbacks(state),
                "latency_s": round(time.perf_counter() - t0, 2), "tokens": _token_totals(cb.usage_metadata),
                "scores": M.ground_truth_scores(follow_up_item(item), state),
                "draft_response": (state.get("draft_response") or "")[:4000], "state": rescore_state(state)}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    agg = M.aggregate([r["scores"] for r in ok])
    for v, a in agg.items():
        lat = sorted(r["latency_s"] for r in ok if r["variant"] == v)
        a["latency_p50_s"] = lat[len(lat) // 2] if lat else None
        a["latency_p95_s"] = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))] if lat else None
        a["errors"] = sum(1 for r in rows if r["variant"] == v and r.get("status") == "error")
        toks = [r.get("tokens", {}).get("total_tokens") for r in ok if r["variant"] == v and r.get("tokens")]
        outs = [r.get("tokens", {}).get("output_tokens") for r in ok if r["variant"] == v and r.get("tokens")]
        a["mean_total_tokens"] = round(sum(toks) / len(toks)) if toks else None
        a["mean_output_tokens"] = round(sum(outs) / len(outs)) if outs else None
        a["e1_with_candidates"] = sum(1 for r in ok if r["variant"] == v and r["sub_use_case"] == "E1" and r["state_excerpt"]["n_candidates"] > 0)
        a["e2_relaxation_attempted"] = sum(1 for r in ok if r["variant"] == v and r["sub_use_case"] == "E2" and r["scores"]["relaxation_attempted"])
        a["answers_with_fallback"] = sum(1 for r in ok if r["variant"] == v and r.get("fallbacks"))
        # n/a (None) when there is nothing to measure: a no-model run, or rows recorded before calls carried a source
        model_f1 = [r["scores"]["tool_selection_model_chosen"]["f1"] for r in ok
                    if r["variant"] == v and "tool_selection_model_chosen" in r["scores"]
                    and (r.get("settings") or {}).get("profile", r.get("profile")) != "none"]
        a["tool_selection_f1_model_chosen"] = sum(model_f1) / len(model_f1) if model_f1 else None
        reads = [r["scores"].get("extraction_model_only") for r in ok if r["variant"] == v and r["scores"].get("extraction_model_only")]
        a["model_reading_search_rate"] = (sum(x["search_correct"] for x in reads) / sum(x["search_total"] for x in reads)) if reads else None
        facts = sum(x["facts_total"] for x in reads)
        a["model_reading_facts_rate"] = (sum(x["facts_correct"] for x in reads) / facts) if facts else None
    return agg


def print_table(agg: Dict[str, Any]) -> None:
    cols = ["n", "tool_selection_f1", "tool_selection_f1_model_chosen", "model_reading_search_rate",
            "model_reading_facts_rate", "parameter_em_rate", "csr", "routing_exact_rate", "mean_tool_calls",
            "latency_p50_s", "latency_p95_s", "mean_total_tokens", "errors", "answers_with_fallback",
            "e1_with_candidates", "e2_relaxation_attempted"]
    print("\nvariant | " + " | ".join(cols))
    for v, a in agg.items():
        cells = []
        for c in cols:
            x = a.get(c)
            cells.append("-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x)))
        print(f"{v:7s} | " + " | ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--ids", default="", help="comma-separated query ids; default all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the results file (e.g. seed2)")
    ap.add_argument("--rerun", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute scores from stored state with the current metrics")
    ap.add_argument("--dataset", default="", help="'varied' for evaluation/category_e_varied.json (or a path); default "
                                                   "the 15 grounded questions")
    args = ap.parse_args()

    profile = active_profile()
    queries = dataset_path(args.dataset)
    global RUN_SETTINGS, DATASET_NAME, DATASET_SHA
    DATASET_NAME = queries.name
    DATASET_SHA = hashlib.sha1(queries.read_bytes()).hexdigest()[:12]
    RUN_SETTINGS = run_settings(profile)
    path = results_path(profile, args.tag)
    items = json.loads(queries.read_text())
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.rescore:
        rows = list(load_done(path).values())
        # each row is scored against the dataset it was run on (its settings name it); rows from before 30 Oct record
        # none and are looked up in the known datasets by id
        sets = {q.name: {i["id"]: i for i in json.loads(q.read_text())} for q in (QUERIES, VARIED, queries) if q.exists()}
        shas = {q.name: hashlib.sha1(q.read_bytes()).hexdigest()[:12] for q in (QUERIES, VARIED, queries) if q.exists()}
        any_set = {k: i for by_id in sets.values() for k, i in by_id.items()}
        n = unnamed = 0
        for r in rows:
            if r.get("status") == "ok" and r.get("state"):
                named = (r.get("settings") or {}).get("dataset")
                sha = (r.get("settings") or {}).get("dataset_sha")
                if named and (named not in sets or (sha and sha != shas[named])):
                    print(f"  skipped {r['id']}: it was run on {named}; pass --dataset with that file to rescore it")
                    continue
                item = (sets[named] if named else any_set).get(r["id"])
                if item is None:
                    print(f"  skipped {r['id']}: not in {named or 'the known datasets'}")
                    continue
                unnamed += not named
                r["scores"] = M.score_item(item, r["state"]); n += 1
                fu = r.get("follow_up") or {}
                if fu.get("status") == "ok" and fu.get("state") and item.get("follow_up"):
                    fu["scores"] = M.ground_truth_scores(follow_up_item(item), fu["state"])
        if unnamed:
            print(f"  {unnamed} rows record no dataset (run before 30 Oct): scored against the dataset with the same id")
        path.rename(path.with_suffix(f".{datetime.now().strftime('%Y%m%d-%H%M%S')}.prescore.bak"))
        path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        print(f"rescored {n} rows in {path}")
        args.summary_only = True

    if args.summary_only:
        rows = list(load_done(path).values())
        agg = summarise(rows)
        path.with_suffix(".summary.json").write_text(json.dumps(agg, indent=2))
        print_table(agg)
        return

    # resume: only successful rows count as done; a crashed or timed-out row is run again
    existing = {} if args.rerun else load_done(path)
    # a crashed row from before 30 Oct records no settings; it is rerun, not taken as another dataset
    other_sets = {(r.get("settings") or {}).get("dataset") for r in existing.values()
                  if r.get("settings") or r.get("status") == "ok"} - {DATASET_NAME}
    other_sets |= {f"another file named {DATASET_NAME}" for r in existing.values()
                   if (r.get("settings") or {}).get("dataset") == DATASET_NAME
                   and (r.get("settings") or {}).get("dataset_sha") not in (None, DATASET_SHA)}
    if other_sets:
        named = ", ".join(sorted(s or "an unknown dataset (rows from before 30 Oct)" for s in other_sets))
        sys.exit(f"{path.name} already holds results for {named}; use another --tag for {DATASET_NAME}")
    # only rows from this exact code and these settings count as done; others (or rows that recorded none) are rerun
    done = {k: r for k, r in existing.items() if r.get("status") == "ok" and same_settings(r.get("settings"), RUN_SETTINGS)}
    stale = sum(1 for r in existing.values() if r.get("status") == "ok") - len(done)
    if stale:
        print(f"{stale} earlier rows come from other code or settings (or recorded none): rerunning them")
    if args.rerun and path.exists():
        path.rename(path.with_suffix(f".{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak.jsonl"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    todo = [(v, it) for v in variants for it in items if (v, it["id"]) not in done]
    print(f"profile={profile['name']} model={profile.get('model')} results={path}")
    print(f"{len(done)} rows already done, {len(todo)} to run")

    unreachable = 0
    with path.open("a", encoding="utf-8") as f:
        for n, (v, it) in enumerate(todo, 1):
            if unreachable >= 2:  # the server or tunnel is gone: every further row would fail the same way
                print(f"MODEL_UNREACHABLE: stopped after {n - 1}/{len(todo)}; run the same command again to resume")
                break
            mod = load_variant(v)
            print(f"[{n}/{len(todo)}] variant {v} {it['id']} ...", end="", flush=True)
            row = run_one(mod, it, v, profile, run_id)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if row["status"] == "ok":
                s = row["scores"]
                print(f" {row['latency_s']}s f1={s['tool_selection']['f1']:.2f} em={s['parameter_em']['rate']} "
                      f"cands={row['state_excerpt']['n_candidates']} errs={len(row['state_excerpt']['errors'] or [])}")
            else:
                print(f" {row['status'].upper()} {row.get('error', '')}")
            unreachable = unreachable + 1 if row["status"] == "model_unreachable" else 0

    rows = list(load_done(path).values())
    agg = summarise(rows)
    path.with_suffix(".summary.json").write_text(json.dumps(agg, indent=2))
    print_table(agg)
    print(f"\nsummary -> {path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
