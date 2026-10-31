"""
LangSmith datasets and experiments for the HDB-Compass question sets (evaluation/langsmith_evals.py).

    python -m evaluation.langsmith_evals --dataset category_a --upload-only          # create/update the dataset only
    LLM_PROFILE=none python -m evaluation.langsmith_evals --dataset category_a --variant C --limit 3
    python -m evaluation.langsmith_evals --dataset category_e --variant C

What it does
  * Datasets  hdb-compass-category-a / -c / -e are built from evaluation/category_{a,c,e}_queries.json. Idempotent:
    a dataset is looked up by name, and every example has a stable id (uuid5 of dataset name + question id), so a
    second upload updates the examples in place instead of duplicating them.
  * Experiments  run the chosen variant (A, B or C; default C) over the examples with langsmith's `evaluate` and score
    each answer with the same rules the local runner uses (evaluation/architecture_metrics.py): route correctness, tool
    precision/recall/F1 against the expected tools, ground-truth facts (policy key clauses for C, blocks and
    financial figures for E, the statistics figures for A) and an answerability check (no menu, no error, a real answer).
  * Add-on packs: C and E run with HDB_ADDONS=off like run_category_e_prelim.py (results stay comparable); A runs with
    HDB_ADDONS=app, the way the app answers. The variable is set for the run only.
  * The credentials and destination are checked first (agents/tracing.py preflight); a failing check stops the run
    with the diagnosis instead of an upload error per example.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import architecture_metrics as M  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "evaluation"
DATASETS: Dict[str, Dict[str, Any]] = {
    "category_a": {"name": "hdb-compass-category-a", "file": "category_a_queries.json", "addons": "app",
                   "description": "Category A: historical resale statistics questions with ground-truth figures"},
    "category_b": {"name": "hdb-compass-category-b", "file": "category_b_queries.json", "addons": "off",
                   "description": "Category B: spatial reasoning, primary school boundaries, multi-amenity, and dual-transit commute"},
    "category_c": {"name": "hdb-compass-category-c", "file": "category_c_queries.json", "addons": "off",
                   "description": "Category C: HDB policy questions scored on key clauses"},
    "category_e": {"name": "hdb-compass-category-e", "file": "category_e_queries.json", "addons": "off",
                   "description": "Category E: multi-constraint search, grants and affordability with ground truth"},
}
ID_NAMESPACE = uuid.UUID("5b0d1c2e-7a41-4c0e-9d57-48a7d0c0ffee")
OUTPUT_DROP = ("query",)   # the question lives in the inputs; everything else is the reference for the evaluators


# ----------------------------------------------------------------------------------------------- dataset building
def load_items(key: str) -> List[Dict[str, Any]]:
    import json
    return json.loads((EVAL_DIR / DATASETS[key]["file"]).read_text(encoding="utf-8"))


def example_id(dataset_name: str, qid: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{dataset_name}:{qid}"))


def build_examples(key: str, items: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """One LangSmith example per question: inputs the target reads, outputs the reference the evaluators score against."""
    name = DATASETS[key]["name"]
    out = []
    for it in (items if items is not None else load_items(key)):
        out.append({
            "id": example_id(name, it["id"]),
            "inputs": {"query": it["query"], "initial_map_state": it.get("initial_map_state"), "question_id": it["id"]},
            "outputs": {k: v for k, v in it.items() if k not in OUTPUT_DROP},
            "metadata": {"question_id": it["id"], "category": it.get("category"), "sub_use_case": it.get("sub_use_case"),
                         "dataset_key": key},
        })
    return out


def ensure_dataset(client: Any, key: str, items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Create the dataset if it is missing, then upsert every example by its stable id. Returns name, id, counts."""
    spec = DATASETS[key]
    if client.has_dataset(dataset_name=spec["name"]):
        ds = client.read_dataset(dataset_name=spec["name"])
        created = False
    else:
        ds = client.create_dataset(spec["name"], description=spec["description"])
        created = True
    examples = build_examples(key, items)
    existing = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
    new = [e for e in examples if e["id"] not in existing]
    old = [e for e in examples if e["id"] in existing]
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    for e in old:
        client.update_example(example_id=e["id"], inputs=e["inputs"], outputs=e["outputs"], metadata=e["metadata"])
    return {"name": spec["name"], "id": str(ds.id), "created": created, "added": len(new), "updated": len(old),
            "total": len(examples)}


# -------------------------------------------------------------------------------------------------------- target
def summarise_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """The run's outputs: small, readable in the UI, and enough for every evaluator (no full tool payloads)."""
    clar = state.get("clarification") or {}
    return {
        "answer": state.get("draft_response") or "",
        "variant": state.get("variant"),
        "active_subagents": list(state.get("active_subagents") or []),
        "tool_calls": [{k: r.get(k) for k in ("agent", "tool", "arguments", "ok", "error")}
                       for r in state.get("tool_call_log") or []],
        "clarification_kind": clar.get("kind") if isinstance(clar, dict) else None,
        "errors": [str(e) for e in state.get("errors") or []],
    }


def make_target(variant: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    from evaluation.run_category_e_prelim import load_variant
    mod = load_variant(variant)

    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return summarise_state(mod.run(inputs["query"], inputs.get("initial_map_state"), question_id=inputs.get("question_id")))
    return target


# ------------------------------------------------------------------------------------------------------ evaluators
def _item(example: Any) -> Dict[str, Any]:
    inputs = dict(getattr(example, "inputs", None) or {})
    ref = dict(getattr(example, "outputs", None) or {})
    return {**ref, "id": inputs.get("question_id") or ref.get("id"), "query": inputs.get("query")}


def _state(run: Any) -> Dict[str, Any]:
    o = dict(getattr(run, "outputs", None) or {})
    return {"variant": o.get("variant"), "draft_response": o.get("answer") or "",
            "active_subagents": o.get("active_subagents") or [], "tool_call_log": o.get("tool_calls") or [],
            "errors": o.get("errors") or []}


def route_correct(run: Any, example: Any) -> Dict[str, Any]:
    item, st = _item(example), _state(run)
    if not st["variant"] or not item.get("expected_agents"):
        return {"key": "route_correct", "score": None, "comment": "no expected routing or variant"}
    r = M.routing_accuracy(item["expected_agents"], st["active_subagents"], st["variant"])
    if not r.get("applicable"):
        return {"key": "route_correct", "score": None, "comment": "not applicable to this variant"}
    return {"key": "route_correct", "score": 1.0 if r["exact"] else 0.0, "comment": f"expected {r['expected']}, got {r['actual']}"}


def tool_f1(run: Any, example: Any) -> Dict[str, Any]:
    item, st = _item(example), _state(run)
    r = M.tool_selection_prf(item.get("expected_mcp_calls") or [], st["tool_call_log"])
    return {"results": [
        {"key": "tool_f1", "score": r["f1"], "comment": f"expected {r['expected']}, called {r['predicted']}"},
        {"key": "tool_recall", "score": r["recall"]},
        {"key": "tool_precision", "score": r["precision"]}]}


def _stat_figures(truth: Dict[str, Any]) -> List[float]:
    figs: List[float] = []
    for g in truth.get("groups") or []:
        for k in ("median_price", "count"):
            if g.get(k) is not None:
                figs.append(float(g[k]))
    return figs


def figures_present(draft: str, truth: Dict[str, Any]) -> Dict[str, Any]:
    """Category A: how many of the ground-truth figures (each group's sales count and median price) appear in the answer."""
    figs = _stat_figures(truth)
    if not figs:
        return {"found": 0, "total": 0, "fraction": None}
    found = sum(1 for f in figs if M.mentions_amount(draft, f, tol=max(1.0, abs(f) * 0.001)))
    return {"found": found, "total": len(figs), "fraction": found / len(figs)}


def ground_truth(run: Any, example: Any) -> Dict[str, Any]:
    item, st = _item(example), _state(run)
    truth = item.get("ground_truth_facts") or {}
    if "groups" in truth:   # Category A statistics
        r = figures_present(st["draft_response"], truth)
        return {"results": [
            {"key": "figures_present", "score": r["fraction"], "comment": f"{r['found']} of {r['total']} ground-truth figures in the answer"},
            {"key": "ground_truth_success", "score": 1.0 if r["found"] == r["total"] else 0.0}]}
    gt = M.ground_truth_scores(item, st)
    if gt is None:
        return {"key": "ground_truth_success", "score": None, "comment": "question has no ground truth"}
    res = [{"key": "ground_truth_success", "score": 1.0 if gt.get("success") else 0.0}]
    if gt.get("kind") == "policy":
        res.append({"key": "clause_coverage", "score": gt["clause_coverage"],
                    "comment": f"{gt['clauses_found']} of {gt['clauses_total']} key clauses; missing {gt['clauses_missing']}"})
    elif "block_precision" in gt:
        res += [{"key": "block_precision", "score": gt["block_precision"]}, {"key": "block_recall", "score": gt["block_recall"]}]
    return {"results": res}


def answerable(run: Any, example: Any) -> Dict[str, Any]:
    """The assistant answered: some text, no menu of options, no tool error recorded."""
    o = dict(getattr(run, "outputs", None) or {})
    problems = []
    if not (o.get("answer") or "").strip():
        problems.append("empty answer")
    if o.get("clarification_kind") == "menu":
        problems.append("answered with the menu")
    if o.get("errors"):
        problems.append(f"{len(o['errors'])} error(s)")
    if any(c.get("ok") is False for c in o.get("tool_calls") or []):
        problems.append("a tool call failed")
    return {"key": "answerable", "score": 0.0 if problems else 1.0, "comment": "; ".join(problems) or "answered"}


EVALUATORS = [route_correct, tool_f1, ground_truth, answerable]


# --------------------------------------------------------------------------------------------------------- running
@contextlib.contextmanager
def addons_mode(mode: str) -> Iterator[None]:
    old = os.environ.get("HDB_ADDONS")
    os.environ["HDB_ADDONS"] = mode
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("HDB_ADDONS", None)
        else:
            os.environ["HDB_ADDONS"] = old


def experiment_prefix(key: str, variant: str) -> str:
    return f"hdb-eval-{key.replace('_', '-')}-variant-{variant.lower()}"


def run_experiment(client: Any, key: str, variant: str, limit: Optional[int] = None,
                   evaluators: Optional[Sequence[Callable]] = None, target: Optional[Callable] = None) -> Any:
    spec = DATASETS[key]
    items = load_items(key)[:limit] if limit else load_items(key)
    ids = [example_id(spec["name"], it["id"]) for it in items]
    data = list(client.list_examples(dataset_name=spec["name"], example_ids=ids))
    if not data:
        raise SystemExit(f"dataset {spec['name']} has no examples; run with --upload-only first")
    with addons_mode(spec["addons"]):
        return client.evaluate(
            target or make_target(variant), data=data, evaluators=list(evaluators or EVALUATORS),
            experiment_prefix=experiment_prefix(key, variant),
            metadata={"variant": variant, "dataset_key": key, "addons": spec["addons"],
                      "model_profile": os.getenv("LLM_PROFILE", "unknown"), "limit": limit},
            max_concurrency=1)


def check_credentials() -> Dict[str, Any]:
    from agents import tracing
    tracing._load_env()
    pf = tracing.preflight()
    if not pf["ok"]:
        print(tracing.format_check(pf))
        raise SystemExit("LangSmith is not usable with the current settings (see above); nothing was uploaded.")
    return pf


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LangSmith datasets and experiments for HDB-Compass")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--variant", default="C", choices=["A", "B", "C"])
    ap.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    ap.add_argument("--upload-only", action="store_true", help="create/update the dataset and stop")
    args = ap.parse_args(argv)
    check_credentials()
    from langsmith import Client
    client = Client()
    info = ensure_dataset(client, args.dataset)
    print(f"dataset {info['name']} ({info['id']}): {'created' if info['created'] else 'existing'}, "
          f"{info['added']} added, {info['updated']} updated, {info['total']} examples")
    if args.upload_only:
        return 0
    res = run_experiment(client, args.dataset, args.variant, args.limit)
    print(f"experiment: {getattr(res, 'experiment_name', res)}")
    try:
        totals: Dict[str, List[float]] = {}
        for row in res:   # no pandas needed: average each feedback key over the rows
            for r in (row.get("evaluation_results") or {}).get("results", []):
                if isinstance(r.score, (int, float)):
                    totals.setdefault(r.key, []).append(float(r.score))
        for k, v in sorted(totals.items()):
            print(f"  {k:<20} {sum(v) / len(v):.3f}  (n={len(v)})")
    except Exception as e:  # noqa: BLE001
        print(f"(scores table unavailable: {e})")
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
