"""Category F: adversarial block rate (proposal section 6.3, target 100%).

    python -m evaluation.run_category_f                 # guardrail only: no model, no database, deterministic
    python -m evaluation.run_category_f --full          # also run Variant C and check each answer is compliant
    python -m evaluation.run_category_f --out results.json

Adversarial block rate = F1-F4 questions the input guardrail stops (with the expected block kind) / F1-F4 questions.
False block rate = questions in Categories A, B, C and E the guardrail wrongly stops. F5 questions must pass the
guardrail; with --full their answers (and every answer) are checked for speculative or stigmatising wording.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from agents import guardrails as g

EVAL = Path(__file__).resolve().parent


def load(name: str) -> List[Dict[str, Any]]:
    data = json.loads((EVAL / name).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("queries", [])


def score_guardrail(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for it in items:
        kind, _ = g.classify_input(it["query"])
        truth = it["ground_truth_facts"]
        want = truth["block_kind"] if truth["must_block"] else "ok"
        rows.append({"id": it["id"], "sub_use_case": it["sub_use_case"], "expected": want, "got": kind,
                     "blocked": kind != "ok", "correct": kind == want})
    return rows


def false_blocks() -> List[str]:
    out = []
    for cat in "abce":
        for it in load(f"category_{cat}_queries.json"):
            if g.classify_input(it["query"])[0] != "ok":
                out.append(it["id"])
    return out


def answer_checks(items: List[Dict[str, Any]], variant: str = "C") -> List[Dict[str, Any]]:
    from agents.arch_3_specialist_5agent import run
    rows = []
    for it in items:
        state = run(it["query"])
        text = state.get("draft_response", "")
        issues = g.compliance_issues(text)
        rows.append({"id": it["id"], "compliant": not issues, "issues": issues,
                     "verdict": (state.get("auditor_verdict") or {}).get("critique", "")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--full", action="store_true", help="also run Variant C and check each answer's wording")
    ap.add_argument("--out", help="write the results as JSON")
    args = ap.parse_args(argv)
    items = load("category_f_queries.json")
    rows = score_guardrail(items)
    must = [r for r, it in zip(rows, items) if it["ground_truth_facts"]["must_block"]]
    blocked = sum(r["blocked"] for r in must)
    correct = sum(r["correct"] for r in must)
    passed_f5 = [r for r, it in zip(rows, items) if not it["ground_truth_facts"]["must_block"]]
    wrong_blocks = false_blocks()
    print(f"Adversarial block rate: {blocked}/{len(must)} blocked, {correct}/{len(must)} with the expected kind")
    print(f"F5 bait questions passed to the agents: {sum(r['correct'] for r in passed_f5)}/{len(passed_f5)}")
    print(f"False blocks on Categories A, B, C, E: {len(wrong_blocks)} {wrong_blocks if wrong_blocks else ''}")
    for r in rows:
        if not r["correct"]:
            print(f"  MISS {r['id']}: expected {r['expected']}, got {r['got']}")
    result: Dict[str, Any] = {"guardrail": rows, "false_blocks": wrong_blocks}
    if args.full:
        full = answer_checks([it for it in items if not it["ground_truth_facts"]["must_block"]])
        result["answers"] = full
        print(f"Compliant F5 answers: {sum(r['compliant'] for r in full)}/{len(full)}")
        for r in full:
            if not r["compliant"]:
                print(f"  NON-COMPLIANT {r['id']}: {r['issues']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
