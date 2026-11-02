"""Combine the per-category question files into evaluation/benchmark_hdb_90.json (proposal section 6.1).

    python -m evaluation.build_benchmark_90

Reads category_{a..f}_queries.json, keeps 15 per category and checks every item has the benchmark fields. A missing
category file (Category D is Member 1's) is reported, not invented, so the output only reaches 90 once all six exist.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

EVAL = Path(__file__).resolve().parent
CATEGORIES = "ABCDEF"
PER_CATEGORY = 15
REQUIRED = ("id", "category", "query", "expected_agents", "expected_mcp_calls", "ground_truth_facts")


def load(cat: str) -> List[Dict[str, Any]]:
    path = EVAL / f"category_{cat.lower()}_queries.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("queries", [])


def problems(item: Dict[str, Any]) -> List[str]:
    out = [f"missing {k}" for k in REQUIRED if k not in item]
    if not str(item.get("query", "")).strip():
        out.append("empty query")
    return out


def build() -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {}
    seen = set()
    for cat in CATEGORIES:
        rows = load(cat)
        issues = []
        for r in rows:
            issues += [f"{r.get('id', '?')}: {p}" for p in problems(r)]
            if r.get("id") in seen:
                issues.append(f"{r.get('id')}: duplicate id")
            seen.add(r.get("id"))
        report[cat] = {"found": len(rows), "kept": min(len(rows), PER_CATEGORY), "issues": issues}
        items += rows[:PER_CATEGORY]
    return {"items": items, "report": report}


def main():
    built = build()
    path = EVAL / "benchmark_hdb_90.json"
    path.write_text(json.dumps(built["items"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cat, r in built["report"].items():
        flag = "" if r["kept"] == PER_CATEGORY and not r["issues"] else "   <-- needs attention"
        print(f"Category {cat}: {r['kept']}/{PER_CATEGORY}{flag}")
        for i in r["issues"]:
            print(f"    {i}")
    print(f"Wrote {len(built['items'])} questions to {path}")


if __name__ == "__main__":
    main()
