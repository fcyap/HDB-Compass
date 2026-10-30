"""
Ground-truth results across several Category E or C runs (evaluation/compare_runs.py). Member 4.

One run per question and variant can be luck; this puts repeated runs side by side with the mean and range, so a
claim such as "C answers most questions fully right" carries its spread. Each cell also carries a Wilson 95%
interval, the spread from having only 15 questions. Rows that never reached the model (status model_unreachable)
are left out and counted. A file holding both categories is broken down by category.

    .venv/bin/python -m evaluation.compare_runs rep1 rep2 rep3          # results tags
    .venv/bin/python -m evaluation.compare_runs evaluation/results/prelim_vllm_..._rep1.jsonl ...

For the varied-phrasing set (ids like "E1-01~casual") it also breaks each run down by style.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import architecture_metrics as M  # noqa: E402

RESULTS = PROJECT_ROOT / "evaluation" / "results"
MEASURES = [("fully right", lambda gt: gt.get("success")),
            ("exact blocks", lambda gt: gt.get("blocks_exact")),
            ("E1 grants and max price", lambda gt: gt.get("grants_correct") and gt.get("max_price_correct")
             if "grants_correct" in gt else None),
            ("E1 stamp duty and upfront cash", lambda gt: gt.get("upfront_correct"))]


def resolve(arg: str) -> Path:
    p = Path(arg)
    if p.suffix == ".jsonl":
        return p if p.is_absolute() else (Path.cwd() / p)
    hits = sorted(RESULTS.glob(f"prelim_*_{arg}.jsonl"))
    if len(hits) > 1:
        # "regcheck" also matches "..._varied_regcheck"; the exact tag is the shortest file name
        shortest = min(len(h.stem) for h in hits)
        hits = [h for h in hits if len(h.stem) == shortest]
    if len(hits) != 1:
        sys.exit(f"tag {arg!r} matches {len(hits)} result files in {RESULTS}")
    return hits[0]


def load(path: Path) -> List[Dict[str, Any]]:
    latest: Dict[tuple, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["variant"], r["id"])] = r     # a resumed row replaces the earlier attempt
    return list(latest.values())


def counts(rows: List[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    ok = [r for r in rows if r["variant"] == variant and r.get("status") == "ok"]
    out: Dict[str, Any] = {"runs": len(ok),
                           "unreachable": sum(r["variant"] == variant and r.get("status") == "model_unreachable" for r in rows)}
    for name, f in MEASURES:
        vals = [f((r.get("scores") or {}).get("ground_truth") or {}) for r in ok]
        vals = [v for v in vals if v is not None]
        out[name] = (sum(bool(v) for v in vals), len(vals))
    fus = [r["follow_up"] for r in ok if r.get("follow_up")]
    fus = [f for f in fus if f.get("status") == "ok"]
    out["follow-up"] = (sum(bool((f.get("scores") or {}).get("success")) for f in fus), len(fus))
    return out


def style_table(runs: Dict[str, List[Dict[str, Any]]]) -> None:
    """Fully right per phrasing style, for runs of the varied set (ids "E1-01~casual")."""
    styles = sorted({r["id"].split("~", 1)[1] for rows in runs.values() for r in rows if "~" in r["id"]})
    if not styles:
        return
    print("\n| Variant | Style | " + " | ".join(runs) + " |")
    print("|---|---|" + "---|" * len(runs))
    variants = sorted({r["variant"] for rows in runs.values() for r in rows})
    for v in variants:
        for st in styles:
            cells = []
            for rows in runs.values():
                part = [r for r in rows if r["id"].endswith("~" + st)]
                c = counts(part, v)
                cells.append(M.rate_with_ci(*c["fully right"], short=True))
            print(f"| {v} | {st} | " + " | ".join(cells) + " |")


def category(r: Dict[str, Any]) -> str:
    return M.category_of(r.get("sub_use_case"), r["id"])


def table(runs: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """One line per variant, category and measure; a measure that does not apply to a category (blocks for a policy
    question, say) is left out. Each cell is right/total with its Wilson 95% interval."""
    variants = sorted({r["variant"] for rows in runs.values() for r in rows})
    cats = sorted({category(r) for rows in runs.values() for r in rows})
    names = [n for n, _ in MEASURES] + ["follow-up"]
    out = ["| Variant | Category | Measure | " + " | ".join(runs) + " | Mean | Range |",
           "|---|---|---|" + "---|" * (len(runs) + 2)]
    for v in variants:
        for c in cats:
            per = {tag: counts([r for r in rows if category(r) == c], v) for tag, rows in runs.items()}
            for n in names:
                if not any(per[t][n][1] for t in runs):
                    continue
                cells = [M.rate_with_ci(*per[t][n], short=True) for t in runs]
                rates = [per[t][n][0] / per[t][n][1] for t in runs if per[t][n][1]]
                out.append(f"| {v} | {c} | {n} | " + " | ".join(cells) +
                           f" | {mean(rates):.0%} | {min(rates):.0%}–{max(rates):.0%} |")
            if any(per[t]["unreachable"] for t in runs):
                out.append(f"| {v} | {c} | rows that never reached the model (left out) | " +
                           " | ".join(str(per[t]['unreachable']) for t in runs) + " | | |")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    # label each column with the tag as typed (or the file name), so "regcheck" and "varied_regcheck" stay apart
    runs = {(Path(a).stem if a.endswith(".jsonl") else a): load(resolve(a)) for a in sys.argv[1:]}
    print("\n".join(table(runs)))
    print("\nCells: right/total (Wilson 95% interval). Category C fully right: every key clause in the answer "
          "(Member 3's matching rule); see the analysis report for the clauses known to be wrong.")
    style_table(runs)


if __name__ == "__main__":
    main()
