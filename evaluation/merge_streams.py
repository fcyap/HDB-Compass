"""Merge per-stream result files of one run into a single results file.

The re-runs in the README split the 90 questions into disjoint subsets (`run_category_e_prelim.py --ids ...`), ran each
subset as its own process, then merged the per-stream files. For every (variant, id) pair the LAST row read wins, so a
later retry replaces an earlier failed attempt (for example a `model_unreachable` row), exactly as the runner's own
resume logic does.

    python -m evaluation.merge_streams --out evaluation/results/prelim_vllm_<model>_<tag>.jsonl \
        evaluation/results/prelim_vllm_<model>_<tag>s0.jsonl evaluation/results/prelim_vllm_<model>_<tag>s1.jsonl
    python -m evaluation.run_architecture_eval --report --no-ablation --tag <tag>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_rows(paths: list[Path]) -> list[dict]:
    """Rows from `paths` in the order given, one per (variant, id), the last occurrence winning."""
    merged: dict[tuple[str, str], dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                merged[(row["variant"], row["id"])] = row
    return list(merged.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="per-stream .jsonl files, oldest first")
    ap.add_argument("--out", required=True, type=Path, help="merged .jsonl file to write")
    args = ap.parse_args(argv)
    rows = merge_rows(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"{len(rows)} rows ({ok} ok) from {len(args.inputs)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
