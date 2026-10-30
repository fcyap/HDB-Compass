"""
RAG Retrieval Pipeline Ablation Study Runner (evaluation/run_rag_ablation.py)
Proposal Section 6.2 (Experiment 2, Page 11) & File Structure (Page 14).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Category C).

Compares three retrieval configurations across the 15 Category C golden queries:
  1. RAG-1: Naive Dense Vector Search (top-5 cosine similarity)
  2. RAG-2: Hybrid Search (BM25 + Dense fused via Reciprocal Rank Fusion, top-5)
  3. RAG-3: Hybrid Search + Cross-Encoder Reranking (top-20 RRF filtered to top-5)

Evaluates:
  - Context Recall: % of ground truth gold policy clauses retrieved in top-5
  - Context Precision: Rank-weighted precision (RAGAS formulation)
  - Mean Reciprocal Rank (MRR): 1 / rank of first relevant passage
  - Latency: P50 and P95 latency (ms) per query
  - Generates publication-ready comparative tables and qualitative failure breakdown
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.policy_rag_tools import execute_rag_pipeline

QUERIES_FILE = PROJECT_ROOT / "evaluation" / "category_c_queries.json"
RESULTS_JSON = PROJECT_ROOT / "evaluation" / "rag_ablation_results.json"


def clean_text(text: str) -> str:
    """Strips markdown formatting (asterisks, hashes, backticks, pipes) and lowercases."""
    return re.sub(r"[*#`|_]", " ", text).lower()


def is_clause_in_passage(clause: Any, p_text_clean: str, p_sec_clean: str) -> bool:
    full_clean = f"{p_sec_clean} {p_text_clean}"
    if isinstance(clause, (list, tuple)):
        return all(clean_text(str(token)).strip() in full_clean for token in clause)
    c_clean = clean_text(str(clause)).strip()
    if c_clean in full_clean:
        return True
    words = [w for w in re.findall(r"\b\w+\b", c_clean) if len(w) > 2]
    if words and sum(1 for w in words if w in full_clean) / len(words) >= 0.75:
        return True
    return False


def evaluate_query_retrieval(
    test_case: Dict[str, Any],
    pipeline: str
) -> Dict[str, Any]:
    """Runs a single query under a specific RAG pipeline and computes retrieval metrics."""
    query = test_case["query"]
    gt = test_case.get("ground_truth_facts", {})
    raw_clauses = gt.get("key_clauses", [])
    target_sections = [clean_text(s) for s in gt.get("target_sections", [])]

    t0 = time.perf_counter()
    res = execute_rag_pipeline(query=query, pipeline=pipeline, top_k=5)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    passages = res.passages

    # Pre-clean passages once
    cleaned_passages = [
        (clean_text(p.text), clean_text(p.section))
        for p in passages
    ]

    # 1. Context Recall: fraction of gold clauses captured in any retrieved passage
    matched_clauses = 0
    for clause in raw_clauses:
        if any(is_clause_in_passage(clause, p_txt, p_sec) for p_txt, p_sec in cleaned_passages):
            matched_clauses += 1
    context_recall = matched_clauses / max(1, len(raw_clauses))

    # 2. Section Relevance & Context Precision (RAGAS formula)
    relevance_vector = []
    first_relevant_rank = None

    for rank_idx, (p_txt, p_sec) in enumerate(cleaned_passages, start=1):
        is_relevant = (
            any(is_clause_in_passage(c, p_txt, p_sec) for c in raw_clauses) or
            any(s in p_sec for s in target_sections)
        )
        relevance_vector.append(1 if is_relevant else 0)
        if is_relevant and first_relevant_rank is None:
            first_relevant_rank = rank_idx
            first_relevant_rank = rank_idx

    total_relevant = sum(relevance_vector)
    if total_relevant == 0:
        context_precision = 0.0
    else:
        cum_precisions = []
        rel_so_far = 0
        for k, rel in enumerate(relevance_vector, start=1):
            if rel == 1:
                rel_so_far += 1
                cum_precisions.append(rel_so_far / k)
        context_precision = sum(cum_precisions) / max(1, len(cum_precisions))

    # 3. MRR
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

    return {
        "id": test_case["id"],
        "sub_use_case": test_case["sub_use_case"],
        "pipeline": pipeline,
        "latency_ms": round(latency_ms, 2),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
        "mrr": round(mrr, 4),
        "matched_clauses": f"{matched_clauses}/{len(raw_clauses)}",
        "top_passage_section": passages[0].section if passages else "",
        "top_score": passages[0].rerank_score if passages else 0.0,
    }


def run_ablation():
    print("=" * 78)
    print(" HDB-Compass: Experiment 2 — RAG Retrieval Pipeline Ablation Study ")
    print(" Owned by Member 3 (Advanced RAG & Regulatory Knowledge Base) ")
    print("=" * 78)

    if not QUERIES_FILE.exists():
        from evaluation.build_category_c import build
        build()

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    pipelines = ["RAG-1", "RAG-2", "RAG-3"]
    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pipelines}

    print(f"\nBenchmarking {len(cases)} Category C queries across configurations: {pipelines}\n")

    for p in pipelines:
        print(f"--> Running Configuration: {p} ...")
        for case in cases:
            res = evaluate_query_retrieval(case, pipeline=p)
            all_results[p].append(res)
        print(f"    Completed {len(cases)} queries for {p}.")

    # Summary Statistics Aggregation
    summary: Dict[str, Dict[str, float]] = {}
    for p in pipelines:
        recalls = [r["context_recall"] for r in all_results[p]]
        precisions = [r["context_precision"] for r in all_results[p]]
        mrrs = [r["mrr"] for r in all_results[p]]
        latencies = [r["latency_ms"] for r in all_results[p]]

        latencies_sorted = sorted(latencies)
        p50_idx = int(0.50 * len(latencies_sorted))
        p95_idx = int(0.95 * len(latencies_sorted))

        summary[p] = {
            "mean_context_recall": round(statistics.mean(recalls), 4),
            "mean_context_precision": round(statistics.mean(precisions), 4),
            "mean_mrr": round(statistics.mean(mrrs), 4),
            "p50_latency_ms": round(latencies_sorted[p50_idx], 2),
            "p95_latency_ms": round(latencies_sorted[p95_idx], 2),
        }

    # Print Comparative Table
    print("\n" + "=" * 84)
    print(" EXPERIMENT 2 RESULTS SUMMARY (Category C Golden Dataset N=15)")
    print("=" * 84)
    print(f"{'Pipeline Configuration':<35} | {'Context Recall':<14} | {'Context Precision':<17} | {'MRR':<8} | {'P50 Latency (ms)':<16}")
    print("-" * 84)
    for p, stats in summary.items():
        name = {
            "RAG-1": "RAG-1 (Naive Dense Vector)",
            "RAG-2": "RAG-2 (Hybrid BM25 + Dense RRF)",
            "RAG-3": "RAG-3 (Hybrid + Cross-Encoder)"
        }[p]
        print(f"{name:<35} | {stats['mean_context_recall']:<14.4f} | {stats['mean_context_precision']:<17.4f} | {stats['mean_mrr']:<8.4f} | {stats['p50_latency_ms']:<16.2f}")
    print("=" * 84)

    # Detailed Sub-use case breakdown
    print("\nSub-Use Case Breakdown (Context Recall by Subcategory):")
    print("-" * 65)
    print(f"{'Sub-Use Case':<32} | {'RAG-1':<9} | {'RAG-2':<9} | {'RAG-3':<9}")
    print("-" * 65)
    for sub in ["C1", "C2", "C3"]:
        sub_name = {
            "C1": "C1: Standard / Plus / Prime",
            "C2": "C2: Scheme & Age Eligibility",
            "C3": "C3: Wait-Out & EIP Quotas"
        }[sub]
        r1 = statistics.mean([r["context_recall"] for r in all_results["RAG-1"] if r["sub_use_case"] == sub])
        r2 = statistics.mean([r["context_recall"] for r in all_results["RAG-2"] if r["sub_use_case"] == sub])
        r3 = statistics.mean([r["context_recall"] for r in all_results["RAG-3"] if r["sub_use_case"] == sub])
        print(f"{sub_name:<32} | {r1:<9.4f} | {r2:<9.4f} | {r3:<9.4f}")
    print("-" * 65)

    # Save to JSON
    output_payload = {
        "summary": summary,
        "queries_evaluated": len(cases),
        "detailed_results": all_results,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nDetailed ablation results saved to: {RESULTS_JSON}\n")


if __name__ == "__main__":
    run_ablation()
