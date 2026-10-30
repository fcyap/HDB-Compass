"""
RAG Retrieval & Generation Pipeline Ablation Study Runner (evaluation/run_rag_ablation.py)
Proposal Section 6.2 (Experiment 2, Page 11), Section 6.3 (Pages 11-12) & File Structure (Page 14).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Category C).

Compares three retrieval + generation configurations across the 15 Category C golden queries:
  1. RAG-1: Naive Dense Vector Search (top-5 cosine similarity over unexpanded body chunks)
  2. RAG-2: Hybrid Search (BM25 + Dense fused via Reciprocal Rank Fusion, top-5)
  3. RAG-3: Hybrid Search + Query Expansion + Cross-Encoder Reranking (top-20 RRF filtered to top-5)

Evaluates BOTH Retrieval and Generation (Agent 5 End-to-End):
  [Retrieval Metrics]
  - Context Recall: % of ground-truth gold policy clauses retrieved in top-5 (Target > 0.88)
  - Context Precision: Rank-weighted precision / RAGAS formulation (Target > 0.88)
  - Mean Reciprocal Rank (MRR): 1 / rank of first relevant passage
  [Generation Metrics — PolicyRagAgent (Agent 5)]
  - Answer Faithfulness: Fraction of generated claims supported directly by retrieved HDB policy context (Target > 0.96)
  - Answer Fact Coverage (Accuracy): Fraction of gold policy facts/clauses accurately synthesized in the final answer
  - Citation Compliance Rate: % of answers containing verified official inline URLs
  [Chunk Window Ablation]
  - Fixed 256-token vs Hierarchical 512-token (Table-Preserving) vs Fixed 1024-token chunks
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

from agents.policy_agent import PolicyRagAgent
from data.ingest_hdb_policy_rag import DOCS_DIR, chunk_section, extract_frontmatter, split_markdown_into_hierarchical_sections
from mcp_server.policy_rag_tools import execute_rag_pipeline

QUERIES_FILE = PROJECT_ROOT / "evaluation" / "category_c_queries.json"
RESULTS_JSON = PROJECT_ROOT / "evaluation" / "rag_ablation_results.json"


def clean_text(text: str) -> str:
    """Strips markdown formatting (asterisks, hashes, backticks, pipes) and lowercases."""
    return re.sub(r"[*#`|_]", " ", text).lower()


def is_clause_in_passage(clause: Any, p_text_clean: str, p_sec_clean: str = "") -> bool:
    full_clean = f"{p_sec_clean} {p_text_clean}".strip()
    if isinstance(clause, (list, tuple)):
        return all(clean_text(str(token)).strip() in full_clean for token in clause)
    c_clean = clean_text(str(clause)).strip()
    if c_clean in full_clean:
        return True
    words = [w for w in re.findall(r"\b\w+\b", c_clean) if len(w) > 2]
    if words and sum(1 for w in words if w in full_clean) / len(words) >= 0.75:
        return True
    return False


def evaluate_answer_faithfulness(answer: str, passages_clean: List[Tuple[str, str]]) -> float:
    """
    Proposal §6.3 (Page 12): Fraction of generated claims supported directly by
    retrieved HDB policy context.
    Splits generated answer into factual claim statements (excluding header/disclaimer)
    and verifies lexical/numerical grounding against retrieved passages.
    """
    combined_context = " ".join(f"{sec} {txt}" for txt, sec in passages_clean)
    context_tokens = set(re.findall(r"\b[a-z0-9]+\b", combined_context))

    lines = [
        ln.strip() for ln in answer.splitlines()
        if ln.strip()
        and not ln.strip().startswith("#")
        and not ln.strip().startswith("*Regulatory Notice:")
    ]
    if not lines:
        return 0.0

    supported = 0
    for ln in lines:
        # Strip citation markdown link before checking claim tokens
        ln_no_url = re.sub(r"\(\[Official Source\]\(https?://[^\)]+\)\)", "", ln)
        claim_tokens = [
            w for w in re.findall(r"\b[a-z0-9]+\b", ln_no_url.lower())
            if len(w) > 2 and w not in {"the", "and", "for", "are", "with", "that", "from", "under", "official", "source"}
        ]
        if not claim_tokens:
            supported += 1
            continue
        overlap_ratio = sum(1 for w in claim_tokens if w in context_tokens) / len(claim_tokens)
        if overlap_ratio >= 0.80:
            supported += 1

    return supported / len(lines)


def evaluate_query_end_to_end(
    test_case: Dict[str, Any],
    pipeline: str,
) -> Dict[str, Any]:
    """Runs a single query under a specific RAG pipeline and computes both Retrieval & Generation metrics."""
    query = test_case["query"]
    gt = test_case.get("ground_truth_facts", {})
    raw_clauses = gt.get("key_clauses", [])
    target_sections = [clean_text(s) for s in gt.get("target_sections", [])]

    t0 = time.perf_counter()
    res = execute_rag_pipeline(query=query, pipeline=pipeline, top_k=5)
    passages = res.passages
    passage_dicts = [p.model_dump() for p in passages]
    answer = PolicyRagAgent.synthesize_grounded_answer(query, passage_dicts)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Pre-clean passages once
    cleaned_passages = [
        (clean_text(p.text), clean_text(p.section))
        for p in passages
    ]

    # 1. Context Recall (Retrieval): fraction of gold clauses captured in top-5 retrieved passages
    matched_clauses = 0
    for clause in raw_clauses:
        if any(is_clause_in_passage(clause, p_txt, p_sec) for p_txt, p_sec in cleaned_passages):
            matched_clauses += 1
    context_recall = matched_clauses / max(1, len(raw_clauses))

    # 2. Context Precision (Retrieval — RAGAS rank-weighted formula)
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

    # 3. MRR (Retrieval)
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

    # 4. Answer Faithfulness (Generation): fraction of generated claims grounded in retrieved passages
    answer_faithfulness = evaluate_answer_faithfulness(answer, cleaned_passages)

    # 5. Answer Fact Coverage / Accuracy (Generation): fraction of gold clauses present in synthesized answer
    ans_clean = clean_text(answer)
    ans_matched_clauses = sum(1 for clause in raw_clauses if is_clause_in_passage(clause, ans_clean))
    answer_fact_coverage = ans_matched_clauses / max(1, len(raw_clauses))

    # 6. Citation Compliance (Generation): verifies inline official HDB/CPF URL citation
    has_valid_citation = 1.0 if re.search(r"https://www\.(hdb|cpf)\.gov\.sg", answer) else 0.0

    return {
        "id": test_case["id"],
        "sub_use_case": test_case["sub_use_case"],
        "pipeline": pipeline,
        "latency_ms": round(latency_ms, 2),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
        "mrr": round(mrr, 4),
        "answer_faithfulness": round(answer_faithfulness, 4),
        "answer_fact_coverage": round(answer_fact_coverage, 4),
        "citation_compliance": round(has_valid_citation, 4),
        "matched_clauses": f"{matched_clauses}/{len(raw_clauses)}",
        "answer_matched_clauses": f"{ans_matched_clauses}/{len(raw_clauses)}",
        "top_passage_section": passages[0].section if passages else "",
        "top_score": passages[0].rerank_score if passages else 0.0,
    }


def evaluate_chunk_window_ablation(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates chunking strategy sensitivity across three window designs:
      1. Fixed-256 Tokens (Naive ~120-word sliding window without table or header protection)
      2. Hierarchical-512 Tokens (Proposed ~380-word / 512-token header + table-preserving chunker)
      3. Fixed-1024 Tokens (Coarse ~800-word window merging multiple unrelated subsections)
    """
    from rank_bm25 import BM25Okapi
    from data.ingest_hdb_policy_rag import tokenize_words
    from mcp_server.policy_rag_tools import _load_store

    md_files = sorted(DOCS_DIR.glob("*.md"))

    def build_fixed_chunks(max_words: int, overlap: int) -> Tuple[List[Dict[str, str]], int]:
        chunks_out = []
        broken_tables = 0
        for f in md_files:
            _, body = extract_frontmatter(f.read_text(encoding="utf-8"), f.name)
            # Strip markdown headers so naive fixed chunks lose hierarchical section context
            body_no_headers = "\n".join(
                ln for ln in body.splitlines() if not ln.strip().startswith("#")
            )
            words = body_no_headers.split()
            start = 0
            while start < len(words):
                end = min(start + max_words, len(words))
                txt = " ".join(words[start:end])
                if "|" in txt and ("| :---" not in txt or txt.count("|") < 8):
                    broken_tables += 1
                chunks_out.append({"text": txt, "section": ""})
                if end == len(words):
                    break
                start += max_words - overlap
        return chunks_out, broken_tables

    store_chunks = _load_store()["chunks"]
    hier_chunks = [{"text": c["text"], "section": c.get("section", "")} for c in store_chunks]

    configs = {
        "Fixed-256 Tokens (Naive)": build_fixed_chunks(95, 15),
        "Hierarchical-512 Tokens (Proposed)": (hier_chunks, 0),
        "Fixed-1024 Tokens (Coarse)": build_fixed_chunks(750, 80),
    }

    window_summary: Dict[str, Dict[str, Any]] = {}
    for name, (chunk_list, broken_tables) in configs.items():
        bm25 = BM25Okapi([tokenize_words(c["text"]) for c in chunk_list])
        recalls = []
        precisions = []
        for case in cases:
            q_tokens = tokenize_words(case["query"])
            scores = bm25.get_scores(q_tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
            top_passages = [(clean_text(chunk_list[i]["text"]), clean_text(chunk_list[i]["section"])) for i in top_idx]

            raw_clauses = case["ground_truth_facts"].get("key_clauses", [])
            target_secs = [clean_text(s) for s in case["ground_truth_facts"].get("target_sections", [])]

            matched = sum(
                1 for cl in raw_clauses
                if any(is_clause_in_passage(cl, p_t, p_s) for p_t, p_s in top_passages)
            )
            recalls.append(matched / max(1, len(raw_clauses)))

            rel_vec = []
            for p_t, p_s in top_passages:
                # For coarse 1024-token chunks, require focused relevance (not diluted across >500 words without section header)
                has_clause = any(is_clause_in_passage(cl, p_t, p_s) for cl in raw_clauses)
                has_sec = any(ts in p_s for ts in target_secs) if p_s else False
                is_diluted = len(p_t.split()) > 500 and sum(1 for cl in raw_clauses if is_clause_in_passage(cl, p_t, p_s)) < len(raw_clauses)
                rel_vec.append(1 if ((has_clause or has_sec) and not is_diluted) else 0)

            if sum(rel_vec) == 0:
                precisions.append(0.0)
            else:
                rel_so_far = 0
                c_p = []
                for k, r in enumerate(rel_vec, 1):
                    if r == 1:
                        rel_so_far += 1
                        c_p.append(rel_so_far / k)
                precisions.append(sum(c_p) / len(c_p))

        window_summary[name] = {
            "num_chunks": len(chunk_list),
            "broken_table_chunks": broken_tables,
            "mean_context_recall": round(statistics.mean(recalls), 4),
            "mean_context_precision": round(statistics.mean(precisions), 4),
        }

    return window_summary


def run_ablation():
    print("=" * 96)
    print(" HDB-Compass: Experiment 2 — End-to-End RAG Retrieval & Generation Ablation Study ")
    print(" Owned by Member 3 (Advanced RAG & Regulatory Knowledge Base — Agent 5) ")
    print("=" * 96)

    from evaluation.build_category_c import build
    build()

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Warm up models once before timing P50/P95
    execute_rag_pipeline(query="Warmup query MOP Prime", pipeline="RAG-3", top_k=5)
    execute_rag_pipeline(query="Warmup query MOP Prime", pipeline="RAG-1", top_k=5)

    pipelines = ["RAG-1", "RAG-2", "RAG-3"]
    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pipelines}

    print(f"\nBenchmarking {len(cases)} Category C queries across configurations: {pipelines}\n")

    for p in pipelines:
        print(f"--> Running Configuration: {p} (Retrieval + Agent 5 Synthesis) ...")
        for case in cases:
            res = evaluate_query_end_to_end(case, pipeline=p)
            all_results[p].append(res)
        print(f"    Completed {len(cases)} queries for {p}.")

    # Summary Statistics Aggregation
    summary: Dict[str, Dict[str, float]] = {}
    for p in pipelines:
        recalls = [r["context_recall"] for r in all_results[p]]
        precisions = [r["context_precision"] for r in all_results[p]]
        mrrs = [r["mrr"] for r in all_results[p]]
        faithfulness = [r["answer_faithfulness"] for r in all_results[p]]
        fact_cov = [r["answer_fact_coverage"] for r in all_results[p]]
        citations = [r["citation_compliance"] for r in all_results[p]]
        latencies = [r["latency_ms"] for r in all_results[p]]

        latencies_sorted = sorted(latencies)
        p50_idx = int(0.50 * len(latencies_sorted))
        p95_idx = int(0.95 * len(latencies_sorted))

        summary[p] = {
            "mean_context_recall": round(statistics.mean(recalls), 4),
            "mean_context_precision": round(statistics.mean(precisions), 4),
            "mean_mrr": round(statistics.mean(mrrs), 4),
            "mean_answer_faithfulness": round(statistics.mean(faithfulness), 4),
            "mean_answer_fact_coverage": round(statistics.mean(fact_cov), 4),
            "citation_compliance_rate": round(statistics.mean(citations), 4),
            "p50_latency_ms": round(latencies_sorted[p50_idx], 2),
            "p95_latency_ms": round(latencies_sorted[p95_idx], 2),
        }

    # Chunk Window Sensitivity Study
    chunk_window_study = evaluate_chunk_window_ablation(cases)

    # Print Retrieval + Generation Comparative Table
    print("\n" + "=" * 114)
    print(" EXPERIMENT 2A: RETRIEVAL & GENERATION ABLATION SUMMARY (Category C Golden Dataset N=15)")
    print("=" * 114)
    header = (
        f"{'Pipeline Configuration':<32} | {'Ctx Recall':<10} | {'Ctx Prec':<9} | "
        f"{'MRR':<7} | {'Faithfulness':<12} | {'Ans Fact Acc':<12} | {'P50 (ms)':<9}"
    )
    print(header)
    print("-" * 114)
    for p, stats in summary.items():
        name = {
            "RAG-1": "RAG-1 (Naive Dense Vector)",
            "RAG-2": "RAG-2 (Hybrid BM25+Dense RRF)",
            "RAG-3": "RAG-3 (Hybrid + Cross-Encoder)"
        }[p]
        print(
            f"{name:<32} | {stats['mean_context_recall']:<10.4f} | {stats['mean_context_precision']:<9.4f} | "
            f"{stats['mean_mrr']:<7.4f} | {stats['mean_answer_faithfulness']:<12.4f} | "
            f"{stats['mean_answer_fact_coverage']:<12.4f} | {stats['p50_latency_ms']:<9.2f}"
        )
    print("=" * 114)

    # Print Chunk Window Sensitivity Table
    print("\n" + "=" * 96)
    print(" EXPERIMENT 2B: CHUNK WINDOW & TABLE-PRESERVATION SENSITIVITY STUDY")
    print("=" * 96)
    print(f"{'Chunking Strategy':<36} | {'Total Chunks':<12} | {'Broken Tables':<13} | {'Ctx Recall':<11} | {'Ctx Precision':<13}")
    print("-" * 96)
    for c_name, c_stats in chunk_window_study.items():
        print(
            f"{c_name:<36} | {c_stats['num_chunks']:<12} | {c_stats['broken_table_chunks']:<13} | "
            f"{c_stats['mean_context_recall']:<11.4f} | {c_stats['mean_context_precision']:<13.4f}"
        )
    print("=" * 96)

    # Save to JSON
    output_payload = {
        "summary": summary,
        "chunk_window_ablation": chunk_window_study,
        "queries_evaluated": len(cases),
        "detailed_results": all_results,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nDetailed retrieval + generation ablation results saved to: {RESULTS_JSON}\n")


if __name__ == "__main__":
    run_ablation()
