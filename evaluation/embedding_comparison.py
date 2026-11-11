"""Experiment 2C: which sentence-embedding model for the policy RAG (Member 3).

Runs as part of evaluation/run_rag_ablation.py (2A-2E); this module can also be run on its own.

Three embedders on the same chunks (data/policy_store.pkl) and the same 30 Category C + E questions as Experiment 2:
  - sentence-transformers/all-MiniLM-L6-v2 (current; 22M parameters, 384 dimensions)
  - BAAI/bge-small-en-v1.5 (33M, 384)
  - BAAI/bge-base-en-v1.5 (110M, 768)
BGE models are trained with an instruction prefix on the query side, so queries get it and passages do not.

Each embedder is scored twice, top-5, with the gold clauses of the fact sheet (same formulas as run_rag_ablation):
  - dense only: cosine search alone, which isolates the embedder;
  - RAG-2: BM25 + dense for the question and its expansion, RRF, cross-encoder rerank: what the app runs.
Costs: time to embed the corpus, median query-embedding time (CPU), model size. No LLM calls.

    .venv/bin/python -m evaluation.embedding_comparison
Prints the table; the full run (run_rag_ablation) saves the results in evaluation/rag_ablation_results.json.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from evaluation.run_rag_ablation import clean_text, is_clause_in_passage, load_benchmark_cases
from mcp_server.policy_rag_tools import (_load_store, expand_query, reciprocal_rank_fusion, rerank_cross_encoder,
                                         retrieve_bm25)

MODELS: List[Dict[str, Any]] = [
    {"name": "all-MiniLM-L6-v2", "hf": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "", "params_m": 22},
    {"name": "bge-small-en-v1.5", "hf": "BAAI/bge-small-en-v1.5",
     "query_prefix": "Represent this sentence for searching relevant passages: ", "params_m": 33},
    {"name": "bge-base-en-v1.5", "hf": "BAAI/bge-base-en-v1.5",
     "query_prefix": "Represent this sentence for searching relevant passages: ", "params_m": 110},
]
TOP_K = 5


def score(top: List[Dict[str, Any]], case: Dict[str, Any]) -> Tuple[float, float, float]:
    """(context recall, context precision MAP@5, MRR) of the top passages against the case's gold clauses."""
    gt = case.get("ground_truth_facts", {})
    clauses = gt.get("key_clauses", [])
    sections = [clean_text(x) for x in gt.get("target_sections", [])]
    cleaned = [(clean_text(c["text"]), clean_text(c.get("section", ""))) for c in top]
    recall = sum(1 for cl in clauses if any(is_clause_in_passage(cl, t, s) for t, s in cleaned)) / max(1, len(clauses))
    rel = [1 if (any(is_clause_in_passage(cl, t, s) for cl in clauses) or any(x in s for x in sections)) else 0
           for t, s in cleaned]
    hits, cum = 0, []
    for k, r in enumerate(rel, 1):
        if r:
            hits += 1
            cum.append(hits / k)
    precision = sum(cum) / len(cum) if cum else 0.0
    mrr = next((1.0 / k for k, r in enumerate(rel, 1) if r), 0.0)
    return recall, precision, mrr


def evaluate(model_cfg: Dict[str, Any], store: Dict[str, Any], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    chunks = store["chunks"]
    model = SentenceTransformer(model_cfg["hf"])
    t0 = time.perf_counter()
    matrix = model.encode([c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True,
                          batch_size=32).astype(np.float32)
    corpus_s = time.perf_counter() - t0

    query_times: List[float] = []

    def dense(query: str, top_k: int) -> List[Tuple[int, float]]:
        t = time.perf_counter()
        vec = model.encode([model_cfg["query_prefix"] + query], convert_to_numpy=True, normalize_embeddings=True)[0]
        query_times.append((time.perf_counter() - t) * 1000)
        sims = matrix @ vec
        order = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in order]

    rows = []
    for case in cases:
        q = case["query"]
        d = score([chunks[i] for i, _ in dense(q, TOP_K)], case)
        fused: Dict[int, float] = {}
        for sq in dict.fromkeys((q, expand_query(q))):          # the same candidate pool as RAG-2
            for idx, s in reciprocal_rank_fusion(dense(sq, 20), retrieve_bm25(sq, store, top_k=20), k=60, top_n=20):
                fused[idx] = fused.get(idx, 0.0) + s
        pool = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:20]
        top = [chunks[i] for i, _ in rerank_cross_encoder(q, pool, store, top_n=TOP_K, rerank_mode="ce_only")]
        r = score(top, case)
        rows.append({"id": case["id"], "category": case["category"],
                     "dense_recall": d[0], "dense_precision": d[1], "dense_mrr": d[2],
                     "rag2_recall": r[0], "rag2_precision": r[1], "rag2_mrr": r[2]})

    def mean(key: str, cat: str = "") -> float:
        return round(statistics.mean(x[key] for x in rows if not cat or x["category"] == cat), 4)

    return {
        "model": model_cfg["hf"], "params_millions": model_cfg["params_m"],
        "dimensions": int(matrix.shape[1]),
        "corpus_embed_seconds": round(corpus_s, 1),
        "query_embed_ms_p50": round(statistics.median(query_times), 1),
        **{f"{k}": mean(k) for k in ("dense_recall", "dense_precision", "dense_mrr",
                                     "rag2_recall", "rag2_precision", "rag2_mrr")},
        "dense_recall_c": mean("dense_recall", "C"), "dense_recall_e": mean("dense_recall", "E"),
        "rag2_recall_c": mean("rag2_recall", "C"), "rag2_recall_e": mean("rag2_recall", "E"),
        "per_question": rows,
    }


def print_table(results: Dict[str, Dict[str, Any]], num_chunks: int, num_questions: int) -> None:
    print("\n" + "=" * 132)
    print(f" EXPERIMENT 2C: EMBEDDING MODEL ({num_chunks} chunks, N={num_questions}, top-{TOP_K})")
    print("=" * 132)
    print(f"{'Model':<20} | {'Params':<6} | {'Dim':<4} | {'Dense R':<7} | {'Dense P':<7} | {'Dense MRR':<9} | "
          f"{'RAG-2 R':<7} | {'RAG-2 P':<7} | {'RAG-2 MRR':<9} | {'Corpus s':<8} | {'Query ms':<8}")
    print("-" * 132)
    for name, r in results.items():
        print(f"{name:<20} | {r['params_millions']:<6} | {r['dimensions']:<4} | {r['dense_recall']:<7.4f} | "
              f"{r['dense_precision']:<7.4f} | {r['dense_mrr']:<9.4f} | {r['rag2_recall']:<7.4f} | "
              f"{r['rag2_precision']:<7.4f} | {r['rag2_mrr']:<9.4f} | {r['corpus_embed_seconds']:<8} | "
              f"{r['query_embed_ms_p50']:<8}")
    print("=" * 132)


def main() -> None:
    c, e, _ = load_benchmark_cases()
    cases = c + e
    store = _load_store()
    results = {}
    for cfg in MODELS:
        print(f"--> {cfg['name']} ...")
        results[cfg["name"]] = evaluate(cfg, store, cases)
    print_table(results, len(store["chunks"]), len(cases))


if __name__ == "__main__":
    main()
