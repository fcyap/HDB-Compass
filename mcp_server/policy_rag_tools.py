"""
FastMCP Tool Server: Advanced Regulatory Policy RAG (mcp_server/policy_rag_tools.py)
Proposal Section 4.3 (Page 9) & Section 6.2 (Page 11).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Category C).

Implements the official tool signature:
    retrieve_hdb_policy_knowledge(query: str, policy_domain: Optional[str] = None) -> Dict[str, Any]

Architecture:
1. Query Expansion / Domain Recognition:
   - Detects key acronyms (MOP, EIP, HFE, EHG, PHG, Prime, Plus, Standard)
   - Infers or verifies policy_domain ("classification", "eligibility", "grants", "financing", "eip")
2. Hybrid Retrieval via Reciprocal Rank Fusion (RRF):
   - BM25 sparse keyword search (exact acronym, number & policy phrase matching)
   - Dense semantic vector search (SentenceTransformer cosine similarity)
   - Fuses rank positions with standard RRF formula (k=60) to retrieve top-20 candidate chunks
3. Cross-Encoder Reranking:
   - Passes top-20 candidate pairs into a cross-encoder to compute query-passage relevance
   - Selects top-5 most relevant passages with normalized rerank_score
4. Supports Ablation Study Pipelines:
   - RAG-1: Naive Dense Vector Search (top-5)
   - RAG-2: Hybrid Search (BM25 + Dense via RRF, top-5)
   - RAG-3: Hybrid Search + Cross-Encoder Reranking (top-20 -> top-5)
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastmcp import FastMCP

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schemas import PolicyPassage, RetrieveHdbPolicyKnowledgeOutput

DATA_DIR = PROJECT_ROOT / "data"
STORE_PKL = DATA_DIR / "policy_store.pkl"

mcp = FastMCP("HDB-Policy-RAG-Tools")

# ---------------------------------------------------------------------------
# Module-level singletons (lazy loaded for maximum performance & zero latency)
# ---------------------------------------------------------------------------
_STORE: Optional[Dict[str, Any]] = None
_EMBED_MODEL: Any = None
_CROSS_ENCODER: Any = None


def _load_store() -> Dict[str, Any]:
    global _STORE
    if _STORE is None:
        if not STORE_PKL.exists():
            # Auto-ingest if store does not exist yet
            from data.ingest_hdb_policy_rag import ingest_policy_knowledge
            ingest_policy_knowledge()
        with open(STORE_PKL, "rb") as f:
            _STORE = pickle.load(f)
    return _STORE


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[Warning] SentenceTransformer load failed: {e}")
            _EMBED_MODEL = False
    return _EMBED_MODEL


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[Warning] CrossEncoder load failed ({e}); using hybrid reciprocal rank score.")
            _CROSS_ENCODER = False
    return _CROSS_ENCODER


# ---------------------------------------------------------------------------
# 1. Query Expansion & Domain Detection
# ---------------------------------------------------------------------------
ACRONYM_EXPANSIONS = {
    r"\bmop\b": "Minimum Occupation Period MOP",
    r"\beip\b": "Ethnic Integration Policy EIP quota",
    r"\bhfe\b": "HDB Flat Eligibility HFE letter",
    r"\behg\b": "Enhanced CPF Housing Grant EHG",
    r"\bphg\b": "Proximity Housing Grant PHG 4km",
    r"\bprime\b": "Prime location housing clawback subsidy recovery 10-year MOP",
    r"\bplus\b": "Plus location flat subsidy recovery 10-year MOP",
    r"\bstandard\b": "Standard flat 5-year MOP",
    r"\bclawback\b": "subsidy recovery clawback first resale",
    r"\bwait-?out\b": "15-month wait-out period private property senior 55 exemption 30-month",
    r"\bsingle[s]?\b": "Single Singapore Citizen Scheme 35 years old 2-room Flexi",
    r"\bright-?sizing\b": "right-sizing senior citizen 55 exempt 4-room",
}


def expand_query(query: str) -> str:
    """Expands query with policy acronyms and relevant synonyms."""
    expanded = query
    for pattern, replacement in ACRONYM_EXPANSIONS.items():
        if re.search(pattern, query, re.I):
            expanded += f" {replacement}"
    return expanded


def infer_domain(query: str, explicit_domain: Optional[str] = None) -> Optional[str]:
    """Infers policy domain if not explicitly provided."""
    if explicit_domain and explicit_domain.lower() in ("classification", "eligibility", "grants", "financing", "eip"):
        return explicit_domain.lower()
    q = query.lower()
    if any(k in q for k in ("prime", "plus", "standard", "clawback", "subsidy recovery")):
        return "classification"
    if any(k in q for k in ("grant", "ehg", "phg", "family grant", "housing grant", "stacking")):
        return "grants"
    if any(k in q for k in ("eip", "ethnic", "spr quota", "buyback", "racial")):
        return "eip"
    if any(k in q for k in ("msr", "ltv", "tdsr", "hfe", "loan", "downpayment", "stamp duty", "bsd")):
        return "financing"
    if any(k in q for k in ("single", "age 35", "wait-out", "wait out", "senior", "55", "eligibility", "scheme", "fiance", "condo")):
        return "eligibility"
    return None


# ---------------------------------------------------------------------------
# 2. Dense Vector Retrieval
# ---------------------------------------------------------------------------
def retrieve_dense(query: str, store: Dict[str, Any], top_k: int = 20) -> List[Tuple[int, float]]:
    """Cosine similarity over dense vector embeddings."""
    embed_model = _get_embed_model()
    chunks = store["chunks"]
    dense_matrix = store["dense_embeddings"]

    if embed_model:
        query_vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    else:
        # Fallback: token frequency cosine similarity
        words = set(re.findall(r"\b\w+\b", query.lower()))
        sims = []
        for i, c in enumerate(chunks):
            c_words = set(re.findall(r"\b\w+\b", c["text"].lower()))
            overlap = len(words & c_words) / max(1, len(words | c_words))
            sims.append(overlap)
        scores = np.array(sims, dtype=np.float32)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    scores = np.dot(dense_matrix, query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


# ---------------------------------------------------------------------------
# 3. BM25 Sparse Retrieval
# ---------------------------------------------------------------------------
def retrieve_bm25(query: str, store: Dict[str, Any], top_k: int = 20) -> List[Tuple[int, float]]:
    """BM25 sparse keyword search."""
    from data.ingest_hdb_policy_rag import tokenize_words
    bm25 = store["bm25"]
    tokens = tokenize_words(query)
    if not tokens:
        tokens = tokenize_words(expand_query(query))
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


# ---------------------------------------------------------------------------
# 4. Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    dense_results: List[Tuple[int, float]],
    bm25_results: List[Tuple[int, float]],
    k: int = 60,
    top_n: int = 20,
) -> List[Tuple[int, float]]:
    """
    Combines ranked lists using Reciprocal Rank Fusion:
        RRF_score(d) = sum(1 / (k + rank(d)))
    """
    rrf_scores: Dict[int, float] = {}

    for rank, (doc_idx, _) in enumerate(dense_results):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

    for rank, (doc_idx, _) in enumerate(bm25_results):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


# ---------------------------------------------------------------------------
# 5. Cross-Encoder Reranking
# ---------------------------------------------------------------------------
def rerank_cross_encoder(
    query: str,
    candidate_indices: List[Tuple[int, float]],
    store: Dict[str, Any],
    top_n: int = 5,
) -> List[Tuple[int, float]]:
    """
    Reranks top candidate chunks using a CrossEncoder model for deep semantic interaction.
    """
    cross_encoder = _get_cross_encoder()
    chunks = store["chunks"]

    if not candidate_indices:
        return []

    # If CrossEncoder is available
    if cross_encoder:
        pairs = [[query, chunks[idx]["text"]] for idx, _ in candidate_indices]
        try:
            scores = cross_encoder.predict(pairs)
            # Sigmoid normalization for cross-encoder logit scores
            norm_scores = 1.0 / (1.0 + np.exp(-np.array(scores, dtype=np.float32)))
            max_rrf = max((s for _, s in candidate_indices), default=1.0)
            reranked = []
            for i, (idx, rrf_s) in enumerate(candidate_indices):
                norm_rrf = rrf_s / max_rrf if max_rrf > 0 else 0.0
                blended = 0.65 * float(norm_scores[i]) + 0.35 * norm_rrf
                reranked.append((idx, round(blended, 4)))
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_n]
        except Exception as e:
            print(f"[Warning] CrossEncoder prediction failed ({e}); falling back to RRF scores.")

    # Fallback normalization over RRF scores
    max_score = max((s for _, s in candidate_indices), default=1.0)
    return [(idx, round(s / max_score, 4)) for idx, s in candidate_indices[:top_n]]


# ---------------------------------------------------------------------------
# 6. Core Retrieval Execution (RAG-1, RAG-2, RAG-3)
# ---------------------------------------------------------------------------
def execute_rag_pipeline(
    query: str,
    policy_domain: Optional[str] = None,
    pipeline: str = "RAG-3",
    top_k: int = 5,
) -> RetrieveHdbPolicyKnowledgeOutput:
    """Executes the selected RAG retrieval configuration."""
    store = _load_store()
    chunks = store["chunks"]

    domain = infer_domain(query, policy_domain)
    expanded = expand_query(query)

    # Configuration selection
    if pipeline == "RAG-1":
        # RAG-1: Naive Dense Vector Search (top-5 cosine similarity)
        dense_results = retrieve_dense(query, store, top_k=top_k)
        selected_candidates = [(idx, score) for idx, score in dense_results]

    elif pipeline == "RAG-2":
        # RAG-2: Hybrid Search (BM25 + Dense fused via RRF, top-5)
        dense_results = retrieve_dense(query, store, top_k=20)
        bm25_results = retrieve_bm25(expanded, store, top_k=20)
        rrf_results = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=top_k)
        max_score = max((s for _, s in rrf_results), default=1.0)
        selected_candidates = [(idx, round(s / max_score, 4)) for idx, s in rrf_results]

    else:
        # RAG-3: Hybrid Search (BM25 + Dense top-20) + Cross-Encoder Reranking to top-5
        pipeline = "RAG-3"
        dense_results = retrieve_dense(query, store, top_k=20)
        bm25_results = retrieve_bm25(expanded, store, top_k=20)
        rrf_candidates = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=20)
        selected_candidates = rerank_cross_encoder(query, rrf_candidates, store, top_n=top_k)

    # Domain re-weighting: if explicit domain match, slight prior boost
    if domain:
        boosted = []
        for idx, score in selected_candidates:
            c = chunks[idx]
            b_score = score * 1.15 if c.get("policy_category") == domain else score
            boosted.append((idx, min(1.0, b_score)))
        boosted.sort(key=lambda x: x[1], reverse=True)
        selected_candidates = boosted[:top_k]

    # Build typed PolicyPassage models
    passages: List[PolicyPassage] = []
    for idx, score in selected_candidates:
        c = chunks[idx]
        passages.append(PolicyPassage(
            text=c["text"],
            source_url=c.get("source_url", "https://www.hdb.gov.sg"),
            section=c.get("section", ""),
            policy_category=c.get("policy_category", ""),
            flat_classification=c.get("flat_classification"),
            effective_date=c.get("effective_date"),
            rerank_score=round(float(score), 4),
        ))

    return RetrieveHdbPolicyKnowledgeOutput(passages=passages, pipeline=pipeline)


# ---------------------------------------------------------------------------
# 7. FastMCP Tool Interface
# ---------------------------------------------------------------------------
@mcp.tool()
def retrieve_hdb_policy_knowledge(
    query: str,
    policy_domain: Optional[str] = None,
    pipeline: Optional[str] = "RAG-3",
) -> Dict[str, Any]:
    """
    Advanced Hybrid RAG Policy Knowledge Retrieval tool for official HDB and CPF policies.
    Combines BM25 sparse keyword search, dense vector embeddings, and cross-encoder reranking
    to provide high-precision, hallucination-resistant policy passages with source citations.
    """
    selected_pipeline = pipeline if pipeline in ("RAG-1", "RAG-2", "RAG-3") else "RAG-3"
    result = execute_rag_pipeline(
        query=query,
        policy_domain=policy_domain,
        pipeline=selected_pipeline,
        top_k=5,
    )
    return result.model_dump()


if __name__ == "__main__":
    test_queries = [
        "Prime vs Plus MOP and clawback percentage",
        "Single citizen 35 years old buying 2-room Flexi BTO",
        "Senior 58 right-sizing from private condo exemption from 15-month wait-out",
        "Enhanced Housing Grant income ceiling and maximum grant for family",
    ]
    for q in test_queries:
        print(f"\n==================== Query: {q} ====================")
        res = retrieve_hdb_policy_knowledge(q)
        print(f"Pipeline: {res['pipeline']}, Passages retrieved: {len(res['passages'])}")
        for i, p in enumerate(res["passages"][:2], 1):
            print(f"[{i}] score={p['rerank_score']} | section={p['section']} | url={p['source_url']}")
            print(f"    {p['text'][:140]}...")
