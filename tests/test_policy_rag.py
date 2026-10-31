"""
Test Suite for Member 3: Advanced RAG & Regulatory Knowledge Base (tests/test_policy_rag.py).
Tests corpus ingestion, BM25 keyword matching, dense vector retrieval,
RRF fusion, cross-encoder reranking, and FastMCP tool schema compliance.

Run:
    .venv/bin/pytest tests/test_policy_rag.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schemas import RetrieveHdbPolicyKnowledgeOutput
from agents.tool_registry import ToolRunner
from mcp_server.policy_rag_tools import (
    _load_store,
    expand_query,
    infer_domain,
    retrieve_dense,
    retrieve_bm25,
    reciprocal_rank_fusion,
    rerank_cross_encoder,
    execute_rag_pipeline,
    retrieve_hdb_policy_knowledge,
)
from agents.policy_agent import PolicyRagAgent


def test_policy_store_contains_expected_chunks():
    store = _load_store()
    assert "chunks" in store
    assert "bm25" in store
    assert "dense_embeddings" in store
    chunks = store["chunks"]
    assert len(chunks) >= 40, f"Expected at least 40 chunks, got {len(chunks)}"

    categories = {c["policy_category"] for c in chunks}
    assert "classification" in categories
    assert "eligibility" in categories
    assert "grants" in categories
    assert "financing" in categories
    assert "eip" in categories


def test_query_expansion_and_domain_inference():
    q = "What is the clawback and MOP for Prime BTO?"
    expanded = expand_query(q)
    assert "Minimum Occupation Period" in expanded or "clawback" in expanded
    assert infer_domain(q) == "classification"

    q_grant = "How much EHG can we get with $7000 income?"
    assert infer_domain(q_grant) == "grants"

    q_eip = "Can an Indian family buy if block EIP quota is full?"
    assert infer_domain(q_eip) == "eip"

    q_senior = "Senior 58 right sizing from private condo"
    assert infer_domain(q_senior) == "eligibility"


def test_bm25_retrieves_exact_acronyms():
    store = _load_store()
    # Test MOP acronym
    res = retrieve_bm25("MOP 10-year clawback", store, top_k=5)
    assert len(res) == 5
    top_chunk = store["chunks"][res[0][0]]
    assert "MOP" in top_chunk["text"] or "clawback" in top_chunk["text"].lower()


def test_dense_retrieval_returns_high_similarity():
    store = _load_store()
    res = retrieve_dense("singles buying resale flats at age 35", store, top_k=5)
    assert len(res) == 5
    top_chunk = store["chunks"][res[0][0]]
    assert "35" in top_chunk["text"] or "single" in top_chunk["text"].lower()


def test_reciprocal_rank_fusion_logic():
    dense_mock = [(1, 0.9), (2, 0.8), (3, 0.7)]
    bm25_mock = [(2, 15.0), (4, 12.0), (1, 10.0)]
    fused = reciprocal_rank_fusion(dense_mock, bm25_mock, k=60, top_n=3)
    # Document 2 is ranked #2 in dense and #1 in BM25 -> should be top ranked in RRF
    top_doc_idx = fused[0][0]
    assert top_doc_idx == 2


def test_cross_encoder_reranks_and_blends_scores():
    store = _load_store()
    query = "Senior citizen age 55 exempt from 15-month wait-out"
    dense = retrieve_dense(query, store, top_k=10)
    bm25 = retrieve_bm25(query, store, top_k=10)
    rrf = reciprocal_rank_fusion(dense, bm25, k=60, top_n=10)
    reranked = rerank_cross_encoder(query, rrf, store, top_n=5)

    assert len(reranked) == 5
    assert all(0.0 <= score <= 1.0 for _, score in reranked)
    top_chunk = store["chunks"][reranked[0][0]]
    assert "55" in top_chunk["text"] or "exempt" in top_chunk["text"].lower()


@pytest.mark.parametrize("pipeline", ["RAG-1", "RAG-2", "RAG-3"])
def test_all_rag_pipelines_return_valid_schema(pipeline):
    res = execute_rag_pipeline(
        query="Prime vs Plus flat clawback and rental restrictions",
        pipeline=pipeline,
        top_k=5
    )
    assert isinstance(res, RetrieveHdbPolicyKnowledgeOutput)
    assert res.pipeline == pipeline
    assert len(res.passages) == 5
    for p in res.passages:
        assert p.text
        assert p.source_url.startswith("http")
        assert p.rerank_score >= 0.0


def test_mcp_tool_runner_and_contract():
    runner = ToolRunner("retrieve_hdb_policy_knowledge")
    out = runner(query="EHG grant table for first-timer family", policy_domain="grants")
    assert "error" not in out
    validated = RetrieveHdbPolicyKnowledgeOutput(**out)
    assert len(validated.passages) == 5
    assert runner.log[-1].ok is True
    assert runner.log[-1].tool == "retrieve_hdb_policy_knowledge"


def test_policy_agent_standalone_execution():
    agent = PolicyRagAgent(pipeline="RAG-3")
    res = agent.run("What are the differences between Standard and Plus flats?")
    assert "passages" in res
    assert "answer" in res
    assert len(res["passages"]) == 5
    assert len(res["answer"]) > 50
