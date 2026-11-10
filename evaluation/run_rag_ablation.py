"""
RAG Retrieval & Generation Pipeline Ablation Study Runner (evaluation/run_rag_ablation.py)
Proposal Section 6.2 (Experiment 2, Page 11), Section 6.3 (Pages 11-12) & File Structure (Page 14).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Categories C & E).

Compares three retrieval + generation configurations across the 30 Golden Benchmark queries
(15 Category C Regulatory queries + 15 Category E Multi-Objective queries):
  1. RAG-1: Naive Dense Vector Search (top-5 cosine similarity over unexpanded body chunks)
  2. RAG-2: Hybrid Search (BM25 + dense, RRF k=60, top-20) + Cross-Encoder Reranking to top-5
  3. RAG-3: Query Rewriting + HyDE + Local Entity-Graph Expansion, then the same Cross-Encoder Reranking
  (--llm only) LONG-CTX: no retrieval; every policy document goes into the prompt (Topic 8 §2 baseline)
Both RAG-2 and RAG-3 rerank with the cross-encoder alone (config policy_rag.rerank_mode). Variants measured during
development and dropped from the default run to keep the comparison simple: a blended rerank score (Category E recall
0.55 vs 0.83) and per-sub-question decomposition (no gain over RAG-2); see the git history of this file.

Evaluates BOTH Retrieval and Generation (Agent 5 End-to-End) using formal RAGAS formulation:
  [Retrieval Metrics]
  - Context Recall (RAGAS): % of ground-truth gold policy clauses retrieved in top-5 (Target > 0.88)
  - Context Precision (RAGAS): Rank-weighted Mean Average Precision (MAP@5) over retrieved chunks (Target > 0.88)
  - Mean Reciprocal Rank (MRR): 1 / rank of first relevant passage
  [Generation Metrics — PolicyRagAgent (Agent 5)]
  - Answer Faithfulness (lexical): fraction of answer lines whose words appear in the retrieved context. It scores the
    extractive synthesize_grounded_answer, which copies passage lines, so it is ~1.0 by construction; it is a sanity
    check, not RAGAS faithfulness. Run with --llm for the RAGAS-style scores below.
  - [--llm] LLM-judge Faithfulness: the LLM answer users see is split into claims and a judge model marks each as
    supported or not by the context (RAGAS faithfulness); LLM-judge Answer Relevance (0-1); LLM Answer Fact Coverage
  - Answer Fact Coverage (Accuracy): Fraction of gold policy facts/clauses accurately synthesized in the final answer
  - Citation Compliance Rate: % of answers containing verified official inline URLs
  [Chunk Window Ablation (Experiment 2B)]
  - Fixed 256-token vs Hierarchical 512-token (Table-Preserving) vs Fixed 1024-token chunks
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.policy_agent import PolicyRagAgent
from data.ingest_hdb_policy_rag import DOCS_DIR, chunk_section, extract_frontmatter, split_markdown_into_hierarchical_sections
from mcp_server.policy_rag_tools import execute_rag_pipeline

PIPELINE_NAMES = {
    "RAG-1": "RAG-1 (Naive Dense Vector)",
    "RAG-2": "RAG-2 (Hybrid + Cross-Encoder)",
    "RAG-3": "RAG-3 (Rewrite + HyDE + Graph)",
    "LONG-CTX": "LONG-CTX (all docs, no RAG)",
}

CATEGORY_C_FILE = PROJECT_ROOT / "evaluation" / "category_c_queries.json"
CATEGORY_E_FILE = PROJECT_ROOT / "evaluation" / "category_e_queries.json"
RESULTS_JSON = PROJECT_ROOT / "evaluation" / "rag_ablation_results.json"

# Ground-truth policy expectations for Category E queries (Proposal §6.2, Page 11). The clauses were re-checked against
# the captured official pages (Nov 2026; every clause appears in data/policy_documents); target_sections now name the
# official pages that hold them (matched against each chunk's section breadcrumb, which starts with the page title).
CATEGORY_E_POLICY_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "E1-01": {
        "target_sections": [
            "Enhanced CPF Housing Grant for Families",
            "CPF Housing Grants for Families Buying Resale Flats",
            "Housing Loan from HDB",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["$9,000", "income ceiling"],
            ["Family Grant", "$80,000"],
            ["95", "years"],
            ["30%", "MSR"],
        ],
    },
    "E1-02": {
        "target_sections": [
            "CPF Housing Grants for Families Buying Resale Flats",
            "Enhanced CPF Housing Grant for Families",
            "Housing Loan from HDB",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["Family Grant", "$50,000"],
            ["$9,000", "income ceiling"],
            ["95", "years"],
            ["30%", "MSR"],
        ],
    },
    "E1-03": {
        "target_sections": [
            "Enhanced CPF Housing Grant for Families",
            "CPF Housing Grants for Families Buying Resale Flats",
            "Housing Loan from HDB",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["$9,000", "income ceiling"],
            ["Family Grant", "$80,000"],
            ["95", "years"],
            ["30%", "MSR"],
        ],
    },
    "E1-04": {
        "target_sections": [
            "Enhanced CPF Housing Grant for Families",
            "CPF Housing Grants for Families Buying Resale Flats",
            "Housing Loan from HDB",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["$9,000", "income ceiling"],
            ["Family Grant", "$80,000"],
            ["95", "years"],
            ["30%", "MSR"],
        ],
    },
    "E1-05": {
        "target_sections": [
            "Singles",
            "CPF Housing Grants for Families Buying Resale Flats",
            "Enhanced CPF Housing Grant for Families",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["Single Singapore Citizen", "35"],
            ["Singles Grant", "$40,000"],
            ["$4,500"],
            ["30%", "MSR"],
        ],
    },
    "E1-06": {
        "target_sections": [
            "CPF Housing Grants for Families Buying Resale Flats",
            "Enhanced CPF Housing Grant for Families",
            "Housing Loan from HDB",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["Family Grant", "$50,000"],
            ["$9,000", "income ceiling"],
            ["95", "years"],
            ["30%", "MSR"],
        ],
    },
    "E1-07": {
        "target_sections": [
            "CPF Housing Grants for Families Buying Resale Flats",
            "Enhanced CPF Housing Grant for Families",
            "CPF Housing Grant for Singles Buying Resale Flats",
            "Housing Loan from HDB",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["$70,000"],
            ["$9,000", "income ceiling"],
            ["95", "years"],
            ["30%", "MSR"],
        ],
    },
    "E1-08": {
        "target_sections": [
            "Enhanced CPF Housing Grant for Families",
            "CPF Housing Grants for Families Buying Resale Flats",
            "Housing Loan from HDB",
            "Credit to Finance a Flat Purchase",
        ],
        "key_clauses": [
            ["$9,000", "income ceiling"],
            ["Family Grant", "$80,000"],
            ["95", "years"],
            ["30%", "MSR"],
        ],
    },
    "E2-01": {
        "target_sections": [
            "Standard, Plus, and Prime Housing Framework",
            "Conditions After Buying a Resale Flat",
            "Housing Loan from HDB",
            "Finding a Resale Flat",
        ],
        "key_clauses": [
            ["standard"],
            ["mop"],
            ["resale"],
            ["remaining lease"],
        ],
    },
    "E2-02": {
        "target_sections": [
            "Standard, Plus, and Prime Housing Framework",
            "Conditions After Buying a Resale Flat",
            "Housing Loan from HDB",
            "Finding a Resale Flat",
        ],
        "key_clauses": [
            ["standard"],
            ["mop"],
            ["resale"],
            ["remaining lease"],
        ],
    },
    "E2-03": {
        "target_sections": [
            "Standard, Plus, and Prime Housing Framework",
            "Conditions After Buying a Resale Flat",
            "Housing Loan from HDB",
            "Finding a Resale Flat",
        ],
        "key_clauses": [
            ["standard"],
            ["mop"],
            ["resale"],
            ["remaining lease"],
        ],
    },
    "E2-04": {
        "target_sections": [
            "Standard, Plus, and Prime Housing Framework",
            "Conditions After Buying a Resale Flat",
            "Housing Loan from HDB",
            "Finding a Resale Flat",
        ],
        "key_clauses": [
            ["standard"],
            ["mop"],
            ["resale"],
            ["remaining lease"],
        ],
    },
    "E2-05": {
        "target_sections": [
            "Standard, Plus, and Prime Housing Framework",
            "Conditions After Buying a Resale Flat",
            "Housing Loan from HDB",
            "Finding a Resale Flat",
        ],
        "key_clauses": [
            ["standard"],
            ["mop"],
            ["resale"],
            ["remaining lease"],
        ],
    },
    "E2-06": {
        "target_sections": [
            "Standard, Plus, and Prime Housing Framework",
            "Conditions After Buying a Resale Flat",
            "Housing Loan from HDB",
            "Finding a Resale Flat",
        ],
        "key_clauses": [
            ["standard"],
            ["mop"],
            ["resale"],
            ["remaining lease"],
        ],
    },
    "E2-07": {
        "target_sections": [
            "Standard, Plus, and Prime Housing Framework",
            "Conditions After Buying a Resale Flat",
            "Housing Loan from HDB",
            "Finding a Resale Flat",
        ],
        "key_clauses": [
            ["standard"],
            ["mop"],
            ["resale"],
            ["remaining lease"],
        ],
    },
}


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
    Proposal §6.3 (Page 12) & RAGAS formulation: Fraction of generated claims supported directly by
    retrieved HDB policy context.
    Splits generated answer into factual claim statements (excluding headers/disclaimers)
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


JUDGE_SYSTEM = (
    "You grade answers from a Singapore HDB policy assistant. You are given CONTEXT, a QUESTION and an ANSWER. "
    "1. Split the ANSWER into its atomic factual claims (figures, conditions, eligibility rules, periods). Ignore "
    "citations, greetings and statements that something is not in the documents. 2. Mark each claim supported=true "
    "only if the CONTEXT states it (paraphrase is fine; a figure must match exactly). 3. Rate relevance from 0 to 1: "
    "how directly and completely the ANSWER addresses the QUESTION. Reply with JSON only: "
    '{"claims": [{"claim": "...", "supported": true}], "relevance": 0.0}'
)


def judge_llm():
    """(client, "profile:model") for the judge. LLM_JUDGE_PROFILE (default groq) names a profile from another model
    family than the generator, so the judge does not grade its own model's answers; when that profile cannot be built
    (no key), the active profile judges and the label says so."""
    from agents.llm import ModelConfigurationError, _configured_profile, active_profile, get_llm
    name = os.getenv("LLM_JUDGE_PROFILE", "groq").strip()
    model = (os.getenv("LLM_JUDGE_MODEL") or "").strip() or None   # e.g. openai/gpt-oss-120b; the app's model is untouched
    try:
        prof = _configured_profile(name)
        return (get_llm(profile=name, model=model, temperature=0, max_tokens=2000),
                f"{name}:{model or prof['model']}")
    except ModelConfigurationError as exc:
        print(f"    [judge] profile {name!r} unavailable ({exc}); the active profile judges")
        prof = active_profile()
        return get_llm(temperature=0, max_tokens=2000), f"{prof['name']}:{prof.get('model')} (same as generator)"


_JUDGE: Optional[Tuple[Any, str]] = None
_FALLBACK_JUDGE: Optional[Tuple[Any, str]] = None
# Groq's free tier allows 8,000 tokens per minute, so one request cannot be larger; the long-context baseline's
# judge request (the whole corpus) is judged by the generator's model instead, and the row records which judge it was.
JUDGE_MAX_PROMPT_TOKENS = int(os.getenv("LLM_JUDGE_MAX_PROMPT_TOKENS", "5500"))


def _fallback_judge() -> Tuple[Any, str]:
    global _FALLBACK_JUDGE
    if _FALLBACK_JUDGE is None:
        from agents.llm import active_profile, get_llm
        prof = active_profile()
        _FALLBACK_JUDGE = (get_llm(temperature=0, max_tokens=2000), f"{prof['name']}:{prof.get('model')} (same as generator)")
    return _FALLBACK_JUDGE


def llm_judge(question: str, answer: str, context: str) -> Dict[str, Optional[float]]:
    """RAGAS-style faithfulness (supported claims / claims) and answer relevance from a judge model.
    None when no model is configured or its reply cannot be read, so a failed judge never scores as 0 or 1."""
    global _JUDGE
    from langchain_core.messages import HumanMessage, SystemMessage
    if _JUDGE is None:
        _JUDGE = judge_llm()
    llm, label = _JUDGE
    if (len(context) + len(answer) + len(question)) // 4 > JUDGE_MAX_PROMPT_TOKENS:
        llm, label = _fallback_judge()
    if llm is None or not answer.strip():
        return {"llm_faithfulness": None, "llm_answer_relevance": None, "claims": None, "judge": None}
    try:
        reply = llm.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(
            content=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}")]).content
        if isinstance(reply, list):
            reply = " ".join(str(x.get("text", "")) for x in reply if isinstance(x, dict))
        data = json.loads(re.search(r"\{.*\}", str(reply), re.S).group(0))
        claims = data.get("claims") or []
        supported = sum(1 for c in claims if c.get("supported") is True)
        relevance = data.get("relevance")
        return {"llm_faithfulness": round(supported / len(claims), 4) if claims else None,
                "llm_answer_relevance": round(float(relevance), 4) if relevance is not None else None,
                "claims": len(claims), "judge": label}
    except Exception as exc:  # noqa: BLE001
        print(f"    [judge] {label}: unreadable reply ({type(exc).__name__}: {str(exc)[:120]}); scored as missing")
        return {"llm_faithfulness": None, "llm_answer_relevance": None, "claims": None, "judge": label}


def full_corpus_context() -> str:
    """Every policy document, as the long-context baseline sends it (frontmatter keeps each source URL)."""
    return "\n\n".join(f"=== {f.name} ===\n{f.read_text(encoding='utf-8')}" for f in sorted(DOCS_DIR.glob("*.md")))


def answer_metrics(test_case: Dict[str, Any], answer: str, context: str) -> Dict[str, Any]:
    """Generation metrics for an LLM answer: judge faithfulness/relevance plus gold-clause coverage."""
    raw_clauses = test_case.get("ground_truth_facts", {}).get("key_clauses", [])
    ans_clean = clean_text(answer)
    matched = sum(1 for clause in raw_clauses if is_clause_in_passage(clause, ans_clean))
    judged = llm_judge(test_case["query"], answer, context)
    return {
        **judged,
        "llm_answer_fact_coverage": round(matched / max(1, len(raw_clauses)), 4),
        "llm_citation_compliance": 1.0 if re.search(r"https://www\.(hdb|cpf)\.gov\.sg", answer) else 0.0,
    }


def evaluate_long_context(test_case: Dict[str, Any], corpus: str) -> Dict[str, Any]:
    """LONG-CTX baseline: no retrieval, the whole corpus in the prompt. Retrieval metrics do not apply (None)."""
    t0 = time.perf_counter()
    answer = PolicyRagAgent.generate_llm_answer(test_case["query"], [], context_override=corpus)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "id": test_case["id"], "category": test_case.get("category", "C"),
        "sub_use_case": test_case.get("sub_use_case", ""), "pipeline": "LONG-CTX",
        "latency_ms": round(latency_ms, 2),
        "context_recall": None, "context_precision": None, "mrr": None,
        "answer_faithfulness": None, "answer_fact_coverage": None, "citation_compliance": None,
        "context_tokens_approx": int(len(corpus.split()) * 1.3),
        **answer_metrics(test_case, answer, corpus),
    }


def evaluate_query_end_to_end(
    test_case: Dict[str, Any],
    pipeline: str,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Runs a single query under a specific RAG pipeline and computes both Retrieval & Generation metrics."""
    query = test_case["query"]
    gt = test_case.get("ground_truth_facts", {})
    raw_clauses = gt.get("key_clauses", [])
    target_sections = [clean_text(s) for s in gt.get("target_sections", [])]

    t0 = time.perf_counter()
    passages = execute_rag_pipeline(query=query, pipeline=pipeline, top_k=5).passages
    passage_dicts = [p.model_dump() for p in passages]
    answer = PolicyRagAgent.synthesize_grounded_answer(query, passage_dicts)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Pre-clean passages once
    cleaned_passages = [
        (clean_text(p.text), clean_text(p.section))
        for p in passages
    ]

    # 1. Context Recall (RAGAS formulation): fraction of gold clauses captured in top-5 retrieved passages
    matched_clauses = 0
    for clause in raw_clauses:
        if any(is_clause_in_passage(clause, p_txt, p_sec) for p_txt, p_sec in cleaned_passages):
            matched_clauses += 1
    context_recall = matched_clauses / max(1, len(raw_clauses))

    # 2. Context Precision (RAGAS rank-weighted MAP@5 formula)
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

    # 4. Answer Faithfulness (Generation — RAGAS formulation): fraction of generated claims grounded in retrieved context
    answer_faithfulness = evaluate_answer_faithfulness(answer, cleaned_passages)

    # 5. Answer Fact Coverage / Accuracy (Generation): fraction of gold clauses present in synthesized answer
    ans_clean = clean_text(answer)
    ans_matched_clauses = sum(1 for clause in raw_clauses if is_clause_in_passage(clause, ans_clean))
    answer_fact_coverage = ans_matched_clauses / max(1, len(raw_clauses))

    # 6. Citation Compliance (Generation): verifies inline official HDB/CPF URL citation
    has_valid_citation = 1.0 if re.search(r"https://www\.(hdb|cpf)\.gov\.sg", answer) else 0.0

    extra: Dict[str, Any] = {}
    if use_llm:
        llm_answer = PolicyRagAgent.generate_llm_answer(query, passage_dicts)
        context = "\n\n".join(f"(Source: {p['source_url']}) {p['text']}" for p in passage_dicts)
        extra.update(answer_metrics(test_case, llm_answer, context))

    return {
        **extra,
        "id": test_case["id"],
        "category": test_case.get("category", "C"),
        "sub_use_case": test_case.get("sub_use_case", ""),
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

    mean_words = round(statistics.mean(len(c["text"].split()) for c in hier_chunks))
    configs = {
        "Fixed-256 Tokens (Naive)": build_fixed_chunks(95, 15),
        # the semantic chunker caps at 380 words but cuts at every low-similarity sentence boundary, so its chunks
        # are far smaller than the cap; the label states the real mean size instead of "512 tokens"
        f"Semantic, current store (~{mean_words} words)": (hier_chunks, 0),
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

            raw_clauses = case.get("ground_truth_facts", {}).get("key_clauses", [])
            target_secs = [clean_text(s) for s in case.get("ground_truth_facts", {}).get("target_sections", [])]

            matched = sum(
                1 for cl in raw_clauses
                if any(is_clause_in_passage(cl, p_t, p_s) for p_t, p_s in top_passages)
            )
            recalls.append(matched / max(1, len(raw_clauses)))

            rel_vec = []
            for p_t, p_s in top_passages:
                # For coarse 1024-token chunks, require focused relevance
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


def _mean(results: List[Dict[str, Any]], key: str) -> Optional[float]:
    """Mean over the rows that have the metric (None when none do, e.g. retrieval metrics for LONG-CTX)."""
    values = [r[key] for r in results if r.get(key) is not None]
    return round(statistics.mean(values), 4) if values else None


def evaluate_chunk_threshold_sweep(cases: List[Dict[str, Any]],
                                   thresholds: Tuple[float, ...] = (0.25, 0.35, 0.45, 0.55),
                                   max_words: int = 380) -> Dict[str, Dict[str, Any]]:
    """Topic 8 "chunk size has a sweet spot": rebuild the semantic chunks at several similarity thresholds (a lower
    threshold cuts less often, so chunks get bigger) and score each through the real RAG-2 path (BM25 + dense + RRF +
    cross-encoder rerank, top-5), not BM25 alone. No model calls."""
    from rank_bm25 import BM25Okapi
    from data.ingest_hdb_policy_rag import build_policy_corpus, compute_dense_embeddings, tokenize_words
    from mcp_server.policy_rag_tools import reciprocal_rank_fusion, rerank_cross_encoder, retrieve_bm25, retrieve_dense

    sweep: Dict[str, Dict[str, Any]] = {}
    for threshold in thresholds:
        chunks = build_policy_corpus(DOCS_DIR, threshold=threshold, max_words=max_words)
        store = {"chunks": chunks, "bm25": BM25Okapi([tokenize_words(c["text"]) for c in chunks]),
                 "dense_embeddings": compute_dense_embeddings(chunks)}
        recalls, precisions = [], []
        for case in cases:
            q = case["query"]
            fused = reciprocal_rank_fusion(retrieve_dense(q, store, top_k=20), retrieve_bm25(q, store, top_k=20), k=60, top_n=20)
            top = [chunks[i] for i, _ in rerank_cross_encoder(q, fused, store, top_n=5)]
            cleaned = [(clean_text(c["text"]), clean_text(c.get("section", ""))) for c in top]
            gt = case.get("ground_truth_facts", {})
            clauses = gt.get("key_clauses", [])
            sections = [clean_text(x) for x in gt.get("target_sections", [])]
            recalls.append(sum(1 for cl in clauses if any(is_clause_in_passage(cl, t, sec) for t, sec in cleaned))
                           / max(1, len(clauses)))
            rel = [1 if (any(is_clause_in_passage(cl, t, sec) for cl in clauses) or any(x in sec for x in sections)) else 0
                   for t, sec in cleaned]
            hits, cum = 0, []
            for k, r in enumerate(rel, 1):
                if r:
                    hits += 1
                    cum.append(hits / k)
            precisions.append(sum(cum) / len(cum) if cum else 0.0)
        sweep[f"threshold {threshold:.2f}, cap {max_words}"] = {
            "num_chunks": len(chunks),
            "mean_words": round(statistics.mean(len(c["text"].split()) for c in chunks), 1),
            "mean_context_recall": round(statistics.mean(recalls), 4),
            "mean_context_precision": round(statistics.mean(precisions), 4),
        }
    return sweep


def compute_group_summary(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    latencies_sorted = sorted(r["latency_ms"] for r in results)
    p50_idx = int(0.50 * len(latencies_sorted))
    p95_idx = int(0.95 * len(latencies_sorted))

    summary: Dict[str, Optional[float]] = {
        "mean_context_recall": _mean(results, "context_recall"),
        "mean_context_precision": _mean(results, "context_precision"),
        "mean_mrr": _mean(results, "mrr"),
        "mean_answer_faithfulness": _mean(results, "answer_faithfulness"),
        "mean_answer_fact_coverage": _mean(results, "answer_fact_coverage"),
        "citation_compliance_rate": _mean(results, "citation_compliance"),
        "p50_latency_ms": round(latencies_sorted[p50_idx], 2),
        "p95_latency_ms": round(latencies_sorted[p95_idx], 2),
    }
    for key in ("llm_faithfulness",
                "llm_answer_relevance", "llm_answer_fact_coverage", "llm_citation_compliance"):
        if any(r.get(key) is not None for r in results):
            summary[f"mean_{key}"] = _mean(results, key)
    return summary


def _fmt(value: Optional[float], width: int) -> str:
    return f"{value:<{width}.4f}" if value is not None else f"{'n/a':<{width}}"


def print_table(title: str, summary: Dict[str, Dict[str, Optional[float]]]) -> None:
    llm = any("mean_llm_faithfulness" in s for s in summary.values())
    width = 160 if llm else 130
    print("\n" + "=" * width)
    print(f" {title} ")
    print("=" * width)
    header = (
        f"{'Pipeline Configuration':<32} | {'Ctx Recall':<10} | {'Ctx Prec':<9} | {'MRR':<7} | "
        f"{'Lex Faith':<9} | {'Ans Fact Acc':<12} | {'P50 (ms)':<9}"
    )
    if llm:
        header += f" | {'LLM Faith':<9} | {'Relevance':<9} | {'LLM Facts':<9}"
    print(header)
    print("-" * width)
    for p, stats in summary.items():
        row = (
            f"{PIPELINE_NAMES.get(p, p):<32} | {_fmt(stats['mean_context_recall'], 10)} | "
            f"{_fmt(stats['mean_context_precision'], 9)} | {_fmt(stats['mean_mrr'], 7)} | "
            f"{_fmt(stats['mean_answer_faithfulness'], 9)} | {_fmt(stats['mean_answer_fact_coverage'], 12)} | "
            f"{stats['p50_latency_ms']:<9.2f}"
        )
        if llm:
            row += (f" | {_fmt(stats.get('mean_llm_faithfulness'), 9)} | {_fmt(stats.get('mean_llm_answer_relevance'), 9)}"
                    f" | {_fmt(stats.get('mean_llm_answer_fact_coverage'), 9)}")
        print(row)
    print("=" * width)


def run_ablation(use_llm: bool = False, limit: Optional[int] = None):
    print("=" * 96)
    print(" HDB-Compass: Experiment 2 — End-to-End RAG Retrieval & Generation Ablation Study ")
    print(" Owned by Member 3 (Advanced RAG & Regulatory Knowledge Base — Categories C & E) ")
    print(" Framework: Formal RAGAS Formulation (Context Recall, Context Precision MAP@5, Faithfulness) ")
    print("=" * 96)

    # 1. Build and load Category C golden queries (15 queries)
    from evaluation.build_category_c import build
    build()

    with open(CATEGORY_C_FILE, "r", encoding="utf-8") as f:
        cat_c_cases = json.load(f)
    # a question no captured page answers has no passages to retrieve; it tests the "not in the documents" rule of the
    # answer instead, so it is left out of the retrieval metrics
    unanswerable = [c["id"] for c in cat_c_cases if not c["ground_truth_facts"].get("answerable", True)]
    cat_c_cases = [c for c in cat_c_cases if c["ground_truth_facts"].get("answerable", True)]
    if unanswerable:
        print(f"Category C questions with no answer in the captured pages (left out of retrieval metrics): {unanswerable}")

    # 2. Load Category E golden queries (15 queries) and enrich with policy ground truth
    with open(CATEGORY_E_FILE, "r", encoding="utf-8") as f:
        cat_e_raw = json.load(f)

    cat_e_cases = []
    for raw in cat_e_raw:
        case = dict(raw)
        qid = case["id"]
        # Attach statutory policy ground-truth expectation for Experiment 2
        case["ground_truth_facts"] = CATEGORY_E_POLICY_GROUND_TRUTH.get(qid, {
            "target_sections": ["Conditions After Buying a Resale Flat", "Housing Loan from HDB"],
            "key_clauses": [["5-year MOP"], ["30%", "MSR"], ["75%", "LTV"]],
        })
        cat_e_cases.append(case)

    if limit:
        # a sample of each category (the --llm run makes two model calls per query and pipeline)
        cat_c_cases, cat_e_cases = cat_c_cases[:limit], cat_e_cases[:limit]
    all_cases = cat_c_cases + cat_e_cases

    # Warm up models once before timing P50/P95
    execute_rag_pipeline(query="Warmup query MOP Prime", pipeline="RAG-3", top_k=5)
    execute_rag_pipeline(query="Warmup query MOP Prime", pipeline="RAG-1", top_k=5)

    pipelines = ["RAG-1", "RAG-2", "RAG-3"] + (["LONG-CTX"] if use_llm else [])
    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pipelines}
    corpus = full_corpus_context() if use_llm else ""

    print(f"\nBenchmarking {len(cat_c_cases)} Category C + {len(cat_e_cases)} Category E = {len(all_cases)} total queries across {pipelines}\n")

    for p in pipelines:
        print(f"--> Running Configuration: {p} (Retrieval + Agent 5 Synthesis) ...")
        for case in all_cases:
            res = (evaluate_long_context(case, corpus) if p == "LONG-CTX"
                   else evaluate_query_end_to_end(case, pipeline=p, use_llm=use_llm))
            all_results[p].append(res)
        print(f"    Completed {len(all_cases)} queries for {p}.")

    # Summary Statistics Aggregation by Group
    cat_c_summary: Dict[str, Dict[str, float]] = {}
    cat_e_summary: Dict[str, Dict[str, float]] = {}
    combined_summary: Dict[str, Dict[str, float]] = {}

    for p in pipelines:
        cat_c_res = [r for r in all_results[p] if r["category"] == "C"]
        cat_e_res = [r for r in all_results[p] if r["category"] == "E"]

        cat_c_summary[p] = compute_group_summary(cat_c_res)
        cat_e_summary[p] = compute_group_summary(cat_e_res)
        combined_summary[p] = compute_group_summary(all_results[p])

    # Chunk Window Sensitivity Study over all 30 benchmark queries
    chunk_window_study = evaluate_chunk_window_ablation(all_cases)

    # Print Breakdown Tables
    print_table(f"EXPERIMENT 2A: RETRIEVAL & GENERATION ABLATION — Category C: Regulatory Rules (N={len(cat_c_cases)})", cat_c_summary)
    print_table(f"EXPERIMENT 2A: RETRIEVAL & GENERATION ABLATION — Category E: Multi-Objective Rules (N={len(cat_e_cases)})", cat_e_summary)
    print_table(f"EXPERIMENT 2A: RETRIEVAL & GENERATION ABLATION — Combined Benchmark (Categories C + E, N={len(all_cases)})", combined_summary)

    # Print Chunk Window Sensitivity Table
    print("\n" + "=" * 96)
    print(" EXPERIMENT 2B: CHUNK WINDOW & TABLE-PRESERVATION SENSITIVITY STUDY (N=30)")
    print("=" * 96)
    print(f"{'Chunking Strategy':<36} | {'Total Chunks':<12} | {'Broken Tables':<13} | {'Ctx Recall':<11} | {'Ctx Precision':<13}")
    print("-" * 96)
    for c_name, c_stats in chunk_window_study.items():
        print(
            f"{c_name:<36} | {c_stats['num_chunks']:<12} | {c_stats['broken_table_chunks']:<13} | "
            f"{c_stats['mean_context_recall']:<11.4f} | {c_stats['mean_context_precision']:<13.4f}"
        )
    print("=" * 96)

    threshold_sweep = evaluate_chunk_threshold_sweep(all_cases, thresholds=(0.15, 0.25, 0.35, 0.45, 0.55))
    print("\n" + "=" * 96)
    print(f" EXPERIMENT 2C: SEMANTIC-CHUNK THRESHOLD SWEEP THROUGH RAG-2 (N={len(all_cases)})")
    print("=" * 96)
    print(f"{'Similarity threshold':<24} | {'Chunks':<7} | {'Mean words':<10} | {'Ctx Recall':<11} | {'Ctx Precision':<13}")
    print("-" * 96)
    for name, row in threshold_sweep.items():
        print(f"{name:<24} | {row['num_chunks']:<7} | {row['mean_words']:<10} | "
              f"{row['mean_context_recall']:<11.4f} | {row['mean_context_precision']:<13.4f}")
    print("=" * 96)

    # Save to JSON (backward-compatible summary key + group breakdowns)
    output_payload = {
        "summary": combined_summary,
        "category_c_summary": cat_c_summary,
        "category_e_summary": cat_e_summary,
        "combined_summary": combined_summary,
        "chunk_window_ablation": chunk_window_study,
        "chunk_threshold_sweep": threshold_sweep,
        "unanswerable_category_c": unanswerable,
        "queries_evaluated": len(all_cases),
        "llm_mode": use_llm,
        "judge_model": _JUDGE[1] if _JUDGE else None,
        "detailed_results": all_results,
    }
    # a sampled or LLM-judged run gets its own file, so it never overwrites the full deterministic results
    suffix = ("_llm" if use_llm else "") + (f"_limit{limit}" if limit else "")
    out_path = RESULTS_JSON.with_name(RESULTS_JSON.stem + suffix + RESULTS_JSON.suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nDetailed retrieval + generation ablation results saved to: {out_path}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 2: RAG retrieval & generation ablation")
    parser.add_argument("--llm", action="store_true",
                        help="also score the LLM answers with an LLM judge and add the LONG-CTX baseline (model calls)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries of each category")
    args = parser.parse_args()
    run_ablation(use_llm=args.llm, limit=args.limit)
