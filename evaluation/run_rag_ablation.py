"""
Experiment 2: the policy RAG, from chunking to generation (evaluation/run_rag_ablation.py).
Proposal Section 6.2 (Experiment 2, Page 11), Section 6.3 (Pages 11-12) & File Structure (Page 14).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Categories C & E).

All parts use the same 30 benchmark questions (15 Category C regulatory + 15 Category E multi-objective) and score
retrieval against exact clauses from the verified fact sheet (evaluation/policy_fact_sheet.yaml):
  2A  Chunking strategy: fixed 256-token vs semantic table-preserving (the store) vs fixed 1024-token chunks, BM25
      top-5. Context recall, context precision, broken tables.
  2B  Chunk similarity threshold: the semantic chunker rebuilt at 0.15-0.55, scored through RAG-2.
  2C  Embedding model: all-MiniLM-L6-v2 vs bge-small-en-v1.5 vs bge-base-en-v1.5, dense-only and inside RAG-2
      (evaluation/embedding_comparison.py).
  2D  Retrieval pipeline, top-5:
        RAG-1  naive dense search (cosine over unexpanded chunk bodies)
        RAG-2  BM25 + dense for the question and its expansion, RRF k=60 to 20 candidates, cross-encoder rerank
        RAG-3  RAG-2 plus an LLM query rewrite and entity-graph expansion
      Context recall, context precision (MAP@5), MRR, latency.
  2E  Generation, Category C only: RAG-2's passages (the app's pipeline) answered by three generator models, gemini-3.5-flash-lite
      (the app's), gemini-3.5-flash and qwen3.8-27b (LLM_E2E_GENERATORS), with the app's answer prompt. Same passages
      throughout, so only the writer changes. (The 11 Nov run also compared the app's concise prompt with a
      "complete" one; the concise prompt was kept, see agents/policy_agent.py.) The `ragas` library scores Faithfulness and Answer Relevancy (judge Groq gpt-oss-120b) and, once
      per question, RAGAS Context Recall and Context Precision against the fact-sheet reference (Groq gpt-oss-20b);
      the judges are OpenAI models, a different family from every generator, so no model grades its own answers.
      Fact coverage (gold clauses in the answer) and citation compliance are scored by code. Answers and scores are
      cached in evaluation/.cache/: a run stopped by a quota leaves the missing scores empty, and the next run fills
      only those.
      Category E is scored on retrieval (2A-2D) only. Its questions target the market, spatial and financial agents
      (their expected_agents); the policy rules they depend on (CATEGORY_E_POLICY_GROUND_TRUTH) test whether retrieval
      finds them, but the policy agent alone cannot answer the listings and budgets they ask for, so its answers to
      them are not a fair measure of generation.
Variants measured during development and dropped to keep the comparison simple: a blended rerank score (Category E
recall 0.55 vs 0.83) and per-sub-question decomposition (no gain over RAG-2); see the git history of this file.

    .venv/bin/python -m evaluation.run_rag_ablation                    # 2A-2E -> evaluation/rag_ablation_results.json
    .venv/bin/python -m evaluation.run_rag_ablation --retrieval-only   # 2A-2D only, no model calls
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.policy_agent import PolicyRagAgent
from data.ingest_hdb_policy_rag import DOCS_DIR, chunk_section, extract_frontmatter, split_markdown_into_hierarchical_sections
from mcp_server.policy_rag_tools import execute_rag_pipeline

PIPELINE_NAMES = {
    "RAG-1": "RAG-1 (Naive Dense Vector)",
    "RAG-2": "RAG-2 (Hybrid + Cross-Encoder)",
    "RAG-3": "RAG-3 (RAG-2 + Rewrite + Graph)",
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


def _judge(profile_env: str, model_env: str, default_model: Optional[str]) -> Tuple[Any, str]:
    """(client, "profile:model") for a ragas judge: a Groq profile by default, so the judge is from another model family
    than the generator (Gemini) and never grades its own answers. When that profile cannot be built (no key), the
    active profile judges and the label says so."""
    from agents.llm import ModelConfigurationError, _configured_profile, active_profile, get_llm
    name = os.getenv(profile_env, "groq").strip()
    model = (os.getenv(model_env) or "").strip() or default_model
    try:
        prof = _configured_profile(name)
        # gpt-oss is a reasoning model: at 2,000 tokens its reasoning used the budget and ragas's statement lists were
        # cut off (LLMDidNotFinishException), so the judge reasons briefly and has room to finish
        extra: Dict[str, Any] = {"reasoning_effort": "low"} if prof.get("provider") == "groq" else {}
        # LLM_JUDGE_API_KEY: a key of its own for the judge profile, so judging has its own free-tier quota
        if (os.getenv("LLM_JUDGE_API_KEY") or "").strip():
            extra["api_key"] = os.getenv("LLM_JUDGE_API_KEY", "").strip()
        llm = get_llm(profile=name, model=model, temperature=0, max_tokens=4096, **extra)
        # LLM_JUDGE_PAID_TIER: the judge's account is on a paid tier, so the app's free-tier limiter (config) is not
        # applied to the judges; the provider's own limits still answer 429, which ragas_judge waits out and retries
        if os.getenv("LLM_JUDGE_PAID_TIER", "").strip().lower() in ("1", "true", "yes"):
            from agents.rate_limit import RateLimitCallback
            llm.callbacks = [cb for cb in (llm.callbacks or []) if not isinstance(cb, RateLimitCallback)]
        return llm, f"{name}:{model or prof['model']}"
    except ModelConfigurationError as exc:
        print(f"    [judge] profile {name!r} unavailable ({exc}); the active profile judges")
        return _fallback_judge()


_JUDGE: Optional[Tuple[Any, str]] = None             # generation: faithfulness, answer relevancy
_RETRIEVAL_JUDGE: Optional[Tuple[Any, str]] = None   # retrieval: ragas context recall / precision
_FALLBACK_JUDGE: Optional[Tuple[Any, str]] = None
# Groq's free tier allows 8,000 tokens per minute, so one request cannot be larger (the reserved output does not
# count: a 5,883-token prompt with max_tokens 4,096 was accepted). 6,500 (characters / 4) leaves room for ragas's own
# prompt text and keeps all but about one RAG context on the Groq judge; the long-context baseline's request (the whole
# corpus) is judged by the generator's model instead, and the row records which judge it was.
JUDGE_MAX_PROMPT_TOKENS = int(os.getenv("LLM_JUDGE_MAX_PROMPT_TOKENS", "6500"))


def _fallback_judge() -> Tuple[Any, str]:
    """The judge for a request larger than the Groq free tier's per-minute token limit: the generator's model, labelled
    so in the row."""
    global _FALLBACK_JUDGE
    if _FALLBACK_JUDGE is None:
        from agents.llm import active_profile, get_llm
        prof = active_profile()
        _FALLBACK_JUDGE = (get_llm(temperature=0, max_tokens=4096), f"{prof['name']}:{prof.get('model')} (same as generator)")
    return _FALLBACK_JUDGE


def _too_big(*texts: Any) -> bool:
    return sum(len(t) if isinstance(t, str) else sum(len(x) for x in t) for t in texts) // 4 > JUDGE_MAX_PROMPT_TOKENS


def _generation_judge(*texts: Any) -> Tuple[Any, str]:
    """The judge for faithfulness and answer relevancy (LLM_JUDGE_PROFILE / LLM_JUDGE_MODEL, default Groq
    openai/gpt-oss-120b). A request too large for Groq's free tier goes to the fallback judge."""
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = _judge("LLM_JUDGE_PROFILE", "LLM_JUDGE_MODEL", "openai/gpt-oss-120b")
    return _fallback_judge() if _JUDGE[1].startswith("groq:") and _too_big(*texts) else _JUDGE


def _context_judge(*texts: Any) -> Tuple[Any, str]:
    """The judge for context recall and precision (LLM_RETRIEVAL_JUDGE_PROFILE / LLM_RETRIEVAL_JUDGE_MODEL, default
    Groq openai/gpt-oss-20b). These grade the retrieved passages, not a model's answer, so a smaller judge is enough."""
    global _RETRIEVAL_JUDGE
    if _RETRIEVAL_JUDGE is None:
        _RETRIEVAL_JUDGE = _judge("LLM_RETRIEVAL_JUDGE_PROFILE", "LLM_RETRIEVAL_JUDGE_MODEL", "openai/gpt-oss-20b")
    return _fallback_judge() if _RETRIEVAL_JUDGE[1].startswith("groq:") and _too_big(*texts) else _RETRIEVAL_JUDGE


def llm_judge(question: str, answer: str, contexts: List[str]) -> Dict[str, Optional[float]]:
    """RAGAS Faithfulness and Answer Relevancy (evaluation/ragas_judge.py)."""
    from evaluation.ragas_judge import generation_scores
    llm, label = _generation_judge(question, answer, contexts)
    return generation_scores(llm, label, question, answer, contexts)


def retrieval_judge(question: str, contexts: List[str], reference: str) -> Dict[str, Optional[float]]:
    """RAGAS Context Recall and Context Precision against the fact-sheet reference."""
    from evaluation.ragas_judge import retrieval_scores
    llm, label = _context_judge(question, reference, contexts)
    return retrieval_scores(llm, label, question, contexts, reference)


_FACTS: Optional[Dict[str, str]] = None


def reference_answer(test_case: Dict[str, Any]) -> str:
    """The reference answer ragas's context metrics compare against: the fact-sheet facts the question points to
    (Category C), else its gold clauses as a list (Category E, whose ground truth is clauses only)."""
    global _FACTS
    if _FACTS is None:
        import yaml
        sheet = yaml.safe_load((PROJECT_ROOT / "evaluation" / "policy_fact_sheet.yaml").read_text(encoding="utf-8"))
        _FACTS = {f["id"]: f["fact"] for f in sheet.get("facts", []) if f.get("id") and f.get("fact")}
    gt = test_case.get("ground_truth_facts", {})
    facts = [_FACTS[i] for i in gt.get("fact_ids", []) if i in _FACTS]
    if facts:
        return " ".join(facts)
    clauses = [" ".join(c) if isinstance(c, (list, tuple)) else str(c) for c in gt.get("key_clauses", [])]
    return "; ".join(clauses)


# Model calls are cached per (pipeline, question, passages), so a run interrupted by a free-tier quota or a closed
# session resumes where it stopped instead of paying for every answer again. Delete the file to regenerate.
CACHE_FILE = PROJECT_ROOT / "evaluation" / ".cache" / "experiment_2e.json"
_CACHE: Optional[Dict[str, Any]] = None


# 2E: the generator models compared on RAG-2's passages, as profile:model. The first is the app's model. None of them
# may be an OpenAI model: the judges are OpenAI's gpt-oss, and no model grades its own family's answers.
GENERATORS = [g.strip() for g in os.getenv(
    "LLM_E2E_GENERATORS", "gemini:gemini-3.5-flash-lite,gemini:gemini-3.5-flash,groq:qwen/qwen3.8-27b").split(",")
    if g.strip()]
_GENERATOR_CLIENTS: Dict[str, Any] = {}


def generator_client(spec: str) -> Any:
    """The client for a "profile:model" generator (built once)."""
    if spec not in _GENERATOR_CLIENTS:
        from agents.llm import get_llm
        profile, model = spec.split(":", 1)
        _GENERATOR_CLIENTS[spec] = get_llm(profile=profile, model=model)
    return _GENERATOR_CLIENTS[spec]


# The answer prompt the cached answers were written with (agents/policy_agent.py). Change it when that prompt changes.
PROMPT_VERSION = "prompt=concise-2026-11-12"

SCORE_GROUPS = {"generation": ("llm_faithfulness", "llm_answer_relevance"),
                "retrieval": ("ragas_context_recall", "ragas_context_precision")}
JUDGE_LABEL_FIELD = {"generation": "judge", "retrieval": "retrieval_judge"}
Judge = Tuple[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Dict[str, Any]]]


def _cached(key_parts: List[str], answer: Callable[[], Dict[str, Any]], judges: Dict[str, Judge]) -> Dict[str, Any]:
    """The cached row for these key parts. The answer is generated once; each judge group (generation, retrieval) is
    run again while one of its scores is missing or was given by another judge model than the one configured now, so
    a resumed run never pays for an answer twice and every row of a table has the same judge. Scores replaced because
    the judge changed are kept under "<group>_previous_judges" for comparing judges."""
    import hashlib
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    key = hashlib.sha1("\x1f".join(key_parts).encode()).hexdigest()
    row = dict(_CACHE.get(key) or {})
    changed = False
    if not row.get("llm_answer"):
        row = answer()
        changed = True
    for group, (expected_label, judge) in judges.items():
        fields, label_field = SCORE_GROUPS[group], JUDGE_LABEL_FIELD[group]
        current = f"ragas + {expected_label(row)}"
        stale = row.get(label_field) not in (None, current)
        if stale and all(row.get(f) is not None for f in fields):
            previous = row.setdefault(f"{group}_previous_judges", [])
            previous.append({label_field: row[label_field], **{f: row[f] for f in fields}})
        if stale or any(row.get(f) is None for f in fields):
            row.update(judge(row))
            changed = True
    if changed:
        _CACHE[key] = row
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(_CACHE, indent=1))
    return row


def answer_metrics(test_case: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """Code-scored answer metrics: gold clauses found in the answer, and whether it cites an official page."""
    raw_clauses = test_case.get("ground_truth_facts", {}).get("key_clauses", [])
    matched = sum(1 for clause in raw_clauses if is_clause_in_passage(clause, clean_text(answer or "")))
    return {
        "llm_answer_fact_coverage": round(matched / max(1, len(raw_clauses)), 4),
        "llm_citation_compliance": 1.0 if re.search(r"https://www\.(hdb|cpf)\.gov\.sg", answer or "") else 0.0,
        "llm_answer": answer,
    }


def evaluate_query_end_to_end(
    test_case: Dict[str, Any],
    pipeline: str,
    use_llm: bool = False,
    generator: Optional[str] = None,
) -> Dict[str, Any]:
    """One question through one pipeline: retrieval metrics (2D) and, with use_llm, the LLM answer and the four ragas
    metrics (2E)."""
    query = test_case["query"]
    gt = test_case.get("ground_truth_facts", {})
    raw_clauses = gt.get("key_clauses", [])
    target_sections = [clean_text(s) for s in gt.get("target_sections", [])]

    t0 = time.perf_counter()
    passages = execute_rag_pipeline(query=query, pipeline=pipeline, top_k=5).passages
    latency_ms = (time.perf_counter() - t0) * 1000.0
    passage_dicts = [p.model_dump() for p in passages]
    cleaned_passages = [(clean_text(p.text), clean_text(p.section)) for p in passages]

    # Context recall: share of the gold clauses found in the top-5 passages
    matched_clauses = sum(1 for clause in raw_clauses
                          if any(is_clause_in_passage(clause, t, sec) for t, sec in cleaned_passages))
    context_recall = matched_clauses / max(1, len(raw_clauses))

    # Context precision (MAP@5): a passage is relevant when it holds a gold clause or comes from a target section
    relevance = [1 if (any(is_clause_in_passage(c, t, sec) for c in raw_clauses) or any(x in sec for x in target_sections))
                 else 0 for t, sec in cleaned_passages]
    hits, cum = 0, []
    for k, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            cum.append(hits / k)
    context_precision = sum(cum) / len(cum) if cum else 0.0
    mrr = next((1.0 / k for k, rel in enumerate(relevance, start=1) if rel), 0.0)

    extra: Dict[str, Any] = {}
    if use_llm:
        contexts = [f"(Source: {p['source_url']}) {p['text']}" for p in passage_dicts]

        # generator: a "profile:model" other than the app's. The app's own row is the one that also judges retrieval
        # (the passages are the same for every generator, so their RAGAS context scores are computed once).
        # PROMPT_VERSION keeps answers written with an earlier answer prompt from being reused.
        variant = [f"generator={generator}"] if generator else []
        judges: Dict[str, Judge] = {
            "generation": (lambda row: _generation_judge(query, row["llm_answer"] or "", contexts)[1],
                           lambda row: llm_judge(query, row["llm_answer"] or "", contexts))}
        if not variant:
            judges["retrieval"] = (lambda row: _context_judge(query, reference_answer(test_case), contexts)[1],
                                   lambda row: retrieval_judge(query, contexts, reference_answer(test_case)))
        client = generator_client(generator) if generator else None
        extra = _cached(
            [pipeline, test_case["id"], query, *contexts, PROMPT_VERSION] + variant,
            lambda: answer_metrics(test_case, PolicyRagAgent.generate_llm_answer(query, passage_dicts, llm=client)),
            judges)

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
        # share of the returned passages that hold a gold clause or come from a target section
        "relevant_passage_share": round(sum(relevance) / len(relevance), 4) if relevance else 0.0,
        "matched_clauses": f"{matched_clauses}/{len(raw_clauses)}",
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
    """Mean over the rows that have the metric (None when none do)."""
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
        "mean_relevant_passage_share": _mean(results, "relevant_passage_share"),
        "p50_latency_ms": round(latencies_sorted[p50_idx], 2),
        "p95_latency_ms": round(latencies_sorted[p95_idx], 2),
    }
    for key in ("llm_faithfulness", "ragas_context_recall", "ragas_context_precision",
                "llm_answer_relevance", "llm_answer_fact_coverage", "llm_citation_compliance"):
        if any(r.get(key) is not None for r in results):
            summary[f"mean_{key}"] = _mean(results, key)
    return summary


def _fmt(value: Optional[float], width: int) -> str:
    return f"{value:<{width}.4f}" if value is not None else f"{'n/a':<{width}}"


def load_benchmark_cases() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """(Category C answerable cases, Category E cases with policy ground truth, unanswerable C ids) for Experiment 2."""
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
    return cat_c_cases, cat_e_cases, unanswerable


def _bar(width: int = 120) -> None:
    print("=" * width)


def print_retrieval_table(title: str, summary: Dict[str, Dict[str, Optional[float]]]) -> None:
    _bar()
    print(f" {title}")
    _bar()
    print(f"{'Pipeline':<32} | {'Ctx Recall':<10} | {'Ctx Precision':<13} | {'MRR':<7} | {'Relevant psg':<12} | "
          f"{'P50 (ms)':<9} | {'P95 (ms)':<9}")
    print("-" * 120)
    for p, st in summary.items():
        print(f"{PIPELINE_NAMES.get(p, p):<32} | {_fmt(st['mean_context_recall'], 10)} | "
              f"{_fmt(st['mean_context_precision'], 13)} | {_fmt(st['mean_mrr'], 7)} | "
              f"{_fmt(st.get('mean_relevant_passage_share'), 12)} | "
              f"{st['p50_latency_ms']:<9.2f} | {st['p95_latency_ms']:<9.2f}")
    _bar()


def print_generation_table(title: str, summary: Dict[str, Dict[str, Optional[float]]]) -> None:
    _bar()
    print(f" {title}")
    _bar()
    print(f"{'Pipeline':<32} | {'Faithfulness':<12} | {'Ans Relevancy':<13} | {'Ctx Recall':<10} | "
          f"{'Ctx Precision':<13} | {'Fact Coverage':<13} | {'Citations':<9}")
    print("-" * 120)
    for p, st in summary.items():
        print(f"{PIPELINE_NAMES.get(p, p):<32} | {_fmt(st.get('mean_llm_faithfulness'), 12)} | "
              f"{_fmt(st.get('mean_llm_answer_relevance'), 13)} | {_fmt(st.get('mean_ragas_context_recall'), 10)} | "
              f"{_fmt(st.get('mean_ragas_context_precision'), 13)} | {_fmt(st.get('mean_llm_answer_fact_coverage'), 13)} | "
              f"{_fmt(st.get('mean_llm_citation_compliance'), 9)}")
    _bar()


def _by_category(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    return {group: {p: compute_group_summary([r for r in rows if not cat or r["category"] == cat])
                    for p, rows in results.items()}
            for group, cat in (("category_c", "C"), ("category_e", "E"), ("combined", ""))}


def run_ablation(generation: bool = True, limit: Optional[int] = None):
    """Experiment 2, in the order of the pipeline: 2A chunking strategy, 2B chunk threshold, 2C embedding model,
    2D retrieval pipeline, 2E generation (an LLM writes each answer, the ragas library judges it)."""
    _bar(96)
    print(" HDB-Compass: Experiment 2 - Policy RAG, from chunking to generation (Member 3)")
    print(" 2A chunking strategy | 2B chunk threshold | 2C embedding | 2D retrieval | 2E generation (LLM + RAGAS)")
    _bar(96)

    cat_c_cases, cat_e_cases, unanswerable = load_benchmark_cases()
    if limit:
        cat_c_cases, cat_e_cases = cat_c_cases[:limit], cat_e_cases[:limit]
    all_cases = cat_c_cases + cat_e_cases
    n = len(all_cases)
    print(f"\n{len(cat_c_cases)} Category C + {len(cat_e_cases)} Category E = {n} questions\n")

    # --- 2A: chunking strategy ------------------------------------------------------------------------------------
    print("--> 2A chunking strategy ...")
    exp_2a = evaluate_chunk_window_ablation(all_cases)
    _bar(96)
    print(f" EXPERIMENT 2A: CHUNKING STRATEGY - fixed-size vs semantic, table-preserving (BM25 top-5, N={n})")
    _bar(96)
    print(f"{'Chunking Strategy':<36} | {'Total Chunks':<12} | {'Broken Tables':<13} | {'Ctx Recall':<11} | {'Ctx Precision':<13}")
    print("-" * 96)
    for name, st in exp_2a.items():
        print(f"{name:<36} | {st['num_chunks']:<12} | {st['broken_table_chunks']:<13} | "
              f"{st['mean_context_recall']:<11.4f} | {st['mean_context_precision']:<13.4f}")
    _bar(96)

    # --- 2B: chunk threshold ----------------------------------------------------------------------------------------
    print("--> 2B chunk similarity threshold ...")
    exp_2b = evaluate_chunk_threshold_sweep(all_cases, thresholds=(0.15, 0.25, 0.35, 0.45, 0.55))
    _bar(96)
    print(f" EXPERIMENT 2B: SEMANTIC-CHUNK SIMILARITY THRESHOLD, scored through RAG-2 (N={n})")
    _bar(96)
    print(f"{'Similarity threshold':<24} | {'Chunks':<7} | {'Mean words':<10} | {'Ctx Recall':<11} | {'Ctx Precision':<13}")
    print("-" * 96)
    for name, row in exp_2b.items():
        print(f"{name:<24} | {row['num_chunks']:<7} | {row['mean_words']:<10} | "
              f"{row['mean_context_recall']:<11.4f} | {row['mean_context_precision']:<13.4f}")
    _bar(96)

    # --- 2C: embedding model ----------------------------------------------------------------------------------------
    print("--> 2C embedding model ...")
    from evaluation.embedding_comparison import MODELS, evaluate as evaluate_embedder, print_table as print_embedders
    from mcp_server.policy_rag_tools import _load_store
    store = _load_store()
    exp_2c = {m["name"]: evaluate_embedder(m, store, all_cases) for m in MODELS}
    print_embedders(exp_2c, len(store["chunks"]), n)

    # --- 2D: retrieval pipeline -------------------------------------------------------------------------------------
    print("--> 2D retrieval pipelines ...")
    execute_rag_pipeline(query="Warmup query MOP Prime", pipeline="RAG-3", top_k=5)   # warm the models before timing
    execute_rag_pipeline(query="Warmup query MOP Prime", pipeline="RAG-1", top_k=5)
    # the app's answers (2E) are written alongside RAG-2's retrieval, for Category C only (see the docstring)
    retrieval = {p: [evaluate_query_end_to_end(case, pipeline=p,
                                               use_llm=generation and p == "RAG-2" and case["category"] == "C")
                     for case in all_cases]
                 for p in ("RAG-1", "RAG-2", "RAG-3")}
    exp_2d = _by_category(retrieval)
    for group, label in (("category_c", "Category C: regulatory rules"), ("category_e", "Category E: multi-objective"),
                         ("combined", "Combined C + E")):
        print_retrieval_table(f"EXPERIMENT 2D: RETRIEVAL PIPELINES, top-5 - {label}", exp_2d[group])

    # --- 2E: generation -----------------------------------------------------------------------------------------
    exp_2e = None
    generation_results: Dict[str, List[Dict[str, Any]]] = {}
    if generation:
        app_model = GENERATORS[0]
        policy_cases = [case for case in all_cases if case["category"] == "C"]
        generation_results = {f"RAG-2 · {app_model.split(':', 1)[1]}":
                              [r for r in retrieval["RAG-2"] if r["category"] == "C"]}
        for spec in GENERATORS[1:]:
            print(f"--> 2E generation: RAG-2 passages, written by {spec} ...")
            label = f"RAG-2 · {spec.split(':', 1)[1]}"
            generation_results[label] = [{**evaluate_query_end_to_end(case, "RAG-2", use_llm=True, generator=spec),
                                          "pipeline": label} for case in policy_cases]
        exp_2e = {"category_c": {p: compute_group_summary(rows) for p, rows in generation_results.items()}}
        print_generation_table(f"EXPERIMENT 2E: GENERATION on RAG-2's passages, Category C (N={len(policy_cases)}) - "
                               f"ragas; judges {_JUDGE[1] if _JUDGE else '-'} / "
                               f"{_RETRIEVAL_JUDGE[1] if _RETRIEVAL_JUDGE else '-'}", exp_2e["category_c"])

    payload = {
        "questions": n,
        "unanswerable_category_c": unanswerable,
        "experiment_2a_chunking_strategy": exp_2a,
        "experiment_2b_chunk_threshold": exp_2b,
        "experiment_2c_embedding_model": {k: {kk: vv for kk, vv in v.items() if kk != "per_question"}
                                          for k, v in exp_2c.items()},
        "experiment_2d_retrieval": exp_2d,
        "experiment_2e_generation": exp_2e,
        "generators": GENERATORS if generation else None,
        "judge_model": _JUDGE[1] if _JUDGE else None,
        "retrieval_judge_model": _RETRIEVAL_JUDGE[1] if _RETRIEVAL_JUDGE else None,
        "judge_token_usage": _judge_usage() if generation else None,
        "judges_out_of_daily_quota": _judge_exhausted() if generation else None,
        "detailed_results": {"retrieval_2d": retrieval, "generation_2e": generation_results},
    }
    suffix = ("" if generation else "_retrieval_only") + (f"_limit{limit}" if limit else "")
    out_path = RESULTS_JSON.with_name(RESULTS_JSON.stem + suffix + RESULTS_JSON.suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {out_path}\n")
    if generation:
        missing = [(p, r["id"]) for p, rows in generation_results.items() for r in rows
                   if r.get("llm_faithfulness") is None or r.get("llm_answer_relevance") is None
                   or (p == f"RAG-2 · {GENERATORS[0].split(':', 1)[1]}"
                       and (r.get("ragas_context_recall") is None or r.get("ragas_context_precision") is None))]
        print(f"Judge tokens used this run: {_judge_usage()}")
        if missing:
            print(f"{len(missing)} answers still have a missing score {_judge_exhausted() and '(daily quota reached)'}; "
                  "run the same command again later and only those are judged.")


def _judge_usage() -> Dict[str, Any]:
    from evaluation.ragas_judge import token_usage
    return token_usage()


def _judge_exhausted() -> Dict[str, str]:
    from evaluation.ragas_judge import exhausted_judges
    return exhausted_judges()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 2 (2A-2E): policy RAG from chunking to generation")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip 2E (no model calls), e.g. to re-check retrieval after a corpus change without "
                             "spending the free-tier quota; the results go to a separate file")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions of each category")
    args = parser.parse_args()
    run_ablation(generation=not args.retrieval_only, limit=args.limit)
