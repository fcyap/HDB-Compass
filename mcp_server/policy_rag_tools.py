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
   - RAG-2 returns top-5; RAG-3 uses a score-drop cutoff with a minimum of two passages
4. Supports Ablation Study Pipelines:
   - RAG-1: Naive Dense Vector Search (top-5)
   - RAG-2: Hybrid Search + Cross-Encoder Reranking (top-20 -> top-5)
   - RAG-3: Advanced RAG (query rewriting + HyDE + graph expansion + reranking)
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
            # Try loading from local cache first (instant, <0.2s, offline-safe)
            try:
                _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception:
                # If not cached yet (e.g. teammate's first run), download once from Hugging Face Hub
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
            # Try loading from local cache first (instant, <0.2s, offline-safe)
            try:
                _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
            except Exception:
                # If not cached yet (e.g. teammate's first run), download once from Hugging Face Hub
                _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[Warning] CrossEncoder load failed ({e}); using hybrid reciprocal rank score.")
            _CROSS_ENCODER = False
    return _CROSS_ENCODER


# ---------------------------------------------------------------------------
# 1. Query Expansion & Domain Detection
# ---------------------------------------------------------------------------
# Query expansion (Topic 8: query rewriting against vocabulary mismatch), used when no model rewrites the question.
# Terms come from the official pages' own wording and carry no figures (figures go stale, and the earlier table injected
# the legacy corpus's wrong "3% to 6%" / "6% to 9%" and its "15-month wait-out", which pulled searches away from HDB's
# actual rules). tests/test_policy_corpus.py checks every word here appears in the corpus.
ACRONYM_EXPANSIONS = {
    r"\bmop\b": "Minimum Occupation Period MOP",
    r"\beip\b": "Ethnic Integration Policy EIP quota block neighbourhood",
    r"\bhfe\b": "HDB Flat Eligibility HFE letter eligible to buy a new or resale flat CPF housing grants HDB housing loan",
    r"\behg\b": "Enhanced CPF Housing Grant EHG",
    r"\bphg\b": "Proximity Housing Grant PHG live with or near parents",
    r"\bprime\b": "Prime flat subsidy recovery minimum occupation period renting out of whole flat",
    r"\bplus\b": "Plus flat subsidy recovery minimum occupation period renting out of whole flat",
    r"\bstandard\b": "Standard flat minimum occupation period",
    r"\bclawback\b": "subsidy recovery percentage of the resale price",
    r"\bwait-?out\b": "private residential property dispose of the interest within 6 months 30 months before HFE letter application",
    r"\b(?:condo|condominium|private property)\b": "private residential property dispose of the interest within 6 months",
    r"\bsingle[s]?\b": "singles aged 35 and above 2-room Flexi flat",
    r"\bright-?sizing\b": "right-sizing seniors 55 and above Silver Housing Bonus private residential property",
}


def expand_query(query: str) -> str:
    """Expands query with policy acronyms and relevant synonyms."""
    expanded = query
    for pattern, replacement in ACRONYM_EXPANSIONS.items():
        if re.search(pattern, query, re.I):
            expanded += f" {replacement}"
    return expanded


# The policy_category values the ingest writes (one per source document).
POLICY_DOMAINS = ("classification", "eligibility", "grants", "financing", "eip", "bto_priority", "resale_process")

# Words that put a question in a domain, checked in this order (the first match is infer_domain's answer).
DOMAIN_TRIGGERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # "plus" alone also reads "plus $30,000 in cash" as a Plus-flat question
    ("classification", ("prime", "plus flat", "plus bto", "plus resale", "plus and", "plus or", "standard", "clawback",
                        "subsidy recovery")),
    ("eip", ("eip", "ethnic", "spr quota", "buyback", "racial")),
    ("eligibility", ("single", "age 35", "34-year", "wait-out", "wait out", "senior", "55", "eligibility", "scheme",
                     "fiance", "condo")),
    ("grants", ("grant", "ehg", "phg", "family grant", "housing grant", "stacking")),
    ("financing", ("msr", "ltv", "tdsr", "hfe", "loan", "downpayment", "stamp duty", "bsd")),
)


def valid_domain(domain: Optional[str]) -> Optional[str]:
    """The domain in the form the chunks store it, or None when it is not one of POLICY_DOMAINS."""
    d = (domain or "").strip().lower()
    return d if d in POLICY_DOMAINS else None


def _trigger_hits(query: str) -> List[Tuple[str, int]]:
    """(domain, number of its trigger words in the query) for each domain with at least one, in DOMAIN_TRIGGERS order.
    A trigger matches at the start of a word, so "plus" does not fire inside "surplus" ("grant" still finds "grants")."""
    q = query.lower()
    hits = [(domain, sum(1 for k in words if re.search(r"(?<![a-z0-9])" + re.escape(k), q)))
            for domain, words in DOMAIN_TRIGGERS]
    return [(domain, n) for domain, n in hits if n]


def detect_domains(query: str) -> List[str]:
    """Every policy domain a question touches, most trigger words first (ties keep DOMAIN_TRIGGERS order). Two or more
    means the question spans several source documents, and the agent decomposes it into one retrieval per domain
    (Topic 8, decomposition); a cap on the searches then keeps the best-supported domains."""
    hits = _trigger_hits(query)
    return [domain for domain, _ in sorted(hits, key=lambda item: -item[1])]


def decompose_query(query: str, max_parts: int = 4) -> List[str]:
    """Sub-queries for a question that spans several policy domains: for each domain (most trigger words first), the
    sentences of the question that mention it. [] for a single-topic question. Measured on Category E1 with RAG-2,
    whole question + these sub-queries searched over the whole corpus raised context recall@5 from 0.28 to 0.44;
    the same domains as hard document filters with the whole question as text lowered it (0.22), because a clause
    often sits in another document than its topic's (the 95-year lease rule, MSR)."""
    domains = detect_domains(query)
    if len(domains) < 2:
        return []
    triggers = dict(DOMAIN_TRIGGERS)
    sentences = [x.strip() for x in re.split(r"(?<=[.?!;])\s+", query) if x.strip()]
    parts: List[str] = []
    for domain in domains:
        hit = [x for x in sentences
               if any(re.search(r"(?<![a-z0-9])" + re.escape(k), x.lower()) for k in triggers[domain])]
        part = " ".join(hit)
        if part and part != query.strip() and part not in parts:
            parts.append(part)
    return parts[:max_parts]


def infer_domain(query: str, explicit_domain: Optional[str] = None) -> Optional[str]:
    """Infers policy domain if not explicitly provided (the first domain whose trigger words appear)."""
    if valid_domain(explicit_domain):
        return valid_domain(explicit_domain)
    hits = _trigger_hits(query)
    return hits[0][0] if hits else None


# ---------------------------------------------------------------------------
# 2. Dense Vector Retrieval
# ---------------------------------------------------------------------------
_NAIVE_DENSE_MATRIX: Optional[np.ndarray] = None


def _top(scores: np.ndarray, top_k: int, allowed: Optional[set]) -> List[Tuple[int, float]]:
    """Indices of the top_k scores, only among `allowed` chunk indices when given."""
    order = [int(i) for i in np.argsort(scores)[::-1] if allowed is None or int(i) in allowed]
    return [(idx, float(scores[idx])) for idx in order[:top_k]]


def retrieve_dense(query: str, store: Dict[str, Any], top_k: int = 20, naive_body_only: bool = False,
                   allowed: Optional[set] = None) -> List[Tuple[int, float]]:
    """Cosine similarity over dense vector embeddings (only over `allowed` chunk indices when given)."""
    global _NAIVE_DENSE_MATRIX
    embed_model = _get_embed_model()
    chunks = store["chunks"]
    dense_matrix = store["dense_embeddings"]

    if naive_body_only and embed_model:
        if _NAIVE_DENSE_MATRIX is None:
            # Strip hierarchical breadcrumb header and table pipes to simulate naive fixed-chunk dense search
            raw_bodies = [
                re.sub(r"^\[.*?\]\s*", "", c["text"]).splitlines()[0][:220]
                for c in chunks
            ]
            _NAIVE_DENSE_MATRIX = embed_model.encode(raw_bodies, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        dense_matrix = _NAIVE_DENSE_MATRIX

    if embed_model:
        query_vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    else:
        # Fallback: token frequency cosine similarity
        words = set(re.findall(r"\b\w+\b", query.lower()))
        sims = []
        for i, c in enumerate(chunks):
            txt = re.sub(r"^\[.*?\]\s*", "", c["text"])[:220] if naive_body_only else c["text"]
            c_words = set(re.findall(r"\b\w+\b", txt.lower()))
            overlap = len(words & c_words) / max(1, len(words | c_words))
            sims.append(overlap)
        return _top(np.array(sims, dtype=np.float32), top_k, allowed)

    return _top(np.dot(dense_matrix, query_vec), top_k, allowed)


# ---------------------------------------------------------------------------
# 3. BM25 Sparse Retrieval
# ---------------------------------------------------------------------------
def retrieve_bm25(query: str, store: Dict[str, Any], top_k: int = 20, allowed: Optional[set] = None) -> List[Tuple[int, float]]:
    """BM25 sparse keyword search (only over `allowed` chunk indices when given)."""
    from data.ingest_hdb_policy_rag import tokenize_words
    bm25 = store["bm25"]
    tokens = tokenize_words(query)
    if not tokens:
        tokens = tokenize_words(expand_query(query))
    return _top(np.asarray(bm25.get_scores(tokens)), top_k, allowed)


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
def _clean_for_cross_encoder(text: str) -> str:
    """Normalizes Markdown table syntax and formatting into clean natural prose for cross-encoder."""
    t = re.sub(r"\|[\s:-]+\|", " ", text)
    t = re.sub(r"[|#*_`]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def rerank_cross_encoder(
    query: str,
    candidate_indices: List[Tuple[int, float]],
    store: Dict[str, Any],
    top_n: int = 5,
    rerank_mode: str = "blend",
) -> List[Tuple[int, float]]:
    """
    Reranks top-20 RRF candidate chunks down to top-5 using a CrossEncoder model
    for deep query-passage token interaction.
    rerank_mode: "blend" (default) = 0.25 cross-encoder + 0.35 RRF + 0.40 query-term coverage;
    "ce_only" = the cross-encoder score alone (the Topic 8 two-phase re-ranker), for the ablation.
    """
    cross_encoder = _get_cross_encoder()
    chunks = store["chunks"]

    if not candidate_indices:
        return []

    q_tokens = [w for w in re.findall(r"\b[a-z0-9]+\b", query.lower()) if len(w) > 1]

    # If CrossEncoder is available
    if cross_encoder:
        pairs = [[query, _clean_for_cross_encoder(chunks[idx]["text"])] for idx, _ in candidate_indices]
        try:
            scores = cross_encoder.predict(pairs)
            raw_scores = np.nan_to_num(np.array(scores, dtype=np.float32), nan=0.0, posinf=50.0, neginf=-50.0)
            norm_scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -50.0, 50.0)))
            max_rrf = max((s for _, s in candidate_indices), default=1.0)
            reranked = []
            for i, (idx, rrf_s) in enumerate(candidate_indices):
                norm_rrf = rrf_s / max_rrf if max_rrf > 0 else 0.0
                section = chunks[idx].get("section", "")
                # The full breadcrumb repeats the same document title in every chunk. Score the local section and
                # body instead, otherwise a PHG chunk inherits words from its parent heading.
                local_section = section.rsplit(" > ", 1)[-1]
                body = re.sub(r"^\[.*?\]\s*", "", chunks[idx]["text"])
                c_text_lower = f"{local_section} {body}".lower()
                token_cov = (
                    sum(1 for w in q_tokens if w in c_text_lower) / max(1, len(q_tokens))
                    if q_tokens else 0.0
                )
                # Balance semantic matching with exact coverage of the user's substantive terms. The latter prevents
                # a generic parent/PHG chunk from outranking a passage whose local section names the requested rule.
                blended = (float(norm_scores[i]) if rerank_mode == "ce_only"
                           else 0.25 * float(norm_scores[i]) + 0.35 * norm_rrf + 0.40 * token_cov)
                reranked.append((idx, blended))
            reranked.sort(key=lambda x: x[1], reverse=True)
            # Keep scores in [0, 1] without saturating every strong match at exactly 1.0. The old min(1.0, ...) made
            # the score output useless for diagnosing ranking and hid small but meaningful ordering differences.
            max_blended = max((score for _, score in reranked), default=1.0)
            return [(idx, round(score / max_blended, 4)) for idx, score in reranked[:top_n]]
        except Exception as e:
            print(f"[Warning] CrossEncoder prediction failed ({e}); falling back to RRF scores.")

    # Fallback normalization over RRF scores
    max_score = max((s for _, s in candidate_indices), default=1.0)
    return [(idx, round(s / max_score, 4)) for idx, s in candidate_indices[:top_n]]


# ---------------------------------------------------------------------------
# 6. Advanced retrieval helpers (RAG-3)
# ---------------------------------------------------------------------------
def _rewrite_and_hypothesize(query: str) -> Tuple[str, str]:
    """Return a search rewrite and hypothetical passage. HyDE text is retrieval-only, never evidence."""
    try:
        from agents.llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm(temperature=0, max_tokens=500)
        if llm is None:
            return query, ""
        response = llm.invoke([
            SystemMessage(content=(
                "Prepare search inputs for a Singapore HDB policy retrieval system. "
                "Return exactly two lines: REWRITE: a concise keyword-rich version of the question; "
                "HYDE: a short hypothetical policy passage likely to answer it. "
                "Do not invent specific legal figures when uncertain."
            )),
            HumanMessage(content=query),
        ])
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
        content = str(content or "")
        rewrite = re.search(r"(?im)^\s*REWRITE:\s*(.+)$", content)
        hyde = re.search(r"(?im)^\s*HYDE:\s*(.+(?:\n(?!\w+:).+)*)", content)
        return (rewrite.group(1).strip() if rewrite else query,
                hyde.group(1).strip() if hyde else "")
    except Exception as exc:
        print(f"[Notice] RAG-3 query augmentation unavailable; using original query ({exc}).")
        return query, ""


def _graph_expand_candidates(seed_results: List[Tuple[int, float]], store: Dict[str, Any], top_n: int = 30,
                             allowed: Optional[set] = None) -> List[Tuple[int, float]]:
    """Small corpus graph linking passages through policy entities and shared document sections."""
    chunks = store["chunks"]
    entity_patterns = {
        "MOP": r"\bMOP\b|minimum occupation period", "EIP": r"\bEIP\b|ethnic integration",
        "EHG": r"\bEHG\b|enhanced.*grant", "PHG": r"\bPHG\b|proximity housing grant",
        "HFE": r"\bHFE\b|flat eligibility", "Prime": r"\bPrime\b", "Plus": r"\bPlus\b",
        "Standard": r"\bStandard\b", "wait-out": r"wait.?out|waiting period", "CPF": r"\bCPF\b",
        "single": r"\bsingle[s]?\b", "senior": r"\bsenior|55 and above|right.?siz",
    }
    entities = []
    for chunk in chunks:
        text = f"{chunk.get('section', '')} {chunk.get('text', '')}"
        entities.append({key for key, pattern in entity_patterns.items() if re.search(pattern, text, re.I)})
    scores: Dict[int, float] = {idx: float(score) for idx, score in seed_results}
    for idx, score in seed_results:
        root = chunks[idx].get("section", "").split(" > ", 1)[0]
        for other, other_entities in enumerate(entities):
            if other == idx or (allowed is not None and other not in allowed):
                continue
            shared = entities[idx] & other_entities
            same_document = chunks[other].get("section", "").split(" > ", 1)[0] == root
            graph_weight = min(0.65, 0.15 * len(shared)) if shared else (0.08 if same_document else 0.0)
            if graph_weight:
                scores[other] = max(scores.get(other, 0.0), float(score) * graph_weight)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def _sage_gradient_select(
    ranked: List[Tuple[int, float]], min_k: int = 2, drop_ratio: float = 0.30
) -> List[Tuple[int, float]]:
    """Keep the leading reranked chunks until a sharp relevance-score drop (SAGE selection step)."""
    if not ranked:
        return []
    ranked = sorted(ranked, key=lambda item: item[1], reverse=True)
    keep = min(max(1, min_k), len(ranked))
    for i in range(keep, len(ranked)):
        previous = max(float(ranked[i - 1][1]), 1e-8)
        current = max(float(ranked[i][1]), 0.0)
        if current / previous < drop_ratio:
            break
        keep += 1
    return ranked[:keep]


def _policy_rag_config() -> Dict[str, Any]:
    try:
        import yaml
        cfg = yaml.safe_load((PROJECT_ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")) or {}
        return cfg.get("policy_rag") or {}
    except Exception:  # noqa: BLE001
        return {}


def configured_pipeline() -> str:
    """policy_rag.active_pipeline from config/experiment_config.yaml ("RAG-2" when unset or unknown): the pipeline the
    app's policy search uses unless a caller names one."""
    pipeline = str(_policy_rag_config().get("active_pipeline") or "RAG-2")
    return pipeline if pipeline in ("RAG-1", "RAG-2", "RAG-3") else "RAG-2"


def configured_rerank_mode() -> str:
    """policy_rag.rerank_mode from config/experiment_config.yaml ("ce_only" when unset or unreadable)."""
    try:
        import yaml
        cfg = yaml.safe_load((PROJECT_ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")) or {}
        mode = str((cfg.get("policy_rag") or {}).get("rerank_mode") or "ce_only")
    except Exception:  # noqa: BLE001
        mode = "ce_only"
    return mode if mode in ("ce_only", "blend") else "ce_only"


def best_relevance(query: str, store: Dict[str, Any]) -> Optional[float]:
    """The cross-encoder's raw score for the best of the question's 20 hybrid candidates (the user's own words, not a
    rewrite): the signal for "the knowledge base has no rule on this". Measured on the official-page corpus (Nov 2026):
    every Category C and short policy question scores -1.0 or more ("How does the HFE letter work?" +4.6), every
    off-topic one -3.7 or less ("How do I buy a condo?" -3.7, "best durian in Singapore" -8.1); raw BM25 could not
    separate them (a short policy question scored 11.5, "How do I renew my passport?" 14.0). None when the
    cross-encoder cannot load."""
    cross_encoder = _get_cross_encoder()
    if not cross_encoder:
        return None
    candidates = reciprocal_rank_fusion(retrieve_dense(query, store, top_k=20), retrieve_bm25(query, store, top_k=20),
                                        k=60, top_n=20)
    if not candidates:
        return None
    try:
        scores = cross_encoder.predict([[query, _clean_for_cross_encoder(store["chunks"][i]["text"])]
                                        for i, _ in candidates])
        return round(float(np.max(np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-50.0))), 3)
    except Exception:  # noqa: BLE001
        return None


# 7. Core Retrieval Execution (RAG-1, RAG-2, RAG-3)
# ---------------------------------------------------------------------------
def execute_rag_pipeline(
    query: str,
    policy_domain: Optional[str] = None,
    pipeline: Optional[str] = None,
    top_k: int = 5,
    rerank_mode: Optional[str] = None,
) -> RetrieveHdbPolicyKnowledgeOutput:
    """Executes the selected RAG retrieval configuration. None = the config's policy_rag.active_pipeline /
    policy_rag.rerank_mode."""
    store = _load_store()
    pipeline = pipeline or configured_pipeline()
    rerank_mode = rerank_mode or configured_rerank_mode()
    chunks = store["chunks"]

    domain = infer_domain(query, policy_domain)
    # An explicit, valid policy_domain scopes the search to that source document. This is what lets the agent
    # decompose a multi-topic question into one retrieval per domain; without it every sub-call would return the same
    # top chunks. An inferred domain only boosts (RAG-3), since a guess must not hide the right chunk.
    scope = valid_domain(policy_domain)
    allowed = {i for i, c in enumerate(chunks) if c.get("policy_category") == scope} if scope else None
    if allowed is not None and not allowed:
        allowed = None
    # Configuration selection
    if pipeline == "RAG-1":
        # RAG-1: Naive Dense Vector Search (top-5 cosine similarity over unexpanded body text)
        dense_results = retrieve_dense(query, store, top_k=top_k, naive_body_only=True, allowed=allowed)
        selected_candidates = [(idx, max(0.0, score)) for idx, score in dense_results]

    elif pipeline == "RAG-2":
        # Hybrid candidates for the question and for its corpus-vocabulary expansion (ACRONYM_EXPANSIONS, no model
        # call), fused by RRF, then reranked by the cross-encoder reading the user's own question. The expansion
        # left the 30 benchmark questions unchanged (Category E precision 0.902 -> 0.907) and brought HDB's own
        # description of the HFE letter into "How does the HFE letter work?".
        fused: Dict[int, float] = {}
        for search_query in dict.fromkeys((query, expand_query(query))):
            dense_results = retrieve_dense(search_query, store, top_k=20, allowed=allowed)
            bm25_results = retrieve_bm25(search_query, store, top_k=20, allowed=allowed)
            for idx, score in reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=20):
                fused[idx] = fused.get(idx, 0.0) + score
        rrf_results = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:20]
        selected_candidates = rerank_cross_encoder(query, rrf_results, store, top_n=top_k, rerank_mode=rerank_mode)

    else:
        # RAG-3: rewritten + HyDE retrieval, graph expansion, then cross-encoder reranking.
        pipeline = "RAG-3"
        rewritten, hyde = _rewrite_and_hypothesize(query)
        rewritten = expand_query(rewritten)
        fused: Dict[int, float] = {}
        # the user's own words are searched too: an expansion or rewrite can drop the best match (C3-01's answer
        # ranked 2nd for the plain question and fell out of the pool for the expanded one)
        for search_query in dict.fromkeys(q for q in (query, rewritten, hyde) if q):
            dense_results = retrieve_dense(search_query, store, top_k=20, allowed=allowed)
            bm25_results = retrieve_bm25(search_query, store, top_k=20, allowed=allowed)
            for idx, score in reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=20):
                fused[idx] = fused.get(idx, 0.0) + score
        graph_candidates = _graph_expand_candidates(
            sorted(fused.items(), key=lambda item: item[1], reverse=True)[:20], store, allowed=allowed
        )
        # two-phase (Topic 8 §4): the rewrite, HyDE and expansion widen the candidate pool; the cross-encoder then
        # reads the user's own question with each candidate, so expansion terms cannot steer the final order
        selected_candidates = rerank_cross_encoder(query, graph_candidates, store, top_n=30, rerank_mode=rerank_mode)

        # Domain re-weighting: if explicit domain match, slight prior boost
        if domain:
            boosted = []
            for idx, score in selected_candidates:
                c = chunks[idx]
                b_score = score * 1.08 if c.get("policy_category") == domain else score
                boosted.append((idx, b_score))
            boosted.sort(key=lambda x: x[1], reverse=True)
            max_boosted = max((score for _, score in boosted), default=1.0)
            selected_candidates = [(idx, round(score / max_boosted, 4)) for idx, score in boosted]
        # Score-drop cutoff, never more than top_k: without the cap every one of the 30 reranked candidates came
        # back, since the blended scores fall smoothly and no single step crossed the drop ratio.
        selected_candidates = _sage_gradient_select(
            selected_candidates, min_k=min(2, top_k), drop_ratio=0.30
        )[:top_k]

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
            publisher=c.get("publisher"),
            retrieved=c.get("retrieved"),
            rerank_score=round(float(score), 4),
        ))

    # raw keyword score of the best passage, so a caller can tell "nothing in the knowledge base matches" from a weak match
    best = retrieve_bm25(query, store, top_k=1)
    relevance = best_relevance(query, store)
    return RetrieveHdbPolicyKnowledgeOutput(passages=passages, pipeline=pipeline,
                                            best_keyword_score=round(best[0][1], 3) if best else None,
                                            best_relevance_score=relevance)


# ---------------------------------------------------------------------------
# 6b. Agentic decomposition: several searches merged into one result
# ---------------------------------------------------------------------------
POLICY_MAX_PASSAGES = 10   # merged across the agent's searches; the answer shows at most three sections anyway


def merge_policy_results(previous: Any, new: Any) -> Any:
    """A second policy search adds to the first instead of replacing it (agentic re-query / decomposition).
    Passages are interleaved by rank (each search's best first), duplicates dropped, at most POLICY_MAX_PASSAGES;
    best_keyword_score is the strongest of the searches, so the no-match floor still sees the best evidence."""
    if not (isinstance(previous, dict) and isinstance(new, dict)) or "error" in new:
        return previous if isinstance(previous, dict) and "error" not in previous else new
    if "error" in previous:
        return new
    runs = previous.get("_runs") or [previous.get("passages") or []]
    runs = runs + [new.get("passages") or []]
    merged, seen = [], set()
    for rank in range(max(len(r) for r in runs)):
        for r in runs:
            if rank < len(r):
                key = (r[rank].get("text") if isinstance(r[rank], dict) else str(r[rank]))
                if key not in seen:
                    seen.add(key)
                    merged.append(r[rank])
    scores = [x for x in (previous.get("best_keyword_score"), new.get("best_keyword_score")) if x is not None]
    relevances = [x for x in (previous.get("best_relevance_score"), new.get("best_relevance_score")) if x is not None]
    return {**previous, "passages": merged[:POLICY_MAX_PASSAGES], "_runs": runs,
            "best_keyword_score": max(scores) if scores else None,
            "best_relevance_score": max(relevances) if relevances else None}


def retrieve_decomposed(
    query: str,
    pipeline: Optional[str] = None,
    top_k: int = 5,
    max_calls: int = 5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The policy agent's decomposition as code: one search over the whole question, then one per sub-query of a
    question spanning several policy domains (decompose_query), merged with merge_policy_results.
    Returns (merged result, trajectory); the trajectory records each call and how many new passages it added, the
    Topic 8 "did each retrieval add new evidence" metric."""
    calls = [query] + decompose_query(query, max_parts=max_calls - 1)
    merged: Optional[Dict[str, Any]] = None
    trajectory: List[Dict[str, Any]] = []
    seen: set = set()
    for search in calls:
        result = execute_rag_pipeline(query=search, pipeline=pipeline, top_k=top_k).model_dump()
        new = [p for p in result["passages"] if p["text"] not in seen]
        seen.update(p["text"] for p in result["passages"])
        trajectory.append({"query": search[:120], "returned": len(result["passages"]), "new_passages": len(new)})
        merged = result if merged is None else merge_policy_results(merged, result)
    return merged or {"passages": [], "pipeline": pipeline, "best_keyword_score": None}, trajectory


# ---------------------------------------------------------------------------
# 7. FastMCP Tool Interface
# ---------------------------------------------------------------------------
@mcp.tool()
def retrieve_hdb_policy_knowledge(
    query: str,
    policy_domain: Optional[str] = None,
    pipeline: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Advanced Hybrid RAG Policy Knowledge Retrieval tool for official HDB and CPF policies.
    Combines BM25 sparse keyword search, dense vector embeddings, and cross-encoder reranking
    to provide high-precision, hallucination-resistant policy passages with source citations.

    policy_domain: leave empty to search every document. Set it (classification, eligibility, grants, financing,
    eip, bto_priority) to search only that document, e.g. one call per topic for a question covering several.
    """
    selected_pipeline = pipeline if pipeline in ("RAG-1", "RAG-2", "RAG-3") else configured_pipeline()
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
