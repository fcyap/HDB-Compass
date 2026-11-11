"""
HDB Policy Regulatory Corpus Ingestion & Hierarchical Chunking (data/ingest_hdb_policy_rag.py)
Proposal Section 3.3 (Page 5) & File Structure (Page 14).
Owned by Member 3: Advanced RAG & Regulatory Knowledge Base (Category C).

Features:
1. Hierarchical, SAGE-inspired semantic segmentation:
   - Preserves section hierarchy and Markdown tables
   - Groups adjacent sentences by embedding similarity into focused chunks
   - Uses a similarity fallback when the sentence embedding model is unavailable
2. Rich Metadata Tagging:
   - policy_category (classification, eligibility, grants, financing, eip)
   - flat_classification (Standard, Plus, Prime, or None)
   - effective_date, source_url, section breadcrumbs, keywords/acronyms
3. Dual Index Generation:
   - BM25 sparse index (using rank_bm25.BM25Okapi)
   - Dense vector embeddings (SentenceTransformer all-MiniLM-L6-v2 with deterministic fallback)
4. Saves precomputed bundle to:
   - data/policy_store.pkl (chunks, metadata, BM25 model, dense embeddings)
   - data/policy_chunks.json (human-inspectable metadata & text)
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "policy_documents"
STORE_PKL = DATA_DIR / "policy_store.pkl"
CHUNKS_JSON = DATA_DIR / "policy_chunks.json"
_SEGMENTATION_MODEL: Any = None
SAGE_SIMILARITY_THRESHOLD = 0.35   # chosen on Category C by the Nov 2026 sweep (Experiment 2B, run_rag_ablation) over the official-page corpus

# Acronyms and domain keywords to explicitly index
DOMAIN_KEYWORDS = [
    "MOP", "EIP", "HFE", "EHG", "PHG", "BTO", "CPF", "MSR", "TDSR", "LTV", "BSD",
    "PRIME", "PLUS", "STANDARD", "FLEXI", "CLAWBACK", "SUBSIDY RECOVERY",
    "WAIT-OUT", "15-MONTH", "30-MONTH", "SINGLES SCHEME", "JOINT SINGLES",
    "BUYBACK", "RIGHT-SIZING", "SENIOR", "AGE 35", "AGE 55"
]


def extract_frontmatter(content: str, filename: str = "") -> Tuple[Dict[str, str], str]:
    """Extract metadata header fields like Source URL, Effective Date, Policy Category."""
    meta = {}
    lines = content.splitlines()
    body_lines = []
    in_preamble = True

    for line in lines:
        if in_preamble:
            match = re.match(r"\*\*(.+?):\*\*\s*(.+)", line.strip())
            if match:
                key = match.group(1).lower().replace(" ", "_")
                val = match.group(2).strip()
                meta[key] = val
                continue
            elif line.startswith("## "):
                in_preamble = False
        body_lines.append(line)

    # Filename fallback if not in header
    fn = filename.lower()
    if "policy_category" not in meta:
        if "classification" in fn:
            meta["policy_category"] = "classification"
        elif "eligibility" in fn:
            meta["policy_category"] = "eligibility"
        elif "grant" in fn:
            meta["policy_category"] = "grants"
        elif "financ" in fn:
            meta["policy_category"] = "financing"
        elif "eip" in fn:
            meta["policy_category"] = "eip"
        else:
            meta["policy_category"] = "general"

    return meta, "\n".join(body_lines)


def tokenize_words(text: str) -> List[str]:
    """Lowercased token list preserving alphanumeric tokens and hyphens."""
    return [t.lower() for t in re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b", text)]


def detect_classification(text: str) -> Optional[str]:
    """Identify if a chunk specifically targets Prime, Plus, or Standard flat rules."""
    t = text.lower()
    has_prime = "prime" in t
    has_plus = "plus" in t
    has_std = "standard" in t
    if has_prime and not (has_plus or has_std):
        return "Prime"
    if has_plus and not (has_prime or has_std):
        return "Plus"
    if has_std and not (has_prime or has_plus):
        return "Standard"
    if has_prime and has_plus and not has_std:
        return "Prime / Plus"
    return None


def detect_keywords(text: str) -> List[str]:
    """Extract present domain keywords from text."""
    t_upper = text.upper()
    return [kw for kw in DOMAIN_KEYWORDS if kw in t_upper]


def split_markdown_into_hierarchical_sections(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Splits markdown by headers while maintaining hierarchical breadcrumbs.
    Never splits markdown tables internally.
    """
    lines = markdown_text.splitlines()
    sections: List[Dict[str, Any]] = []

    doc_title = "HDB Policy"
    h2_title = ""
    h3_title = ""
    current_content: List[str] = []

    def flush_section():
        nonlocal current_content
        text = "\n".join(current_content).strip()
        if text:
            breadcrumb = " > ".join(filter(None, [doc_title, h2_title, h3_title]))
            sections.append({
                "breadcrumb": breadcrumb,
                "h2": h2_title,
                "h3": h3_title,
                "text": text,
            })
        current_content = []

    for line in lines:
        if line.startswith("# "):
            flush_section()
            doc_title = line[2:].strip()
            h2_title = ""
            h3_title = ""
        elif line.startswith("## "):
            flush_section()
            h2_title = line[3:].strip()
            h3_title = ""
        elif line.startswith("### "):
            flush_section()
            h3_title = line[4:].strip()
        else:
            current_content.append(line)

    flush_section()
    return sections


def _sentence_units(text: str) -> List[str]:
    """Split prose/list text into sentence-sized units without breaking Markdown list items."""
    units: List[str] = []
    in_list_after_lead = False   # the last unit is a lead-in ending ":" (with any list items already joined to it)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line):
            if in_list_after_lead and units:
                # a list completes the lead-in above it ("... eligible to:" + its items): one unit, never split
                units[-1] += "\n" + line
                continue
            units.append(line)
        else:
            units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line) if part.strip())
        in_list_after_lead = bool(units) and units[-1].rstrip().endswith(":")
    return units


def _adjacent_similarity(units: List[str]) -> List[float]:
    """Similarity for adjacent sentences, using MiniLM or a deterministic token-overlap fallback."""
    global _SEGMENTATION_MODEL
    if len(units) < 2:
        return []
    if _SEGMENTATION_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            try:
                _SEGMENTATION_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception:
                _SEGMENTATION_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            print(f"[Warning] SAGE semantic segmenter unavailable ({exc}); using token-overlap boundaries.")
            _SEGMENTATION_MODEL = False
    if _SEGMENTATION_MODEL:
        vectors = _SEGMENTATION_MODEL.encode(units, convert_to_numpy=True, normalize_embeddings=True)
        return [float(np.dot(vectors[i], vectors[i + 1])) for i in range(len(units) - 1)]
    tokens = [set(tokenize_words(unit)) for unit in units]
    return [
        len(tokens[i] & tokens[i + 1]) / max(1, len(tokens[i] | tokens[i + 1]))
        for i in range(len(units) - 1)
    ]


_LIST_ITEM = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")


def _join_units(units: List[str]) -> str:
    """Units of one chunk as text: sentences of a paragraph joined by a space, but a list item (or a lead-in carrying
    its items) on a line of its own, so an item never runs into the sentence after it ("... overseas The MOP starts")."""
    text = ""
    for unit in units:
        if text:
            last_line = text.splitlines()[-1].strip()
            separate = "\n" in unit or _LIST_ITEM.match(unit) or _LIST_ITEM.match(last_line)
            text += "\n" if separate else " "
        text += unit
    return text


def _semantic_chunks(text: str, max_words: int, threshold: float = SAGE_SIMILARITY_THRESHOLD) -> List[str]:
    """Coarsely bound prose, then cut at low-similarity adjacent sentence boundaries."""
    units = _sentence_units(text)
    if not units:
        return []
    similarities = _adjacent_similarity(units)
    result: List[str] = []
    current: List[str] = []
    current_words = 0
    for i, unit in enumerate(units):
        count = len(unit.split())
        # never cut right after a lead-in ("... you will know upfront if you are eligible to:"): the list that follows
        # completes it, and a lead-in and its items read as dissimilar sentences (Topic 8 / SAGE failure A: a chunk
        # boundary splitting the answer)
        after_lead_in = bool(current) and current[-1].rstrip().endswith(":")
        if current and (current_words + count > max_words or (similarities[i - 1] < threshold and not after_lead_in)):
            result.append(_join_units(current))
            current = []
            current_words = 0
        current.append(unit)
        current_words += count
    if current:
        result.append(_join_units(current))
    return result


def chunk_section(section: Dict[str, Any], max_words: int = 380, threshold: Optional[float] = None) -> List[str]:
    """
    SAGE-inspired segmentation: preserve section breadcrumbs and tables, split prose at low-similarity sentence boundaries.
    """
    text = section["text"]
    breadcrumb = section["breadcrumb"]

    # If section contains a table, isolate table block
    parts = []
    current_block: List[str] = []
    in_table = False

    for line in text.splitlines():
        if "|" in line:
            if not in_table:
                if current_block:
                    parts.append("\n".join(current_block))
                    current_block = []
                in_table = True
            current_block.append(line)
        else:
            if in_table:
                parts.append("\n".join(current_block))
                current_block = []
                in_table = False
            current_block.append(line)

    if current_block:
        parts.append("\n".join(current_block))

    chunks: List[str] = []
    for part in parts:
        part = re.sub(r"(?m)^\s*---+\s*$", "", part).strip()
        if not part:
            continue
        if "|" in part:
            # Tables remain atomic so a retrieved cell is never separated from its headers/row labels.
            subparts = [part]
        else:
            subparts = _semantic_chunks(part, max_words=max_words,
                                        threshold=SAGE_SIMILARITY_THRESHOLD if threshold is None else threshold)
        for sub_text in subparts:
            chunks.append(f"[{breadcrumb}]\n{sub_text}")

    return chunks


def build_policy_corpus(docs_dir: Path, threshold: Optional[float] = None, max_words: int = 380) -> List[Dict[str, Any]]:
    """Parse all markdown files in docs_dir into typed, metadata-enriched chunks."""
    all_chunks: List[Dict[str, Any]] = []
    md_files = sorted(docs_dir.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No markdown documents found in {docs_dir}")

    chunk_idx = 1
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        meta, body = extract_frontmatter(content, filename=md_file.name)
        sections = split_markdown_into_hierarchical_sections(body)

        for sec in sections:
            sub_chunks = chunk_section(sec, max_words=max_words, threshold=threshold)
            for chk_text in sub_chunks:
                cat = meta.get("policy_category", "general")
                url = meta.get("source_url", "https://www.hdb.gov.sg")
                # the date the page states; the rebuilt corpus writes it as "Page Date" ("not stated on the page"
                # when it has none), the legacy documents as "Effective Date". Never a made-up default.
                eff_date = meta.get("effective_date") or meta.get("page_date")
                if eff_date and not re.match(r"\d{4}-\d{2}-\d{2}$", eff_date):
                    eff_date = None
                flat_cls = detect_classification(chk_text)
                kws = detect_keywords(chk_text)

                record = {
                    "chunk_id": f"hdb_policy_{chunk_idx:03d}",
                    "text": chk_text,
                    "section": sec["breadcrumb"],
                    "policy_category": cat,
                    "flat_classification": flat_cls,
                    "effective_date": eff_date,
                    "source_url": url,
                    "keywords": kws,
                    "doc_filename": md_file.name,
                    # provenance from the page header written by data/policy_sources/extract_sources.py (absent in
                    # the legacy hand-written documents)
                    "publisher": meta.get("publisher"),
                    "retrieved": meta.get("retrieved"),
                    "chunking_method": "sage_semantic_similarity",
                }
                all_chunks.append(record)
                chunk_idx += 1

    return all_chunks


def compute_dense_embeddings(chunks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Computes dense vector representations for all chunks using SentenceTransformer.
    Falls back to a deterministic normalized subword TF-IDF / character n-gram matrix if torch/ST fails.
    """
    texts = [c["text"] for c in chunks]

    # Try SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
        # Try local cache first (instant, offline-safe)
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            # If not cached yet (e.g. teammate's first run), download once from Hugging Face Hub
            model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Generating dense embeddings using SentenceTransformer(all-MiniLM-L6-v2) for {len(texts)} chunks...")
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)
    except Exception as e:
        print(f"[Warning] SentenceTransformer load failed ({e}); building deterministic sklearn TF-IDF dense embeddings.")

    # High-dimension L2-normalized TF-IDF fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=384, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    # Ensure L2 normalized
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def ingest_policy_knowledge():
    """Main ingestion pipeline execution."""
    print(f"=== HDB-Compass Policy RAG Ingestion (Member 3) ===")
    print(f"Reading policy documents from: {DOCS_DIR}")

    chunks = build_policy_corpus(DOCS_DIR)
    print(f"Extracted {len(chunks)} SAGE-inspired semantic policy chunks.")

    # 1. Build BM25 Index
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize_words(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("Built BM25 Okapi sparse index.")

    # 2. Build Dense Embeddings
    dense_embeddings = compute_dense_embeddings(chunks)
    print(f"Generated dense embeddings matrix of shape: {dense_embeddings.shape}")

    # 3. Save to disk
    store_data = {
        "chunks": chunks,
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus,
        "dense_embeddings": dense_embeddings,
        "embedding_model": "all-MiniLM-L6-v2",
        "num_chunks": len(chunks),
    }

    with open(STORE_PKL, "wb") as f:
        pickle.dump(store_data, f)
    print(f"Saved binary policy store to: {STORE_PKL} ({STORE_PKL.stat().st_size / 1024:.1f} KB)")

    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved human-readable chunks to: {CHUNKS_JSON}")

    # Vector store: the normalised embedding matrix above, searched by brute-force cosine (one matrix-vector product,
    # about 0.05 ms over ~1,100 chunks). A Chroma copy was written here earlier but no code read it; an ANN index only
    # pays off at hundreds of thousands of chunks (Topic 8, "storing & searching the vectors").


    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest_policy_knowledge()
