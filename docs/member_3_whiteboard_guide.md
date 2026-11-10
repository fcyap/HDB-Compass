# Member 3 Whiteboard Guide: Policy RAG & Agent 5

> **Lesson 12 Policy:** No LLM-generated scripts, no phones, no notes, limited/no slides. Each student must explain
> their own technical contribution unaided using the whiteboard.
>
> This guide is preparation material: what to draw, what to cover in 2.5 minutes, and the facts behind each point.
> It deliberately has no script. Put the points in your own words and rehearse them until you don't need this page.

---

## 1. The diagram to draw (about 30 seconds of drawing, build it while you talk)

```
 OFFLINE                                              ONLINE (per question)
 ───────                                              ─────────────────────
 53 official pages (HDB, CPF, IRAS)                   question
   │  captured, kept word for word                      │
   ▼                                                    ├──► BM25 (exact terms: MOP, EIP, $8,000)
 chunker: sections + tables whole,                      ├──► dense cosine (MiniLM, paraphrases)
 cut at low-similarity sentences                        ▼
   │  1,085 chunks                                   RRF fusion → top 20
   ├──► BM25 index                                      ▼
   └──► embedding matrix (numpy)                     cross-encoder reads question + chunk → top 5
                                                        ▼
                                                     "no rule" gate ─► Agent 5 answers with citations
```

A label worth writing beside it: **"fact sheet: 47 facts, exact quotes"** under the pages.

---

## 2. Timed outline (2 minutes 30 seconds)

| Time | Point to make | What to show on the board |
|---|---|---|
| 0:00–0:20 | **The problem.** My part answers HDB policy questions (MOP, grants, eligibility, EIP). A wrong figure here misleads a buyer, so answers must come only from official sources and cite them. | Write "policy questions → must be grounded + cited". |
| 0:20–0:55 | **The corpus.** 53 official pages (HDB, CPF Board, IRAS), kept verbatim: one document per page with its URL and capture date, so every answer cites the exact page. HDB blocks scripts, so its pages were saved by hand; CPF pages are fetched by script. The ground truth is a fact sheet of 47 facts, each an exact quote that a test checks against its page. | Draw the left (offline) half. |
| 0:55–1:35 | **Retrieval.** Hybrid search: BM25 catches exact terms and numbers, dense embeddings catch paraphrases; rank fusion merges them; a cross-encoder then reads the question with each candidate and keeps the best 5. Chunking keeps tables whole and never splits a lead-in from its list. | Draw the right (online) half, top to bottom. |
| 1:35–2:10 | **Evidence.** Three pipelines on 30 benchmark questions: dense-only (RAG-1), hybrid + cross-encoder (RAG-2), and RAG-2 plus query rewriting and HyDE (RAG-3). RAG-2 reaches 0.82 recall vs 0.42 for dense-only; RAG-3 was not better and was over 10x slower: measure before adding complexity. | Write the RAG-1/2/3 recall numbers in a small table. |
| 2:10–2:30 | **Safety and handover.** A relevance gate (cross-encoder score) makes the agent say "no HDB rule found" instead of guessing; Agent 5 answers only from the passages with source links, and hands them to the auditor. | Point to the gate and Agent 5 at the bottom. |

Pacing check: about 60–70 spoken words per 20 seconds. If you run long, drop the chunking sentence in 0:55–1:35 first.

---

## 3. Facts to have ready (all checked against the code and the 10 Nov 2026 run)

### Corpus
- **Corpus:** 53 pages (41 HDB, 8 CPF, IRAS stamp duty, 3 HDB press releases), 1,085 chunks.
  - HDB pages saved by hand (HDB returns 403 to scripts; we did not get around its bot protection); the HTML stores
    the article in a JSON block, which the extractor reads.
  - Text kept verbatim; only page furniture removed. Navigation-only pages excluded.
- **Fact sheet:** `evaluation/policy_fact_sheet.yaml`, 47 facts, each with an exact quote; `tests/test_policy_corpus.py`
  checks every quote is on its page. Category C ground truth is built from it.

### Retrieval (`mcp_server/policy_rag_tools.py`)
- **Embeddings:** `all-MiniLM-L6-v2`, 384 dimensions, stored as a numpy matrix in `data/policy_store.pkl`;
  search is one matrix-vector product (about 0.05 ms at 1,085 chunks). No vector DB needed at this size.
- **BM25:** `rank_bm25` BM25Okapi.
- **Query expansion:** acronyms and key terms expanded in HDB's own wording (no figures; a test checks every
  expansion word is in the corpus); both the question and its expansion are searched, reranking uses the question.
- **Fusion:** Reciprocal Rank Fusion, k = 60: RRF(d) = Σ 1 / (60 + rank of d in each list); top 20.
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, cross-encoder score only (config `rerank_mode: ce_only`).
  During development a blended score (cross-encoder + fusion + word overlap) was tried and dropped: it let every chunk
  of a page whose title matched look relevant (Category E recall 0.55 vs 0.83).
- **Chunking:** split by headings; tables never split; prose cut where adjacent sentences' similarity drops below
  0.35 (chosen on Category C by a sweep); a line ending ":" stays with its list.
- **"No rule" gate:** best cross-encoder score below -3.2 → "I couldn't find an HDB rule". Raw BM25 could not separate
  short policy questions from off-topic ones.

### Evaluation (`evaluation/run_rag_ablation.py`, 15 Category C + 15 Category E questions)

| Pipeline | Cat C recall | Cat E recall | Combined recall / precision | Speed |
|---|---|---|---|---|
| RAG-1 dense only | 0.57 | 0.27 | 0.42 / 0.62 | 0.2 s |
| RAG-2 hybrid + cross-encoder (the app's pipeline) | 0.81 | 0.83 | 0.82 / 0.91 | 0.4 s |
| RAG-3 rewrite + HyDE + graph | 0.81 | 0.77 | 0.79 / 0.87 | 5.4 s |

- Stronger embedders (BGE-small, BGE-base) gave no clear gain in the full pipeline: BM25 + cross-encoder do most of the
  work.
- Recall/precision here are clause-matching (is the gold clause in the retrieved text?), the RAGAS context-recall idea
  computed without an LLM. `--llm` adds an LLM judge (Groq gpt-oss-120b, a different model family from the Gemini
  generator) for faithfulness and relevance.

---

## 4. Questions you may get

- **"Why not just put all the documents in the prompt (long context)?"** About 79,000 words fits in a large
  context window, and the ablation has a long-context baseline (`--llm`). Retrieval keeps each answer tied to a cited
  page, and 'lost in the middle' makes huge prompts unreliable.
- **"Why not ChromaDB?"** At 1,085 chunks brute-force cosine takes 0.05 ms; an approximate index only pays off at
  hundreds of thousands of chunks. The unused Chroma copy was removed.
- **"Is the agent agentic?"** Retrieval is a tool the policy agent calls in a capped ReAct loop (max 5 searches), with
  decomposition for multi-topic questions. Decomposition was measured during development and did not beat plain RAG-2,
  so the evaluation keeps the three pipelines RAG-1/2/3.
- **"How do you know the answers are right?"** Ground truth comes from the fact sheet: each fact is an exact quote from
  an official page, and a test checks the quote is on that page, so the evaluation never relies on memory.
- **"What's still weak?"** Chunks of one page can split an answer: for C3-01 the search ranks HDB's 2026 wait-out
  removal page first, but the chunk it returns (paragraph 7) lacks the "no longer applies" and "55 and above" lines.
  The page has 7 near-identical copies of the Couples and Families disposal table (one per tab), which crowd the
  candidate pool. HDB's EIP buyback source is a 2022 announcement.
