# Member 3 Presentation Script: Policy RAG

**Target length: about 2 minutes 30 seconds.** Follow the pipeline from left to right on the whiteboard: corpus → chunking → embedding → retrieval → generation → evaluation. The experiment label appears with the stage it tests.

## Spoken script

### Corpus — about 20 seconds

Now I will bring you through our Policy Agent which uses RAG for answering HDB policy questions. We have a corpus of 53 official HDB, CPF and IRAS pages and for ground truth, a fact sheet of 47 facts, each linked to an exact quote from an official page is prepared for evaluation.

### Chunking — Experiments 2A and 2B — about 30 seconds

We conducted 2 experiments on chunking, experiments 2A and 2B. Experiment 2A compared three chunking methods: fixed 256-token chunks, fixed 1,024-token chunks, and semantic chunking. The results suggest semantic chunking is the best fit. Semantic chunking groups text by meaning instead of cutting at a fixed length. In our SAGE-inspired approach the sentence embeddings detect topic shifts and a new chunk starts when neighbouring sentences become less similar. In Experiment 2B, similarity threshold tuning of the sentences were done.

### Embedding — Experiment 2C — about 20 seconds

Then I compared MiniLM, BGE-small and BGE-base.

<!-- In dense-only search, which is the embedding part of RAG-1, recall was 0.63, 0.73 and 0.65 respectively, so the models were not tied;  -->

BGE-small did the best on its own. But when tested inside RAG-2 the recall was effectively tied at 0.82 for all 3 embedding models. MiniLM had the best precision, 0.91, and lowest query-embedding time, so I kept it for the full pipeline.

### Retrieval — Experiment 2D, RAG-1, RAG-2 and RAG-3 — about 35 seconds

For retrieval, I compared three pipelines. RAG-1 uses dense search only and reached 0.42 recall. RAG-2 combines keyword search, dense search, rank fusion and a cross-encoder reranker; it reached 0.82 recall and 0.91 precision in about 0.4 seconds. RAG-3 adds query rewriting, HyDE and graph expansion. It reached 0.78 recall and took about 4.9 seconds. Since RAG-3 was slower and did not improve retrieval, I chose RAG-2.
<!-- Why is RAG-3 worse when it is an add-on from RAG-2? -->

### Generation — Experiment 2E — about 30 seconds

Experiment 2E tests generation. Gemini answers from RAG-2's passages, and a long-context baseline gets all 53 pages instead. A judge from a different model family scores them with RAGAS. On the policy questions, long context was slightly more faithful, 0.90 against 0.86, because our pages still fit in the window. But it uses about fifty times more tokens per answer and stops working as the corpus grows, so we keep RAG-2 and focus on improving its retrieval.

### Evaluation and close — about 20 seconds

Across the pipeline, retrieval is checked against the official pages, and at the final stage we assess whether answers are faithful to their evidence and relevant to the question.

## Whiteboard cues

Write the pipeline across the board, then add one result beneath each experiment as you reach it:

| Stage | Board note |
|---|---|
| Corpus | 53 official pages · 47 exact-quote facts |
| Chunking, 2A–2B | Fixed 256 vs semantic vs fixed 1,024 · semantic selected · threshold 0.35 |
| Embedding, 2C | Dense-only: 0.63 / 0.73 / 0.65 · in RAG-2: 0.82 / 0.81 / 0.82 |
| Retrieval, 2D | RAG-1 0.42 · RAG-2 0.82 · RAG-3 0.78 |
| Generation, 2E | Faithfulness RAG-2 0.86 vs long context 0.90 · ~2k vs ~104k tokens |
| Evaluation | Exact-clause retrieval checks · faithfulness · relevancy · coverage · citations |

## Delivery notes

- Say that the 0.82/0.91 retrieval result is recall/precision across the 30-question benchmark.
- In Experiment 2C, dense-only means the embedder's standalone search, as in RAG-1; RAG-2 measures the same embedders inside the full hybrid pipeline.
- The 2A numbers come from the chunking comparison; the retrieval pipeline comparison is 2D. Keep those experiment labels attached to the right figures.
- The 2E scores are Category C (policy questions); if asked about Category E, its low relevancy comes from questions that also ask for listings and budgets, which other agents answer.
