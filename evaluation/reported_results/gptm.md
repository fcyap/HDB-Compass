<!-- AUTO-RESULTS:gptm START -->
### Run `gptm`: openai/gpt-oss-20b (re-run on code `4ff2767`)

**Serving and context window**

- vLLM 0.33.0 on one 24 GB slice of an NVIDIA RTX PRO 6000 Blackwell, job 363449 (cluster-side `serve_gptoss.sbatch`, same flags as the initial run); MXFP4 weights, 13.8 GiB loaded.
- Context window: native 131,072 tokens; served at **65,536**. KV cache 160,383 tokens (2.45x concurrency at 65,536). Attention backend Triton; tool calls parsed with `openai`, reasoning with `openai_gptoss`. Source: the job's server log.
- Default reasoning effort; the Qwen-specific switches were not sent (`agents/llm.py` drops them for model names without "qwen3"). Smoke test passed (`SMOKE_OK gptm`).
- Per-call limits as in the initial runs: response cap `max_tokens` 4,096 (profile `vllm`; the constraint reading is capped at 1,536 tokens), model-call timeout 180 s, `LLM_MAX_RETRIES=1`, temperature 0, seed 42. The questions were split into 3 parallel streams (disjoint subsets of the 90 questions, A, B and C in each), so up to 3 requests reached the one server at once; the stream files were merged afterwards (last row per variant and question wins).

- Generated for the README from the merged rows with the same table code as the other blocks; the local `evaluation/analysis/checkpoint.py` was not run for this row. Code commit recorded in the rows: `4ff2767` (initial run of this model: `gptoss90`, code `23c9076`).
- Source: `evaluation/results/prelim_vllm_openai_gpt-oss-20b_gptm.jsonl` (270 distinct variant/question rows; a full A/B/C pass is 270), merged from 3 stream files; every stream row was `ok` on its first attempt.
- Rows whose error mentions the model's context limit: 0. Rows timed out (`model_unreachable`): 0.

**Auditor on, A/B/C**

| Variant | Rows | OK | Timed out | Task success (Wilson 95%) | w/o Category F | Tool F1 | Latency mean / p90 (s) | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A | 90 | 90 | 0 | 12/90 (13%, 8-22%) | 12/75 (16%, 9-26%) | 0.62 | 50 / 112 | 35,535 |
| B | 90 | 90 | 0 | 23/90 (26%, 18-35%) | 20/75 (27%, 18-38%) | 0.76 | 115 / 225 | 35,986 |
| C | 90 | 90 | 0 | 75/90 (83%, 74-90%) | 60/75 (80%, 70-87%) | 0.94 | 107 / 217 | 20,179 |

| Category | A ok/n | B ok/n | C ok/n |
|---|---|---|---|
| A | 5/15 | 1/15 | 11/15 |
| B | 4/15 | 5/15 | 14/15 |
| C | 2/15 | 3/15 | 7/15 |
| D | 1/15 | 11/15 | 13/15 |
| E | 0/15 | 0/15 | 15/15 |
| F | 0/15 | 3/15 | 15/15 |
<!-- AUTO-RESULTS:gptm END -->
