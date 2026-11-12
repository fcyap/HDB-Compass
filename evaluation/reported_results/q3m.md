<!-- AUTO-RESULTS:q3m START -->
### Run `q3m`: Qwen/Qwen3-30B-A3B-GPTQ-Int4 (re-run on code `4ff2767`)

**Serving and context window**

- vLLM 0.33.0 on one 24 GB slice of an NVIDIA RTX PRO 6000 Blackwell, job 363453 (cluster-side `serve_q330b.sbatch`, same flags as the initial run); GPTQ 4-bit weights, 15.68 GiB loaded.
- Context window: native 40,960 tokens; served at **32,768** (the most the slice holds). KV cache 43,808 tokens (1.34x concurrency at 32,768). Source: the job's server log.
- Qwen thinking mode disabled (`enable_thinking=false`, ' /no_think' suffix); tool calls parsed with `hermes`. Smoke test passed (`SMOKE_OK q3m`).
- Per-call limits as in the initial runs: response cap `max_tokens` 4,096 (profile `vllm`; the constraint reading is capped at 1,536 tokens), model-call timeout 180 s, `LLM_MAX_RETRIES=1`, temperature 0, seed 42. The questions were split into 2 parallel streams (disjoint subsets of the 90 questions, A, B and C in each), so up to 2 requests reached the one server at once; the stream files were merged afterwards (last row per variant and question wins).

- Generated for the README from the merged rows with the same table code as the other blocks; the local `evaluation/analysis/checkpoint.py` was not run for this row. Code commit recorded in the rows: `4ff2767` (initial run of this model: `q330b90`, code `23c9076`).
- Source: `evaluation/results/prelim_vllm_Qwen_Qwen3-30B-A3B-GPTQ-Int4_q3m.jsonl` (270 distinct variant/question rows; a full A/B/C pass is 270), merged from 2 stream files; every stream row was `ok` on its first attempt.
- Rows whose error mentions the model's context limit: 2 (A B3-02 and B B3-02; both scored as failures). Rows timed out (`model_unreachable`): 0.

**Auditor on, A/B/C**

| Variant | Rows | OK | Timed out | Task success (Wilson 95%) | w/o Category F | Tool F1 | Latency mean / p90 (s) | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A | 90 | 90 | 0 | 21/90 (23%, 16-33%) | 19/75 (25%, 17-36%) | 0.68 | 21 / 52 | 50,766 |
| B | 90 | 90 | 0 | 21/90 (23%, 16-33%) | 21/75 (28%, 19-39%) | 0.80 | 34 / 63 | 34,065 |
| C | 90 | 90 | 0 | 76/90 (84%, 76-91%) | 61/75 (81%, 71-89%) | 0.96 | 21 / 35 | 17,664 |

| Category | A ok/n | B ok/n | C ok/n |
|---|---|---|---|
| A | 11/15 | 5/15 | 10/15 |
| B | 2/15 | 1/15 | 14/15 |
| C | 4/15 | 6/15 | 7/15 |
| D | 2/15 | 9/15 | 15/15 |
| E | 0/15 | 0/15 | 15/15 |
| F | 2/15 | 0/15 | 15/15 |
<!-- AUTO-RESULTS:q3m END -->
