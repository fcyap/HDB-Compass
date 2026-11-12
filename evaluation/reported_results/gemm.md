<!-- AUTO-RESULTS:gemm START -->
### Run `gemm`: nvidia/Gemma-4-26B-A4B-NVFP4 (re-run on code `4ff2767`)

**Serving and context window**

- vLLM 0.33.0 on an NVIDIA L40S (46 GB), job 363448 (cluster-side `serve_gemma4.sbatch`, same flags as the initial run); NVFP4 weights, 18.12 GiB loaded.
- Context window: native 262,144 tokens; served at **65,536**. KV cache 1,199,893 tokens (18.31x concurrency at 65,536). Source: the job's server log.
- Tool and reasoning parser `gemma4`, `--max-num-seqs 16`. The Qwen-specific ' /no_think' suffix and `enable_thinking=false` were not sent (`agents/llm.py` drops both for model names without "qwen3"). The repository smoke test was skipped on purpose, as in the initial run.
- Per-call limits as in the initial runs: response cap `max_tokens` 4,096 (profile `vllm`; the constraint reading is capped at 1,536 tokens), model-call timeout 180 s, `LLM_MAX_RETRIES=1`, temperature 0, seed 42. The questions were split into 5 parallel streams (disjoint subsets of the 90 questions, A, B and C in each), so up to 5 requests reached the one server at once; the stream files were merged afterwards (last row per variant and question wins).

- Generated for the README from the merged rows with the same table code as the other blocks; the local `evaluation/analysis/checkpoint.py` was not run for this row. Code commit recorded in the rows: `4ff2767` (initial run of this model: `gemma4nvfp90`, code `23c9076`).
- Source: `evaluation/results/prelim_vllm_nvidia_Gemma-4-26B-A4B-NVFP4_gemm.jsonl` (270 distinct variant/question rows; a full A/B/C pass is 270), merged from 5 stream files; every stream row was `ok` on its first attempt.
- Rows whose error mentions the model's context limit: 0. Rows timed out (`model_unreachable`): 0.

**Auditor on, A/B/C**

| Variant | Rows | OK | Timed out | Task success (Wilson 95%) | w/o Category F | Tool F1 | Latency mean / p90 (s) | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A | 90 | 90 | 0 | 26/90 (29%, 21-39%) | 19/75 (25%, 17-36%) | 0.69 | 16 / 25 | 35,372 |
| B | 90 | 90 | 0 | 33/90 (37%, 27-47%) | 32/75 (43%, 32-54%) | 0.78 | 34 / 45 | 38,084 |
| C | 90 | 90 | 0 | 72/90 (80%, 71-87%) | 57/75 (76%, 65-84%) | 0.95 | 22 / 37 | 16,031 |

| Category | A ok/n | B ok/n | C ok/n |
|---|---|---|---|
| A | 9/15 | 8/15 | 9/15 |
| B | 1/15 | 5/15 | 14/15 |
| C | 6/15 | 7/15 | 7/15 |
| D | 3/15 | 10/15 | 12/15 |
| E | 0/15 | 2/15 | 15/15 |
| F | 7/15 | 1/15 | 15/15 |
<!-- AUTO-RESULTS:gemm END -->
