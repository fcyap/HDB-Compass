<!-- AUTO-RESULTS:gptoss90 START -->
### Run `gptoss90`: openai/gpt-oss-20b

**Serving and context window**

- vLLM 0.33.0 on one 24 GB slice of an NVIDIA RTX PRO 6000 Blackwell; MXFP4 weights, mixture of experts (32 experts, 4 active per token), alternating full and 128-token sliding-window attention.
- Context window: the model's native maximum is 131,072 tokens; served at **65,536**. Attention backend forced to Triton (FlashInfer cannot be built on the cluster nodes); tool calls parsed with `openai`, reasoning with `openai_gptoss`. KV cache size: see the server log.
- Default reasoning effort was used. The Qwen-specific ' /no_think' suffix and `enable_thinking=false` were not sent: `agents/llm.py` (at `23c9076`) drops both for model names without "qwen3" (corrected 12 Nov from code; the rows do not record the suffix).
- Per-call limits for every model: response cap `max_tokens` 4,096 (profile `vllm` in config/experiment_config.yaml; the constraint reading is capped at 1,536 tokens), model-call timeout 180 s, temperature 0, seed 42, one request at a time.

- Checkpoint: 2026-11-11 11:55 UTC, label: **run finished (final)**. Code commit recorded in the rows: `23c9076`.
- Source: `evaluation/results/prelim_vllm_openai_gpt-oss-20b_gptoss90.jsonl` (270 distinct variant/question rows; a full A/B/C pass is 270).
- Rows whose error mentions the model's context limit: 0. Rows timed out (`model_unreachable`): 0 (rerun on resume).
- This block is generated from the result rows by `evaluation/analysis/checkpoint.py`; the numbers are not typed in.

**Auditor on, A/B/C**

| Variant | Rows | OK | Timed out | Task success (Wilson 95%) | w/o Category F | Tool F1 | Latency mean / p90 (s) | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A | 90 | 90 | 0 | 14/90 (16%, 9-24%) | 14/75 (19%, 11-29%) | 0.61 | 32 / 79 | 33,358 |
| B | 90 | 90 | 0 | 23/90 (26%, 18-35%) | 21/75 (28%, 19-39%) | 0.77 | 68 / 135 | 33,274 |
| C | 90 | 90 | 0 | 73/90 (81%, 72-88%) | 58/75 (77%, 67-85%) | 0.94 | 72 / 143 | 20,009 |

| Category | A ok/n | B ok/n | C ok/n |
|---|---|---|---|
| A | 5/15 | 3/15 | 12/15 |
| B | 4/15 | 4/15 | 13/15 |
| C | 3/15 | 5/15 | 6/15 |
| D | 2/15 | 9/15 | 12/15 |
| E | 0/15 | 0/15 | 15/15 |
| F | 0/15 | 2/15 | 15/15 |
<!-- AUTO-RESULTS:gptoss90 END -->
