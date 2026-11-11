<!-- AUTO-RESULTS:haiku55 START -->
### Run `haiku55`: claude-haiku-5-5 (hosted reference row, not Experiment 1)

**Serving and context window**

- Anthropic API (hosted, proprietary), profile `claude` in `config/hosted_models.yaml`, through `langchain-anthropic`; run with the runner's override `HDB_EVAL_ALLOW_HOSTED=claude`, which prints that these results are not Experiment 1. No GPU, quantisation, KV-cache or `--max-model-len` facts apply (hosted).
- Context window 1,000,000 tokens (published model limit). The client is built with `thinking={"type": "disabled"}` (`agents/llm.py`); no temperature, no seed (`supports_temperature: false`, `supports_seed: false`) and no effort setting are sent, so the API defaults apply. No ' /no_think' suffix and no `enable_thinking` switch are sent (both apply only to model names containing "qwen3").
- Response cap `max_tokens` 4,096 (the constraint reading is capped at 1,536 tokens), model-call timeout 180 s (`LLM_TIMEOUT_S`, recorded in the rows). Three processes ran in parallel, one per variant; each sent one request at a time.

- Generated for the README from the merged rows with the same table code as the other blocks (checked to reproduce the `gemma4nvfp90` and `q38fp8` blocks exactly); the local `evaluation/analysis/checkpoint.py` was not run for this row. Code commit recorded in the rows: `a6e8557` (the other four runs: `23c9076`).
- Source: `evaluation/results/prelim_claude_claude-haiku-5-5_haiku55.jsonl` (270 distinct variant/question rows; a full A/B/C pass is 270), merged from one results file per variant.
- Rows whose error mentions the model's context limit: 0. Rows timed out (`model_unreachable`): 0.

**Auditor on, A/B/C**

| Variant | Rows | OK | Timed out | Task success (Wilson 95%) | w/o Category F | Tool F1 | Latency mean / p90 (s) | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A | 90 | 90 | 0 | 35/90 (39%, 29-49%) | 31/75 (41%, 31-53%) | 0.72 | 13 / 24 | 74,405 |
| B | 90 | 90 | 0 | 42/90 (47%, 37-57%) | 39/75 (52%, 41-63%) | 0.77 | 15 / 24 | 76,184 |
| C | 90 | 90 | 0 | 76/90 (84%, 76-91%) | 61/75 (81%, 71-89%) | 0.94 | 12 / 23 | 30,879 |

| Category | A ok/n | B ok/n | C ok/n |
|---|---|---|---|
| A | 12/15 | 10/15 | 11/15 |
| B | 6/15 | 4/15 | 14/15 |
| C | 10/15 | 10/15 | 9/15 |
| D | 2/15 | 13/15 | 12/15 |
| E | 1/15 | 2/15 | 15/15 |
| F | 4/15 | 3/15 | 15/15 |
<!-- AUTO-RESULTS:haiku55 END -->
