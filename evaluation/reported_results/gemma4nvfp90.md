<!-- AUTO-RESULTS:gemma4nvfp90 START -->
### Run `gemma4nvfp90`: nvidia/Gemma-4-26B-A4B-NVFP4


- Checkpoint: 2026-11-11 13:20 UTC, label: **run finished (final)**. Code commit recorded in the rows: `23c9076`.
- Source: `evaluation/results/prelim_vllm_nvidia_Gemma-4-26B-A4B-NVFP4_gemma4nvfp90.jsonl` (270 distinct variant/question rows; a full A/B/C pass is 270).
- Rows whose error mentions the model's context limit: 0. Rows timed out (`model_unreachable`): 0 (rerun on resume).
- This block is generated from the result rows by `evaluation/analysis/checkpoint.py`; the numbers are not typed in.

**Auditor on, A/B/C**

| Variant | Rows | OK | Timed out | Task success (Wilson 95%) | w/o Category F | Tool F1 | Latency mean / p90 (s) | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| A | 90 | 90 | 0 | 27/90 (30%, 22-40%) | 20/75 (27%, 18-38%) | 0.70 | 9 / 16 | 37,269 |
| B | 90 | 90 | 0 | 31/90 (34%, 25-45%) | 29/75 (39%, 28-50%) | 0.76 | 21 / 32 | 37,583 |
| C | 90 | 90 | 0 | 73/90 (81%, 72-88%) | 58/75 (77%, 67-85%) | 0.95 | 16 / 31 | 16,058 |

| Category | A ok/n | B ok/n | C ok/n |
|---|---|---|---|
| A | 10/15 | 7/15 | 10/15 |
| B | 2/15 | 3/15 | 14/15 |
| C | 6/15 | 6/15 | 7/15 |
| D | 2/15 | 11/15 | 12/15 |
| E | 0/15 | 2/15 | 15/15 |
| F | 7/15 | 2/15 | 15/15 |
<!-- AUTO-RESULTS:gemma4nvfp90 END -->
