# Reported results (Experiment 1, live runs)

The raw result rows (`evaluation/results/prelim_*.jsonl`, 3-12 MB per run) are git-ignored. This folder holds what the
main README cites: per-run summaries and a slim per-question CSV, so the principal quantitative comparisons can be checked without the
raw rows. Detailed behavioural claims use selected evidence extracts or the original rows. Code commit in every row of the four
open-weight initial runs: `23c9076`; in every row of the four re-runs (`q38m`, `q38m-noauditor`, `q3m`, `gptm`,
`gemm`): `4ff2767`; the hosted reference row `haiku55` records `a6e8557` (see below). Benchmark
`evaluation/benchmark_hdb_90.json` (sha `0bbca6193b0a`) in all nine.

| Tag | Model | Status | Files |
|---|---|---|---|
| `q38fp8` | Qwen/Qwen3.8-27B-FP8 (headline) | complete: A, B, C 90 rows each | `q38fp8.csv`, `q38fp8.md`, `architecture_eval_q38fp8.md/.json`, `prelim_vllm_Qwen_Qwen3.8-27B-FP8_q38fp8.summary.json` |
| `q38fp8-noauditor` | same, Variant C with `HDB_AUDITOR=off` | complete: 90 rows | `q38fp8-noauditor.csv`, `prelim_vllm_Qwen_Qwen3.8-27B-FP8_q38fp8-noauditor.summary.json` |
| `q330b90` | Qwen/Qwen3-30B-A3B-GPTQ-Int4 | complete: A, B, C 90 rows each | `q330b90.csv`, `q330b90.md`, `architecture_eval_q330b90.md/.json`, `prelim_vllm_Qwen_Qwen3-30B-A3B-GPTQ-Int4_q330b90.summary.json` |
| `gptoss90` | openai/gpt-oss-20b | complete: A, B, C 90 rows each | `gptoss90.csv`, `gptoss90.md`, `architecture_eval_gptoss90.md/.json` |
| `gemma4nvfp90` | nvidia/Gemma-4-26B-A4B-NVFP4 | complete: A, B, C 90 rows each (third attempt, L40S, smoke test skipped) | `gemma4nvfp90.csv`, `gemma4nvfp90.md`, `architecture_eval_gemma4nvfp90.md/.json` |
| `haiku55` | claude-haiku-5-5 (Anthropic API, profile `claude`): hosted reference row, not Experiment 1 | complete: A, B, C 90 rows each; code `a6e8557` | `haiku55.csv`, `haiku55.md`, `architecture_eval_haiku55.md/.json`, `prelim_claude_claude-haiku-5-5_haiku55.summary.json` |
| `q3m` | Qwen/Qwen3-30B-A3B-GPTQ-Int4, re-run on `4ff2767` | complete: A, B, C 90 rows each (2 parallel streams, merged) | `q3m.csv`, `q3m.md`, `architecture_eval_q3m.md/.json`, `prelim_vllm_Qwen_Qwen3-30B-A3B-GPTQ-Int4_q3m.summary.json` |
| `gptm` | openai/gpt-oss-20b, re-run on `4ff2767` | complete: A, B, C 90 rows each (3 parallel streams, merged) | `gptm.csv`, `gptm.md`, `architecture_eval_gptm.md/.json`, `prelim_vllm_openai_gpt-oss-20b_gptm.summary.json` |
| `gemm` | nvidia/Gemma-4-26B-A4B-NVFP4, re-run on `4ff2767` | complete: A, B, C 90 rows each (5 parallel streams, merged; L40S, smoke test skipped) | `gemm.csv`, `gemm.md`, `architecture_eval_gemm.md/.json`, `prelim_vllm_nvidia_Gemma-4-26B-A4B-NVFP4_gemm.summary.json` |
| `q38m` | Qwen/Qwen3.8-27B-FP8 (headline), re-run on `4ff2767` | complete: A, B, C 90 rows each (8 chunks from a shared queue, merged; 65,536 throughout) | `q38m.csv`, `q38m.md`, `architecture_eval_q38m.md/.json`, `prelim_vllm_Qwen_Qwen3.8-27B-FP8_q38m.summary.json` |
| `q38m-noauditor` | same, Variant C with `HDB_AUDITOR=off`, re-run on `4ff2767` | complete: 90 rows | `q38m-noauditor.csv`, `prelim_vllm_Qwen_Qwen3.8-27B-FP8_q38m-noauditor.summary.json` |
| `nomodel90_e71ad37_st` | no model (`LLM_PROFILE=none`), offline plumbing check | complete | `prelim_none_none_nomodel90_e71ad37_st.summary.json` |

**CSV columns** (one row per variant and question; the last row per pair in the raw file wins, so reruns replace
earlier attempts): `variant, id, category, status, task_success, tool_f1, latency_s, total_tokens, input_tokens,
output_tokens`. Tokens are summed over every model call of one answer, not one prompt. Variant C's guardrail-blocked
Category F rows have 0 tokens and 0.0 s. `latency_s` is rounded to 0.1 s, so a p90 recomputed from the CSV can differ from the README by 0.1 s.

**`<tag>.md`** is the same table as that run's AUTO-RESULTS block in the main README, generated from the rows by the
local helper `evaluation/analysis/checkpoint.py` (p90 at sorted index `floor(0.9*n)`). The README's narrative tables use linearly
interpolated p90, so p90 values can differ by a few seconds.

## Verify without a model or database

```bash
python evaluation/reported_results/verify_results.py
```

The standard-library script checks 2,610 CSV rows (nine runs of 270, plus two sets of 90 auditor-off rows) against
the nine JSON reports and recomputes task counts, Tool F1, Wilson intervals, exact paired McNemar comparisons (within each
run, and same-variant between every pair of runs), the before/after pairs of each initial run and its re-run
(answers that changed in each direction, McNemar on all 90 and without F, change in mean latency and tokens; for
Qwen3.8 also the auditor-off pass), the auditor on/off comparison of both Qwen3.8 runs, and token means. It prints the mean, conventional
median and upper-middle auditor token differences separately. It checks saved results, not answer-key validity.
CSV latency is rounded; the main README's interpolated p90 uses unrounded raw latency.

`qualitative_checks.json` preserves selected scorer fields for four Qwen3.8/Gemma Variant C case comparisons.
The examples concern shortlist completeness and missing financial inputs; they are observations, not a causal
explanation of model quality. Full answer text is not needed for these scorer-level examples.

`q38fp8_context_provenance.csv` records which final rows match the original 32,768-window backup in id, latency
and token counts: A 33, B 79, C 20. False means not retained from that backup, rather than an independently
recorded per-row server configuration. The live rows did not store server context size. Thus the headline model's
initial run mixes windows; the auditor-on side also retains 20 older C rows. Its re-run (`q38m`, `q38m-noauditor`)
ran entirely at 65,536: both of its server jobs logged that window from start-up and no other server produced rows.

`serving_evidence.md` records the Gemma server measurements. Other serving specifications in the README were
transcribed from the earlier local model notes; the original logs are not included in these slim artifacts.

The initial Experiment 1 runs used code `23c9076` and its 53-document policy corpus. The re-runs (`q38m`, `q38m-noauditor`,
`q3m`, `gptm`, `gemm`) used main `4ff2767`: the updated supervisor, complete-answer policy prompt and 55-document corpus, with the
same serving scripts, and with each model's questions split into parallel streams against one server (Qwen3.8: a
shared queue of 8 question chunks served by 8 workers on one server and 4 on a second), so re-run latency is not
comparable with the initial runs. `4ff2767` differs from Haiku's `a6e8557` only in the policy index
text, one chat UI file and documentation. The re-run `.md` snapshots were produced with the same table code as
`haiku55.md`, not by the local checkpoint helper. Current main's 55-document corpus and updated
policy prompt are reported separately in Experiment 2; reproducing historical scores requires the earlier code,
matching data, and serving overrides. The hosted reference row `haiku55` ran later on main `a6e8557` (the merge of
PR #32): same architecture graphs, tool registry, guardrail, auditor, scorer and benchmark, but a changed supervisor,
policy prompt, LLM gateway and 55-document corpus; it is therefore not a controlled member of the four-model
comparison. `haiku55.md` was produced with the same table code as the other snapshots, not by the local checkpoint
helper. The local checkpoint helper mentioned by the snapshot Markdown files is
not distributed; `verify_results.py` provides the tracked path for recomputing the principal comparisons.

## Evidence guide

| Evaluation concern | Evidence |
|---|---|
| Meaningful alternatives and trade-offs | Main README architecture comparison, 75-question headline comparison, latency and token tables |
| Quantitative analysis | Four complete open-weight initial runs, four complete re-runs on the updated code, one hosted reference row, paired tests, before/after comparison, confidence intervals, auditor on/off comparison |
| Qualitative analysis | Main README failure examples and `qualitative_checks.json`; calculator behaviour and Category F caveats |
| Debugging, risks and reflection | Main README earlier-run fixes, Gemma attempts, context-window provenance, threats to validity and next steps |
| Reproducibility | Recorded code/data identifiers, model/serving settings, CSV verification command and current test results |

These artifacts support the report; they do not replace students' own review of the interpretation or their
explanations of each member's contribution.

Final local checks and known test failures are recorded in [`validation.md`](validation.md).
