# Final results validation — 11 Nov 2026

Integration base: main `33615f6`. The historical architecture runs record code `23c9076`; the completed
README draft was `61b4985`. This change publishes documentation and result artifacts; it does not rerun models.

## Tests

```bash
LLM_PROFILE=none HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -W ignore -m pytest -q -ra
```

Full suite: **1,904 passed, 19 skipped, 6 failed** in 372.33 seconds. The six failures are
`tests/test_category_e_extra.py::test_ground_truth_is_reproducible_from_the_tools` for E1-X01 to E1-X05 and E2-X01.
The answer keys carry snapshot `be71fe16c4e7e7548eb7`; the local database carries `d888eb158ece963898a0`.
Rebuilding each of these six cases and comparing all fields except `snapshot_id` confirmed equality. The
failures remain visible; neither tests nor answer keys were changed.

Two independent-truth checks initially skipped because the isolated checkout used a directory symlink that its
recursive database discovery did not traverse. After making the database discoverable, both passed explicitly:

```bash
LLM_PROFILE=none HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -W ignore -m pytest \
  tests/test_independent_truth_check.py::test_b2_04_lists_the_block_beyond_the_price_search_limit \
  tests/test_independent_truth_check.py::test_every_stored_e_and_b_answer_matches_the_raw_data -q
```

Result: **2 passed** in 12.06 seconds. The remaining 17 skips are 15 statistics cases pinned to another snapshot,
one optional local QA-scenario test and one shellcheck check (shellcheck is not installed).
The historical baseline was 1,906 passed, 17 skipped, 6 failed. The same six failures remain in this validation.

## Result and documentation checks

- All 1,170 tracked CSV rows match the final raw rows by variant/question: status, success, Tool F1, input/output/
  total tokens, and latency within CSV rounding (0.05 seconds).
- All four architecture JSON reports match fresh aggregation of the raw results using the repository reporter.
- The tracked standard-library `verify_results.py` recomputes counts, Wilson intervals, paired tests and token means.
- All 13 input-token distribution rows and reference-window exceedance counts in the main README were recomputed.
- Tool inventory confirmed from code: A 25; B data 24; C owners 14 market, 7 spatial, 3 financial, 1 policy.
- Main's Member 3 policy section and Experiment 2 summary are preserved verbatim, including the 55-document results.
- README local Markdown links resolve; no deleted whiteboard-guide or obsolete RAGAS-result link remains.
- Experiment runner shell scripts pass `bash -n`; the Python report command's `--help` parses.
- JSON files parse; whitespace, merge-marker, private-path/token and commit-attribution checks pass.

## Hosted reference row `haiku55` (added 12 Nov 2026)

- The merged raw file (270 rows, all `ok`, commit `a6e8557`) is row-for-row identical to the three per-variant
  originals apart from the added `tag` field.
- `haiku55.csv` against the merged rows: 270 of 270 variant/question pairs, 0 mismatches in status, category,
  success, Tool F1 and tokens; latency within CSV rounding (7 rows sit exactly on the 0.05 s rounding boundary).
- The CSV builder, applied to the other runs' raw rows, reproduces their tracked CSVs (same rows; row order differs
  for runs with reruns).
- `architecture_eval_haiku55.md/.json` come from `python -m evaluation.run_architecture_eval --report --no-ablation
  --tag haiku55` on main `34259a2` (the reporter accepts a hosted row; it only reads saved rows). The same command
  reproduces the tracked `gemma4nvfp90` JSON byte for byte (its Markdown differs only by a trailing blank line).
  The summary JSON comes from `run_category_e_prelim.summarise`, which reproduces the tracked `q330b90` summary
  byte for byte.
- `verify_results.py` now includes `haiku55` and passes: 1,440 tracked rows agree with the five JSON reports.
- Not checked: the effort level the API applied, the billed cost, and per-call prompt sizes (not recorded in the rows).

## Re-runs on code `4ff2767`: `q3m`, `gptm`, `gemm` (added 12 Nov 2026)

- Each merged raw file has 270 rows, all `ok`, all recording commit `4ff2767` with `uncommitted_changes: false`. It is
  identical, row for row, to the last row per variant and question across its stream files (Gemma 4 5 streams of
  54 rows, gpt-oss-20b 3 of 90, Qwen3-30B-A3B 2 of 135); the streams' question sets are disjoint, and no stream file
  has a non-`ok` row, so no row was retried.
- `gemm.csv`, `gptm.csv` and `q3m.csv` against the merged rows: 270 of 270 variant/question pairs each, 0 mismatches
  in status, category, success, Tool F1 and tokens; latency within CSV rounding (24, 15 and 23 rows sit exactly on
  the 0.05 s rounding boundary). The same CSV builder reproduces the tracked `gemma4nvfp90.csv` (same rows).
- `architecture_eval_<tag>.md/.json` come from `python -m evaluation.run_architecture_eval --report --no-ablation
  --tag <tag>` on `4ff2767`; running that command again on the merged rows reproduces all three (JSON byte for
  byte; the tracked Markdown drops the generator's trailing blank line, as for the other runs). The
  summary JSONs come from `run_category_e_prelim.summarise` on the merged rows (the same call reproduces the tracked
  `q330b90` summary byte for byte).
- The `<tag>.md` snapshots and the README's AUTO-RESULTS blocks were produced with the same table code as `haiku55.md`
  (the table code reproduces the tracked `gemma4nvfp90` block exactly).
- Serving: the cluster job logs of the three re-runs (jobs 363448, 363449, 363453) show vLLM 0.33.0, the same GPU
  types and `max_model_len` as the initial runs (L40S at 65,536; Blackwell slice at 65,536 and 32,768), and for Gemma 4
  and Qwen3-30B-A3B the same KV-cache sizes as the initial runs.
- `verify_results.py` now includes `q3m`, `gptm` and `gemm` and the three before/after pairs, and passed: 2,250
  tracked rows agreed with the eight JSON reports at that point (2,610 with the Qwen3.8 re-run, below). Every README number for the re-runs (task counts, per-category
  cells, Tool F1, latency and token means and p90, input-token distributions, context-limit rows, fallback and
  tool-source counts, within-run, same-variant and before/after McNemar tests) was recomputed from the merged raw
  rows; every per-category row sums to the variant total.
- `git diff --stat a6e8557 4ff2767 -- agents mcp_server config evaluation/run_category_e_prelim.py` is empty; the
  Haiku rows were not re-run and its numbers are unchanged.
- Not checked: run-to-run variation on the same code (no repeat run), and the re-run answer texts (the Category F
  and Gemma calculator readings in the README are from the initial-run rows).

## Qwen3.8 re-run on `4ff2767`: `q38m` and `q38m-noauditor` (added 12 Nov 2026)

- The merged raw files have 270 and 90 rows, all `ok`, all recording commit `4ff2767` and `llm_timeout_s` 180. Each
  is identical, row for row, to the last row per variant and question across its 8 stream files; the chunks' question
  sets are disjoint. The stream files hold 16 `model_unreachable` lines on 8 rows (auditor on: A B3-04, C2-01, E1-01;
  B B2-04; C B1-02, B2-05; auditor off: C B1-02, B2-05), all replaced by later `ok` rows.
- `q38m.csv` and `q38m-noauditor.csv` against the merged rows: 270 and 90 pairs, 0 mismatches in status, category,
  success, Tool F1 and tokens; latency within CSV rounding (30 and 6 rows exactly on the 0.05 s boundary). The CSV
  builder reproduces the tracked `q38fp8-noauditor.csv`.
- `architecture_eval_q38m.md/.json`: running `python -m evaluation.run_architecture_eval --report --no-ablation --tag
  q38m` on the merged rows reproduces both (JSON byte for byte; trailing blank line dropped from the Markdown). The
  summary JSONs come from `run_category_e_prelim.summarise`.
- Window: the cluster logs of both server jobs (363485, 363509; NVIDIA L40S, vLLM 0.33.0) show `max_model_len`
  65,536, KV cache 150,732 tokens and 28.81 GiB of weights; the two jobs the first schedule submitted (363483,
  363484) were cancelled without starting, and the first queue item was claimed after job 363485 was up. So every
  re-run row ran at 65,536; the rows themselves do not record the window. Peak load on job 363485 from its log: 13
  requests running, 5 waiting, KV cache 99.5%.
- Auditor-off chunks: the second server's workers could take only auditor-off items (all A/B/C chunks were already
  claimed); which server ran each auditor-off chunk is not recorded, and the final sweep resumed one auditor-off chunk
  on job 363485.
- `verify_results.py` now includes `q38m`, both auditor on/off pairs and the Qwen3.8 before/after pairs (including
  the auditor-off pass), and passes: 2,610 tracked rows agree with the nine JSON reports. All README Qwen3.8 re-run
  figures were recomputed from the merged raw rows; per-category rows sum to the variant totals.

## Interpretation limits

The Qwen3.8 initial run retains 132 successful rows from its original smaller context window, including 20 C rows on
the auditor-on side. The resulting mixed-window comparison is disclosed rather than presented as a controlled uniform-window run.
Apart from the re-runs on `4ff2767` (one run per model), no additional seeds, repeated runs on the
same code, human answer-quality assessment or multi-turn history ablation were performed.
