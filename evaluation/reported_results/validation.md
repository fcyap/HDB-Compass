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

## Interpretation limits

Qwen3.8 retains 132 successful rows from its original smaller context window, including 20 C rows on the auditor-on
side. The resulting mixed-window comparison is disclosed rather than presented as a controlled uniform-window run.
No additional seeds, GPU reruns, human answer-quality assessment or multi-turn history ablation were performed.
