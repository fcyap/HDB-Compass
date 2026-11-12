# Reproducing the project: what you can check, and what each step needs

Three levels, from cheapest to most expensive. Run commands from the repository root.

| Level | Needs | What it shows |
|---|---|---|
| 1. Check the reported numbers | Python 3.11+ only | every count, confidence interval and paired test in the README, recomputed from the tracked CSVs |
| 2. Run the code and tests | the environment, plus the market database for the data-dependent tests | the agents, tools, guardrail and auditor behave as specified (no language model needed) |
| 3. Re-run the model evaluation | an SMU cluster GPU account (open-weight models) or an Anthropic key (hosted row) | the 90-question benchmark on a model of your choice |

## Level 1: verify the reported results (about a second, no model, GPU or database)
```bash
python evaluation/reported_results/verify_results.py
```
Expected last line: `Verified 2,610 tracked rows and agreement with per-model JSON task counts and Tool F1.` (the standard
library only). What the checks cover: [`evaluation/reported_results/validation.md`](../evaluation/reported_results/validation.md).

## Level 2: environment, database, tests
1. **Environment** (Python 3.11 or 3.12):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   python -m pip install -r requirements.lock      # every package pinned; resolved for Linux, macOS and Windows
   # or: python -m pip install -r requirements.txt  (direct dependencies only; newer versions may behave differently)
   ```
   Checked on 12 Nov 2026: a fresh Python 3.12 environment from `requirements.lock` (CPU torch) passed the full suite with
   the local database (2,003 passed, 23 skipped, 0 failed, about 6 minutes). On Linux the lock includes torch's CUDA
   libraries, so the environment is about 6 GB even without a GPU; the top of `requirements.lock` gives a smaller CPU-only
   route through `requirements.txt` (not checked against the full suite). macOS and Windows installs were resolved but not tried.
2. **Configuration:** copy `.env.example` to `.env`; for local runs `LLM_PROFILE=none` needs no keys
   (see [Setup](../README.md#setup)). Seeds and model settings live in `config/experiment_config.yaml`
   (temperature 0, seed 42 for the open-weight runs).
3. **Market database** (public data.gov.sg resale file, downloaded and checksum-cached):
   ```bash
   python -m data.build_market_db
   ```
4. **Tests:**
   ```bash
   python -m pytest tests/ -q          # about 2,000 tests, about 6 minutes on a laptop CPU
   ```
   - The Category E answer keys are pinned to database snapshot `be71fe16c4e7e7548eb7`. A database built from a later download
     has a different snapshot id, so the six
     `tests/test_category_e_extra.py::test_ground_truth_is_reproducible_from_the_tools` cases are **skipped with that
     reason** instead of failing. Set `HDB_REQUIRE_PINNED_SNAPSHOT=1` to make them fail when the snapshot differs (use this
     only with the team's database).
   - Without the database (checked on 12 Nov 2026 with `HDB_COMPASS_DB_PATH` pointing at a missing file): 1,851 passed, 16 skipped,
     135 failed in 8 minutes; the failures are tests that read the database (analysis questions, reply composer, price check, demo
     readiness and similar). Build the database first (step 3); this is also why CI runs only the result check, not the test suite.
   - `sentence-transformers` must be installed, or eight policy tests fail (it is in both requirements files).
5. **Run the graph without a model:**
   ```bash
   python -c 'from agents.arch_3_specialist_5agent import run; print(run("Find 4-room flats in Bishan under 850k with at least 60 years lease.")["draft_response"])'
   ```

## Level 3: re-run the model evaluation
**Open-weight models on the SMU cluster** (needs the SMU VPN, an account on the student partition, and `CLUSTER_LOGIN` in
`.env`; commands below are as recorded in the README, not re-run when this page was written):
```bash
# on the cluster: copy cluster/serve_vllm.sbatch to ~/hdb-cluster/, then submit it with the model's preset
sbatch --constraint='a40|l40s' --export=ALL,MODEL_PRESET=qwen38 ~/hdb-cluster/serve_vllm.sbatch
#   presets: qwen3-30b (default; --constraint=rtx_pro_6000_blackwell_max), qwen38, gptoss
#   (--constraint=rtx_pro_6000_blackwell_max), gemma4 (--constraint='a40|l40s')
# locally:
export LLM_PROFILE=vllm LLM_MODEL=Qwen/Qwen3.8-27B-FP8 LLM_TIMEOUT_S=180 LLM_MAX_RETRIES=1
bash cluster/run_architecture_eval.sh mytag     # A, B, C with the auditor on, then C with the auditor off; writes the report
```
The presets reproduce the flags in the README table
[Setup used for these runs](../README.md#setup-used-for-these-runs) (parsers, context window, memory utilisation, extra
flags). `run_live_eval.sh` reuses a running `hdb-vllm` job, so submit the preset job first. Download the model with
`cluster/setup_env.sbatch` and the same `MODEL_REPO`. Per-model GPU memory limits decide which model fits where (see the
README's context-window section).

**Faster runs with parallel streams:** run the runner on disjoint subsets of the 90 question ids against one server, then
merge:
```bash
python -m evaluation.run_category_e_prelim --dataset evaluation/benchmark_hdb_90.json --variants A,B,C --ids <comma-separated ids> --tag mytags0
python -m evaluation.merge_streams --out evaluation/results/prelim_vllm_<model>_mytag.jsonl evaluation/results/prelim_vllm_<model>_mytags0.jsonl ...
python -m evaluation.run_architecture_eval --report --no-ablation --tag mytag
```
The last row per (variant, question) wins, as in the reported re-runs. Parallel streams share one server, so latency is
not comparable with a one-request-at-a-time run.

**Reproducing a specific reported run:** use the commit named in that run's rows (`23c9076` initial runs, `4ff2767`
re-runs, `a6e8557` hosted row) and the benchmark `evaluation/benchmark_hdb_90.json` (sha `0bbca6193b0a`). The
orchestration scripts used for the re-runs (a local driver and the cluster-side server copies) are not all in the
repository; the presets and `merge_streams.py` replace the parts that matter.

**Hosted reference row (paid, optional):** `ANTHROPIC_API_KEY` in `.env`, `HDB_EVAL_ALLOW_HOSTED=claude`; commands in the
README under "Setup used for these runs". The reported row has already been run once; do not repeat it without a reason.

## What cannot be reproduced exactly
- Model outputs from a different serving stack, GPU or vLLM version (greedy decoding, but not bit-identical across setups);
  one run per code version was made, so small differences between runs are expected.
- The database snapshot: see above.
- Hosted-model results (API defaults, no seed).
