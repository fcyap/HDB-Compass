#!/usr/bin/env bash
# One live Category E run on the cluster model, start to finish. Member 4.
# Starts the hdb-vllm job if none is running, waits for the server, opens the tunnel, runs the smoke test,
# optionally the 11-prompt tool check, then Category E (with follow-ups) under a results tag. Closes the tunnel
# and cancels the job at the end, also on failure or Ctrl-C, so the GPU slice is never left running.
#
#   bash cluster/run_live_eval.sh realdata5                 # A, B and C
#   bash cluster/run_live_eval.sh realdata4 C               # resume: only rows not yet done are run
#   bash cluster/run_live_eval.sh realdata5 A,B,C --tool-check
#   bash cluster/run_live_eval.sh rep1,rep2,rep3            # repeats in one job (the model loads once)
#   EVAL_DATASET=varied bash cluster/run_live_eval.sh varied1 C   # the 60 varied-phrasing questions
#   HDB_CODE_WRITTEN_FACTS=0 bash cluster/run_live_eval.sh nofacts1 C   # ablation: the model writes the facts too
#     (the run settings record the switch; use a new tag, since rows from the other setting are rerun, not mixed)
#   EVAL_DATASET=extra bash cluster/run_live_eval.sh extra1 A,B,C # the 6 extra questions (new towns, school, upfront)
#
# Needs: SMU VPN, CLUSTER_LOGIN in the gitignored .env, and an open shared session so no password is asked
# mid-run. Open it once in your own terminal (it asks for your password there and stays open):
#   ssh -M -S ~/.ssh/hdb-cluster.sock -o ControlPersist=8h -fN "$CLUSTER_LOGIN"
# KEEP_JOB=1 leaves the GPU job running for a following run (the next run reuses it; the last one must not set it).
# Rows whose model calls could not reach the server are marked model_unreachable and rerun on the next resume.
set -uo pipefail
TAGS="${1:?usage: run_live_eval.sh TAG[,TAG...] [VARIANTS] [--tool-check]}"
VARIANTS="${2:-A,B,C}"
TOOL_CHECK="${3:-}"
DATASET="${EVAL_DATASET:-}"   # EVAL_DATASET=varied (or extra) runs evaluation/category_e_varied.json (category_e_extra.json)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SOCK="${CLUSTER_SOCK:-$HOME/.ssh/hdb-cluster.sock}"
if [[ -z "${CLUSTER_LOGIN:-}" && -f .env ]]; then
  CLUSTER_LOGIN="$(grep -E '^CLUSTER_LOGIN=' .env | tail -1 | cut -d= -f2-)"
fi
LOGIN="${CLUSTER_LOGIN:?Set CLUSTER_LOGIN=<your-cluster-id>@<cluster-address> in .env}"
ssh -S "$SOCK" -O check "$LOGIN" 2>/dev/null || { echo "No shared session at $SOCK: open it first (see the header)."; exit 1; }
SSH=(ssh -S "$SOCK" "$LOGIN")
export LLM_PROFILE=vllm

JOB="$("${SSH[@]}" 'squeue --me -h -n hdb-vllm -o %i | head -1')"
if [[ -z "$JOB" ]]; then
  JOB="$("${SSH[@]}" 'rm -f ~/vllm_endpoint.txt; sbatch --parsable ~/hdb-cluster/serve_vllm.sbatch')"
  echo "submitted job $JOB"
else
  echo "using running job $JOB"
fi
cleanup() {
  bash cluster/tunnel.sh --close >/dev/null 2>&1 || true
  if [[ -n "${KEEP_JOB:-}" ]]; then
    echo "left job $JOB running (KEEP_JOB) for the next run"
  else
    "${SSH[@]}" "scancel $JOB" && echo "cancelled job $JOB"
  fi
}
trap cleanup EXIT

echo "== waiting for the job to start"
for _ in $(seq 1 240); do
  "${SSH[@]}" "grep -qw $JOB ~/vllm_endpoint.txt 2>/dev/null" && break
  sleep 15
done
"${SSH[@]}" "grep -qw $JOB ~/vllm_endpoint.txt 2>/dev/null" || { echo "STEP_FAILED job $JOB never started"; exit 1; }
echo "JOB_STARTED $("${SSH[@]}" 'cat ~/vllm_endpoint.txt')"

bash cluster/tunnel.sh || { echo "STEP_FAILED tunnel"; exit 1; }
KEY="$(grep -E '^VLLM_API_KEY=' .env | tail -1 | cut -d= -f2-)"
echo "== waiting for the server to load the model"
for _ in $(seq 1 120); do
  curl -sf -m 5 -H "Authorization: Bearer $KEY" http://localhost:8001/v1/models >/dev/null && break
  sleep 15
done
curl -sf -m 5 -H "Authorization: Bearer $KEY" http://localhost:8001/v1/models >/dev/null || { echo "STEP_FAILED server never ready"; exit 1; }
echo "SERVER_READY"
.venv/bin/python cluster/smoke_test.py 2>&1 | tail -2 | grep -q "OK: well-formed" || { echo "STEP_FAILED smoke test"; exit 1; }
echo "SMOKE_OK"

if [[ "$TOOL_CHECK" == "--tool-check" ]]; then
  .venv/bin/python -m evaluation.tool_selection_check --variants "$VARIANTS" 2>&1 | grep -E "^\[|intended tools|^\|"
  echo "TOOLCHECK_DONE"
fi
for TAG in ${TAGS//,/ }; do
  .venv/bin/python evaluation/run_category_e_prelim.py ${DATASET:+--dataset "$DATASET"} --tag "$TAG" --variants "$VARIANTS" 2>&1 \
    | grep -E "^\[|already done|ERROR|UNREACHABLE|Traceback|^variant|^[ABC] "
  echo "RUN_DONE $TAG"
done
