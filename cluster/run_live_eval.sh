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
# Needs: SMU VPN, CLUSTER_LOGIN in the gitignored .env, and the shared session: log in with  ssh $CLUSTER_LOGIN
# in your terminal first (it asks for your password there). ~/.ssh/config keeps that login open for 8 h
# (ControlMaster auto, ControlPath ~/.ssh/hdb-cluster.sock, ControlPersist 8h) and this script only reuses it:
# every ssh call uses BatchMode=yes, so a closed session stops the run with a message instead of a password prompt.
# KEEP_JOB=1 leaves the GPU job running for a following run (the next run reuses it; the last one must not set it).
# The job's id is kept in evaluation/results/cluster_job.txt (gitignored) until it is cancelled: a later run that
# cannot reach the cluster prints the exact  scancel <id>  to run after logging in, and one that can reuses the job.
# Rows whose model calls could not reach the server are marked model_unreachable and rerun on the next resume.
# Model calls wait at most LLM_TIMEOUT_S=90 s with LLM_MAX_RETRIES=1 (set either to override): the slowest healthy
# row on record took 92 s for all its calls together. A lost server is caught before each row by the runner's quick
# check instead, which stops the run (exit code 3) and with it this script, so no later tag waits on a dead tunnel.
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
# The GPU job an earlier run of this script started or reused and did not cancel (KEEP_JOB, or a lost session)
JOB_FILE="${CLUSTER_JOB_FILE:-$ROOT/evaluation/results/cluster_job.txt}"
KEPT="$(cat "$JOB_FILE" 2>/dev/null || true)"
ssh -o BatchMode=yes -S "$SOCK" -O check "$LOGIN" 2>/dev/null \
  || { echo "No shared session at $SOCK: on the VPN, log in with  ssh $LOGIN  in your terminal, then run again."
       [[ -n "$KEPT" ]] && echo "GPU job $KEPT from an earlier run may still be running: after logging in, run  scancel $KEPT" \
                        && echo "(or run this script again once logged in: it reuses the job, then cancels it)"
       exit 1; }
SSH=(ssh -o BatchMode=yes -S "$SOCK" "$LOGIN")
export LLM_PROFILE=vllm
export CLUSTER_BATCH=1   # tunnel.sh: fail instead of asking for a password
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-90}" LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"

RUNNING="$("${SSH[@]}" 'squeue --me -h -n hdb-vllm -o %i')" || { echo "STEP_FAILED could not list the GPU jobs"; exit 1; }
if [[ -n "$KEPT" ]] && grep -qx "$KEPT" <<< "$RUNNING"; then
  JOB="$KEPT"
  echo "using job $JOB left running by an earlier run"
else
  [[ -n "$KEPT" ]] && echo "job $KEPT from an earlier run has ended"
  rm -f "$JOB_FILE"
  JOB="$(head -1 <<< "$RUNNING")"
  if [[ -z "$JOB" ]]; then
    JOB="$("${SSH[@]}" 'rm -f ~/vllm_endpoint.txt; sbatch --parsable ~/hdb-cluster/serve_vllm.sbatch')"
    JOB="${JOB%%;*}"   # --parsable prints id;cluster on a multi-cluster setup
    echo "submitted job $JOB"
  else
    echo "using running job $JOB"
  fi
fi
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "STEP_FAILED no job id from the cluster: $JOB"; exit 1; }
mkdir -p "$(dirname "$JOB_FILE")" && echo "$JOB" > "$JOB_FILE"   # removed again once the job is cancelled
cleanup() {
  bash cluster/tunnel.sh --close >/dev/null 2>&1 || true
  if [[ -n "${KEEP_JOB:-}" ]]; then
    echo "left job $JOB running (KEEP_JOB) for the next run; cancel it later with: scancel $JOB"
  elif "${SSH[@]}" "scancel $JOB"; then
    rm -f "$JOB_FILE"
    echo "cancelled job $JOB"
  else
    echo "Could not cancel GPU job $JOB (session closed?). Log in with  ssh $LOGIN  and run: scancel $JOB"
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
  # exit code 3: the model server stopped answering; the next tag would only find the same dead tunnel
  [[ "${PIPESTATUS[0]}" == 3 ]] && { echo "STEP_FAILED $TAG: the model server stopped answering"; exit 1; }
  echo "RUN_DONE $TAG"
done
