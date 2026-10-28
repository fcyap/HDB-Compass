#!/usr/bin/env bash
# Demo day: start HDB-Compass on the presenting laptop with the model on the SMU cluster GPU. Member 4.
#
#   bash cluster/demo_start.sh             # cluster model (VPN + shared session needed)
#   bash cluster/demo_start.sh --offline   # no model: real tools and graph, template answers, starts in seconds
#
# Cluster mode starts (or reuses) the hdb-vllm job, opens the tunnel, checks the model answers, then runs the backend
# (Variant C on the cluster model) and the frontend and opens the app. While it runs it checks the model every 30 s
# and prints a warning if it stops answering. Ctrl-C stops the backend and frontend, closes the tunnel and cancels
# the job (add --keep-job to leave the GPU job running, e.g. between two demo slots).
#
# Before cluster mode, open the shared session once in your own terminal (asks for your password):
#   ssh -M -S ~/.ssh/hdb-cluster.sock -o ControlPersist=8h -o ServerAliveInterval=30 -fN "$CLUSTER_LOGIN"
# Fallback during the demo: Ctrl-C, then  bash cluster/demo_start.sh --offline
set -uo pipefail
MODE=cluster
KEEP_JOB=0
for a in "$@"; do
  case "$a" in
    --offline) MODE=offline ;;
    --keep-job) KEEP_JOB=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/data/local/demo_logs"
mkdir -p "$LOGS"
PIDS=()
JOB=""

for port in 8000 3000; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is in use (an earlier backend or frontend?). Stop it first:"
    lsof -nP -iTCP:$port -sTCP:LISTEN | tail -n +2
    exit 1
  fi
done
[[ -d frontend/node_modules ]] || { echo "== installing frontend packages"; (cd frontend && npm install --no-audit --no-fund) || exit 1; }

cleanup() {
  echo; echo "== stopping"
  # npm starts vite as a child, so stop each server's children too, then anything still on our two ports
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && { pkill -TERM -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; }; done
  for port in 8000 3000; do lsof -t -iTCP:$port -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null; done
  if [[ "$MODE" == cluster ]]; then
    bash cluster/tunnel.sh --close >/dev/null 2>&1 || true
    if [[ -n "$JOB" && "$KEEP_JOB" == 0 ]]; then
      "${SSH[@]}" "scancel $JOB" && echo "cancelled GPU job $JOB"
    elif [[ -n "$JOB" ]]; then
      echo "GPU job $JOB left running (--keep-job); cancel it later with: scancel $JOB"
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
nap() { sleep "$1" & wait $!; }   # a plain sleep would hold Ctrl-C until it ends

if [[ "$MODE" == cluster ]]; then
  SOCK="${CLUSTER_SOCK:-$HOME/.ssh/hdb-cluster.sock}"
  [[ -z "${CLUSTER_LOGIN:-}" && -f .env ]] && CLUSTER_LOGIN="$(grep -E '^CLUSTER_LOGIN=' .env | tail -1 | cut -d= -f2-)"
  LOGIN="${CLUSTER_LOGIN:?Set CLUSTER_LOGIN=<your-cluster-id>@<cluster-address> in .env}"
  ssh -S "$SOCK" -O check "$LOGIN" 2>/dev/null || {
    echo "No shared cluster session. On the VPN, run this in your own terminal, then start again:"
    echo "  ssh -M -S ~/.ssh/hdb-cluster.sock -o ControlPersist=8h -o ServerAliveInterval=30 -fN $LOGIN"
    echo "Or start without the model:  bash cluster/demo_start.sh --offline"
    exit 1; }
  SSH=(ssh -S "$SOCK" "$LOGIN")
  JOB="$("${SSH[@]}" 'squeue --me -h -n hdb-vllm -o %i | head -1')"
  if [[ -z "$JOB" ]]; then
    JOB="$("${SSH[@]}" 'rm -f ~/vllm_endpoint.txt; sbatch --parsable ~/hdb-cluster/serve_vllm.sbatch')"
    echo "== submitted GPU job $JOB"
  else
    echo "== using running GPU job $JOB"
  fi
  echo "== waiting for the job to start (queue)"
  for _ in $(seq 1 120); do "${SSH[@]}" "grep -qw $JOB ~/vllm_endpoint.txt 2>/dev/null" && break; nap 15; done
  "${SSH[@]}" "grep -qw $JOB ~/vllm_endpoint.txt 2>/dev/null" || { echo "Job $JOB has not started after 30 min. Use --offline."; exit 1; }
  bash cluster/tunnel.sh || { echo "Tunnel failed. Use --offline."; exit 1; }
  KEY="$(grep -E '^VLLM_API_KEY=' .env | tail -1 | cut -d= -f2-)"
  model_up() { curl -sf -m 5 -H "Authorization: Bearer $KEY" http://localhost:8001/v1/models >/dev/null; }
  echo "== waiting for the model to load (about 5 min)"
  for _ in $(seq 1 80); do model_up && break; nap 15; done
  model_up || { echo "The model did not come up. Use --offline."; exit 1; }
  export LLM_PROFILE=vllm LLM_TIMEOUT_S=60 LLM_MAX_RETRIES=0   # before the smoke test, which uses the active profile
  .venv/bin/python cluster/smoke_test.py > "$LOGS/smoke.log" 2>&1
  grep -q "OK: well-formed" "$LOGS/smoke.log" || { tail -5 "$LOGS/smoke.log"; echo "Smoke test failed (log: $LOGS/smoke.log). Use --offline."; exit 1; }
  echo "== model ready"
else
  export LLM_PROFILE=none
fi

echo "== starting backend (LLM_PROFILE=$LLM_PROFILE), log: $LOGS/backend.log"
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > "$LOGS/backend.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null && break; nap 1; done
curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null || { echo "Backend did not start; see $LOGS/backend.log"; exit 1; }
echo "== starting frontend, log: $LOGS/frontend.log"
(cd frontend && exec npm run dev) > "$LOGS/frontend.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do curl -sf -m 2 http://localhost:3000 >/dev/null && break; nap 1; done
curl -sf -m 2 http://localhost:3000 >/dev/null || { echo "Frontend did not start; see $LOGS/frontend.log"; exit 1; }
open http://localhost:3000 2>/dev/null || true
echo
echo "HDB-Compass is running at http://localhost:3000  (mode: $MODE). Ctrl-C to stop."
[[ "$MODE" == cluster ]] && echo "If a model warning appears: Ctrl-C, then  bash cluster/demo_start.sh --offline"

while true; do
  nap 30
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "!! a server stopped; see $LOGS/"; exit 1; }
  done
  if [[ "$MODE" == cluster ]] && ! model_up; then
    echo "!! $(date +%H:%M:%S) THE CLUSTER MODEL IS NOT ANSWERING (VPN, session or job). Switch: Ctrl-C, then --offline"
  fi
done
