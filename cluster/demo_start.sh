#!/usr/bin/env bash
# Demo day: start HDB-Compass on the presenting laptop with the model on the SMU cluster GPU. Member 4.
#
#   bash cluster/demo_start.sh             # cluster model (VPN + shared session needed)
#   bash cluster/demo_start.sh --offline   # no model: real tools and graph, template answers, starts in seconds
#   bash cluster/demo_start.sh --preview   # serve a production build of the frontend instead of the dev server
#   bash cluster/demo_start.sh --keep-job  # Ctrl-C leaves the GPU job running (e.g. between two demo slots)
#   bash cluster/demo_start.sh --dry-run   # checks this laptop and prints the plan; starts nothing, no ssh
#
# Cluster mode starts (or reuses) the hdb-vllm job, opens the tunnel, checks the model answers, then runs the backend
# (Variant C on the cluster model) and the frontend and opens the app. While it runs it checks the model every 30 s
# and prints a warning if it stops answering, and keeps the laptop awake (caffeinate, macOS only). Ctrl-C stops the
# backend and frontend, closes the tunnel and CANCELS THE GPU JOB unless --keep-job was given.
# A model call that takes longer than LLM_TIMEOUT_S (default 25 s; set it in the shell to change) falls back to the
# template answer, so a stalled server costs one short wait instead of a frozen demo.
#
# Before cluster mode, log in with  ssh $CLUSTER_LOGIN  in your terminal (on the VPN; it asks for your password).
# ~/.ssh/config keeps that login open as the shared session for 8 h, and this script only reuses it: every ssh call
# here uses BatchMode=yes, so it fails with a message instead of asking for a password. The config entry for the
# cluster host needs:  ControlMaster auto,  ControlPath ~/.ssh/hdb-cluster.sock,  ControlPersist 8h
# Fallback during the demo: Ctrl-C, then  bash cluster/demo_start.sh --offline
set -uo pipefail
MODE=cluster
KEEP_JOB=0
PREVIEW=0
DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --offline) MODE=offline ;;
    --keep-job) KEEP_JOB=1 ;;
    --preview) PREVIEW=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/data/local/demo_logs"
DIST="$ROOT/data/local/demo_dist"   # the --preview build goes here, so nothing is written inside frontend/
mkdir -p "$LOGS"
PIDS=()
JOB=""
CAFFEINATE=""
export CLUSTER_BATCH=1   # tunnel.sh: fail instead of asking for a password
SOCK="${CLUSTER_SOCK:-$HOME/.ssh/hdb-cluster.sock}"
TIMEOUT_S="${LLM_TIMEOUT_S:-25}"

# A dry run reports every problem and carries on to the plan; a real start stops at the first one.
BLOCKED=0
blocked() {
  printf '%s\n' "$@"
  [[ "$DRY_RUN" == 1 ]] || exit 1
  BLOCKED=1
}

for port in 8000 3000; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    blocked "Port $port is in use (an earlier backend or frontend?). Stop it first:" \
      "$(lsof -nP -iTCP:$port -sTCP:LISTEN | tail -n +2)"
  fi
done

# Node and the policy index: a missing Node stops the start; the rest are warnings that say early what will break.
if ! command -v node >/dev/null 2>&1; then
  blocked "Node is not installed; the frontend needs it (install Node 22 LTS)."
else
  # The Node range comes from the installed Vite, or from the Vite major in frontend/package.json before npm install.
  NODE_NOTE="$(node -e '
    const fs = require("fs");
    const read = (p) => { try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return null; } };
    const pkg = read("frontend/package.json") || {};
    const want = (pkg.devDependencies || {}).vite || (pkg.dependencies || {}).vite || "";
    const major = parseInt(want.replace(/^[^0-9]*/, ""), 10);
    const vite = read("frontend/node_modules/vite/package.json");
    const range = (vite && vite.engines && vite.engines.node)
      || (major >= 7 ? "^20.19.0 || >=22.12.0" : major >= 5 ? "^18.0.0 || >=20.0.0" : "");
    const have = process.versions.node.split(".").map(Number);
    const atLeast = (r) => have[0] !== r[0] ? have[0] > r[0] : have[1] !== r[1] ? have[1] > r[1] : have[2] >= r[2];
    const fits = range.split("||").some((alt) => {
      const m = alt.trim().match(/^(\^|>=)(\d+)\.(\d+)\.(\d+)$/);
      if (!m) return true;   // a range written another way: do not guess, do not warn
      const r = [+m[2], +m[3], +m[4]];
      return atLeast(r) && (m[1] === ">=" || have[0] === r[0]);
    });
    if (!fits) console.log(`Vite ${want} needs Node ${range}; this laptop has Node ${process.versions.node}.`);
  ' 2>/dev/null)"
  [[ -n "$NODE_NOTE" ]] && echo "!! $NODE_NOTE The frontend may not start: install Node 22 LTS."
fi
.venv/bin/python -c 'import rank_bm25' 2>/dev/null \
  || echo "!! rank_bm25 is not installed: the real policy search cannot load its index. Install: .venv/bin/pip install rank_bm25"

if [[ "$MODE" == cluster ]]; then
  [[ -z "${CLUSTER_LOGIN:-}" && -f .env ]] && CLUSTER_LOGIN="$(grep -E '^CLUSTER_LOGIN=' .env | tail -1 | cut -d= -f2-)"
  LOGIN="${CLUSTER_LOGIN:-}"
  [[ -n "$LOGIN" ]] || blocked "Set CLUSTER_LOGIN=<your-cluster-id>@<cluster-address> in .env"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo
  echo "== dry run: nothing started, no ssh. A real start would:"
  [[ -d frontend/node_modules ]] || echo "   install the frontend packages (npm install)"
  [[ "$PREVIEW" == 1 ]] && echo "   build the frontend into $DIST (vite build)"
  command -v caffeinate >/dev/null 2>&1 && echo "   keep the laptop awake while it runs (caffeinate)"
  if [[ "$MODE" == cluster ]]; then
    echo "   check the shared session: ssh -o BatchMode=yes -S $SOCK -O check ${LOGIN:-<CLUSTER_LOGIN>}"
    echo "   start or reuse the hdb-vllm GPU job, open the tunnel and smoke-test the model"
    echo "   run the backend with LLM_PROFILE=vllm LLM_TIMEOUT_S=$TIMEOUT_S LLM_MAX_RETRIES=${LLM_MAX_RETRIES:-0}"
  else
    echo "   run the backend with LLM_PROFILE=none (no model)"
  fi
  if [[ "$PREVIEW" == 1 ]]; then
    echo "   serve the production build on http://localhost:3000 (vite preview)"
  else
    echo "   serve the frontend dev server on http://localhost:3000 (npm run dev)"
  fi
  if [[ "$MODE" == cluster && "$KEEP_JOB" == 1 ]]; then
    echo "   on Ctrl-C: stop the servers and the tunnel, leave the GPU job running (--keep-job)"
  elif [[ "$MODE" == cluster ]]; then
    echo "   on Ctrl-C: stop the servers and the tunnel, and cancel the GPU job"
  fi
  [[ "$BLOCKED" == 1 ]] && echo "!! fix the problems above first: a real start would stop there."
  exit 0
fi

[[ -d frontend/node_modules ]] || { echo "== installing frontend packages"; (cd frontend && npm install --no-audit --no-fund) || exit 1; }
if [[ "$PREVIEW" == 1 ]]; then
  # Built before the cluster wait, so a build error shows now and not after the model has loaded
  echo "== building the frontend, log: $LOGS/frontend.log"
  (cd frontend && exec ./node_modules/.bin/vite build --outDir "$DIST" --emptyOutDir) > "$LOGS/frontend.log" 2>&1 \
    || { tail -5 "$LOGS/frontend.log"; echo "Frontend build failed; see $LOGS/frontend.log, or start without --preview"; exit 1; }
fi

cleanup() {
  echo; echo "== stopping"
  # npm starts vite as a child, so stop each server's children too, then anything still on our two ports
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && { pkill -TERM -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; }; done
  for port in 8000 3000; do lsof -t -iTCP:$port -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null; done
  [[ -n "$CAFFEINATE" ]] && kill "$CAFFEINATE" 2>/dev/null
  if [[ "$MODE" == cluster ]]; then
    bash cluster/tunnel.sh --close >/dev/null 2>&1 || true
    if [[ -n "$JOB" && "$KEEP_JOB" == 0 ]]; then
      if "${SSH[@]}" "scancel $JOB"; then
        echo "cancelled GPU job $JOB"
      else
        echo "Could not cancel GPU job $JOB (session closed?). Log in with  ssh $LOGIN  and run: scancel $JOB"
      fi
    elif [[ -n "$JOB" ]]; then
      echo "GPU job $JOB left running (--keep-job); cancel it later with: scancel $JOB"
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
nap() { sleep "$1" & wait $!; }   # a plain sleep would hold Ctrl-C until it ends

# Keep the display and the laptop awake for as long as this script runs (macOS; elsewhere there is no caffeinate)
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dims -w $$ &
  CAFFEINATE=$!
fi

if [[ "$MODE" == cluster ]]; then
  ssh -o BatchMode=yes -S "$SOCK" -O check "$LOGIN" 2>/dev/null || {
    echo "No shared cluster session. On the VPN, log in with  ssh $LOGIN  in your terminal, then start again."
    echo "Or start without the model:  bash cluster/demo_start.sh --offline"
    exit 1; }
  SSH=(ssh -o BatchMode=yes -S "$SOCK" "$LOGIN")
  JOB="$("${SSH[@]}" 'squeue --me -h -n hdb-vllm -o %i | head -1')"
  if [[ -z "$JOB" ]]; then
    JOB="$("${SSH[@]}" 'rm -f ~/vllm_endpoint.txt; sbatch --parsable ~/hdb-cluster/serve_vllm.sbatch')"
    echo "== submitted GPU job $JOB"
  else
    echo "== using running GPU job $JOB"
  fi
  [[ "$KEEP_JOB" == 0 ]] && echo "   (Ctrl-C cancels this job; start with --keep-job to leave it running)"
  echo "== waiting for the job to start (queue)"
  for _ in $(seq 1 120); do "${SSH[@]}" "grep -qw $JOB ~/vllm_endpoint.txt 2>/dev/null" && break; nap 15; done
  "${SSH[@]}" "grep -qw $JOB ~/vllm_endpoint.txt 2>/dev/null" || { echo "Job $JOB has not started after 30 min. Use --offline."; exit 1; }
  bash cluster/tunnel.sh || { echo "Tunnel failed. Use --offline."; exit 1; }
  KEY="$(grep -E '^VLLM_API_KEY=' .env | tail -1 | cut -d= -f2-)"
  model_up() { curl -sf -m 5 -H "Authorization: Bearer $KEY" http://localhost:8001/v1/models >/dev/null; }
  echo "== waiting for the model to load (about 5 min)"
  for _ in $(seq 1 80); do model_up && break; nap 15; done
  model_up || { echo "The model did not come up. Use --offline."; exit 1; }
  export LLM_PROFILE=vllm LLM_TIMEOUT_S="$TIMEOUT_S" LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-0}"
  # The first request after loading can be slow, so the smoke test (which also warms the model) gets 60 s
  LLM_TIMEOUT_S=60 .venv/bin/python cluster/smoke_test.py > "$LOGS/smoke.log" 2>&1
  grep -q "OK: well-formed" "$LOGS/smoke.log" || { tail -5 "$LOGS/smoke.log"; echo "Smoke test failed (log: $LOGS/smoke.log). Use --offline."; exit 1; }
  echo "== model ready ($(grep -o 'latency=[0-9.]*s' "$LOGS/smoke.log"); answers time out after ${LLM_TIMEOUT_S} s)"
else
  export LLM_PROFILE=none
fi

echo "== starting backend (LLM_PROFILE=$LLM_PROFILE), log: $LOGS/backend.log"
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > "$LOGS/backend.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null && break; nap 1; done
curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null || { echo "Backend did not start; see $LOGS/backend.log"; exit 1; }
if [[ "$PREVIEW" == 1 ]]; then
  echo "== serving the frontend build, log: $LOGS/frontend.log"
  (cd frontend && exec ./node_modules/.bin/vite preview --outDir "$DIST" --port 3000 --strictPort) >> "$LOGS/frontend.log" 2>&1 &
else
  echo "== starting frontend, log: $LOGS/frontend.log"
  (cd frontend && exec npm run dev) > "$LOGS/frontend.log" 2>&1 &
fi
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
