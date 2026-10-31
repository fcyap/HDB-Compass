#!/usr/bin/env bash
# Forward laptop:8001 -> <gpu node>:<port> through the Origami login node, and copy the
# server's API key into HDB-Compass/.env (VLLM_API_KEY, gitignored). Run on VPN after the
# hdb-vllm job is RUNNING.  Member 4.
#   bash cluster/tunnel.sh          # then:  LLM_PROFILE=vllm .venv/bin/python cluster/smoke_test.py
#   bash cluster/tunnel.sh --close  # remove the forward
# Log in first with  ssh $CLUSTER_LOGIN  in your terminal: ~/.ssh/config keeps that login open as the
# shared session ~/.ssh/hdb-cluster.sock, which this script reuses without a password (BatchMode=yes).
# Without it, plain ssh asks for your password, unless CLUSTER_BATCH is set (demo_start.sh and
# run_live_eval.sh set it): then it stops with a message instead.
set -euo pipefail
SOCK="${CLUSTER_SOCK:-$HOME/.ssh/hdb-cluster.sock}"
LOCAL_PORT="${LOCAL_PORT:-8001}"
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
# Each member uses their own cluster account: set CLUSTER_LOGIN in the shell or in the gitignored .env
if [[ -z "${CLUSTER_LOGIN:-}" && -f "$ENV_FILE" ]]; then
  CLUSTER_LOGIN="$(grep -E '^CLUSTER_LOGIN=' "$ENV_FILE" | tail -1 | cut -d= -f2-)"
fi
LOGIN="${CLUSTER_LOGIN:?Set CLUSTER_LOGIN=<your-cluster-id>@<cluster-address> in .env or your shell}"
SSH=(ssh)
if [[ -S "$SOCK" ]] && ssh -o BatchMode=yes -S "$SOCK" -O check "$LOGIN" 2>/dev/null; then
  SSH=(ssh -o BatchMode=yes -S "$SOCK")
elif [[ "${1:-}" == "--close" ]]; then
  echo "No shared session: no forward to close."; exit 0   # forwards live in the shared session and end with it
elif [[ -n "${CLUSTER_BATCH:-}" ]]; then
  echo "No shared session at $SOCK: log in with  ssh $LOGIN  in your terminal."; exit 1
fi

ENDPOINT="$("${SSH[@]}" "$LOGIN" 'cat ~/vllm_endpoint.txt 2>/dev/null' || true)"
read -r NODE PORT JOB <<< "$ENDPOINT"
if [[ -z "${NODE:-}" || -z "${PORT:-}" ]]; then echo "No ~/vllm_endpoint.txt on the cluster: is the hdb-vllm job running? (squeue --me)"; exit 1; fi

if [[ "${1:-}" == "--close" ]]; then
  "${SSH[@]}" -O cancel -L "$LOCAL_PORT:$NODE:$PORT" "$LOGIN" && echo "forward closed"; exit 0
fi

# API key -> .env without printing it
KEY="$("${SSH[@]}" "$LOGIN" 'cat ~/.hdb_vllm_key')"
touch "$ENV_FILE"; chmod 600 "$ENV_FILE"
grep -v '^VLLM_API_KEY=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
printf 'VLLM_API_KEY=%s\n' "$KEY" >> "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
unset KEY

if [[ "${SSH[*]}" == *"-S"* ]]; then
  "${SSH[@]}" -O forward -L "$LOCAL_PORT:$NODE:$PORT" "$LOGIN"
  echo "Forward added: localhost:$LOCAL_PORT -> $NODE:$PORT (job $JOB), lives as long as the shared session."
else
  echo "Tunnelling localhost:$LOCAL_PORT -> $NODE:$PORT (job $JOB). Ctrl-C to close."
  exec ssh -N -L "$LOCAL_PORT:$NODE:$PORT" "$LOGIN"
fi
