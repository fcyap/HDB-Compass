#!/usr/bin/env bash
# ==============================================================================
# HDB-Compass Local Startup Script
# Starts both FastAPI backend (port 8000) and React/Vite frontend (port 3000)
# in a single command. Pressing Ctrl+C stops both servers cleanly.
#
# Usage:
#   ./start_app.sh
#   ./start_app.sh --profile ollama      # use local Ollama model
#   ./start_app.sh --profile groq        # use Groq API
#   ./start_app.sh --no-open             # don't open browser automatically
# ==============================================================================

set -uo pipefail

# ANSI color codes for friendly terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 0. Load .env if present
if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env" 2>/dev/null || true
  set +a
fi

# Default configuration
CLI_PROFILE=""
AUTO_OPEN=1

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|-p)
      CLI_PROFILE="$2"
      shift 2
      ;;
    --no-open)
      AUTO_OPEN=0
      shift
      ;;
    --help|-h)
      echo "Usage: ./start_app.sh [options]"
      echo ""
      echo "Options:"
      echo "  -p, --profile <name>  LLM profile (none, ollama, groq, vllm, openai) [default: auto-detected from .env]"
      echo "  --no-open             Do not automatically open the browser"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run ./start_app.sh --help for available options."
      exit 1
      ;;
  esac
done

if [ -n "$CLI_PROFILE" ]; then
  PROFILE="$CLI_PROFILE"
elif [ -n "${LLM_PROFILE:-}" ]; then
  PROFILE="$LLM_PROFILE"
elif [ -n "${GROQ_API_KEY:-}" ]; then
  PROFILE="groq"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
  PROFILE="openai"
else
  PROFILE="none"
fi

export LLM_PROFILE="$PROFILE"

echo -e "${BOLD}${BLUE}============================================================${NC}"
echo -e "${BOLD}${BLUE}             🧭 Launching HDB-Compass Web Application       ${NC}"
echo -e "${BOLD}${BLUE}============================================================${NC}"

# 1. Environment & Node.js detection (NVM support)
if [ -s "$HOME/.nvm/nvm.sh" ]; then
  # Source NVM without crashing on subshell errors
  source "$HOME/.nvm/nvm.sh" 2>/dev/null || true
  nvm use 22 >/dev/null 2>&1 || nvm use default >/dev/null 2>&1 || true
fi

NODE_VER="$(node -v 2>/dev/null || echo "not found")"
if [ "$NODE_VER" = "not found" ]; then
  echo -e "${RED}[ERROR] Node.js is not found. Please install Node.js 22 LTS or check PATH.${NC}"
  exit 1
fi
echo -e "${GREEN}[✓] Node environment:${NC} $NODE_VER (npm $(npm -v 2>/dev/null || echo '?'))"

# 2. Python environment detection
PYTHON=""
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/venv/bin/python" ]; then
  PYTHON="$ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo -e "${RED}[ERROR] Python 3 not found! Please create a virtualenv at .venv.${NC}"
  exit 1
fi
echo -e "${GREEN}[✓] Python environment:${NC} $($PYTHON --version 2>&1) ($PYTHON)"
echo -e "${GREEN}[✓] LLM Profile:${NC}       ${BOLD}$PROFILE${NC}"

# 3. Clean up lingering processes on ports 8000 & 3000
cleanup_port() {
  local port=$1
  local pids
  pids=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo -e "${YELLOW}[!] Port $port is already in use by PID(s): $pids. Freeing port...${NC}"
    kill -9 $pids 2>/dev/null || true
    sleep 0.5
  fi
}

cleanup_port 8000
cleanup_port 3000

# 4. Process cleanup on exit/interruption
BACKEND_PID=""
FRONTEND_PID=""

shutdown_all() {
  echo ""
  echo -e "${YELLOW}[*] Shutting down HDB-Compass servers...${NC}"
  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [ -n "$BACKEND_PID" ]; then
    kill $(pgrep -P "$BACKEND_PID" 2>/dev/null) "$BACKEND_PID" 2>/dev/null || true
  fi
  cleanup_port 8000
  cleanup_port 3000
  echo -e "${GREEN}[✓] All servers stopped cleanly. Goodbye!${NC}"
  exit 0
}

trap shutdown_all SIGINT SIGTERM EXIT

# 5. Check data snapshots
if [ ! -f "$ROOT/data/hdb_market.duckdb" ]; then
  echo -e "${YELLOW}[i] Note: data/hdb_market.duckdb not found. Resale price queries fall back or require: python -m data.build_market_db${NC}"
fi

# 6. Start Backend
echo -e "${BLUE}[*] Starting FastAPI Backend on http://127.0.0.1:8000 ...${NC}"
"$PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend health check
echo -n "    Waiting for backend to be ready"
READY=0
for i in {1..20}; do
  if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  echo -n "."
  sleep 0.5
done
echo ""

if [ "$READY" -eq 1 ]; then
  echo -e "${GREEN}[✓] Backend is healthy at http://127.0.0.1:8000 (PID $BACKEND_PID)${NC}"
else
  echo -e "${YELLOW}[!] Backend did not respond to /health in time, but process is running.${NC}"
fi

# 6. Start Frontend
echo -e "${BLUE}[*] Starting React/Vite Frontend on http://localhost:3000 ...${NC}"
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo -e "${YELLOW}[*] Installing frontend dependencies first...${NC}"
  (cd "$ROOT/frontend" && npm install)
fi

(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

# Brief pause to let Vite print its ready banner
sleep 2

echo ""
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e "${BOLD}${GREEN}  🚀 HDB-Compass is now running!                           ${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e "  • ${BOLD}Frontend (Web UI):${NC}    ${BOLD}http://localhost:3000${NC}"
echo -e "  • ${BOLD}Backend (FastAPI):${NC}    ${BOLD}http://127.0.0.1:8000${NC}"
echo -e "  • ${BOLD}API Health Check:${NC}     ${BOLD}http://127.0.0.1:8000/health${NC}"
echo -e "  • ${BOLD}API Documentation:${NC}    ${BOLD}http://127.0.0.1:8000/docs${NC}"
echo -e "------------------------------------------------------------"
echo -e "  ${YELLOW}Press Ctrl+C in this terminal to shut down both servers.${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo ""

# Auto-open browser if requested
if [ "$AUTO_OPEN" -eq 1 ]; then
  if command -v open >/dev/null 2>&1; then
    open "http://localhost:3000" 2>/dev/null || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:3000" 2>/dev/null || true
  fi
fi
# Keep script running and wait for background processes
wait "$FRONTEND_PID" "$BACKEND_PID" 2>/dev/null || true