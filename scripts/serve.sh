#!/usr/bin/env bash
# Serve the built frontend + collector API on one port (production).
# Prereq: npm run build:wasm && npm run build (dist/ must exist).
#
#   PORT=8000 PUSHT_EPISODES_DIR=/workspace/episodes scripts/serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m uvicorn server.main:app --host 0.0.0.0 --port "${PORT:-8000}"
