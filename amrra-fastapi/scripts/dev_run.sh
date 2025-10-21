#!/usr/bin/env bash
set -euo pipefail
export REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
export CORPUS_DIR=${CORPUS_DIR:-./corpus}
uvicorn app.api:app --reload --port 8000