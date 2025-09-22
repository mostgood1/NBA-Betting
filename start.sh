#!/bin/bash
set -euo pipefail

echo "🚀 Starting NBA Betting (static frontend) on Render..."

# Ensure directories
mkdir -p logs

python --version || true

export PYTHONUNBUFFERED=1

echo "Using PORT=${PORT:-5000} WEB_CONCURRENCY=${WEB_CONCURRENCY:-1} WEB_THREADS=${WEB_THREADS:-4}"

exec gunicorn app:app \
  --bind 0.0.0.0:${PORT:-5000} \
  --workers ${WEB_CONCURRENCY:-1} \
  --worker-class gthread \
  --threads ${WEB_THREADS:-4} \
  --timeout 120 \
  --log-level info
