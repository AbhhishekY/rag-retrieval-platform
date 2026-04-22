#!/usr/bin/env bash
# bootstrap.sh — full setup + run on a fresh machine (Mac/Linux)
# Usage: bash bootstrap.sh
set -e

echo ""
echo "  RAG Retrieval Platform — Bootstrap"
echo "  ────────────────────────────────────"

# 1. Check Python version
PY=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PY" | cut -d. -f1)
MINOR=$(echo "$PY" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || [ "$MINOR" -lt 11 ]; then
  echo "  ✗  Python 3.11+ required (found $PY)"
  exit 1
fi
echo "  ✓  Python $PY"

# 2. Create venv if missing
if [ ! -d ".venv" ]; then
  echo "  →  Creating .venv..."
  python3 -m venv .venv
fi

# 3. Activate
source .venv/bin/activate
echo "  ✓  venv activated"

# 4. Install dependencies
echo "  →  Installing dependencies (first run: ~2 min)..."
pip install -e ".[dev]" --quiet
echo "  ✓  Dependencies installed"

# 5. Run the full pipeline
echo ""
python run.py "$@"
