#!/usr/bin/env bash
# Phase 3: measure every model that has no results yet.
# Prepares Python and the virtual environment on first use, then runs benchmark.py.
# Any arguments are passed straight through, e.g.  benchmark.sh --models base,v1
set -euo pipefail
cd "$(dirname "$0")"
./setup.sh
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
exec "$PY" benchmark.py "$@"
