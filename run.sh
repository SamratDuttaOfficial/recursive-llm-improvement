#!/usr/bin/env bash
# Phase 1: write exercises and build the corpus of good answers.
# Prepares Python and the virtual environment on first use, then runs run.py.
# Any arguments are passed straight through, e.g.  run.sh --generator latest --new
set -euo pipefail
cd "$(dirname "$0")"
./setup.sh
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
exec "$PY" run.py "$@"
