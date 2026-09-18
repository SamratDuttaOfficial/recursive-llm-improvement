#!/usr/bin/env bash
# Phase 2: fine-tune on the answers the corpus holds.
# Prepares Python and the virtual environment on first use, then runs finetune.py.
# Any arguments are passed straight through, e.g.  finetune.sh --from-model latest
set -euo pipefail
cd "$(dirname "$0")"
./setup.sh
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
exec "$PY" finetune.py "$@"
