#!/usr/bin/env bash
# All three phases in order (optional convenience).
# Prepares Python and the virtual environment on first use, then runs pipeline.py.
# Any arguments are passed straight through, e.g.  pipeline.sh --cycles 2
set -euo pipefail
cd "$(dirname "$0")"
./setup.sh
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
exec "$PY" pipeline.py "$@"
