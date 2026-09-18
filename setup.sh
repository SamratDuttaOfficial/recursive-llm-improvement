#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Prepares everything this project needs, and touches nothing outside its own
#  folder. No sudo, no PATH changes, no effect on any Python you already use
#  for other work.
#
#    .python/   a private CPython, downloaded ONLY when this machine has none
#               that is new enough. It lives in this folder and nowhere else.
#    .venv/     the environment every script runs in
#
#  Safe to run as often as you like: it checks before it does anything, and
#  does nothing at all once the environment exists.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
PYDIR="$ROOT/.python"
VENV="$ROOT/.venv"
VPY="$VENV/bin/python"
[ -x "$VPY" ] || [ ! -x "$VENV/Scripts/python.exe" ] || VPY="$VENV/Scripts/python.exe"
PYVER="3.12.7"
PBSTAG="20241016"

if [ -x "$VPY" ]; then
  [ "${1:-}" = "-v" ] && echo "[setup] environment already prepared: $VENV"
  exit 0
fi

usable() {                       # usable <python> -> 0 when it is 3.9 or newer
  [ -x "$1" ] || command -v "$1" >/dev/null 2>&1 || return 1
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1
}

# ---- 1. is there already a Python we may use? -----------------------------
PY=""
for cand in "$PYDIR/bin/python3" python3 python; do
  if usable "$cand"; then PY="$cand"; break; fi
done

if [ -n "$PY" ]; then
  echo "[setup] using the Python already on this machine: $PY  ($("$PY" -c 'import sys;print(sys.version.split()[0])'))"
else
  # ---- 2. none found: fetch a private one just for this project ------------
  case "$(uname -s)" in
    Darwin) case "$(uname -m)" in
              arm64|aarch64) TRIPLE="aarch64-apple-darwin" ;;
              *)             TRIPLE="x86_64-apple-darwin" ;;
            esac ;;
    Linux)  case "$(uname -m)" in
              aarch64|arm64) TRIPLE="aarch64-unknown-linux-gnu" ;;
              *)             TRIPLE="x86_64-unknown-linux-gnu" ;;
            esac ;;
    *)      echo "[setup] unsupported platform $(uname -s); install Python 3.9+ and re-run"; exit 1 ;;
  esac

  echo "[setup] no Python 3.9+ found on this machine."
  echo "[setup] downloading a private CPython $PYVER into $PYDIR"
  echo "[setup] (this copy is used only by this project - nothing else changes)"

  BASE="https://github.com/astral-sh/python-build-standalone/releases"
  URL="$BASE/download/$PBSTAG/cpython-$PYVER+$PBSTAG-$TRIPLE-install_only.tar.gz"
  # Prefer whatever the newest release offers, falling back to the pinned URL above. The API
  # percent-encodes the '+' in the file name, so both spellings have to be accepted here.
  LATEST=$(curl -fsSL -H 'User-Agent: rli-setup' \
             "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" 2>/dev/null \
           | grep -oE "https://[^\"]*cpython-3\.12\.[0-9]+(%2B|\+)[0-9]+-$TRIPLE-install_only\.tar\.gz" \
           | head -1 || true)
  [ -n "$LATEST" ] && URL="$LATEST"
  echo "[setup] $URL"

  TGZ="$(mktemp -t rli-cpython.XXXXXX).tar.gz"
  if ! curl -fSL --retry 3 -o "$TGZ" "$URL"; then
    echo
    echo "[setup] Could not download a private Python automatically."
    echo "[setup] Install Python 3.9 or newer and run this again - it will use it"
    echo "[setup] and change nothing on your system."
    exit 1
  fi
  rm -rf "$PYDIR" "${PYDIR}_tmp"
  mkdir -p "${PYDIR}_tmp"
  tar -xzf "$TGZ" -C "${PYDIR}_tmp"
  mv "${PYDIR}_tmp/python" "$PYDIR"
  rm -rf "${PYDIR}_tmp" "$TGZ"
  PY="$PYDIR/bin/python3"
  [ -x "$PY" ] || { echo "[setup] the downloaded Python did not unpack as expected"; exit 1; }
  echo "[setup] private Python ready: $PY"
fi

# ---- 3. the virtual environment every script runs in ----------------------
echo "[setup] creating the virtual environment in $VENV"
"$PY" -m venv "$VENV"
[ -x "$VPY" ] || VPY="$VENV/Scripts/python.exe"   # a venv made under Git Bash on Windows
"$VPY" -m pip install -q --upgrade pip wheel
echo "[setup] ready. Heavier pieces (PyTorch or MLX, ruff, ...) install themselves when first needed."
