#!/usr/bin/env bash
#
# Run NDLOCR‑Lite web server inside a Python virtual environment.
#
# This script creates a virtual environment in the repository root (``.venv``)
# if it does not already exist, installs required Python packages, and then
# launches the FastAPI server defined in ``server/main.py``.  Any
# additional command line arguments are passed through to the Python
# interpreter.

set -e

# Directory of the virtual environment relative to this script
VENV=".venv"

# --- Python version check (3.14+ は非対応) ---
PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -gt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 14 ]; }; then
    echo ""
    echo "[警告] Python $PY_VER は非対応です（推奨: 3.11〜3.13）"
    echo "       Python 3.13 がインストール済みであれば run-py313.sh を使用してください。"
    echo ""
    read -r -p "続行する場合は Enter、中止する場合は Ctrl+C を押してください..."
fi

# --- Copy config.toml.example → config.toml if not present ---
if [ ! -f "config.toml" ] && [ -f "config.toml.example" ]; then
    echo "Copying config.toml.example to config.toml"
    cp "config.toml.example" "config.toml"
fi

# --- Select requirements file based on OS ---
# macOS does not support CUDA, so use CPU-only onnxruntime.
# Windows and Linux use onnxruntime-gpu (falls back to CPU automatically).
if [ "$(uname -s)" = "Darwin" ]; then
    REQ_FILE="requirements-cpu.txt"
else
    REQ_FILE="requirements-gpu.txt"
fi

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment in $VENV"
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip
  echo "Installing dependencies from $REQ_FILE ..."
  pip install -r "$REQ_FILE"
else
  source "$VENV/bin/activate"
fi

# Launch the server; forward any arguments to Python
python server/main.py "$@"
