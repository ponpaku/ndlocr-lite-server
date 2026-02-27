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

# --- Copy config.toml.example → config.toml if not present ---
if [ ! -f "config.toml" ] && [ -f "config.toml.example" ]; then
    echo "Copying config.toml.example to config.toml"
    cp "config.toml.example" "config.toml"
fi

# --- Read runtime.package from config.toml (default: gpu) ---
# Supported values: gpu / directml / cpu
# !! runtime.package を変更した場合は .venv を削除して再実行してください !!
NDLOCR_RUNTIME="gpu"
if [ -f "config.toml" ]; then
    _pkg=$(sed -n '/^\[runtime\]/,/^\[/{/^package[[:space:]]*=/{s/.*=[[:space:]]*"\([^"]*\)".*/\1/;p;q}}' config.toml 2>/dev/null)
    [ -n "$_pkg" ] && NDLOCR_RUNTIME="$_pkg"
fi

case "$NDLOCR_RUNTIME" in
    gpu)       REQ_FILE="requirements-gpu.txt" ;;
    directml)  REQ_FILE="requirements-directml.txt" ;;
    cpu)       REQ_FILE="requirements-cpu.txt" ;;
    *)         REQ_FILE="requirements-gpu.txt" ;;
esac

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
