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
  pip install --prefer-binary -r "$REQ_FILE"
else
  source "$VENV/bin/activate"
fi

# Launch the server; forward any arguments to Python
python server/main.py "$@"
