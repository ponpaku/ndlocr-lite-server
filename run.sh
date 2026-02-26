#!/usr/bin/env bash
#
# Run NDLOCR‑Lite web server inside a Python virtual environment.
#
# This script creates a virtual environment in the repository root (``.venv``)
# if it does not already exist, installs required Python packages, and then
# launches the FastAPI server defined in ``local_server/main.py``.  Any
# additional command line arguments are passed through to the Python
# interpreter, allowing you to customise the host/port via Uvicorn options.

set -e

# Directory of the virtual environment relative to this script
VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment in $VENV"
  python3 -m venv "$VENV"
  # Activate the virtual environment for installation
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip
  # Install dependencies from requirements.txt if present.  Support both
  # a top‑level requirements.txt (for monorepos) and the one in
  # ndlocr‑lite‑gui/requirements.txt when this script is in the project root.
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
  elif [ -f "ndlocr-lite-gui/requirements.txt" ]; then
    pip install -r "ndlocr-lite-gui/requirements.txt"
  fi
  # Ensure FastAPI server dependencies are installed
  pip install fastapi uvicorn pypdfium2 pillow
else
  # Activate existing virtual environment
  source "$VENV/bin/activate"
fi

# Launch the server; forward any arguments to Python
python server/main.py "$@"