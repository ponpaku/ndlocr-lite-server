#!/usr/bin/env bash
#
# Run NDLOCR-Lite web server using Python 3.13.
#
# Python 3.14+ is not supported due to onnxruntime-gpu requiring numpy<2.3.
# Use this script when your default python3 is 3.14+ but Python 3.13 is also
# installed.  Install Python 3.13 via your package manager or pyenv if needed.
#

set -e

VENV=".venv313"

# --- Check that python3.13 is available ---
if ! command -v python3.13 &>/dev/null; then
    echo ""
    echo "[エラー] python3.13 が見つかりません。"
    echo "         sudo apt install python3.13 python3.13-venv などでインストールしてください。"
    echo ""
    exit 1
fi

# --- Copy config.toml.example → config.toml if not present ---
if [ ! -f "config.toml" ] && [ -f "config.toml.example" ]; then
    echo "Copying config.toml.example to config.toml"
    cp "config.toml.example" "config.toml"
fi

# --- Select requirements file based on OS ---
if [ "$(uname -s)" = "Darwin" ]; then
    REQ_FILE="requirements-cpu.txt"
else
    REQ_FILE="requirements-gpu.txt"
fi

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment in $VENV using python3.13"
    python3.13 -m venv "$VENV"
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip
    echo "Installing dependencies from $REQ_FILE ..."
    pip install --prefer-binary -r "$REQ_FILE"
else
    source "$VENV/bin/activate"
fi

# Launch the server; forward any arguments to Python
python server/main.py "$@"
