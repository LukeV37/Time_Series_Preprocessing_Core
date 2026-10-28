#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
else
    source "$VENV_DIR/bin/activate"
fi
