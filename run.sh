#!/usr/bin/env bash
# Start the centering bench. Creates the virtual environment on first run.
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    ./.venv/bin/pip install --quiet --upgrade pip
    ./.venv/bin/pip install --quiet -r requirements.txt
fi

# Scans are read from the parent folder unless BENCH_SCANS says otherwise.
./.venv/bin/python server.py
status=$?

# Keep the terminal open if double-clicked and something goes wrong.
if [ $status -ne 0 ]; then
    echo "Server exited with an error (status $status)."
    read -n 1 -s -r -p "Press any key to close..."
    echo
fi
