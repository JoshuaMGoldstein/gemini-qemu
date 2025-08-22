#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.llmvm/pids/puppy.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Puppy VM (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "VM stopped"
    else
        echo "VM not running"
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found"
fi
