#!/bin/bash

# Run script for 3D Generative Models Backend
# This script starts the FastAPI server

set -e

echo "🚀 Starting 3D Generative Models Backend..."


# Check if configuration files exist
if [ ! -f "config/system.yaml" ]; then
    echo "❌ Configuration file config/system.yaml not found"
    echo "Please run ./scripts/setup.sh to create configuration files"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default values
HOST=${P3D_HOST:-"0.0.0.0"}
PORT=${P3D_PORT:-7842}
RELOAD=${P3D_RELOAD:-"false"}
LOG_LEVEL=${P3D_LOG_LEVEL:-"info"}

echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Reload: $RELOAD"
echo "   Log Level: $LOG_LEVEL"
echo ""

# Start server based on environment
if [ "$RELOAD" = "true" ]; then
    echo "🔄 Starting development server with auto-reload..."
    uvicorn api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    echo "🚀 Starting production server (current ONLY a single worker supported)..."
    # Single worker
    uvicorn api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL"

fi
