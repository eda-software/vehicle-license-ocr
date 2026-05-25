#!/bin/bash
# Moikan OCR Service Startup Script

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=== Moikan OCR Service Startup ==="

if [ -d "venv" ]; then
    echo "Starting OCR service using virtual environment..."
    ./venv/bin/python ocr.py
else
    echo "Virtual environment not found! Running with system python..."
    python3 ocr.py
fi
