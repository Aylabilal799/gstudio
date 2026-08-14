#!/bin/bash

# Navigate to project directory
cd /root/gstudio

# Activate Python Virtual Environment
source venv/bin/activate

# Ensure output directory exists
mkdir -p output/jobs

# Stop any existing server on port 5454
fuser -k 5454/tcp 2>/dev/null || true

echo "=================================================="
echo "[+] Starting HTTP Video Server serving /root/gstudio on port 5454..."
python3 -m http.server 5454 --directory /root/gstudio &
HTTP_PID=$!

echo "[+] Starting GStudio Discord Bot..."
echo "=================================================="

# Run Discord Bot (Foreground)
python discord_bot.py

# Cleanup HTTP server process if bot stops
kill $HTTP_PID 2>/dev/null || true
