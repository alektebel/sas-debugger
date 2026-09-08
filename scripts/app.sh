#!/usr/bin/env bash
# Run the live-debug demo web app. Open the printed URL in a browser.
#   PORT=8010 bash scripts/app.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8010}"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# Optional: point the app at a llama.cpp server so the demo's "Engine" toggle
# can run the real model instead of the scripted trace (see CPU_LLAMACPP.md).
#   LLAMA_SERVER_URL=http://127.0.0.1:8080 bash scripts/app.sh

# (Re)create the bundled sample assets (two .egp projects + one .xlsx).
"$PY" examples/make_app_assets.py

echo "Demo app running — open http://127.0.0.1:${PORT} in your browser."
echo "Press Ctrl-C to stop."
PORT="$PORT" "$PY" examples/app/server.py
