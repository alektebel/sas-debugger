#!/usr/bin/env bash
# Serve the exported GGUF with llama.cpp on CPU. Leave this running in one
# terminal; scripts/eval_cpu.sh and scripts/app.sh talk to it over HTTP.
#
#   LLAMA_CPP=~/llama.cpp bash scripts/llama_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
GGUF="${GGUF:-checkpoints/sft-0.5b/gguf/sas-reconcile-0.5b-Q8_0.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"                     # episode prompts run ~2.4k tokens; 4096 is too
                                       # tight once the answer is added. A 0.5B KV
                                       # cache at 8192 is ~0.1 GB — free on 32 GB.
THREADS="${THREADS:-$(nproc)}"
PARALLEL="${PARALLEL:-1}"

SERVER="$(command -v llama-server || true)"
for c in "$LLAMA_CPP/build/bin/llama-server" "$LLAMA_CPP/llama-server"; do
  [ -n "$SERVER" ] && break
  [ -x "$c" ] && SERVER="$c"
done
if [ -z "$SERVER" ]; then
  echo "error: llama-server not found. Build llama.cpp or set LLAMA_CPP." >&2
  exit 1
fi
if [ ! -f "$GGUF" ]; then
  echo "error: $GGUF not found. Run: bash scripts/export_gguf.sh" >&2
  exit 1
fi

echo "llama-server on http://$HOST:$PORT  (ctx $CTX, $THREADS threads, CPU)"
exec "$SERVER" -m "$GGUF" --host "$HOST" --port "$PORT" \
  -c "$CTX" -t "$THREADS" -np "$PARALLEL" --no-warmup
