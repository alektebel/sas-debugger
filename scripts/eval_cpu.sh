#!/usr/bin/env bash
# Evaluate the model vs the deterministic oracle on a CPU-only box, through a
# running llama-server (start it with scripts/llama_server.sh first).
#
#   bash scripts/eval_cpu.sh                       # full held-out split
#   SAMPLE_LIMIT=40 BEST_OF_N=6 bash scripts/eval_cpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

TEST="${TEST:-data/generated/sft.jsonl}"
OUT="${OUT:-eval/cpu-llamacpp-report.json}"
LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://127.0.0.1:8080}"

args=(--test_jsonl "$TEST" --backend llamacpp --llama_url "$LLAMA_SERVER_URL"
      --json_out "$OUT")
[ -n "${SAMPLE_LIMIT:-}" ] && args+=(--sample_limit "$SAMPLE_LIMIT")
[ -n "${BEST_OF_N:-}" ] && args+=(--best_of_n "$BEST_OF_N")

"$PY" -m slm.evaluate "${args[@]}"
echo "wrote $OUT"
