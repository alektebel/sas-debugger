#!/usr/bin/env bash
# Regenerate the verified SFT dataset (>=2k episodes, stratified >=10% test).
set -euo pipefail
cd "$(dirname "$0")/.."

# No GPU and no torch needed here: episode generation is SQLite + stdlib.
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

N_PER_DEFECT="${N_PER_DEFECT:-34}"
ROWS="${ROWS:-1000}"
OUT="${OUT:-data/generated/sft.jsonl}"

"$PY" -m slm.episodes "$N_PER_DEFECT" "$ROWS" "$OUT"
echo "wrote $OUT"
