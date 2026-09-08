#!/usr/bin/env bash
# Export the trained QLoRA adapter to a GGUF that llama.cpp can serve on CPU.
#
#   base fp16 + LoRA  ->  merged HF  ->  F16 GGUF  ->  Q8_0 GGUF (~0.5 GB)
#
# Requirements (one-off, on any machine — a GPU is NOT needed):
#   pip install -r requirements-export.txt      # torch(cpu) + transformers + peft
#   git clone https://github.com/ggml-org/llama.cpp && cmake -B build && cmake --build build -j
#
# Usage:
#   LLAMA_CPP=~/llama.cpp bash scripts/export_gguf.sh
#   QUANT=Q4_K_M LLAMA_CPP=~/llama.cpp bash scripts/export_gguf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

BASE="${BASE:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/sft-0.5b/lora}"
MERGED="${MERGED:-checkpoints/sft-0.5b/merged}"
GGUF_DIR="${GGUF_DIR:-checkpoints/sft-0.5b/gguf}"
QUANT="${QUANT:-Q8_0}"          # Q8_0 on a 0.5B: 32 GB of RAM makes Q4 pointless
NAME="${NAME:-sas-reconcile-0.5b}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"

F16="$GGUF_DIR/$NAME-F16.gguf"
OUT="$GGUF_DIR/$NAME-$QUANT.gguf"

CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
if [ ! -f "$CONVERT" ]; then
  echo "error: $CONVERT not found. Set LLAMA_CPP=/path/to/llama.cpp" >&2
  exit 1
fi

QUANTIZE="$(command -v llama-quantize || true)"
for c in "$LLAMA_CPP/build/bin/llama-quantize" "$LLAMA_CPP/llama-quantize"; do
  [ -n "$QUANTIZE" ] && break
  [ -x "$c" ] && QUANTIZE="$c"
done

mkdir -p "$GGUF_DIR"

echo "==> [1/3] merge LoRA into the fp16 base"
"$PY" -m slm.export_gguf --base_model "$BASE" --adapter "$ADAPTER" --out "$MERGED"

echo "==> [2/3] convert merged HF model to F16 GGUF"
"$PY" "$CONVERT" "$MERGED" --outfile "$F16" --outtype f16

if [ -z "$QUANTIZE" ]; then
  echo "warning: llama-quantize not found; keeping F16 GGUF only." >&2
  echo "         build llama.cpp, then: llama-quantize $F16 $OUT $QUANT" >&2
  OUT="$F16"
else
  echo "==> [3/3] quantize F16 -> $QUANT"
  "$QUANTIZE" "$F16" "$OUT" "$QUANT"
fi

echo
echo "GGUF ready: $OUT"
ls -lh "$OUT"
echo "Serve it with:  LLAMA_CPP=$LLAMA_CPP GGUF=$OUT bash scripts/llama_server.sh"
