# Running on a CPU-only box (32 GB RAM, llama.cpp)

The project was built on an RTX 5060 Ti with QLoRA/bitsandbytes. This document
is the path for a machine with **no GPU, 32 GB of RAM and llama.cpp**.

## What actually needed porting

Most of the repo never touched the GPU. Verified by import graph:

| Component | GPU? | Notes |
|---|---|---|
| `slm/oracle.py`, `slm/episodes.py`, `slm/lineage.py`, `slm/reward.py` | no | SQLite + stdlib only |
| `vendor/*` (defect catalog, DB generator, expr oracle, SAS logic tree) | no | stdlib only |
| `examples/deep_debug_run.py`, `examples/app/` | no | SQLite + `openpyxl` |
| `slm/evaluate.py` | **was** | hard-coded `.to("cuda")` + 4-bit bnb |
| `slm/train_sft.py` | **yes** | QLoRA NF4 needs CUDA — training stays on the GPU box |

So the port is one seam: **generation**. It now goes through
`slm/backends.py`, which offers `llamacpp` (HTTP to `llama-server`) and
`transformers` (the original CUDA path, with a plain-CPU fallback).

Training is deliberately *not* ported. bitsandbytes 4-bit requires CUDA, and a
CPU LoRA run of the 378 training steps would take on the order of a day
against ~58 minutes on the GPU. Train on the GPU box, export a GGUF, run the
GGUF anywhere.

## One-time setup

```bash
# 1. runtime deps (no torch)
python3 -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt

# 2. llama.cpp
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -B ~/llama.cpp/build -S ~/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j --target llama-server llama-quantize

# 3. one-off GGUF export (needs torch once; the CPU wheel is enough)
python3 -m venv .venv-export
.venv-export/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-export/bin/pip install -r requirements-export.txt
PY=.venv-export/bin/python LLAMA_CPP=~/llama.cpp bash scripts/export_gguf.sh
```

`scripts/export_gguf.sh` merges `checkpoints/sft-0.5b/lora` into the fp16
`Qwen/Qwen2.5-0.5B-Instruct`, converts to F16 GGUF, then quantizes to **Q8_0**
(~0.5 GB). On 32 GB of RAM there is no reason to go below Q8_0 for a 0.5B —
`QUANT=Q4_K_M` is available if you want to measure the degradation.

## Running

```bash
# terminal 1 — the model server
LLAMA_CPP=~/llama.cpp bash scripts/llama_server.sh          # 127.0.0.1:8080

# terminal 2 — evaluation against the deterministic oracle
bash scripts/gen_data.sh                                    # no GPU needed
bash scripts/eval_cpu.sh                                    # full held-out split
SAMPLE_LIMIT=40 BEST_OF_N=6 bash scripts/eval_cpu.sh        # best-of-N subset

# terminal 2 — the live-debug demo app
LLAMA_SERVER_URL=http://127.0.0.1:8080 bash scripts/app.sh  # 127.0.0.1:8010
```

The demo app has an **Engine** selector:

* *Scripted reference trace* — the original deterministic pass. No model.
* *SFT 0.5B via llama.cpp* — the model actually writes the SQL, and every query
  it emits is executed against the real trap/clean databases and graded by the
  same oracle. It samples up to 4 times and keeps the first attempt that
  catches the planted cycle with no false positives.

The page probes `/api/backend` on load and tells you whether a llama-server is
reachable; with none, `mode=auto` falls back to the scripted trace.

## Resource envelope (32 GB box)

| Step | RAM | Notes |
|---|---|---|
| LoRA merge (one-off) | ~3 GB | fp16 0.5B on CPU |
| F16 GGUF conversion | ~2 GB | |
| `llama-server` Q8_0 @ `-c 8192` | ~1 GB | weights ~0.5 GB + KV cache |
| Episode generation / oracle | < 0.5 GB | SQLite |

Throughput on CPU is the real cost, not memory: expect single-digit seconds per
episode for a 0.5B at ~2.4k-token prompts, so the 268-episode held-out split is
tens of minutes, and best-of-N (N=6) multiplies that. Use `SAMPLE_LIMIT` while
iterating.

## Things that bit us, and what the code does about them

Both were reproduced against a real `llama-server`, not assumed:

1. **`HTTP 400 … exceeds the available context size`.** Episode prompts run
   ~2.4k tokens; a server started with a small `-c` rejects them outright.
   `scripts/llama_server.sh` therefore defaults to `-c 8192`, and
   `slm/backends.py` rewrites the error to say exactly how to fix it.
2. **`HTTP 500 … does not match the expected peg-native format`.**
   llama-server's chat-template parser rejects some sampled outputs. Aborting a
   268-episode run over one of those is worse than scoring it as the failure it
   is, so `LlamaCppBackend` counts them (`llama_server_errors` in the eval JSON)
   and returns an empty completion. Pass `tolerate_server_errors=False` to make
   it raise instead.

## Honest caveats

* **The published numbers were measured on 4-bit NF4 CUDA, not on Q8_0 GGUF.**
  The 0.70 greedy / 0.90 best-of-N figures in `TRAINING_REPORT.md` come from the
  bitsandbytes path. Merging LoRA into the fp16 base and re-quantizing to Q8_0
  is a different numerical path; re-run `scripts/eval_cpu.sh` and compare rather
  than assuming the numbers carry over.
* **The demo app's model mode is out-of-distribution.** The checkpoint was
  trained on the single-table `ciclos_calibrados` defect catalog
  (`vendor/defect_catalog.py`), while the demo debugs a bespoke 8-table
  pipeline. Expect it to do worse there than the 0.70 held-out number; that mode
  is a generalisation stress test, and the scripted trace remains the reference
  pass. `scripts/eval_cpu.sh` is the in-distribution measurement.

## Tests

`tests/test_cpu_llamacpp.py` covers the CPU path with the stdlib only — no
torch, no GPU, no model download. A stub HTTP server speaks the subset of the
llama-server API the backend uses, so the chain
*episode → backend → generated JSON → oracle grading → metrics* is exercised end
to end, including that importing the CPU modules never pulls in torch.

```bash
python3 -m unittest discover -s tests -v
```
