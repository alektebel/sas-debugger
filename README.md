# sas-debugger — SAS Reconcile SLM

**Small language model that localises the root cause of a discrepancy between
two SAS-produced tables.**

## The objective

In risk/regulatory reporting (IRB, IFRS 9, CRR), a SAS pipeline produces a final
table (e.g. `ciclos_calibrados`: PD/LGD/EAD/ECL/RWA per recovery cycle) by
chaining many transformations over several source tables. When a captain reviews
the output and flags a set of "cycles" they doubt — `1000–100000` rows in an
Excel — the hardest question to answer is:

> **Which table, which field, and which transformation makes this value wrong?**

A senior analyst can reason through the lineage, but that is slow and does not
scale to hundreds of suspicious cycles. This project post-trains a **small
language model (0.5–4B)** to do it automatically:

- It is given the SAS **projects** (`.egp` files) that produced the tables,
  the **schema + lineage** of the pipeline, and the **suspected cycles**.
- Its action space is a *debugging loop*: request a **key-table extraction**,
  emit a **diagnostic SQL query**, and produce a **localisation**
  (`table / columns / transformation / reason`).
- The result is the diagnostic SQL and the root-cause attribution the captain
  can run and act on.

## Why this is trainable without real bank data

Ground truth is **correct by construction**. We inject *exactly one* known defect
into a synthetic database (`vendor/generate_db`), so the oracle always knows the
true table / column / transformation. This makes the task fully supervised and
the reward fully **execution-verifiable** — the grading does not depend on an LLM
judge, only on whether the emitted SQL catches the planted row and stays clean on
the reference database.

## Pipeline

```
vendor/          RegLLM prior art (defect catalog, DB generator, expr oracle,
                 SAS logic tree, sample SAS) — vendored with attribution
slm/oracle.py    deterministic grading oracle (ground truth by construction)
slm/episodes.py  oracle-verified episode / SFT data generator (stratified train/test)
slm/lineage.py   compact schema + 7-layer lineage context
slm/train_sft.py QLoRA SFT (HF transformers + peft + bitsandbytes, no vLLM)
slm/evaluate.py  eval vs oracle + base baselines, incl. best-of-N
slm/reward.py    stage-2 GRPO reward (execution-verifiable)
slm/grpo_config.py   stage-2 GRPO config (PREPARED, GATED — not launched)
examples/        worked deep-table example + live-debug demo app (FastAPI + SSE)
scripts/         runnable wrappers (gen_data / train / eval / grpo_prep / app)
TRAINING_REPORT.md  full environment / data / training / eval report
```

## Results (held-out test split, n = 268)

Metrics graded by the execution-verified oracle: `valid` = SQL runs; `catch` =
catches the planted row; `FP` = fires on the clean reference; `loc` = root-cause
table + primary suspect column correct.

| Model | valid | catch | FP | loc | **success** |
|-------|------:|------:|---:|----:|:-----------:|
| Deterministic oracle (ceiling) | 1.00 | 1.00 | 0.00 | 1.00 | **1.00** |
| Untrained base `Qwen2.5-0.5B-Instruct` | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| **QLoRA SFT (0.5B, greedy)** | 0.86 | 0.72 | 0.02 | 0.70 | **0.70** |
| SFT + best-of-N (N=6) | 1.00 | 0.90 | 0.00 | 0.90 | **0.90** |

The SFT model learns the tool-use contract (86% valid queries, 86% useful
key-table extractions, only 1.9% false positives) and the deep lineage, but a
0.5B model cannot perfectly separate all 67 defect classes, so best-of-N is the
clean way to close the gap. Stage-2 GRPO is prepared but **gated** (see
`TRAINING_REPORT.md`).

## Live-debug demo app

```bash
bash scripts/app.sh            # http://127.0.0.1:8010
```

Load the two corresponding `.egp` projects + the `.xlsx` of suspect cycles and
watch the agent debug in real time — reasoning, key-table extractions,
diagnostic SQL, oracle verdicts and the final localisation (streamed via SSE).
Bundled samples are in `examples/app_assets/`.

The **Engine** selector switches between the deterministic reference trace and
the SFT'd 0.5B generating the SQL itself through llama.cpp; either way every
query is executed against the real trap/clean databases and graded by the same
oracle.

## Quickstart — CPU only, no GPU (32 GB + llama.cpp)

Training needs the GPU box; everything else does not. Full walkthrough,
resource envelope and caveats: **[CPU_LLAMACPP.md](CPU_LLAMACPP.md)**.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-cpu.txt   # no torch

PY=.venv-export/bin/python LLAMA_CPP=~/llama.cpp bash scripts/export_gguf.sh
LLAMA_CPP=~/llama.cpp bash scripts/llama_server.sh          # serves Q8_0, ~1 GB RAM

bash scripts/gen_data.sh                                    # SQLite + stdlib
bash scripts/eval_cpu.sh                                    # eval through llama.cpp
LLAMA_SERVER_URL=http://127.0.0.1:8080 bash scripts/app.sh  # demo, model in the loop
python3 -m unittest discover -s tests                       # CPU-path tests
```

## Quickstart — GPU (training)

```bash
python3 -m venv --system-site-packages .venv     # reuses a working CUDA torch
.venv/bin/pip install -r requirements.txt

bash scripts/gen_data.sh       # 2,278 oracle-verified episodes (stratified)
bash scripts/train.sh          # QLoRA SFT -> checkpoints/sft-0.5b/
bash scripts/eval.sh           # eval vs oracle + baselines -> eval/sft-report.json
bash scripts/app.sh            # live-debug demo (optional)
```

## Notes

- Env: RTX 5060 Ti 16 GB (Blackwell, compute 12.0). **No vLLM** (Blackwell
  compatibility risk).
- Generation is backend-pluggable (`slm/backends.py`): `llamacpp` (HTTP to
  `llama-server`, CPU) or `transformers` (4-bit CUDA). Only `slm/train_sft.py`
  still requires a GPU.
- RegLLM prior art is vendored under `vendor/` with attribution headers; this
  project has no runtime dependency on RegLLM.
- Synthetic data only — no real customer or bank data anywhere.
- Stage-2 GRPO is intentionally **not launched**: it is gated on SFT +
  best-of-N plateauing below target with a ≥0.9 execution-verifiable reward.
