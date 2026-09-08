# SAS Reconcile SLM

New PwC project. Idea under design (see firstmate `slm-db-debug-council` report).

## Goal

With a small language model (0.5b-4b), debug where discrepancies between two SAS-produced
databases/tables come from. The captain uploads the SAS projects (.egp zip files) and an
Excel of "cycles" (1000-100000 rows) they have doubts about; the model localizes which
table/field/transformation makes a value wrong, may request key-table extractions, and emits
the diagnostic SQL query it needs.

## Current status

- **Post-training pipeline built (SFT-first).** See `TRAINING_REPORT.md` for the
  full environment/data/training/eval report and how to rerun.
- A working QLoRA SFT of `Qwen/Qwen2.5-0.5B-Instruct` is in
  `checkpoints/sft-0.5b/` (adapter in `lora/`); held-out success ≈ 0.70 greedy,
  0.90 with best-of-N. Stage-2 GRPO is **prepared but gated** (not launched).

## Post-training pipeline (rerunnable)

- `vendor/` — RegLLM prior art vendored (defect catalog, DB generator, expr
  oracle, SAS logic tree, sample SAS). No runtime dependency on regllm.
- `slm/oracle.py` — deterministic grading oracle (ground truth by construction).
- `slm/episodes.py` — oracle-verified episode/SFT data generator (stratified
  train/test). `slm/lineage.py` — schema + 7-layer lineage context.
- `slm/train_sft.py` — QLoRA SFT (HF transformers+peft, custom collator, no vLLM).
- `slm/evaluate.py` — eval vs oracle + base baselines, incl. best-of-N.
- `slm/reward.py`, `slm/grpo_config.py` — stage-2 GRPO reward + gated config.
- `scripts/{gen_data,train,eval,grpo_prep}.sh` — runnable wrappers.
- Worked example of the whole agent pass on a deep 8-table pipeline:
  `examples/deep_debug_worked_example.md` (reproduce with
  `examples/deep_debug_run.py`).
- Live-debug demo app (FastAPI + SSE, watch the agent stream its trace):
  `scripts/app.sh` -> http://127.0.0.1:8010 (uses the bundled two sample
  `.egp` projects + `.xlsx` of suspect cycles in `examples/app_assets/`).
- Env: `.venv` (Python 3.14, `--system-site-packages` reuses the CUDA torch
  2.14+cu130 already installed). GPU: RTX 5060 Ti 16 GB (Blackwell). Do NOT
  use vLLM on this GPU.

## CPU-only / llama.cpp path

Generation is backend-pluggable (`slm/backends.py`: `llamacpp` over HTTP to
`llama-server`, or `transformers`). **Only `slm/train_sft.py` needs CUDA** —
everything else (oracle, episode generation, lineage, reward, `vendor/`, the
demo app) is SQLite + stdlib. Do not reintroduce a module-level `import torch`
outside `train_sft.py`; `tests/test_cpu_llamacpp.py` asserts against it.
Setup, scripts, resource envelope and the two llama-server failure modes we
hit: `CPU_LLAMACPP.md`.

## Reference assets

- SAS field-lineage parser (from-scratch): `~/Documents/Development/learn-stuff-from-scratch/sas-lineage-tool/`
  - `solutions/lineage_parser.py`, `template_lineage_parser.py`
- RegLLM SAS logic tree: `regllm/src/sas_logic_tree.py`
- Candidate external parser: `alektebel/sas-lineage` on GitHub

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
