"""Merge the QLoRA adapter into the base model, ready for GGUF conversion.

llama.cpp cannot load a PEFT adapter on top of a GGUF at inference time in the
form this project ships it, so the CPU path merges first:

    base (Qwen2.5-0.5B-Instruct, fp16)  +  checkpoints/sft-0.5b/lora
        -> checkpoints/sft-0.5b/merged  (HF format, fp16)
        -> convert_hf_to_gguf.py        (F16 GGUF)
        -> llama-quantize               (Q8_0 GGUF, ~0.5 GB)

Only the merge step needs torch/transformers/peft; it is a one-off you can run
on any machine (including the CUDA training box) and the resulting .gguf is
what the 32 GB CPU box actually loads.  ``scripts/export_gguf.sh`` drives the
whole chain.

Note the adapter was trained on the *4-bit NF4* base while this merges into the
fp16 base.  That is the standard QLoRA export path and is what the published
0.70 greedy / 0.90 best-of-N numbers should be re-checked against on CPU —
see CPU_LLAMACPP.md.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def merge(base_model: str, adapter_dir: str, out_dir: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading base {base_model} (fp16, CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True)

    print(f"[2/3] applying adapter {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir, torch_dtype=torch.float16)
    model = model.merge_and_unload()
    model.config.torch_dtype = torch.float16

    print(f"[3/3] saving merged model -> {out}")
    model.save_pretrained(out, safe_serialization=True)

    # Prefer the adapter's tokenizer (it carries the trained chat template);
    # fall back to the base model's.
    try:
        tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(out)

    # convert_hf_to_gguf.py reads the chat template from tokenizer_config.json;
    # the training run stored it as a sibling chat_template.jinja.
    tmpl = Path(adapter_dir) / "chat_template.jinja"
    if tmpl.exists() and not (out / "chat_template.jinja").exists():
        shutil.copy2(tmpl, out / "chat_template.jinja")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/sft-0.5b/lora")
    ap.add_argument("--out", default="checkpoints/sft-0.5b/merged")
    args = ap.parse_args()
    out = merge(args.base_model, args.adapter, args.out)
    print(f"merged model written to {out}")


if __name__ == "__main__":
    main()
