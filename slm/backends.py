"""Pluggable generation backends.

The project was built against a CUDA box (RTX 5060 Ti, QLoRA/bitsandbytes).
Everything except *generation* is pure-Python + SQLite and already runs on a
CPU-only machine; this module isolates the one piece that is not, so the same
evaluation / agent code can run either against:

  * ``transformers``  — HF transformers, 4-bit NF4 on CUDA (the training box) or
                        plain fp32/bf16 on CPU (slow, but no bitsandbytes).
  * ``llamacpp``      — a running ``llama-server`` (llama.cpp) speaking the
                        OpenAI-compatible HTTP API.  No torch, no CUDA; a
                        32 GB CPU box loading a Q8_0 GGUF of the merged
                        0.5B model needs well under 1 GB.

Only ``urllib`` from the stdlib is used for the llama.cpp path, so the CPU
install needs neither ``torch`` nor ``requests`` (see ``requirements-cpu.txt``).

Backend selection (CLI ``--backend`` overrides these):
    SLM_BACKEND=llamacpp|transformers      default: llamacpp if LLAMA_SERVER_URL
                                           is set, else transformers
    LLAMA_SERVER_URL=http://127.0.0.1:8080 llama-server base URL
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

class LlamaServerError(RuntimeError):
    """A non-2xx answer from llama-server, carrying the HTTP status."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


DEFAULT_LLAMA_URL = "http://127.0.0.1:8080"
DEFAULT_MAX_NEW_TOKENS = 256
# Kept identical to the CUDA eval path so the two backends are comparable.
REPETITION_PENALTY = 1.05


class Backend:
    """Generate an assistant turn from a (system, user) chat pair."""

    name = "backend"

    def generate(self, system: str, user: str, *, do_sample: bool = False,
                 temperature: float = 0.7,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# llama.cpp (llama-server, OpenAI-compatible HTTP)
# --------------------------------------------------------------------------
class LlamaCppBackend(Backend):
    """Talk to a running ``llama-server``.

    Start it with ``bash scripts/llama_server.sh`` (or by hand):

        llama-server -m checkpoints/sft-0.5b/gguf/sas-reconcile-0.5b-Q8_0.gguf \
                     --host 127.0.0.1 --port 8080 -c 4096 -t $(nproc)

    ``/v1/chat/completions`` applies the chat template baked into the GGUF, so
    the prompt formatting matches what ``slm.train_sft`` trained on.
    """

    name = "llamacpp"

    def __init__(self, url: str = DEFAULT_LLAMA_URL, *, timeout: float = 300.0,
                 model: str = "sas-reconcile", seed: int | None = 42,
                 tolerate_server_errors: bool = True):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.seed = seed
        # llama-server answers 500 when its chat-template parser cannot make
        # sense of the tokens the model emitted ("does not match the expected
        # peg-native format").  Observed in practice on degenerate samples.
        # Aborting a 268-episode evaluation over one of those is worse than
        # scoring it as the failure it is, so by default we count it and carry
        # on; set tolerate_server_errors=False to let it raise.
        self.tolerate_server_errors = tolerate_server_errors
        self.server_errors = 0

    # -- health ------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        """(reachable, detail).  Never raises — callers degrade gracefully."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                body = json.loads(r.read().decode("utf-8") or "{}")
            status = body.get("status", "ok")
            return status in ("ok", "no slot available"), f"{self.url}: {status}"
        except urllib.error.HTTPError as e:  # 503 while the model still loads
            return False, f"{self.url}: HTTP {e.code} (model still loading?)"
        except Exception as e:  # connection refused, DNS, timeout
            return False, f"{self.url}: {type(e).__name__}: {e}"

    def require_health(self) -> None:
        ok, detail = self.health()
        if not ok:
            raise RuntimeError(
                f"llama-server is not reachable at {self.url} ({detail}).\n"
                f"Start it with:  bash scripts/llama_server.sh\n"
                f"Build the GGUF first with:  bash scripts/export_gguf.sh")

    # -- generation --------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise LlamaServerError(
                f"llama-server {path} -> HTTP {e.code}: {detail}", e.code) from e

    def generate(self, system: str, user: str, *, do_sample: bool = False,
                 temperature: float = 0.7,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            # temperature 0 == greedy in llama.cpp, matching do_sample=False.
            "temperature": float(temperature) if do_sample else 0.0,
            "max_tokens": int(max_new_tokens),
            "repeat_penalty": REPETITION_PENALTY,
            "stream": False,
        }
        if self.seed is not None and not do_sample:
            payload["seed"] = self.seed
        try:
            body = self._post("/v1/chat/completions", payload)
        except LlamaServerError as e:
            if e.code == 400 and "context size" in str(e):
                # The episode prompts run ~2.4k tokens; a server started with a
                # small -c rejects them outright.  Say so instead of leaving the
                # caller to decode llama.cpp's error.
                raise LlamaServerError(
                    f"{e}\nThe prompt does not fit the server's context. "
                    f"Restart it with a bigger window, e.g. "
                    f"CTX=8192 bash scripts/llama_server.sh", e.code) from e
            if not (self.tolerate_server_errors and e.code >= 500):
                raise
            self.server_errors += 1
            return ""  # graded as a failure by the oracle, which it is
        choices = body.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""


# --------------------------------------------------------------------------
# HF transformers (CUDA 4-bit, or CPU full-precision)
# --------------------------------------------------------------------------
class TransformersBackend(Backend):
    """The original CUDA path, with a CPU fallback.

    On CUDA it loads the 4-bit NF4 model exactly as training did.  With no CUDA
    visible it loads fp32 on CPU instead — bitsandbytes 4-bit needs a GPU, so
    "CPU + 4-bit" is not a thing.  For a 0.5B that is ~2 GB of RAM and roughly
    an order of magnitude slower than llama.cpp; prefer ``LlamaCppBackend``.
    """

    name = "transformers"

    def __init__(self, base_model: str, adapter_dir: str | None = None):
        import torch  # local import: the CPU install has no torch

        from slm.train_sft import load_tokenizer
        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.tok = load_tokenizer(base_model)

        if self.cuda:
            from slm.train_sft import load_4bit_model
            model = load_4bit_model(base_model)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.float32, trust_remote_code=True)

        if adapter_dir and adapter_dir not in ("none", "None", ""):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        if self.cuda:
            model.to("cuda")
        self.model = model

    def generate(self, system: str, user: str, *, do_sample: bool = False,
                 temperature: float = 0.7,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        tok, model = self.tok, self.model
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
        else:
            prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                      f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with self.torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                repetition_penalty=REPETITION_PENALTY,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    def close(self) -> None:
        model, self.model = getattr(self, "model", None), None
        del model
        if getattr(self, "cuda", False):
            self.torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def resolve_backend_name(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("SLM_BACKEND")
    if env:
        return env
    return "llamacpp" if os.environ.get("LLAMA_SERVER_URL") else "transformers"


def make_backend(backend: str | None = None, *, base_model: str | None = None,
                 adapter_dir: str | None = None,
                 llama_url: str | None = None) -> Backend:
    name = resolve_backend_name(backend)
    if name == "llamacpp":
        url = llama_url or os.environ.get("LLAMA_SERVER_URL", DEFAULT_LLAMA_URL)
        be = LlamaCppBackend(url)
        be.require_health()
        return be
    if name == "transformers":
        if not base_model:
            raise ValueError("the transformers backend needs --base_model")
        return TransformersBackend(base_model, adapter_dir)
    raise ValueError(f"unknown backend {name!r} (expected 'llamacpp' or 'transformers')")
