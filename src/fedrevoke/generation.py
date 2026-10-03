"""generation.py -- FedRevoke generator backends (INTERFACES.md Section 5).

Unified protocol::

    class Generator(Protocol):
        def generate(self, prompts: list[str], max_new_tokens: int = 200) -> list[str]: ...

Implementations:
    MockGenerator   deterministic stub, zero network, zero dependencies; for unit tests and --smoke
    HFGenerator     transformers backend (4bit bitsandbytes -> fp16 auto-fallback, batch / OOM degradation)
    GGUFGenerator   optional backend (llama-cpp-python; raises ImportError if not installed)

Lazy-loading convention: no backend downloads/loads a model in __init__; the model is only built
on the first call to generate(), so unit tests do not touch the network or occupy VRAM.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

try:  # allow standalone import when config.py is not ready yet
    from .config import GEN_ID, GEN_ID_SMALL  # type: ignore
except Exception:  # pragma: no cover - fallback constants consistent with INTERFACES.md Section 1
    GEN_ID = "Qwen/Qwen2.5-7B-Instruct"
    GEN_ID_SMALL = "Qwen/Qwen2.5-1.5B-Instruct"

__all__ = [
    "Generator",
    "MockGenerator",
    "HFGenerator",
    "GGUFGenerator",
    "build_generator",
    "GEN_ID",
    "GEN_ID_SMALL",
]

_CONTEXT_MARKERS = ("context:", "context\n")
_QUESTION_MARKERS = ("question:", "q:")


@runtime_checkable
class Generator(Protocol):
    """Generator protocol: takes a batch of prompts, returns an equally long list of texts."""

    def generate(self, prompts: list[str], max_new_tokens: int = 200) -> list[str]:  # pragma: no cover
        ...


def _as_prompt_list(prompts: Any) -> list[str]:
    if prompts is None:
        return []
    if isinstance(prompts, str):
        return [prompts]
    return ["" if p is None else str(p) for p in prompts]


# --------------------------------------------------------------------------- #
# 1) Mock generator
# --------------------------------------------------------------------------- #
class MockGenerator:
    """Deterministic extractive stub generator (network forbidden).

    Behavior: extract the first evidence passage from the prompt's Context section (at most
    max_new_tokens * chars_per_token characters) as the answer, prefixed with the fixed prefix
    keyword. This simulates the leak channel of "a RAG generator copying retrieved evidence",
    so in verify.ElicitationTest:

    * when deleted documents are still retrievable -> the extracted source text hits gold -> high elicit_rate;
    * after deleted documents are fully removed    -> the extracted text is irrelevant -> elicit_rate is about 0.

    Parameters
    ----
    keyword : output prefix (empty string means no prefix).
    chars_per_token : conversion factor from max_new_tokens to a character budget (default 4, approximating English).
    history : records the inputs of each generate call, so unit tests can assert whether prompt
              construction includes the evidence.
    """

    def __init__(self, keyword: str = "MOCK", chars_per_token: int = 4) -> None:
        self.keyword = "" if keyword is None else str(keyword)
        self.chars_per_token = max(1, int(chars_per_token))
        self.history: list[list[str]] = []
        self.n_generated = 0

    # -- Internal ---------------------------------------------------------- #
    @staticmethod
    def _extract_context(prompt: str) -> str:
        """Extract the Context section (up to Question); return the whole prompt when there is no Context."""
        text = prompt or ""
        lowered = text.lower()
        start = 0
        for marker in _CONTEXT_MARKERS:
            pos = lowered.find(marker)
            if pos >= 0:
                start = pos + len(marker)
                break
        segment = text[start:]
        seg_low = segment.lower()
        cut = len(segment)
        for marker in _QUESTION_MARKERS:
            pos = seg_low.find(marker)
            if pos >= 0:
                cut = min(cut, pos)
        segment = segment[:cut]
        return " ".join(segment.split()).strip()

    def generate(self, prompts: list[str], max_new_tokens: int = 200) -> list[str]:
        prompt_list = _as_prompt_list(prompts)
        self.history.append(list(prompt_list))
        try:
            budget = max(1, int(max_new_tokens)) * self.chars_per_token
        except (TypeError, ValueError):
            budget = 200 * self.chars_per_token
        outputs: list[str] = []
        for prompt in prompt_list:
            body = self._extract_context(prompt)[:budget].strip()
            if self.keyword:
                outputs.append((self.keyword + ": " + body).strip() if body else self.keyword)
            else:
                outputs.append(body)
        self.n_generated += len(outputs)
        return outputs

    def info(self) -> dict:
        return {"backend": "mock", "keyword": self.keyword, "n_generated": self.n_generated}


# --------------------------------------------------------------------------- #
# 2) HuggingFace generator
# --------------------------------------------------------------------------- #
class HFGenerator:
    """transformers causal-LM backend.

    Loading strategy (all executed on the first generate; __init__ touches neither network nor VRAM):
      1. device="cuda" but torch.cuda is unavailable -> automatically fall back to cpu (recorded in warnings).
      2. load_in_4bit=True and CUDA available -> BitsAndBytesConfig(nf4) quantized loading;
         bitsandbytes missing / loading error -> automatically fall back to fp16 (fp32 on CPU).
      3. Catch CUDA OOM during generation: first empty_cache and halve max_new_tokens (floor at
         min_new_tokens); if it still fails, split the batch, and only then go item by item. With
         strict=False a failed item returns an empty string and is recorded in errors, so a long
         experiment does not crash for the whole run.

    Known risks (Windows + RTX 5080 Laptop 16G + sm_120, driver 617.14):
      * Official bitsandbytes Windows wheels lag in sm_120 support; 4bit may raise
        RuntimeError/ImportError directly -- already covered by the fp16 fallback; 7B fp16 needs
        more than 15G of VRAM and will OOM on a 16G card, so smoke/ablation should prefer
        GEN_ID_SMALL (1.5B fp16 is about 3.2G).
      * If neither 4bit nor fp16 is available, switch to GGUFGenerator (llama-cpp-python) or
        MockGenerator to get the pipeline through, then return to GPU.
    """

    def __init__(
        self,
        model_id: str = GEN_ID,
        load_in_4bit: bool = True,
        device: str = "cuda",
        max_batch: int = 4,
        default_max_new_tokens: int = 200,
        min_new_tokens: int = 32,
        max_input_tokens: int = 2048,
        use_chat_template: bool | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        strict: bool = False,
    ) -> None:
        self.model_id = str(model_id)
        self.load_in_4bit = bool(load_in_4bit)
        self.device = str(device)
        self.max_batch = max(1, int(max_batch))
        self.default_max_new_tokens = max(1, int(default_max_new_tokens))
        self.min_new_tokens = max(1, int(min_new_tokens))
        self.max_input_tokens = max(64, int(max_input_tokens))
        self.use_chat_template = use_chat_template
        self.torch_dtype = torch_dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.strict = bool(strict)

        self.backend: str | None = None       # "bnb-4bit" / "fp16" / "fp32"
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.oom_events = 0
        self.n_generated = 0
        self.effective_max_new_tokens = self.default_max_new_tokens

        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    # -- Loading ----------------------------------------------------------- #
    @staticmethod
    def _bitsandbytes_available() -> bool:
        try:
            return importlib.util.find_spec("bitsandbytes") is not None
        except Exception:
            return False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.warnings.append("CUDA unavailable; HFGenerator falls back to CPU (fp32)")
            self.device = "cpu"
        use_cuda = self.device.startswith("cuda")

        if self.torch_dtype is not None:
            dtype = getattr(torch, str(self.torch_dtype).replace("torch.", ""), torch.float16)
        else:
            dtype = torch.float16 if use_cuda else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "padding_side", "right") != "left":
            tokenizer.padding_side = "left"  # generative batches need left padding

        model = None
        want_4bit = self.load_in_4bit and use_cuda
        if want_4bit and not self._bitsandbytes_available():
            self.warnings.append("bitsandbytes not installed; skipping 4bit quantization and falling back to fp16/fp32")
            want_4bit = False
        if want_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quant_cfg,
                    device_map={"": 0},
                    trust_remote_code=self.trust_remote_code,
                )
                self.backend = "bnb-4bit"
            except Exception as exc:  # missing sm_120 wheel / driver mismatch, etc.
                self.warnings.append(
                    "4bit loading failed, automatically falling back to fp16: {0}: {1}".format(type(exc).__name__, exc)
                )
                model = None

        if model is None:
            model = self._load_dense(AutoModelForCausalLM, dtype)
            self.backend = "fp16" if dtype == torch.float16 else "fp32"
            try:
                model.to(self.device)
            except Exception as exc:
                self.warnings.append("model.to({0}) failed, switching to device_map=auto: {1}".format(self.device, exc))
                model = self._load_dense(AutoModelForCausalLM, dtype, device_map="auto")
        model.eval()

        if self.use_chat_template is None:
            self.use_chat_template = getattr(tokenizer, "chat_template", None) is not None
        self._model = model
        self._tokenizer = tokenizer

    def _load_dense(self, auto_cls: Any, dtype: Any, **extra: Any) -> Any:
        """Load dense weights by dtype: transformers>=5 uses dtype=, 4.x uses torch_dtype=."""
        last_exc: BaseException | None = None
        for dtype_kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
            kwargs = dict(extra)
            kwargs.update(dtype_kw)
            try:
                return auto_cls.from_pretrained(
                    self.model_id, trust_remote_code=self.trust_remote_code, **kwargs
                )
            except Exception as exc:  # version differences / parameter name not accepted
                last_exc = exc
        raise last_exc if last_exc is not None else RuntimeError("model loading failed")

    # -- Generation -------------------------------------------------------- #
    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        try:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    def _is_oom(self, exc: BaseException) -> bool:
        name = type(exc).__name__
        text = str(exc).lower()
        return ("outofmemory" in name.lower()) or ("out of memory" in text) or ("cuda error" in text and "memory" in text)

    def _run_batch(self, texts: Sequence[str], budget: int) -> list[str]:
        torch = self._torch
        tok = self._tokenizer
        model = self._model
        enc = tok(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=int(budget),
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
        prompt_len = int(enc["input_ids"].shape[1])
        generated = out[:, prompt_len:]
        return [tok.decode(row, skip_special_tokens=True).strip() for row in generated]

    def _generate_chunk(self, texts: list[str], budget: int) -> list[str]:
        try:
            return self._run_batch(texts, budget)
        except Exception as exc:
            if self._is_oom(exc):
                self.oom_events += 1
                try:
                    self._torch.cuda.empty_cache()
                except Exception:
                    pass
                new_budget = max(self.min_new_tokens, int(budget) // 2)
                self.effective_max_new_tokens = min(self.effective_max_new_tokens, new_budget)
                if new_budget != budget:
                    try:
                        return self._run_batch(texts, new_budget)
                    except Exception as exc2:
                        exc = exc2
                if len(texts) > 1:
                    self.max_batch = max(1, min(self.max_batch, len(texts) // 2 or 1))
                    mid = max(1, len(texts) // 2)
                    return self._generate_chunk(texts[:mid], new_budget) + self._generate_chunk(
                        texts[mid:], new_budget
                    )
            if self.strict:
                raise
            self.errors.append("{0}: {1}".format(type(exc).__name__, exc))
            warnings.warn("HFGenerator generation failed (strict=False, returning empty string): {0}".format(exc))
            if len(texts) > 1:
                mid = max(1, len(texts) // 2)
                return self._generate_chunk(texts[:mid], budget) + self._generate_chunk(texts[mid:], budget)
            return [""]

    def generate(self, prompts: list[str], max_new_tokens: int = 200) -> list[str]:
        prompt_list = _as_prompt_list(prompts)
        if not prompt_list:
            return []
        self._ensure_model()
        try:
            budget = max(1, int(max_new_tokens))
        except (TypeError, ValueError):
            budget = self.default_max_new_tokens
        self.effective_max_new_tokens = budget
        formatted = [self._format_prompt(p) for p in prompt_list]
        outputs: list[str] = []
        for start in range(0, len(formatted), self.max_batch):
            chunk = formatted[start : start + self.max_batch]
            outputs.extend(self._generate_chunk(chunk, budget))
        self.n_generated += len(outputs)
        return outputs

    # -- Resources and info ------------------------------------------------ #
    def unload(self) -> None:
        """Release the model and VRAM (so backends can be swapped for ablation within one process)."""
        self._model = None
        self._tokenizer = None
        if self._torch is not None:
            try:
                import gc

                gc.collect()
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    def info(self) -> dict:
        peak_mb = 0.0
        if self._torch is not None:
            try:
                peak_mb = float(self._torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
            except Exception:
                peak_mb = 0.0
        return {
            "backend": self.backend or "unloaded",
            "model_id": self.model_id,
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
            "max_batch": self.max_batch,
            "use_chat_template": bool(self.use_chat_template),
            "n_generated": self.n_generated,
            "oom_events": self.oom_events,
            "effective_max_new_tokens": self.effective_max_new_tokens,
            "peak_vram_mb": peak_mb,
            "warnings": list(self.warnings),
            "errors": list(self.errors[:10]),
        }


# --------------------------------------------------------------------------- #
# 3) GGUF generator (optional backend)
# --------------------------------------------------------------------------- #
class GGUFGenerator:
    """llama-cpp-python backend (CPU/GPU hybrid, as a fallback when 4bit fails).

    Depends on llama-cpp-python, for which this machine (Windows + Python 3.14) usually has no
    usable wheel. If it is not installed, the constructor raises ImportError directly, with
    alternative-path guidance (do not pip install inside this project; the environment is
    managed by the project environment setup).
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        max_batch: int = 4,
        chat_format: str | None = None,
        verbose: bool = False,
        strict: bool = False,
    ) -> None:
        if importlib.util.find_spec("llama_cpp") is None:
            raise ImportError(
                "GGUFGenerator requires llama-cpp-python, which is not installed in the current environment. "
                "Alternatives: (a) use an fp16 small model with HFGenerator(load_in_4bit=False); "
                "(b) use MockGenerator to run the pipeline; "
                "(c) install llama-cpp-python in .venv (requires a build toolchain)."
            )
        self.repo_id = str(repo_id)
        self.filename = str(filename)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.max_batch = max(1, int(max_batch))
        self.chat_format = chat_format
        self.verbose = bool(verbose)
        self.strict = bool(strict)
        self.backend = "gguf"
        self.errors: list[str] = []
        self.n_generated = 0
        self._llm: Any = None

    def _resolve_model_path(self) -> str:
        local = Path(self.repo_id)
        if local.is_dir():
            candidate = local / self.filename
            if candidate.exists():
                return str(candidate)
            raise FileNotFoundError("{1} not found in local directory {0}".format(local, self.filename))
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ImportError("huggingface_hub is missing; cannot download GGUF weights: {0}".format(exc)) from exc
        return hf_hub_download(repo_id=self.repo_id, filename=self.filename)

    def _ensure_model(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama  # already verified importable in __init__

        model_path = self._resolve_model_path()
        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": self.verbose,
        }
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format
        self._llm = Llama(**kwargs)

    def generate(self, prompts: list[str], max_new_tokens: int = 200) -> list[str]:
        prompt_list = _as_prompt_list(prompts)
        if not prompt_list:
            return []
        self._ensure_model()
        outputs: list[str] = []
        for prompt in prompt_list:
            try:
                result = self._llm(
                    prompt,
                    max_tokens=int(max_new_tokens),
                    temperature=0.0,
                    echo=False,
                )
                text = ""
                choices = result.get("choices") if isinstance(result, dict) else None
                if choices:
                    text = str(choices[0].get("text", ""))
                outputs.append(text.strip())
            except Exception as exc:
                if self.strict:
                    raise
                self.errors.append("{0}: {1}".format(type(exc).__name__, exc))
                outputs.append("")
        self.n_generated += len(outputs)
        return outputs

    def info(self) -> dict:
        return {
            "backend": "gguf",
            "repo_id": self.repo_id,
            "filename": self.filename,
            "n_gpu_layers": self.n_gpu_layers,
            "n_generated": self.n_generated,
            "errors": list(self.errors[:10]),
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_generator(kind: str = "mock", **kwargs: Any) -> Generator:
    """Construct a generator by name: mock / hf / gguf (case and underscore insensitive)."""
    key = str(kind or "mock").strip().lower().replace("-", "_")
    if key in ("mock", "mock_generator", "dummy", "smoke"):
        return MockGenerator(**kwargs)
    if key in ("hf", "hf_generator", "transformers", "hf4bit"):
        return HFGenerator(**kwargs)
    if key in ("gguf", "gguf_generator", "llama_cpp", "llamacpp"):
        return GGUFGenerator(**kwargs)
    raise ValueError("unknown generator type: {0!r} (options: mock / hf / gguf)".format(kind))
