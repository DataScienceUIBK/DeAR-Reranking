# clients.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import os
import time
import torch

# ---- Base interface ---------------------------------------------------------

class BaseLLMClient:
    def chat(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        raise NotImplementedError

# ---- OpenAI / Azure OpenAI -------------------------------------------------

@dataclass
class OpenAIConfig:
    # For OpenAI: set OPENAI_API_KEY
    # For Azure: set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
    provider: str = "openai"  # "openai" | "azure"
    model: Optional[str] = None          # e.g., "gpt-4o-mini"
    temperature: float = 0.5
    max_tokens: int = 1024
    request_timeout: int = 60

class OpenAIClient(BaseLLMClient):
    def __init__(self, cfg: OpenAIConfig):
        self.cfg = cfg
        if cfg.provider == "azure":
            from openai import AzureOpenAI
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
            if not endpoint or not api_key:
                raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
            self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
            self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or cfg.model
            if not self.deployment:
                raise ValueError("Set AZURE_OPENAI_DEPLOYMENT or pass cfg.model for Azure.")
        else:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Set OPENAI_API_KEY.")
            self.client = OpenAI(api_key=api_key)
            if not cfg.model:
                raise ValueError("Pass cfg.model for OpenAI.")

    def chat(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        temperature = gen_kwargs.get("temperature", self.cfg.temperature)
        max_tokens = gen_kwargs.get("max_new_tokens", self.cfg.max_tokens)
        timeout = gen_kwargs.get("request_timeout", self.cfg.request_timeout)

        if isinstance(getattr(self.client, "azure_endpoint", None), str):
            # Azure
            resp = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return resp.choices[0].message.content
        else:
            # OpenAI
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return resp.choices[0].message.content

# ---- Anthropic (Claude) ----------------------------------------------------

@dataclass
class AnthropicConfig:
    model: str = "claude-3-haiku-20240307"
    max_tokens: int = 1024
    temperature: float = 0.5

class AnthropicClient(BaseLLMClient):
    def __init__(self, cfg: AnthropicConfig):
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY.")
        self.client = Anthropic(api_key=api_key)
        self.cfg = cfg

    def chat(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        # convert OpenAI-style messages to Anthropic's
        sys_prompt = " ".join(m["content"] for m in messages if m["role"] == "system").strip() or None
        chat_messages = [m for m in messages if m["role"] in ("user", "assistant")]
        temperature = gen_kwargs.get("temperature", self.cfg.temperature)
        max_tokens = gen_kwargs.get("max_new_tokens", self.cfg.max_tokens)
        resp = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=sys_prompt,
            messages=chat_messages,
        )
        return "".join(block.text for block in resp.content if getattr(block, "text", None))

# ---- Hugging Face local CausalLM ------------------------------------------

@dataclass
class HFConfig:
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    adapter_path: Optional[str] = None   # PEFT LoRA or AdapterHub path
    adapter_strategy: str = "auto"       # "auto" | "peft" | "adapters"
    torch_dtype: str = "bfloat16"        # "auto"|"float16"|"bfloat16"|"float32"
    device: str = "auto"                 # "auto"|"cuda"|"cpu"
    temperature: float = 0.5
    max_new_tokens: int = 1024

class HFClient(BaseLLMClient):
    def __init__(self, cfg: HFConfig):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.cfg = cfg
        dtype_map = {"auto": None, "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(cfg.torch_dtype, None)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=dtype,
            device_map={"": "cuda:0"} if (cfg.device == "auto" and torch.cuda.is_available()) else None,
        )

        # Adapters (PEFT or AdapterHub)
        if cfg.adapter_path:
            loaded = False
            
            if cfg.adapter_strategy in ("auto", "peft"):
                #try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)
                loaded = True
                print("adapters load successfully")
                # except Exception:
                #     if cfg.adapter_strategy == "peft":
                #         raise
            if not loaded and cfg.adapter_strategy in ("auto", "adapters"):
                #try:
                    # requires adapter-transformers
                self.model.load_adapter(cfg.adapter_path)
                loaded = True
                # except Exception:
                #     if cfg.adapter_strategy == "adapters":
                #         raise
            if not loaded:
                raise RuntimeError("Could not load adapter: tried PEFT and AdapterHub paths.")

        self.use_chat_template = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template

        # padding safety for batching inside generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        if cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available()):
            self.model.to("cuda")

    def _render(self, messages: List[Dict[str, str]]) -> str:
        if self.use_chat_template:
            try:
                return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        # fallback manual
        s = []
        for m in messages:
            role = m["role"]
            if role == "system":
                s.append(f"[SYSTEM] {m['content']}")
            elif role == "user":
                s.append(f"[USER] {m['content']}")
            elif role == "assistant":
                s.append(f"[ASSISTANT] {m['content']}")
        return "\n".join(s)

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        prompt = self._render(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        temperature = gen_kwargs.get("temperature", self.cfg.temperature)
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.cfg.max_new_tokens)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ---- Factory ---------------------------------------------------------------

def create_client(kind: str, **kwargs) -> BaseLLMClient:
    """
    kind: "openai", "azure", "anthropic", "hf"
    """
    if kind == "openai":
        return OpenAIClient(OpenAIConfig(provider="openai", **kwargs))
    if kind == "azure":
        return OpenAIClient(OpenAIConfig(provider="azure", **kwargs))
    if kind == "anthropic":
        return AnthropicClient(AnthropicConfig(**kwargs))
    if kind == "hf":
        return HFClient(HFConfig(**kwargs))
    raise ValueError(f"Unknown client kind: {kind}")
