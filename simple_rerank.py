import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

DTYPE = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def load_ranker(model_or_adapter_path: str, torch_dtype: str = "auto", device: str = "auto"):
    """
    Pass EITHER:
      - a PEFT LoRA adapter dir / HF repo id  (contains adapter_config.json)
      - OR a merged/original model dir / HF repo id (normal Transformers model)

    Returns: (tokenizer, model, device_str)
    """
    dtype = DTYPE[torch_dtype]

    # Try to interpret as a PEFT adapter first
    is_peft = False
    peft_cfg = None
    try:
        peft_cfg = PeftConfig.from_pretrained(model_or_adapter_path)
        is_peft = True
    except Exception:
        is_peft = False

    if is_peft:
        # ----- LoRA adapter path -----
        base_id = peft_cfg.base_model_name_or_path
        tok = AutoTokenizer.from_pretrained(base_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"

        base = AutoModelForSequenceClassification.from_pretrained(
            base_id, num_labels=1, torch_dtype=dtype
        )
        model = PeftModel.from_pretrained(base, model_or_adapter_path)
        model = model.merge_and_unload()
    else:
        # ----- Merged / original model path -----
        tok = AutoTokenizer.from_pretrained(model_or_adapter_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"

        # Note: for merged/original, we DO NOT force num_labels=1 — we load as saved.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_or_adapter_path, torch_dtype=dtype
        )

    model.eval()

    # Make batching safe for LLaMA-style models
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    if getattr(model.config, "bos_token_id", None) is None and tok.bos_token_id is not None:
        model.config.bos_token_id = tok.bos_token_id
    if getattr(model.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
        model.config.eos_token_id = tok.eos_token_id

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tok, model, device


@torch.inference_mode()
def score_pair(tokenizer, model, device, query: str, passage: str, title: str = "",
               q_max_len: int = 32, p_max_len: int = 196) -> float:
    inputs = tokenizer(
        f"query: {query}",
        f"document: {title} {passage}",
        return_tensors="pt",
        truncation=True,
        max_length=q_max_len + p_max_len,
        padding="max_length",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    return float(logits.squeeze().item())


@torch.inference_mode()
def rerank(tokenizer, model, device, query: str, docs: List[Tuple[str, str]],
           q_max_len: int = 32, p_max_len: int = 196, batch_size: int = 64):
    """
    docs: list of (title, passage)
    Returns: list of (index, score) sorted by score desc (index is original docs index).
    """
    scores = []
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        q_texts = [f"query: {query}"] * len(chunk)
        d_texts = [f"document: {t} {p}" for t, p in chunk]
        inputs = tokenizer(
            q_texts, d_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=q_max_len + p_max_len,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits.squeeze(-1)  # (B,)
        scores.extend(logits.detach().cpu().tolist())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked  # [(doc_idx, score), ...]


# ------------------ demo ------------------
# 1) Load either a PEFT adapter OR a merged model
# peft_or_model_path = ""      # PEFT adapter dir
# peft_or_model_path = ""               # merged/original model dir
peft_or_model_path = "abdoelsayed/dear-8b-reranker-ce-v1"
tokenizer, model, device = load_ranker(peft_or_model_path, torch_dtype="bfloat16")

# 2) Single pair
q = "What is llama?"
title = "Llama"
passage = "The llama is a domesticated South American camelid..."
print(score_pair(tokenizer, model, device, q, passage, title))

# 3) Rerank a list for one query
q = "When did Thomas Edison invent the light bulb?"
docs = [
    ("", "Lightning strike at Seoul National University"), #0
    ("", "Thomas Edison tried to invent a device for car but failed"),#1
    ("", "Coffee is good for diet"),#2
    ("", "KEPCO fixes light problems"),#3
    ("", "Thomas Edison invented the light bulb in 1879"),#4
]
ranking = rerank(tokenizer, model, device, q, docs)
print(ranking)
#DeAR-P-8B-BC
#[(4, -2.015625), (1, -5.6875), (2, -6.375), (0, -6.5), (3, -6.78125)]

#DeAR-P-8B-RL
#[(4, -2.984375), (1, -6.375), (3, -7.4375), (0, -7.75), (2, -8.125)]

#DeAR-P-3B-BC
# [(4, -6.0625), (1, -10.1875), (0, -11.125), (3, -11.625), (2, -12.0625)]

#DeAR-P-3B-RL
# [(4, -1.3046875), (1, -5.125), (3, -6.3125), (0, -6.4375), (2, -6.96875)]