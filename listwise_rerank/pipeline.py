# pipeline.py
from __future__ import annotations
import copy
from typing import Dict, List

from .clients import BaseLLMClient
from .prompts import build_messages, extract_after_think, parse_ranking_to_indices

def apply_permutation(item, order, start, end):
    slice_hits = copy.deepcopy(item["hits"][start:end])
    order = [i for i in order if 0 <= i < len(slice_hits)]
    missing = [i for i in range(len(slice_hits)) if i not in order]
    order += missing

    N = len(slice_hits)
    for j, idx in enumerate(order):
        new_hit = copy.deepcopy(slice_hits[idx])
        new_hit["score"] = float(N - j)   # make score reflect new order
        item["hits"][start + j] = new_hit
    return item



def permutation_once(item: Dict, start: int, end: int, agent: BaseLLMClient, max_passage_tokens: int = 300, **gen_kwargs) -> Dict:
    query = item["query"]
    hits = item["hits"][start:end]
    messages = build_messages(query, hits, max_tokens_per_passage=max_passage_tokens)
    raw = agent.chat(messages, **gen_kwargs)
    print(raw)
    text = extract_after_think(raw)
    order = parse_ranking_to_indices(text, n=len(hits))
    return apply_permutation(item, order, start, end)

def sliding_windows(item: Dict, rank_start: int, rank_end: int, window_size: int, step: int,
                    agent: BaseLLMClient, **gen_kwargs) -> Dict:
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_once(item, start_pos, end_pos, agent, **gen_kwargs)
        end_pos -= step
        start_pos -= step
    return item
