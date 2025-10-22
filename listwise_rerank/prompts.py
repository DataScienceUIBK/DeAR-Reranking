# prompts.py
import re
from typing import List, Dict

SYSTEM_PROMPT = "You are RankLLM, an assistant that ranks passages by relevance to the query."

def prefix_messages(query: str, num: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"I will provide you with {num} passages, each indicated by number identifier [].\nRank the passages based on their relevance to query: {query}."},
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]

def post_prompt(query: str, num: int) -> str:
    return (
        f"Search Query: {query}.\n"
        f"Rank the {num} passages above based on their relevance to the search query. "
        "The passages should be listed in descending order using identifiers.\n"
        "Please follow the steps below:\n"
        "Step 1. List the information requirements to answer the query.\n"
        "Step 2. For each requirement, find the passages containing that information.\n"
        "Step 3. Rank passages that best cover clear and diverse information. Include all passages.\n"
        "Output format strictly: [2] > [1] > [3] ... Only output the ranking using identifiers; no explanations."
    )

def build_messages(query: str, hits: List[Dict], max_tokens_per_passage: int = 300) -> List[Dict[str, str]]:
    msgs = prefix_messages(query, len(hits))
    for i, hit in enumerate(hits, start=1):
        text = hit["content"].replace("Title: Content: ", "").strip()
        snippet = " ".join(text.split()[:max_tokens_per_passage])
        msgs.append({"role": "user", "content": f"[{i}] {snippet}"})
        msgs.append({"role": "assistant", "content": f"Received passage [{i}]."})
    msgs.append({"role": "user", "content": post_prompt(query, len(hits))})
    return msgs

def extract_after_think(text: str) -> str:
    m = re.search(r"</think>\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

def parse_ranking_to_indices(ranking_text: str, n: int) -> List[int]:
    """
    Convert text like "[3] > [1] > [2]" or "3 1 2" into zero-based indices within [0..n-1].
    Deduplicate and append any missing indices (stable).
    """
    # Keep only digits and spaces
    cleaned = "".join(ch if ch.isdigit() else " " for ch in ranking_text).strip()
    if not cleaned:
        return list(range(n))
    ints = [int(x) - 1 for x in cleaned.split() if x.isdigit()]
    # Keep valid, dedupe preserving order
    seen = set()
    result = []
    for idx in ints:
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            result.append(idx)
    # append any missing
    for idx in range(n):
        if idx not in seen:
            result.append(idx)
    return result
