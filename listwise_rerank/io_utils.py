# io_utils.py
from __future__ import annotations
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from .trec_eval import EvalFunction

def load_rankgpt_json(path: str, selected_qids: Optional[set] = None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for entry in tqdm(data, desc=f"load {path}"):
        if selected_qids is None or entry.get("qid") in selected_qids:
            out.append({"query": entry["query"], "hits": entry["hits"], "qid": entry.get("qid")})
    return out

def write_trec(results, out_path: str) -> None:
    with open(out_path, "w") as f:
        for item in results:
            hits = item["hits"]
            N = len(hits)
            for rank, hit in enumerate(hits, start=1):
                # monotonic score consistent with reranked order
                score = float(N - rank + 1)
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {score:.6f} rerank\n")


def maybe_eval(topics_name: str, trec_path: str):
    """
    Calls your existing trec_eval python wrapper if available:
    from trec_eval import EvalFunction
    """
    
    EvalFunction.write_file  # attribute check
    res = EvalFunction.main(topics_name, trec_path)
    return res
    
