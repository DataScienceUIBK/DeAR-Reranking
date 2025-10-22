# main.py
import argparse
import json
import os
from tqdm import tqdm

from .datasets import THE_TOPICS
from .clients import create_client
from .pipeline import sliding_windows
from .io_utils import load_rankgpt_json, write_trec, maybe_eval

def parse_args():
    ap = argparse.ArgumentParser("Listwise CoT Reranking")
    ap.add_argument("--input_json", required=True, help="RankGPT-style JSON with hits per query.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_key", required=True, choices=THE_TOPICS.keys())
    ap.add_argument("--client", default="hf", choices=["hf", "openai", "azure", "anthropic"])

    # HF options
    ap.add_argument("--hf_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf_adapter", default=None)
    ap.add_argument("--hf_adapter_strategy", default="auto", choices=["auto", "peft", "adapters"])
    ap.add_argument("--hf_dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--hf_device", default="auto", choices=["auto", "cuda", "cpu"])

    # OpenAI/Azure/Anthropic
    ap.add_argument("--model", default=None, help="OpenAI/Azure/Anthropic model or deployment name")

    # Sliding window
    ap.add_argument("--rank_start", type=int, default=0)
    ap.add_argument("--rank_end", type=int, default=30)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--step", type=int, default=10)

    # Generation
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_passage_tokens", type=int, default=300)

    # Misc
    ap.add_argument("--metrics_file", default="metrics.json")
    ap.add_argument("--skip_eval", action="store_true", help="Skip trec_eval hook")

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build client
    if args.client == "hf":
        agent = create_client(
            "hf",
            model_name_or_path=args.hf_model,
            adapter_path=args.hf_adapter,
            adapter_strategy=args.hf_adapter_strategy,
            torch_dtype=args.hf_dtype,
            device=args.hf_device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.client == "openai":
        agent = create_client("openai", model=args.model, temperature=args.temperature, max_tokens=args.max_new_tokens)
    elif args.client == "azure":
        agent = create_client("azure", model=args.model, temperature=args.temperature, max_tokens=args.max_new_tokens)
    else:  # anthropic
        agent = create_client("anthropic", model=args.model or "claude-3-haiku-20240307",
                              temperature=args.temperature, max_tokens=args.max_new_tokens)

    # Load input
    rank_results = load_rankgpt_json(args.input_json)

    # Rerank
    out_items = []
    for item in tqdm(rank_results, desc="listwise rerank"):
        new_item = sliding_windows(
            item=item,
            rank_start=args.rank_start,
            rank_end=args.rank_end,
            window_size=args.window_size,
            step=args.step,
            agent=agent,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_passage_tokens=args.max_passage_tokens,
        )
        out_items.append(new_item)
        #break

    # Write TREC run and (optional) eval
    trec_path = os.path.join(args.out_dir, os.path.basename(args.input_json) + ".trec")
    write_trec(out_items, trec_path)

    metrics = {}
    if not args.skip_eval:
        ds_topics = THE_TOPICS[args.dataset_key]
        metrics = maybe_eval(ds_topics, trec_path) or {}

    # Save JSON metrics (if any)
    with open(os.path.join(args.out_dir, args.metrics_file), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reranked JSON (same schema as input)
    with open(os.path.join(args.out_dir, "reranked.json"), "w") as f:
        json.dump(out_items, f, indent=2)

if __name__ == "__main__":
    main()
    """
    # Example with local Llama 3.1 8B Instruct + optional adapter
    python -m listwise_rerank.main \
    --input_json ./results/DeAR-P-8B-BC/ranked_dl19.json \
    --out_dir ./out_list_wise/dl19-llama31 \
    --dataset_key dl19 \
    --client hf \
    --hf_model meta-llama/Llama-3.1-8B \
    --hf_adapter ./models/Listwise_Models/Llama-3.1-8B-think-30-epochs-rank32
    """
    """# Example with Azure OpenAI (set env: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT)
python -m listwise_rerank.main \
  --input_json /path/to/ranked_news.json \
  --out_dir ./out/news-azure-gpt4o \
  --dataset_key news \
  --client azure \
  --model gpt-4o-mini \
  --temperature 0.5
    """