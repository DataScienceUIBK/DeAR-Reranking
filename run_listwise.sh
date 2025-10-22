#!/usr/bin/env bash
set -euo pipefail


# common args
HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"   # (chat/instruct tends to parse better)
HF_ADAPTER="abdoelsayed/dear-8b-reranker-listwise-lora-v1"
IN_DIR="./results/dear-8b-reranker-ce-lora-v1"
OUT_DIR="./results/dear-8b-reranker-ce-lora-v1"

# the datasets you looped over
DATASETS=(covid) #dl19 dl20 news nfc touche scifact signal robust04 dbpedia

mkdir -p "${OUT_DIR}"

for ds in "${DATASETS[@]}"; do
  in_json="${IN_DIR}/ranked_${ds}.json"
  out_path="${OUT_DIR}/${ds}-dear8b"
  echo "=== Running ${ds} ==="
  python -m listwise_rerank.main \
    --input_json "${in_json}" \
    --out_dir "${out_path}" \
    --dataset_key "${ds}" \
    --client hf \
    --hf_model "${HF_MODEL}" \
    --hf_adapter "${HF_ADAPTER}" \
    --hf_adapter_strategy auto \
    --hf_dtype bfloat16 \
    --hf_device auto \
    --rank_start 0 \
    --rank_end 30 \
    --window_size 20 \
    --step 10 \
    --temperature 0.5 \
    --max_new_tokens 1024 \
    --max_passage_tokens 300
done
