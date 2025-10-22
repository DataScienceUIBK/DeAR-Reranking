#!/bin/bash

# Folder containing your bm25 JSONL files
BM25_DIR="./data/bm25_beir_dl19_20"
BM25_DIR_PREDICT="./results/dear-8b-reranker-ce-lora-v1"

# Log file to store evaluation results
LOG_FILE="$BM25_DIR_PREDICT/evaluation_results.log"

# Path to your reranker model and tokenizer
RERANKER_CHECKPOINT="abdoelsayed/dear-8b-reranker-ce-lora-v1"


TOKENIZER="meta-llama/Llama-3.1-8B"
TEACHER_MODEL="abdoelsayed/llama2-13b-rankllama-teacher"
# Mapping datasets to their qrels
declare -A QRELS=(
    [dl19]="dl19-passage"
    [dl20]="dl20-passage"
    [covid]="beir-v1.0.0-trec-covid-test"
    [nfc]="beir-v1.0.0-nfcorpus-test"
    #[arguana]="beir-v1.0.0-arguana-test"
    [touche]="beir-v1.0.0-webis-touche2020-test"
    [dbpedia]="beir-v1.0.0-dbpedia-entity-test"
    [scifact]="beir-v1.0.0-scifact-test"
    [signal]="beir-v1.0.0-signal1m-test"
    [news]="beir-v1.0.0-trec-news-test"
    #[fiqa]="beir-v1.0.0-fiqa-test"
    #[scidocs]="beir-v1.0.0-scidocs-test"
    #[quora]="beir-v1.0.0-quora-test"
    #[fever]="beir-v1.0.0-fever-test"
    [robust04]="beir-v1.0.0-robust04-test"
)

# Create prediction directory if it doesn't exist
mkdir -p "$BM25_DIR_PREDICT"

# Clean previous log file
echo "Evaluation Results" > $LOG_FILE

# Loop through all bm25 JSONL files in the directory
for jsonl_file in $BM25_DIR/*.jsonl; do

    dataset=$(basename "$jsonl_file" .jsonl)
    echo "Processing $dataset..."

    if [ ! -f "$BM25_DIR_PREDICT/${dataset}.txt" ]; then
        # Run reranker inference
        CUDA_VISIBLE_DEVICES=0 python pointwise_reranker/inference.py \
            --output_dir=temp \
            --model_name_or_path $RERANKER_CHECKPOINT \
            --tokenizer_name $TOKENIZER \
            --encode_in_path $jsonl_file \
            --fp16 \
            --teacher_model_name_or_path $TEACHER_MODEL \
            --per_device_eval_batch_size 64 \
            --temperature 2 \
            --alpha 0.1 \
            --q_max_len 32 \
            --p_max_len 196 \
            --dataset_name json \
            --encoded_save_path "$BM25_DIR_PREDICT/${dataset}.txt"
    else
        echo "Inference file exists for $dataset, skipping inference."
    fi

    # Convert results to TREC format
    python pointwise_reranker/convert_result_to_trec.py \
        --input "$BM25_DIR_PREDICT/${dataset}.txt" \
        --output "$BM25_DIR_PREDICT/ranked_${dataset}.trec"
    echo "[${dataset}] Converting TREC to JSON..."
    python pointwise_reranker/convert_trec_to_json.py --path "$BM25_DIR_PREDICT" --out "$BM25_DIR_PREDICT"
    echo "[${dataset}] JSON saved (e.g., $BM25_DIR_PREDICT/ranked_${dataset}.json)"
    # Run evaluation using correct qrels
    eval_result=$(python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 ${QRELS[$dataset]} "$BM25_DIR_PREDICT/ranked_${dataset}.trec")

    # Save result to log
    echo "Dataset: $dataset" >> $LOG_FILE
    echo "$eval_result" >> $LOG_FILE
    echo "---------------------------" >> $LOG_FILE

    echo "$dataset done."
done

# Final log output
echo "All evaluations completed. Check $LOG_FILE for details."
