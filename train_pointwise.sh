#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --partition=
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --cpus-per-task=
#SBATCH --gpus-per-node=
#SBATCH --time=
#SBATCH --output=.out
#SBATCH --error=.err

module load anaconda3/2023.09-0
module load cuda/12.3
source activate retreiver

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

deepspeed --include localhost:0,1,2,3 reranker_train.py \
  --deepspeed ./config/ds_config.json \
  --output_dir ./models/test_training \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --teacher_model_name_or_path abdoelsayed/llama2-13b-rankllama-teacher \
  --temperature 2 \
  --alpha 0.1 \
  --save_steps 200 \
  --dataset_name Tevatron/msmarco-passage \
  --bf16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --train_n_passages 2 \
  --learning_rate 1e-4 \
  --q_max_len 32 \
  --p_max_len 196 \
  --num_train_epochs 2 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --dataset_proc_num 32
