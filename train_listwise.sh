#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH --account=
#SBATCH --partition==
#SBATCH --time =
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=train_list_wise

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export BNB_CUDA_VERSION=121


# Torchrun-based LLaMA-Factory training
srun --jobid $SLURM_JOBID bash -c '
python -m torch.distributed.run \
--nproc_per_node=$GPUS_PER_NODE \
--nnodes=$SLURM_NNODES \
--node_rank=$SLURM_PROCID \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
src/train.py \
--stage sft \
--model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
--do_train \
--dataset DeAR-COT \
--template llama3 \
--finetuning_type lora \
--lora_rank 16 \
--output_dir ./output \
--overwrite_cache \
--preprocessing_num_workers 16 \
--dataloader_num_workers 4 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-4 \
--num_train_epochs 50.0 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--logging_steps 10 \
--save_steps 500 \
--plot_loss \
--report_to wandb \
--cutoff_len 5000 \
--bf16 \
'
