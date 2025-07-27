#!/bin/bash

conda activate hescape

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# export MASTER_PORT=12802
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr

# unset SLURM_CPU_BIND

unset SLURM_CPU_BIND
NCCL_DEBUG=INFO
python experiments/hescape_pretrain/train.py --config-name pan_5k_finetune_SIGLIP.yaml launcher=juelich --multirun
