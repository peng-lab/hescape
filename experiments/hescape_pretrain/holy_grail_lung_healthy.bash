#!/bin/bash

# conda activate hescape

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1

export MASTER_PORT=12802
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr

# unset SLURM_CPU_BIND
NCCL_DEBUG=INFO
# python experiments/hescape_pretrain/train.py --config-name holy_grail_lung_healthy.yaml launcher=juelich --multirun

uv run experiments/hescape_pretrain/train.py --config-name holy_grail_lung_healthy.yaml model.litmodule.img_enc_name=h0-mini model.litmodule.img_finetune=false model.litmodule.gene_enc_name=drvi
