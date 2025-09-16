#!/bin/bash

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

source .venv/bin/activate
uv run experiments/hescape_pretrain/train.py \
    --config-name=local_config.yaml \
    launcher=local \
    training.lightning.trainer.devices=1 \
    datamodule.batch_size=256 \
    datamodule.num_workers=8
