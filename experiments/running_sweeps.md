# Running Sweeps and Experiment Structure

## Overview

Experimental configs are organized within the `experiments/` directory, which follows a modular layout. The experiments folder is structured as follow:
```
experiments
├── configs                  # main folder for experimental configuration files
│   ├── datamodule/          # dataset configs for each panel
│   ├── launcher/            # cluster & local launcher configs (slurm, local)
│   ├── model/               # default model configs
│   ├── paths/               # dataset and pretrained weights paths
│   ├── training/            # default training configs
│   ├── core_config_*.yaml   # (development only)
│   ├── holy_grail_*.yaml    # benchmark sweep configs to recreate results
│   └── local_config.yaml    # default config for local runs
│
├── drvi_pretrain/           # DRVI-specific pretraining files
│   ├── train.py
│   ├── inference.py
│   ├── drvi_train.bash
│   └── drvi_results.ipynb
│
├── hescape_pretrain/        # pretraining related files
│   ├── train.py             # training script
│   ├── local.bash           # launcher for local runs
│   └── holy_grail_*.bash    # benchmark sweep launchers for each dataset
│
├── yaml_configs/            # (development only)
│   └── ...                  
│
└── running_sweeps.md        
```

## Hydra-based Experiment Management

HESCAPE uses [Hydra](https://hydra.cc) for configuration management and sweep orchestration.  
Each experiment is defined by a YAML configuration in `experiments/configs/` and launched via a `.bash` script in `experiments/hescape_pretrain/`.

Hydra combines multiple configuration files (for the model, data, training, and environment) into a single experiment specification. You can override parameters directly from the command line or define multiple values for a parameter to perform *grid sweeps*.

Example `configs/holy_grail_lung_healthy.yaml`:

```yaml
hydra:
  sweeper:
    params:
      model.litmodule.img_enc_name: h0-mini, uni
      model.litmodule.gene_enc_name: drvi, nicheformer
      model.litmodule.loss: CLIP, SIGLIP
      datamodule.batch_size: 64, 256
```
Running this config will automatically expand into all combinations of the parameter values, launching one job per configuration (in this example: 2×2×2×2 = 16 runs).

### Launching Sweeps

Sweeps can be launched either locally or on an HPC cluster via Slurm (recommended for large-scale benchmarks).

#### Local Example

For quick iteration or smoke testing on a single GPU:
```bash
uv run experiments/hescape_pretrain/train.py \
  --config-name=local_config.yaml \
  launcher=local \
  training.lightning.trainer.devices=1 \
  datamodule.batch_size=8 \
  training.lightning.trainer.max_steps=200
```

This command runs a single experiment (no sweeping) using the local launcher. Alternatively, `experiments/hescape_pretrain/local.bash` can be used to launch local runs.

#### Slurm Sweep Example (Benchmark Reproduction)

Each `holy_grail_*.bash` file launches the corresponding benchmark sweep on an HPC environment configured via the appropriate launcher YAML (e.g. configs/launcher/juelich.yaml).

Example `experiments/hescape_pretrain/holy_grail_lung_healthy.bash` 
```bash 
#!/bin/bash
source .venv/bin/activate

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset SLURM_CPU_BIND
NCCL_DEBUG=INFO

uv run experiments/hescape_pretrain/train.py \
  --config-name=holy_grail_lung_healthy.yaml \
  launcher=juelich \
  --multirun
```
Key points:
- `--multirun` triggers a Hydra sweep
- `launcher=juelich` loads the Slurm configuration from `configs/launcher/juelich.yaml`
- The sweep expands all parameter combinations under `hydra.sweeper.params` in the config file
- Logs, checkpoints, and metrics are stored automatically under `experiments/hescape_pretrain/holy_grail_lung_healthy/`

### Customizing a Sweep
To create your own sweep:

1. Copy an existing benchmark config
2. Edit the `hydra.sweeper.params` section to define the grid

3. Launch using a custom launcher file or via CLI with:
```bash
uv run experiments/hescape_pretrain/train.py --config-name=my_experiment.yaml --multirun
````
