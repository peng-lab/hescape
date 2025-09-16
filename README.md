# HESCAPE: A Large-Scale Benchmark for Cross-Modal Learning in Spatial Transcriptomics
## Multimodal Contrastive Pretraining for Spatial Transcriptomics and Histology

[ [arXiv](https://arxiv.org/abs/2508.01490) | [Blog](https://www.linkedin.com/pulse/hescape-benchmark-visiongenomics-alignment-spatial-rushin-gindra-rfrsf/?trackingId=FQFxbUmNyOlFcCLuvwqPcA%3D%3D) | [Data](https://huggingface.co/datasets/Peng-AI/hescape-pyarrow) | [Cite](https://github.com/peng-lab/hescape?tab=readme-ov-file#citation) \]

**HESCAPE** s a large-scale, pan-organ benchmark for cross-modal contrastive pretraining in spatial transcriptomics (6 gene panels, 54 donors). We evaluate state-of-the-art image and gene encoders across multiple pretraining strategies on two downstream tasks: gene-mutation classification and gene-expression prediction. We find alignment is driven primarily by the gene encoder, with spatially pretrained gene models outperforming non-spatial and simple baselines. Paradoxically, contrastive pretraining improves mutation classification but degrades expression prediction, likely due to batch effects. HESCAPE provides standardized datasets, evaluation protocols, and tools to advance batch-robust multimodal learning.

<img src="figures/schematic.png" alt="HESCAPE framework" width="800" />

<!-- ## What does HESCAPE offer?
🤗 **Huggingface multi-modal datase**t: We release a histology-transcriptomics 1-1 mapped streamable pyarrow datasets for 6 independent 10x Xenium gene panels respectively.

💡 **Pretraining at scale**: We provide a framework that can systematically evaluate 4 gene expression encoders (DRVI, Nicheformer, scFoundation, MLP) and 5 pathology foundation models (Gigapath, UNI, CONCH, H0-mini, CtransPath) across CLIP-style contrastive objectives. We enable users to evaluate their own custom pathology or gene models in a multi-modal CLIP setting.

🔁 **Cross-modal retrieval tasks**: We test how well image patches retrieve matching gene vectors (I2G) and vice versa (G2I), with Recall@5 scores revealing insights into encoder generalizability and dataset-specific tuning.

🧬 **Enables downstream tasks**: We provide tools for users to use HESCAPE trained models for inference in downstream tasks like classification of clinically relevant mutations like MSI, BRAF, KRAS, and Gene expression prediction from Histology. -->

## TL;DR Quickstart (3 steps)
1. **Install (uv):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    git clone https://github.com/peng-lab/hescape.git
    cd hescape
    uv sync
    ```
2. **Load a small sample dataset**
    ```python
    from datasets import load_dataset
    ds = load_dataset("Peng-AI/hescape-pyarrow", name="human-lung-healthy-panel", split="train", num_proc=4)
    print(ds)  # peek at the attributes
    ```
3. **Run a 60-second smoke test training (single GPU, local)**
    ```bash
    uv run experiments/hescape_pretrain/train.py \
    --config-name=local_config.yaml \
    launcher=local \
    training.lightning.trainer.max_steps=200 \
    training.lightning.trainer.devices=1 \
    datamodule.batch_size=8 \
    datamodule.num_workers=4
    ```
> [!NOTE]
> The `launcher=local` parameter is used to run the training locally. This can be useful for debugging or running experiments on a local machine with 1+ gpu. For distributed training on HPC with Slurm, take a look at [running_sweeps.md](https://github.com/peng-lab/hescape/blob/main/experiments/running_sweeps.md).

## HESCAPE installation
Supported: uv (recommended), Conda, pip(PyPI):

- ### uv
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh

    git clone https://github.com/peng-lab/hescape.git
    cd hescape
    uv sync

    # contributions are welcome!
    uv pip install -e ".[dev]"
    ```
- ### conda
    ```bash
    conda create -n "hescape" python=3.11
    conda activate hescape
    git clone https://github.com/peng-lab/hescape.git
    cd hescape
    pip install -e .
    ```

## Using the Dataset
We provide 5 independent datasets to use with HESCAPE, each based on a specific 10x Xenium gene panel. These datasets are loaded as follows:
```python
from datasets import load_dataset

# Example: load the human breast panel
ds = load_dataset(
    "Peng-AI/hescape-pyarrow",
    name="human-breast-panel",
    split="train",
    streaming=True,
    # cache_dir="/path/to/cache",
    # num_proc=4
)
print(ds)
```

While you can stream the data to perform training, it is recommended to store the dataset locally for faster access and easier management. As you run the training script, the dataset gets downloaded automatically to the default HuggingFace cache.

To store the dataset locally for other uses, disable streaming by setting `streaming=False` and specify a `cache_dir` to store the dataset locally. You can also specify a number of processes to use for data loading by setting `num_proc` in the load_dataset function.

Check the huggingface [hescape-pyarrow DatasetCard](https://huggingface.co/datasets/Peng-AI/hescape-pyarrow) for more information

## Directory structure
The HESCAPE repository takes pretrained weights for pre-built images and genes to train the model. ***The directory structure is crucial for the training process to work correctly.*** The repository is structured as follows:

```bash
├── hescape (from github)
│   ├── README.md
│   ├── data
│   ├── experiments
│   ├── notebooks
│   ├── pyproject.toml
│   ├── src
│   ├── tests
│   ├── uv.lock
│   └── ...
├── pretrain_weights
│   ├── gene
│   │   ├── nicheformer
│   │   ├── drvi
│   │   └── <predefined gene models> ...
│   └── image
│        ├── h0-mini
│        ├── uni
│        └── <predefined image models> ...
```
## Training
- Single-GPU local
    ```bash
    source .venv/bin/activate

    uv run experiments/hescape_pretrain/train.py \
      --config-name=local_config.yaml \
      launcher=local \
      model.litmodule.img_enc_name=h0-mini \
      model.litmodule.gene_enc_name=drvi \
      training.lightning.trainer.devices=1 \
      datamodule.batch_size=256 \
      datamodule.num_workers=8
    ```
- Multi-GPU local DDP(Lightning)
    ```bash
    uv run experiments/hescape_pretrain/train.py \
      --config-name=local_config.yaml \
      launcher=local \
      training.lightning.trainer.devices=4 \
      training.lightning.trainer.strategy=ddp\
      datamodule.batch_size=256 \
      datamodule.num_workers=8
    ```
- Slurm example (Quick recipe)
    ```bash
    srun --nodes=1 --ntasks-per-node=4 --cpus-per-task=12 --gres=gpu:4 \
     --mem=480G --time=02:00:00 --partition=<part> ... \
     bash -lc '
        export WANDB_MODE=offline
        export HYDRA_FULL_ERROR=1
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export NCCL_DEBUG=INFO
        source .venv/bin/activate
        uv run experiments/hescape_pretrain/train.py \
             --config-name=default_config.yaml
     '

Training is launched via Hydra-based configuration with default configurations stored in [local_config.yaml](https://github.com/peng-lab/hescape/blob/main/experiments/configs/local_config.yaml).

To modify the default parameters, you can modify the `local_config.yaml` file directly or override arguments from the CLI as shown above. The paramteres that can be modified are listed in the next section.


## Understanding Hyperparameter Configuration
Our framework uses Hydra for flexible experiment configuration.

### Common Hyperparameters

| Config Key                             | Description                                 |  Values                                             |
|----------------------------------------|---------------------------------------------|-----------------------------------------------------|
| `model.litmodule.img_enc_name`         | Vision encoder backbone                     | `h0-mini`, `gigapath`, `ctranspath`, `uni`, `conch`, `custom` |
| `model.litmodule.gene_enc_name`        | Gene encoder architecture                   | `mlp`, `scfoundation`, `nicheformer`, `drvi`, `custom`       |
| `model.litmodule.img_proj`             | Projection head for image features          | `mlp`, `linear`, `transformer`                      |
| `model.litmodule.gene_proj`            | Projection head for gene features           | `mlp`, `linear`                                     |
| `model.litmodule.loss`                 | Contrastive loss type                       | `CLIP`, `SIGLIP`                                    |
| `model.litmodule.optimizer.lr`         | Learning rate                               | `1e-3`, `3e-4`, etc.                                |
| `model.litmodule.temperature`          | CLIP temperature parameter                  | `0.05`, `0.07`, etc.                                |
| `training.train` / `training.test`     | Toggle training or test mode                | `true`, `false`                                     |
| `training.lightning.trainer.max_steps` | Number of steps during training             | `20_000` etc.                                       |
| `datamodule.batch_size`                | Batch size for Dataloader                   | `64`, `256`, etc.                                   |
| `datamodule.num_workers`               | Subprocesses to use for data loading        | `4`, `8`, etc.                                      |

### Running a Config Sweep
Benchmark Sweeps with different parameters are only possible in a slurm environment with a ddp setup. Hydra automatically runs grid search over all specified values. For example:

```
model.litmodule.img_enc_name: h0-mini, uni
model.liftmodule.gene_enc_name: drvi, nicheformer
```
This will run all combinations:
`h0-mini + drvi`, `h0-mini + nicheformer`, `uni + drvi`, `uni + nicheformer`

Running Sweeps have been explained in [running_sweeps.md](https://github.com/peng-lab/hescape/blob/main/experiments/running_sweeps.md).

## Inference Demo
We provide a Jupyter notebook [image_model_loading.ipynb](https://github.com/peng-lab/hescape/blob/main/notebooks/image_model_loading.ipynb) that demonstrates how to load a pretrained model and extract features from histology images for mutation and gene expression prediction.


## Benchmark Results

Test Recall@5 subset for both Image-to-Gene (I2G) and Gene-to-Image (G2I) tasks across different tissue panels.
**Note**: “—” indicates out-of-memory during training. **Bold** = best result, _Underlined_ = second-best.

| Model                   | 5K I2G | 5K G2I | Multi-Tissue I2G | Multi-Tissue G2I | ImmOnc I2G | ImmOnc G2I | Colon I2G | Colon G2I | Breast I2G | Breast G2I | Lung I2G | Lung G2I |
|------------------------|-----------|-----------|------------------|------------------|--------|--------|-----------|-----------|------------|------------|----------|----------|
| mlp-gigapath           | 0.257     | 0.257     | 0.297            | 0.215            | 0.179  | 0.132  | 0.313     | 0.297     | 0.390      | 0.288      | 0.510    | 0.493    |
| mlp-optimus            | 0.235     | 0.235     | 0.209            | 0.153            | 0.173  | 0.119  | 0.296     | 0.291     | 0.309      | 0.235      | 0.358    | 0.336    |
| scfoundation-gigapath  | —         | —         | —                | —                | 0.251  | 0.207  | 0.294     | 0.249     | 0.348      | 0.365      | 0.590    | 0.543    |
| scfoundation-optimus   | —         | —         | —                | —                | 0.206  | 0.171  | 0.315     | 0.272     | 0.388      | 0.377      | 0.427    | 0.345    |
| nicheformer-gigapath   | 0.241     | 0.255     | 0.274            | 0.285            | 0.247  | 0.267  | 0.261     | 0.269     | 0.414      | 0.447      | 0.473    | 0.554    |
| nicheformer-optimus    | 0.243     | 0.273     | 0.261            | 0.277            | 0.212  | 0.215  | 0.290     | 0.278     | 0.418      | 0.451      | 0.424    | 0.498    |
| drvi-gigapath          | _0.315_   | **0.359** | **0.322**        | **0.417**        | **0.344**| **0.334**| 0.388   | 0.394     | _0.461_    | _0.436_    | **0.649**| **0.709**|
| drvi-optimus           | 0.299     | 0.321     | 0.271            | 0.342            | 0.287  | 0.267  | **0.412** | _0.397_   | **0.465**  | **0.461**  | 0.562    | 0.612    |
| drvi-uni               | **0.322** | _0.341_   | _0.312_          | _0.396_          | _0.326_| _0.318_| _0.404_  | **0.401** | 0.450      | 0.436      | _0.610_  | _0.678_  |

We provide more details about our full collection of results for all multi-modal combinations [here](https://github.com/peng-lab/hescape/blob/main/hescape_results.md).

## Updates

<!-- - **21.09.25**: HESCAPE has been accepted to ICCV-CVAMD Workshop 2025! We will be in Hawai'i from Oct 19th to 23rd. Send us a message if you wanna learn more about HESCAPE (rushin.gindra@helmholtz-munich.de).  -->
<!-- - **16.09.25**: HESCAPE is now public with implementation documentation. New image model (LUNIT) and DRVI model weights are available -->

- **02.09.25**: 6 new datasets released. You can find them on [huggingface](https://huggingface.co/datasets/Peng-AI/hescape-pyarrow).

## To-Do's
- [ ] Benchmark your own model
- [x] Documentation
- [x] New Xenium datasets
- [ ] New Visium datasets


## Issues
- GitHub issues are prefered
- If GitHub issues are not possible, email `rushin.gindra@helmholtz-munich.de`

## Contributing guide
- We are open to contributions from the multi-modal community.
- Feel free to reach out with a pull-request or via email if you have a prospective idea and need some assistance with implementing it.

## Acknowledgements
The project was built as an adaptation of functions from cool repositories such as [OpenClip](https://github.com/mlfoundations/open_clip), [HuggingFace Datasets](https://huggingface.co/docs/hub/en/datasets) and [Timm](https://github.com/huggingface/pytorch-image-models/) . We thank all authors and open-source developers for their contribution.


## Citation

Gindra, R. H., Palla, G., Nguyen, M., Wagner, S. J., Tran, M., Theis, F. J., Saur, D., Crawford, L., & Peng, T.
A Large-Scale Benchmark of Cross-Modal Learning for Histology and Gene Expression in Spatial Transcriptomics. arXiv preprint arXiv:2508.01490, August 2025.

```
@misc{gindra2025largescalebenchmarkcrossmodallearning,
      title={A Large-Scale Benchmark of Cross-Modal Learning for Histology and Gene Expression in Spatial Transcriptomics},
      author={Rushin H. Gindra and Giovanni Palla and Mathias Nguyen and Sophia J. Wagner and Manuel Tran and Fabian J Theis and Dieter Saur and Lorin Crawford and Tingying Peng},
      year={2025},
      eprint={2508.01490},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN},
      url={https://arxiv.org/abs/2508.01490},
}
```
