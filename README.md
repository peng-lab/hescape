# HESCAPE: A Large-Scale Benchmark for Cross-Modal Learning in Spatial Transcriptomics
## Multimodal Contrastive Pretraining for Spatial Transcriptomics and Histology

[ [arXiv](https://arxiv.org/abs/2508.01490) | [Blog](https://www.linkedin.com/pulse/hescape-benchmark-visiongenomics-alignment-spatial-rushin-gindra-rfrsf/?trackingId=FQFxbUmNyOlFcCLuvwqPcA%3D%3D) | [Data](https://huggingface.co/datasets/marr-peng-lab/paired_ts8_human_breast_panel) | [Cite](https://github.com/peng-lab/hescape?tab=readme-ov-file#citation) \]

**Abstract**: Spatial transcriptomics enables simultaneous measurement of gene expression and tissue morphology, offering unprecedented insights into cellular organization and disease mechanisms. However, the field lacks comprehensive benchmarks for evaluating multimodal learning methods that leverage both histology images and gene expression data. Here, we present **HESCAPE**, a large-scale benchmark for cross-modal contrastive pretraining in spatial transcriptomics, built on a curated pan-organ dataset spanning 6 different gene panels and 54 donors. We systematically evaluated state-of-the-art image and gene expression encoders across multiple pre-training strategies and assessed their effectiveness on two downstream tasks: gene mutation classification and gene expression prediction. Our benchmark demonstrates that gene expression encoders are the primary determinant of strong representational alignment, and that gene models pretrained on spatial transcriptomics data outperform both those trained without spatial data and simple baseline ap- proaches. However, downstream task evaluation reveals a striking contradiction: while contrastive pretraining consistently improves gene mutation classification performance, it degrades direct gene expression prediction compared to baseline encoders trained without cross-modal objectives. We identify batch effects as a key factor that interferes with effec- tive cross-modal alignment. Our findings highlight the critical need for batch-robust multimodal learning approaches in spatial transcriptomics. To accelerate progress in this direction, we release HESCAPE, providing standardized datasets, evaluation protocols, and benchmarking tools for the community

<img src="figures/schematic.png" alt="HESCAPE framework" width="800" />

## What does HESCAPE offer?
🤗 **Huggingface multi-modal datase**t: We release a histology-transcriptomics 1-1 mapped streamable pyarrow datasets for 6 independent 10x Xenium gene panels respectively.

💡 **Pretraining at scale**: We provide a framework that can systematically evaluate 4 gene expression encoders (DRVI, Nicheformer, scFoundation, MLP) and 5 pathology foundation models (Gigapath, UNI, CONCH, H0-mini, CtransPath) across CLIP-style contrastive objectives. We enable users to evaluate their own custom pathology or gene models in a multi-modal CLIP setting.

🔁 **Cross-modal retrieval tasks**: We test how well image patches retrieve matching gene vectors (I2G) and vice versa (G2I), with Recall@5 scores revealing insights into encoder generalizability and dataset-specific tuning.

🧬 **Enables downstream tasks**: We provide tools for users to use HESCAPE trained models for inference in downstream tasks like classification of clinically relevant mutations like MSI, BRAF, KRAS, and Gene expression prediction from Histology.


## To-Do's
- [ ] Benchmark your own model
- [ ] New Visium/Xenium datasets
- [ ] Data preprocessing pipeline
- [ ] Docker Container

## HESCAPE installation
We support installation via uv, PyPI, Conda, and Docker (coming soon):

### pip / uv (recommended)

```
cd hescape
uv venv --python 3.11
pip install git+https://github.com/peng-lab/hescape.git@main

```
### conda
```
git clone https://github.com//peng-lab/hescape.git
cd hescape
conda create -n "hescape" python=3.11
conda activate hescape
pip install -e .
```
<!-- ### docker -->

<!--
## Dataset Download

- **HEST**
- **10x Genomics Website (Xenium)**
- **10x Genomics Website (Visium)**
- **HTAN - WUSTL (2D/3D)**
- **HTAN - HTAPP**
- **GEO database**

## Dataset Preprocessing
- **zarr**
- **HestData objects**
- **Huggingface datasets**
(link to notebook or datasets)

Run all conversions via:
```
python preprocess.py --
```
-->
## Hyperparameter Configuration
Our framework uses Hydra for flexible experiment configuration. You can adjust hyperparameters, model settings, or training options directly via a single config file or from the command line.

### Editing the config file
The main configuration file /experiments/configs/default_config.yaml is in YAML format. Here's an example snippet:
```
sweeper:
  params:
    model.litmodule.img_enc_name: h0-mini, gigapath, conch
    model.litmodule.img_proj: mlp
    model.litmodule.loss: CLIP
    datamodule.batch_size: 64
    training.train: true
```

To modify:
- Change values inline (e.g., batch_size: 256)
- Use comma-separated values to sweep over multiple options
### Running a Config Sweep
Hydra automatically runs grid search over all specified values. For example:

```
model.litmodule.img_enc_name: h0-mini, uni
model.liftmodule.gene_enc_name: drvi, nicheformer
```
This will run all combinations:
`h0-mini + drvi`, `h0-mini + nicheformer`, `uni + drvi`, `uni + nicheformer`


### Override from CLI
You can override config values when launching a job:
```
python experiments/hescape_pretrain/train.py model.litmodule.img_enc_name=uni datamodule.batch_size=128
```

### Common Hyperparameters

| Config Key                             | Description                                 |  Values                                             |
|----------------------------------------|---------------------------------------------|-----------------------------------------------------|
| `model.litmodule.img_enc_name`         | Vision encoder backbone                     | `h0-mini`, `gigapath`, `ctranspath`, `uni`, `conch` |
| `model.litmodule.gene_enc_name`        | Gene encoder architecture                   | `mlp`, `scfoundation`, `nicheformer`, `drvi`        |
| `model.litmodule.img_proj`             | Projection head for image features          | `mlp`, `linear`, `transformer`                      |
| `model.litmodule.gene_proj`            | Projection head for gene features           | `mlp`, `linear`                                     |
| `model.litmodule.loss`                 | Contrastive loss type                       | `CLIP`, `SIGLIP`                                    |
| `model.litmodule.optimizer.lr`         | Learning rate                               | `1e-3`, `3e-4`, etc.                                |
| `model.litmodule.temperature`          | XXX                                         | `0.05`, `0.07`, etc.                                |
| `training.train` / `training.test`     | Toggle training or test mode                | `true`, `false`                                     |
| `training.lightning.trainer.max_steps` | Number of steps during training             | `20_000` etc.                                       |
| `datamodule.batch_size`                | Batch size for Dataloader                   | `64`, `256`, etc.                                   |
| `datamodule.num_workers`               | Subprocesses to use for data loading        | `4`, `8`, etc.                                      |

For complete configuration pattern, check out [default_config.yaml](https://github.com/peng-lab/hescape/blob/main/experiments/configs/default_config.yaml)

## Training
Training is launched via Hydra-based configuration. Running `experiments/hescape_pretrain/train.py` without any additional parameters will perform training with parameters from default_config.yaml.
```
python experiments/hescape_pretrain/train.py --config-name default_config.yaml
```

## Inference Demo
We provide a Jupyter notebook [image_model_loading.ipynb](https://github.com/peng-lab/hescape/blob/main/notebooks/image_model_loading.ipynb) that demonstrates how to load a pretrained model and extract features from histology images for mutation and gene expression prediction.


## Benchmark
### Benchmark Results

### Complete Test Recall@5 Results (I2G and G2I)
Complete Test Recall@5 Results for both Image-to-Gene (I2G) and Gene-to-Image (G2I) tasks across different tissue panels.
**Note**: “—” indicates out-of-memory during training. **Bold** = best result, _Underlined_ = second-best.

| Model                   | 5K I2G | 5K G2I | Multi-Tissue I2G | Multi-Tissue G2I | ImmOnc I2G | ImmOnc G2I | Colon I2G | Colon G2I | Breast I2G | Breast G2I | Lung I2G | Lung G2I |
|------------------------|-----------|-----------|------------------|------------------|--------|--------|-----------|-----------|------------|------------|----------|----------|
| mlp-ctranspath         | 0.103     | 0.106     | 0.138            | 0.098            | 0.110  | 0.094  | 0.098     | 0.122     | 0.116      | 0.117      | 0.103    | 0.126    |
| mlp-conch              | 0.228     | 0.228     | 0.241            | 0.178            | 0.187  | 0.130  | 0.300     | 0.258     | 0.383      | 0.260      | 0.443    | 0.418    |
| mlp-gigapath           | 0.257     | 0.257     | 0.297            | 0.215            | 0.179  | 0.132  | 0.313     | 0.297     | 0.390      | 0.288      | 0.510    | 0.493    |
| mlp-optimus            | 0.235     | 0.235     | 0.209            | 0.153            | 0.173  | 0.119  | 0.296     | 0.291     | 0.309      | 0.235      | 0.358    | 0.336    |
| mlp-uni                | 0.247     | 0.246     | 0.255            | 0.171            | 0.243  | 0.130  | 0.320     | 0.317     | 0.346      | 0.248      | 0.493    | 0.493    |
| scfoundation-ctranspath| —         | —         | —                | —                | 0.119  | 0.126  | 0.105     | 0.098     | 0.138      | 0.122      | 0.125    | 0.121    |
| scfoundation-conch     | —         | —         | —                | —                | 0.219  | 0.177  | 0.287     | 0.262     | 0.331      | 0.309      | 0.503    | 0.458    |
| scfoundation-gigapath  | —         | —         | —                | —                | 0.251  | 0.207  | 0.294     | 0.249     | 0.348      | 0.365      | 0.590    | 0.543    |
| scfoundation-optimus   | —         | —         | —                | —                | 0.206  | 0.171  | 0.315     | 0.272     | 0.388      | 0.377      | 0.427    | 0.345    |
| scfoundation-uni       | —         | —         | —                | —                | 0.236  | 0.173  | 0.297     | 0.244     | 0.358      | 0.351      | 0.543    | 0.479    |
| nicheformer-ctranspath | 0.105     | 0.115     | 0.126            | 0.117            | 0.127  | 0.127  | 0.092     | 0.106     | 0.110      | 0.126      | 0.122    | 0.138    |
| nicheformer-conch      | 0.237     | 0.274     | 0.258            | 0.286            | 0.224  | 0.247  | 0.267     | 0.253     | 0.366      | 0.410      | 0.412    | 0.496    |
| nicheformer-gigapath   | 0.241     | 0.255     | 0.274            | 0.285            | 0.247  | 0.267  | 0.261     | 0.269     | 0.414      | 0.447      | 0.473    | 0.554    |
| nicheformer-optimus    | 0.243     | 0.273     | 0.261            | 0.277            | 0.212  | 0.215  | 0.290     | 0.278     | 0.418      | 0.451      | 0.424    | 0.498    |
| nicheformer-uni        | 0.259     | 0.291     | 0.269            | 0.288            | 0.243  | 0.261  | 0.252     | 0.250     | 0.416      | 0.442      | 0.449    | 0.538    |
| drvi-ctranspath        | 0.106     | 0.116     | 0.116            | 0.147            | 0.138  | 0.135  | 0.111     | 0.123     | 0.162      | 0.144      | 0.143    | 0.163    |
| drvi-conch             | 0.266     | 0.316     | 0.298            | 0.363            | 0.300  | 0.290  | 0.356     | 0.362     | 0.397      | 0.397      | 0.539    | 0.597    |
| drvi-gigapath          | _0.315_   | **0.359** | **0.322**        | **0.417**        | **0.344**| **0.334**| 0.388   | 0.394     | _0.461_    | _0.436_    | **0.649**| **0.709**|
| drvi-optimus           | 0.299     | 0.321     | 0.271            | 0.342            | 0.287  | 0.267  | **0.412** | _0.397_   | **0.465**  | **0.461**  | 0.562    | 0.612    |
| drvi-uni               | **0.322** | _0.341_   | _0.312_          | _0.396_          | _0.326_| _0.318_| _0.404_  | **0.401** | 0.450      | 0.436      | _0.610_  | _0.678_  |

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
