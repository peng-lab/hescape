import os

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
from nicheformer.data import NicheformerDataset
from nicheformer.models import Nicheformer
from torch.utils.data import DataLoader
from tqdm import tqdm

pl.seed_everything(42)


def create_adata(spatial_root, hest1k_id, reference, gene_to_token_mapping):
    """Preprocess spatial gene expression data from HEST-1K"""
    # create data path
    spatial_path = f"{spatial_root}/{hest1k_id}.h5ad"
    # read spatial data
    adata = ad.read_h5ad(spatial_path)
    # add a batch label
    adata.obs["batch"] = hest1k_id
    # add label to barcodes
    adata.obs_names = [f"{hest1k_id}_{idx}" for idx in adata.obs_names]
    # store the raw counts
    adata.layers["raw_counts"] = adata.X.copy()

    # update with ensemble ids
    niche_adata = adata.copy()
    niche_adata.var["gene_ids"] = niche_adata.var["gene_ids"].astype("string")
    valid_genes = niche_adata.var["gene_ids"].notna()
    niche_adata = niche_adata[:, valid_genes]
    niche_adata.var.index = niche_adata.var["gene_ids"].values

    niche_adata = ad.concat([reference, niche_adata], join="outer", axis=0)
    niche_adata = niche_adata[1:].copy()

    niche_adata.obs["modality"] = 4  # spatial
    niche_adata.obs["specie"] = 5  # human

    # if 'nicheformer_split' not in adata.obs.columns:
    niche_adata.obs["nicheformer_split"] = "train"

    # extract current gene names
    new_gene_names = niche_adata.var_names
    # map gene names to token ids
    token_ids = [gene_to_token_mapping.get(gene, -1) for gene in new_gene_names]
    # add token ids to adata
    niche_adata.var["token_id"] = token_ids
    # filter out genes with token id -1
    niche_adata = niche_adata[:, niche_adata.var["token_id"] != -1]

    return adata, niche_adata


def main():
    config = {
        "data_path": "/mnt/raw/datasets/hest1k/st_drvi/discrete",  #'path/to/your/data.h5ad',  # Path to your AnnData file
        "technology_mean_path": "/home/ubuntu/SpatialCLIP/pretrain_weights/nicheformer/xenium_mean_script.npy",  #'path/to/technology_mean.npy',  # Path to technology mean file
        "reference_path": "/home/ubuntu/SpatialCLIP/pretrain_weights/nicheformer/nicheformer_reference.h5ad",  # Path to reference AnnData
        "checkpoint_path": "/home/ubuntu/SpatialCLIP/pretrain_weights/nicheformer/nicheformer.ckpt",  # Path to model checkpoint
        "output_path": "/mnt/raw/datasets/hest1k/st_nicheformer/",  # Where to save the result, it is a new h5ad
        "output_dir": ".",  # Directory for any intermediate outputs
        "batch_size": 32,
        "max_seq_len": 1500,
        "aux_tokens": 30,
        "chunk_size": 1000,  # to prevent OOM
        "num_workers": 10,
        "precision": "bf16-mixed",
        "embedding_layer": -1,  # Which layer to extract embeddings from (-1 for last layer)
        "embedding_name": "embeddings",  # Name suffix for the embedding key in adata.obsm
    }

    nicheformer_reference = ad.read_h5ad(config["reference_path"])
    gene_names = nicheformer_reference.var_names
    gene_to_token_mapping = {gene: idx for idx, gene in enumerate(gene_names)}
    technology_mean = np.load(config["technology_mean_path"])

    model = Nicheformer.load_from_checkpoint(config["checkpoint_path"], strict=False)
    model.eval()
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=config["output_dir"],
        precision=config.get("precision", 32),
    )

    for hest1k_id in os.listdir(config["data_path"]):
        hest1k_id = hest1k_id.split(".")[0]

        if os.path.exists(f"{config['output_path']}/{hest1k_id}.h5ad"):
            print(f"{config['output_path']}/{hest1k_id}.h5ad already exists, skipping")
            continue
        adata, niche_adata = create_adata(config["data_path"], hest1k_id, nicheformer_reference, gene_to_token_mapping)
        # raise
        dataset = NicheformerDataset(
            adata=niche_adata,
            technology_mean=technology_mean,
            split="train",
            max_seq_len=1500,
            aux_tokens=config.get("aux_tokens", 30),
            chunk_size=config.get("chunk_size", 1000),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
        )

        print(f"Extracting embeddings...{hest1k_id}")
        embeddings = []
        device = model.embeddings.weight.device

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Get embeddings from the model
                emb = model.get_embeddings(
                    batch=batch,
                    layer=config.get("embedding_layer", -1),  # Default to last layer
                )
                embeddings.append(emb.cpu().numpy())

        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)

        embedding_key = f"X_niche_{config.get('embedding_name', 'embeddings')}"
        adata.obsm[embedding_key] = embeddings

        # revert index back with gene names
        adata.write_h5ad(f"{config['output_path']}/{hest1k_id}.h5ad")


if __name__ == "__main__":
    main()
