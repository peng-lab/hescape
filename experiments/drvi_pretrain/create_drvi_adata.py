import argparse
import os
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from spatialclip._utils import find_root

# ignore ImplicitModificationWarning
warnings.filterwarnings("ignore", category=ad.ImplicitModificationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

root_dir = Path(find_root())
data_root = root_dir / "data" / "htan-wustl"
st_root = data_root / "st"
save_root = data_root / "st_tissue_specific"
os.makedirs(save_root, exist_ok=True)

genes_1370 = np.load(root_dir / "experiments/drvi_pretrain/xenium_human_1370.npy")

metadata_csv_file = data_root / "HTAN_WUSTL_v1_0_0.csv"


def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for spatial data preprocessing")
    parser.add_argument("--species", choices=["Homo sapiens", "Mus musculus"], default="Homo sapiens")
    parser.add_argument("--st_technology", choices=["Visium", "Xenium"], default="Visium")
    return parser.parse_args()


def subset_anndata(adata_dict, meta_df, target_genes):
    org = "hsapiens"
    # annotations = sc.queries.biomart_annotations(
    #     org=org, attrs=["ensembl_gene_id", "external_gene_name"], use_cache=False
    # )
    # gene_name_to_ensembl = dict(zip(annotations["external_gene_name"], annotations["ensembl_gene_id"], strict=False))

    processed_adata_dict = {}
    target_genes = [gene.upper() for gene in target_genes]

    for name, adata in adata_dict.items():
        adata.var_names = adata.var_names.str.upper()

        existing_genes = adata.var_names.intersection(target_genes)
        missing_genes = list(set(target_genes) - set(existing_genes))

        new_data = np.zeros((adata.n_obs, len(target_genes)))
        new_adata = ad.AnnData(
            X=new_data,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=target_genes),
            uns=adata.uns.copy(),
            obsm=adata.obsm.copy(),
        )

        new_adata[:, existing_genes].X = adata[:, existing_genes].X.copy()
        new_adata.obs["organ"] = meta_df.loc[meta_df["id"] == name, "organ"].values[0]
        new_adata.var["gene_names"] = new_adata.var_names.values
        # new_adata.var["gene_ids"] = new_adata.var["gene_names"].map(gene_name_to_ensembl)

        processed_adata_dict[name] = new_adata

        # print(f"Processed {name}: {len(existing_genes)} common genes, {len(missing_genes)} filled with zero.")

    return processed_adata_dict


def main():
    args = get_args()

    meta_df = pd.read_csv(metadata_csv_file)

    meta_df = meta_df[
        (meta_df["species"] == args.species)
        & (meta_df["st_technology"] == args.st_technology)  # "Spatial Transcriptomics"
    ]

    unique_organs = meta_df["organ"].unique()
    print(f"Unique organs: {unique_organs}")

    h5_samples = [h5_name.split(".")[0] for h5_name in os.listdir(st_root)]

    os.makedirs(save_root, exist_ok=True)

    for organ in unique_organs:
        print(f"Processing organ: {organ}")
        # Filter samples for the current organ
        organ_mask = meta_df["organ"] == organ
        current_meta_df = meta_df[organ_mask]

        # If the current_meta_df only has 1 sample, ignore it.
        if len(current_meta_df) <= 1:
            continue

        current_meta_df = current_meta_df[current_meta_df["id"].isin(h5_samples)]

        adata_dict = {}
        for st in current_meta_df["id"].to_list():
            adata_dict[st.split(".")[0]] = sc.read_h5ad(st_root / f"{st}.h5ad")

        processed_adata_dict = subset_anndata(adata_dict, current_meta_df, target_genes=list(genes_1370))

        drvi_adata = ad.concat(
            processed_adata_dict,
            join="inner",
            merge="same",
            uns_merge=None,
            label="batch",
            keys=None,
            index_unique=None,
        )

        print(drvi_adata.obs["batch"].value_counts())

        if args.species == "Homo sapiens":
            species = "human"
        elif args.species == "Mus musculus":
            species = "mouse"
        drvi_adata.obs_names_make_unique()
        drvi_adata.write(save_root / f"htan_{args.st_technology.lower()}_{species.lower()}_{organ.lower()}.h5ad")


if __name__ == "__main__":
    main()
