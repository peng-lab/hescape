import argparse
import os
import warnings
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import anndata as ad
import drvi
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from drvi.model import DRVI
from scipy.stats import median_abs_deviation

# from spatialclip._utils import find_root

# root_dir = Path(find_root())

# TEST_LIST = [
#     "TENX95",  # IDC
#     "TENX99",
#     "NCBI783",
#     "NCBI785",
#     "MEND139",  # PRAD
#     "MEND140",
#     "MEND141",
#     "MEND142",
#     "MEND143",
#     "MEND144",
#     "MEND145",
#     "MEND146",
#     "MEND147",
#     "MEND148",
#     "MEND149",
#     "MEND150",
#     "MEND151",
#     "MEND152",
#     "MEND153",
#     "MEND154",
#     # "MEND155",
#     "MEND156",
#     "MEND157",
#     "MEND158",
#     "MEND159",
#     "MEND160",
#     "MEND161",
#     "MEND162",
#     "TENX116",  # PAAD
#     "TENX126",
#     "TENX140",
#     "TENX115",  # SKCM
#     "TENX117",
#     "TENX111",  # COAD
#     "TENX147",
#     "TENX148",
#     "TENX149",
#     "ZEN36",  # READ
#     "ZEN40",
#     "ZEN48",
#     "ZEN49",
#     "INT1",  # ccRCC
#     "INT2",
#     "INT3",
#     "INT4",
#     "INT5",
#     "INT6",
#     "INT7",
#     "INT8",
#     "INT9",
#     "INT10",
#     "INT11",
#     "INT12",
#     "INT13",
#     "INT14",
#     "INT15",
#     "INT16",
#     "INT17",
#     "INT18",
#     "INT19",
#     "INT20",
#     "INT21",
#     "INT22",
#     "INT23",
#     "INT24",
#     "NCBI642",  # HCC
#     "NCBI643",
#     "TENX118",  # LUNG
#     "TENX141",
#     "NCBI681",  # IDC-LymphNode
#     "NCBI682",
#     "NCBI683",
#     "NCBI684",
# ]

TEST_LIST = [
    "TENX114",
    "TENX111"  # colon
    "NCBI884",
    "NCBI882",
    "NCBI866",
    "NCBI858",
    "NCBI883",
    "NCBI861",
    "NCBI857",
    "NCBI881",  # lung healthy
    "NCBI783",
    "NCBI785",  # breast
    "TENX106",
    "TENX120",
    "TENX125",  # multi-tissue
    "TENX117",  # skin,
    "TENX141",
    "TENX140",  # immuno-onco
    "TENX157",
    "TENX158",
    "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun",  # 5k prime
]


def get_parser():
    """
    Create an argument parser for the script.

    Returns
    -------
    parser (argparse.ArgumentParser): The argument parser.
    """
    # Get the main project directory where git is initialized
    DATA_ROOT = Path("/p/project1/hai_spatial_clip/data/xenium/preprocessing_adatas")
    CHECKPOINT_PATH = Path("/p/project1/hai_spatial_clip/pretrain_weights/gene/")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Path to h5 data samples", default=DATA_ROOT, required=False)
    parser.add_argument(
        "--checkpoint-path", help="Path to save model weights and embeddings", default=CHECKPOINT_PATH, required=False
    )
    # gene-panel as an argument
    parser.add_argument("--h5-file", help="Gene panel used for training", default="human_colon_panel_adata.h5ad")

    return parser


def is_outlier(adata, metric: str, nmads: int):
    """
    Identify outliers in a dataset based on a specified metric.

    Parameters
    ----------
    adata (AnnData): Annotated data matrix.
    metric (str): The column name in the `.obs` attribute to check for outliers.
    nmads (int): Number of median absolute deviations from the median to consider as an outlier.

    Returns
    -------
    bool: True if the value is an outlier, False otherwise.
    """
    M = adata.obs[metric]
    return (M < np.median(M) - nmads * median_abs_deviation(M)) | (np.median(M) + nmads * median_abs_deviation(M) < M)


def filter_adata(adata, train=False):
    """
    Filter the dataset based on whether it's for training or inference.

    Parameters
    ----------
    adata (AnnData): Annotated data matrix.
    args (argparse.Namespace): Parsed command-line arguments.

    Returns
    -------
    adata (AnnData): Filtered Annotated data matrix.
    """
    if train:
        return adata[~adata.obs["name"].isin(TEST_LIST)].copy()
    else:
        return adata[adata.obs["name"].isin(TEST_LIST)].copy()


def preprocess_adata(adata):
    """
    Preprocess the dataset by filtering outliers, normalizing, and logging the data.

    Parameters
    ----------
    adata (AnnData): Annotated data matrix.

    Returns
    -------
    adata (AnnData): Preprocessed Annotated data matrix.
    """
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)
    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", 5)
        | is_outlier(adata, "log1p_n_genes_by_counts", 5)
        | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
    )
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (adata.obs["pct_counts_mt"] > 8)
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    adata.layers["counts"] = adata.X.copy()
    return adata


def train_drvi(args, adata):
    """
    Train the DRVI model on the provided dataset.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    adata (AnnData): Annotated data matrix.
    """
    # Configure DRVI for training
    covariate_keys = args.covariate_keys if args.covariate_keys else None
    DRVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=covariate_keys, is_count_data=True)
    model = DRVI(
        adata,
        categorical_covariates=covariate_keys if covariate_keys else (),
        n_latent=128,
        encoder_dims=[128, 128],
        decoder_dims=[128, 128],
    )
    print(model.view_anndata_setup(adata))
    print(model)

    n_obs = adata.obs.shape[0]
    max_epochs = np.min([round((20000 / n_obs) * 400), 400])
    max_epochs = round(max_epochs)
    if max_epochs >= 400:
        plan_kwargs = None
    else:
        plan_kwargs = {"n_epochs_kl_warmup": round(max_epochs * 0.5)}

    adata_reference = adata[:10].copy()
    adata_reference.write(args.checkpoint_path / f"drvi_{args.h5_file.split('_adata.h5ad')[0]}" / "drvi_reference.h5ad")

    model.train(
        max_epochs=max_epochs,
        train_size=1.0,
        batch_size=128,
        early_stopping=False,
        early_stopping_patience=20,
        use_gpu=True,
        accelerator="gpu",
        devices=-1,
        plan_kwargs=plan_kwargs,
    )
    model.save(
        args.checkpoint_path / f"drvi_{args.h5_file.split('_adata.h5ad')[0]}" / "drvi",
        overwrite=True,
    )

    plt.figure()
    plt.plot(model.history["reconstruction_loss_train"]["reconstruction_loss_train"], label="train")
    # plt.plot(model.history["reconstruction_loss_validation"]["reconstruction_loss_validation"], label="validation")
    plt.legend()
    plt.savefig(args.figures_dir / f"reconstruction_loss_{args.h5_file.split('.')[0]}.png")


def infer_drvi(args, adata, train=False):
    """
    Perform inference using the trained DRVI model on the provided dataset.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    adata (AnnData): Annotated data matrix.
    """
    phase = "train" if train else "test"

    model_path = args.checkpoint_path / f"drvi_{args.h5_file.split('_adata.h5ad')[0]}" / "drvi"
    DRVI.prepare_query_anndata(adata, str(model_path))
    drvi_q = DRVI.load_query_data(adata, str(model_path))
    drvi_q.is_trained = True
    latent = drvi_q.get_latent_representation()
    adata.obsm["drvi"] = latent

    # plot drvi embed umap
    embed = ad.AnnData(latent, obs=adata.obs)
    sc.pp.subsample(embed, fraction=1.0)  # Shuffling for better visualization of overlapping colors
    sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
    sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
    sc.pl.umap(
        embed,
        color=["name", "cancer", "tissue"],
        ncols=1,
        frameon=False,
        save=f"_embed_{args.h5_file.split('.')[0]}_{phase}.png",
    )

    # plot raw expression umap
    sc.pp.normalize_total(adata, inplace=True, key_added="norm_factor")
    sc.pp.subsample(adata, fraction=1.0)  # Shuffling for better visualization of overlapping colors
    sc.pp.neighbors(adata, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
    sc.tl.umap(adata, spread=1.0, min_dist=0.5, random_state=123)
    sc.pl.umap(
        adata,
        color=["name", "cancer", "tissue"],
        ncols=1,
        frameon=False,
        save=f"_counts_{args.h5_file.split('.')[0]}_{phase}.png",
    )

    drvi.utils.tl.set_latent_dimension_stats(drvi_q, embed)
    # embed.var.sort_values("reconstruction_effect", ascending=False)[:5]
    latent_plt = drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2, remove_vanished=True, show=False)
    latent_plt.savefig(args.figures_dir / f"latent_dimension_stats_{args.h5_file.split('.')[0]}_{phase}.png")


def main(args):
    """
    Main function to orchestrate the training or inference process.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    """
    adata = sc.read(args.data_root / args.h5_file)

    # VV Important
    # Sort the variable names to ensure consistent ordering
    var_names = adata.var_names
    var_names = sorted(var_names)
    adata = adata[:, var_names].copy()

    train_adata = filter_adata(adata, train=True)
    train_adata = preprocess_adata(train_adata).copy()
    train_adata.obs["cancer"] = train_adata.obs["cancer"].astype("category")
    train_drvi(args, train_adata)
    infer_drvi(args, train_adata, train=True)

    # adata = filter_adata(adata, args) # Not filtering right now. We see later
    test_adata = filter_adata(adata, train=False)
    if test_adata.shape[0] == 0:
        print("No test data found. Skipping inference.")
        return
    test_adata.obs["cancer"] = test_adata.obs["cancer"].astype("category")
    infer_drvi(args, test_adata, train=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.checkpoint_path = Path(args.checkpoint_path)
    os.makedirs(args.checkpoint_path / f"drvi_{args.h5_file.split('_adata.h5ad')[0]}", exist_ok=True)
    args.covariate_keys = ["name"]
    # args.h5_file = f"human_{args.gene_panel}_adata.h5ad"

    sc._settings.ScanpyConfig.figdir = args.checkpoint_path / "drvi_figures"
    os.makedirs(sc._settings.ScanpyConfig.figdir, exist_ok=True)
    args.figures_dir = Path(sc._settings.ScanpyConfig.figdir)
    main(args)
