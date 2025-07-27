import argparse
import os
import warnings
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import scanpy as sc
from spatialclip._utils import find_root
from train import infer_drvi, preprocess_adata

root_dir = Path(find_root())


def get_parser():
    """
    Create an argument parser for the script.

    Returns
    -------
    parser (argparse.ArgumentParser): The argument parser.
    """
    # Get the main project directory where git is initialized
    DATA_ROOT = root_dir / "data" / "hest1k" / "st_tissue_specific"
    CHECKPOINT_PATH = root_dir / "pretrain_weights" / "drvi"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Path to h5 data samples", default=DATA_ROOT, required=False)
    parser.add_argument(
        "--checkpoint-path", help="Path to save model weights and embeddings", default=CHECKPOINT_PATH, required=False
    )
    parser.add_argument("--species", choices=["Homo sapiens", "Mus musculus"], default="Homo sapiens")
    parser.add_argument("--st_technology", choices=["Visium", "Xenium"], default="Visium")
    parser.add_argument("--organ", help="Organ(s) for which to run the analysis", default="Ovary", required=False)
    return parser


def main(args):
    """
    Main function to orchestrate the training or inference process.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    """
    adata = sc.read(args.data_root / args.h5_file)

    # adata = filter_adata(adata, args) # Not filtering right now. We see later

    adata = preprocess_adata(adata)
    print(f"Infering DRVI model for organs {args.organ}...")
    infer_drvi(args, adata)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.checkpoint_path, exist_ok=True)
    args.covariate_keys = ["batch"]
    args.species = "human" if args.species == "Homo sapiens" else "mouse"
    args.h5_file = f"hest_htan_{args.st_technology.lower()}_{args.species.lower()}_{args.organ.lower()}.h5ad"
    sc._settings.ScanpyConfig.figdir = args.checkpoint_path / "infer_figures"
    os.makedirs(sc._settings.ScanpyConfig.figdir, exist_ok=True)
    args.figures_dir = Path(sc._settings.ScanpyConfig.figdir)
    main(args)
