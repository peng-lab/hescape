import anndata as ad
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from nicheformer import Nicheformer
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from tqdm import tqdm

# ------------------------------------------------------------------------------------------


def normalize_counts(x: np.array):
    """Normalize each cell by total counts over all genes."""
    x = x.copy()
    counts = np.array(x.sum(axis=1)).flatten()

    # avoid zero devision error
    counts += (counts == 0).astype(float)

    # normalize to 10000. counts
    scaling_factor = 10000.0 / counts

    if issparse(x):
        sparsefuncs.inplace_row_scale(x, scaling_factor)
    else:
        np.multiply(x, scaling_factor.reshape((-1, 1)), out=x)

    return x


def tokenize_genes(adata: ad.AnnData):
    """Rank value encoding where genes are ranked by their expression."""
    # get the expression matrix and token ids
    exp_matrix = adata.X.toarray()
    token_ids = adata.var["token_id"].values

    # normalize each cell / spot by its mean
    gene_means = np.mean(exp_matrix, axis=0)
    mask = gene_means != 0
    exp_matrix[:, mask] /= gene_means[mask]

    # initialize arrays to store sorted data
    sorted_exp = np.zeros_like(exp_matrix)
    sorted_ids = np.zeros_like(exp_matrix, dtype=int)

    # perform vectorized sorting
    sorted_idx = np.argsort(-exp_matrix, axis=1)
    for idx in tqdm(range(exp_matrix.shape[0])):
        sorted_exp[idx, :] = exp_matrix[idx, sorted_idx[idx, :]]
        sorted_ids[idx, :] = token_ids[sorted_idx[idx, :]] + 30

    return sorted_exp, sorted_ids


def extract_gene_features(model: nn.Module, data: torch.Tensor):
    """Extract gene features from last hidden layer."""
    model.cuda()
    model.eval()

    # collect features in list
    features = []
    target_length = 1500

    # loop over all cells / spots
    for idx in tqdm(range(data.shape[0])):
        # prepare inputs by reshaping and padding
        inputs = torch.tensor(data[idx][:1500]).unsqueeze(0).cuda()
        padding = target_length - inputs.shape[1]
        inputs = F.pad(inputs, (0, padding))

        # make a forward pass to get the features
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = nicheformer(inputs, None)["transformer_output"]
                feature = torch.mean(outputs, dim=1).squeeze(0)
                features.append(feature.detach().cpu().numpy())

    # convert list to NumPy array
    features = np.array(features)

    return features


def create_adata(spatial_root, hest1k_id):
    """Preprocess spatial gene expression data from HEST-1K"""
    # create data path
    spatial_path = f"{spatial_root}/{hest1k_id}.h5ad"
    # read spatial data
    adata = ad.read_h5ad(spatial_path)
    # add a batch label
    adata.obs["batch"] = hest1k_id
    # add label to barcodes
    adata.obs.index = [f"{hest1k_id}_{idx}" for idx in adata.obs.index]
    # store the raw counts
    adata.layers["raw_counts"] = adata.X.copy()

    # update with ensemble ids
    adata.var.index = adata.var["gene_ids"]
    # extract current gene names
    new_gene_names = adata.var_names
    # map gene names to token ids
    token_ids = [gene_to_token_mapping.get(gene, -1) for gene in new_gene_names]
    # add token ids to adata
    adata.var["token_id"] = token_ids
    # filter out genes with token id -1
    adata = adata[:, adata.var["token_id"] != -1]

    return adata


# ---------- Load Nicheformer Model ---------- #

state_path = "/home/ubuntu/SpatialCLIP/pretrain_weights/nicheformer/nicheformer.ckpt"
state_dict = torch.load(state_path)["state_dict"]

nicheformer = Nicheformer()
nicheformer.load_state_dict(state_dict, strict=False)

# load reference adata object
reference = sc.read_h5ad("/home/ubuntu/SpatialCLIP/pretrain_weights/nicheformer/nicheformer_reference.h5ad")
reference.obs.reset_index(drop=True, inplace=True)

# Xenium technology means


# map gene names to token ids
gene_names = reference.var_names
gene_to_token_mapping = {gene: idx for idx, gene in enumerate(gene_names)}

# ----------- Load HEST-1K Dataset ----------- #

hest1k_ids = ["TENX68", "TENX53", "TENX39", "TENX24", "TENX14", "TENX13", "NCBI776"]

# save_root = ''
# os.makedirs(save_root, exist_ok=True)

spatial_root = "/mnt/raw/datasets/hest1k/st_processed_esm"

for hest1k_id in tqdm(hest1k_ids, desc="Loop over HEST-1K"):
    adata = create_adata(spatial_root, hest1k_id)

    sorted_gex, sorted_ids = tokenize_genes(adata)
    gene_features = extract_gene_features(nicheformer, sorted_ids)
