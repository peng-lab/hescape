from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from datasets import load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from hescape.constants import DatasetEnum

ENUMS = [DatasetEnum.NAME, DatasetEnum.GEXP, DatasetEnum.SOURCE, DatasetEnum.TISSUE]
# DatasetEnum.TISSUE,
# DatasetEnum.ASSAY,
# DatasetEnum.SOURCE,
# DatasetEnum.PRESERVATION_METHOD


SELECTION = [DatasetEnum.NAME, DatasetEnum.GEXP, DatasetEnum.IMG, DatasetEnum.SOURCE, DatasetEnum.TISSUE]


class scFoundationTransform(torch.nn.Module):
    """
    scFoundation-based bulk transform adaptation for HF dataset.
    data X is always raw! It's not normalized or log-transformed.
    input_type: bulk
    pre_normalized: F (false)

    """

    def __init__(self, scFoundation_gene_index_path, data_gene_reference_path) -> None:
        super().__init__()

        gene_list_df = pd.read_csv(scFoundation_gene_index_path, header=0, delimiter="\t")
        self.gene_list = list(gene_list_df["gene_name"])

        self.data_gene_ref = ad.read_h5ad(data_gene_reference_path)
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.data_gene_ref.var_names)}

    def main_gene_selection(self, X_tensor: torch.Tensor, gene_list: list):
        """
        Describe:
            rebuild the input adata to select target genes encode protein
        Parameters:
            adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        Returns:
            adata_new->`~anndata.AnnData` object
            to_fill_columns->list: zero padding gene\
        """
        gene_order = list(self.data_gene_ref.var_names)  # original column order
        gene_to_idx = self.gene_to_idx

        selected_columns = []
        to_fill_columns = []

        for gene in gene_list:
            if gene in gene_to_idx:
                col_tensor = X_tensor[:, gene_to_idx[gene]].unsqueeze(1)
            else:
                col_tensor = torch.zeros(X_tensor.size(0), 1, device=X_tensor.device)
                to_fill_columns.append(gene)
            selected_columns.append(col_tensor)

        out_tensor = torch.cat(selected_columns, dim=1)
        var = pd.DataFrame(index=gene_list)
        var["mask"] = [1 if g in to_fill_columns else 0 for g in gene_list]
        return out_tensor, to_fill_columns, var

    def forward(self, X: torch.Tensor):
        """
        Given:
            X: Tensor of shape (N, <# panel genes>) with gene expression data ordered as in data_gene_ref.
            Convert gene feature into 19264
            Get totalcount of each row (cell | spot) --> log
            Normalizes gene expression counts per cell and applies log transformation.
            append total_counts to input_gene_x
            fetch data_gene_ids

        """
        X = X.squeeze()  # remove the extra dimension if it exists

        if X.shape[1] < len(self.gene_list):
            gexpr_feature, to_fill_columns, var = self.main_gene_selection(X, self.gene_list)
            # here gexpr_feature is a DataFrame with shape (N, 19264)
        else:
            gexpr_feature = X

        counts_per_cell = torch.sum(gexpr_feature, dim=1)
        counts_greater_than_zero = counts_per_cell[counts_per_cell > 0]  # compute counts_greater_that_zero for the X.
        after, _ = torch.median(counts_greater_than_zero, dim=0)
        counts_per_cell += counts_per_cell == 0  # Avoid division by zero
        normalized_counts_per_cell = counts_per_cell / after
        normalized = torch.div(gexpr_feature, normalized_counts_per_cell[:, None])
        gexpr_feature = torch.log1p(normalized)

        totalcount = torch.log10(counts_per_cell[:, None])
        pretrain_gene_x = torch.cat([gexpr_feature, totalcount, totalcount], dim=1)

        # value_labels = pretrain_gene_x > 0

        # data_gene_ids = torch.arange(19266).repeat(pretrain_gene_x.shape[0], 1)
        # x, x_padding = gatherData(pretrain_gene_x, value_labels, pad_token_id=103) # fixing pad_token_id to 103 for now

        # (N, 19264 + 2) where 19264 is the number of genes in the scFoundation reference.
        return pretrain_gene_x


class NicheformerTransform(torch.nn.Module):
    """Nicheformer-based transform adaptation for HF dataset."""

    def __init__(
        self,
        nicheformer_reference_path,
        technology_mean_path,
        data_gene_reference_path,
        max_seq_len=1500,
        aux_tokens=30,
    ) -> None:
        super().__init__()

        # Load references.
        nicheformer_ref = ad.read_h5ad(nicheformer_reference_path)
        data_gene_ref = ad.read_h5ad(data_gene_reference_path)

        # Create mapping from nicheformer gene names to token IDs.
        nicheformer_gene_to_token_mapping = {gene: idx for idx, gene in enumerate(nicheformer_ref.var_names)}

        # Load technology mean (e.g., xenium mean) and always use it.
        technology_mean = np.load(technology_mean_path)
        self.technology_mean = torch.from_numpy(technology_mean)

        data_gene_ref.var.index = data_gene_ref.var["gene_ids"]
        # extract the current gene_ids in the data gene reference
        new_gene_names = data_gene_ref.var["gene_ids"]
        # map gene names to token ids
        token_ids = [nicheformer_gene_to_token_mapping.get(gene, -1) for gene in new_gene_names]
        # add token ids to adata
        data_gene_ref.var["token_id"] = token_ids

        # filter out genes with token id -1. but we need to do this for all spots in HF dataset as well.
        # create mask on genes with token id not equal to -1.
        self.gene_mask = data_gene_ref.var["token_id"] != -1
        # data_gene_ref = data_gene_ref[:, data_gene_ref.var["token_id"] != -1]
        data_gene_ref = data_gene_ref[:, data_gene_ref.var["token_id"] != -1]
        self.token_ids = torch.tensor(data_gene_ref.var["token_id"].values)

        self.max_seq_len = max_seq_len
        self.aux_tokens = aux_tokens

    def tokenize_genes(self, X: torch.Tensor):
        # filter X based on gene_mask
        exp_matrix = X[:, self.gene_mask]
        token_ids = self.token_ids  # these are valid token ids

        # norm and scale to 10000 counts
        counts_per_cell = torch.mean(exp_matrix, dim=1)
        counts_per_cell += counts_per_cell == 0
        scaling_factor = 10_000 / counts_per_cell
        exp_matrix = exp_matrix * scaling_factor[:, None]

        tech_mean = self.technology_mean
        tech_mean = torch.nan_to_num(tech_mean)
        tech_mean += tech_mean == 0
        # print(tech_mean.shape)
        tech_mean = tech_mean[token_ids]
        # print(tech_mean.shape)
        # raise

        exp_matrix = exp_matrix / tech_mean[None, :]

        # initialize arrays to store sorted data
        sorted_exp = torch.zeros_like(exp_matrix)
        sorted_ids = torch.zeros_like(exp_matrix, dtype=int)

        # perform vectorized sorting
        sorted_idx = torch.argsort(-exp_matrix, dim=1)
        for idx in range(exp_matrix.shape[0]):
            sorted_exp[idx, :] = exp_matrix[idx, sorted_idx[idx, :]]
            sorted_ids[idx, :] = token_ids[sorted_idx[idx, :]] + self.aux_tokens

        return sorted_exp, sorted_ids

    def forward(self, X: torch.Tensor):
        """
        Given:
          X: Tensor of shape (N, 1370) with gene expression data ordered as in data_gene_ref.
          counts_per_cell: Tensor of shape (N,) with cell-specific total counts.

        This method expands X to shape (N, M) where M is len(concat_gene_names), placing
        values in the correct columns. Then, it normalizes the data.
        """
        sorted_exp, sorted_ids = self.tokenize_genes(X.squeeze())

        tokens_final = torch.zeros((sorted_ids.shape[0], self.max_seq_len), dtype=torch.long)

        # additional padding to make shape of X (N, max_seq_len) for each cell.
        for idx in range(sorted_ids.shape[0]):
            tokens = torch.tensor(sorted_ids[idx][: self.max_seq_len]).unsqueeze(0)
            padding = self.max_seq_len - tokens.shape[1]
            tokens = F.pad(tokens, (0, padding))
            tokens = tokens.to(torch.int)
            tokens_final[idx, :] = tokens

        return tokens_final


class NormalizeCounts(torch.nn.Module):
    """Normalizes gene expression counts per cell and applies log transformation."""

    def forward(self, X: torch.Tensor):
        # sum counts per row in X

        # dim needs to be 2, because X is of shape (N, 1, G) where N is the number of cells, 1 is just redundant, and G is the number of genes.
        # might have to change this later if the shape of X changes.
        counts_per_cell = torch.sum(X, dim=2)
        counts_greater_than_zero = counts_per_cell[counts_per_cell > 0]  # compute counts_greater_that_zero for the X.
        after, _ = torch.median(counts_greater_than_zero, dim=0)
        counts_per_cell += counts_per_cell == 0  # Avoid division by zero
        counts_per_cell = counts_per_cell / after
        normalized = torch.div(X, counts_per_cell[:, None])
        return torch.log1p(normalized)


class LogNormOnly(torch.nn.Module):
    """Normalizes gene expression counts per cell and applies log transformation."""

    def forward(self, X: torch.Tensor):
        return torch.log1p(X)


class ApplyTransforms(torch.nn.Module):
    def __init__(self, transforms: dict):
        super().__init__()

        self.img_transform = transforms["img_transform"]
        self.gene_transform = transforms["gene_transform"]

    def forward(self, x):
        for i in ENUMS:
            x[i] = torch.tensor(x[i]).contiguous()

        if self.img_transform:
            x[DatasetEnum.IMG] = [img.contiguous() for img in self.img_transform(x[DatasetEnum.IMG])]
            x[DatasetEnum.IMG] = torch.stack(x[DatasetEnum.IMG]).contiguous()

        if self.gene_transform:
            # Current general rule! x[DatasetEnum.GEXP] is always a tensor of shape (N, 1, G) where N is the number of cells, 1 is just redundant, and G is the number of genes.
            # We need to convert it to a tensor of shape (N, G) where N is the number of cells and G is the number of genes.
            # This is currently required because drvi uses (N, 1, G) shape for gene expression data.
            # Every other model uses (N, G) shape for gene expression data.

            x[DatasetEnum.GEXP] = self.gene_transform(x[DatasetEnum.GEXP])
            x[DatasetEnum.GEXP] = x[DatasetEnum.GEXP].contiguous()

        return x


# Predefined transformations for different models
TRANSFORMS = {
    "default": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "conch": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512)]),
            T.Resize((480, 480), antialias=True, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((480, 480)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    ),
    "optimus": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ]
    ),
    "h0-mini": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ]
    ),
    "uni": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "ctranspath": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "gigapath": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "augment": T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.GaussianBlur((5, 9), (0.1, 5.0)),
            T.RandomAdjustSharpness(2),
        ]
    ),
    "default_gene": NormalizeCounts(),
    "log1p_only": LogNormOnly(),
    "nicheformer": NicheformerTransform,
    "scFoundation": scFoundationTransform,
}


class ImageGexpDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        data_gene_reference_path: Path,
        img_model_name: Literal["ctranspath", "densenet", "uni", "optimus", "conch", "gigapath", "h0-mini"] | str,
        gene_model_name: Literal["drvi", "nicheformer", "scfoundation", "generic"] | str,
        source_key: str = "source",
        source_value=None,
        batch_size: int = 256,
        persistent_workers=True,
        pin_memory=True,
        num_workers: int = 4,
        seed: int = 42,
        augment: bool = True,
        split_key: str = None,
        split_train_csv: str = None,
        split_val_csv: str = None,
        split_test_csv: str = None,  # This is mandatory! Cannot skip!
        **kwargs: Any,
    ):
        """
        Initializes the ImageGexpDataModule for handling image and gene expression datasets.

        Args:
            dataset_path (Path): Path to the dataset directory.
            img_model_name (str): Name of the Image model to determine the transformation pipeline. Defaults to "default".
            gene_model_name (str): Name of the Gene model to determine the transformation for genes. Defaults to "drvi".
            source_key (str, optional): Key in the dataset to filter data. Defaults to "source".
            source_value (Any, optional): Value for filtering data based on the `source_key`. Defaults to None.
            batch_size (int, optional): Number of samples per batch for DataLoaders. Defaults to 256.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            augment (bool, optional): Whether to apply data augmentation on the training set. Defaults to True.
            split_test_key (str, optional): Key in the dataset used for splitting test data. Defaults to None.
            split_test_value (Any, optional): Value for filtering test data based on the `split_test_key`. Defaults to None.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.data_gene_reference_path = data_gene_reference_path
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.seed = seed
        self.source_key = source_key
        self.source_value = source_value

        self.split_key = split_key
        self.split_train_csv = split_train_csv
        self.split_val_csv = split_val_csv
        self.split_test_csv = split_test_csv

        # Configure transformations
        self.img_transform = (
            TRANSFORMS[img_model_name] if img_model_name in TRANSFORMS.keys() else TRANSFORMS["default"]
        )
        self.img_transform_augment = (
            T.Compose([self.img_transform, TRANSFORMS["augment"]]) if augment else self.img_transform
        )

        # Load gene model
        if gene_model_name in ["nicheformer"]:
            self.gene_transform = TRANSFORMS["nicheformer"](
                nicheformer_reference_path="/p/project1/hai_spatial_clip/pretrain_weights/gene/nicheformer/nicheformer_reference.h5ad",
                technology_mean_path="/p/project1/hai_spatial_clip/pretrain_weights/gene/nicheformer/xenium_mean_script.npy",
                data_gene_reference_path=self.data_gene_reference_path,
                max_seq_len=1500,
                aux_tokens=30,
            )

        elif gene_model_name in ["generic"]:
            self.gene_transform = TRANSFORMS["default_gene"]

        elif gene_model_name in ["drvi"]:
            self.gene_transform = TRANSFORMS["log1p_only"]

        elif gene_model_name in ["scFoundation"]:
            self.gene_transform = TRANSFORMS["scFoundation"](
                scFoundation_gene_index_path="/p/project1/hai_spatial_clip/pretrain_weights/gene/scFoundation/OS_scRNA_gene_index.19264.tsv",
                data_gene_reference_path=self.data_gene_reference_path,
            )

        # Load dataset
        dataset = load_from_disk(dataset_path)
        self.idx_all = self._filter_dataset_indices(dataset, source_key, source_value)

    def _filter_dataset_indices(self, dataset, key, value):
        """Filters dataset indices based on key and value."""
        universe = np.array(dataset[key])
        target = dataset.features[key].str2int(value)
        idx_target = np.where(np.isin(universe, target))[0]
        return idx_target

    def _split_indices(self):
        """Splits dataset into train, validation, and test indices."""
        if self.split_key and self.split_test_csv and self.split_train_csv and self.split_val_csv:
            universe = np.array(self.dataset[self.split_key])

            train_ids = pd.read_csv(self.split_train_csv)["id"].values.tolist()
            train = self.dataset.features[self.split_key].str2int(train_ids)
            idx_train = np.where(np.isin(universe, train))[0]

            val_ids = pd.read_csv(self.split_val_csv)["id"].values.tolist()
            val = self.dataset.features[self.split_key].str2int(val_ids)
            idx_val = np.where(np.isin(universe, val))[0]

            test_ids = pd.read_csv(self.split_test_csv)["id"].values.tolist()
            test = self.dataset.features[self.split_key].str2int(test_ids)
            idx_test = np.where(np.isin(universe, test))[0]

            return idx_train, idx_val, idx_test

        else:
            return None, None, None

    def setup(self, stage=None):
        """Prepares train, validation, and test datasets."""
        self.dataset = load_from_disk(self.dataset_path)
        self.dataset = self.dataset.select_columns(SELECTION)
        self.dataset = self.dataset.select(self.idx_all)
        idx_train, idx_val, idx_test = self._split_indices()
        assert idx_train is not None, "Train split is not defined."
        assert idx_val is not None, "Validation split is not defined."
        assert idx_test is not None, "Test split is not defined."

        self.test_dataset = self.dataset.select(idx_test)
        self.train_dataset = self.dataset.select(idx_train)
        self.val_dataset = self.dataset.select(idx_val)

        self._apply_transforms()

    def _apply_transforms(self):
        """Applies transforms to train, validation, and test datasets."""
        self.train_dataset.set_transform(
            ApplyTransforms({"img_transform": self.img_transform_augment, "gene_transform": self.gene_transform})
        )
        self.val_dataset.set_transform(
            ApplyTransforms({"img_transform": self.img_transform, "gene_transform": self.gene_transform})
        )
        if hasattr(self, "test_dataset"):
            self.test_dataset.set_transform(
                ApplyTransforms({"img_transform": self.img_transform, "gene_transform": self.gene_transform})
            )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_dataset, shuffle=True, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.val_dataset, shuffle=False, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return self._create_dataloader(
            self.test_dataset, shuffle=False, pin_memory=self.pin_memory, persistent_workers=False
        )

    def predict_dataloader(self):
        return self._create_dataloader(self.dataset, shuffle=False, pin_memory=False, persistent_workers=False)

    def _create_dataloader(self, dataset, shuffle, pin_memory=True, persistent_workers=False):
        """Creates a DataLoader for the given dataset."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )


if __name__ == "__main__":
    # dataset_path = "/mnt/volume/cligen-dir/hfdataset/image_gexp_counts.dataset"
    hf_path = Path("/p/project1/hai_spatial_clip/data/xenium/encoded_files")
    dataset_path = str(hf_path / "paired_ts8_human_breast_panel.dataset")

    data_gene_reference_path = Path(
        "/p/project1/hai_spatial_clip/hescape/data/human_breast_panel/nicheformer_reference.h5ad"
    )

    split_path = Path("/p/project1/hai_spatial_clip/hescape/data/human_breast_panel")
    split_train_csv = str(split_path / "train.csv")
    split_val_csv = str(split_path / "val.csv")
    split_test_csv = str(split_path / "test.csv")

    dm = ImageGexpDataModule(
        dataset_path=dataset_path,
        data_gene_reference_path=data_gene_reference_path,
        img_model_name="h0-mini",
        gene_model_name="scFoundation",
        source_key="tissue",
        source_value=["breast"],
        batch_size=5,
        num_workers=4,
        seed=42,
        augment=True,
        split_key="name",
        split_train_csv=split_train_csv,
        split_val_csv=split_val_csv,
        split_test_csv=split_test_csv,
    )
    dm.prepare_data()
    dm.setup()

    # print(len(dm.train_dataset.features["id"].names))
    # print(dm.val_dataset)
    # print(dm.test_dataset)
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    print(dm.test_dataset)

    for idx, batch in enumerate(train_loader):
        # print(f"Batch keys: {batch.keys()}")
        # print(f"Gexp dims: {batch[DatasetEnum.GEXP].shape}")
        print(f"Gexp dims: {batch[DatasetEnum.GEXP].shape}")
        # save image to file
        # torchvision.utils.save_image(batch[DatasetEnum.IMG][0], f"img_{idx}.png")
        # print(f"Batch size: {len(batch)}")
        # print(f"Gene exp: {batch[DatasetEnum.GEXP]}")
        # print how many non-zero values exist in each row of batch[DatasetEnum.GEXP]
        # print(f"Non-zeros per gene expression vector: {torch.count_nonzero(batch[DatasetEnum.GEXP], dim=1)}")

        if idx == 10:
            break
