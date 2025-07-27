import os
import pickle
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scanpy as sc
import torch
from eval_data import MultiDatasetSentenceCollator, MultiDatasetSentences
from scanpy import AnnData
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def figshare_download(url, save_path):
    """
    Figshare download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    if os.path.exists(save_path):
        return
    else:
        # Check if directory exists
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        print("Downloading " + save_path + " from " + url + " ..." + "\n")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    # If the downloaded filename ends in tar.gz then extraact it
    if save_path.endswith(".tar.gz"):
        with tarfile.open(save_path) as tar:
            tar.extractall(path=os.path.dirname(save_path))
            print("Done!")


def get_species_to_pe(EMBEDDING_DIR):
    """
    Given an embedding directory, return all embeddings as a dictionary coded by species.
    Note: In the current form, this function is written such that the directory needs all of the following species embeddings.
    """
    EMBEDDING_DIR = Path(EMBEDDING_DIR)

    embeddings_paths = {
        "human": EMBEDDING_DIR / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        "mouse": EMBEDDING_DIR / "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
        "frog": EMBEDDING_DIR / "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
        "zebrafish": EMBEDDING_DIR / "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
        "mouse_lemur": EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": EMBEDDING_DIR / "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
        "macaca_fascicularis": EMBEDDING_DIR
        / "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
        "macaca_mulatta": EMBEDDING_DIR / "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
    }

    species_to_pe = {species: torch.load(pe_dir) for species, pe_dir in embeddings_paths.items()}

    species_to_pe = {species: {k.upper(): v for k, v in pe.items()} for species, pe in species_to_pe.items()}
    return species_to_pe


def get_spec_chrom_csv(path="/dfs/project/cross-species/yanay/code/all_to_chrom_pos.csv"):
    """Get the species to chrom csv file"""
    gene_to_chrom_pos = pd.read_csv(path)
    gene_to_chrom_pos["spec_chrom"] = pd.Categorical(
        gene_to_chrom_pos["species"] + "_" + gene_to_chrom_pos["chromosome"]
    )  # add the spec_chrom list
    return gene_to_chrom_pos


def adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset):
    """Given a :path: to an h5ad"""
    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata.var_names]).long()
    print(len(np.unique(pe_row_idxs)))

    spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")

    gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]

    dataset_chroms = gene_chrom["spec_chrom"].cat.codes  # now this is correctely indexed by species and chromosome
    print("Max Code:", max(dataset_chroms))
    dataset_pos = gene_chrom["start"].values
    return pe_row_idxs, dataset_chroms, dataset_pos


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


def load_gene_embeddings_adata(
    adata: AnnData, protein_embeddings_dir: Path, species: list, embedding_model: str
) -> tuple[AnnData, dict[str, torch.FloatTensor]]:
    """Loads gene embeddings for all the species/genes in the provided data.

    :param data: An AnnData object containing gene expression data for cells.
    :param species: Species corresponding to this adata
    :param embedding_model: The gene embedding model whose embeddings will be loaded.
    :return: A tuple containing:
               - A subset of the data only containing the gene expression for genes with embeddings in all species.
               - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim).
    """
    EMBEDDING_DIR = Path(protein_embeddings_dir)

    MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH = {
        "ESM2": {
            "human": EMBEDDING_DIR / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
            "mouse": EMBEDDING_DIR / "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
            "frog": EMBEDDING_DIR / "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
            "zebrafish": EMBEDDING_DIR / "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
            "mouse_lemur": EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
            "pig": EMBEDDING_DIR / "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
            "macaca_fascicularis": EMBEDDING_DIR
            / "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
            "macaca_mulatta": EMBEDDING_DIR / "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
        }
    }

    # Get species names
    species_names = species
    species_names_set = set(species_names)

    # Get embedding paths for the model
    species_to_gene_embedding_path = MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH[embedding_model]
    available_species = set(species_to_gene_embedding_path)

    # Ensure embeddings are available for all species
    if not (species_names_set <= available_species):
        raise ValueError(f"The following species do not have gene embeddings: {species_names_set - available_species}")

    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(species_to_gene_embedding_path[species]).items()
        }
        for species in species_names
    }

    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(
        *[set(gene_symbol_to_embedding) for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()]
    )
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}

    # Subset data to only use genes with embeddings
    adata = adata[:, adata.var_names.isin(genes_to_use)]

    return adata


def anndata_to_sc_dataset(
    adata: sc.AnnData,
    protein_embeddings_dir: Path,
    species: str = "human",
    labels: list = [],
    hv_genes=None,
    embedding_model="ESM2",
) -> AnnData:
    # Subset to just genes we have embeddings for
    adata = load_gene_embeddings_adata(
        adata=adata, protein_embeddings_dir=protein_embeddings_dir, species=[species], embedding_model=embedding_model
    )

    if hv_genes is not None:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hv_genes)  # Expects Count Data

        hv_index = adata.var["highly_variable"]
        adata = adata[:, hv_index]  # Subset to hv genes only

    if len(labels) > 0:
        labels = adata.obs.loc[:, labels].values

    return adata


def process_raw_anndata(row, h5_folder_path, npz_folder_path, protein_embeddings_dir, skip, additional_filter, root):
    path = row.path
    if not os.path.isfile(root + "/" + path):
        print("**********************************")
        print(f"***********{root + '/' + path} File Missing****")
        print("**********************************")
        print(path, root)
        return None

    name = path.replace(".h5ad", "")
    proc_path = path.replace(".h5ad", "_proc.h5ad")
    if skip:
        if os.path.isfile(h5_folder_path + proc_path):
            print(f"{name} already processed. Skipping")
            return None, None, None

    print(f"Proccessing {name}")

    species = row.species

    ad = sc.read(root + "/" + path)
    labels = []
    if "cell_type" in ad.obs.columns:
        labels.append("cell_type")

    if additional_filter:
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.filter_cells(ad, min_genes=25)

    # edit func anndata_to_sc_dataset to only give modified adata. Remove SingelCellDataset
    adata = anndata_to_sc_dataset(ad, protein_embeddings_dir, species=species, labels=labels, hv_genes=None)
    adata = adata.copy()

    if additional_filter:
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.filter_cells(ad, min_genes=25)

    num_cells = adata.X.shape[0]
    num_genes = adata.X.shape[1]

    adata_path = h5_folder_path + proc_path
    adata.write(adata_path)

    arr = data_to_torch_X(adata.X).numpy()

    print("this is a nice check to make sure it's counts", arr.max())  # this is a nice check to make sure it's counts
    filename = npz_folder_path + f"{name}_counts.npz"
    shape = arr.shape
    print(name, shape)
    fp = np.memmap(filename, dtype="int64", mode="w+", shape=shape)
    fp[:] = arr[:]
    fp.flush()

    return adata, num_cells, num_genes


def get_ESM2_embeddings(args):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(args.token_file)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
        # 1895 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS))  # Add the chrom tensors to the end
        all_pe.requires_grad = False

    return all_pe


class AnndataProcessor:
    def __init__(self, args):
        self.args = args
        self.h5_folder_path = self.args.dir
        self.npz_folder_path = self.args.dir
        self.scp = ""

        # Check if paths exist, if not, create them
        self.check_paths()

        # Set up the anndata
        self.adata_name = self.args.adata_path.split("/")[-1]
        self.adata_root_path = self.args.adata_path.replace(self.adata_name, "")
        self.name = self.adata_name.replace(".h5ad", "")
        self.proc_h5_path = self.h5_folder_path + f"{self.name}_proc.h5ad"
        self.adata = None

        # Set up the row
        row = pd.Series()
        row.path = self.adata_name
        row.species = self.args.species
        self.row = row

        # Set paths once to be used throughout the class
        os.makedirs(self.args.dir, exist_ok=True)
        self.pe_idx_path = self.args.dir + "/" + f"{self.name}_pe_idx.torch"
        self.chroms_path = self.args.dir + "/" + f"{self.name}_chroms.pkl"
        self.starts_path = self.args.dir + "/" + f"{self.name}_starts.pkl"
        self.shapes_dict_path = self.args.dir + "/" + f"{self.name}_shapes_dict.pkl"

    def check_paths(self):
        """Check if the paths exist, if not, create them"""
        figshare_download("https://figshare.com/ndownloader/files/42706558", self.args.spec_chrom_csv_path)
        figshare_download("https://figshare.com/ndownloader/files/42706555", self.args.offset_pkl_path)
        if not os.path.exists(self.args.protein_embeddings_dir):
            figshare_download(
                "https://figshare.com/ndownloader/files/42715213", "model_files/protein_embeddings.tar.gz"
            )
        figshare_download("https://figshare.com/ndownloader/files/42706585", self.args.token_file)
        if self.args.adata_path is None:
            print("Using sample AnnData: 10k pbmcs dataset")
            self.args.adata_path = "./data/10k_pbmcs_proc.h5ad"
            figshare_download("https://figshare.com/ndownloader/files/42706966", self.args.adata_path)
        if self.args.model_loc is None:
            print("Using sample 4 layer model")
            self.args.model_loc = "./model_files/4layer_model.torch"
            figshare_download("https://figshare.com/ndownloader/files/42706576", self.args.model_loc)

    def save_shapes_dict(self, name, num_cells, num_genes, shapes_dict_path):
        shapes_dict = {name: (num_cells, num_genes)}
        with open(shapes_dict_path, "wb+") as f:
            pickle.dump(shapes_dict, f)
            print("Wrote Shapes Dict")

    def preprocess_anndata(self):
        self.adata, num_cells, num_genes = process_raw_anndata(
            self.row,
            self.h5_folder_path,
            self.npz_folder_path,
            self.args.protein_embeddings_dir,
            self.args.skip,
            self.args.filter,
            root=self.adata_root_path,
        )
        if (num_cells is not None) and (num_genes is not None):
            self.save_shapes_dict(self.name, num_cells, num_genes, self.shapes_dict_path)

        if self.adata is None:
            self.adata = sc.read(self.proc_h5_path)

    def generate_idxs(self):
        if os.path.exists(self.pe_idx_path) and os.path.exists(self.chroms_path) and os.path.exists(self.starts_path):
            print("PE Idx, Chrom and Starts files already created")

        else:
            species_to_pe = get_species_to_pe(self.args.protein_embeddings_dir)
            with open(self.args.offset_pkl_path, "rb") as f:
                species_to_offsets = pickle.load(f)

            gene_to_chrom_pos = get_spec_chrom_csv(self.args.spec_chrom_csv_path)
            dataset_species = self.args.species
            spec_pe_genes = list(species_to_pe[dataset_species].keys())
            offset = species_to_offsets[dataset_species]
            pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
                self.adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset
            )

            # Save to the temp dict
            torch.save({self.name: pe_row_idxs}, self.pe_idx_path)
            with open(self.chroms_path, "wb+") as f:
                pickle.dump({self.name: dataset_chroms}, f)
            with open(self.starts_path, "wb+") as f:
                pickle.dump({self.name: dataset_pos}, f)

    def run_evaluation(self):
        with open(self.shapes_dict_path, "rb") as f:
            shapes_dict = pickle.load(f)
        run_eval(self.adata, self.name, self.pe_idx_path, self.chroms_path, self.starts_path, shapes_dict, self.args)


def run_eval(adata, name, pe_idx_path, chroms_path, starts_path, shapes_dict, args):
    # Load in the real token embeddings
    all_pe = get_ESM2_embeddings(args)
    pe_embedding = nn.Embedding.from_pretrained(all_pe)
    batch_size = args.batch_size

    #### Run the model ####
    # Dataloaders
    dataset = MultiDatasetSentences(
        sorted_dataset_names=[name],
        shapes_dict=shapes_dict,
        args=args,
        npzs_dir=args.dir,
        dataset_to_protein_embeddings_path=pe_idx_path,
        datasets_to_chroms_path=chroms_path,
        datasets_to_starts_path=starts_path,
    )
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=multi_dataset_sentence_collator, num_workers=0
    )
    pbar = tqdm(dataloader)
    with torch.no_grad():
        i = 0
        for batch in pbar:
            protein_token_indices, _, _ = batch[0], batch[1], batch[2]  # protein_token_indices, mask, idxs
            protein_token_indices = protein_token_indices.permute(1, 0)

            # load the protein embeddings for the indices only
            batch_sentences = pe_embedding(protein_token_indices.long())
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)  # Normalize token outputs now

            print(batch_sentences.shape)
            i += 1
            if i == 10:
                break
