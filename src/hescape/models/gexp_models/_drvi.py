import anndata as ad
import drvi
import torch


def _build_drvi_model(path: str):
    return DRVIModel(path)


class DRVIModel(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()
        adata = ad.read_h5ad(path / "drvi_reference.h5ad")
        self.trunk = drvi.model.DRVI.load(path / "drvi", adata=adata).module.z_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_m, q_v, latent = self.trunk(x, cat_full_tensor=None)

        return q_m.squeeze()
