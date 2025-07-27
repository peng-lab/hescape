# takes all different image encoders in a single class function
from __future__ import annotations

import warnings

# filter all User and FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# from cellnet.tabnet.tab_network import TabNet
from peft import LoraConfig, get_peft_model
from timm.layers.mlp import Mlp

from hescape.models._utils import print_trainable_parameters
from hescape.models.gexp_models._drvi import _build_drvi_model
from hescape.models.gexp_models._nicheformer import _build_nicheformer_model
from hescape.models.gexp_models._scfoundation import _build_scfoundation_model
from hescape.models.gexp_models._utils import freeze_batch_norm_2d


# takes all gexp encoders in a single class
# onehot taken from https://github.com/scverse/scvi-tools/blob/1.0.2/scvi/nn/_utils.py#L4
def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


class GexpEncoder(nn.Module):
    """ImageEncoder that wraps timm models."""

    def __init__(
        self,
        input_genes: int,
        model_name: str | None = None,
        n_region: int | None = None,
        n_tissue: int | None = None,
        embed_dim: int = -1,
        finetune: bool = False,
        proj: str = "identity",
        **kwargs: Any,
    ):
        super().__init__()

        checkpoint_root = Path(kwargs.get("checkpoint_path", None))
        self.drvi_model_dir = kwargs.get("drvi_model_dir", None)

        self.input_genes = input_genes
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.n_region = n_region
        self.n_tissue = n_tissue
        self.proj = proj
        self.trunk, num_features = self._build_trunk(self.model_name, checkpoint_root, **kwargs)

        if not finetune:
            self.freeze()

        if finetune:
            self.trunk = self.get_ft_model(model_name, self.trunk, lora=finetune)

        self.head = self._build_head(proj, num_features, embed_dim)

    def _build_trunk(self, model_name: str, checkpoint_root: Path, **kwargs: Any) -> nn.Module:
        """Build the trunk."""
        """

        elif model_name == "scfoundation":
            # fetch pretrain weights for nicheformer
            trunk = nn.Identity()  ## TBD
            num_features = 512

         """

        if model_name == "nicheformer":
            checkpoint_path = checkpoint_root / model_name / "nicheformer.ckpt"
            trunk = _build_nicheformer_model(checkpoint_path)
            num_features = trunk.num_features
            print(f"Successfully loaded weights for {model_name}")

        elif model_name == "drvi":
            checkpoint_path = checkpoint_root / self.drvi_model_dir
            trunk = _build_drvi_model(checkpoint_path)

            num_features = 128
            print(f"Successfully loaded weights for {model_name}")

        elif model_name == "scFoundation":
            checkpoint_path = checkpoint_root / model_name / "scFoundation.ckpt"
            trunk = _build_scfoundation_model(checkpoint_path)
            trunk.norm = torch.nn.BatchNorm1d(trunk.model_config["encoder"]["hidden_dim"], affine=False, eps=1e-6)
            num_features = trunk.model_config["encoder"]["hidden_dim"]
            print(f"Successfully loaded weights for {model_name}")

        elif model_name == "generic":
            trunk = GenericEncoder(num_input=self.input_genes, num_layer=3, embed_dim=self.embed_dim)
            num_features = self.embed_dim

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if self.n_region is not None:
            self.region_onehot = partial(one_hot, n_cat=self.n_region)
            num_features += self.n_region
        else:
            self.region_onehot = None

        if self.n_tissue is not None:
            self.tissue_onehot = partial(one_hot, n_cat=self.n_tissue)
            num_features += self.n_tissue
        else:
            self.tissue_onehot = None

        return trunk, num_features

    def get_ft_model(self, model_name: str, trunk, lora: bool = False) -> object:
        """
        Returns a fine-tuned model based on the given model name and trunk.

        Args:
            model_name (str): The name of the model.
            trunk: The trunk model.
            lora (bool, optional): Whether to use LoRA. Defaults to False.

        Returns
        -------
            object: The fine-tuned model.
        """
        # Define LoRA configurations for each model
        lora_configs = {
            "nicheformer": {"r": 8, "lora_alpha": 16, "target_modules": ["self_attn.out_proj", "linear1", "linear2"]},
            # "scfoundation": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
        }

        if lora:
            # Get the LoRA configuration for the given model
            if model_name in lora_configs.keys():
                config = lora_configs.get(model_name)
                if config:
                    # Create a LoRA configuration object
                    lora_config = LoraConfig(
                        r=config["r"],
                        lora_alpha=config["lora_alpha"],
                        target_modules=config["target_modules"],
                        lora_dropout=0.1,
                        bias="none",
                    )
                    # Return the fine-tuned model with LoRA
                    return get_peft_model(trunk, lora_config)
                else:
                    # Handle unknown model names
                    raise ValueError(f"Unknown model name: {model_name}")

        # If LoRA is not enabled, if it's an scFoundation model,
        if model_name == "scFoundation":
            for _, p in trunk.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in trunk.pos_emb.named_parameters():
                p.requires_grad = False

            for na, param in trunk.encoder.named_parameters():
                param.requires_grad = False
            for na, param in trunk.encoder.transformer_encoder[-2].named_parameters():
                # print('trunk.encoder.transformer_encoder ',na,' have grad')
                param.requires_grad = True

        return trunk  # it simply returns the trunk for generic and drvi. finetunes final layers for scFoundation and lora ft for nicheformer

    def _build_head(self, proj: str, in_features: int, embed_dim: int) -> nn.Sequential:
        """Build a projection head (Linear or MLP)."""
        head_layers = OrderedDict()
        if proj == "linear":
            head_layers["proj"] = nn.Linear(in_features, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(in_features, 2 * embed_dim, embed_dim, drop=0.2, norm_layer=nn.LayerNorm)
        elif proj == "identity":
            head_layers["proj"] = nn.Identity()
        else:
            raise ValueError(f"Unknown projection type: {proj}")

        return nn.Sequential(head_layers)

    def freeze(self, freeze_bn_stats=True):
        if self.model_name != "generic":
            """Freeze model params."""
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)

    def forward(self, x, region=None, tissue=None):
        """Forward pass."""
        x = self.trunk(x)
        if self.region_onehot is not None:
            region = self.region_onehot(region[:, None])
            x = torch.cat([x, region], dim=1)
        if self.tissue_onehot is not None:
            tissue = self.tissue_onehot(tissue[:, None])
            x = torch.cat([x, tissue], dim=1)
        x = self.head(x)
        return x


class GenericEncoder(nn.Module):
    def __init__(
        self,
        num_input: int,
        num_layer: int,
        embed_dim: int,
        **kwargs: Any,
    ):
        super().__init__()
        self.nn = nn.Sequential()
        self.nn.add_module("input", nn.Linear(int(num_input), int(embed_dim)))
        for i in range(num_layer):
            self.nn.add_module(f"batchnorm_{i}", nn.BatchNorm1d(embed_dim))
            self.nn.add_module(f"relu_{i}", nn.ReLU())
            self.nn.add_module(f"dropout_{i}", nn.Dropout(p=0.1))
            self.nn.add_module(f"linear_{i}", nn.Linear(embed_dim, embed_dim))
        self.nn.add_module("batchnorm_out", nn.BatchNorm1d(embed_dim))
        self.nn.add_module("relu_out", nn.ReLU())

    def forward(self, x):
        return self.nn(x.squeeze())


if __name__ == "__main__":
    # Create an instance of the ImageEncoder class
    encoder = GexpEncoder(
        input_genes=5001,
        model_name="scFoundation",
        embed_dim=128,
        finetune=True,
        proj="mlp",
        checkpoint_path="/p/project1/hai_spatial_clip/pretrain_weights/gene",
        n_tissue=None,
        n_region=None,
        # drvi_model_dir="drvi_human_5k_panel",
    )

    # encoder.freeze()

    encoder = encoder.to("cuda")
    dummy_input = torch.Tensor(8, 19266).uniform_().to("cuda")  # .to(torch.int)
    output = encoder(dummy_input)
    print(output.shape)  # Output shape: [batch_size, num_features]

    # print parameter names which have gradient set to True
    # for name, param in encoder.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # for name, module in encoder.named_modules():
    #     print(name)

    print_trainable_parameters("drvi", encoder)
