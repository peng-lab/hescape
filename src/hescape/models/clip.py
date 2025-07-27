from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from open_clip.loss import ClipLoss, SigLipLoss
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef

from hescape.constants import DatasetEnum
from hescape.models._utils import print_trainable_parameters
from hescape.models.gexp_models import GexpEncoder
from hescape.models.image_models import ImageEncoder

LOCAL_RANK = "LOCAL_RANK"

REGRESSION_METRICS = [
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
]

CLASSIFICATION_METRICS = [
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
]


class CLIPModel(nn.Module):
    """CLIP model."""

    def __init__(
        self,
        input_genes: int,
        embed_dim: int,
        img_enc_name: Literal["ctranspath", "densenet", "uni", "optimus", "conch", "gigapath"],
        gene_enc_name: Literal["drvi", "nicheformer", "scfoundation", "uce", "generic"],
        loss: Literal["CLIP", "SIGLIP"],
        img_finetune: bool = False,
        gene_finetune: bool = False,
        n_tissue: int | None = None,
        n_region: int | None = None,
        image_size: int = 224,
        temperature: float = 0.07,
        world_size: int = 1,
        rank: int = 0,
        cfg: DictConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            model_name=img_enc_name,
            embed_dim=embed_dim,
            finetune=img_finetune,
            checkpoint_path=kwargs.get("img_enc_path", None),
            proj=kwargs.get("img_proj", "linear"),
        )

        self.gene_enc_name = gene_enc_name
        self.gexp_encoder = GexpEncoder(
            input_genes=input_genes,
            model_name=gene_enc_name,
            checkpoint_path=kwargs.get("gene_enc_path", None),
            drvi_model_dir=kwargs.get("drvi_model_dir", None),
            n_region=n_region,
            n_tissue=n_tissue,
            finetune=gene_finetune,  # (Always fine-tune gene encoder)
            embed_dim=embed_dim,
            proj=kwargs.get("gene_proj", "linear"),
            # idx_genes_target=cfg.paths.idx_genes_target,
        )

        print_trainable_parameters(img_enc_name, self.image_encoder)
        print_trainable_parameters(gene_enc_name, self.gexp_encoder)

        # ------------------------
        # 2) CLIP Loss Setup
        # ------------------------
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        if loss == "CLIP":
            self.logit_bias = None
            loss_fn = ClipLoss(world_size=world_size, rank=rank)
        elif loss == "SIGLIP":
            self.logit_bias = nn.Parameter(torch.ones([]) * -10)
            loss_fn = SigLipLoss(world_size=world_size, rank=rank)
        else:
            raise ValueError(f"Loss {loss} not supported.")
        self.loss = loss_fn

    def forward(self, batch: dict[str, Tensor], norm: bool = True):
        """Forward pass: returns (img_embed, gexp_embed, logit_scale.exp())."""
        gexp_encoder_input = batch[DatasetEnum.GEXP]
        region, tissue = None, None

        # Encode gene expressions
        gexp_embed = self.gexp_encoder(gexp_encoder_input, tissue, region)
        # Encode images
        img_embed = self.image_encoder(batch[DatasetEnum.IMG])

        if norm:
            return (
                F.normalize(img_embed, p=2, dim=-1),
                F.normalize(gexp_embed, p=2, dim=-1),
                self.logit_scale.exp(),
            )
        return img_embed, gexp_embed, self.logit_scale.exp()

    def compute_loss(self, img_embed, gexp_embed):  # default 1.0
        """
        Compute the total loss comprising:
          - Contrastive CLIP loss.
        """
        if self.logit_bias is not None:
            contrastive_loss = self.loss(
                img_embed, gexp_embed, logit_scale=self.logit_scale.exp(), logit_bias=self.logit_bias
            )
        else:
            contrastive_loss = self.loss(img_embed, gexp_embed, logit_scale=self.logit_scale.exp())

        return contrastive_loss
