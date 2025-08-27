from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Literal

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from hescape._logging import logger
from hescape.models import CLIPModel


def world_info_from_env():
    # from https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


class ClampCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.model.logit_scale.clamp_(min=0, max=math.log(100))


class PretrainModule(LightningModule):
    def __init__(
        self,
        input_genes: int,
        embed_dim: int,
        img_enc_name: Literal["ctranspath", "uni", "conch", "optimus", "densenet", "gigapath", "lunit"],
        gene_enc_name: Literal["drvi", "nicheformer", "uce", "scfoundation", "generic"],
        loss: Literal["CLIP", "SIGLIP"],
        img_finetune: bool,
        gene_finetune: bool,
        img_proj: Literal["linear", "mlp", "transformer"],
        gene_proj: Literal["linear", "mlp", "identity"],
        n_tissue: int,
        n_region: int,
        image_size: int,
        temperature: float,
        lr: float,
        weight_decay: float,
        cfg: DictConfig,
        lambda_scheduler: Callable | None,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        local_rank, global_rank, world_size = world_info_from_env()

        logger.info(f"LOCAL RANK: {local_rank}")
        logger.info(f"GLOBAL RANK: {global_rank}")
        logger.info(f"WORLD SIZE: {world_size}")

        if torch.cuda.is_available():
            logger.info(f"CUDA DEVICE NAME: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"CUDA DEVICE INDEX: {torch.cuda.current_device()}")
            logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")
        else:
            logger.info("CUDA is not available.")

        self.model = CLIPModel(
            input_genes=input_genes,
            embed_dim=embed_dim,
            img_enc_name=img_enc_name,
            loss=loss,
            img_finetune=img_finetune,
            gene_finetune=gene_finetune,
            gene_enc_name=gene_enc_name,
            image_size=image_size,
            n_tissue=n_tissue,
            n_region=n_region,
            temperature=temperature,
            world_size=world_size,
            rank=local_rank,
            img_enc_path=self.cfg.paths.pretrain_weights.img_enc_path,
            gene_enc_path=self.cfg.paths.pretrain_weights.gene_enc_path,
            drvi_model_dir=self.cfg.paths.anatomy.pretrain_weights.drvi_model_dir,
            cfg=cfg,
            img_proj=img_proj,
            gene_proj=gene_proj,
        )
        self.lambda_scheduler: Callable[[int], float] | None = lambda_scheduler
        self.lr = lr
        self.weight_decay = weight_decay

        # evaluations
        self.eval_outputs = {"val": [], "test": []}
        self.strategy = self.cfg.training.lightning.trainer.strategy
        self.eval_batch_key = self.cfg.training.evaluations.batch_key
        self.eval_label_key = self.cfg.training.evaluations.label_key

    def configure_optimizers(self):
        params = []
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)

        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), lr=self.lr, weight_decay=self.weight_decay)
        if self.lambda_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lambda_scheduler)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        loss, metrics = self.shared_step(batch, "train")
        loss_params = {}
        logit_bias = getattr(self.model, "logit_bias", None)
        if logit_bias is not None:
            loss_params["logit_bias"] = logit_bias.item()
        loss_params["logit_scale"] = self.model.logit_scale.item()
        metrics.update(loss_params)
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        _, metrics = self.shared_step(batch, "val")
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return metrics

    def test_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        _, metrics = self.shared_step(batch, "test")
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return metrics

    def shared_step(self, batch, stage: str):
        img_embed, gexp_embed, logit_scale = self.model(batch, norm=False)
        # For batch "ID" and "GEXP"
        # source_ids = batch[DatasetEnum.ID].to(torch.int64)
        # source_exp = batch[DatasetEnum.GEXP].to(img_embed.dtype)

        contrastive_loss = self.model.compute_loss(  # recon_loss, cls_loss
            img_embed=img_embed, gexp_embed=gexp_embed
        )

        metrics = {f"{stage}_loss": contrastive_loss}

        knn_recall_i2g = get_clip_metrics(img_embed, gexp_embed, logit_scale=logit_scale, stage=stage)
        metrics.update(knn_recall_i2g)

        return contrastive_loss, metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch)


def get_clip_metrics(image_features, gene_features, logit_scale, stage: str):
    metrics = {}
    logits_per_image = logit_scale * image_features @ gene_features.T
    logits_per_gene = logits_per_image.T

    logits = {"image_to_gene": logits_per_image, "gene_to_image": logits_per_gene}
    ground_truth = torch.arange(len(gene_features), device=gene_features.device).contiguous().view(-1, 1)

    # metrics for +ve and -ve pairs
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.float()
        metrics[f"{stage}/{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{stage}/{name}_median_rank"] = torch.floor(torch.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{stage}/{name}_R@{k}"] = torch.mean((preds < k).float())

    return metrics
