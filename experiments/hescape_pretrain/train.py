import warnings

# os.environ["NCCL_P2P_LEVEL"] = "PIX"
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OmegaConf.register_new_resolver("eval", eval)
import faulthandler

from pytorch_lightning import seed_everything

faulthandler.enable()


def train(cfg: DictConfig) -> None:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
    from pytorch_lightning.loggers import Logger

    from hescape.modules.pretrain_module import ClampCallback

    torch.set_float32_matmul_precision("medium")
    from hescape._logging import logger as hescape_logger

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        hescape_logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            hescape_logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        hescape_logger.info("No GPUs available.")

    if cfg.datamodule.get("seed"):
        pl.seed_everything(cfg.datamodule.seed, workers=True)

    hescape_logger.info("Instantiating datamodule...")
    dm: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    hescape_logger.info("Instantiating model...")
    lr_lambda = hydra.utils.instantiate(cfg.model.cosine_scheduler)
    model: LightningModule = hydra.utils.instantiate(cfg.model.litmodule)
    model = model(lambda_scheduler=lr_lambda, cfg=cfg)

    hescape_logger.info("Instantiating callbacks and logger...")
    callbacks: list[Callback] = []
    for _, cb in cfg.training.callbacks.items():
        callbacks.append(hydra.utils.instantiate(cb))
    callbacks.append(ClampCallback())

    logger: list[Logger] = []
    for name, lg in cfg.training.logger.items():
        lgr = hydra.utils.instantiate(lg)

        # if name == "wandb":
        #     metadata = {
        #         "img_enc_name": cfg.model.litmodule.img_enc_name,
        #         "gene_enc_name": cfg.model.litmodule.gene_enc_name,
        #         "img_proj": cfg.model.litmodule.img_proj,
        #         "img_finetune": cfg.model.litmodule.img_finetune,
        #         "gene_proj": cfg.model.litmodule.gene_proj,
        #         "gene_finetune": cfg.model.litmodule.gene_finetune,
        #         "loss": cfg.model.litmodule.loss,
        #         "seed": cfg.datamodule.seed,
        #         "panel": cfg.name,
        #     }
        #     print(metadata)

        #     @rank_zero_only
        #     def update_wandb_config(logger, metadata):
        #         logger.experiment.config.update(metadata)

        #     update_wandb_config(lgr, metadata)

        logger.append(lgr)

    hescape_logger.info("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(cfg.training.lightning.trainer)
    trainer = trainer(callbacks=callbacks, logger=logger)
    if cfg.training.train:
        hescape_logger.info("Training...")
        trainer.fit(
            model,
            # datamodule=dm,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        hescape_logger.info("Testing...")
        trainer.test(
            model,
            verbose=True,
            dataloaders=test_loader,
        )


@hydra.main(config_path="./../configs", config_name="breast_clip_pretrain", version_base="1.2")
def main(cfg: DictConfig) -> None:
    from rich.pretty import pprint

    sweep_params = HydraConfig.get().job.override_dirname

    img_enc_name = cfg.model.litmodule.img_enc_name
    gene_enc_name = cfg.model.litmodule.gene_enc_name
    img_proj = cfg.model.litmodule.img_proj
    img_finetune = cfg.model.litmodule.img_finetune
    gene_proj = cfg.model.litmodule.gene_proj  # Mostly fixed to linear
    gene_finetune = cfg.model.litmodule.gene_finetune

    seed = cfg.datamodule.seed
    # batch_size = cfg.datamodule.batch_size
    loss = cfg.model.litmodule.loss

    # set seed for reproducibility
    seed_everything(seed, workers=True)

    try:
        job_id = f"{HydraConfig.get().job.id}"  # _{HydraConfig.get().job.num}"
    except Exception:
        job_id = "local"
    wandb_name = f"{job_id}"  # -{img_enc_name}-{img_proj}-{gene_enc_name}-{gene_proj}-{seed}-{loss}"
    cfg.training.logger.wandb.name = wandb_name
    cfg.paths.anatomy.output = f"{cfg.paths.anatomy.output}/{wandb_name}"

    pprint(OmegaConf.to_container(cfg, resolve=True))
    train(cfg)

    print(f"SWEEP PARAMS {sweep_params}, {cfg.paths.anatomy.output}")
    print(f"modelcheckpoint dirpath: {cfg.training.callbacks.model_checkpoint.dirpath}")
    print(f"csv logger name: {cfg.training.logger.csv.save_dir}")
    # print(f"encoder_path: {cfg.model.litmodule.encoder_path}")

    # os.makedirs(cfg.training.logger.wandb.save_dir, exist_ok=True)


if __name__ == "__main__":
    main()
