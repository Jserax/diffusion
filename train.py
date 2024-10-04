import hydra
import wandb
from omegaconf import OmegaConf

with hydra.initialize(version_base="1.3", config_path="."):
    wandb.login()
    cfg = hydra.compose("train.yaml")
    model = hydra.utils.instantiate(cfg["model"])
    train_loader = hydra.utils.instantiate(cfg["train_loader"])
    callbacks = hydra.utils.instantiate(cfg.callbacks)
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model, train_loader)
    trainer.logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    wandb.finish()
