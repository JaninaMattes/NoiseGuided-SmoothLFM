"""Experiment tracking: one place that decides where a run is logged.

Every run always writes a local ``metrics.csv`` (CSVLogger). On top of that,
Weights & Biases and MLflow can be switched on independently in the config:

    use_wandb: true            # online W&B   (or use_wandb_offline: true)
    use_mlflow: true           # MLflow, local ./mlruns by default
    log_images: true           # also log sample grids, not only scalars

If neither W&B nor MLflow is enabled, TensorBoard is used so that a run is
never "blind". Lightning fans scalar metrics (``self.log``) out to all loggers;
images are not covered by Lightning, so :func:`log_image` does that here.
"""
from typing import Sequence

import numpy as np
from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf


def build_loggers(cfg: DictConfig, log_dir: str, run_name: str) -> list[Logger]:
    """Create the loggers requested by ``cfg`` for a run stored in ``log_dir``."""
    hparams = OmegaConf.to_container(cfg, resolve=True)
    tags = [str(cfg.user), *map(str, cfg.tags)]
    loggers: list[Logger] = []

    if cfg.use_wandb or cfg.use_wandb_offline:
        loggers.append(WandbLogger(
            save_dir=log_dir,
            name=run_name,
            project=cfg.wandb_project,
            tags=tags,
            config=hparams,
            offline=cfg.use_wandb_offline,
            group="DDP",
        ))
    if cfg.use_mlflow:
        loggers.append(MLFlowLogger(
            experiment_name=cfg.mlflow_experiment,
            run_name=run_name,
            tracking_uri=cfg.mlflow_tracking_uri,
            tags={"user": str(cfg.user), "tags": ",".join(tags), "log_dir": log_dir},
        ))
    if not loggers:
        loggers.append(TensorBoardLogger(save_dir=log_dir, name="", version="", default_hp_metric=False))

    csv_logger = CSVLogger(log_dir, name="", version="", flush_logs_every_n_steps=500)
    csv_logger.log_hyperparams(hparams)
    loggers.append(csv_logger)
    return loggers


@rank_zero_only
def log_run_info(loggers: Sequence[Logger], info: dict, config_path: str) -> None:
    """Attach run metadata (command, batch size, ...) and the resolved config file to the run."""
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.summary.update({k: str(v) for k, v in info.items()})
            logger.experiment.save(config_path, policy="now")
        elif isinstance(logger, MLFlowLogger):
            for key, value in info.items():
                logger.experiment.set_tag(logger.run_id, f"info.{key}", str(value)[:5000])
            logger.experiment.log_artifact(logger.run_id, config_path)


@rank_zero_only
def log_image(loggers: Sequence[Logger], key: str, image: np.ndarray, step: int) -> None:
    """Log one ``(H, W, C)`` uint8 image to every logger that supports images."""
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.log_image(key=key, images=[image], step=step)
        elif isinstance(logger, MLFlowLogger):
            logger.experiment.log_image(logger.run_id, image, key=key, step=step)
        elif isinstance(logger, TensorBoardLogger):
            logger.experiment.add_image(key, image, global_step=step, dataformats="HWC")
