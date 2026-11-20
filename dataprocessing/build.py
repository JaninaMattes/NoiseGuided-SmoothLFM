"""Build the guidance-free ImageNet-256 dataset (Stage-2 training data).

For every source image: resize/centre-crop to 256, encode with the frozen first stage to ``x_1``,
integrate the forward ODE of the frozen class-conditional SiT-XL/2 (ω = 1) and store ``x_t`` for all
configured timesteps together with the image and its label. Runs are resumable: samples already
in the output file are skipped.
"""
import logging
import subprocess
from typing import Optional

import torch
from jutils import freeze
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from dataprocessing.forward_ode import GuidanceFreeEncoder
from dataprocessing.hdf5_writer import HDF5DatasetWriter
from dataprocessing.sources import ClassFolderImages
from ldm.legacy import instantiate

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_frozen(config: DictConfig, device: torch.device) -> torch.nn.Module:
    model = instantiate(config).to(device)
    freeze(model)
    return model.eval()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generation_settings(cfg: DictConfig) -> dict:
    """Everything that defines the content of the dataset; a resumed run must match it exactly."""
    return {
        "timesteps": list(cfg.timesteps),
        "ode_steps": cfg.ode_steps,
        "cfg_scale": cfg.cfg_scale,
        "image_size": cfg.image_size,
        "class_labels": OmegaConf.to_container(cfg.class_labels) if cfg.class_labels is not None else None,
        "first_stage": cfg.first_stage.name,
        "flow_encoder": cfg.flow_encoder.name,
        "max_per_class": {name: split.max_per_class for name, split in cfg.splits.items()},
    }


def build_split(name: str, images: ClassFolderImages, encoder: GuidanceFreeEncoder, writer: HDF5DatasetWriter,
                cfg: DictConfig, device: torch.device) -> None:
    done = writer.num_samples(name)
    if done >= len(images):
        logger.info(f"[{name}] complete ({done} samples)")
        return
    logger.info(f"[{name}] {done}/{len(images)} samples present, generating the rest")
    loader = DataLoader(Subset(images, range(done, len(images))), batch_size=cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers)
    for step, (batch, labels) in enumerate(loader, start=1):
        latents = encoder(batch.to(device), labels.to(device))
        writer.append(name, batch.numpy(), labels.numpy(), {t: x.cpu().numpy() for t, x in latents.items()})
        if step % cfg.log_every == 0:
            logger.info(f"[{name}] {writer.num_samples(name)}/{len(images)}")
    logger.info(f"[{name}] done: {writer.num_samples(name)} samples")


def build_dataset(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    encoder = GuidanceFreeEncoder(
        first_stage=load_frozen(cfg.first_stage, device),
        flow=load_frozen(cfg.flow_encoder, device),
        timesteps=cfg.timesteps,
        num_steps=cfg.ode_steps,
        cfg_scale=cfg.cfg_scale,
        num_classes=cfg.num_classes,
    )
    provenance = {
        "git_commit": git_commit(),
        "source_dirs": {name: split.source_dir for name, split in cfg.splits.items()},
        "first_stage": OmegaConf.to_container(cfg.first_stage, resolve=True),
        "flow_encoder": OmegaConf.to_container(cfg.flow_encoder, resolve=True),
    }
    with HDF5DatasetWriter(cfg.output, generation_settings(cfg), list(cfg.timesteps),
                           compression=cfg.compression, provenance=provenance) as writer:
        for name, split in cfg.splits.items():
            images = ClassFolderImages(split.source_dir, cfg.class_labels, split.max_per_class, cfg.image_size)
            build_split(name, images, encoder, writer, cfg, device)
    logger.info(f"Dataset written to {cfg.output}")
