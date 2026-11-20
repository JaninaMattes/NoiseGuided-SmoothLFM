"""Backwards compatibility for checkpoints written before the module renames.

Lightning checkpoints store the full ``target:`` strings of every sub-model in their
hyper-parameters. When a module is renamed, those strings must be mapped to the new
location, otherwise the thesis checkpoints can no longer be instantiated. All renames
are listed here, in one place; :func:`instantiate` applies them transparently.
"""
from typing import Any

from jutils import instantiate_from_config
from omegaconf import DictConfig, ListConfig, OmegaConf

LEGACY_TARGETS = {
    "ldm.trainer_bvae_ti2.TrainerModuleLatentBetaVae": "ldm.trainer_beta_vae.TrainerModuleLatentBetaVae",
    "ldm.trainer_bvae.TrainerModuleLatentBetaVae": "ldm.trainer_beta_vae.TrainerModuleLatentBetaVae",
    "ldm.trainer_rf_vae.TrainerModuleLatentFlow": "ldm.trainer_flow_decoder.TrainerModuleLatentFlow",
    "ldm.models.wrapper.beta_vae-t2i.BVaeLDMWrapper": "ldm.models.wrapper.pretrained_beta_vae.PretrainedBetaVAE",
    "ldm.models.wrapper.beta_vae.BVaeLDMWrapper": "ldm.models.wrapper.pretrained_beta_vae.PretrainedBetaVAE",
    "ldm.frameworks.beta_vae.bvae_model_t2i.BetaVAEModel": "ldm.frameworks.beta_vae.bvae_model.BetaVAEModel",
    "ldm.frameworks.rectified_flow.fm.RectifiedFlowModel": "ldm.frameworks.flow.fm.FlowModel",
    "ldm.models.autoencoder.encoder.skip_vit_t2i.SkipViTEncoder": "ldm.models.autoencoder.encoder.skip_vit.SkipViTEncoder",
    "ldm.models.autoencoder.decoder.skip_vit_t2i.SkipViTDecoder": "ldm.models.autoencoder.decoder.skip_vit.SkipViTDecoder",
    "ldm.models.autoencoder.encoder.vanilla_vit_t2i.VanillaViTEncoder": "ldm.models.autoencoder.encoder.vanilla_vit.VanillaViTEncoder",
    "ldm.models.autoencoder.decoder.vanilla_vit_t2i.VanillaViTDecoder": "ldm.models.autoencoder.decoder.vanilla_vit.VanillaViTDecoder",
}


def migrate_targets(config: Any) -> Any:
    """Return a plain-container copy of ``config`` with every legacy ``target`` replaced."""
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return {
            key: LEGACY_TARGETS.get(value, value) if key == "target" else migrate_targets(value)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [migrate_targets(value) for value in config]
    return config


def instantiate(config: Any) -> Any:
    """``jutils.instantiate_from_config`` that also understands legacy target names.

    OmegaConf input stays OmegaConf, so constructors receive the same types as before.
    """
    migrated = migrate_targets(config)
    if isinstance(config, (DictConfig, ListConfig)):
        migrated = OmegaConf.create(migrated)
    return instantiate_from_config(migrated)
