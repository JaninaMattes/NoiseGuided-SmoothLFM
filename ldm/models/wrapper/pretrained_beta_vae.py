"""Frozen, pre-trained β-VAE used as the self-guidance encoder of the flow matching decoder.

The architecture is rebuilt from the hyper-parameters stored in the Lightning checkpoint of
``ldm.trainer_beta_vae.TrainerModuleLatentBetaVae`` and the weights are loaded strictly, so a
renamed or changed module fails loudly instead of silently running with random weights.
"""
import os

import torch
from jutils import freeze

from ldm.frameworks.beta_vae.bvae_model import BetaVAEModel

MODEL_PREFIX = "model."     # parameters of the β-VAE inside the Lightning module


class PretrainedBetaVAE(BetaVAEModel):
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[PretrainedBetaVAE] Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        params = checkpoint["hyper_parameters"]["vae_cfg"]["params"]
        super().__init__(
            encoder_cfg=params["encoder_cfg"],
            decoder_cfg=params["decoder_cfg"],
            beta=params["beta"],
            kld_weight=params["kld_weight"],
            loss_type=params["loss_type"],
        )
        state_dict = {
            key[len(MODEL_PREFIX):]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(MODEL_PREFIX)
        }
        self.load_state_dict(state_dict, strict=True)
        freeze(self)
        self.eval()
        print(f"[PretrainedBetaVAE] Loaded {ckpt_path}")
