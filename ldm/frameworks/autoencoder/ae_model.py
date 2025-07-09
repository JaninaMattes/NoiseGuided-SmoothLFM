from ldm.models.nn.out.outputs import AEDecoderOutput, AEModelOutput
from ldm.models.architecture.base_architectures import BaseAE, BaseDecoder, BaseEncoder
import os
import sys
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code taken and adapted from:
# - https://github.com/clementchadebec/benchmark_VAE/blob/master/models/ae.py
from jutils import instantiate_from_config


project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../../'))
sys.path.append(project_root)


def exists(val):
    return val is not None


class AE(BaseAE):
    """Vanilla Autoencoder model."""

    def __init__(
        self,
        encoder_cfg: Union[dict, BaseEncoder],
        decoder_cfg: Union[dict, BaseDecoder]
    ):
        super().__init__()
        self.model_name = "AE"

        # Instantiate the encoder and decoder
        self.encoder = encoder_cfg if isinstance(
            encoder_cfg, nn.Module) else instantiate_from_config(encoder_cfg)
        self.decoder = decoder_cfg if isinstance(
            decoder_cfg, nn.Module) else instantiate_from_config(decoder_cfg)

    def forward(self, x: torch.Tensor, normalize=False) -> AEModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        # embedding and reconstruction
        out = self.encode(x, normalize=normalize)
        z = out.get("z_enc", None)
        out = self.decode(z, denorm=normalize)
        recon_z = out.get("z_dec", None)

        # reconstruction loss
        loss = self.loss_function(recon_z, x)

        output = AEModelOutput(
            loss=loss,
            sample=recon_z,
        )
        return output

    def encode(self, x: torch.Tensor, normalize=False) -> torch.Tensor:
        """Encodes the input data

        Args:
            x (torch.Tensor): The latent input data
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor, denorm=False) -> torch.Tensor:
        """Decodes the input data

        Args:
            z (torch.Tensor): The latent output data
        """
        out = self.decoder(z)
        recon_z = out.z_dec
        return AEDecoderOutput(z_dec=recon_z, noise=None)

    def loss_function(self, recons_x, x) -> torch.Tensor:
        """ Reconstruction loss."""
        return F.mse_loss(recons_x, x)

    def initialize_weights(self):
        """Initialize the weights of the model"""
        def _basic_init(module):
            self.apply(_basic_init)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE().to(dev)
