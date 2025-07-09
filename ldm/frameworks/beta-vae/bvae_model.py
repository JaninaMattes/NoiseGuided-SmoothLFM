#     Code taken and adapted from:
#    - https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
#    - https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/
#    - https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
#
#    Related papers:
#    - Beta-VAE:                             https://openreview.net/forum?id=Sy2fzU9gl
#    - Beta-VAE with capacity constraint:    https://arxiv.org/pdf/1804.03599.pdfs
from ldm.models.nn.out.outputs import BetaVAEModelOutput, DecoderOutput, EncoderOutput
from ldm.models.nn.dist.distributions import DiagonalGaussianDistribution
from ldm.models.architecture.base_architectures import BaseDecoder, BaseEncoder, BaseVAE
from ldm.frameworks.beta_vae.annealer import Annealer, BetaAnnealer
import os
import sys
import math
from typing import Union, Optional

import torch
from torch.nn import functional as F
import torch.nn as nn

from jutils import instantiate_from_config
from jutils import exists, freeze, default


project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)


class BetaVAEModel(BaseVAE):
    """
    A VAE model with KL loss for encoding latents into parametrized latent codes and decoding latent representations into latents.
    latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
    """

    def __init__(
        self,
        encoder_cfg: Union[dict, BaseEncoder],
        decoder_cfg: Union[dict, BaseDecoder],
        annealing_cfg: Optional[Union[dict, Annealer]] = None,
        beta_annealing_cfg: Optional[Union[dict, BetaAnnealer]] = None,
        beta: float = 1e-4,
        gamma: float = 1000.,
        max_capacity: int = 25,
        capacity_max_iter: int = 1e5,
        loss_type: str = 'B',
        kld_weight: float = 1.0,
        kl_anneal_steps: int = 10000,
        kl_warmup_steps: int = 1000,
        num_iter: int = 0,
        use_bottleneck_layer: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = "BetaVAE"

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.kld_weight = kld_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_anneal_steps = kl_anneal_steps
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter
        self.num_iter = num_iter
        self.use_vae = use_bottleneck_layer

        # Instantiate the encoder and decoder
        self.encoder = encoder_cfg if isinstance(
            encoder_cfg, nn.Module) else instantiate_from_config(encoder_cfg)
        self.decoder = decoder_cfg if isinstance(
            decoder_cfg, nn.Module) else instantiate_from_config(decoder_cfg)

        # Add annealing agents
        self.annealing_agent = None
        if exists(annealing_cfg):
            if isinstance(annealing_cfg, Annealer):
                self.annealing_agent = annealing_cfg
            else:
                self.annealing_agent = instantiate_from_config(annealing_cfg)

        self.beta_annealing_agent = None
        if exists(beta_annealing_cfg):
            if isinstance(beta_annealing_cfg, BetaAnnealer):
                self.beta_annealing_agent = beta_annealing_cfg
            else:
                self.beta_annealing_agent = instantiate_from_config(
                    beta_annealing_cfg)
            print(f"Using beta annealing.")

        # Initialize weights
        self.initialize_weights()

    def forward(self, sample: torch.Tensor, target: torch.Tensor = None):
        r"""
        Repo: https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py

        Args:
            sample (`torch.Tensor`): Input tensor of shape `(B, nc, H, W)`.
            sample_dist (`bool`): Whether to sample from the dist distribution or use the mode.
            normalize (`bool`): Whether to normalize the latent space based on SD Unit Gaussian.

        """
        dist = self.encode(sample)['latent_dist']

        if self.use_vae:
            z = self.reparametrize(dist)
            mu_z = dist.mean
            logvar_z = dist.log_var
        else:
            z = dist    # deterministic code (no bottleneck / regularization)
            mu_z = torch.zeros_like(z)
            logvar_z = torch.zeros_like(z)

        generated = self.decode(z)['sample']

        if not exists(target):
            target = sample

        # Compute loss
        out = self.loss_function(target, generated, dist)
        total_loss, recon_loss, kld_loss = out['loss'], out['recon_loss'], out['kld_loss']

        return BetaVAEModelOutput(loss=total_loss, recon_loss=recon_loss, kld_loss=kld_loss, sample=generated, mu_z=mu_z, logvar_z=logvar_z)

    def encode(self, z):
        """
        Parametrisation of the approximate dist distribution q(z|x).
        Return the parameters `mu` and `std` of the Normal distribution.

        z = eps * std + mu
        """

        out = self.encoder(z)
        z_enc = out['z_enc']
        if self.use_vae:
            # Distribution with mean and log-variance
            dist = DiagonalGaussianDistribution(z_enc, deterministic=False)
        else:
            # Deterministic code (no bottleneck / regularization)
            dist = z_enc

        return EncoderOutput(latent_dist=dist)

    def decode(self, z):
        """
        Infer the parameters of the data distribution p(x|z).
        Return the parameters `mu` and `std` of the Normal distribution.
        Outputs a constant identity matrix for the log-variance.
        """
        recon_z = self.decoder(z)['z_dec']
        # Placeholder for log-variance
        logvar_dec = torch.zeros_like(recon_z)
        return DecoderOutput(sample=recon_z, logvar_dec=logvar_dec)

    def loss_function(self, sample, generated, dist):
        """ Combines reconstruction loss and KL divergence for total loss.
            Repo: https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
        """
        recons_loss = 0.5 * F.mse_loss(
            generated.reshape(sample.shape[0], -1),  # flatten
            sample.reshape(sample.shape[0], -1),
            reduction="none",
        ).sum(dim=-1)

        if self.use_vae:
            # KL divergence
            kld_loss = dist.kl()
        else:
            # No KL divergence in Deterministic AE mode
            kld_loss = torch.tensor(0.0, device=sample.device)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            if self.beta_annealing_agent is not None:
                beta = self.beta_annealing_agent()
            else:
                beta = self.beta
            loss = recons_loss + beta * self.kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(sample.device)
            C = torch.clamp(self.C_max / self.C_stop_iter *
                            self.num_iter, 0, self.C_max.item())
            loss = recons_loss + self.gamma * \
                self.kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return dict(loss=loss.mean(dim=0), recon_loss=recons_loss.mean(dim=0), kld_loss=kld_loss.mean(dim=0))

    def reparametrize(self, dist):
        """
        Reparameterization trick to sample from the approximate dist distribution.
        """
        if not self.use_vae:
            raise RuntimeError(
                "Reparameterization called in Deterministic AE mode.")
        return dist.sample()

    def update_kld_weight(self):
        """ KL-Annealing: Update kld_weight using cosine annealing and sync across GPUs """
        if self.annealing_agent is not None:
            self.annealing_agent.step()

    def update_beta_annealing(self):
        """ Beta-Annealing: Update beta using cosine annealing and sync across GPUs """
        if self.beta_annealing_agent is not None:
            self.beta_annealing_agent.step()

    def update_num_iter(self):
        """ Update the number of iterations """
        self.num_iter += 1
        if self.num_iter > self.C_stop_iter:
            self.num_iter = self.C_stop_iter

    def encode_to_latent(self, x):
        """Encodes an input image to its latent representation."""
        with torch.autocast("cuda"):
            z = self.encode(x)['latent_dist'].sample()
        return z

    def decode_from_latent(self, z):
        """Decodes a latent representation to its image space."""
        with torch.autocast("cuda"):
            sample = self.decode(z)['sample']
        return sample

    @torch.no_grad()
    def sample_prior(self, num_samples=16, latent_dim=1024, device='cuda'):
        """ Samples images from the prior latent distribution and return 
            a corresponding image space map.
        """
        z = torch.randn(num_samples, latent_dim, device=device)
        with torch.autocast("cuda"):
            sample = self.decode(z)['sample']
        return sample

    @torch.no_grad()
    def sample_dist(self, x, device='cuda'):
        """ Generate from dist distribution given input image x."""
        with torch.autocast("cuda"):
            z = self.encode_to_latent(x.to(device))
            sample = self.decode_from_latent(z.to(device))
        return sample

    def query_beta(self):
        """ Returns the current beta value. """
        if self.beta_annealing_agent is not None:
            return self.beta_annealing_agent()
        else:
            return self.beta

    def initialize_weights(self):
        """Initialize the weights of the model"""
        def _basic_init(module):
            self.apply(_basic_init)

    ##############################################
    #           Latent Manipulation             #
    ##############################################

    @torch.no_grad()
    def interpolate_between(self, x1, x2, num_samples=16, device='cuda'):
        """Interpolates between two images in latent space."""
        # Encode base codes
        z1 = self.encode_to_latent(x1.to(device))
        z2 = self.encode_to_latent(x2.to(device))

        # Interpolate between latent vectors
        t = torch.linspace(0, 1, steps=num_samples,
                           device=z1.device, dtype=z1.dtype)
        interps = [(1 - t[i]) * z1 + t[i] * z2 for i in range(num_samples)]
        interps = torch.cat(interps)

        # Decode interpolated latent vectors
        sample = self.decode_from_latent(interps)
        return sample

    @torch.no_grad()
    def tweak_latent(self, x, dim, value, device='cuda'):
        """Tweaks a latent dimension in the input image."""
        z = self.encode_to_latent(x.to(device))
        # Tweak a dimension
        z[:, dim] = value
        sample = self.decode_from_latent(z)
        return sample


if __name__ == '__main__':
    # --- Dummy Code for Quick Testing ---
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define configuration for Deterministic AE mode (no bottleneck)
    encoder_config_ae = {
        "target": "ldm.models.autoencoder.encoder.skip_vit_t2i.SkipViTEncoder",
        "params": {
            "in_channels": 4,
            "image_size": 32,
            "patch_size": 2,
            "embed_dim": 512,
            "num_layers": 1,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "latent_dim": None,  # No latent dimension in AE mode
            "use_bottleneck_layer": False,
            "moments_factor": 1.0,
            "skip": True,
            "skip_indices": [0],
            "skip_layer_type": 'cross_attn',
            "compile": True
        }
    }

    decoder_config_ae = {
        "target": "ldm.models.autoencoder.decoder.skip_vit_t2i.SkipViTDecoder",
        "params": {
            "in_channels": 4,
            "image_size": 32,
            "patch_size": 2,
            "latent_dim": 512,  # No bottleneck
            "embed_dim": 512,
            "num_layers": 1,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "skip": True,
            "skip_indices": [0],
            "skip_layer_type": 'cross_attn',
            "use_act_layer": False,
            "compile": True
        }
    }

    try:
        ae_model = BetaVAEModel(
            encoder_cfg=encoder_config_ae,
            decoder_cfg=decoder_config_ae,
            use_bottleneck_layer=False,
            loss_type='B'
        ).to(DEV)
        print(
            f"Successfully instantiated BetaVAEModel in Deterministic AE mode: {ae_model.use_vae}")

        dummy_sample = torch.randn(16, 4, 32, 32).to(DEV)

        print("Performing a forward pass with dummy data...")
        output = ae_model(dummy_sample)

        print("Forward pass complete.")
        print(f"Output keys: {output.keys()}")
        print(f"Reconstruction loss: {output['recon_loss'].item():.4f}")
        print(
            f"KL loss: {output['kld_loss'].item():.4f} (Expected 0.0 in Deterministic AE mode)")
        print(f"Total loss: {output['loss'].item():.4f}")
        print(f"Generated sample shape: {output['sample'].shape}")
        print(f"Latent code shape: {output['mu_z'].shape}")

    except Exception as e:
        print(f"An error occurred during the test: {e}")

    print("Quick test finished.")
