#     Code taken and adapted from:
#    - https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
#    - https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/
#    - https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
#
#    Related papers:
#    - Beta-VAE:                             https://openreview.net/forum?id=Sy2fzU9gl
#    - Beta-VAE with capacity constraint:    https://arxiv.org/pdf/1804.03599.pdfs

import math
from typing import Union

import torch
from torch.nn import functional as F
import torch.nn as nn

from jutils import instantiate_from_config
from jutils import exists, freeze, default

from ldm.models.architecture.base_architectures import BaseDecoder, BaseEncoder, BaseVAE
from ldm.models.nn.dist.distributions import DiagonalGaussianDistribution
from ldm.models.nn.out.outputs import BetaVAEModelOutput, DecoderOutput, EncoderOutput


class BetaVAEModel(BaseVAE):
    """
    A VAE model with KL loss for encoding latents into parametrized latent codes and decoding latent representations into latents.
    latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
    """

    def __init__(
        self,
        encoder_cfg: Union[dict, BaseEncoder],
        decoder_cfg: Union[dict, BaseDecoder],
        beta: int = 1e-4,
        gamma: float = 1000.,
        max_capacity: int = 25,
        capacity_max_iter: int = 1e5,
        loss_type: str = 'B',
        kld_weight: float = 0.0,
        kl_anneal_steps: int = 10000,
        kl_warmup_steps: int = 1000,
        add_dct_component: bool = False,
        num_iter: int = 0,
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
        self.add_dct_component = add_dct_component

        # Instantiate the encoder and decoder
        self.encoder = encoder_cfg if isinstance(
            encoder_cfg, nn.Module) else instantiate_from_config(encoder_cfg)
        self.decoder = decoder_cfg if isinstance(
            decoder_cfg, nn.Module) else instantiate_from_config(decoder_cfg)

        # Activation
        self.softplus = nn.Softplus()

        # Initialize weights
        self.initialize_weights()

    def forward(self, sample: torch.Tensor, target: torch.Tensor = None, dct_freq: torch.Tensor = None):
        r"""
        Args:
            sample (`torch.Tensor`): Input tensor of shape `(B, nc, H, W)`.
            sample_posterior (`bool`): Whether to sample from the posterior distribution or use the mode.
            normalize (`bool`): Whether to normalize the latent space based on SD Unit Gaussian.
            dct conditioning (`torch.Tensor`): Whether to use DCT frequency conditioning for decoder.
        """
        posterior = self.encode(sample)['latent_dist']
        z = self.reparametrize(posterior)
        # Decode the latent representation
        generated = self.decode(z, dct_freq=dct_freq)['sample']

        # Compute loss
        if not exists(target):
            target = sample

        out = self.loss_function(target, generated, posterior)
        total_loss, recon_loss, kld_loss = out['loss'], out['recon_loss'], out['kld_loss']

        return BetaVAEModelOutput(loss=total_loss, recon_loss=recon_loss, kld_loss=kld_loss, sample=generated, mu_z=posterior.mean, logvar_z=posterior.log_var)

    def encode(self, z):
        """
        Parametrisation of the approximate posterior distribution q(z|x).
        Return the parameters `mu` and `std` of the Normal distribution.

        z = eps * std + mu
        """
        out = self.encoder(z)
        parameter = out['z_enc']
        posterior = DiagonalGaussianDistribution(
            parameter, deterministic=False)
        # Parametrize [mu, logvar]
        return EncoderOutput(latent_dist=posterior)

    def decode(self, z, dct_freq: torch.Tensor = None):
        """
        Infer the parameters of the data distribution p(x|z).
        Return the parameters `mu` and `std` of the Normal distribution.
        Outputs a constant identity matrix for the log-variance.
        """
        recon_z = self.decoder(z, dct_freq=dct_freq)['z_dec']
        logvar_dec = torch.zeros_like(recon_z)

        return DecoderOutput(sample=recon_z, logvar_dec=logvar_dec)

    def reparametrize(self, posterior):
        """
        Reparameterization trick to sample from the approximate posterior distribution.
        """
        return posterior.sample()

    def loss_function(self, target, generated, posterior):
        """ Combines reconstruction loss and KL divergence for total loss.
            Repo: https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
        """
        self.num_iter += 1

        # (1) Reconstruction loss
        recons_loss = posterior.mse(target, generated)

        # (2) KL divergence
        kld_loss = posterior.kl()

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * self.kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(target.device)
            C = torch.clamp(self.C_max/self.C_stop_iter *
                            self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * \
                self.kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return dict(loss=loss, recon_loss=recons_loss, kld_loss=kld_loss)

    def update_kld_weight(self):
        """ KL-Annealing: Update kld_weight using cosine annealing and sync across GPUs """
        raise NotImplementedError

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
    def sample_posterior(self, x, device='cuda'):
        """ Generate from posterior distribution given input image x."""
        with torch.autocast("cuda"):
            z = self.encode_to_latent(x.to(device))
            sample = self.decode_from_latent(z.to(device))
        return sample

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

    def initialize_weights(self):
        """Initialize the weights of the model"""
        def _basic_init(module):
            self.apply(_basic_init)


if __name__ == '__main__':
    pass
