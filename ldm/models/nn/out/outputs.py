
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.utils import BaseOutput


@dataclass
class EncoderOutput(BaseOutput):
    f"""
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    
    taken from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_outputs.py#L7
    """

    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821
    
    
    
@dataclass
class DecoderOutput(BaseOutput):
    f"""
    Output of the decoding method.
    
    Args:
        recon_x (`torch.Tensor`): Reconstructed image of shape (B, NC, H, W).
    """
    sample: torch.Tensor
    logvar_dec: torch.Tensor



@dataclass
class BetaVAEModelOutput(BaseOutput):
    f"""
    The output of [`BetaVAEModel`].

    Args:
        sample (`torch.Tensor`):
            The reconstructed image output conditioned on the `encoder_hidden_states` input.
        mu_z (`torch.Tensor`):
            The mean of the latent distribution output conditioned on the `encoder_hidden_states` input.
        logvar_z (`torch.Tensor`):
            The log variance of the latent distribution output conditioned on the `encoder_hidden_states` input.
    """
    loss: torch.Tensor
    recon_loss: torch.Tensor
    kld_loss: torch.Tensor
    sample: torch.Tensor
    mu_z: torch.Tensor
    logvar_z: torch.Tensor



@dataclass
class AEModelOutput(BaseOutput):
    f"""
    The output of [`BetaVAEModel`].

    Args:
        sample (`torch.Tensor`):
            The reconstructed image output conditioned on the `encoder_hidden_states` input.
        mu_z (`torch.Tensor`):
            The mean of the latent distribution output conditioned on the `encoder_hidden_states` input.
        logvar_z (`torch.Tensor`):
            The log variance of the latent distribution output conditioned on the `encoder_hidden_states` input.
    """
    loss: torch.Tensor
    sample: torch.Tensor


        
@dataclass
class AEEncoderOutput(BaseOutput):
    f"""
    The output of [`AEEncoder`].
    """
    z_enc: torch.Tensor
    skips: Optional[list[torch.Tensor]] = None      # skip connections
    
    

@dataclass
class AEDecoderOutput(BaseOutput):
    f"""
    The output of [`AEDecoder`].
    """
    z_dec: torch.Tensor
    noise: Optional[torch.Tensor] = None    # noise added to the latent space
    
    
@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821
    