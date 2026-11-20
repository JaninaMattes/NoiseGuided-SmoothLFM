"""Parameter-free stand-ins for the pre-trained components, used only by smoke tests.

They have the same interface and tensor shapes as the real models, so the full pipeline
(dataset generation, training, logging) can run on a CPU without downloading any weights.
"""
import torch
import torch.nn.functional as F
from torch import nn

LATENT_SCALE = 0.18215      # same scale as the SD autoencoder


class PoolingAutoencoder(nn.Module):
    """Stand-in for the SD KL-autoencoder: 256x256x3 images <-> 32x32x4 latents (f = 8)."""

    def __init__(self, factor: int = 8):
        super().__init__()
        self.factor = factor

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(x, self.factor)
        return torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1) * LATENT_SCALE

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.interpolate(z[:, :3] / LATENT_SCALE, scale_factor=self.factor, mode="nearest")
