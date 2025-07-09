# Code based on: https://github.com/joh-schb/image-ldm/blob/main/ldm/models/transformer/repa.py
import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dit import DiT


def build_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )


class REPA(DiT):
    def __init__(
        self,
        *args,
        hidden_size=1152,
        z_dim=768,
        encoder_depth=8,
        projector_dim=2048,
        **kwargs
    ):
        super().__init__(*args, hidden_size=hidden_size, **kwargs)
        self.encoder_depth = encoder_depth
        self.projector = build_mlp(hidden_size, projector_dim, z_dim)
        self.initialize_weights()

    def forward(self, x, t, y, return_z=False):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                # (N, D)
        y = self.y_embedder(y, self.training)       # (N, D)
        c = t_embed + y                             # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            if (i + 1) == self.encoder_depth:
                z = self.projector(x.reshape(-1, D)).reshape(N, T, -1)  # (N, T, z_dim)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if return_z:
            return x, z
        return x


def REPA_XL_2(**kwargs):
    return REPA(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def REPA_L_2(**kwargs):
    return REPA(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def REPA_B_2(**kwargs):
    return REPA(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


REPA_models = {
    'DiT-XL/2': REPA_XL_2,
    'DiT-L/2':  REPA_L_2,
    'DiT-B/2':  REPA_B_2,
}