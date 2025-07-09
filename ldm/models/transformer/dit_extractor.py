# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from jutils import Namespace

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dit import modulate
from dit import TimestepEmbedder, LabelEmbedder
from dit import get_2d_sincos_pos_embed


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def cp(x):
    return x.clone().detach()


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.activations = Namespace(attn=Namespace(), mlp=Namespace(), output=None)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # ===== Attention =====
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        self.activations.attn.residual = cp(x)

        norm_x = self.norm1(x)
        self.activations.attn.norm_x = cp(norm_x)

        mod_x = modulate(norm_x, shift_msa, scale_msa)
        self.activations.attn.mod_x = cp(mod_x)

        attn_out = self.attn(mod_x)
        self.activations.attn.attn_out = cp(attn_out)

        self.activations.attn.gate_msa = cp(gate_msa)

        gate_x = gate_msa.unsqueeze(1) * attn_out
        self.activations.attn.gate_x = cp(gate_x)

        x = x + gate_x

        # ===== MLP =====
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        self.activations.mlp.residual = cp(x)

        norm_x = self.norm2(x)
        self.activations.mlp.norm_x = cp(norm_x)

        mod_x = modulate(norm_x, shift_mlp, scale_mlp)
        self.activations.mlp.mod_x = cp(mod_x)

        mlp_out = self.mlp(mod_x)
        self.activations.mlp.mlp_out = cp(mlp_out)

        self.activations.mlp.gate_mlp = cp(gate_mlp)

        gate_x = gate_mlp.unsqueeze(1) * mlp_out
        self.activations.mlp.gate_x = cp(gate_x)

        x = x + gate_x

        self.activations.output = cp(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Extractor(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,                 # Class label dropout probability
        num_classes=1000,
        out_channels=None,
        learn_sigma=False,          # LEGACY (True for DiT and SiT)
        return_sigma=False,         # LEGACY (True for DiT, False for SiT, but not used at all)
        use_checkpointing=False,
        load_from_ckpt=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        if learn_sigma:
            self.out_channels = in_channels * 2
        else:
            self.out_channels = out_channels if out_channels is not None else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_checkpointing = use_checkpointing
        self.hidden_size = hidden_size
        
        self.return_sigma = return_sigma

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            uc = "+ unconditional" if class_dropout_prob > 0 else ""
            print(f"[DiT] Class-conditional ({num_classes} classes {uc})")
        else:
            self.y_embedder = None
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        if load_from_ckpt is not None:
            dev = next(self.parameters()).device
            print(f"[DiT] Loading weights from {load_from_ckpt}")
            self.load_state_dict(torch.load(load_from_ckpt, map_location=dev))

        self.block_activations = {}

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        
        if self.y_embedder is not None:
            # Add a null class label for unconditional generation
            if y is None:
                unconditional_idx = self.num_classes if self.y_embedder.dropout_prob > 0 else self.num_classes - 1
                y = torch.full((x.size(0),), unconditional_idx, dtype=torch.long, device=x.device)

            if y.ndim > 1:
                y = y.squeeze(1)

            y = self.y_embedder(y, self.training)                   # (N, D)            
            c = t + y                                               # (N, D)
        else:
            c = t

            if y.ndim > 1:
                y = y.squeeze(1)

            y = self.y_embedder(y, self.training)                   # (N, D)
            c = t + y                                               # (N, D)
        else:
            c = t
        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            self.block_activations[i] = block.activations
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma and not self.return_sigma:        # LEGACY
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        if cfg_scale == 1.0:                                # without CFG
            print(f"[DiffusionFlow] CFG scale is 1.0, no CFG applied")
            
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)





#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT_Extractor(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT_Extractor(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT_Extractor(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT_Extractor(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT_Extractor(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT_Extractor(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT_Extractor(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT_Extractor(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT_Extractor(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT_Extractor(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT_Extractor(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT_Extractor(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


if __name__ == "__main__":
    ipt = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 1000, (2,))
    
    """ Conditional """
    net = DiT_models['DiT-B/8']()
    out = net(ipt, t, y)

    print("Conditional")
    print(f"{'Params':<10}: {sum([p.numel() for p in net.parameters() if p.requires_grad]):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Output':<10}: {out.shape}")
    
    """ Unconditional """
    net = DiT_models['DiT-B/8'](num_classes=-1)
    out = net(ipt, t)

    print("Unconditional")
    print(f"{'Params':<10}: {sum([p.numel() for p in net.parameters() if p.requires_grad]):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Output':<10}: {out.shape}")
