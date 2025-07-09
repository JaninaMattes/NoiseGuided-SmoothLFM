import os
import sys
import torch
import torch.nn as nn
import numpy as np
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dit import DiT


class RegisterDiT(DiT):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        *args,
        hidden_size=1152,
        n_registers=1,
        load_from_ckpt=None,
        **kwargs
    ):
        super().__init__(*args, hidden_size=hidden_size, load_from_ckpt=None, **kwargs)

        self.n_registers = n_registers
        if n_registers > 0:
            self.registers = nn.Parameter(torch.randn(1, n_registers, hidden_size))
        
        # manually load checkpoint after registers have been added
        if load_from_ckpt is not None:
            dev = next(self.parameters()).device
            print(f"[DiT] Loading weights from {load_from_ckpt}")
            self.load_state_dict(torch.load(load_from_ckpt, map_location=dev))

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        
        # add register tokens
        if self.n_registers > 0:
            registers = self.registers.repeat(x.shape[0], 1, 1)
            x = torch.cat([registers, x], dim=1)

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
            
        for block in self.blocks:
            if self.use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)
            else:
                x = block(x, c)                      # (N, T, D)

        # remove register tokens
        if self.n_registers > 0:
            x = x[:, self.n_registers:]

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma and not self.return_sigma:        # LEGACY
            x, _ = x.chunk(2, dim=1)
        return x


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return RegisterDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return RegisterDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return RegisterDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return RegisterDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return RegisterDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return RegisterDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return RegisterDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return RegisterDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return RegisterDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return RegisterDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return RegisterDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return RegisterDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


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
    net = DiT_models['DiT-B/8'](n_registers=2)
    out = net(ipt, t, y)

    print("Conditional")
    print(f"{'Params':<10}: {sum([p.numel() for p in net.parameters() if p.requires_grad]):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Output':<10}: {out.shape}")
    
    """ Unconditional """
    net = DiT_models['DiT-B/8'](n_registers=2, num_classes=-1)
    out = net(ipt, t)

    print("Unconditional")
    print(f"{'Params':<10}: {sum([p.numel() for p in net.parameters() if p.requires_grad]):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Output':<10}: {out.shape}")

    """ No Registers """
    net = DiT_models['DiT-B/8'](n_registers=0, num_classes=-1)
    out = net(ipt, t)

    print("No Registers")
    print(f"{'Params':<10}: {sum([p.numel() for p in net.parameters() if p.requires_grad]):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Output':<10}: {out.shape}")
