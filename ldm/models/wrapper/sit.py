import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets.utils import download_url
from functools import partial

from jutils import freeze

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

from ldm.models.context_diffusion.sit_context import SiT_models

    
    
""" SiT Wrapper """


class SiTLDMWrapper(nn.Module):
    def __init__(
        self, 
        model_type, 
        learn_sigma=False, 
        legacy_attn=True, 
        ckpt_path=None, 
        sit_model=SiT_models, 
        requires_grad=True, 
        **kwargs
    ):
        super().__init__()
        if model_type not in SiT_models:
            raise ValueError(f"Model type {model_type} not found in SiT_models. Available options are: {list(SiT_models.keys())}")
        
        self.model = sit_model[model_type](learn_sigma=learn_sigma, **kwargs)     
        self.model.eval()  # Set the model to evaluation mode
        
        if not requires_grad:
            freeze(self.model)
            print(f"[SiTWrapper] Model is frozen.")
        
        if ckpt_path:
            # Load the checkpoint
            state_dict = torch.load(ckpt_path, weights_only=True)
            
            # Add missing keys
            missing_keys = set(self.model.state_dict().keys()) - set(state_dict.keys())
            for key in missing_keys:
                state_dict[key] = self.model.state_dict()[key]
            print(f'[SiTWrapper] Loading checkpoint from {ckpt_path}.')
            
            # Resize the positional embedding if needed
            if 'pos_embed' in state_dict:
                pos_embed_shape = self.model.pos_embed.shape
                ckpt_pos_embed = state_dict['pos_embed']
                if ckpt_pos_embed.shape != pos_embed_shape:
                    print(f"Resizing pos_embed from {ckpt_pos_embed.shape} to {pos_embed_shape}.")
                    state_dict['pos_embed'] = self._resize_pos_embed(ckpt_pos_embed, pos_embed_shape)
            
            # Load the state dict
            self.model.load_state_dict(state_dict)

        # Initialize the positional embeddings
        self._initialize_pos_embed()

    def _resize_pos_embed(self, ckpt_pos_embed, target_shape):
        """
        Resize the positional embedding tensor to match the target shape.
        """
        if ckpt_pos_embed.shape != target_shape:
            resized_pos_embed = torch.zeros(target_shape)
            resized_pos_embed[:, :ckpt_pos_embed.shape[1], :] = ckpt_pos_embed 
            return resized_pos_embed
        return ckpt_pos_embed

    def _initialize_pos_embed(self):
        # Generate fresh positional embeddings for the model
        cls_token, extra_tokens = bool(self.model.y_embedder), self.model.extras  # 1 if class-conditional, 0 otherwise
        grid_size = int(self.model.x_embedder.num_patches ** 0.5)  
        pos_embed = get_2d_sincos_pos_embed(self.model.pos_embed.shape[-1], grid_size, cls_token=cls_token, extra_tokens=extra_tokens)
        self.model.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, t, **cond_kwargs):
        return self.model(x, t, **cond_kwargs)



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



if __name__ == "__main__":
    # Test SiTContext wrapper
    model_type = "SiT-S/2" 
    input_size = 32
    in_channels = 4
    mlp_ratio = 4
    num_classes = 1000
    learn_sigma = False
    legacy_attn = True
    cat_context = True
    context_size = 1024
    compile = True  
    ckpt_path = "checkpoints/SiT-S-2/step_095000_clean.pt" 

    model = SiTLDMWrapper(
        model_type,
        learn_sigma=learn_sigma,
        legacy_attn=legacy_attn,
        ckpt_path=ckpt_path,
        input_size=input_size,
        in_channels=in_channels,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        cat_context=cat_context,
        context_size=context_size,
        compile=compile
    )
    print(model)
    
    # Test forward pass with bs=1
    dev = next(model.parameters()).device
    ipt = torch.randn(1, 4, 32, 32).to(dev)
    t = torch.tensor([0.5]).to(dev)
    context = torch.randn(1, 1024).to(dev)
    with torch.no_grad():
        output = model(ipt, t, y=None, context=context)
    print(f"Output shape: {output.shape}")


    # Test forward pass with bs=16
    ipt = torch.randn(16, 4, 32, 32).to(dev)
    t = torch.tensor([0.5]*16).to(dev)
    context = torch.randn(16, 1024).to(dev)
    with torch.no_grad():
        output = model(ipt, t, y=None, context=context)
    print(f"Output shape: {output.shape}")
    
    # Test forward pass with bs=16 and y
    ipt = torch.randn(16, 4, 32, 32).to(dev)
    t = torch.tensor([0.5]*16).to(dev)
    y = torch.randint(0, 1000, (16,)).to(dev)
    context = torch.randn(16, 1024).to(dev)
    with torch.no_grad():
        output = model(ipt, t, y=y, context=context)
    print(f"Output shape: {output.shape}")