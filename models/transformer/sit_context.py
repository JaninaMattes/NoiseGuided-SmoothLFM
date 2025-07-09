# Code adapted from:
# - https://github.com/willisma/SiT/blob/main/models.py
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import os
import sys
import numpy as np
import math
from functools import partial

import torch
import torch.nn as nn


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

from ldm.models.nn.layers.attention import Attention
from ldm.models.nn.layers.mlp import Mlp
from ldm.models.nn.layers.patch_embed import PatchEmbed
    

""" Jit Compile """

COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=True, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f
    
    
    
""" Modulation Functions """

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        
        if COMPILE: self.forward = compile_fn(self.forward)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        if COMPILE: self.forward = compile_fn(self.forward)
        

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
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
        
        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    

class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x




""" Context Embedder (Simple MLP) """

class ContextEmbedder(nn.Module):
    """ Convert context vector (B, context_dim) → (B, embed_dim). """
    def __init__(self, context_embedding_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x):
        return self.mlp(x)
    


class ContextEmbedderWithCFG(nn.Module):
    """ Convert context vector (B, context_size) → (B, embed_dim)."""
    def __init__(self, context_embedding_size, hidden_size, dropout_prob):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.dropout_prob = dropout_prob
        self.null_token = nn.Parameter(torch.zeros(1, hidden_size))                 # learned null-token for dropped context vectors
        
        if COMPILE: self.forward = compile_fn(self.forward)


    def token_drop(self, context, drop_prob):
        """ Drops context vector (B, context_size) → (B, embed_dim) for classifier-free guidance.
            Torch supported.
            
            Drop mask:
            - Drops elements per batch (for each element a random number is drawn, if its less it's replaced with a null token)
        """
        drop_mask = torch.rand(context.size(0), device=context.device) < drop_prob  # (N,) 
        null_context = self.null_token.expand(context.shape[0], -1)                 # (N, D)
        return torch.where(drop_mask.unsqueeze(1), null_context, context)           


    def forward(self, x):
        orig_dtype = x.dtype                            
        context_emb = self.mlp(x.to(self.null_token.dtype))  
        context_emb = context_emb.to(orig_dtype)       

        if self.dropout_prob > 0:
            context_emb = self.token_drop(context_emb, self.dropout_prob)

        return context_emb

  

class ContextSiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=None,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,                 # Class label dropout probability
        num_classes=1000,                       # -1 for no class labels
        context_size=1024,                      # Context vector dimensionality
        context_dropout_prob=0.1,               # Context dropout probability
        cat_context=False,                      # Concatenate context to image tokens
        learn_sigma=True,                       # Learnable sigma for diffusion process
        return_sigma=False,                     # Return sigma in forward pass
        use_checkpointing=False,
        load_from_ckpt=None,
        compile=True,                           # JIT compile the model
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile
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
            
        # Context embedding + CFG
        if cat_context:
            if context_dropout_prob > 0:
                self.context_embedder = ContextEmbedderWithCFG(context_size, hidden_size, context_dropout_prob)
                uc = "+ unconditional" if context_dropout_prob > 0 else ""
                print(f"[DiT] Context-conditional ({context_size} dim, context {uc})")
            else:
                self.context_embedder = ContextEmbedder(context_size, hidden_size)
                print(f"[DiT] Context-conditional ({context_size} dim)")
        else:
            self.context_embedder = None
            print("[DiT] Unconditional")

        # Will use fixed sin-cos embedding:
        self.extras = int(cat_context)
        num_patches = self.x_embedder.num_patches + self.extras
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        if load_from_ckpt is not None:
            dev = next(self.parameters()).device
            print(f"[DiT] Loading weights from {load_from_ckpt}")
            self.load_state_dict(torch.load(load_from_ckpt, map_location=dev))

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    
    def forward(self, x, t, y=None, context=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        context: (N, context_size) tensor of 1D context vector
        """
        x = self.x_embedder(x)                                      # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                                      # (N, D)
        
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

        if self.context_embedder is not None and context is not None:
            context_emb = self.context_embedder(context).unsqueeze(1)   # (N, D) -> (N, 1, D)
            x = torch.cat([x, context_emb], dim=1)                      # (N, T+1, D) append context to img tokens
            
        x = x + self.pos_embed                                          # (N, T+1, D)
        
        for block in self.blocks:
            if self.use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)
            else:
                x = block(x, c)                                         # (N, T, D)
        
        if self.context_embedder is not None and context is not None:
            x = x[:, :-1]                                               # remove context token
        
        x = self.final_layer(x, c)                                      # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                                          # (N, out_channels, H, W)
        if self.learn_sigma and not self.return_sigma:                  # LEGACY
            x, _ = x.chunk(2, dim=1)
        return x
        
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
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
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


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


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        cls_token, extra_tokens = bool(self.y_embedder), self.extras  # 1 if class-conditional, 0 otherwise
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=cls_token, extra_tokens=extra_tokens)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize context embedding MLP:
        if self.context_embedder is not None:
            nn.init.normal_(self.context_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.context_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)



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


#################################################################################
#                                   ContextSiT Configs                          #
#################################################################################

def SiT_XL_2(**kwargs):
    return ContextSiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return ContextSiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return ContextSiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return ContextSiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return ContextSiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return ContextSiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return ContextSiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return ContextSiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return ContextSiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return ContextSiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return ContextSiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return ContextSiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}




if __name__ == '__main__':
    
    """ Simple test for concatenation """ 
    
    N = 1  # Batch size
    D = 2  # Dimensionality of each token
    T = 2  # Number of tokens
    
    # Create input tensor (N, T, D)
    ipt = torch.randn(N, T, D)  
    print(f"Input tensor (x): {ipt.shape}")

    # Embedded context vector (N, 1, D)
    context_emb = torch.zeros(N, 1, D)  
    print(f"Context emb (before concat): {context_emb.shape}")
    print(f"Context tensor (context_emb): {context_emb}")
    
    # Concatenate (N, T+1, D)
    x = torch.cat([ipt, context_emb], dim=1)  
    print(f"Cat output (after concat): {x.shape}")
    print(f"Cat tensor (x): {x}")
    
    # Remove context token (N, T, D)
    x = x[:, :-1, :]  
    print(f"Removed context (after removal): {x.shape}")
    print(f"Output tensor (x): {x}")
