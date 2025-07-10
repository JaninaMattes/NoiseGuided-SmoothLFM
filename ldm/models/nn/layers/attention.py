# Code adapted from:
# - https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/mlp.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
import os
import math
import sys
import warnings

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


from torch.jit import Final

from timm.layers import use_fused_attn



""" Layer/Module Helpers """

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module




""" XFormers Fused Attention """

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")




#################################################
#                Standard Attention             #
#################################################

""" Attention Layers """

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.num_headsead_dim = dim // num_heads
        self.scale = self.num_headsead_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.num_headsead_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.num_headsead_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.num_headsead_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



#################################################
#            MemoryEfficient Attention          #
#################################################
# Code adapted from:
# - https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py


""" Dinov2 based Attention Layers """


class DinoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    
# Dinov2 Attention with Memory Efficient Attention
class MemEffAttention(DinoAttention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



#################################################
#               QKV Attention SiT               #
#################################################


""" QKV Attention Layers """  
        
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """
    def __init__(self, efficient_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.embed_dimropout = dropout
        self.efficient_attn = efficient_attn
        if self.efficient_attn:
            try:
                _ = nn.functional.scaled_dot_product_attention
            except AttributeError:
                print("Please update PyTorch to 2.0 or higher to use efficient attention.")
                self.efficient_attn = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q, k, v: (n, ..., l, c) tensors of Queries, Keys, Values. The ...
                can be any number of batch dimensions (e.g. heads).
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        if self.efficient_attn:
            res = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.embed_dimropout)
        else:
            ch = q.shape[-1]
            scale = 1. / math.sqrt(ch)
            dot = torch.einsum('...td, ...kd -> ...tk', q, k) * scale
            weight = torch.softmax(dot, dim=-1)
            if self.embed_dimropout > 0.0:
                weight = torch.dropout(weight, p=self.embed_dimropout, train=self.training)
            res = torch.einsum('...dt, ...tv -> ...dv', weight, v)
        return res


class LinearQKVAttention(nn.Module):
    """
    A module which performs linear QKV attention.
    (https://arxiv.org/abs/1812.01243)
    """
    def __init__(self, l2_norm_v: bool = False):
        super().__init__()
        self.l2_norm_v = l2_norm_v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q, k, v: (n, ..., l, c) tensors of Queries, Keys, Values. The ...
                can be any number of batch dimensions (e.g. heads).
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        ch = q.shape[-1]
        scale = 1. / math.sqrt(ch)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * scale
        if self.l2_norm_v:
            v = torch.nn.functional.normalize(v, dim=-1)
        context = torch.einsum('...nd, ...ne -> ...de', k, v)
        res = torch.einsum('...nd, ...de -> ...ne', q, context)
        return res



class SpatialSelfAttention(nn.Module):
    """
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64,
                 use_linear: bool = False, use_efficient_attn: bool = True):
        super().__init__()
        self.embed_dimim = dim
        self.num_headseads = heads
        self.embed_dimim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv1d(dim, self.inner_dim * 3, 1)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(nn.Conv1d(self.inner_dim, self.embed_dimim, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (b, c, *spatial), where spatial can be (f, h, w) or (h, w).
        Returns:
            x: Tensor after attention, MHSA(x) + residual.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)                                     # (b, c, f * h * w)
        qkv = self.qkv(self.norm(x))                                # (b, 3 * c * nh, f * h * w)
        qkv = qkv.reshape(b, self.num_headseads, qkv.shape[-1], -1)         # (b, nh, f * h * w, 3 * c)
        q, k, v = qkv.chunk(3, dim=-1)                              # (b, nh, f * h * w, c) each
        h = self.attention(q, k, v)                                 # (b, nh, f * h * w, c)
        h = h.reshape(b, self.inner_dim, -1)                        # (b, nh * c, f * h * w)
        h = self.proj_out(h)                                        # (b, c, f * h * w)
        return (x + h).reshape(b, c, *spatial)




#################################################
#       Causal Self Attention with Masking      #
#################################################

# Code adapted from:
# - https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/masks/causal.py
# - https://gist.github.com/wolfecameron/26863dbbc322b15d2e224a2569868256
# - https://www.kaggle.com/code/aisuko/causal-self-attention


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


"""Standard Causal Attention with optional Nested Dropout Masking."""


class CausalSelfAttention(nn.Module):
    """ Standard Causal Attention with Masking. """
    
    def __init__(self, embed_dim, num_heads, max_seq_len=256, bias=False, dropout=0.1, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        
        # key, query, value projections for all heads in batch 
        # output is 3x the dimension for key, query, value per head
        self.c_attn = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # dropout module
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # torch.tril for lower triangular matrix (causal masking)
        # causal mask to ensure that attention is only applied to previous tokens (left in the input sequence)
        self.register_buffer("mask", torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).view(1, 1, self.max_seq_len, self.max_seq_len))
        
        
    def forward(self, x):
        B, T, D = x.size()                                                  # B: batch size, T: num tokens, D: embedding dimensionality

        # Compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)              # [B, T, d]

        # Reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) # [B, H, T, d // H]
        q = q.view(B, T, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

        # Masking Additional Attention Weights With Dropout
        # compute the attention matrix, perform masking, and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [B, H, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # compute output vectors for each token
        y = att @ v # [B, H, T, d // H]

        # Concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        y = self.resid_dropout(self.c_proj(y))
        
        return y






if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Attention Layers
    ipt = torch.randn((1, 32, 8, 8))
    print("Input:", ipt.shape)
    attn = SpatialSelfAttention(32, heads=4, dim_head=64, use_linear=True)
    print(f"Params: {sum(p.numel() for p in attn.parameters()):,}")
    out = attn(ipt)
    print("Output:", out.shape)
    
    
    # Test Causal Self Attention
    ipt = torch.randn((1, 256, 384))
    print("Input:", ipt.shape)
    attn = CausalSelfAttention(384, num_heads=8, max_seq_len=256).to(dev)
    print(f"Params: {sum(p.numel() for p in attn.parameters()):,}")
    out = attn(ipt.to(dev))
    print("Output:", out.shape)