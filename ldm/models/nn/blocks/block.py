# Code adapted from:
# - https://github.com/facebookresearch/dinov2/blob/main/dinov2/block.py    # vision_transformer.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py
# 
import os
import warnings
import numpy as np

from typing import Optional

import torch
import torch.nn as nn

from timm.models.vision_transformer import Attention as LegacyAttention

from ldm.models.nn.layers.mlp import Mlp
from ldm.models.nn.layers.layer_scale import LayerScale
from ldm.models.nn.layers.dropout.drop import DropPath

from ldm.models.architecture.default_architectures import AttentionSkipLayer, ConcatSkipLayer, LowRankSkipLayer, SimpleSkipLayer
from ldm.models.nn.layers.attention import Attention
from ldm.models.nn.layers.layer_scale import LayerScale


""" XFormers Fused Attention """

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Block)")

    
    
""" Helper Functions """

class View(nn.Module):
    """Reshapes input tensor."""
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Embedding(nn.Module):
    """Embedding layer."""
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


#################################################################################
#                                 Core ViT Model                                #
#################################################################################

class ResidualModule(nn.Module):
    def __init__(
            self,
            inner_module: nn.Module
        ):
        super().__init__()
        self.inner_module = inner_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inner_module(x)


class FeedForwardBlock(nn.Module):
    """ FeedForwardBlock class for the MLP of a transformer block in the Transformer Encoder.
        The linear MLP layers are local and translationally equivariant,
        while the self-attention layers are global and permutation invariant.

        Args:
            hidden_size (int): Hidden size of the model.
            mlp_size (int): Size of the MLP.
            p_dropout (float): Dropout probability.

        Returns:
            torch.Tensor: Output tensor of the feedforward block.
    """

    def __init__(
            self,
            hidden_size: int,
            mlp_size: int,
            p_dropout: float
        ):
        super().__init__()
        self.dropout = p_dropout
        self.hidden_size = hidden_size # kept fixed
        self.mlp_size = mlp_size

        self.linear1 = nn.Linear(self.hidden_size, self.mlp_size)
        self.dropout1 = nn.Dropout(self.dropout)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.mlp_size, self.hidden_size)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class SelfAttentionTransformerBlock(nn.Module):
    """ SelfAttentionTransformerBlock class for the transformer block in the Transformer Encoder.
        The linear MLP layers are local and translationally equivariant,
        while the self-attention layers are global and permutation invariant.

        Args:
            hidden_size (int): Hidden size of the model.
            num_heads (int): Number of heads in the multi-head self-attention.
            p_dropout (float): Dropout probability.

        Returns:
            torch.Tensor: Output tensor of the transformer block.
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            p_dropout: float
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.mlp_size = self.hidden_size * 4 # Standard in the literature

        # Layer normalization - important to stabilize training
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=p_dropout, batch_first=True)

        # MLP block
        self.mlp = FeedForwardBlock(self.hidden_size, self.mlp_size, self.p_dropout)
        self.dropout = nn.Dropout(self.p_dropout)

  
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Residual connection
        x = x

        # Self-attention (global and permutation invariant)
        x = self.norm1(x)
        x = self.mha(x, x, x)[0]
        x = self.dropout(x)
        
        # Residual connection
        x = x + x

        # Residual connection
        x = x

        # MLP (local and translationally equivariant)
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        # Residual connection
        x = x + x

        return x



#################################################################################
#                                 Core DiT Model                                #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, legacy=False, drop=0, **block_kwargs):
        super().__init__()
        attn_class = LegacyAttention if legacy else Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=drop)
        
        # Learnable parameters for unconditional embedding
        self.unconditional_embedding = nn.Parameter(torch.randn(hidden_size), requires_grad=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c=None):
        """ 
        Zero conditioned adaptive layer normalization if no conditioning is provided.
        """
        # Learnable unconditional embedding if no conditioning is provided
        if c is None:
            c = self.unconditional_embedding.unsqueeze(0).expand(x.size(0), -1)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
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


# ------------------------------------
# Autoencoder Blocks
# ------------------------------------
class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    
# ------------------------------------
# Linear Blocks
# ------------------------------------
class LinearBlock(nn.Module):
    """ Linear Block with flexible activation and normalization layers.
        taken from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
    """
    def __init__(self, 
                 in_features, 
                 out_features, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.GroupNorm,
                 group_num=32,
                 bias=True,
                 drop=0., 
                 nonlinearity=True):
        super(LinearBlock, self).__init__()
        if norm_layer is nn.GroupNorm and out_features % group_num != 0:
          raise ValueError(f"`group_num` ({group_num}) must evenly divide `out_features` ({out_features}).")

        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            act_layer() if nonlinearity else nn.Identity(),
            nn.Dropout(drop),
            norm_layer(group_num, out_features) if norm_layer is not None else nn.Identity()
        )
        
    def forward(self, x):
        return self.linear(x)
    
    
    
class LinearResBlock(nn.Module):
    """ Linear Residual Block with bottleneck architecture and pre-act_layer.
        taken inspiration from: https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
    """
    
    def __init__(self, in_features, out_features, act_layer=nn.SiLU, bottleneck_ratio=0.25, 
                 downsample: Optional[nn.Module] = None, drop_rate=0.0):
        super(LinearResBlock, self).__init__()
        
        bottleneck_features = int(out_features * bottleneck_ratio)
        
        self.act_layer = act_layer()
        self.drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()

        # Block 1: First bottleneck layer (reduce dimensions)
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_features),
            self.act_layer,
            nn.Linear(in_features, bottleneck_features),
        )

        # Block 2: Second bottleneck layer (expand dimensions)
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, bottleneck_features),
            self.act_layer,
            self.drop,
            nn.Linear(bottleneck_features, out_features),
        )

        # Shortcut (skip connection)
        if in_features != out_features or downsample is not None:
            self.skip_connection = downsample if downsample is not None else nn.Sequential(
                nn.Linear(in_features, out_features),
            )
        else:
            self.skip_connection = nn.Identity()


    def forward(self, x):
        identity = x

        # Forward pass 
        x = self.in_layers(x)
        x = self.out_layers(x)
        
        # Shortcut connection
        return self.skip_connection(identity) + x
    
    
# ------------------------------------
# Convolutional Blocks
# ------------------------------------
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act_layer):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.GroupNorm(32, out_channels)
        self.act_layer = act_layer() if callable(act_layer) else act_layer

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)
        return x

class UpsampleBlock(nn.Module):
    """Upsample block with flexible act_layer."""
    def __init__(self, in_channels, out_channels, scale_factor, mode='bilinear', groups=8, act_layer=nn.GELU):
        super(UpsampleBlock, self).__init__()
        
        # Pass arguments to ConvBlock with dynamic group validation
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, act_layer=act_layer)
        )

    def forward(self, x):
        return self.block(x)

    
class TransposeConvBlock(nn.Module):
    """Transpose convolutional block with flexible act_layer."""
    # H_out​=(Hin​−1)* stride − 2*padding+kernel_size + output_padding
    # W_out​=(Win​−1)* stride − 2*padding+kernel_size + output_padding
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=8, act_layer=nn.GELU):
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(groups, out_channels), # Better than BatchNorm for small batch sizes
            act_layer() if callable(act_layer) else act_layer 
        )

    def forward(self, x):
        return self.block(x)
    

class InterpolateConvBlock(nn.Module):
    """Interpolates and applies convolutional block with bilinear mode."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, size, act_layer=nn.GELU):
        super(InterpolateConvBlock, self).__init__()
        self.block = nn.Sequential(
            Interpolate(size=size, mode='bilinear', align_corners=True),
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding, act_layer),
            nn.GroupNorm(32, out_channels),
            act_layer() if callable(act_layer) else act_layer # Initialize act_layer function
        )

    def forward(self, x):
        return self.block(x)


class Interpolate(nn.Module):
    """ Interpolates input tensor.
        Source: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2

        Args:
            size: size of the output tensor
            mode: interpolation mode
            align_corners: align corners can lead to issues if True with conv.
    """
    def __init__(self, size, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)


# ------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with bottleneck architecture and pre-act_layer.
    Supports same-scale, downscaling, and upscaling operations.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        act_layer (callable): Activation function (default: LeakyReLU).
        scale (str): Scaling mode - 'same', 'downscale', or 'upscale'.
        bias (bool): Whether to use bias in convolution layers (default: False).
    """
    def __init__(self, in_channels, out_channels, act_layer=nn.LeakyReLU, scale="same", bias=False):
        super(ResBlock, self).__init__()
        # Activation function
        self.act_layer = act_layer() if callable(act_layer) else act_layer

        # Bottleneck channel size
        mid_channels = max(in_channels // 2, out_channels // 2)

        # Define convolution layers based on scaling mode
        if scale == "same":
            stride = 1
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=bias)
        elif scale == "downscale":
            stride = 2
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=bias)
        elif scale == "upscale":
            stride = 1
            self.conv2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=bias)
        else:
            raise ValueError("Scale must be one of ['same', 'downscale', 'upscale']")

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=bias)

        # Sequential block of operations
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            self.act_layer,
            self.conv1,
            nn.GroupNorm(32, mid_channels),
            self.act_layer,
            self.conv2,
            nn.GroupNorm(32, mid_channels),
            self.act_layer,
            self.conv3
        )

        # Identity/skip connection
        if in_channels != out_channels or scale in ["downscale", "upscale"]:
            if scale == "upscale":
                self.id = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
                )
            else:
                self.id = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
        else:
            self.id = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.id(x)

# ------------------------------------
# Vision Transformer Blocks
# ------------------------------------
class ResidualModule(nn.Module):
    def __init__(
            self,
            inner_module: nn.Module
        ):
        super().__init__()
        self.inner_module = inner_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inner_module(x)

class FeedForwardBlock(nn.Module):
    """FeedForwardBlock for the Transformer Encoder."""
    def __init__(self, hidden_size: int, mlp_size: int, p_dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionTransformerBlock(nn.Module):
    """ SelfAttentionTransformerBlock class for the transformer block in the Transformer Encoder.
        The linear MLP layers are local and translationally equivariant,
        while the self-attention layers are global and permutation invariant.

        Args:
            hidden_size (int): Hidden size of the model.
            num_heads (int): Number of heads in the multi-head self-attention.
            p_dropout (float): Dropout probability.

        Returns:
            torch.Tensor: Output tensor of the transformer block.
    """

    def __init__(self, hidden_size: int, num_heads: int, p_dropout: float, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Multi-head self-attention
        # Note: The `batch_first=True` results in (batch, seq, feature) output shape
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=p_dropout, batch_first=True)

        # Feed-Forward Block
        # self.mlp = FeedForwardBlock(hidden_size, mlp_hidden_dim, p_dropout)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=p_dropout)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: Self-Attention with Residual Connection
        x = x
        x = self.norm1(x)
        x = self.mha(x, x, x)[0]
        x = self.dropout(x)
        x = x + x

        # Block 2: MLP with Residual Connection
        x = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + x

        return x


class DiTAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, p_dropout=0., **block_kwargs):
        super().__init__()
        
        # Learnable Layer Normalization with adaptive parameters
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        
        # Multi-Head Self-Attention
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        # Adaptive Scaling Mechanism
        self.scale1 = nn.Parameter(torch.ones(hidden_size))
        self.shift1 = nn.Parameter(torch.zeros(hidden_size))
        
        self.scale2 = nn.Parameter(torch.ones(hidden_size))
        self.shift2 = nn.Parameter(torch.zeros(hidden_size))
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=approx_gelu, 
                       drop=p_dropout)
    
    def forward(self, x):
        # Block1: Self-attention with Adaptive Normalization
        x = x
        x = self.norm1(x)
        x = self.scale1 * x + self.shift1  # Adaptive scaling
        x = self.attn(x)
        x = x + x
        
        # Block 2: MLP with Adaptive Normalization
        x = x
        x = self.norm2(x)
        x = self.scale2 * x + self.shift2  # Adaptive scaling
        x = self.mlp(x)
        x = x + x
        
        return x
    
# ------------------------------------
# Vision Transformer Blocks (Timm)
# https://github.com/baofff/U-ViT/blob/main/libs/uvit.py
# Github:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
# ------------------------------------ 


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# Code adapted from:
# - https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L95
class BlockSkip(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp, 
            skip: bool = False,  
            use_checkpoint: bool = False
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer
        )
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


        
""" Compiled Block with Layer Scale and Drop Path """

class BlockUVit(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    

# ------------------------------------
# Vision Transformer Blocks
# ------------------------------------
class VisionTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 4,
            image_size: int = 32,
            layers: int = 6,
            hidden_size: int = 256,
            mlp_size: int = 512,
            num_heads: int = 8,
            num_classes: int = 10,
            p_dropout: float = 0.2,
        ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.layers = layers
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.p_dropout = p_dropout


        # ---------------------------------
        # ------ Transformer Encoder ------
        # ---------------------------------

        # Image patches / token
        self.num_patches = (self.image_size // self.patch_size) ** 2 # Number of patches (L) or tokens
        self.patch_dim = self.in_channels * (self.patch_size ** 2)   # Dimension of the patch after flattening (D)

        # Patch embedding - linear projection of the patches
        self.patch_embed = nn.Linear(self.patch_dim, self.hidden_size)

        # Positional encoding - learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))

        # CLS token - classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(*[
            SelfAttentionTransformerBlock(self.hidden_size, self.num_heads, self.p_dropout)
            for _ in range(self.layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(self.p_dropout)

        # Classification head
        self.norm = nn.LayerNorm(self.hidden_size)
        self.classification_head = nn.Linear(self.hidden_size, self.num_classes)

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        # Initialize weights
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        # Initialize weights of the model
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)


    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes an image tensor of shape (B, C, H, W) and transforms it to a sequence of patches (B, L, D), 
        with a learnable linear projection after flattening, and a standard additive positional encoding applied. 
        
        Note that the act_layers in (Vision) Transformer implementations are typically passed around in channels-_last_ layout, 
        different from typical PyTorch norms.

        The linear projection of flattened image patches produces lower-dimensional linear embddings from flattened patches and adds positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Embedded patch sequence tensor with positional encodings applied and shape (B, L, D)
        """
        B, C, H, W = x.shape

        # Reshape and flatten the image patches
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()             # Size: (B, H, W, C, patch_size, patch_size)
        x = x.view(B, -1, C * self.patch_size * self.patch_size) # Size: (B, L, D) with D: C * patch_size * patch_size

        # Linear projection of the patches
        x = self.patch_embed(x)

        # Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]

        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings for the CLS token
        x[:, 0, :] = x[:, 0, :] + self.pos_embed[:, 0, :]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Takes an image tensor of shape (B, C, H, W), applies patching, a standard ViT
           and then an output projection of the CLS token
           to finally create a class logit prediction of shape (B, N_cls)

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output logits of shape (B, N_cls)
        """
        # Patchify input image + pos embeddings
        x = self.patchify(x)

        # Apply dropout
        x = self.dropout(x)

        # Transformer blocks
        x = self.transformer_blocks(x)

        # Classification head
        x = self.norm(x[:, 0])              # select only the learned CLS token with Size: (B, D)
        x = self.classification_head(x)

        return x
    





# ------------------------------------
# ResMLP Architecture Blocks
# source: https://github.com/rishikksh20/ResMLP-pytorch
# ------------------------------------
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

# Helpers
def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class PreAffinePostLayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x

def ResMLP(*, image_size, patch_size, dim, depth, num_classes, expansion_factor=4):
    image_height, image_width = pair(image_size)
    assert (image_height % patch_size) == 0 and (image_width % patch_size) == 0, 'image height and width must be divisible by patch size'
    num_patches = (image_height // patch_size) * (image_width // patch_size)
    wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[nn.Sequential(
            wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
            wrapper(i, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Linear(dim * expansion_factor, dim)
            ))
        ) for i in range(depth)],
        Affine(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
    