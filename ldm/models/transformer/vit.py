# Code adpated from:
# - https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
import torch
import torch.nn as nn

from typing import Optional, Type

import torch
from torch.jit import Final
import torch.nn as nn
import torch.nn.functional as F


from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
)

from ldm.models.transformer.hilo import HiloBlock


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

#################################################################################
#                                 Core ViT Model                                #
#################################################################################
 
    
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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
    

class ViTBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
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
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ViTResPostBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.init_values = init_values
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x
    
    
    
    
    


#################################################################################
#                                 Skip Layers                                   #
#################################################################################

class SimpleSkipLayer(nn.Module):
    """
    Simple skip layer.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, skip_input):
        skip_input = self.linear(skip_input)
        return x + skip_input
    
class LowRankSkipLayer(nn.Module):
    """
    Low-rank skip layer.
    """
    def __init__(self, embed_dim, rank=None):
        super().__init__()
        if rank is None:
            rank = max(128, embed_dim // 2)
        self.compress = nn.Linear(embed_dim, rank)
        self.expand = nn.Linear(rank, embed_dim)

    def forward(self, x, skip_input):
        skip_compressed = self.compress(skip_input)
        skip_expanded = self.expand(skip_compressed)
        return x + skip_expanded
    

class ConcatSkipLayer(nn.Module):
    """
    Concatenation skip layer with optional compression.
    """
    def __init__(self, embed_dim, compress=False, rank=None):
        super().__init__()
        if compress:
            if rank is None:
                rank = max(128, embed_dim // 2)
        else:
            rank = embed_dim

        self.skip_linear = nn.Linear(2 * embed_dim, rank)
        self.expand = nn.Linear(rank, embed_dim) if compress else nn.Identity()


    def forward(self, x, skip_input):
        x = self.skip_linear(torch.cat([x, skip_input], dim=-1))             
        x = self.expand(x)                    
        return x



class AttentionSkipLayer(nn.Module):
    """
    Cross-attention skip layer with residual connections and dropout.
    Helps improve feature fusion by attending to skip features.
    """
    def __init__(self, embed_dim, dropout=0., proj_drop=0., num_heads=2, batch_first=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=proj_drop, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, skip_input):
        # Perform attention and add dropout for regularization
        attn_output, _ = self.attn(self.norm1(x), self.norm2(skip_input), self.norm2(skip_input))
        return x + self.dropout(attn_output)

    
    
class HiloSkipLayer(nn.Module):
    """
    HiLo skip layer with improved modularity.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, net_type='decoder'):
        super().__init__()
        self.net_type = net_type
        self.hilo = HiloBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

    def forward(self, x, skip_input):
        class_token, patch_token = skip_input[:, :1], skip_input[:, 1:]

        if self.net_type == 'encoder':
            # Compute residual for encoder
            B, N, C = patch_token.shape
            H = W = int(N ** 0.5)
            assert H * W == N, "Invalid input size after removing class token"
            
            self.hilo.H = H
            self.hilo.W = W
            
            patch_token = self.hilo(patch_token, H, W)
            residual = torch.cat([class_token, patch_token], dim=1)

        elif self.net_type == 'decoder':
            # Compute residual for decoder
            B, N = class_token.shape
            H = W = int(N ** 0.5)
            assert H * W == N, "Invalid input size after removing patch tokens"
            
            self.hilo.H = H
            self.hilo.W = W
            
            class_token = self.hilo(class_token)
            residual = torch.cat([class_token, patch_token], dim=1)

        return x + residual


###############################
# Legacy code
###############################
class CompiledBlock(nn.Module):
    """ Vision Transformer block with layer scale and drop path.
        Allows for JIT compilation of the attention layer in the block.
    """
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
            skip=False,
            skip_layer_type='default',
            use_checkpoint=False
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # Jit compile attention layer
        self.attn = self.compile(attn)
        
        self.skip = skip
        self.use_checkpoint = use_checkpoint
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
        
        # Define skip layer
        if skip_layer_type == 'default':
            # Based on U-ViT implementation
            self.skip_layer = ConcatSkipLayer(dim) if self.skip else nn.Identity()
        elif skip_layer_type == 'simple':
            self.skip_layer = SimpleSkipLayer(dim) if self.skip else nn.Identity()
        elif skip_layer_type == 'low_rank':
            self.skip_layer = LowRankSkipLayer(dim) if self.skip else nn.Identity()
        elif skip_layer_type == 'attention':
            skip_layer = AttentionSkipLayer(dim, dropout=proj_drop, proj_drop=proj_drop) if self.skip else nn.Identity()
            self.skip_layer = self.compile(skip_layer)
        else:
            raise ValueError(f"Invalid skip layer type: {skip_layer_type}")
        
        
    def forward(self, x, skip=None):
        # taken from: https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L138
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)
        
        
    def _forward(self, x: torch.Tensor, skips=None) -> torch.Tensor:
        if self.skip and skips is not None:
            x = self.skip_layer(x, skips)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
    
    def compile(self, layer):
        """Compile layer with JIT for faster execution."""
        # Nvidia Jetson Xavier NX does not support inductor backend
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            backend = 'inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager'
        else:
            backend = 'aot_eager'  # Default for CPU-only mode
        compiled_layer = torch.compile(layer, fullgraph=True, backend=backend)
        return compiled_layer
    