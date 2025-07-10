from functools import partial
import os
import sys

import torch
import torch.nn as nn
from typing import Optional


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(project_root)


from ldm.models.nn.layers.mlp import Mlp
from ldm.models.nn.layers.layer_scale import LayerScale
from ldm.models.nn.layers.dropout.drop import DropPath
from ldm.models.nn.layers.dropout.nested_drop import NestedDropout
from ldm.models.nn.layers.attention import CausalSelfAttention


    
""" Causal Attention Transformer Block with Nested Dropout.
    This block is used in the FlexTok-inspired Tokenizer Module.
"""

class CattBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        max_seq_len: int = 256, 
        bias: bool = False, 
        dropout: float = 0.1, 
        nested_dropout: bool = False,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,  
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        act_layer: Optional[nn.Module] = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        nested_drop_layer: nn.Module = NestedDropout,
        proj_drop: float = 0.0
    ):
        """ Causal Attention Transformer Block with Nested Dropout.
        
        Args:
            dim (int): Number of features in the input tensor.
            num_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length.
            bias (bool): Whether to include bias in the attention calculation.
            dropout (float): Dropout rate.
            nested_dropout (bool): Whether to include nested dropout.
            norm_layer (nn.Module): Normalization layer.
            init_values (float): Initial values for the LayerScale.
            drop_path (float): Drop path rate.
            mlp_ratio (float): Ratio of the hidden layer size to the input size.
            act_layer (nn.Module): Activation layer.
            proj_drop (float): Dropout rate after the projection
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, bias=bias, dropout=dropout)
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
        self.nested_dropout = nested_drop_layer() if nested_dropout else None
        self.max_seq_len = max_seq_len
        

    def forward(self, x, k_keep=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if self.nested_dropout is not None:
            x = self.nested_dropout(x, k_keep=k_keep)
        return x





if __name__ == '__main__':
    # Test CattBlock
    block = CattBlock(dim=384, num_heads=8, max_seq_len=256, nested_dropout=True)
    tokens = torch.randn(1, 256, 384)  # (Batch, Sequence_len, Embedding_dim)
    
    print("Original Tokens Shape:", tokens.shape)
    print(f"Original Tokens:\n{tokens}")

    masked_tokens = block(tokens)
    print("Dropped Tokens Shape:", masked_tokens.shape)
    print(f"Dropped Tokens:\n{masked_tokens}")
    assert tokens.shape == masked_tokens.shape, "Shape Mismatch"
