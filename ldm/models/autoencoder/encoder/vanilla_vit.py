# Code adapted from:
# - https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L95
# - https://github.com/facebookresearch/DiT/blob/main/models.py
from functools import partial
from typing import Optional

from torch.jit import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import use_fused_attn

from ldm.models.nn.layers.dropout.drop import DropPath
from ldm.models.nn.layers.layer_scale import LayerScale
from ldm.models.nn.layers.mlp import Mlp
from ldm.models.nn.layers.patch_embed import PatchEmbed
from ldm.models.architecture.base_architectures import BaseEncoder
from ldm.models.autoencoder.nn import get_2d_sincos_pos_embed
from ldm.tools.fourier.fft import FrequencyUnit
from ldm.models.nn.out.outputs import AEEncoderOutput

    
""" Jit Compile """

COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=True, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f
    

""" ViT Encoder """
# ------------------------------------
# Vision Transformer Blocks (Timm)
# https://github.com/baofff/U-ViT/blob/main/libs/uvit.py
# Github:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py


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

        if COMPILE: self.forward = compile_fn(self.forward)
        
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

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    

class FinalLayer(nn.Module):
    """
    The final layer of the ViT encoder.
    Args:
        latent_dim (int): The dimension of the latent space.
        embed_dim (int): The dimension of the embedding space.
        moments_factor (float): The factor to multiply the embedding dimension by (for VAE, mu and logvar).
    """
    def __init__(self, latent_dim, embed_dim, moments_factor=1):
        super().__init__()
        out_features = int(latent_dim * moments_factor)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_features)
        )

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, x):
        return self.mlp(x)



class VanillaViTEncoder(BaseEncoder):
    """ 
    Vision Transformer (ViT) decoder with skip connections.
    The implementation is based on the ViT model from timm.
    """
    def __init__(self, 
                 image_size=32, 
                 patch_size=1, 
                 in_channels=4,
                 latent_dim=512, 
                 embed_dim=512, 
                 num_layers=12, 
                 num_heads=16,
                 mlp_ratio: float = 4.,
                 attn_drop: float = 0., 
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 qkv_bias: bool = False,
                qk_scale: bool = False,
                 norm_layer=nn.LayerNorm, 
                 act_layer: nn.Module = nn.GELU,
                 pos_embed_learned=False, 
                 use_fft_features=False, 
                 use_weighted_fft=False, 
                 use_checkpoint=False,
                 moments_factor=1,    # Factor to multiply the embedding dimension by (for VAE, mu and logvar)
                compile=True
            ):
        super(VanillaViTEncoder, self).__init__()
        global COMPILE
        COMPILE = compile
        self.use_weighted_fft = use_weighted_fft
        self.use_fft_features = use_fft_features
        self.use_checkpoint = use_checkpoint   
             
        self.pos_embed_learned = pos_embed_learned
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim, bias=True)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
        # Positional Embedding
        self.extras = 1               # class token
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + self.extras, embed_dim) if pos_embed_learned else
            torch.zeros(1, self.num_patches + self.extras, embed_dim), requires_grad=pos_embed_learned
        )

        # Transformer blocks (default: 12)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_norm=qk_scale, 
                proj_drop=proj_drop, 
                attn_drop=attn_drop, 
                drop_path=drop_path, 
                act_layer=act_layer, 
                norm_layer=norm_layer  
            )
            for _ in range(num_layers)
        ])

        self.final_layer = FinalLayer(latent_dim, embed_dim, moments_factor)
     
        
        # Optional: FFT features
        if use_fft_features:
            self.fourier_unit = FrequencyUnit()
            self.norm_layer = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-5)
        
        # Optional: Weighted FFT features
        if self.use_weighted_fft:
            self.beta_weight = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            
        # Initialize weights     
        self.initialize_weights()   

    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                                                         # (B, num_patches, embed_dim)
        x = x + self.pos_embed[:, 1:, :]                                                # Add positional embedding

        # Add class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)                                           # (B, num_patches + 1, embed_dim)                              

        for blk in self.blocks:
            x = blk(x)                                                                  

        # Extract class token
        x = x[:, 0]                                                                     # Extract class token
        x = self.final_layer(x)
        return AEEncoderOutput(z_enc=x, skips=None)
        
        
    def initialize_weights(self):
        # taken from: https://github.com/facebookresearch/DiT/blob/main/models.py
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # initialization
        if self.pos_embed_learned:
            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
            nn.init.normal_(self.pos_embed, mean=0, std=.02)
        else:    
            pose_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True, extra_tokens=self.extras)
            self.pos_embed.data.copy_(torch.from_numpy(pose_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # initialize cls token
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # Initialize skip layers weights
        if self.use_weighted_fft:
            torch.nn.init.normal_(self.beta_weight, std=.02)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.mlp[-1].weight, 0)
        nn.init.constant_(self.final_layer.mlp[-1].bias, 0)

