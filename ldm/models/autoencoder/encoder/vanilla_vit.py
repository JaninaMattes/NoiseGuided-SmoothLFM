# Code adapted from:
# - https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L95
# - https://github.com/facebookresearch/DiT/blob/main/models.py
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.layers import use_fused_attn

from ldm.models.transformer.vit import CompiledBlock
from ldm.models.architecture.base_architectures import BaseEncoder
from ldm.models.nn.out.outputs import AEEncoderOutput

from ldm.models.nn.layers.dropout.drop import DropPath
from ldm.models.nn.layers.layer_scale import LayerScale
from ldm.models.nn.layers.mlp import Mlp
from ldm.models.nn.layers.patch_embed import PatchEmbed
from ldm.models.autoencoder.nn import get_2d_sincos_pos_embed
from ldm.tools.fourier.fft import FrequencyUnit
    
    
######################################
# Legacy Code
######################################from functools import partial

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
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        out_features = int(latent_dim * moments_factor)
        self.linear = nn.Linear(embed_dim, out_features, bias=True)
        
    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

        
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
                 qkv_bias=True, 
                 qk_scale=None, 
                 norm_layer=nn.LayerNorm, 
                 act_layer: nn.Module = nn.GELU,
                 pos_embed_learned=False, 
                 use_fft_features=False, 
                 use_weighted_fft=False, 
                 skip=False, 
                 skip_layer_type='default', 
                 use_checkpoint=False,
                 moments_factor=1    # Factor to multiply the embedding dimension by (for VAE, mu and logvar)
            ):
        super().__init__()
        self.model_name = "VanillaViTEncoder"
        
        self.skip = skip
        self.use_weighted_fft = use_weighted_fft
        self.use_fft_features = use_fft_features
        self.use_checkpoint = use_checkpoint   
             
        self.pos_embed_learned = pos_embed_learned
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim, bias=True)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
        # Positional Embedding
        self.extra_tokens = 1               # class token
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + self.extra_tokens, embed_dim) if pos_embed_learned else
            torch.zeros(1, self.num_patches + self.extra_tokens, embed_dim), requires_grad=pos_embed_learned
        )

        # Transformer blocks (default: 12)
        self.blocks = nn.ModuleList([
            CompiledBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_scale, norm_layer=norm_layer,
                          act_layer=act_layer, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path,
                          skip=skip, skip_layer_type=skip_layer_type, use_checkpoint=use_checkpoint)
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
        self._init_weights()   
            
        # Jit compile the forward function
        self._compile()
        
    
    def forward(self, x):
        """ Forward pass """
        return self.compiled_forward(x)
    
    
    def _forward(self, x):
        B = x.shape[0]

        # Embed patches
        x = self.patch_embed(x)                                                         # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]                                                # Add positional embedding

        # Add class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)                                           # (B, num_patches + 1, embed_dim)                              

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)                                                                  # Add long-range skip information

        # Extract class token
        x = x[:, 0]                                                                     # Extract class token
        x = self.final_layer(x)
        
        return AEEncoderOutput(z_enc=x, skips=None)
        
        
    def _init_weights(self):
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
            pose_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True, extra_tokens=self.extra_tokens)
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
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def _compile(self):
        """JIT-compile PyTorch code into optimized kernels if available."""
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            backend = 'inductor' if major >= 7 else 'aot_eager'
        else:
            backend = 'aot_eager'  # Default to a safe backend for CPU

        # Compile forward function
        compile_fn = partial(torch.compile, fullgraph=True, backend=backend)
        self.compiled_forward = compile_fn(self._forward)