from functools import partial
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.vision_transformer import PatchEmbed

from ldm.models.nn.layers.patch_embed import PatchEmbed
from ldm.models.architecture.base_architectures import BaseEncoder
from ldm.models.autoencoder.nn import get_2d_sincos_pos_embed
from ldm.models.transformer.vit import CompiledBlock
from ldm.tools.fourier.fft import FrequencyUnit
from ldm.models.nn.out.outputs import AEEncoderOutput


##########################
# Legacy code
##########################

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

        
class SkipViTEncoder(BaseEncoder):
    """ 
    Vision Transformer (ViT) encoder with skip connections.
    The implementation is based on the ViT model from timm.
    """
    def __init__(self, 
                 in_channels=4, 
                 image_size=32, 
                 patch_size=1, 
                 latent_dim=512, 
                 embed_dim=512, 
                 num_layers=12, 
                 num_heads=16,
                 mlp_ratio: float = 4.,
                 attn_drop: float = 0., 
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer=nn.LayerNorm, 
                 act_layer: nn.Module = nn.GELU,
                 pos_embed_learned=False, 
                 use_fft_features=False, 
                 use_weighted_fft=False, 
                 skip=True, 
                 skip_layers=None,
                 skip_layer_type='default', 
                 use_checkpoint=False, 
                 moments_factor=1 # for VAE - mu and logvar
                 ):

        super(SkipViTEncoder, self).__init__()
        self.skip = skip
        self.skip_layers = skip_layers if skip_layers is not None else []       # List of skip-layers                            # List of layers to apply skip connections  
        self.use_weighted_fft = use_weighted_fft
        self.use_fft_features = use_fft_features
        self.use_checkpoint = use_checkpoint   
             
        self.pos_embed_learned = pos_embed_learned
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim, bias=True)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if use_fft_features:
            self.fourier_unit = FrequencyUnit()
            self.norm_layer = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-5)
            
        # Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) if pos_embed_learned else
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=pos_embed_learned
        )

        # Transformer blocks (default: 12)
        self.blocks = nn.ModuleList([
            CompiledBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                          act_layer=act_layer, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path,
                          skip=skip, skip_layer_type=skip_layer_type, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])

        self.final_layer = FinalLayer(latent_dim, embed_dim, moments_factor)
        
        # Optional: Weighted FFT features
        if self.use_weighted_fft:
            self.beta_weight = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            
        # Initialize weights     
        self.initialize_weights()   
            
        # Jit compile the forward function - select backend dynamically
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            backend = 'inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager'
        else:
            backend = 'aot_eager'  # Default for CPU-only mode
        compile_fn = partial(torch.compile, fullgraph=True, backend=backend)
        self.compiled_forward = compile_fn(self._forward)  
        
    def forward(self, x):
        return self.compiled_forward(x)
    
    def _forward(self, x):
        B = x.shape[0]

         # Embed patches
        x = self.patch_embed(x.to(dtype=torch.float32))                                # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]                                                # Add positional embedding

        # Add class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)                                           # (B, num_patches + 1, embed_dim)
        
        # Optional: Initialize skip connections
        skips = x if self.skip else None  # Shape (B, num_patches + 1, embed_dim)

        # Apply Transformer blocks
        for i, blk in enumerate(self.blocks, 1):
            if self.skip and i in self.skip_layers:
                skips = x                                                               # Update skip connections
            
                # Add high-frequency components if applicable
                # if self.use_fft_features:
                #     x = self.add_fft_features(x, skips)    
            else:
                skips = None                                                            # Reset skips for non-skip layers

            x = blk(x, skips)                                                           # (B, num_patches + 1, embed_dim)
            

        # Extract class token
        x = x[:, 0]                                                                     # Extract class token
        x = self.final_layer(x)
        
        return AEEncoderOutput(z_enc=x, skips=None)
    
    
    def add_fft_features(self, x, skips):
        """Adds additional high-frequency FFT features."""
        # Separate class token and patch tokens (contains img patches)
        patch_token = skips[:, 1:]
        
        # Compute FFT features
        fft_features = self.fourier_unit.highpass_filter(patch_token)
        fft_features = self.norm_layer(fft_features)
        
        if self.use_weighted_fft:
            fft_features = self.sigmoid(self.beta_weight) * self.norm_layer(fft_features)

        return x + fft_features
            
        
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
            pose_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True, extra_tokens=1)
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
        