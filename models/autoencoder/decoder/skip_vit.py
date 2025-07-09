import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


from ldm.models.architecture.base_architectures import BaseDecoder
from ldm.models.autoencoder.nn import get_2d_sincos_pos_embed
from ldm.models.transformer.vit import CompiledBlock
from ldm.tools.fourier.fft import FrequencyUnit
from ldm.models.nn.out.outputs import AEDecoderOutput
    

##########################
# Legacy code
##########################
class EmbedLayer(nn.Module):
    """
    The first layer of DiT.
    """
    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        self.norm_input = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(latent_dim, embed_dim, bias=True)
        
    def forward(self, x):
        x = self.norm_input(x)
        x = self.linear(x)
        return x
       
       
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    taken from: https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, embed_dim, patch_size, out_channels, use_act_layer=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels, bias=True)
        self.act = nn.Tanh() if use_act_layer else nn.Identity()    # For pixel-space outputs in range [-1, 1]
        
    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        x = self.act(x)
        return x
    
    
class SkipViTDecoder(BaseDecoder):
    """ 
    Vision Transformer (ViT) decoder with skip connections.
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
                 use_act_layer: bool = False,
                 pos_embed_learned=False, 
                 use_fft_features=False, 
                 use_weighted_fft=False, 
                 skip=False, 
                 skip_layers=None,
                 skip_layer_type='default', 
                 use_checkpoint=False
                 ):
        super(SkipViTDecoder, self).__init__()
        self.skip = skip
        self.skip_layers = skip_layers if skip_layers is not None else []       # List of skip-layers            
        self.use_weighted_fft = use_weighted_fft
        self.use_fft_features = use_fft_features
        self.use_checkpoint = use_checkpoint   
          
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2                      # 32 ** 2 = 1024
        self.pos_embed_learned = pos_embed_learned    
        self.embed_layer = EmbedLayer(latent_dim, embed_dim)
        self.masked_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
    
        if self.skip and self.use_fft_features:
            self.fourier_unit = FrequencyUnit()
            self.norm_layer = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
                 
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

        self.final_layer = FinalLayer(embed_dim, patch_size, in_channels, use_act_layer=use_act_layer)        # Jit compile the encoder and decoder
   
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

        # Embed class token
        x = self.embed_layer(x)                                                         # (B, latent_dim) -> (B, embed_dim)
        x = x.unsqueeze(1)                                                              # (B, 1, embed_dim)

        # Expand masked patches
        mask_tokens = self.masked_tokens.expand(B, -1, -1)                              # (B, num_patches, embed_dim)
        x = torch.cat((x, mask_tokens), dim=1)                                          # (B, num_patches + 1, embed_dim)
        assert x.shape == (B, self.num_patches + 1, self.embed_dim), \
            f"Expected shape {(B, self.num_patches + 1, self.embed_dim)}, got {x.shape}"

        # Add positional embedding
        x = x + self.pos_embed                                                          # (B, num_patches + 1, embed_dim)

        # Optional: Initialize skip connections
        skips = x if self.skip else None  # Shape (B, num_patches + 1, embed_dim)
        for i, blk in enumerate(self.blocks, 1):
            if self.skip and i in self.skip_layers:
                skips = x                                                               # Update skip connections
            
                # Add high-frequency components if applicable
                # if self.use_fft_features:
                #     x = self.add_fft_features(x, skips)    
            else:
                skips = None                                                            # Reset skips for non-skip layers

            x = blk(x, skips)                                                           # (B, num_patches + 1, embed_dim)
            

        # Remove class token
        x = x[:, 1:, :]                                                                 # Remove class token, retain patch embeddings

        # Project back to spatial dimensions
        x = self.final_layer(x)                                                         # (B, num_patches, patch_dim)
        x = self.unpatchify(x)                                                          # (B, C, H, W)

        return AEDecoderOutput(z_dec=x, noise=None)


    def add_fft_features(self, x, skips):
        """Adds additional high-frequency FFT features."""
        # Separate class token and patch tokens (contains img patches)
        class_token = skips[:, :1]
        
        # Compute FFT features
        fft_features = self.fourier_unit.highpass_filter(class_token)
        fft_features = self.norm_layer(fft_features)
        
        if self.use_weighted_fft:
            # Apply a learnable weight to the FFT features
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

        # Initialize masked tokens
        torch.nn.init.normal_(self.masked_tokens, std=.02)
        
        # Initialize skip layers weights
        if self.use_weighted_fft:
            torch.nn.init.normal_(self.beta_weight, std=.02)
        
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.in_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs