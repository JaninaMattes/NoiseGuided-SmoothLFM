# Code adapted from:
# - https://github.com/facebookresearch/mae/blob/main/models_vit.py
# - https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L95
# - https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
from functools import partial
from typing import Optional, Type

import einops

import torch
from torch.jit import Final
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
)

from ldm.models.architecture.base_architectures import BaseDecoder
from ldm.models.nn.init.weight_init import trunc_normal_
from ldm.models.nn.layers.dropout.drop import DropPath
from ldm.models.nn.layers.layer_scale import LayerScale
from ldm.models.nn.layers.mlp import Mlp
from ldm.tools.fourier.fft import FrequencyUnit
from ldm.models.nn.out.outputs import AEDecoderOutput
from ldm.models.autoencoder.nn import get_2d_sincos_pos_embed


""" Jit Compile """

COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=True, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f



if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')




""" Skip Vision Transformer (ViT) Encoder """


class AttentionSkip(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    
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
    ) -> None:
        super().__init__(
            dim, num_heads, mlp_ratio, qkv_bias, qk_norm, proj_drop, 
            attn_drop, init_values, drop_path, act_layer, norm_layer, mlp_layer
        )
        self.attn = AttentionSkip(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


""" Skip Connection Layers """

class CrossAttnSkipLayer(nn.Module):
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
    
        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x, skip_input):
        # Perform attention and add dropout for regularization
        attn_output, _ = self.attn(self.norm1(x), self.norm2(skip_input), self.norm2(skip_input))
        return x + self.dropout(attn_output)


class SelfAttnSkipLayer(nn.Module):
    """
    Self-attention skip layer with residual connections and dropout.
    Helps improve feature fusion by attending to skip features.
    """
    def __init__(self, embed_dim, dropout=0., proj_drop=0., num_heads=2, batch_first=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=proj_drop, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
    
        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x, skip_input):
        # Perform attention and add dropout for regularization
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        return x + self.dropout(attn_output)
    
    

class ConcatSkipLayer(nn.Module):
    """
    Concatenation skip layer with residual connections and dropout.
    Helps improve feature fusion by concatenating skip features.
    
    Code adapted from:
    - https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L95
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.skip_linear = nn.Linear(2 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        if COMPILE: self.forward = compile_fn(self.forward)
    
    def forward(self, x, skip_input):
        x = self.skip_linear(torch.cat([x, skip_input], dim=-1))
        return self.norm(x)
    

class DCTEmbedLayer(nn.Module):
    """
    Conditioning of the token stream x on DCT features of target image (x1,..,xn)
    x:        (B, 257, embed_dim) – with CLS
    dct_freq: (B, 4, 32, 32)      - latent high-frequency image features 
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        # Project DCT features to the same dimension as the token stream
        # Note: The input to the DCT projection layer is (B, 4, 32, 32)
        self.dct_proj = nn.Sequential(
            nn.Conv2d(4, embed_dim, kernel_size=1),
            nn.SiLU(),
        )

        self.flatten = nn.Flatten(2)                            # (B, embed_dim, H*W)
        
        # Cross-attention layer
        # Note: The input to the cross-attention layer is (B, 1024, embed_dim)
        # The output is (B, 257, embed_dim) - with CLS
        self.cross_attn = CrossAttnSkipLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            proj_drop=dropout,
            batch_first=True,
        )

    def forward(self, x, dct_freq):
        """
        x:         (B, 257, embed_dim) – with CLS
        dct_freq:  (B, 4, 32, 32)
        """
        dct_feat = self.dct_proj(dct_freq)                   # (B, embed_dim, 32, 32)
        dct_feat = self.flatten(dct_feat).transpose(1, 2)    # (B, 1024, embed_dim)

        # Apply cross-attention on all tokens (including CLS)
        x = self.cross_attn(x, dct_feat)                     # (B, 257, embed_dim)
        return x
     
        
    

""" Embeddings """
    
class EmbedLayer(nn.Module):
    """
    The first layer of DiT.
    """
    def __init__(self, latent_dim, embed_dim, act_layer: nn.Module = nn.SiLU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, embed_dim, bias=True),
            act_layer(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x):
        return self.mlp(x)

        

""" Final Layer """

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    taken from: https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, embed_dim, patch_size, out_channels, act_layer: nn.Module = nn.Tanh, use_act_layer=False):
        super().__init__()
        out_dim = int(patch_size * patch_size * out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            act_layer() if use_act_layer else nn.Identity()     # For pixel-space outputs in range [-1, 1]
        )

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x):
        return self.mlp(x)
    


""" Skip Vision Transformer (ViT) Decoder """


class SkipViTDecoder(BaseDecoder):
    """ 
    Vision Transformer (ViT) decoder with optional skip connections.
    The implementation is based on the ViT model from timm (Huggingface).
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
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer=nn.LayerNorm, 
                 pos_embed_learned=False, 
                 use_fft_features=False, 
                 use_weighted_fft=False, 
                 skip=True, 
                 skip_indices=(3, 5, 7, 9),
                 skip_layer_type='default',
                 dct_indices=(9, 11,), 
                 cond_dct_freq=True,
                 use_act_layer=False,
                 use_checkpoint=False, 
                 compile=True
                 ):
        super(SkipViTDecoder, self).__init__()
        global COMPILE
        COMPILE = compile
        self.skip = skip
        self.skip_indices = skip_indices                                # List of skip-layers 
        self.dct_indices = dct_indices                                  # List of DCT layers (default: last layer)
        self.cond_dct_freq = cond_dct_freq           
        self.use_weighted_fft = use_weighted_fft
        self.use_fft_features = use_fft_features
        self.use_checkpoint = use_checkpoint   
        
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2                      # 32 ** 2 = 1024
        self.pos_embed_learned = pos_embed_learned    
        self.embed_layer = EmbedLayer(latent_dim, embed_dim, act_layer=act_layer)
        self.masked_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
                      
        # Positional Embedding
        self.extra_tokens = 1                                                   # cls token
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + self.extra_tokens, embed_dim) if pos_embed_learned else
            torch.zeros(1, self.num_patches + self.extra_tokens, embed_dim), requires_grad=pos_embed_learned
        )

        # Transformer blocks (default: 12)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[BlockSkip(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer
            ) for i in range(num_layers)]
        )

        self.final_layer = FinalLayer(embed_dim, patch_size, in_channels, use_act_layer=use_act_layer)         

        # Optional: Skip connections
        if skip:
            if skip_layer_type == 'default':
                self.skip_layer = ConcatSkipLayer(embed_dim)
            elif skip_layer_type == 'self_attn':
                self.skip_layer = SelfAttnSkipLayer(embed_dim)
            elif skip_layer_type == 'cross_attn':
                self.skip_layer = CrossAttnSkipLayer(embed_dim)
            else:
                raise ValueError(f"Invalid skip layer type: {skip_layer_type}")
        else:
            self.skip_layer = None
        
        # Optional: condition on DCT features
        if self.cond_dct_freq:
            self.dct_embed_layer = DCTEmbedLayer(embed_dim)
        else:
            self.dct_embed_layer = None
        
        # Optional: FFT features
        if self.use_fft_features:
            self.fourier_unit = FrequencyUnit()
            self.norm_layer = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        # Optional: Weighted FFT features
        if self.use_weighted_fft:
            self.beta_weight = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)
            self.sigmoid = nn.Sigmoid()
            
        # Initialize weights     
        self.initialize_weights()
    
    
    def forward(self, x, dct_freq=None):
        B = x.shape[0]
        x = self.embed_layer(x)                                                         # (B, latent_dim) -> (B, embed_dim)
        x = x.unsqueeze(1)                                                              # (B, 1, embed_dim)

        mask_tokens = self.masked_tokens.expand(B, -1, -1)                              # (B, num_patches, embed_dim)
        x = torch.cat((x, mask_tokens), dim=1)                                          # (B, num_patches + 1, embed_dim)
        assert x.shape == (B, self.num_patches + 1, self.embed_dim), \
            f"Expected shape {(B, self.num_patches + 1, self.embed_dim)}, got {x.shape}"

        x = x + self.pos_embed                                                          # (B, num_patches + 1, embed_dim)

        # Skip connections
        skips = x if self.skip else None                                               # Shape (B, num_patches + 1, embed_dim)
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)
            
            # normalize skip if used
            if i in self.skip_indices and self.skip_layer is not None:
                x = self.skip_layer(x, skips)                                                            
                skips = x
            
                # optional: add FFT features
                if self.use_fft_features:
                    x = self.add_fft_features(x, skips)  
            
            # Optional: DCT features in last block(s) (normalized)
            if i in self.dct_indices and dct_freq is not None and self.dct_embed_layer is not None:
                x = self.dct_embed_layer(x, dct_freq)                                   # (B, num_patches + 1, embed_dim)
                
        x = x[:, 1:, :]                                                                 # Remove class token, retain patch embeddings
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
            pose_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True, extra_tokens=self.extra_tokens)
            self.pos_embed.data.copy_(torch.from_numpy(pose_embed).float().unsqueeze(0))

        # Initialize masked tokens
        torch.nn.init.normal_(self.masked_tokens, std=.02)
        
        # Initialize skip layers weights
        if self.use_weighted_fft:
            torch.nn.init.normal_(self.beta_weight, std=.02)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.mlp[0].weight, 0)
        nn.init.constant_(self.final_layer.mlp[0].bias, 0)
        
        
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