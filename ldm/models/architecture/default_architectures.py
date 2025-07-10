from functools import partial
from typing import Callable, List
import torch
import torch.nn as nn


from ldm.models.architecture.base_architectures import BaseEncoder
from ldm.models.transformer.hilo import HiloBlock
from ldm.tools.fourier.fft import FrequencyUnit
from ldm.models.nn.out.outputs import AEDecoderOutput, AEEncoderOutput



""" MLP Layers """

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


""" Other Layers """ 

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
        if x.shape[-1] != skip_input.shape[-1]:
            raise ValueError(f"Shape mismatch: x ({x.shape[-1]}) and skip_input ({skip_input.shape[-1]}) must have the same feature dimension.")
        
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


    
class Encoder_AE_MLP(BaseEncoder):
    """ taken from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py#L13
    """
   
    def __init__(
        self, 
        in_channels: int = 4,
        image_size: int = 32,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.GroupNorm,
        group_num: int = 32,
        drop: float = .0,
        skip_layer_type='default',
        use_unet_skips: bool = False,
        skip_layers: bool = False,
        use_checkpoint=False
    ):
        super(Encoder_AE_MLP, self).__init__()
        
        input_dim = in_channels * image_size * image_size  # Flatten input image
        layers = []

        # Downsample 
        for i in range(num_layers):
            out_dim = hidden_dim // (2 ** i) 
            layers.append(
                LinearBlock(
                    in_features=input_dim,
                    out_features=out_dim, 
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    group_num=group_num
                )
            )
            input_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(out_dim, latent_dim)
        self.drop = nn.Dropout(drop)
        self.use_unet_skips = use_unet_skips
        self.skip_layers = skip_layers
        self.use_checkpoint = use_checkpoint   
        
        if self.use_unet_skips:
            self.fourier_unit = FrequencyUnit()
        
        # taken from: https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L95
        if skip_layers:
            if skip_layer_type == 'linear':
                self.skip_layer = LowRankSkipLayer()
            if skip_layer_type == 'concat':
                self.skip_layer = ConcatSkipLayer()
            else:
                self.skip_layer = SimpleSkipLayer()
                
        self.initialize_weights()
        
    
    def forward(self, x, output_layer_levels: List[int] = None):
        """Forward pass of the encoder."""
        if self.use_checkpoint:
            x, skips = torch.utils.checkpoint.checkpoint(self._forward, x, output_layer_levels)
        else:
            x, skips = self._forward(x, output_layer_levels)
        return AEEncoderOutput(
            z_enc=x,
            skips=skips
        )
            
        
    def _forward(self, x, output_layer_levels: List[int] = None):
        """Forward pass of the encoder."""
        B = x.size(0)
        x = x.view(B, -1)  # Flatten (B, C * H * W)
        
        skips = [] if self.use_unet_skips else None
        residual = x if self.skip_layers else None 
        for i, layer in enumerate(self.encoder, start=1):
            x = layer(x)
            
            # Unet skip connections
            if self.use_unet_skips and i in output_layer_levels:
                x_high_freq = self.fourier_unit.highpass_filter(x)
                skips.append(x_high_freq)

            # Long-range skip connections
            if self.skip_layers and i % 2 == 0:
                x = self.skip_layer(x, residual)
                residual = x   
                   
        x = self.head(x)    # Bottleneck
        x = self.drop(x)
        
        return x, skips
    
    
    
    def initialize_weights(self):
        # initialize transformer layers
        self.apply(self._basic_init)
        
    def _basic_init(self, m, gain: float = 1.0, bias: bool = True):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
            if bias and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            torch.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.LayerNorm):
            torch.init.constant_(m.weight, 1)
            torch.init.constant_(m.bias, 0) 
        
            
            
class SkipEmbeddingLayer(nn.Module):
    """
    Simple skip layer with dynamic linear layer initialization.
    """
    def __init__(self):
        super().__init__()
        self.linear = None  # Delay initialization
        
    def forward(self, x, skip_input):
        # Dynamically initialize the linear layer
        if self.linear is None:
            input_dim = skip_input.size(-1)
            self.linear = nn.Linear(input_dim, input_dim).to(skip_input.device)

        skip_input = self.linear(skip_input)
        return x + skip_input
    
    
    
    
class Decoder_AE_MLP(BaseEncoder):
   
    def __init__(
        self, 
        out_channels: int = 4,
        image_size: int = 32,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.GroupNorm,
        group_num: int = 32,
        drop: float = .0,
        skip_layer_type='default',
        use_unet_skips: bool = False,
        skip_layers: bool = False,
        use_checkpoint=False
    ):
        super(Decoder_AE_MLP, self).__init__()
        self.image_size = image_size
        
        output_dim = out_channels * image_size * image_size  # Flattened image size
        layers = []
        
        input_dim = latent_dim  # Bottleneck
        
        # Upsample
        for i in range(num_layers):
            out_dim = min(hidden_dim * (2 ** i), output_dim)  
            layers.append(
                LinearBlock(
                    in_features=input_dim,
                    out_features=out_dim, 
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    group_num=group_num
                )
            )
            # Update for next layer
            input_dim = out_dim 

        self.decoder = nn.Sequential(*layers)
        self.head = nn.Linear(out_dim, output_dim) 
        self.drop = nn.Dropout(drop)
        self.use_unet_skips = use_unet_skips
        self.skip_layers = skip_layers
        self.use_checkpoint = use_checkpoint   
     
        if self.use_unet_skips:
            self.linear_embedding = SkipEmbeddingLayer()
               
        # taken from: https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L95
        if skip_layers:
            if skip_layer_type == 'linear':
                self.skip_layer = LowRankSkipLayer()
            if skip_layer_type == 'concat':
                self.skip_layer = ConcatSkipLayer()
            else:
                self.skip_layer = SimpleSkipLayer()
                
        self.initialize_weights()
   
    
    def forward(self, x, skips=None, output_layer_levels: List[int] = None):
        if self.use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(self._forward, x, skips, output_layer_levels)
        else:
            x = self._forward(x, skips, output_layer_levels)
        return AEDecoderOutput(z_dec=x, noise=None)
         
        
    def _forward(self, x, skips=None, output_layer_levels: List[int] = None):
        """Forward pass of the encoder."""
        B = x.size(0)

        # Initialize residual
        residual = x if self.skip_layers else None
        for i, layer in enumerate(self.decoder, start=1):
            x = layer(x)
            
            # Unet skip connections
            if self.use_unet_skips and i in output_layer_levels:
                x_high_freq = skips.pop()
                x = self.linear_embedding(x, x_high_freq)

            # Apply long-range skip connections
            if self.skip_layers and i % 2 == 0:
                x = self.skip_layer(x, residual)
                residual = x
                
                  
        # Output layer                                   
        x = self.head(x)
        x = self.drop(x)
        
        x = x.view(B, -1, self.image_size, self.image_size)  # Reshape to image
        
        return x
    
    
    def initialize_weights(self):
        # initialize transformer layers
        self.apply(self._basic_init)


    def _basic_init(self, m, gain: float = 1.0, bias: bool = True):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
            if bias and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            torch.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.LayerNorm):
            torch.init.constant_(m.weight, 1)
            torch.init.constant_(m.bias, 0) 