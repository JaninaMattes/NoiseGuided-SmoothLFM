
# --------------------------------
# MLP Encoder
# --------------------------------
import torch 
from torch import nn

from functools import partial
from typing import Callable

from ldm.models.nn.blocks.block import LinearBlock, LinearResBlock


""" MLP-based Encoder """


class SmallMLPEncoder(nn.Module):
    """Encoder network for 4-channel 32x32 input."""
    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 32,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.GroupNorm,
        group_num: int = 32
    ):
        super(SmallMLPEncoder, self).__init__()
        
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
        self.head = nn.Linear(out_dim, latent_dim * 2)  # Output [mu, logvar] for VAE
        
        self._init_weights()


    def forward(self, x):
        """Forward pass through the encoder."""
        B = x.size(0)
        x = x.view(B, -1)  # Flatten (B, C, H, W) -> (B, C * H * W)
        
        x = self.encoder(x) # (B, hidden_dim)
        x = self.head(x)  # Learned [mu, logvar]
        
        return x 
    
    
    def _init_weights(self):
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
                
                
    
    
# --------------------------------

class ResMLPEncoder(nn.Module):
    """MLP-based Encoder network for 4-channel 32x32 input.
    Outputs latent representation as mean and logvar (for VAEs).
    """
    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 32,
        latent_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 4,
        drop_rate: float = 0.,
        bottleneck_ratio: float = 0.25,
        act_layer: Callable[..., nn.Module] = nn.GELU
    ):
        super(ResMLPEncoder, self).__init__()
        
        # Input dimension
        input_dim = in_channels * image_size * image_size
        
        # Build Encoder Layers
        layers = []
        for i in range(num_layers):
            layers.append(
                LinearResBlock(
                    in_features=input_dim if i == 0 else hidden_dim,
                    out_features=hidden_dim,
                    act_layer=act_layer,
                    bottleneck_ratio=bottleneck_ratio,
                    drop_rate=drop_rate
                )
            )
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, latent_dim * 2)
        
        self._init_weights()
        
        # Jit compile the forward function
        self._compile()
         
    
    def forward(self, x):
        """ Forward pass """
        return self.compiled_forward(x)
    
    
    def _forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            mean: Latent mean (B, latent_dim)
            logvar: Latent log-variance (B, latent_dim)
        """
        # Flatten input (B, C, H, W) -> (B, C * H * W)
        B = x.size(0)
        x = x.view(B, -1)

        # Pass through Encoder
        x = self.encoder(x)  # (B, hidden_dim)
        x = self.head(x)     # (B, latent_dim * 2)

        return x



    def _init_weights(self):
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