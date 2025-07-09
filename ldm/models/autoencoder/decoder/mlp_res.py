# -------------------
# MLP Decoder
# -------------------
import torch 
from torch import nn

from functools import partial
from typing import Callable

from ldm.models.nn.activations.activations import ScaledTanh
from ldm.models.nn.blocks.block import LinearBlock, LinearResBlock


""" MLP-based Decoder """


class SmallMLPDecoder(nn.Module):
    """Decoder network for 4-channel 32x32 output."""
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
        scale_tanh: bool = False
    ):
        super(SmallMLPDecoder, self).__init__()
        self.image_size = image_size

        # Output dimension
        output_dim = out_channels * image_size * image_size  # Flattened image size
        
        layers = []
        input_dim = latent_dim  # Start from latent space size
        
        # Upsample
        for i in range(num_layers):
            out_dim = min(hidden_dim * (2 ** i), output_dim)  
            layers.append(
                LinearBlock(
                    in_features=input_dim,
                    out_features=out_dim, 
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    group_num=group_num,
                )
            )
            # Update for next layer
            input_dim = out_dim 

        self.decoder = nn.Sequential(*layers)
        self.head = nn.Linear(out_dim, output_dim) 
        
        # Scaled tanh
        self.act_tanh = ScaledTanh() if scale_tanh else nn.Identity()
        
        self._init_weights()

        # Jit compile the forward function
        self._compile()
         
         
    def forward(self, x):
        """ Forward pass """
        return self.compiled_forward(x)
    
    
    def _forward(self, x):
        """Forward pass through the decoder."""
        x = self.decoder(x)                 # (B, out_dim)
        x = self.head(x) 
        x = self.act_tanh(x)                # (B, out_channels * image_size^2)
        
        x = x.view(x.size(0), -1, self.image_size, self.image_size)   # Unflatten to image shape (B, 4, 32, 32)
        
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




""" ResMLP Decoder """

class ResMLPDecoder(nn.Module):
    """MLP-based Decoder network for latent representation.
    Reconstructs input of shape (B, 4, 32, 32) from latent code z.
    """
    def __init__(
        self,
        out_channels: int = 4,
        image_size: int = 32,
        latent_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 4,
        drop_rate: float = 0.,
        bottleneck_ratio: float = 0.25,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_conv_head: bool = False, # Use Conv2d head for spatial refinement
        use_scale_tanh: bool = False
    ):
        super(ResMLPDecoder, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.use_conv_head = use_conv_head

        # Outout dimension
        output_dim = out_channels * image_size * image_size

        # Build Decoder Layers
        layers = []
        input_dim = latent_dim
        for i in range(num_layers):
            out_dim = min(hidden_dim * (2 ** i), output_dim)  # Gradually approach the output size
            layers.append(
                LinearResBlock(
                    in_features=input_dim,
                    out_features=out_dim,
                    act_layer=act_layer,
                    bottleneck_ratio=bottleneck_ratio,
                    drop_rate=drop_rate
                )
            )
            input_dim = out_dim  # Update for next layer

        self.decoder = nn.Sequential(*layers)

        # Head: Project to flattened output space
        self.head = nn.Linear(input_dim, output_dim)
        self.s_act_layer = ScaledTanh() if use_scale_tanh else nn.Identity()

        # Optional Convolutional Head
        if self.use_conv_head:
            self.conv_head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Identity()  
            )
        else:
            self.conv_head = None

        self._init_weights()
        
        # Jit compile the forward function
        self._compile()


    def forward(self, x):
        """ Forward pass """
        return self.compiled_forward(x)
    
    
    def _forward(self, x):
        """
        Args:
            z: Latent code of shape (B, latent_dim).
        
        Returns:
            x_recon: Reconstructed image of shape (B, out_channels, image_size, image_size).
        """
        features = self.decoder(x)
        x_flat = self.head(features)                                    # (B, out_channels * image_size^2)

        # Unflatten 
        B = z.size(0)
        x_recon = x_flat.view(B, -1, self.image_size, self.image_size)  # (B, out_channels, H, W)

        # Optional: Convolutional Head
        if self.use_conv_head:
            x_recon = self.conv_head(x_recon)

        x_recon = self.s_act_layer(x_recon)                             # Scaled Tanh activation
        return x_recon



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