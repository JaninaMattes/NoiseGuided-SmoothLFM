# Code acdapted from:
# - https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models/blob/main/model/NoiseTransformer.py
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm import create_model


COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=True, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f
    

class NoiseTransformer(nn.Module):
    
    def __init__(self, out_channels=4, resolution=32):
        """Learn residual noise from the noisy input code (B, 4, 32, 32)"""
        super().__init__()
        self.out_channels = out_channels
        self.resolution = resolution
        
        self.upsample = lambda x: F.interpolate(x, [224,224])
        self.downsample = lambda x: F.interpolate(x, [resolution,resolution])
        self.upconv = nn.Conv2d(7,4,(1,1),(1,1),(0,0))
        self.downconv = nn.Conv2d(4,3,(1,1),(1,1),(0,0))
        # self.upconv = nn.Conv2d(7,4,(1,1),(1,1),(0,0))
        self.swin = create_model("swin_tiny_patch4_window7_224",pretrained=True)
        
        if COMPILE: self.forward = compile_fn(self.forward)


    def forward(self, x, residual=False):
        if residual:
            residual = x
            x = self.upconv(self.downsample(self.swin.forward_features(self.downconv(self.upsample(x))))) + residual
        else:
            x = self.upconv(self.downsample(self.swin.forward_features(self.downconv(self.upsample(x)))))

        return x
    
